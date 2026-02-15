#ifndef __SEEDVR2_HPP__
#define __SEEDVR2_HPP__

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include "vae.hpp"

namespace SeedVR2 {

// Constants for graph size
constexpr int SEEDVR2_GRAPH_SIZE = 81920;

// --- SeedVR2 Parameters ---

struct SeedVR2Params {
    int64_t vid_dim          = 2560;
    int64_t txt_dim          = 2560;
    int64_t emb_dim          = 15360;  // 6 * vid_dim
    int64_t heads            = 20;
    int64_t head_dim         = 128;
    int64_t expand_ratio     = 4;
    float norm_eps           = 1e-5f;
    int num_layers           = 32;
    int mm_layers            = 10;
    std::vector<int> patch_size = {1, 2, 2}; // [t, h, w]
    int vid_in_channels      = 33;
    int vid_out_channels     = 16;
    int txt_in_dim           = 5120;
};

// --- Basic Blocks ---

class RMSNorm : public UnaryBlock {
protected:
    int64_t dim;
    float eps;
    bool elementwise_affine;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        if (!elementwise_affine) return;
        ggml_type wtype = GGML_TYPE_F32;
        auto iter       = tensor_storage_map.find(prefix + "weight");
        if (iter == tensor_storage_map.end()) {
            iter = tensor_storage_map.find(prefix + "gamma");
        }
        if (iter != tensor_storage_map.end()) {
            params["weight"] = ggml_new_tensor(ctx, wtype, iter->second.n_dims, &iter->second.ne[0]);
        } else {
            params["weight"] = ggml_new_tensor_1d(ctx, wtype, dim);
        }
    }

public:
    RMSNorm(int64_t dim, float eps = 1e-6f, bool elementwise_affine = true) 
        : dim(dim), eps(eps), elementwise_affine(elementwise_affine) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
        auto h = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        if (elementwise_affine) {
            struct ggml_tensor* w = params["weight"];
            // Broadcast weight across other dimensions
            h = ggml_mul(ctx->ggml_ctx, h, w);
        }
        return h;
    }
};

// SwiGLU MLP
class SwiGLUMLP : public GGMLBlock {
protected:
    int64_t dim;
    int64_t hidden_dim;

public:
    SwiGLUMLP(int64_t dim, int64_t expand_ratio) : dim(dim) {
        hidden_dim = (int64_t)(2 * dim * expand_ratio / 3);
        int64_t multiple_of = 256;
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

        blocks["proj_in_gate"] = std::shared_ptr<GGMLBlock>(new Linear(dim, hidden_dim, false));
        blocks["proj_in"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, hidden_dim, false));
        blocks["proj_out"]     = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, dim, false));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto proj_in_gate = std::dynamic_pointer_cast<Linear>(blocks["proj_in_gate"]);
        auto proj_in      = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
        auto proj_out     = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

        auto gate = proj_in_gate->forward(ctx, x);
        gate      = ggml_silu(ctx->ggml_ctx, gate);
        auto up   = proj_in->forward(ctx, x);
        auto h    = ggml_mul(ctx->ggml_ctx, gate, up);
        return proj_out->forward(ctx, h);
    }
};

// AdaSingle Modulation
class AdaSingle : public GGMLBlock {
protected:
    int64_t dim;
    std::vector<std::string> layers;
    bool has_in;
    bool has_out;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        ggml_type wtype = GGML_TYPE_F32;
        for (const auto& l : layers) {
            if (has_in) {
                params[l + "_shift"] = ggml_new_tensor_1d(ctx, wtype, dim);
                params[l + "_scale"] = ggml_new_tensor_1d(ctx, wtype, dim);
            }
            if (has_out) {
                params[l + "_gate"]  = ggml_new_tensor_1d(ctx, wtype, dim);
            }
        }
    }

public:
    AdaSingle(int64_t dim, const std::vector<std::string>& layers, bool has_in = true, bool has_out = true) 
        : dim(dim), layers(layers), has_in(has_in), has_out(has_out) {}

    struct ggml_tensor* modulate(GGMLRunnerContext* ctx, struct ggml_tensor* hid, struct ggml_tensor* emb, const std::string& layer, const std::string& mode) {
        int idx = -1;
        for (int i = 0; i < (int)layers.size(); i++) {
            if (layers[i] == layer) {
                idx = i;
                break;
            }
        }
        if (idx == -1) return hid;

        // emb is [6 * dim, b]
        // We need to slice it to get shiftA, scaleA, gateA
        // In PyTorch: emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        // where g=3 if l=2, or g=6 if l=1. l * g is always 6.
        
        int64_t l = layers.size();
        int64_t g = 6 / l;
        
        // In GGML, emb is [6 * dim, b]. Reshape to [g, l, dim, b]
        int64_t b = (ggml_n_dims(emb) > 1) ? emb->ne[1] : 1;
        struct ggml_tensor* emb_view = ggml_reshape_4d(ctx->ggml_ctx, emb, g, l, dim, b);
        
        // shiftA = emb_view[0, idx, :, :]
        struct ggml_tensor* shiftA = ggml_view_2d(ctx->ggml_ctx, emb_view, dim, b, emb_view->nb[2], 
                                                 0 * emb_view->nb[0] + idx * emb_view->nb[1]);
        // scaleA = emb_view[1, idx, :, :]
        struct ggml_tensor* scaleA = ggml_view_2d(ctx->ggml_ctx, emb_view, dim, b, emb_view->nb[2], 
                                                 1 * emb_view->nb[0] + idx * emb_view->nb[1]);
        // gateA = emb_view[2, idx, :, :] (if g >= 3)
        struct ggml_tensor* gateA = nullptr;
        if (g >= 3) {
            gateA = ggml_view_2d(ctx->ggml_ctx, emb_view, dim, b, emb_view->nb[2], 
                                 2 * emb_view->nb[0] + idx * emb_view->nb[1]);
        }

        if (mode == "in") {
            struct ggml_tensor* shiftB = params[layer + "_shift"];
            struct ggml_tensor* scaleB = params[layer + "_scale"];
            // return hid * (scaleA + scaleB) + (shiftA + shiftB)
            auto scale = ggml_add(ctx->ggml_ctx, scaleA, scaleB);
            auto shift = ggml_add(ctx->ggml_ctx, shiftA, shiftB);
            auto h = ggml_mul(ctx->ggml_ctx, hid, scale);
            return ggml_add(ctx->ggml_ctx, h, shift);
        } else if (mode == "out") {
            struct ggml_tensor* gateB = params[layer + "_gate"];
            auto gate = ggml_add(ctx->ggml_ctx, gateA, gateB);
            return ggml_mul(ctx->ggml_ctx, hid, gate);
        }
        return hid; 
    }
};

// --- DiT Components ---

class SeedVR2RoPE : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        // Load rope.rope.freqs
        // prefix comes as "blocks.X.attn.rope."
        auto iter = tensor_storage_map.find(prefix + "rope.freqs");
        if (iter != tensor_storage_map.end()) {
             params["rope.freqs"] = ggml_new_tensor(ctx, iter->second.type, iter->second.n_dims, &iter->second.ne[0]);
        }
    }
public:
    SeedVR2RoPE() {}
    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // Placeholder forward - we don't apply RoPE yet but we load weights
        return x;
    }
};

class NaMMAttention : public GGMLBlock {
protected:
    int64_t vid_dim;
    int64_t txt_dim;
    int64_t heads;
    int64_t head_dim;
    bool shared_weights;

public:
    NaMMAttention(int64_t vid_dim, int64_t txt_dim, int64_t heads, int64_t head_dim, bool shared_weights)
        : vid_dim(vid_dim), txt_dim(txt_dim), heads(heads), head_dim(head_dim), shared_weights(shared_weights) {
        
        int64_t inner_dim = heads * head_dim;
        int64_t qkv_dim = inner_dim * 3;

        if (shared_weights) {
            blocks["proj_qkv.all"] = std::shared_ptr<GGMLBlock>(new Linear(vid_dim, qkv_dim, false)); 
            blocks["proj_out.all"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, vid_dim, true)); 
            blocks["norm_q.all"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
            blocks["norm_k.all"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
        } else {
            blocks["proj_qkv.vid"] = std::shared_ptr<GGMLBlock>(new Linear(vid_dim, qkv_dim, false));
            blocks["proj_qkv.txt"] = std::shared_ptr<GGMLBlock>(new Linear(txt_dim, qkv_dim, false));
            blocks["proj_out.vid"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, vid_dim, true));
            blocks["proj_out.txt"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, txt_dim, true));
            blocks["norm_q.vid"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
            blocks["norm_q.txt"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
            blocks["norm_k.vid"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
            blocks["norm_k.txt"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-5f, true));
        }
        blocks["rope"] = std::shared_ptr<GGMLBlock>(new SeedVR2RoPE());
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward_dual(GGMLRunnerContext* ctx, struct ggml_tensor* vid, struct ggml_tensor* txt) {
        struct ggml_tensor* vid_qkv;
        struct ggml_tensor* txt_qkv;
        
        if (shared_weights) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj_qkv.all"]);
            vid_qkv = proj->forward(ctx, vid);
            txt_qkv = proj->forward(ctx, txt);
        } else {
            auto proj_vid = std::dynamic_pointer_cast<Linear>(blocks["proj_qkv.vid"]);
            auto proj_txt = std::dynamic_pointer_cast<Linear>(blocks["proj_qkv.txt"]);
            vid_qkv = proj_vid->forward(ctx, vid);
            txt_qkv = proj_txt->forward(ctx, txt);
        }
        
        // Split QKV: [3*inner_dim, N] -> ([inner_dim, N], [inner_dim, N], [inner_dim, N])
        auto vid_qkv_vec = split_qkv(ctx->ggml_ctx, vid_qkv);
        auto txt_qkv_vec = split_qkv(ctx->ggml_ctx, txt_qkv);
        
        auto v_q = vid_qkv_vec[0];
        auto v_k = vid_qkv_vec[1];
        auto v_v = vid_qkv_vec[2];
        
        auto t_q = txt_qkv_vec[0];
        auto t_k = txt_qkv_vec[1];
        auto t_v = txt_qkv_vec[2];
        
        // Norm QK
        if (shared_weights) {
            auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q.all"]);
            auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k.all"]);
            v_q = norm_q->forward(ctx, v_q);
            t_q = norm_q->forward(ctx, t_q);
            v_k = norm_k->forward(ctx, v_k);
            t_k = norm_k->forward(ctx, t_k);
        } else {
            auto norm_q_vid = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q.vid"]);
            auto norm_q_txt = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q.txt"]);
            auto norm_k_vid = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k.vid"]);
            auto norm_k_txt = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k.txt"]);
            v_q = norm_q_vid->forward(ctx, v_q);
            t_q = norm_q_txt->forward(ctx, t_q);
            v_k = norm_k_vid->forward(ctx, v_k);
            t_k = norm_k_txt->forward(ctx, t_k);
        }
        
        // TODO: RoPE
        
        // Global self-attention on (vid + txt)
        auto q = ggml_concat(ctx->ggml_ctx, v_q, t_q, 1);
        auto k = ggml_concat(ctx->ggml_ctx, v_k, t_k, 1);
        auto v = ggml_concat(ctx->ggml_ctx, v_v, t_v, 1);
        
        auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, heads, nullptr, false, false, ctx->flash_attn_enabled);
        
        // Split output back to vid and txt
        auto v_out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, vid->ne[1]);
        auto t_out = ggml_ext_slice(ctx->ggml_ctx, out, 1, vid->ne[1], out->ne[1]);
        
        // Proj Out
        if (shared_weights) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj_out.all"]);
            v_out = proj->forward(ctx, v_out);
            t_out = proj->forward(ctx, t_out);
        } else {
            auto proj_vid = std::dynamic_pointer_cast<Linear>(blocks["proj_out.vid"]);
            auto proj_txt = std::dynamic_pointer_cast<Linear>(blocks["proj_out.txt"]);
            v_out = proj_vid->forward(ctx, v_out);
            t_out = proj_txt->forward(ctx, t_out);
        }
        
        return {v_out, t_out};
    }
};

class NaSwinAttention : public NaMMAttention {
public:
    NaSwinAttention(int64_t vid_dim, int64_t txt_dim, int64_t heads, int64_t head_dim, bool shared_weights)
        : NaMMAttention(vid_dim, txt_dim, heads, head_dim, shared_weights) {}
};

class NaMMSRTransformerBlock : public GGMLBlock {
protected:
    bool shared_weights;
    bool is_last_layer;

public:
    NaMMSRTransformerBlock(int64_t vid_dim, int64_t txt_dim, int64_t emb_dim, int64_t heads, int64_t head_dim, int64_t expand_ratio, bool shared_weights, bool is_last_layer)
        : shared_weights(shared_weights), is_last_layer(is_last_layer) {
        
        if (shared_weights) {
            blocks["attn_norm.all"] = std::shared_ptr<GGMLBlock>(new RMSNorm(vid_dim, 1e-5f, false)); 
            blocks["mlp_norm.all"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(vid_dim, 1e-5f, false));
            blocks["mlp.all"]       = std::shared_ptr<GGMLBlock>(new SwiGLUMLP(vid_dim, expand_ratio));
            blocks["ada.all"]       = std::shared_ptr<GGMLBlock>(new AdaSingle(vid_dim, {"attn", "mlp"}, true, true));
        } else {
            blocks["attn_norm.vid"] = std::shared_ptr<GGMLBlock>(new RMSNorm(vid_dim, 1e-5f, false));
            blocks["attn_norm.txt"] = std::shared_ptr<GGMLBlock>(new RMSNorm(txt_dim, 1e-5f, false));
            blocks["mlp_norm.vid"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(vid_dim, 1e-5f, false));
            blocks["mlp_norm.txt"]  = std::shared_ptr<GGMLBlock>(new RMSNorm(txt_dim, 1e-5f, false));
            blocks["mlp.vid"]       = std::shared_ptr<GGMLBlock>(new SwiGLUMLP(vid_dim, expand_ratio));
            blocks["mlp.txt"]       = std::shared_ptr<GGMLBlock>(new SwiGLUMLP(txt_dim, expand_ratio));
            blocks["ada.vid"]       = std::shared_ptr<GGMLBlock>(new AdaSingle(vid_dim, {"attn", "mlp"}, true, true));
            blocks["ada.txt"]       = std::shared_ptr<GGMLBlock>(new AdaSingle(txt_dim, {"attn", "mlp"}, true, true));
        }

        blocks["attn"] = std::shared_ptr<GGMLBlock>(new NaSwinAttention(vid_dim, txt_dim, heads, head_dim, shared_weights));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx, struct ggml_tensor* vid, struct ggml_tensor* txt, struct ggml_tensor* emb) {
        struct ggml_tensor* v_identity = vid;
        struct ggml_tensor* t_identity = txt;
        
        struct ggml_tensor* v_attn;
        struct ggml_tensor* t_attn;
        
        if (shared_weights) {
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["attn_norm.all"]);
            auto ada  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.all"]);
            v_attn = norm->forward(ctx, vid);
            t_attn = norm->forward(ctx, txt);
            v_attn = ada->modulate(ctx, v_attn, emb, "attn", "in");
            t_attn = ada->modulate(ctx, t_attn, emb, "attn", "in");
        } else {
            auto norm_v = std::dynamic_pointer_cast<RMSNorm>(blocks["attn_norm.vid"]);
            auto norm_t = std::dynamic_pointer_cast<RMSNorm>(blocks["attn_norm.txt"]);
            auto ada_v  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.vid"]);
            auto ada_t  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.txt"]);
            v_attn = norm_v->forward(ctx, vid);
            t_attn = norm_t->forward(ctx, txt);
            v_attn = ada_v->modulate(ctx, v_attn, emb, "attn", "in");
            t_attn = ada_t->modulate(ctx, t_attn, emb, "attn", "in");
        }
        
        auto attn = std::dynamic_pointer_cast<NaSwinAttention>(blocks["attn"]);
        auto res = attn->forward_dual(ctx, v_attn, t_attn);
        struct ggml_tensor* v_out = res.first;
        struct ggml_tensor* t_out = res.second;
        
        if (shared_weights) {
            auto ada = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.all"]);
            v_out = ada->modulate(ctx, v_out, emb, "attn", "out");
            t_out = ada->modulate(ctx, t_out, emb, "attn", "out");
        } else {
            auto ada_v = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.vid"]);
            auto ada_t = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.txt"]);
            v_out = ada_v->modulate(ctx, v_out, emb, "attn", "out");
            t_out = ada_t->modulate(ctx, t_out, emb, "attn", "out");
        }
        
        vid = ggml_add(ctx->ggml_ctx, v_identity, v_out);
        txt = ggml_add(ctx->ggml_ctx, t_identity, t_out);
        
        // MLP
        v_identity = vid;
        t_identity = txt;
        
        struct ggml_tensor* v_mlp;
        struct ggml_tensor* t_mlp;
        
        if (shared_weights) {
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["mlp_norm.all"]);
            auto ada  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.all"]);
            auto mlp  = std::dynamic_pointer_cast<SwiGLUMLP>(blocks["mlp.all"]);
            v_mlp = norm->forward(ctx, vid);
            t_mlp = norm->forward(ctx, txt);
            v_mlp = ada->modulate(ctx, v_mlp, emb, "mlp", "in");
            t_mlp = ada->modulate(ctx, t_mlp, emb, "mlp", "in");
            v_mlp = mlp->forward(ctx, v_mlp);
            t_mlp = mlp->forward(ctx, t_mlp);
            v_mlp = ada->modulate(ctx, v_mlp, emb, "mlp", "out");
            t_mlp = ada->modulate(ctx, t_mlp, emb, "mlp", "out");
        } else {
            auto norm_v = std::dynamic_pointer_cast<RMSNorm>(blocks["mlp_norm.vid"]);
            auto norm_t = std::dynamic_pointer_cast<RMSNorm>(blocks["mlp_norm.txt"]);
            auto ada_v  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.vid"]);
            auto ada_t  = std::dynamic_pointer_cast<AdaSingle>(blocks["ada.txt"]);
            auto mlp_v  = std::dynamic_pointer_cast<SwiGLUMLP>(blocks["mlp.vid"]);
            auto mlp_t  = std::dynamic_pointer_cast<SwiGLUMLP>(blocks["mlp.txt"]);
            v_mlp = norm_v->forward(ctx, vid);
            t_mlp = norm_t->forward(ctx, txt);
            v_mlp = ada_v->modulate(ctx, v_mlp, emb, "mlp", "in");
            t_mlp = ada_t->modulate(ctx, t_mlp, emb, "mlp", "in");
            v_mlp = mlp_v->forward(ctx, v_mlp);
            t_mlp = mlp_t->forward(ctx, t_mlp);
            v_mlp = ada_v->modulate(ctx, v_mlp, emb, "mlp", "out");
            t_mlp = ada_t->modulate(ctx, t_mlp, emb, "mlp", "out");
        }
        
        vid = ggml_add(ctx->ggml_ctx, v_identity, v_mlp);
        txt = ggml_add(ctx->ggml_ctx, t_identity, t_mlp);
        
        return {vid, txt};
    }
};

class SeedVR2DiT : public GGMLBlock {
protected:
    SeedVR2Params params;

public:
    SeedVR2DiT(SeedVR2Params params) : params(params) {
        blocks["vid_in.proj"] = std::shared_ptr<GGMLBlock>(new Linear(params.vid_in_channels * params.patch_size[1] * params.patch_size[2], params.vid_dim, true));
        blocks["txt_in"]      = std::shared_ptr<GGMLBlock>(new Linear(params.txt_in_dim, params.vid_dim, true));
        blocks["emb_in.proj_in"]  = std::shared_ptr<GGMLBlock>(new Linear(256, params.vid_dim, true));
        blocks["emb_in.proj_hid"] = std::shared_ptr<GGMLBlock>(new Linear(params.vid_dim, params.vid_dim, true));
        blocks["emb_in.proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(params.vid_dim, params.emb_dim, true));

        for (int i = 0; i < params.num_layers; i++) {
            bool shared = (i >= params.mm_layers);
            bool last   = (i == params.num_layers - 1);
            blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new NaMMSRTransformerBlock(params.vid_dim, params.vid_dim, params.emb_dim, params.heads, params.head_dim, params.expand_ratio, shared, last));
        }

        blocks["vid_out_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.vid_dim, 1e-5f, true));
        blocks["vid_out_ada"]  = std::shared_ptr<GGMLBlock>(new AdaSingle(params.vid_dim, {"out"}, true, false)); 
        blocks["vid_out.proj"] = std::shared_ptr<GGMLBlock>(new Linear(params.vid_dim, params.vid_out_channels * params.patch_size[1] * params.patch_size[2], true));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* context) {
        // x: [channels, tokens] -> [132, t*h*w]
        // t: [1] (timestep scalar)
        // context: [in_dim, seq_len] -> [5120, 77]
        
        // 1. Timestep embedding
        auto t_emb = ggml_timestep_embedding(ctx->ggml_ctx, t, 256, 10000);
        
        auto emb_proj_in  = std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_in"]);
        auto emb_proj_hid = std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_hid"]);
        auto emb_proj_out = std::dynamic_pointer_cast<Linear>(blocks["emb_in.proj_out"]);
        
        auto emb = emb_proj_in->forward(ctx, t_emb);
        emb = ggml_silu(ctx->ggml_ctx, emb);
        emb = emb_proj_hid->forward(ctx, emb);
        emb = ggml_silu(ctx->ggml_ctx, emb);
        emb = emb_proj_out->forward(ctx, emb);
        
        // 2. Video input projection
        auto vid_in_proj = std::dynamic_pointer_cast<Linear>(blocks["vid_in.proj"]);
        auto vid = vid_in_proj->forward(ctx, x);
        
        // 3. Text input projection
        auto txt_in = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
        auto txt = txt_in->forward(ctx, context);
        
        // 4. Transformer blocks
        for (int i = 0; i < params.num_layers; i++) {
            auto block = std::dynamic_pointer_cast<NaMMSRTransformerBlock>(blocks["blocks." + std::to_string(i)]);
            auto res = block->forward(ctx, vid, txt, emb);
            vid = res.first;
            txt = res.second;
        }
        
        // 5. Output
        auto vid_out_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["vid_out_norm"]);
        auto vid_out_ada  = std::dynamic_pointer_cast<AdaSingle>(blocks["vid_out_ada"]);
        auto vid_out_proj = std::dynamic_pointer_cast<Linear>(blocks["vid_out.proj"]);
        
        vid = vid_out_norm->forward(ctx, vid);
        vid = vid_out_ada->modulate(ctx, vid, emb, "out", "in");
        vid = vid_out_proj->forward(ctx, vid);
        
        return vid;
    }
};

class SeedVR2GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;

public:
    SeedVR2GroupNorm(int num_groups, int64_t num_channels, float eps = 1e-6f, bool affine = true)
        : num_groups(num_groups), num_channels(num_channels), eps(eps), affine(affine) {
        if (affine) {
            params["weight"] = nullptr; // Will be allocated in init_params
            params["bias"]   = nullptr;
        }
    }

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        if (affine) {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // x: [W, H, T, C] (channels at dim 3)
        // ggml_group_norm expects channels at dim 2 (standard NCHW where C is dim 2)
        // So we permute [W, H, T, C] -> [W, H, C, T] to put C at dim 2.
        
        x = ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2); // [W, H, C, T]
        x = ggml_cont(ctx->ggml_ctx, x);
        
        // SeedVR2 GroupNorm is spatial-only (per frame).
        // ggml_group_norm normalizes over ne[0] and ne[1] (W, H) for each group in ne[2] (C),
        // independently for each batch element in ne[3] (T).
        // So [W, H, C, T] is exactly what we want.
        
        x = ggml_group_norm(ctx->ggml_ctx, x, num_groups, eps);
        
        if (affine) {
            struct ggml_tensor* w = params["weight"];
            struct ggml_tensor* b = params["bias"];
            // w, b are [C].
            // After permute, x is [W, H, C, T].
            // We want to broadcast w, b along C (dim 2).
            // Reshape w, b to [1, 1, C, 1]
            w = ggml_reshape_4d(ctx->ggml_ctx, w, 1, 1, num_channels, 1);
            b = ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, num_channels, 1);
            x = ggml_mul(ctx->ggml_ctx, x, w);
            x = ggml_add(ctx->ggml_ctx, x, b);
        }
        
        // Permute back: [W, H, C, T] -> [W, H, T, C]
        x = ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2);
        x = ggml_cont(ctx->ggml_ctx, x);
        
        return x;
    }
};

class CausalConv3d : public GGMLBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::tuple<int, int, int> kernel_size;
    std::tuple<int, int, int> stride;
    std::tuple<int, int, int> padding;
    std::tuple<int, int, int> dilation;
    bool bias;

    void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        params["weight"] = ggml_new_tensor_4d(ctx,
                                              GGML_TYPE_F16,
                                              std::get<2>(kernel_size),
                                              std::get<1>(kernel_size),
                                              std::get<0>(kernel_size),
                                              in_channels * out_channels);

        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    CausalConv3d(int64_t in_channels,
                 int64_t out_channels,
                 std::tuple<int, int, int> kernel_size,
                 std::tuple<int, int, int> stride   = {1, 1, 1},
                 std::tuple<int, int, int> padding  = {0, 0, 0},
                 std::tuple<int, int, int> dilation = {1, 1, 1},
                 bool bias                          = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(std::move(kernel_size)),
          stride(std::move(stride)),
          padding(std::move(padding)),
          dilation(std::move(dilation)),
          bias(bias) {}

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        int64_t t = x->ne[2];
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = bias ? params["bias"] : nullptr;

        int kd = std::get<0>(kernel_size);
        int kh = std::get<1>(kernel_size);
        int kw = std::get<2>(kernel_size);
        
        int ph = std::get<1>(padding);
        int pw = std::get<2>(padding);
        
        int sd = std::get<0>(stride);
        int sh = std::get<1>(stride);
        int sw = std::get<2>(stride);
        
        int dd = std::get<0>(dilation);
        int dh = std::get<1>(dilation);
        int dw = std::get<2>(dilation);

        // Decompose 3D conv into loop of 2D convs
        
        // Pre-slice weights [kw, kh, 1, IC*OC]
        std::vector<struct ggml_tensor*> w_slices(kd);
        for (int k = 0; k < kd; k++) {
            struct ggml_tensor* w_k = ggml_view_4d(ctx->ggml_ctx, w, kw, kh, 1, in_channels * out_channels,
                                                   w->nb[1], w->nb[2], w->nb[3], k * w->nb[2]);
            
            w_k = ggml_cont(ctx->ggml_ctx, w_k); 
            w_k = ggml_reshape_4d(ctx->ggml_ctx, w_k, kw, kh, in_channels, out_channels);
            w_slices[k] = w_k;
        }

        // Pre-slice input frames
        std::vector<struct ggml_tensor*> x_frames(t);
        for (int i = 0; i < t; i++) {
            struct ggml_tensor* x_i = ggml_view_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], 1, x->ne[3],
                                                   x->nb[1], x->nb[2], x->nb[3], i * x->nb[2]);
            x_i = ggml_cont(ctx->ggml_ctx, x_i);
            x_i = ggml_reshape_4d(ctx->ggml_ctx, x_i, x->ne[0], x->ne[1], in_channels, 1);
            x_frames[i] = x_i;
        }

        int64_t t_out = (t - 1) / sd + 1;
        std::vector<struct ggml_tensor*> out_frames;
        
        for (int i = 0; i < t_out; i++) {
            struct ggml_tensor* frame_sum = nullptr;
            
            for (int k = 0; k < kd; k++) {
                // SeedVR2 Causal Inflation: head is replicated. 
                // This means for t_in < 0, we use frame 0.
                int t_in = i * sd - (kd - 1) * dd + k * dd;
                t_in = std::max(0, std::min((int)t - 1, t_in));
                
                struct ggml_tensor* conv_res = ggml_ext_conv_2d(ctx->ggml_ctx, x_frames[t_in], w_slices[k], nullptr,
                                                                sw, sh, pw, ph, dw, dh, false);
                
                if (frame_sum == nullptr) {
                    frame_sum = conv_res;
                } else {
                    frame_sum = ggml_add(ctx->ggml_ctx, frame_sum, conv_res);
                }
            }
            
            if (b != nullptr) {
                struct ggml_tensor* b_reshaped = ggml_reshape_4d(ctx->ggml_ctx, b, 1, 1, out_channels, 1);
                frame_sum = ggml_add(ctx->ggml_ctx, frame_sum, b_reshaped);
            }
            
            // Reshape frame_sum [W, H, OC, 1] -> [W, H, 1, OC]
            frame_sum = ggml_reshape_4d(ctx->ggml_ctx, frame_sum, frame_sum->ne[0], frame_sum->ne[1], 1, out_channels);
            out_frames.push_back(frame_sum);
        }
        
        struct ggml_tensor* result = out_frames[0];
        for (size_t i = 1; i < out_frames.size(); i++) {
            result = ggml_concat(ctx->ggml_ctx, result, out_frames[i], 2);
        }
        
        return result;
    }
};

class VAEResnetBlock : public GGMLBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    bool use_shortcut;

public:
    VAEResnetBlock(int64_t in_channels, int64_t out_channels, bool use_shortcut)
        : in_channels(in_channels), out_channels(out_channels), use_shortcut(use_shortcut) {
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new SeedVR2GroupNorm(32, in_channels, 1e-6f, true));
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, out_channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new SeedVR2GroupNorm(32, out_channels, 1e-6f, true));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_channels, out_channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        if (use_shortcut) {
            blocks["conv_shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_channels, out_channels, {1, 1, 1}));
        }
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* identity = x;
        if (use_shortcut) {
            identity = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_shortcut"])->forward(ctx, identity);
        }
        
        auto norm1 = std::dynamic_pointer_cast<SeedVR2GroupNorm>(blocks["norm1"]);
        auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
        auto norm2 = std::dynamic_pointer_cast<SeedVR2GroupNorm>(blocks["norm2"]);
        auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);
        
        struct ggml_tensor* h = x;
        h = norm1->forward(ctx, h);
        h = ggml_silu(ctx->ggml_ctx, h);
        h = conv1->forward(ctx, h);
        
        h = norm2->forward(ctx, h);
        h = ggml_silu(ctx->ggml_ctx, h);
        h = conv2->forward(ctx, h);
        
        return ggml_add(ctx->ggml_ctx, h, identity);
    }
};

class VAEAttnBlock : public GGMLBlock {
protected:
    int64_t channels;

public:
    VAEAttnBlock(int64_t channels) : channels(channels) {
        blocks["group_norm"] = std::shared_ptr<GGMLBlock>(new SeedVR2GroupNorm(32, channels, 1e-6f, true));
        blocks["to_q"]       = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
        blocks["to_k"]       = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
        blocks["to_v"]       = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
        blocks["to_out.0"]   = std::shared_ptr<GGMLBlock>(new Linear(channels, channels, true));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        auto identity = x;
        x = std::dynamic_pointer_cast<SeedVR2GroupNorm>(blocks["group_norm"])->forward(ctx, x);
        
        int64_t w = x->ne[0];
        int64_t h = x->ne[1];
        int64_t t = x->ne[2];
        int64_t c = x->ne[3];
        int64_t n = h * w; // Spatial sequence length
        
        // Permute to [C, W, H, T] so C is fastest
        // [W, H, T, C] -> [C, W, H, T]
        // Src0(W)->Dst1, Src1(H)->Dst2, Src2(T)->Dst3, Src3(C)->Dst0
        x = ggml_permute(ctx->ggml_ctx, x, 1, 2, 3, 0); 
        x = ggml_cont(ctx->ggml_ctx, x);

        // Reshape for projection: [C, W*H*T]
        // We project all frames together
        struct ggml_tensor* x_flat = ggml_reshape_2d(ctx->ggml_ctx, x, c, n * t);
        
        auto q = std::dynamic_pointer_cast<Linear>(blocks["to_q"])->forward(ctx, x_flat);
        auto k = std::dynamic_pointer_cast<Linear>(blocks["to_k"])->forward(ctx, x_flat);
        auto v = std::dynamic_pointer_cast<Linear>(blocks["to_v"])->forward(ctx, x_flat);
        
        // Reshape for attention: [C, N, T] -> batch over T
        // q is [C, N*T]. We want [C, N, T].
        // ne[0]=C, ne[1]=N*T.
        // We want ne[0]=C, ne[1]=N, ne[2]=T.
        q = ggml_reshape_3d(ctx->ggml_ctx, q, c, n, t);
        k = ggml_reshape_3d(ctx->ggml_ctx, k, c, n, t);
        v = ggml_reshape_3d(ctx->ggml_ctx, v, c, n, t);
        
        // Attention: Spatial only (per frame)
        struct ggml_tensor* out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, false, ctx->flash_attn_enabled);
        
        // Reshape back for output proj: [C, N*T]
        out = ggml_reshape_2d(ctx->ggml_ctx, out, c, n * t);
        
        out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"])->forward(ctx, out);
        
        // Reshape back to [C, W, H, T]
        // We need to be careful. The linear output is [C, N*T] (contiguous C first).
        // We want [C, W, H, T].
        out = ggml_reshape_4d(ctx->ggml_ctx, out, c, w, h, t);
        
        // Permute back to [W, H, T, C]
        // [C, W, H, T] -> [W, H, T, C]
        // Src0(C)->Dst3, Src1(W)->Dst0, Src2(H)->Dst1, Src3(T)->Dst2
        // Args: axis0, axis1, axis2, axis3
        // new_ne[0] = old_ne[axis0]
        // new_ne[1] = old_ne[axis1]
        // new_ne[2] = old_ne[axis2]
        // new_ne[3] = old_ne[axis3]
        // We want new_ne[0] = W = old_ne[1] -> axis0=1
        // We want new_ne[1] = H = old_ne[2] -> axis1=2
        // We want new_ne[2] = T = old_ne[3] -> axis2=3
        // We want new_ne[3] = C = old_ne[0] -> axis3=0
        out = ggml_permute(ctx->ggml_ctx, out, 3, 0, 1, 2);
        out = ggml_cont(ctx->ggml_ctx, out);
        
        return ggml_add(ctx->ggml_ctx, out, identity);
    }
};

class VAEUpsample3D : public GGMLBlock {
protected:
    int64_t channels;
    bool temporal_up;

public:
    VAEUpsample3D(int64_t channels, bool temporal_up = true) : channels(channels), temporal_up(temporal_up) {
        int64_t upscale_ratio = temporal_up ? 8 : 4;
        blocks["upscale_conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels * upscale_ratio, {1, 1, 1}));
        blocks["conv"]         = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        const bool disable_temporal_up_runtime = getenv("SD_SEEDVR2_DISABLE_TEMPORAL_UP") != nullptr;
        const bool trace_shapes = getenv("SD_SEEDVR2_TRACE_SHAPES") != nullptr;
        if (trace_shapes) {
            LOG_INFO("VAEUpsample3D start (temporal_up=%d, disable_temporal_up_runtime=%d): [%ld, %ld, %ld, %ld]",
                     temporal_up ? 1 : 0, disable_temporal_up_runtime ? 1 : 0,
                     x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        }
        x = std::dynamic_pointer_cast<CausalConv3d>(blocks["upscale_conv"])->forward(ctx, x);
        
        int64_t w = x->ne[0];
        int64_t h = x->ne[1];
        int64_t t = x->ne[2];
        int64_t c = x->ne[3];

        auto apply_width_x2 = [&]() {
            bool use_ref = getenv("SD_SEEDVR2_UPSAMPLE_HOST_REF") != nullptr;
            if (use_ref) {
                // Explicit gather/scatter reference implementation (slow graph build, but explicit layout)
                // x: [W, H, T, C]
                // We split C into 2 parts: C_0 and C_1.
                // Output is [2W, H, T, C/2] where we interleave w_0(C_0), w_0(C_1), w_1(C_0), w_1(C_1)...
                
                int64_t c2 = c / 2;
                struct ggml_tensor* x_c0 = ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, c2);
                struct ggml_tensor* x_c1 = ggml_ext_slice(ctx->ggml_ctx, x, 3, c2, c);
                
                // We need to interleave these along W (dim 0).
                // Since we can't easily interleave elements in GGML without permute (which is what we are debugging),
                // we iterate W slices and concat them.
                
                std::vector<struct ggml_tensor*> cols;
                for (int i = 0; i < w; i++) {
                    // Slice w=i from c0 and c1
                    struct ggml_tensor* col_0 = ggml_ext_slice(ctx->ggml_ctx, x_c0, 0, i, i + 1); // [1, H, T, C/2]
                    struct ggml_tensor* col_1 = ggml_ext_slice(ctx->ggml_ctx, x_c1, 0, i, i + 1); // [1, H, T, C/2]
                    col_0 = ggml_cont(ctx->ggml_ctx, col_0);
                    col_1 = ggml_cont(ctx->ggml_ctx, col_1);
                    cols.push_back(col_0);
                    cols.push_back(col_1);
                }
                
                // Concat all 2*W columns back into [2W, H, T, C/2]
                // Helper to concat vector efficiently
                struct ggml_tensor* result = cols[0];
                for (size_t i = 1; i < cols.size(); i++) {
                    result = ggml_concat(ctx->ggml_ctx, result, cols[i], 0); // Concat along W
                }
                x = result;
                w *= 2;
                c /= 2;
                if (trace_shapes) {
                    LOG_INFO("VAEUpsample3D(REF) after width x2: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
                }
            } else {
                // axis0 (w) -> 2, axis1 (h*t) -> 0, axis2 (2) -> 1, axis3 (c/2) -> 3
                // New0=Old2(2), New1=Old0(w), New2=Old1(ht), New3=Old3(c/2)
                // Result: [2, W, HT, C2]
                x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h*t, 2, c/2); 
                x = ggml_permute(ctx->ggml_ctx, x, 2, 0, 1, 3);
                x = ggml_cont(ctx->ggml_ctx, x);
                x = ggml_reshape_4d(ctx->ggml_ctx, x, 2*w, h, t, c/2);
                w *= 2;
                c /= 2;
                if (trace_shapes) {
                    LOG_INFO("VAEUpsample3D after width x2: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
                }
            }
        };

        auto apply_height_x2 = [&]() {
            bool use_ref = getenv("SD_SEEDVR2_UPSAMPLE_HOST_REF") != nullptr;
            if (use_ref) {
                // Height reference: Split C -> C_0, C_1. Interleave along H.
                int64_t c2 = c / 2;
                struct ggml_tensor* x_c0 = ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, c2);
                struct ggml_tensor* x_c1 = ggml_ext_slice(ctx->ggml_ctx, x, 3, c2, c);
                
                std::vector<struct ggml_tensor*> rows;
                for (int i = 0; i < h; i++) {
                    // Slice h=i from c0 and c1
                    struct ggml_tensor* row_0 = ggml_ext_slice(ctx->ggml_ctx, x_c0, 1, i, i + 1); // [W, 1, T, C/2]
                    struct ggml_tensor* row_1 = ggml_ext_slice(ctx->ggml_ctx, x_c1, 1, i, i + 1); // [W, 1, T, C/2]
                    row_0 = ggml_cont(ctx->ggml_ctx, row_0);
                    row_1 = ggml_cont(ctx->ggml_ctx, row_1);
                    rows.push_back(row_0);
                    rows.push_back(row_1);
                }
                
                struct ggml_tensor* result = rows[0];
                for (size_t i = 1; i < rows.size(); i++) {
                    result = ggml_concat(ctx->ggml_ctx, result, rows[i], 1); // Concat along H
                }
                x = result;
                h *= 2;
                c /= 2;
                if (trace_shapes) {
                    LOG_INFO("VAEUpsample3D(REF) after height x2: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
                }
            } else {
                // axis0 (w) -> 0, axis1 (h) -> 2, axis2 (2) -> 1, axis3 (...) -> 3
                // Result: [W, 2, H, T*C2]
                x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2, t*(c/2));
                x = ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3);
                x = ggml_cont(ctx->ggml_ctx, x);
                x = ggml_reshape_4d(ctx->ggml_ctx, x, w, 2*h, t, c/2);
                h *= 2;
                c /= 2;
                if (trace_shapes) {
                    LOG_INFO("VAEUpsample3D after height x2: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
                }
            }
        };

        auto apply_temporal = [&]() {
            // axis0 (w*h) -> 0, axis1 (t) -> 2, axis2 (2) -> 1, axis3 (c/2) -> 3
            // Result: [WH, 2, T, C2]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w*h, t, 2, c/2);
            x = ggml_permute(ctx->ggml_ctx, x, 0, 2, 1, 3);
            x = ggml_cont(ctx->ggml_ctx, x);
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2*t, c/2);

            int64_t t_new = 2 * t;
            struct ggml_tensor* part1 = ggml_view_4d(ctx->ggml_ctx, x, w, h, 1, c/2, x->nb[1], x->nb[2], x->nb[3], 0);
            if (t_new > 2) {
                struct ggml_tensor* part2 = ggml_view_4d(ctx->ggml_ctx, x, w, h, t_new - 2, c/2, x->nb[1], x->nb[2], x->nb[3], 2 * x->nb[2]);
                x = ggml_concat(ctx->ggml_ctx, part1, part2, 2);
                t = t_new - 1;
            } else {
                x = part1;
                t = 1;
            }
            c /= 2;
            x = ggml_cont(ctx->ggml_ctx, x);
            if (trace_shapes) {
                LOG_INFO("VAEUpsample3D after temporal up: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
            }
        };

        auto apply_temporal_bypass = [&]() {
            // Keep model parameter shapes unchanged, but bypass temporal inflation in debug mode.
            // Drop the upper temporal-factor half in channel-packed representation.
            x = ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, c/2);
            c /= 2;
            x = ggml_cont(ctx->ggml_ctx, x);
            if (trace_shapes) {
                LOG_INFO("VAEUpsample3D temporal bypass (slice C): [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
            }
        };

        // SeedVR2 factor order is [C_out, Z, Y, X] (fastest -> slowest), so apply temporal first.
        if (temporal_up) {
            if (disable_temporal_up_runtime) {
                apply_temporal_bypass();
            } else {
                apply_temporal();
            }
        }
        apply_width_x2();
        apply_height_x2();

        x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"])->forward(ctx, x);
        if (trace_shapes) {
            LOG_INFO("VAEUpsample3D end: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        }
        return x;
    }
};

class VAEDownsample3D : public GGMLBlock {
protected:
    int64_t channels;
    bool temporal_down;

public:
    VAEDownsample3D(int64_t channels, bool temporal_down = true) : channels(channels), temporal_down(temporal_down) {
        blocks["conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(channels, channels, {temporal_down ? 3 : 1, 3, 3}, {temporal_down ? 2 : 1, 2, 2}, {temporal_down ? 1 : 0, 0, 0}));
    }

    struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
        // Safe pad (0, 1, 0, 1) spatially like in PyTorch
        x = ggml_ext_pad_ext(ctx->ggml_ctx, x, 0, 1, 0, 1, 0, 0, 0, 0);
        return std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"])->forward(ctx, x);
    }
};

class SeedVR2VAE : public GGMLBlock {
public:
    SeedVR2VAE(bool decode_only) {
        if (!decode_only) {
            blocks["encoder.conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, 128, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            std::vector<int> block_channels = {128, 256, 512, 512};
            for (int i = 0; i < 4; i++) {
                int in_c = (i == 0) ? 128 : block_channels[i-1];
                int out_c = block_channels[i];
                blocks["encoder.down_blocks." + std::to_string(i) + ".resnets.0"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(in_c, out_c, in_c != out_c));
                blocks["encoder.down_blocks." + std::to_string(i) + ".resnets.1"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(out_c, out_c, false));
                if (i < 3) {
                    bool temporal_down = (i > 0); // Downsampler 0 has kT=1, 1 and 2 have kT=3
                    blocks["encoder.down_blocks." + std::to_string(i) + ".downsamplers.0"] = std::shared_ptr<GGMLBlock>(new VAEDownsample3D(out_c, temporal_down));
                }
            }
            blocks["encoder.mid_block.resnets.0"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(512, 512, false));
            blocks["encoder.mid_block.attentions.0"] = std::shared_ptr<GGMLBlock>(new VAEAttnBlock(512));
            blocks["encoder.mid_block.resnets.1"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(512, 512, false));
            blocks["encoder.conv_norm_out"] = std::shared_ptr<GGMLBlock>(new SeedVR2GroupNorm(32, 512, 1e-6f, true));
            blocks["encoder.conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(512, 32, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        blocks["decoder.conv_in"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(16, 512, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        blocks["decoder.mid_block.resnets.0"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(512, 512, false));
        blocks["decoder.mid_block.attentions.0"] = std::shared_ptr<GGMLBlock>(new VAEAttnBlock(512));
        blocks["decoder.mid_block.resnets.1"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(512, 512, false));
        std::vector<int> up_channels = {512, 512, 256, 128};
        for (int i = 0; i < 4; i++) {
            int in_c = (i == 0) ? 512 : up_channels[i-1];
            int out_c = up_channels[i];
            blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.0"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(in_c, out_c, in_c != out_c));
            blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.1"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(out_c, out_c, false));
            blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.2"] = std::shared_ptr<GGMLBlock>(new VAEResnetBlock(out_c, out_c, false));
            if (i < 3) {
                bool temporal_up = (i < 2); // temporal_up_num = 2. Blocks 0 and 1 have it. Block 2 does not.
                blocks["decoder.up_blocks." + std::to_string(i) + ".upsamplers.0"] = std::shared_ptr<GGMLBlock>(new VAEUpsample3D(out_c, temporal_up));
            }
        }
        blocks["decoder.conv_norm_out"] = std::shared_ptr<GGMLBlock>(new SeedVR2GroupNorm(32, 128, 1e-6f, true));
        blocks["decoder.conv_out"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(128, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
    }

    struct ggml_tensor* encode(GGMLRunnerContext* ctx, struct ggml_tensor* x) { 
        LOG_INFO("VAE encode: start, input shape: [%ld, %ld, %ld, %ld]", x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        // x: [W, H, 3, 1]
        // VAE expects [W, H, 1, 3]
        x = ggml_permute(ctx->ggml_ctx, x, 0, 1, 3, 2);
        x = ggml_cont(ctx->ggml_ctx, x);
        
        LOG_INFO("VAE encode: conv_in");
        x = std::dynamic_pointer_cast<CausalConv3d>(blocks["encoder.conv_in"])->forward(ctx, x);
        for (int i = 0; i < 4; i++) {
            LOG_INFO("VAE encode: down_block %d", i);
            x = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["encoder.down_blocks." + std::to_string(i) + ".resnets.0"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["encoder.down_blocks." + std::to_string(i) + ".resnets.1"])->forward(ctx, x);
            if (i < 3) {
                x = std::dynamic_pointer_cast<VAEDownsample3D>(blocks["encoder.down_blocks." + std::to_string(i) + ".downsamplers.0"])->forward(ctx, x);
            }
        }
        LOG_INFO("VAE encode: mid_block");
        x = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["encoder.mid_block.resnets.0"])->forward(ctx, x);
        x = std::dynamic_pointer_cast<VAEAttnBlock>(blocks["encoder.mid_block.attentions.0"])->forward(ctx, x);
        x = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["encoder.mid_block.resnets.1"])->forward(ctx, x);
        
        LOG_INFO("VAE encode: out");
        x = std::dynamic_pointer_cast<SeedVR2GroupNorm>(blocks["encoder.conv_norm_out"])->forward(ctx, x);
        x = ggml_silu(ctx->ggml_ctx, x);
        x = std::dynamic_pointer_cast<CausalConv3d>(blocks["encoder.conv_out"])->forward(ctx, x);
        
        // Take mean (first 16 channels) - ne[3] is channels
        x = ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, 16);
        LOG_INFO("VAE encode: end");
        return x;
    }

    struct ggml_tensor* decode(GGMLRunnerContext* ctx, struct ggml_tensor* z) { 
        LOG_INFO("VAE decode: start, input shape: [%ld, %ld, %ld, %ld]", z->ne[0], z->ne[1], z->ne[2], z->ne[3]);
        struct ggml_tensor* h = std::dynamic_pointer_cast<CausalConv3d>(blocks["decoder.conv_in"])->forward(ctx, z);
        LOG_INFO("VAE decode: mid_block");
        h = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["decoder.mid_block.resnets.0"])->forward(ctx, h);
        h = std::dynamic_pointer_cast<VAEAttnBlock>(blocks["decoder.mid_block.attentions.0"])->forward(ctx, h);
        h = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["decoder.mid_block.resnets.1"])->forward(ctx, h);
        for (int i = 0; i < 4; i++) {
            LOG_INFO("VAE decode: up_block %d", i);
            h = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.0"])->forward(ctx, h);
            h = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.1"])->forward(ctx, h);
            h = std::dynamic_pointer_cast<VAEResnetBlock>(blocks["decoder.up_blocks." + std::to_string(i) + ".resnets.2"])->forward(ctx, h);
            if (i < 3) {
                h = std::dynamic_pointer_cast<VAEUpsample3D>(blocks["decoder.up_blocks." + std::to_string(i) + ".upsamplers.0"])->forward(ctx, h);
            }
        }
        LOG_INFO("VAE decode: out");
        h = std::dynamic_pointer_cast<SeedVR2GroupNorm>(blocks["decoder.conv_norm_out"])->forward(ctx, h);
        h = ggml_silu(ctx->ggml_ctx, h);
        h = std::dynamic_pointer_cast<CausalConv3d>(blocks["decoder.conv_out"])->forward(ctx, h);
        
        // VAE output is [W, H, 1, 3] -> Permute to [W, H, 3, 1] for sd_image
        h = ggml_permute(ctx->ggml_ctx, h, 0, 1, 3, 2);
        h = ggml_cont(ctx->ggml_ctx, h);
        
        LOG_INFO("VAE decode: end");
        return h;
    }
    void init(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix) {
        GGMLBlock::init(ctx, tensor_storage_map, prefix);
    }
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        GGMLBlock::get_param_tensors(tensors, prefix);
    }
    void clear_cache() {}
    std::vector<struct ggml_tensor*> _feat_map;
};

// --- Runners ---

struct SeedVR2DiTRunner : public GGMLRunner {
    SeedVR2DiT model;

    SeedVR2DiTRunner(ggml_backend_t backend, bool offload_params_to_cpu, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "")
        : GGMLRunner(backend, offload_params_to_cpu), model(SeedVR2Params()) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override { return "seedvr2_dit"; }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    bool load_from_file(const std::string& file_path) {
        LOG_INFO("loading seedvr2 dit from '%s'", file_path.c_str());
        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) return false;
        if (!alloc_params_buffer()) return false;
        std::map<std::string, ggml_tensor*> dit_tensors;
        get_param_tensors(dit_tensors, "");
        return model_loader.load_tensors(dit_tensors, {}, 1);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* context) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, SEEDVR2_GRAPH_SIZE, false);
        auto runner_ctx = get_context();
        struct ggml_tensor* out = model.forward(&runner_ctx, to_backend(x), to_backend(t), to_backend(context));
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    bool compute(int n_threads, struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* context, struct ggml_tensor** output, struct ggml_context* output_ctx = nullptr) {
         LOG_INFO("SeedVR2DiT compute start");
         auto get_graph = [&]() -> struct ggml_cgraph* { return build_graph(x, t, context); };
        bool res = GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
         LOG_INFO("SeedVR2DiT compute end: %s", res ? "success" : "failed");
         return res;
    }
};

struct SeedVR2VAERunner : public VAE {
    SeedVR2VAE ae;
    SeedVR2VAERunner(ggml_backend_t backend, bool offload_params_to_cpu, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "", bool decode_only = false)
        : ae(decode_only), VAE(backend, offload_params_to_cpu) {
        ae.init(params_ctx, tensor_storage_map, prefix);
    }
    std::string get_desc() override { return "seedvr2_vae"; }
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
        ae.get_param_tensors(tensors, prefix);
    }
    bool load_from_file(const std::string& file_path) {
        LOG_INFO("loading seedvr2 vae from '%s'", file_path.c_str());
        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(file_path)) return false;
        if (!alloc_params_buffer()) return false;
        std::map<std::string, ggml_tensor*> vae_tensors;
        get_param_tensors(vae_tensors, "");
        return model_loader.load_tensors(vae_tensors, {}, 1);
    }
    bool compute(const int n_threads, struct ggml_tensor* z, bool decode_graph, struct ggml_tensor** output, struct ggml_context* output_ctx = nullptr) override {
        LOG_INFO("SeedVR2VAE compute start (%s)", decode_graph ? "decode" : "encode");
        auto get_graph = [&]() -> struct ggml_cgraph* {
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, SEEDVR2_GRAPH_SIZE, false);
            auto runner_ctx = get_context();
            struct ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, to_backend(z)) : ae.encode(&runner_ctx, to_backend(z));
            ggml_build_forward_expand(gf, out);
            return gf;
        };
         bool res = GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        
        LOG_INFO("SeedVR2VAE compute end: %s", res ? "success" : "failed");
        return res;
    }
};

} // namespace SeedVR2

#endif // __SEEDVR2_HPP__