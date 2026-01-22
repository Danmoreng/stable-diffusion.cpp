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
    constexpr int SEEDVR2_GRAPH_SIZE = 20480;

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
            return hid; 
        }
    };

    // --- DiT Components ---

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
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* vid, struct ggml_tensor* txt) {
            return vid; 
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

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* vid, struct ggml_tensor* txt, struct ggml_tensor* emb) {
            return vid; 
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
            // Return a dummy tensor in the correct context
            return ggml_dup_tensor(ctx->ggml_ctx, x); 
        }
    };

    // --- VAE Implementation ---

    struct VAEParameterBlock : public GGMLBlock {
        std::map<std::string, std::vector<int64_t>> param_shapes;
        VAEParameterBlock(std::map<std::string, std::vector<int64_t>> shapes) : param_shapes(shapes) {}
        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            for (auto const& [name, shape] : param_shapes) {
                params[name] = ggml_new_tensor(ctx, GGML_TYPE_F32, (int)shape.size(), shape.data());
            }
        }
    };

    class SeedVR2VAE : public GGMLBlock {
    public:
        SeedVR2VAE(bool decode_only) {
            blocks["decoder.conv_in"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 8192}}, }));
            blocks["decoder.conv_norm_out"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["decoder.conv_out"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {3}}, {"weight", {3, 3, 3, 384}}, }));
            blocks["decoder.mid_block.attentions.0.group_norm"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.mid_block.attentions.0.to_k"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["decoder.mid_block.attentions.0.to_out.0"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["decoder.mid_block.attentions.0.to_q"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["decoder.mid_block.attentions.0.to_v"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["decoder.mid_block.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.mid_block.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.mid_block.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.mid_block.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.mid_block.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.mid_block.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.mid_block.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.mid_block.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.2.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.2.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.resnets.2.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.resnets.2.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.0.upsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.0.upsamplers.0.upscale_conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {4096}}, {"weight", {1, 1, 1, 2097152}}, }));
            blocks["decoder.up_blocks.1.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.resnets.2.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.2.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.resnets.2.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.resnets.2.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.1.upsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["decoder.up_blocks.1.upsamplers.0.upscale_conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {4096}}, {"weight", {1, 1, 1, 2097152}}, }));
            blocks["decoder.up_blocks.2.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 131072}}, }));
            blocks["decoder.up_blocks.2.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.resnets.0.conv_shortcut"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {1, 1, 1, 131072}}, }));
            blocks["decoder.up_blocks.2.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["decoder.up_blocks.2.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.2.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.2.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.2.resnets.2.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.resnets.2.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.resnets.2.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.2.resnets.2.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.2.upsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["decoder.up_blocks.2.upsamplers.0.upscale_conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {1024}}, {"weight", {1, 1, 1, 262144}}, }));
            blocks["decoder.up_blocks.3.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 32768}}, }));
            blocks["decoder.up_blocks.3.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["decoder.up_blocks.3.resnets.0.conv_shortcut"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {1, 1, 1, 32768}}, }));
            blocks["decoder.up_blocks.3.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["decoder.up_blocks.3.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["decoder.up_blocks.3.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["decoder.up_blocks.3.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["decoder.up_blocks.3.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["decoder.up_blocks.3.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["decoder.up_blocks.3.resnets.2.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["decoder.up_blocks.3.resnets.2.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["decoder.up_blocks.3.resnets.2.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["decoder.up_blocks.3.resnets.2.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.conv_in"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 384}}, }));
            blocks["encoder.conv_norm_out"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.conv_out"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {32}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["encoder.down_blocks.0.downsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 1, 16384}}, }));
            blocks["encoder.down_blocks.0.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["encoder.down_blocks.0.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["encoder.down_blocks.0.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.down_blocks.0.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.down_blocks.0.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["encoder.down_blocks.0.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {3, 3, 3, 16384}}, }));
            blocks["encoder.down_blocks.0.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.down_blocks.0.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.down_blocks.1.downsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["encoder.down_blocks.1.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 32768}}, }));
            blocks["encoder.down_blocks.1.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["encoder.down_blocks.1.resnets.0.conv_shortcut"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {1, 1, 1, 32768}}, }));
            blocks["encoder.down_blocks.1.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {128}}, {"weight", {128}}, }));
            blocks["encoder.down_blocks.1.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["encoder.down_blocks.1.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["encoder.down_blocks.1.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {3, 3, 3, 65536}}, }));
            blocks["encoder.down_blocks.1.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["encoder.down_blocks.1.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["encoder.down_blocks.2.downsamplers.0.conv"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.2.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 131072}}, }));
            blocks["encoder.down_blocks.2.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.2.resnets.0.conv_shortcut"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {1, 1, 1, 131072}}, }));
            blocks["encoder.down_blocks.2.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {256}}, {"weight", {256}}, }));
            blocks["encoder.down_blocks.2.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.2.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.2.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.2.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.2.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.3.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.3.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.3.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.3.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.3.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.3.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.down_blocks.3.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.down_blocks.3.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.mid_block.attentions.0.group_norm"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.mid_block.attentions.0.to_k"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["encoder.mid_block.attentions.0.to_out.0"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["encoder.mid_block.attentions.0.to_q"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["encoder.mid_block.attentions.0.to_v"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512, 512}}, }));
            blocks["encoder.mid_block.resnets.0.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.mid_block.resnets.0.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.mid_block.resnets.0.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.mid_block.resnets.0.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.mid_block.resnets.1.conv1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.mid_block.resnets.1.conv2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {3, 3, 3, 262144}}, }));
            blocks["encoder.mid_block.resnets.1.norm1"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
            blocks["encoder.mid_block.resnets.1.norm2"] = std::shared_ptr<GGMLBlock>(new VAEParameterBlock({ {"bias", {512}}, {"weight", {512}}, }));
        }
        void init(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix) {
            GGMLBlock::init(ctx, tensor_storage_map, prefix);
        }
        struct ggml_tensor* encode(GGMLRunnerContext* ctx, struct ggml_tensor* x) { 
            // Return a dummy latent tensor in the correct context
            return ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, x->ne[0]/8, x->ne[1]/8, 16, x->ne[3]);
        }
        struct ggml_tensor* decode(GGMLRunnerContext* ctx, struct ggml_tensor* z) { 
            // Return a dummy image tensor in the correct context
            return ggml_new_tensor_4d(ctx->ggml_ctx, GGML_TYPE_F32, z->ne[0]*8, z->ne[1]*8, 3, z->ne[3]);
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
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
            auto runner_ctx = get_context();
            struct ggml_tensor* out = model.forward(&runner_ctx, to_backend(x), to_backend(t), to_backend(context));
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        bool compute(int n_threads, struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* context, struct ggml_tensor** output, struct ggml_context* output_ctx = nullptr) {
             LOG_INFO("SeedVR2DiT compute start");
             auto get_graph = [&]() -> struct ggml_cgraph* { return build_graph(x, t, context); };
            bool res = GGMLRunner::compute(get_graph, n_threads, false, output, nullptr);
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
                struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
                auto runner_ctx = get_context();
                struct ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, to_backend(z)) : ae.encode(&runner_ctx, to_backend(z));
                ggml_build_forward_expand(gf, out);
                return gf;
            };
                         bool res = GGMLRunner::compute(get_graph, n_threads, false, output, nullptr);
            
            LOG_INFO("SeedVR2VAE compute end: %s", res ? "success" : "failed");
            return res;
        }
    };

} // namespace SeedVR2

#endif // __SEEDVR2_HPP__