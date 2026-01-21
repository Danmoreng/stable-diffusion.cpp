#ifndef __SEEDVR2_HPP__
#define __SEEDVR2_HPP__

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "common.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include "vae.hpp"

namespace SeedVR2 {

    // Constants for graph size
    constexpr int SEEDVR2_GRAPH_SIZE = 20480;
    constexpr int CACHE_T = 2; // From Wan

    // --- Basic Blocks ---

    class RMSNorm : public UnaryBlock {
    protected:
        int64_t dim;
        float eps;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            auto iter = tensor_storage_map.find(prefix + "gamma");
            if (iter != tensor_storage_map.end()) {
                params["gamma"] = ggml_new_tensor(ctx, wtype, iter->second.n_dims, &iter->second.ne[0]);
            } else {
                params["gamma"] = ggml_new_tensor_1d(ctx, wtype, dim);
            }
        }

    public:
        RMSNorm(int64_t dim, float eps = 1e-6f) : dim(dim), eps(eps) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            struct ggml_tensor* w = params["gamma"];
            w = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            // x: [N*IC, ID, IH, IW] -> permute -> norm -> mul -> permute back (Wan style)
            // Or simpler if x is [N, L, C]
            // Adapting Wan RMS_norm behavior for 3D tensors:
            auto h = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2)); // [ID, IH, IW, N*IC]
            h = ggml_rms_norm(ctx->ggml_ctx, h, eps);
            h = ggml_mul(ctx->ggml_ctx, h, w);
            h = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));
            return h;
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

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x, struct ggml_tensor* cache_x = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            struct ggml_tensor* w = params["weight"];
            struct ggml_tensor* b = nullptr;
            if (bias) {
                b = params["bias"];
            }

            int lp0 = std::get<2>(padding);
            int rp0 = std::get<2>(padding);
            int lp1 = std::get<1>(padding);
            int rp1 = std::get<1>(padding);
            int lp2 = 2 * std::get<0>(padding);
            int rp2 = 0;

            if (cache_x != nullptr && lp2 > 0) {
                x = ggml_concat(ctx->ggml_ctx, cache_x, x, 2);
                lp2 -= (int)cache_x->ne[2];
            }

            x = ggml_ext_pad_ext(ctx->ggml_ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    0, 0, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation));
        }
    };

    class Resample : public GGMLBlock {
    protected:
        int64_t dim;
        std::string mode;

    public:
        Resample(int64_t dim, const std::string& mode)
            : dim(dim), mode(mode) {
            if (mode == "upsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
            } else if (mode == "upsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                blocks["time_conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim * 2, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}));
            } else if (mode == "downsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
            } else if (mode == "downsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim, {3, 1, 1}, {2, 1, 1}, {0, 0, 0}));
            } else if (mode == "none") {
                // nn.Identity()
            } else {
                GGML_ASSERT(false && "invalid mode");
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            int64_t c = x->ne[3] / b;
            int64_t t = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            if (mode == "upsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    feat_idx += 1;
                    if (chunk_idx == 0) {
                        // feat_cache[idx] == nullptr, pass
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {  // chunk_idx >= 2
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }
                        if (chunk_idx == 1 && cache_x->ne[2] < 2) {  // Rep
                            cache_x = ggml_pad_ext(ctx->ggml_ctx, cache_x, 0, 0, 0, 0, (int)cache_x->ne[2], 0, 0, 0);
                        }
                        if (chunk_idx == 1) {
                            x = time_conv->forward(ctx, x);
                        } else {
                            x = time_conv->forward(ctx, x, feat_cache[idx]);
                        }
                        feat_cache[idx] = cache_x;
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w * h, t, c, 2);                                   // (2, c, t, h*w)
                        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 3, 1, 2));  // (c, t, 2, h*w)
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2 * t, c);                                   // (c, t*2, h, w)
                    }
                }
            }

            t = x->ne[2];
            if (mode != "none") {
                auto resample_1 = std::dynamic_pointer_cast<Conv2d>(blocks["resample.1"]);

                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)
                if (mode == "upsample2d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "upsample3d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "downsample2d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                } else if (mode == "downsample3d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                }
                x = resample_1->forward(ctx, x);
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)
            }

            if (mode == "downsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == nullptr) {
                        feat_cache[idx] = x;
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x    = ggml_ext_slice(ctx->ggml_ctx, x, 2, -1, x->ne[2]);
                        x               = ggml_concat(ctx->ggml_ctx,
                                                      ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      x,
                                                      2);
                        x               = time_conv->forward(ctx, x);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                }
            }

            return x;
        }
    };

    class AvgDown3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int factor_t;
        int factor_s;
        int factor;
        int64_t group_size;

    public:
        AvgDown3D(int64_t in_channels, int64_t out_channels, int factor_t, int factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(in_channels * factor % out_channels == 0);
            group_size = in_channels * factor / out_channels;
        }
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t B = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            int pad_t = (factor_t - T % factor_t) % factor_t;

            x = ggml_pad_ext(ctx->ggml_ctx, x, 0, 0, 0, 0, pad_t, 0, 0, 0);
            T = x->ne[2];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, factor_t, T / factor_t, C);                                                  // [C, T/factor_t, factor_t, H*W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C, factor_t, T/factor_t, H*W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, factor_s, (H / factor_s) * (T / factor_t), factor_t * C);                        // [C*factor_t, T/factor_t*H/factor_s, factor_s, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C*factor_t, factor_s, T/factor_t*H/factor_s, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s, W / factor_s, (H / factor_s) * (T / factor_t), factor_s * factor_t * C);  // [C*factor_t*factor_s, T/factor_t*H/factor_s, W/factor_s, factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));                                       // [C*factor_t*factor_s, factor_s, T/factor_t*H/factor_s, W/factor_s]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, (W / factor_s) * (H / factor_s) * (T / factor_t), group_size, out_channels);        // [out_channels, group_size, T/factor_t*H/factor_s*W/factor_s]

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [out_channels, T/factor_t*H/factor_s*W/factor_s, group_size]
            x = ggml_mean(ctx->ggml_ctx, x);                                                     // [out_channels, T/factor_t*H/factor_s*W/factor_s, 1]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W / factor_s, H / factor_s, T / factor_t, out_channels);
            return x;
        }
    };

    class DupUp3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s;
        int64_t factor;
        int64_t repeats;

    public:
        DupUp3D(int64_t in_channels, int64_t out_channels, int64_t factor_t, int64_t factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(out_channels * factor % in_channels == 0);
            repeats = out_channels * factor / in_channels;
        }
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    bool first_chunk = false,
                                    int64_t B        = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            auto x_ = x;
            for (int64_t i = 1; i < repeats; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, x_, 2);
            }

            C = out_channels;

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor_s, factor_s * factor_t * C);  // [C*factor_t*factor_s, factor_s, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C*factor_t*factor_s, T*H, W, factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, H * T, factor_s, factor_t * C);  // [C*factor_t, factor_s, T*H, W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C*factor_t, T*H, factor_s, W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, T, factor_t, C);  // [C, factor_t, T, H*factor_s*W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C, T, factor_t, H*factor_s*W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s * H, factor_t * T, C);  // [C, T*factor_t, H*factor_s, W*factor_s]

            if (first_chunk) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, factor_t - 1, x->ne[2]);
            }

            return x;
        }
    };

    class ResidualBlock : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t out_dim;

    public:
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMSNorm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMSNorm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* h = x;
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }

            for (int i = 0; i < 7; i++) {
                if (i == 0 || i == 3) {  // RMSNorm
                    auto layer = std::dynamic_pointer_cast<RMSNorm>(blocks["residual." + std::to_string(i)]);
                    x          = layer->forward(ctx, x);
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
                    x = ggml_silu(ctx->ggml_ctx, x);
                }
            }

            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    class Down_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool down_flag;

    public:
        Down_ResidualBlock(int64_t in_dim,
                           int64_t out_dim,
                           int mult,
                           bool temperal_downsample = false,
                           bool down_flag           = false)
            : mult(mult), down_flag(down_flag) {
            blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new AvgDown3D(in_dim, out_dim, temperal_downsample ? 2 : 1, down_flag ? 2 : 1));

            int i = 0;
            for (; i < mult; i++) {
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                     = out_dim;
            }
            if (down_flag) {
                std::string mode                           = temperal_downsample ? "downsample3d" : "downsample2d";
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                i++;
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* x_copy = x;

            auto avg_shortcut = std::dynamic_pointer_cast<AvgDown3D>(blocks["avg_shortcut"]);

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (down_flag) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
            }

            auto shortcut = avg_shortcut->forward(ctx, x_copy, b);

            x = ggml_add(ctx->ggml_ctx, x, shortcut);

            return x;
        }
    };

    class Up_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool up_flag;

    public:
        Up_ResidualBlock(int64_t in_dim,
                         int64_t out_dim,
                         int mult,
                         bool temperal_upsample = false,
                         bool up_flag           = false)
            : mult(mult), up_flag(up_flag) {
            if (up_flag) {
                blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new DupUp3D(in_dim, out_dim, temperal_upsample ? 2 : 1, up_flag ? 2 : 1));
            }

            int i = 0;
            for (; i < mult; i++) {
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                   = out_dim;
            }
            if (up_flag) {
                std::string mode                         = temperal_upsample ? "upsample3d" : "upsample2d";
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                i++;
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* x_copy = x;

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (up_flag) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);

                auto avg_shortcut = std::dynamic_pointer_cast<DupUp3D>(blocks["avg_shortcut"]);
                auto shortcut     = avg_shortcut->forward(ctx, x_copy, chunk_idx == 0, b);

                x = ggml_add(ctx->ggml_ctx, x, shortcut);
            }

            return x;
        }
    };

    class AttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        AttentionBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
            blocks["to_qkv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 3, {1, 1}));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto to_qkv = std::dynamic_pointer_cast<Conv2d>(blocks["to_qkv"]);
            auto proj   = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            auto identity = x;

            x = norm->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)

            const int64_t n = x->ne[3];
            const int64_t c = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto qkv     = to_qkv->forward(ctx, x);
            auto qkv_vec = split_image_qkv(ctx->ggml_ctx, qkv);

            auto q = qkv_vec[0];
            q      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 2, 0, 1, 3));  // [t, h, w, c]
            q      = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                                      // [t, h * w, c]

            auto k = qkv_vec[1];
            k      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 2, 0, 1, 3));  // [t, h, w, c]
            k      = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                                      // [t, h * w, c]

            auto v = qkv_vec[2];
            v      = ggml_reshape_3d(ctx->ggml_ctx, v, h * w, c, n);  // [t, c, h * w]

            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));                // [t, h * w, c]
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, true, false);  // [t, h * w, c]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [t, c, h * w]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, c, n);                             // [t, c, h, w]

            x = proj->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)

            x = ggml_add(ctx->ggml_ctx, x, identity);
            return x;
        }
    };

    class Encoder3d : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;

    public:
        Encoder3d(int64_t dim                           = 128,
                  int64_t z_dim                         = 4,
                  std::vector<int> dim_mult             = {1, 2, 4, 4},
                  int num_res_blocks                    = 2,
                  std::vector<bool> temperal_downsample = {false, true, true})
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_downsample(temperal_downsample) {
            
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }

            // SeedVR2 VAE likely uses 3 channels input
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                
                for (int j = 0; j < num_res_blocks; j++) {
                    auto block                                       = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                    blocks["downsamples." + std::to_string(index++)] = block;
                    in_dim                                           = out_dim;
                }

                if (i != dim_mult.size() - 1) {
                    std::string mode                                 = temperal_downsample[i] ? "downsample3d" : "downsample2d";
                    auto block                                       = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                    blocks["downsamples." + std::to_string(index++)] = block;
                }
            }

            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(out_dim));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));

            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMSNorm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, z_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMSNorm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // downsamples
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                for (int j = 0; j < num_res_blocks; j++) {
                    auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }

                if (i != dim_mult.size() - 1) {
                    auto layer = std::dynamic_pointer_cast<Resample>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                }
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class Decoder3d : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;

    public:
        Decoder3d(int64_t dim                         = 128,
                  int64_t z_dim                       = 4,
                  std::vector<int> dim_mult           = {1, 2, 4, 4},
                  int num_res_blocks                  = 2,
                  std::vector<bool> temperal_upsample = {true, true, false})
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_upsample(temperal_upsample) {
            
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }

            // init block
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            // middle blocks
            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(dims[0]));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));

            // upsample blocks
            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                
                if (i == 1 || i == 2 || i == 3) {
                    in_dim = in_dim / 2;
                }
                for (int j = 0; j < num_res_blocks + 1; j++) {
                    auto block                                     = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                    blocks["upsamples." + std::to_string(index++)] = block;
                    in_dim                                         = out_dim;
                }

                if (i != dim_mult.size() - 1) {
                    std::string mode                               = temperal_upsample[i] ? "upsample3d" : "upsample2d";
                    auto block                                     = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                    blocks["upsamples." + std::to_string(index++)] = block;
                }
            }

            // output blocks
            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMSNorm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMSNorm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // upsamples
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                for (int j = 0; j < num_res_blocks + 1; j++) {
                    auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                }

                if (i != dim_mult.size() - 1) {
                    auto layer = std::dynamic_pointer_cast<Resample>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                }
            }

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class SeedVR2VAE : public GGMLBlock {
    public:
        bool decode_only                      = true;
        int64_t dim                           = 128;
        int64_t dec_dim                       = 128; // Assuming symmetry unless verified otherwise
        int64_t z_dim                         = 16;  // Standard for SeedVR
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};

        int _conv_num = 33;
        int _conv_idx = 0;
        std::vector<struct ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<struct ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<struct ggml_tensor*>(_conv_num, nullptr);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<struct ggml_tensor*>(_enc_conv_num, nullptr);
        }

    public:
        SeedVR2VAE(bool decode_only = true)
            : decode_only(decode_only) {
            
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_upsample));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x,
                                     int64_t patch_size,
                                     int64_t b = 1) {
            // SeedVR2 doesn't seem to use patchify in VAE, unlike Wan2.2
            // Keeping for structure if needed later, but standard VAE doesn't need it.
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t patch_size,
                                       int64_t b = 1) {
            // SeedVR2 doesn't seem to use unpatchify in VAE
            return x;
        }

        struct ggml_tensor* encode(GGMLRunnerContext* ctx,
                                   struct ggml_tensor* x,
                                   int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            struct ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _enc_conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);  // [b*c, 1, h, w]
                    out     = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1 + 4 * (i - 1), 1 + 4 * i);  // [b*c, 4, h, w]
                    auto out_ = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            out     = conv1->forward(ctx, out);
            auto mu = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            clear_cache();
            return mu;
        }

        struct ggml_tensor* decode(GGMLRunnerContext* ctx,
                                   struct ggml_tensor* z,
                                   int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            struct ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    out     = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            clear_cache();
            return out;
        }

        struct ggml_tensor* decode_partial(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* z,
                                           int i,
                                           int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto x    = conv2->forward(ctx, z);
            auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
            _conv_idx = 0;
            auto out  = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
            return out;
        }
    };

    struct SeedVR2VAERunner : public VAE {
        bool decode_only = true;
        SeedVR2VAE ae;

        SeedVR2VAERunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false)
            : decode_only(decode_only), ae(decode_only), VAE(backend, offload_params_to_cpu) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "seedvr2_vae";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
            struct ggml_cgraph* gf = new_graph_custom(10240 * z->ne[2]);

            z = to_backend(z);

            auto runner_ctx = get_context();

            struct ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        struct ggml_cgraph* build_graph_partial(struct ggml_tensor* z, bool decode_graph, int i) {
            struct ggml_cgraph* gf = new_graph_custom(20480);

            ae.clear_cache();

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                auto feat_cache        = get_cache_tensor_by_name("feat_idx:" + std::to_string(feat_idx));
                ae._feat_map[feat_idx] = feat_cache;
            }

            z = to_backend(z);

            auto runner_ctx = get_context();

            struct ggml_tensor* out = decode_graph ? ae.decode_partial(&runner_ctx, z, i) : ae.encode(&runner_ctx, z);

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                ggml_tensor* feat_cache = ae._feat_map[feat_idx];
                if (feat_cache != nullptr) {
                    cache("feat_idx:" + std::to_string(feat_idx), feat_cache);
                    ggml_build_forward_expand(gf, feat_cache);
                }
            }

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(const int n_threads,
                     struct ggml_tensor* z,
                     bool decode_graph,
                     struct ggml_tensor** output,
                     struct ggml_context* output_ctx = nullptr) override {
            if (true) {
                auto get_graph = [&]() -> struct ggml_cgraph* {
                    return build_graph(z, decode_graph);
                };
                return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
            } else {  // chunk 1 result is weird
                ae.clear_cache();
                int64_t t      = z->ne[2];
                int i          = 0;
                auto get_graph = [&]() -> struct ggml_cgraph* {
                    return build_graph_partial(z, decode_graph, i);
                };
                struct ggml_tensor* out = nullptr;
                bool res                = GGMLRunner::compute(get_graph, n_threads, true, &out, output_ctx);
                ae.clear_cache();
                if (t == 1) {
                    *output = out;
                    return res;
                }

                *output = ggml_new_tensor_4d(output_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], (t - 1) * 4 + 1, out->ne[3]);

                auto copy_to_output = [&]() {
                    for (int64_t i3 = 0; i3 < out->ne[3]; i3++) {
                        for (int64_t i2 = 0; i2 < out->ne[2]; i2++) {
                            for (int64_t i1 = 0; i1 < out->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < out->ne[0]; i0++) {
                                    float value    = ggml_ext_tensor_get_f32(out, i0, i1, i2, i3);
                                    int64_t offset = (i == 0) ? 0 : (1 + (i - 1) * 4);
                                    ggml_ext_tensor_set_f32(*output, value, i0, i1, offset + i2, i3);
                                }
                            }
                        }
                    }
                };

                copy_to_output();

                out = ggml_new_tensor_4d(output_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], 4, out->ne[3]);

                for (i = 1; i < t; i++) {
                    res = res || GGMLRunner::compute(get_graph, n_threads, true, &out);
                    ae.clear_cache();
                    copy_to_output();
                }
                free_cache_ctx_and_buffer();
                return res;
            }
        }
    };

    // --- DiT Implementation (NaDiT / SeedVR2) ---

    static struct ggml_tensor* modulate_add(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_add(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_add(ctx, x, e);
        }
        return x;
    }

    static struct ggml_tensor* modulate_mul(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_mul(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_mul(ctx, x, e);
        }
        return x;
    }

    class SeedVR2SelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t head_dim;

    public:
        SeedVR2SelfAttention(int64_t dim,
                         int64_t num_heads,
                         bool qk_norm = true,
                         float eps    = 1e-6)
            : num_heads(num_heads) {
            head_dim    = dim / num_heads;
            blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, true)); // qkv bias=True usually
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, true));

            if (qk_norm) {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new Identity());
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                            struct ggml_tensor* x) {
            // x: [N, n_token, dim]
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto qkv = qkv_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv); // [q, k, v] each [N, n_token, dim]

            auto q = qkv_vec[0];
            auto k = qkv_vec[1];
            auto v = qkv_vec[2];

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]

            // Standard attention
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, false, ctx->flash_attn_enabled);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    class SeedVR2Block : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        SeedVR2Block(int64_t dim,
                          int64_t ffn_dim,
                          int64_t num_heads,
                          bool qk_norm         = true,
                          float eps            = 1e-6)
            : dim(dim) {
            blocks["norm1"]     = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["attn"]      = std::shared_ptr<GGMLBlock>(new SeedVR2SelfAttention(dim, num_heads, qk_norm, eps));
            
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));

            blocks["mlp.fc1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            blocks["mlp.fc2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
            // act is Silu usually or GeluTanh
        }

        virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                            struct ggml_tensor* x,
                                            struct ggml_tensor* e) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // return [N, n_token, dim]

            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 6, dim] or [N, T, 6, dim]
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);  // ([N, 1, dim], ...) or [N, T, 1, dim]

            auto norm1      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn       = std::dynamic_pointer_cast<SeedVR2SelfAttention>(blocks["attn"]);
            auto norm2      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto mlp_fc1    = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"]);
            auto mlp_fc2    = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"]);

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = attn->forward(ctx, y);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            // ffn
            y = norm2->forward(ctx, x);
            y = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[4]));
            y = modulate_add(ctx->ggml_ctx, y, es[3]);

            y = mlp_fc1->forward(ctx, y);
            y = ggml_silu_inplace(ctx->ggml_ctx, y); // Assuming SiLU, check if GELU is needed
            y = mlp_fc2->forward(ctx, y);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));

            return x;
        }
    };

    struct SeedVR2Params {
        int64_t dim = 2048; // 3B model default
        int64_t ffn_dim = 8192; // 4x dim usually
        int64_t num_heads = 16;
        int num_layers = 28;
        int64_t patch_size = 2;
        int64_t in_channels = 16; // Latent channels
        int64_t out_channels = 16;
        float eps = 1e-6f;
    };

    class SeedVR2DiT : public GGMLBlock {
    protected:
        SeedVR2Params params;

    public:
        SeedVR2DiT(SeedVR2Params params)
            : params(params) {
            
            // Input embedding
            blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(new Conv2d(params.in_channels, params.dim, {params.patch_size, params.patch_size}, {params.patch_size, params.patch_size}));
            
            // Time/Text embedding handling (Simplified for now)
            // t_embedder, y_embedder etc. need to be added based on specific config

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::shared_ptr<GGMLBlock>(new SeedVR2Block(params.dim, params.ffn_dim, params.num_heads, true, params.eps));
                blocks["blocks." + std::to_string(i)] = block;
            }

            // Final layer (Norm + Linear)
            blocks["final_layer.norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.dim, params.eps));
            blocks["final_layer.linear"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.patch_size * params.patch_size * params.out_channels, true));
            // AdaLN modulation for final layer usually has 2 components (shift, scale)
            blocks["final_layer.adaLN_modulation"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, 2 * params.dim, true));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* t,
                                    struct ggml_tensor* context) {
            // Forward pass
            // x: [N, C, H, W]
            
            // Patch embed
            auto x_embedder = std::dynamic_pointer_cast<Conv2d>(blocks["x_embedder"]);
            x = x_embedder->forward(ctx, x); // [N, dim, H/p, W/p]
            // Reshape to sequence [N, L, dim]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0]*x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3)); 

            // Time embedding & Modulation generation (Placeholder - needs implementation of t_embedder)
            // Assuming 't' is already the modulation vector for simplicity in this stub
            auto e = t; 

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<SeedVR2Block>(blocks["blocks." + std::to_string(i)]);
                x = block->forward(ctx, x, e);
            }

            // Final layer
            auto final_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["final_layer.norm"]);
            auto final_linear = std::dynamic_pointer_cast<Linear>(blocks["final_layer.linear"]);
            auto final_adaLN = std::dynamic_pointer_cast<Linear>(blocks["final_layer.adaLN_modulation"]);

            auto m = final_adaLN->forward(ctx, ggml_silu(ctx->ggml_ctx, e)); // [N, 2*dim]
            // Split m into shift, scale
            // ... (implement modulation logic similar to blocks)
            
            x = final_norm->forward(ctx, x);
            // Apply shift/scale
            x = final_linear->forward(ctx, x);

            // Unpatchify
            // ...

            return x;
        }
    };

    // DiT Runner
    struct SeedVR2DiTRunner : public GGMLRunner {
        SeedVR2DiT model;

        SeedVR2DiTRunner(ggml_backend_t backend, bool offload_params_to_cpu, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "")
            : GGMLRunner(backend, offload_params_to_cpu), model(SeedVR2Params()) { // Use default params for now
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "seedvr2_dit"; }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        // Compute function for inference
        bool compute(int n_threads, struct ggml_tensor* x, struct ggml_tensor* t, struct ggml_tensor* context, struct ggml_tensor** output) {
             // Graph build & compute
             return true;
        }
    };

} // namespace SeedVR2

#endif // __SEEDVR2_HPP__