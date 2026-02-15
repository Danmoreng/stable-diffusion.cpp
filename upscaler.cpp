#include "esrgan.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "seedvr2.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include <algorithm>
#include <cmath>
#include <limits>

struct UpscalerGGML {
    ggml_backend_t backend    = nullptr;  // general backend
    ggml_type model_data_type = GGML_TYPE_F16;
    std::shared_ptr<ESRGAN> esrgan_upscaler;
    std::shared_ptr<SeedVR2::SeedVR2DiTRunner> seedvr2_dit;
    std::shared_ptr<SeedVR2::SeedVR2VAERunner> seedvr2_vae;
    std::string esrgan_path;
    int n_threads;
    bool direct   = false;
    int tile_size = 128;
    bool is_seedvr2 = false;

    UpscalerGGML(int n_threads,
                 bool direct   = false,
                 int tile_size = 128)
        : n_threads(n_threads),
          direct(direct),
          tile_size(tile_size) {
    }

    bool load_from_file(const std::string& esrgan_path,
                        const std::string& vae_path,
                        bool offload_params_to_cpu,
                        int n_threads) {
        ggml_log_set(ggml_log_callback_default, nullptr);
#ifdef SD_USE_CUDA
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        backend = ggml_backend_vk_init(0);
#endif
#ifdef SD_USE_OPENCL
        LOG_DEBUG("Using OpenCL backend");
        backend = ggml_backend_opencl_init();
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif
        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }

        // Check for SeedVR2
        if (esrgan_path.find("seedvr") != std::string::npos) {
            is_seedvr2 = true;
            LOG_INFO("Detected SeedVR2 model from path: %s", esrgan_path.c_str());
            
            // Assume VAE is in same dir or passed (TODO: handle VAE path properly)
            // For now, load DiT. 
            // We need a way to specify VAE path. Maybe derive from esrgan_path?
            // E.g. if path is "seedvr2_dit.safetensors", look for "seedvr2_vae.safetensors"
            
            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(esrgan_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
                return false;
            }
            model_loader.set_wtype_override(model_data_type);
            
            seedvr2_dit = std::make_shared<SeedVR2::SeedVR2DiTRunner>(backend, offload_params_to_cpu, model_loader.get_tensor_storage_map());
            if (!seedvr2_dit->load_from_file(esrgan_path)) { // GGMLRunner load_from_file
                 LOG_ERROR("Failed to load SeedVR2 DiT");
                 return false;
            }
            
            // Try to find VAE
            std::string final_vae_path = vae_path;
            if (final_vae_path.empty()) {
                final_vae_path = "models/seedvr/ema_vae_fp16.safetensors"; // Default fallback
            }
            // TODO: Better path resolution
            
            ModelLoader vae_loader;
            if (vae_loader.init_from_file_and_convert_name(final_vae_path)) {
                vae_loader.set_wtype_override(model_data_type);
                seedvr2_vae = std::make_shared<SeedVR2::SeedVR2VAERunner>(backend, offload_params_to_cpu, vae_loader.get_tensor_storage_map());
                if (!seedvr2_vae->load_from_file(final_vae_path)) {
                    LOG_ERROR("Failed to load SeedVR2 VAE from %s", final_vae_path.c_str());
                    return false;
                }
            } else {
                LOG_ERROR("Could not find SeedVR2 VAE at default location: %s", final_vae_path.c_str());
                return false;
            }
            
            return true;
        }

        // Default ESRGAN
        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(esrgan_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        }
        model_loader.set_wtype_override(model_data_type);
        LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
        esrgan_upscaler = std::make_shared<ESRGAN>(backend, offload_params_to_cpu, tile_size, model_loader.get_tensor_storage_map());
        if (direct) {
            esrgan_upscaler->set_conv2d_direct_enabled(true);
        }
        if (!esrgan_upscaler->load_from_file(esrgan_path, n_threads)) {
            return false;
        }
        return true;
    }

    sd_image_t upscale(sd_image_t input_image, uint32_t upscale_factor) {
        if (is_seedvr2) {
            // SeedVR2 Upscaling Logic
            int target_width = input_image.width * upscale_factor;
            int target_height = input_image.height * upscale_factor;
            const bool seedvr2_debug = getenv("SD_SEEDVR2_DEBUG") != nullptr;
            const bool seedvr2_loopback = getenv("SD_SEEDVR2_LOOPBACK") != nullptr;
            
            // Ensure dimensions are divisible by 16 (VAE requirement)
            target_width = (target_width / 16) * 16;
            target_height = (target_height / 16) * 16;

            LOG_INFO("SeedVR2 upscaling to %dx%d", target_width, target_height);

            auto tensor_to_f32 = [&](ggml_tensor * t, std::vector<float> & out) {
                const int64_t n = ggml_nelements(t);
                out.resize(n);

                if (t->buffer) {
                    if (t->type == GGML_TYPE_F32) {
                        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
                        return;
                    }

                    if (t->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> tmp(n);
                        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
                        ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
                        return;
                    }
                } else {
                    // Host memory fallback
                     if (t->type == GGML_TYPE_F32) {
                        memcpy(out.data(), t->data, n * sizeof(float));
                        return;
                    }
                    if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((ggml_fp16_t*)t->data, out.data(), n);
                        return;
                    }
                }

                LOG_ERROR("tensor_to_f32: unsupported type %s", ggml_type_name(t->type));
            };

            auto require_shape = [&](const char * name, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) -> bool {
                if (t == nullptr) {
                    LOG_ERROR("%s is null", name);
                    return false;
                }
                const int64_t got[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                const int64_t exp[4] = { ne0, ne1, ne2, ne3 };
                for (int i = 0; i < 4; ++i) {
                    if (exp[i] >= 0 && got[i] != exp[i]) {
                        LOG_ERROR("%s shape mismatch at dim %d: got %lld expected %lld. Full shape=[%lld,%lld,%lld,%lld]",
                                  name, i, (long long)got[i], (long long)exp[i],
                                  (long long)got[0], (long long)got[1], (long long)got[2], (long long)got[3]);
                        return false;
                    }
                }
                return true;
            };

            auto require_nelements = [&](const char * name, ggml_tensor * t, int64_t expected) -> bool {
                if (t == nullptr) {
                    LOG_ERROR("%s is null", name);
                    return false;
                }
                const int64_t got = ggml_nelements(t);
                if (got != expected) {
                    LOG_ERROR("%s element count mismatch: got %lld expected %lld", name, (long long)got, (long long)expected);
                    return false;
                }
                return true;
            };

            auto check_finite_tensor = [&](const char * name, ggml_tensor * t, bool log_stats = false) -> bool {
                if (t == nullptr) {
                    LOG_ERROR("%s is null", name);
                    return false;
                }
                std::vector<float> host;
                tensor_to_f32(t, host);
                if (host.empty()) {
                    LOG_ERROR("%s could not be converted to float host data", name);
                    return false;
                }
                double sum = 0.0;
                double sum_sq = 0.0;
                float min_val = std::numeric_limits<float>::infinity();
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < host.size(); ++i) {
                    const float v = host[i];
                    if (!std::isfinite(v)) {
                        LOG_ERROR("%s has non-finite value at index %zu: %f", name, i, v);
                        return false;
                    }
                    sum += v;
                    sum_sq += (double)v * (double)v;
                    min_val = std::min(min_val, v);
                    max_val = std::max(max_val, v);
                }
                if (log_stats) {
                    const double n = (double)host.size();
                    const double mean = sum / n;
                    const double var = std::max(0.0, sum_sq / n - mean * mean);
                    const double stddev = std::sqrt(var);
                    LOG_INFO("%s stats: mean=%.6f std=%.6f min=%.6f max=%.6f n=%lld",
                             name, mean, stddev, min_val, max_val, (long long)host.size());
                }
                return true;
            };

            auto check_finite_host = [&](const char * name, const std::vector<float> & host, bool log_stats = false) -> bool {
                if (host.empty()) {
                    LOG_ERROR("%s is empty", name);
                    return false;
                }
                double sum = 0.0;
                double sum_sq = 0.0;
                float min_val = std::numeric_limits<float>::infinity();
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < host.size(); ++i) {
                    const float v = host[i];
                    if (!std::isfinite(v)) {
                        LOG_ERROR("%s has non-finite value at index %zu: %f", name, i, v);
                        return false;
                    }
                    sum += v;
                    sum_sq += (double)v * (double)v;
                    min_val = std::min(min_val, v);
                    max_val = std::max(max_val, v);
                }
                if (log_stats) {
                    const double n = (double)host.size();
                    const double mean = sum / n;
                    const double var = std::max(0.0, sum_sq / n - mean * mean);
                    const double stddev = std::sqrt(var);
                    LOG_INFO("%s stats: mean=%.6f std=%.6f min=%.6f max=%.6f n=%lld",
                             name, mean, stddev, min_val, max_val, (long long)host.size());
                }
                return true;
            };

            // 1. Resize Image (Float Precision)
            int channels = 3;
            std::vector<float> input_data_f(input_image.width * input_image.height * channels);
            for (size_t i = 0; i < input_data_f.size(); ++i) {
                input_data_f[i] = (float)input_image.data[i] / 255.0f;
            }

            std::vector<float> resized_data_f(target_width * target_height * channels);
            stbir_resize_float(input_data_f.data(), input_image.width, input_image.height, 0,
                               resized_data_f.data(), target_width, target_height, 0, channels);

            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(2048 * 1024) * 1024; // 2GB buffer
            params.mem_buffer = nullptr;
            params.no_alloc   = false;
            struct ggml_context* work_ctx = ggml_init(params);

            // 2. Image to Tensor
            ggml_tensor* x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, target_width, target_height, 3, 1);
            
            {
                float* x_ptr = (float*)x->data;
                // Fill tensor from resized_data_f (Interleaved RGB -> Planar RGB) and normalize -1..1
                for (int y = 0; y < target_height; y++) {
                    for (int x_c = 0; x_c < target_width; x_c++) {
                        for (int c = 0; c < 3; c++) {
                            // Source: Interleaved [H, W, C]
                            float val = resized_data_f[(y * target_width + x_c) * 3 + c];
                            
                            // Dest: Planar [W, H, C] (GGML standard: W fastest, then H, then C)
                            // Index = c * (W*H) + y * W + x
                            int dst_idx = c * target_width * target_height + y * target_width + x_c;
                            
                            x_ptr[dst_idx] = val * 2.0f - 1.0f;
                        }
                    }
                }
            }
            if (!require_shape("seedvr2.vae_encode_input", x, target_width, target_height, 3, 1) ||
                !check_finite_tensor("seedvr2.vae_encode_input", x, seedvr2_debug)) {
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            // 3. VAE Encode
            if (getenv("SD_DUMP_TENSORS")) {
                FILE* f = fopen("cpp_vae_input.bin", "wb");
                if (f) {
                    fwrite(x->data, 1, ggml_nbytes(x), f);
                    fclose(f);
                    LOG_INFO("Dumped cpp_vae_input.bin");
                }
            }

            ggml_tensor* latents = nullptr;
            if (!seedvr2_vae->compute(n_threads, x, false, &latents, work_ctx)) { // false = encode
                LOG_ERROR("SeedVR2 VAE encode failed");
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }
            if (latents == nullptr || latents->ne[0] <= 0 || latents->ne[1] <= 0) {
                LOG_ERROR("SeedVR2 VAE encode returned invalid latents");
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }
            const bool latent_shape_ok =
                (latents->ne[2] == 16 && latents->ne[3] == 1) ||
                (latents->ne[2] == 1 && latents->ne[3] == 16);
            if (!latent_shape_ok) {
                LOG_ERROR("Unexpected latent shape: [%lld,%lld,%lld,%lld], expected channel=16 with T=1",
                          (long long)latents->ne[0], (long long)latents->ne[1],
                          (long long)latents->ne[2], (long long)latents->ne[3]);
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }
            if (!require_nelements("seedvr2.vae_latents", latents, latents->ne[0] * latents->ne[1] * 16) ||
                !check_finite_tensor("seedvr2.vae_latents", latents, true)) {
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            if (getenv("SD_DUMP_TENSORS")) {
                FILE* f = fopen("cpp_latents.bin", "wb");
                if (f) {
                    fwrite(latents->data, 1, ggml_nbytes(latents), f);
                    fclose(f);
                    LOG_INFO("Dumped cpp_latents.bin");
                }
            }

            // 4. Prepare DiT Inputs
            // latents from VAE encode: [W/8, H/8, 16, 1]
            int lw = latents->ne[0];
            int lh = latents->ne[1];
            
            // Fetch latents from backend to host
            std::vector<float> latents_host;
            tensor_to_f32(latents, latents_host);

            LOG_INFO("latents type=%s nbytes=%zu ne=%ld",
                 ggml_type_name(latents->type),
                 ggml_nbytes(latents),
                 ggml_nelements(latents));

            // Create x_t (noisy latents) on host for reference/noise addition
            // For SR, x_t is initialized with the latents (or noise+latents). 
            // Here we just copy latents as per previous logic (zero noise assumption for now or simple copy)
            std::vector<float> x_t_host = latents_host; 

            // Patchify parameters
            int patch_size = 2;
            int C_in = 33; // 16 (noisy) + 16 (cond) + 1 (mask)
            if ((lw % patch_size) != 0 || (lh % patch_size) != 0) {
                LOG_ERROR("Latent dimensions must be divisible by patch size %d, got lw=%d lh=%d", patch_size, lw, lh);
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }
            int n_tokens_w = lw / patch_size;
            int n_tokens_h = lh / patch_size;
            int n_tokens = n_tokens_w * n_tokens_h;
            int input_dim = C_in * patch_size * patch_size; // 33 * 4 = 132

            // Create dit_input: [132, N_tokens, 1, 1]
            // GGML shape is [ne0, ne1, ne2, ne3] -> [132, n_tokens, 1, 1]
            ggml_tensor* dit_input = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, input_dim, n_tokens);
            
            std::vector<float> dit_input_host(ggml_nelements(dit_input));
            float* dst = dit_input_host.data();

            // Source pointers
            const float* src_xt = x_t_host.data();
            const float* src_lc = latents_host.data(); // latent cond

            // Helper to get pixel value from (x, y, c) in [W, H, C] layout
            // latents are [W, H, 16]. stride_w = 1, stride_h = W, stride_c = W*H (if planar)
            // Wait, sd_image_to_ggml_tensor creates standard layout.
            // But VAE output layout might be different. 
            // Checked SeedVR2VAE: x = ggml_ext_slice(..., 3, 0, 16).
            // ggml_ext_slice preserves layout. Conv3d output is [W, H, T, C] (permuted back).
            // So stride is: C is fastest? No, usually GGML is column major.
            // ne[0]=W, ne[1]=H, ne[2]=T, ne[3]=C.
            // Element at (x, y, t, c) is data[x + y*W + t*W*H + c*W*H*T].
            // Here T=1. So data[x + y*W + c*W*H].
            
            int W = lw;
            int H = lh;
            int stride_y = W;
            int stride_c = W * H;

            for (int ty = 0; ty < n_tokens_h; ty++) {
                for (int tx = 0; tx < n_tokens_w; tx++) {
                    int token_idx = ty * n_tokens_w + tx;
                    float* token_dst = dst + token_idx * input_dim;
                    
                    // We fill 132 elements for this token.
                    // Order: c varies slowest, py, px fastest? 
                    // Verify: rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
                    // Inner dim is (c p1 p2). c is outer of that block.
                    // So loop c, then py, then px.
                    
                    int dst_idx = 0;

                    // 1. x_t (16 channels)
                    for (int c = 0; c < 16; c++) {
                        for (int py = 0; py < patch_size; py++) {
                            for (int px = 0; px < patch_size; px++) {
                                int sx = tx * patch_size + px;
                                int sy = ty * patch_size + py;
                                int src_idx = sx + sy * stride_y + c * stride_c;
                                token_dst[dst_idx++] = src_xt[src_idx];
                            }
                        }
                    }

                    // 2. latents_cond (16 channels)
                    for (int c = 0; c < 16; c++) {
                        for (int py = 0; py < patch_size; py++) {
                            for (int px = 0; px < patch_size; px++) {
                                int sx = tx * patch_size + px;
                                int sy = ty * patch_size + py;
                                int src_idx = sx + sy * stride_y + c * stride_c;
                                token_dst[dst_idx++] = src_lc[src_idx];
                            }
                        }
                    }

                    // 3. mask (1 channel)
                    for (int py = 0; py < patch_size; py++) {
                        for (int px = 0; px < patch_size; px++) {
                            token_dst[dst_idx++] = 1.0f;
                        }
                    }
                }
            }
            if (!check_finite_host("seedvr2.dit_input_host", dit_input_host, seedvr2_debug)) {
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }
            
            // Copy patchified data to device
            // We explicitly allocate backend buffer for input to avoid GGMLRunner::to_backend issues
            ggml_backend_buffer_t dit_input_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(dit_input));
            if (dit_input_buffer) {
                dit_input->buffer = dit_input_buffer;
                dit_input->data = ggml_backend_buffer_get_base(dit_input_buffer);
                ggml_backend_tensor_set(dit_input, dit_input_host.data(), 0, ggml_nbytes(dit_input));
            } else {
                LOG_ERROR("Failed to allocate backend buffer for dit_input");
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            // t = noise level (0 for upscale usually, or small noise)
            ggml_tensor* t = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
            ggml_set_f32(t, 0.0f); 
            // Also move t to backend
            ggml_backend_buffer_t t_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(t));
            if (t_buffer) {
                t->buffer = t_buffer;
                t->data = ggml_backend_buffer_get_base(t_buffer);
                float t_val = 0.0f;
                ggml_backend_tensor_set(t, &t_val, 0, sizeof(float));
            }

            // Context needs to be [in_dim, seq_len]
            // For SeedVR2 3B, txt_in_dim = 5120
            ggml_tensor* context = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 5120, 77); 
            ggml_set_f32(context, 0.0f); // Empty context
            // Also move context to backend
            ggml_backend_buffer_t context_buffer = ggml_backend_alloc_buffer(backend, ggml_nbytes(context));
            if (context_buffer) {
                context->buffer = context_buffer;
                context->data = ggml_backend_buffer_get_base(context_buffer);
                std::vector<float> zeros(ggml_nelements(context), 0.0f);
                ggml_backend_tensor_set(context, zeros.data(), 0, ggml_nbytes(context));
            }

            // 5. DiT Upscale
            ggml_tensor* upscaled_latents = nullptr;
            if (seedvr2_loopback) {
                LOG_WARN("SD_SEEDVR2_LOOPBACK enabled: bypassing DiT and feeding VAE latents directly to decoder.");
                upscaled_latents = latents;
            } else {
                if (!seedvr2_dit->compute(n_threads, dit_input, t, context, &upscaled_latents, work_ctx)) {
                    LOG_ERROR("SeedVR2 DiT upscale failed");
                    if (dit_input_buffer) ggml_backend_buffer_free(dit_input_buffer);
                    if (t_buffer) ggml_backend_buffer_free(t_buffer);
                    if (context_buffer) ggml_backend_buffer_free(context_buffer);
                    ggml_free(work_ctx);
                    return {0, 0, 0, nullptr};
                }
                if (!require_nelements("seedvr2.dit_output", upscaled_latents, (int64_t)n_tokens * patch_size * patch_size * 16) ||
                    !check_finite_tensor("seedvr2.dit_output", upscaled_latents, true)) {
                    if (dit_input_buffer) ggml_backend_buffer_free(dit_input_buffer);
                    if (t_buffer) ggml_backend_buffer_free(t_buffer);
                    if (context_buffer) ggml_backend_buffer_free(context_buffer);
                    ggml_free(work_ctx);
                    return {0, 0, 0, nullptr};
                }
            }
            
            // Clean up input buffers (if they were allocated)
            if (dit_input_buffer) ggml_backend_buffer_free(dit_input_buffer);
            if (t_buffer) ggml_backend_buffer_free(t_buffer);
            if (context_buffer) ggml_backend_buffer_free(context_buffer);

            if (!seedvr2_loopback && getenv("SD_DUMP_TENSORS")) {
                FILE* f = fopen("cpp_upscaled_latents.bin", "wb");
                if (f) {
                    fwrite(upscaled_latents->data, 1, ggml_nbytes(upscaled_latents), f);
                    fclose(f);
                    LOG_INFO("Dumped cpp_upscaled_latents.bin");
                }
            }

            // 6. Unpatchify & VAE Decode
            // upscaled_latents from DiT: [64, N_tokens]
            // We need to reshape to [lw, lh, 1, 16]
            float vae_scaling_factor = 0.9152f;
            
            ggml_tensor* unpatchified = nullptr;
            
            if (upscaled_latents == latents) {
                // SKIP UNPATCHIFY for loopback test
                unpatchified = latents;
                LOG_INFO("VAE Loopback: Skipping unpatchify, using latents directly.");
            } else {
                LOG_INFO("Unpatchifying DiT output... (lw=%d, lh=%d, n_tokens=%d, scaling=%f)", lw, lh, n_tokens, vae_scaling_factor);
                unpatchified = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, lw, lh, 1, 16);
                
                if (upscaled_latents == nullptr) {
                    LOG_ERROR("upscaled_latents is NULL!");
                    ggml_free(work_ctx);
                    return {0, 0, 0, nullptr};
                }
                
                LOG_INFO("Unpatchifying DiT output... (lw=%d, lh=%d, n_tokens=%d, scaling=%f)", lw, lh, n_tokens, vae_scaling_factor);
                
                // Copy to host vector for easier iteration
                std::vector<float> upscaled_host;
                tensor_to_f32(upscaled_latents, upscaled_host);

                LOG_INFO("dit_out type=%s nbytes=%zu ne=%ld",
                     ggml_type_name(upscaled_latents->type),
                     ggml_nbytes(upscaled_latents),
                     ggml_nelements(upscaled_latents));
                
                float* src = upscaled_host.data();
                float* dst = (float*)unpatchified->data;
                
                int patch_size_w = 2;
                int patch_size_h = 2;
                int channels = 16; // from config
                int n_patches_w = lw / patch_size_w; // 64
                
                // Validate dimensions
                if (ggml_nelements(upscaled_latents) != n_tokens * patch_size_w * patch_size_h * channels) {
                     LOG_ERROR("Dimension mismatch in unpatchify! DiT output size: %lld, expected: %d",
                               (long long)ggml_nelements(upscaled_latents),
                               n_tokens * patch_size_w * patch_size_h * channels);
                     ggml_free(work_ctx);
                     return {0, 0, 0, nullptr};
                }

                // Iterate tokens (patches)
                for (int i = 0; i < n_tokens; i++) {
                    int patch_y = i / n_patches_w;
                    int patch_x = i % n_patches_w;
                    
                    // Base offset for this token in the source array
                    int token_offset = i * (channels * patch_size_w * patch_size_h);
                    
                    for (int py = 0; py < patch_size_h; py++) {
                        for (int px = 0; px < patch_size_w; px++) {
                            for (int c = 0; c < channels; c++) {
                                // CORRECTED READ LOGIC:
                                // Assume DiT output is Planar: [Channel, Height, Width]
                                // Index = c * (H*W) + py * W + px
                                int src_idx_rel = c * (patch_size_h * patch_size_w) + py * patch_size_w + px;
                                float val = upscaled_host[token_offset + src_idx_rel];
                                
                                // Destination Logic (Unchanged, this is correct for GGML)
                                int global_x = patch_x * patch_size_w + px;
                                int global_y = patch_y * patch_size_h + py;
                                
                                // GGML Tensor [W, H, C] -> Index = x + y*W + c*(W*H)
                                int dst_idx = global_x + global_y * lw + c * (lw * lh);
                                
                                dst[dst_idx] = val / vae_scaling_factor;
                            }
                        }
                    }
                }
                LOG_INFO("Unpatchify completed.");
            }
            if (!require_shape("seedvr2.unpatchified_latents", unpatchified, lw, lh, 1, 16) ||
                !check_finite_tensor("seedvr2.unpatchified_latents", unpatchified, true)) {
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            LOG_INFO("Running VAE decode (Tiled)...");
            
            // Prepare for tiling: View [W, H, 1, 16] as [W, H, 16, 1] so sd_tiling slices correctly
            ggml_tensor* tile_input = ggml_view_4d(work_ctx, unpatchified, lw, lh, 16, 1,
                                                  unpatchified->nb[1], unpatchified->nb[2], unpatchified->nb[3], 0);
            
            // Final output tensor [TargetW, TargetH, 3, 1]
            ggml_tensor* decoded = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, target_width, target_height, 3, 1);
            
            // Tiling params
            int vae_tile_size = 64; // Latent tile size (Input to VAE)
            if (getenv("SD_FORCE_SINGLE_TILE")) {
                vae_tile_size = std::max(std::max((int)lw, (int)lh), 64);
                LOG_WARN("!!! SD_FORCE_SINGLE_TILE ENABLED !!! Tile Size: %d", vae_tile_size);
            }
            float vae_overlap = 0.25f;
            int vae_scale = 8;
            const bool disable_tiling = getenv("SD_DISABLE_TILING") != nullptr;

            LOG_INFO("VAE Tiling: input %dx%d, scale %d, tile_size %d", (int)tile_input->ne[0], (int)tile_input->ne[1], vae_scale, vae_tile_size);

            auto pack_vae_decode_input = [&](ggml_context * ctx, ggml_tensor * src_whc) -> ggml_tensor * {
                // src_whc is logical [W,H,16,1] (channels in ne2). We pack into contiguous [W,H,1,16].
                ggml_tensor * packed = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src_whc->ne[0], src_whc->ne[1], 1, 16);
                float * dst = (float *)packed->data;
                const int W = (int)src_whc->ne[0];
                const int H = (int)src_whc->ne[1];
                const int C = 16;
                for (int c = 0; c < C; ++c) {
                    for (int y = 0; y < H; ++y) {
                        for (int x = 0; x < W; ++x) {
                            const size_t src_off = (size_t)x * src_whc->nb[0] +
                                                   (size_t)y * src_whc->nb[1] +
                                                   (size_t)c * src_whc->nb[2];
                            const float v = *(float *)((char *)src_whc->data + src_off);
                            const size_t dst_idx = (size_t)x + (size_t)y * W + (size_t)c * W * H;
                            dst[dst_idx] = v;
                        }
                    }
                }
                return packed;
            };

            auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                // in: [TileW, TileH, 16, 1] (Tile Input)
                // out: [TileW*8, TileH*8, 3, 1] (Tile Output Buffer)
                
                // Temp context for this tile's graph management
                struct ggml_init_params tile_params = { 128 * 1024 * 1024, nullptr, false };
                struct ggml_context* tile_ctx = ggml_init(tile_params);
                
                // Pack into contiguous [TileW, TileH, 1, 16] for SeedVR2VAE to avoid stride/view ambiguity.
                struct ggml_tensor* vae_in = pack_vae_decode_input(tile_ctx, in);
                
                struct ggml_tensor* vae_out = nullptr;
                if (!seedvr2_vae->compute(n_threads, vae_in, true, &vae_out, tile_ctx)) {
                    LOG_ERROR("Tile VAE decode failed");
                } else {
                    LOG_INFO("Tile VAE raw output shape: [%lld, %lld, %lld, %lld], expected [%lld, %lld, 3, 1]",
                             (long long)vae_out->ne[0], (long long)vae_out->ne[1],
                             (long long)vae_out->ne[2], (long long)vae_out->ne[3],
                             (long long)out->ne[0], (long long)out->ne[1]);
                    if (vae_out->ne[0] != out->ne[0] ||
                        vae_out->ne[1] != out->ne[1] ||
                        vae_out->ne[2] != 3 ||
                        vae_out->ne[3] != 1) {
                        LOG_ERROR("Unexpected tile VAE output shape; raw memcpy may corrupt image layout.");
                    }
                    // vae_out is already [TileW*8, TileH*8, 3, 1] (ae.decode handles permutation)
                    memcpy(out->data, vae_out->data, ggml_nbytes(out));
                }
                
                seedvr2_vae->free_compute_buffer();
                ggml_free(tile_ctx);
            };

            if (disable_tiling) {
                LOG_WARN("!!! SD_DISABLE_TILING ENABLED !!! Running single full-frame VAE decode.");
                struct ggml_init_params full_params = { 256 * 1024 * 1024, nullptr, false };
                struct ggml_context* full_ctx = ggml_init(full_params);
                struct ggml_tensor* vae_in = pack_vae_decode_input(full_ctx, tile_input);
                struct ggml_tensor* vae_out = nullptr;
                if (!seedvr2_vae->compute(n_threads, vae_in, true, &vae_out, full_ctx)) {
                    LOG_ERROR("Full-frame VAE decode failed");
                    seedvr2_vae->free_compute_buffer();
                    ggml_free(full_ctx);
                    ggml_free(work_ctx);
                    return {0, 0, 0, nullptr};
                }
                LOG_INFO("Full-frame VAE raw output shape: [%lld, %lld, %lld, %lld], expected [%d, %d, 3, 1]",
                         (long long)vae_out->ne[0], (long long)vae_out->ne[1],
                         (long long)vae_out->ne[2], (long long)vae_out->ne[3],
                         target_width, target_height);
                if (vae_out->ne[0] != target_width ||
                    vae_out->ne[1] != target_height ||
                    vae_out->ne[2] != 3 ||
                    vae_out->ne[3] != 1) {
                    LOG_ERROR("Unexpected full-frame VAE output shape; raw memcpy may corrupt image layout.");
                }
                memcpy(decoded->data, vae_out->data, ggml_nbytes(decoded));
                seedvr2_vae->free_compute_buffer();
                ggml_free(full_ctx);
            } else {
                sd_tiling(tile_input, decoded, vae_scale, vae_tile_size, vae_overlap, on_tiling);
            }
            if (!require_shape("seedvr2.vae_decoded", decoded, target_width, target_height, 3, 1) ||
                !check_finite_tensor("seedvr2.vae_decoded", decoded, true)) {
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            // 7. Tensor to Image (Denormalize -1..1 -> 0..1 -> 0..255)
            LOG_INFO("Converting decoded tensor to final image... output shape: %dx%d", (int)decoded->ne[0], (int)decoded->ne[1]);
            {
                float* dec_ptr = (float*)decoded->data;
                for (int i = 0; i < ggml_nelements(decoded); i++) {
                    dec_ptr[i] = (dec_ptr[i] + 1.0f) * 0.5f;
                }
            }
            ggml_ext_tensor_clamp_inplace(decoded, 0.0f, 1.0f);
            
            uint8_t* output_data = ggml_tensor_to_sd_image(decoded);
            
            ggml_free(work_ctx);

            return { (uint32_t)target_width, (uint32_t)target_height, 3, output_data };
        }

        // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
        sd_image_t upscaled_image = {0, 0, 0, nullptr};
        int output_width          = (int)input_image.width * esrgan_upscaler->scale;
        int output_height         = (int)input_image.height * esrgan_upscaler->scale;
        LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
                 input_image.width, input_image.height, output_width, output_height);

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        // draft context
        struct ggml_context* upscale_ctx = ggml_init(params);
        if (!upscale_ctx) {
            LOG_ERROR("ggml_init() failed");
            return upscaled_image;
        }
        // LOG_DEBUG("upscale work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);
        ggml_tensor* input_image_tensor = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, input_image.width, input_image.height, 3, 1);
        sd_image_to_ggml_tensor(input_image, input_image_tensor);

        ggml_tensor* upscaled = ggml_new_tensor_4d(upscale_ctx, GGML_TYPE_F32, output_width, output_height, 3, 1);
        auto on_tiling        = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
            esrgan_upscaler->compute(n_threads, in, &out);
        };
        int64_t t0 = ggml_time_ms();
        sd_tiling(input_image_tensor, upscaled, esrgan_upscaler->scale, esrgan_upscaler->tile_size, 0.25f, on_tiling);
        esrgan_upscaler->free_compute_buffer();
        ggml_ext_tensor_clamp_inplace(upscaled, 0.f, 1.f);
        uint8_t* upscaled_data = ggml_tensor_to_sd_image(upscaled);
        ggml_free(upscale_ctx);
        int64_t t3 = ggml_time_ms();
        LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
        upscaled_image = {
            (uint32_t)output_width,
            (uint32_t)output_height,
            3,
            upscaled_data,
        };
        return upscaled_image;
    }
};

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = nullptr;
};

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 const char* vae_path_c_str,
                                 bool offload_params_to_cpu,
                                 bool direct,
                                 int n_threads,
                                 int tile_size) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);
    std::string vae_path = vae_path_c_str ? std::string(vae_path_c_str) : "";

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size);
    if (upscaler_ctx->upscaler == nullptr) {
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, vae_path, offload_params_to_cpu, n_threads)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
        free(upscaler_ctx);
        return nullptr;
    }
    return upscaler_ctx;
}

sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
    return upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}
