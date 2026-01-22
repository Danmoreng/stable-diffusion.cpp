#include "esrgan.hpp"
#include "ggml_extend.hpp"
#include "model.h"
#include "seedvr2.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

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
            std::string vae_path = "models/seedvr/ema_vae_fp16.safetensors"; // Default fallback
            // TODO: Better path resolution
            
            ModelLoader vae_loader;
            if (vae_loader.init_from_file_and_convert_name(vae_path)) {
                vae_loader.set_wtype_override(model_data_type);
                seedvr2_vae = std::make_shared<SeedVR2::SeedVR2VAERunner>(backend, offload_params_to_cpu, vae_loader.get_tensor_storage_map());
                if (!seedvr2_vae->load_from_file(vae_path)) {
                    LOG_ERROR("Failed to load SeedVR2 VAE from %s", vae_path.c_str());
                    return false;
                }
            } else {
                LOG_ERROR("Could not find SeedVR2 VAE at default location: %s", vae_path.c_str());
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
            
            // Ensure dimensions are divisible by 16 (VAE requirement)
            target_width = (target_width / 16) * 16;
            target_height = (target_height / 16) * 16;

            LOG_INFO("SeedVR2 upscaling to %dx%d", target_width, target_height);

            // 1. Resize Image
            std::vector<uint8_t> resized_data(target_width * target_height * 3);
            stbir_resize_uint8(input_image.data, input_image.width, input_image.height, 0,
                               resized_data.data(), target_width, target_height, 0, 3);

            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(2048 * 1024) * 1024; // 2GB buffer
            params.mem_buffer = nullptr;
            params.no_alloc   = false;
            struct ggml_context* work_ctx = ggml_init(params);

            // 2. Image to Tensor (Normalize -1 to 1)
            ggml_tensor* x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, target_width, target_height, 3, 1);
            sd_image_t resized_img = { (uint32_t)target_width, (uint32_t)target_height, 3, resized_data.data() };
            sd_image_to_ggml_tensor(resized_img, x);
            
            {
                float* x_ptr = (float*)x->data;
                for (int i = 0; i < ggml_nelements(x); i++) {
                    x_ptr[i] = x_ptr[i] * 2.0f - 1.0f;
                }
            }

            // 3. VAE Encode
            ggml_tensor* latents = nullptr;
            if (!seedvr2_vae->compute(n_threads, x, false, &latents, work_ctx)) { // false = encode
                LOG_ERROR("SeedVR2 VAE encode failed");
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
            
            // Create x_t (noisy latents): [W/8, H/8, 16, 1]
            // For a single pass, we can use zeros or random noise. 
            // In SR models, it's often initialized with the blurred latent.
            ggml_tensor* x_t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, lw, lh, 16, 1);
            memcpy(x_t->data, latents->data, ggml_nbytes(latents)); 

            // Create dit_input: [W/8, H/8, 33, 1]
            // Channels: [0:16] = x_t, [16:32] = latents_cond, [32] = mask
            ggml_tensor* dit_input = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, lw, lh, 33, 1);
            
            // Let's use a cleaner manual copy loop
            {
                float* dst = (float*)dit_input->data;
                float* src_xt = (float*)x_t->data;
                float* src_lc = (float*)latents->data;
                int n_pix = lw * lh;
                
                for (int c = 0; c < 16; c++) {
                    memcpy(dst + c * n_pix, src_xt + c * n_pix, n_pix * sizeof(float));
                }
                for (int c = 0; c < 16; c++) {
                    memcpy(dst + (16 + c) * n_pix, src_lc + c * n_pix, n_pix * sizeof(float));
                }
                // Mask channel
                for (int i = 0; i < n_pix; i++) {
                    dst[32 * n_pix + i] = 1.0f;
                }
            }

            // t = noise level (0 for upscale usually, or small noise)
            ggml_tensor* t = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
            ggml_set_f32(t, 0.0f); 

            // Context needs to be [in_dim, seq_len]
            // For SeedVR2 3B, txt_in_dim = 5120
            ggml_tensor* context = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 5120, 77); 
            ggml_set_f32(context, 0.0f); // Empty context

            // 5. DiT Upscale
            ggml_tensor* upscaled_latents = nullptr;
            if (!seedvr2_dit->compute(n_threads, dit_input, t, context, &upscaled_latents, work_ctx)) {
                LOG_ERROR("SeedVR2 DiT upscale failed");
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            if (getenv("SD_DUMP_TENSORS")) {
                FILE* f = fopen("cpp_upscaled_latents.bin", "wb");
                if (f) {
                    fwrite(upscaled_latents->data, 1, ggml_nbytes(upscaled_latents), f);
                    fclose(f);
                    LOG_INFO("Dumped cpp_upscaled_latents.bin");
                }
            }

            // 6. VAE Decode
            ggml_tensor* decoded = nullptr;
            if (!seedvr2_vae->compute(n_threads, upscaled_latents, true, &decoded, work_ctx)) { // true = decode
                LOG_ERROR("SeedVR2 VAE decode failed");
                ggml_free(work_ctx);
                return {0, 0, 0, nullptr};
            }

            // 7. Tensor to Image (Denormalize -1..1 -> 0..1 -> 0..255)
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
                                 bool offload_params_to_cpu,
                                 bool direct,
                                 int n_threads,
                                 int tile_size) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size);
    if (upscaler_ctx->upscaler == nullptr) {
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, offload_params_to_cpu, n_threads)) {
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
