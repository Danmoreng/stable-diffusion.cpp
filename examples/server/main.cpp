// main.cpp
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

#include "httplib.h"
#include "stable-diffusion.h"

#include "common/common.hpp"

namespace fs = std::filesystem;

// ----------------------- helpers -----------------------
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val = 0, valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6)
        ret.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (ret.size() % 4)
        ret.push_back('=');
    return ret;
}

inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<uint8_t> base64_decode(const std::string& encoded_string) {
    int in_len = encoded_string.size();
    int i      = 0;
    int j      = 0;
    int in_    = 0;
    uint8_t char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = static_cast<uint8_t>(base64_chars.find(char_array_4[i]));

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; i < 3; i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = static_cast<uint8_t>(base64_chars.find(char_array_4[j]));

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++)
            ret.push_back(char_array_3[j]);
    }

    return ret;
}

std::string iso_timestamp_now() {
    using namespace std::chrono;
    auto now      = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _MSC_VER
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

struct SDSvrParams {
    std::string listen_ip = "127.0.0.1";
    int listen_port       = 1234;
    bool normal_exit      = false;
    bool verbose          = false;
    bool color            = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-l",
             "--listen-ip",
             "server listen ip (default: 127.0.0.1)",
             &listen_ip}};

        options.int_options = {
            {"",
             "--listen-port",
             "server listen port (default: 1234)",
             &listen_port},
        };

        options.bool_options = {
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
        };

        auto on_help_arg = [&](int argc, const char** argv, int index) {
            normal_exit = true;
            return -1;
        };

        options.manual_options = {
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
        };
        return options;
    };

    bool process_and_check() {
        if (listen_ip.empty()) {
            LOG_ERROR("error: the following arguments are required: listen_ip");
            return false;
        }

        if (listen_port < 0 || listen_port > 65535) {
            LOG_ERROR("error: listen_port should be in the range [0, 65535]");
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDSvrParams {\n"
            << "  listen_ip: " << listen_ip << ",\n"
            << "  listen_port: \"" << listen_port << "\",\n"
            << "}";
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDSvrParams& svr_params, SDContextParams& ctx_params, SDGenerationParams& default_gen_params) {
    std::vector<ArgOptions> options_vec = {svr_params.get_options(), ctx_params.get_options(), default_gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(svr_params.normal_exit ? 0 : 1);
    }

    if (!svr_params.process_and_check() ||
        !ctx_params.process_and_check(IMG_GEN) ||
        !default_gen_params.process_and_check(IMG_GEN, ctx_params.lora_model_dir)) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

enum class ImageFormat { JPEG,
                         PNG };

std::vector<uint8_t> write_image_to_vector(
    ImageFormat format,
    const uint8_t* image,
    int width,
    int height,
    int channels,
    int quality = 90) {
    std::vector<uint8_t> buffer;

    auto write_func = [&buffer](void* context, void* data, int size) {
        uint8_t* src = reinterpret_cast<uint8_t*>(data);
        buffer.insert(buffer.end(), src, src + size);
    };

    struct ContextWrapper {
        decltype(write_func)& func;
    } ctx{write_func};

    auto c_func = [](void* context, void* data, int size) {
        auto* wrapper = reinterpret_cast<ContextWrapper*>(context);
        wrapper->func(context, data, size);
    };

    int result = 0;
    switch (format) {
        case ImageFormat::JPEG:
            result = stbi_write_jpg_to_func(c_func, &ctx, width, height, channels, image, quality);
            break;
        case ImageFormat::PNG:
            result = stbi_write_png_to_func(c_func, &ctx, width, height, channels, image, width * channels);
            break;
        default:
            throw std::runtime_error("invalid image format");
    }

    if (!result) {
        throw std::runtime_error("write imgage to mem failed");
    }

    return buffer;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->verbose, svr_params->color);
}

int main(int argc, const char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    SDGenerationParams default_gen_params;
    parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);
    log_verbose = svr_params.verbose;
    log_color   = svr_params.color;

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", svr_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", default_gen_params.to_string().c_str());

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(false, false, false);
    sd_ctx_t* sd_ctx              = new_sd_ctx(&sd_ctx_params);

    if (sd_ctx == nullptr) {
        LOG_ERROR("new_sd_ctx_t failed");
        return 1;
    }

    std::mutex sd_ctx_mutex;

    httplib::Server svr;

    // Mount the outputs directory to serve generated images
    if (!svr.set_mount_point("/outputs", "./outputs")) {
        LOG_WARN("failed to mount ./outputs directory, will not serve history images");
    }
    // Mount a directory to serve static files (our web UI)
    if (!svr.set_mount_point("/", "./public")) {
        LOG_WARN("failed to mount ./public directory, will not serve static files");
    }

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // health
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"ok":true,"service":"sd-cpp-http"})", "application/json");
    });

    // models endpoint (minimal)
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json r;
        r["data"] = json::array();
        r["data"].push_back({{"id", "sd-cpp-local"}, {"object", "model"}, {"owned_by", "local"}});
        res.set_content(r.dump(), "application/json");
    });

    // image history endpoint
    svr.Get("/v1/history/images", [&](const httplib::Request&, httplib::Response& res) {
        const std::string output_dir = "outputs";
        json image_list = json::array();
        try {
            if (fs::exists(output_dir) && fs::is_directory(output_dir)) {
                std::vector<fs::path> image_paths;
                for (const auto& entry : fs::directory_iterator(output_dir)) {
                    if (entry.is_regular_file()) {
                        auto ext = entry.path().extension().string();
                        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                            image_paths.push_back(entry.path());
                        }
                    }
                }
                // Sort files descending (newest first based on filename timestamp)
                std::sort(image_paths.begin(), image_paths.end(), [](const fs::path& a, const fs::path& b) {
                    return a.filename().string() > b.filename().string();
                });

                for(const auto& img_path : image_paths) {
                    json item;
                    item["name"] = img_path.filename().string();
                    
                    // Try to load matching .json metadata
                    auto json_path = img_path;
                    json_path.replace_extension(".json");
                    if (fs::exists(json_path)) {
                        try {
                            std::ifstream json_file(json_path);
                            item["params"] = json::parse(json_file);
                        } catch (...) {
                            LOG_WARN("failed to parse metadata: %s", json_path.string().c_str());
                        }
                    }
                    image_list.push_back(item);
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("failed to list image history: %s", e.what());
            res.status = 500;
            res.set_content(R"({"error":"failed to list image history"})", "application/json");
            return;
        }
        res.set_content(image_list.dump(), "application/json");
    });

    // core endpoint: /v1/images/generations
    svr.Post("/v1/images/generations", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }

            json j                    = json::parse(req.body);
            std::string prompt        = j.value("prompt", "");
            int n                     = std::max(1, j.value("n", 1));
            std::string size          = j.value("size", "");
            std::string output_format = j.value("output_format", "png");
            int output_compression    = j.value("output_compression", 100);
            int width                 = 512;
            int height                = 512;
            if (!size.empty()) {
                auto pos = size.find('x');
                if (pos != std::string::npos) {
                    try {
                        width  = std::stoi(size.substr(0, pos));
                        height = std::stoi(size.substr(pos + 1));
                    } catch (...) {
                    }
                }
            }

            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"prompt required"})", "application/json");
                return;
            }

            if (output_format != "png" && output_format != "jpeg") {
                res.status = 400;
                res.set_content(R"({"error":"invalid output_format, must be one of [png, jpeg]"})", "application/json");
                return;
            }
            if (n <= 0)
                n = 1;
            if (n > 8)
                n = 8;  // safety
            if (output_compression > 100) {
                output_compression = 100;
            }
            if (output_compression < 0) {
                output_compression = 0;
            }

            json out;
            out["created"]       = iso_timestamp_now();
            out["data"]          = json::array();
            out["output_format"] = output_format;

            SDGenerationParams gen_params = default_gen_params;
            gen_params.prompt             = prompt;
            gen_params.width              = width;
            gen_params.height             = height;
            gen_params.batch_count        = n;

            if (!gen_params.from_json_str(req.body)) {
                res.status = 400;
                res.set_content(R"({"error":"invalid params"})", "application/json");
                return;
            }

            bool save_image = j.value("save_image", false);

            if (!gen_params.process_and_check(IMG_GEN, "")) {
                res.status = 400;
                res.set_content(R"({"error":"invalid params"})", "application/json");
                return;
            }

            LOG_DEBUG("%s\n", gen_params.to_string().c_str());

            sd_image_t init_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            
            // Handle Img2Img init_image
            if (j.contains("init_image") && j["init_image"].is_string()) {
                std::string b64_init = j["init_image"];
                // Strip data:image/png;base64, prefix if present
                if (b64_init.find("base64,") != std::string::npos) {
                    b64_init = b64_init.substr(b64_init.find("base64,") + 7);
                }
                auto init_bytes = base64_decode(b64_init);
                if (!init_bytes.empty()) {
                    int img_w = gen_params.width;
                    int img_h = gen_params.height;
                    init_image.data = load_image_from_memory(
                        reinterpret_cast<const char*>(init_bytes.data()),
                        init_bytes.size(),
                        img_w, img_h,
                        gen_params.width, gen_params.height, 3);
                    init_image.width = (uint32_t)img_w;
                    init_image.height = (uint32_t)img_h;
                    if (!init_image.data) {
                        LOG_ERROR("failed to load init_image from base64");
                    } else {
                        LOG_INFO("loaded init_image for img2img: %dx%d", img_w, img_h);
                    }
                }
            }

            sd_image_t control_image = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            sd_image_t mask_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 1, nullptr};

            // Allocate dummy data for mask and control if not provided, to avoid crash in sd_image_to_ggml_tensor
            if (mask_image.data == nullptr) {
                mask_image.data = (uint8_t*)malloc(mask_image.width * mask_image.height * mask_image.channel);
                // For Img2Img, a mask of 255 (white) means "denoise everything"
                memset(mask_image.data, 255, mask_image.width * mask_image.height * mask_image.channel);
            }
            if (control_image.data == nullptr) {
                control_image.data = (uint8_t*)calloc(1, control_image.width * control_image.height * control_image.channel);
            }

            std::vector<sd_image_t> pmid_images;

            sd_img_gen_params_t img_gen_params = {
                gen_params.lora_vec.data(),
                static_cast<uint32_t>(gen_params.lora_vec.size()),
                gen_params.prompt.c_str(),
                gen_params.negative_prompt.c_str(),
                gen_params.clip_skip,
                init_image,
                nullptr,
                0,
                gen_params.auto_resize_ref_image,
                gen_params.increase_ref_index,
                mask_image,
                gen_params.width,
                gen_params.height,
                gen_params.sample_params,
                gen_params.strength,
                gen_params.seed,
                gen_params.batch_count,
                control_image,
                gen_params.control_strength,
                {
                    pmid_images.data(),
                    (int)pmid_images.size(),
                    gen_params.pm_id_embed_path.c_str(),
                    gen_params.pm_style_strength,
                },  // pm_params
                ctx_params.vae_tiling_params,
                gen_params.easycache_params,
            };

            sd_image_t* results = nullptr;
            int num_results     = 0;

            {
                std::lock_guard<std::mutex> lock(sd_ctx_mutex);
                results     = generate_image(sd_ctx, &img_gen_params);
                num_results = gen_params.batch_count;
            }

            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                auto image_bytes = write_image_to_vector(output_format == "jpeg" ? ImageFormat::JPEG : ImageFormat::PNG,
                                                         results[i].data,
                                                         results[i].width,
                                                         results[i].height,
                                                         results[i].channel,
                                                         output_compression);
                if (image_bytes.empty()) {
                    LOG_ERROR("write image to mem failed");
                    continue;
                }

                if (save_image) {
                    try {
                        const std::string output_dir = "outputs";
                        if (!fs::exists(output_dir)) {
                            fs::create_directory(output_dir);
                        }
                        // a timestamp in microseconds + seed should be unique enough
                        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                        std::string base_filename = "img-" + std::to_string(timestamp) + "-" + std::to_string(gen_params.seed);
                        std::string img_filename = output_dir + "/" + base_filename + ".png";
                        
                        std::ofstream file(img_filename, std::ios::binary);
                        file.write(reinterpret_cast<const char*>(image_bytes.data()), image_bytes.size());
                        LOG_INFO("saved image to %s", img_filename.c_str());

                        // Save parameters as JSON
                        std::string json_filename = output_dir + "/" + base_filename + ".json";
                        json meta = j; // The original request JSON
                        
                        bool has_init = meta.contains("init_image");
                        if (has_init) {
                            meta.erase("init_image");
                        }
                        meta["is_img2img"] = has_init;
                        meta["seed"] = gen_params.seed;
                        
                        std::ofstream json_file(json_filename);
                        json_file << meta.dump(4);
                        LOG_INFO("saved parameters to %s", json_filename.c_str());

                    } catch (const std::exception& e) {
                        LOG_ERROR("failed to save image or metadata: %s", e.what());
                    }
                }

                // base64 encode
                std::string b64 = base64_encode(image_bytes);
                json item;
                item["b64_json"] = b64;
                out["data"].push_back(item);
            }

            res.set_content(out.dump(), "application/json");
            res.status = 200;

            if (init_image.data) {
                stbi_image_free(init_image.data);
            }
            if (mask_image.data) {
                free(mask_image.data);
            }
            if (control_image.data) {
                free(control_image.data);
            }

        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    svr.Post("/v1/images/edits", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            if (!req.is_multipart_form_data()) {
                res.status = 400;
                res.set_content(R"({"error":"Content-Type must be multipart/form-data"})", "application/json");
                return;
            }

            std::string prompt = req.form.get_field("prompt");
            if (prompt.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"prompt required"})", "application/json");
                return;
            }

            std::string extra_args_str;
            if (req.form.has_field("extra_args")) {
                extra_args_str = req.form.get_field("extra_args");
            }

            size_t image_count = req.form.get_file_count("image[]");
            if (image_count == 0) {
                res.status = 400;
                res.set_content(R"({"error":"at least one image[] required"})", "application/json");
                return;
            }

            std::vector<std::vector<uint8_t>> images_bytes;
            for (size_t i = 0; i < image_count; i++) {
                auto file = req.form.get_file("image[]", i);
                images_bytes.emplace_back(file.content.begin(), file.content.end());
            }

            std::vector<uint8_t> mask_bytes;
            if (req.form.has_field("mask")) {
                auto file = req.form.get_file("mask");
                mask_bytes.assign(file.content.begin(), file.content.end());
            }

            int n = 1;
            if (req.form.has_field("n")) {
                try {
                    n = std::stoi(req.form.get_field("n"));
                } catch (...) {
                }
            }
            n = std::clamp(n, 1, 8);

            std::string size = req.form.get_field("size");
            int width = 512, height = 512;
            if (!size.empty()) {
                auto pos = size.find('x');
                if (pos != std::string::npos) {
                    try {
                        width  = std::stoi(size.substr(0, pos));
                        height = std::stoi(size.substr(pos + 1));
                    } catch (...) {
                    }
                }
            }

            std::string output_format = "png";
            if (req.form.has_field("output_format"))
                output_format = req.form.get_field("output_format");
            if (output_format != "png" && output_format != "jpeg") {
                res.status = 400;
                res.set_content(R"({"error":"invalid output_format, must be one of [png, jpeg]"})", "application/json");
                return;
            }

            std::string output_compression_str = req.form.get_field("output_compression");
            int output_compression             = 100;
            try {
                output_compression = std::stoi(output_compression_str);
            } catch (...) {
            }
            if (output_compression > 100) {
                output_compression = 100;
            }
            if (output_compression < 0) {
                output_compression = 0;
            }

            SDGenerationParams gen_params = default_gen_params;
            gen_params.prompt             = prompt;
            gen_params.width              = width;
            gen_params.height             = height;
            gen_params.batch_count        = n;

            if (!extra_args_str.empty() && !gen_params.from_json_str(extra_args_str)) {
                res.status = 400;
                res.set_content(R"({"error":"invalid extra_args"})", "application/json");
                return;
            }

            if (!gen_params.process_and_check(IMG_GEN, "")) {
                res.status = 400;
                res.set_content(R"({"error":"invalid params"})", "application/json");
                return;
            }

            LOG_DEBUG("%s\n", gen_params.to_string().c_str());

            sd_image_t init_image    = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            sd_image_t control_image = {(uint32_t)gen_params.width, (uint32_t)gen_params.height, 3, nullptr};
            std::vector<sd_image_t> pmid_images;

            std::vector<sd_image_t> ref_images;
            ref_images.reserve(images_bytes.size());
            for (auto& bytes : images_bytes) {
                int img_w           = width;
                int img_h           = height;
                uint8_t* raw_pixels = load_image_from_memory(
                    reinterpret_cast<const char*>(bytes.data()),
                    bytes.size(),
                    img_w, img_h,
                    width, height, 3);

                if (!raw_pixels) {
                    continue;
                }

                sd_image_t img{(uint32_t)img_w, (uint32_t)img_h, 3, raw_pixels};
                ref_images.push_back(img);
            }

            sd_image_t mask_image = {0};
            if (!mask_bytes.empty()) {
                int mask_w        = width;
                int mask_h        = height;
                uint8_t* mask_raw = load_image_from_memory(
                    reinterpret_cast<const char*>(mask_bytes.data()),
                    mask_bytes.size(),
                    mask_w, mask_h,
                    width, height, 1);
                mask_image = {(uint32_t)mask_w, (uint32_t)mask_h, 1, mask_raw};
            } else {
                mask_image.width   = width;
                mask_image.height  = height;
                mask_image.channel = 1;
                mask_image.data    = nullptr;
            }

            sd_img_gen_params_t img_gen_params = {
                gen_params.lora_vec.data(),
                static_cast<uint32_t>(gen_params.lora_vec.size()),
                gen_params.prompt.c_str(),
                gen_params.negative_prompt.c_str(),
                gen_params.clip_skip,
                init_image,
                ref_images.data(),
                (int)ref_images.size(),
                gen_params.auto_resize_ref_image,
                gen_params.increase_ref_index,
                mask_image,
                gen_params.width,
                gen_params.height,
                gen_params.sample_params,
                gen_params.strength,
                gen_params.seed,
                gen_params.batch_count,
                control_image,
                gen_params.control_strength,
                {
                    pmid_images.data(),
                    (int)pmid_images.size(),
                    gen_params.pm_id_embed_path.c_str(),
                    gen_params.pm_style_strength,
                },  // pm_params
                ctx_params.vae_tiling_params,
                gen_params.easycache_params,
            };

            sd_image_t* results = nullptr;
            int num_results     = 0;

            {
                std::lock_guard<std::mutex> lock(sd_ctx_mutex);
                results     = generate_image(sd_ctx, &img_gen_params);
                num_results = gen_params.batch_count;
            }

            json out;
            out["created"]       = iso_timestamp_now();
            out["data"]          = json::array();
            out["output_format"] = output_format;

            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr)
                    continue;
                auto image_bytes = write_image_to_vector(output_format == "jpeg" ? ImageFormat::JPEG : ImageFormat::PNG,
                                                         results[i].data,
                                                         results[i].width,
                                                         results[i].height,
                                                         results[i].channel,
                                                         output_compression);
                std::string b64 = base64_encode(image_bytes);
                json item;
                item["b64_json"] = b64;
                out["data"].push_back(item);
            }

            res.set_content(out.dump(), "application/json");
            res.status = 200;

            if (init_image.data) {
                stbi_image_free(init_image.data);
            }
            if (mask_image.data) {
                stbi_image_free(mask_image.data);
            }
            for (auto ref_image : ref_images) {
                stbi_image_free(ref_image.data);
            }
        } catch (const std::exception& e) {
            res.status = 500;
            json err;
            err["error"]   = "server_error";
            err["message"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    LOG_INFO("listening on: %s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    // cleanup
    free_sd_ctx(sd_ctx);
    return 0;
}
