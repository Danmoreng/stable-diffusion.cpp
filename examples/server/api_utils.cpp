#include "api_utils.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <iostream>

// Since we separated common.hpp/cpp, common.hpp no longer defines STB_IMAGE_WRITE_IMPLEMENTATION
// But common.cpp does. We just need the declarations from common.hpp which includes stb_image_write.h
// However, write_image_to_vector uses stbi_write_jpg_to_func which is declared in stb_image_write.h.
// common.hpp includes stb_image_write.h, so we are good.

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

static const std::string base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/ ";

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

bool is_base64(unsigned char c) {
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

json parse_image_params(const std::string& txt) {
    json j;
    std::istringstream stream(txt);
    std::string line;
    std::string positive_prompt;
    std::string negative_prompt;
    bool in_positive = true;
    bool in_negative = false;

    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        if (line.find("Negative prompt: ") == 0) {
            negative_prompt = line.substr(17);
            in_positive = false;
            in_negative = true;
            continue;
        }

        // The parameters line usually starts with "Steps: "
        if (line.find("Steps: ") == 0) {
            in_positive = false;
            in_negative = false;
            
            // Parse comma-separated key-value pairs
            std::istringstream line_stream(line);
            std::string pair;
            while (std::getline(line_stream, pair, ',')) {
                size_t colon_pos = pair.find(':');
                if (colon_pos != std::string::npos) {
                    std::string key = pair.substr(0, colon_pos);
                    std::string val = pair.substr(colon_pos + 1);
                    
                    // Trim key and val
                    key.erase(0, key.find_first_not_of(" 	"));
                    key.erase(key.find_last_not_of(" 	") + 1);
                    val.erase(0, val.find_first_not_of(" 	"));
                    val.erase(val.find_last_not_of(" 	") + 1);

                    if (key == "Steps") j["sample_steps"] = std::stoi(val);
                    else if (key == "CFG scale") j["cfg_scale"] = std::stof(val);
                    else if (key == "Seed") j["seed"] = std::stoll(val);
                    else if (key == "Sampler") j["sampling_method"] = val;
                    else if (key == "Model") j["model"] = val;
                    else if (key == "Clip skip") j["clip_skip"] = std::stoi(val);
                    else if (key == "Size") {
                        auto x_pos = val.find('x');
                        if (x_pos != std::string::npos) {
                            j["width"] = std::stoi(val.substr(0, x_pos));
                            j["height"] = std::stoi(val.substr(x_pos + 1));
                        }
                    }
                }
            }
            continue;
        }

        if (in_positive) {
            positive_prompt += (positive_prompt.empty() ? "" : "\n") + line;
        } else if (in_negative) {
            negative_prompt += (negative_prompt.empty() ? "" : "\n") + line;
        }
    }

    j["prompt"] = positive_prompt;
    j["negative_prompt"] = negative_prompt;
    return j;
}

std::string get_image_params(const SDContextParams& ctx_params, const SDGenerationParams& gen_params, int64_t seed) {
    std::string parameter_string = gen_params.prompt + "\n";
    if (gen_params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + gen_params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(gen_params.sample_params.sample_steps) + ", ";
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(gen_params.sample_params.sample_method));
    if (gen_params.sample_params.scheduler != SCHEDULER_COUNT) {
        parameter_string += " " + std::string(sd_scheduler_name(gen_params.sample_params.scheduler));
    }
    parameter_string += ", CFG scale: " + std::to_string(gen_params.sample_params.guidance.txt_cfg) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(gen_params.width) + "x" + std::to_string(gen_params.height) + ", ";
    parameter_string += "Model: " + sd_basename(ctx_params.diffusion_model_path.empty() ? ctx_params.model_path : ctx_params.diffusion_model_path) + ", ";

    if (!ctx_params.vae_path.empty()) {
        parameter_string += "VAE: " + sd_basename(ctx_params.vae_path) + ", ";
    }
    if (gen_params.clip_skip != -1) {
        parameter_string += "Clip skip: " + std::to_string(gen_params.clip_skip) + ", ";
    }
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

std::vector<uint8_t> write_image_to_vector(
    ImageFormat format,
    const uint8_t* image,
    int width,
    int height,
    int channels,
    int quality) {
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
