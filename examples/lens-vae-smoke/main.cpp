#include "stable-diffusion.h"

#include "../common/log.h"
#include "../common/media_io.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Args {
    std::string vae;
    std::string latent_npy;
    std::string output = "lens_vae_smoke.png";
    std::string vae_backend = "cpu";
    int width = 0;
    int height = 0;
    bool pack_check = false;
    bool decode = false;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --pack-check | --vae <lens-vae.safetensors> --latent-npy <latent.npy> --output <out.png>\n"
                 "       --latent-npy expects f32 NCHW with shape 1x32xHxW.\n"
                 "       [--vae-backend cpu|cuda] defaults to cpu and is diagnostic only.\n",
                 argv0);
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--vae") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.vae = value;
        } else if (std::strcmp(arg, "--latent-npy") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.latent_npy = value;
        } else if (std::strcmp(arg, "--output") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.output = value;
        } else if (std::strcmp(arg, "--vae-backend") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.vae_backend = value;
            std::transform(args.vae_backend.begin(), args.vae_backend.end(), args.vae_backend.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (args.vae_backend != "cpu" && args.vae_backend != "cuda") {
                std::fprintf(stderr, "--vae-backend must be cpu or cuda\n");
                return false;
            }
        } else if (std::strcmp(arg, "--width") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.width = std::atoi(value);
        } else if (std::strcmp(arg, "--height") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.height = std::atoi(value);
        } else if (std::strcmp(arg, "--pack-check") == 0) {
            args.pack_check = true;
        } else if (std::strcmp(arg, "--decode") == 0) {
            args.decode = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    args.decode = args.decode || !args.vae.empty() || !args.latent_npy.empty();
    if (args.pack_check == args.decode) {
        return false;
    }
    if (args.decode && (args.vae.empty() || args.latent_npy.empty())) {
        return false;
    }
    return true;
}

static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    log_print(level, log, true, false);
}

static bool load_f32_npy(const std::string& path, std::vector<float>& data, std::vector<int64_t>& shape) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "failed to open npy: " << path << "\n";
        return false;
    }
    char magic[6] = {};
    in.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "not an npy file: " << path << "\n";
        return false;
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        in.read(reinterpret_cast<char*>(&h16), 2);
        header_len = h16;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        std::cerr << "unsupported npy version: " << static_cast<int>(major) << "." << static_cast<int>(minor) << "\n";
        return false;
    }
    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (header.find("'descr': '<f4'") == std::string::npos && header.find("\"descr\": \"<f4\"") == std::string::npos) {
        std::cerr << "npy must be f32 little-endian: " << header << "\n";
        return false;
    }
    if (header.find("True") != std::string::npos) {
        std::cerr << "Fortran-order npy is not supported: " << header << "\n";
        return false;
    }
    const size_t lparen = header.find('(');
    const size_t rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) {
        std::cerr << "npy shape missing: " << header << "\n";
        return false;
    }
    shape.clear();
    std::string dims = header.substr(lparen + 1, rparen - lparen - 1);
    size_t start = 0;
    while (start < dims.size()) {
        size_t comma = dims.find(',', start);
        std::string token = dims.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch); }), token.end());
        if (!token.empty()) {
            shape.push_back(std::stoll(token));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (shape.empty()) {
        return false;
    }
    uint64_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) return false;
        count *= static_cast<uint64_t>(dim);
    }
    data.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<uint64_t>(in.gcount()) == count * sizeof(float);
}

static bool run_pack_check() {
    std::vector<float> lens(1 * 32 * 4 * 4);
    for (size_t i = 0; i < lens.size(); ++i) {
        lens[i] = static_cast<float>(i);
    }
    std::vector<float> packed(1 * 128 * 2 * 2, -1.0f);
    sd_lens_vae_latent_desc_t desc;
    sd_lens_vae_latent_desc_init(&desc);
    if (!sd_lens_pack_vae_latent_f32(lens.data(), lens.size(), 1, 32, 4, 4, packed.data(), packed.size(), &desc)) {
        std::cerr << "sd_lens_pack_vae_latent_f32 failed\n";
        return false;
    }
    if (desc.output_n != 1 || desc.output_c != 128 || desc.output_h != 2 || desc.output_w != 2 ||
        desc.input_elements != lens.size() || desc.output_elements != packed.size()) {
        std::cerr << "unexpected pack descriptor\n";
        return false;
    }
    auto src_index = [](int c, int y, int x) {
        return static_cast<size_t>(c * 4 * 4 + y * 4 + x);
    };
    auto dst_index = [](int c, int y, int x) {
        return static_cast<size_t>(c * 2 * 2 + y * 2 + x);
    };
    for (int c = 0; c < 32; ++c) {
        for (int py = 0; py < 2; ++py) {
            for (int px = 0; px < 2; ++px) {
                for (int oy = 0; oy < 2; ++oy) {
                    for (int ox = 0; ox < 2; ++ox) {
                        int pc = c * 4 + oy * 2 + ox;
                        float expected = lens[src_index(c, py * 2 + oy, px * 2 + ox)];
                        if (packed[dst_index(pc, py, px)] != expected) {
                            std::cerr << "pack mismatch at c=" << c << " py=" << py << " px=" << px << " oy=" << oy << " ox=" << ox << "\n";
                            return false;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Lens VAE pack smoke passed: input=1x32x4x4 output=1x128x2x2\n";
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    sd_set_log_callback(sd_log_cb, nullptr);

    if (args.pack_check) {
        return run_pack_check() ? 0 : 1;
    }

    std::vector<float> latent;
    std::vector<int64_t> shape;
    if (!load_f32_npy(args.latent_npy, latent, shape) || shape.size() != 4 || shape[0] != 1 || shape[1] != 32) {
        std::cerr << "latent npy must have shape NCHW with N=1 and C=32\n";
        return 1;
    }
    const int64_t h = shape[2];
    const int64_t w = shape[3];

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.vae_path = args.vae.c_str();
    params.vae_decode_only = true;
    params.free_params_immediately = true;
    params.keep_vae_on_cpu = args.vae_backend != "cuda";
#ifdef _WIN32
    _putenv_s("SDCPP_MODEL_FAMILY_HINT", "lens");
#else
    setenv("SDCPP_MODEL_FAMILY_HINT", "lens", 1);
#endif
    sd_ctx_t* ctx = new_sd_ctx(&params);
    if (ctx == nullptr) {
        std::cerr << "failed to create Lens VAE decode context\n";
        return 1;
    }

    sd_latent_t* sd_latent = sd_latent_import_f32(latent.data(),
                                                 static_cast<uint64_t>(latent.size()),
                                                 static_cast<uint32_t>(w),
                                                 static_cast<uint32_t>(h),
                                                 32);
    if (sd_latent == nullptr) {
        std::cerr << "sd_latent_import_f32 failed\n";
        free_sd_ctx(ctx);
        return 1;
    }

    sd_vae_memory_report_t report;
    sd_image_t* image = sd_decode_latent_normal(ctx, sd_latent, nullptr, &report);
    free_sd_latent(sd_latent);
    if (image == nullptr) {
        std::cerr << "Lens VAE decode failed\n";
        free_sd_ctx(ctx);
        return 1;
    }
    if (!write_image_to_file(args.output, image->data, image->width, image->height, image->channel)) {
        std::cerr << "failed to write output: " << args.output << "\n";
        free_sd_image(image);
        free_sd_ctx(ctx);
        return 1;
    }
    std::cout << "Lens VAE decode smoke wrote " << args.output
              << " image=" << image->width << "x" << image->height << "x" << image->channel
              << " latent=1x32x" << h << "x" << w
              << " vae_backend=" << args.vae_backend
              << " decode_graph_ms=" << report.decode_graph_ms << "\n";
    free_sd_image(image);
    free_sd_ctx(ctx);
    return 0;
}
