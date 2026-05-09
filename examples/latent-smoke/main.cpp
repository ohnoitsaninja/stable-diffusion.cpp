#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/media_io.h"
#include "stable-diffusion.h"

struct Args {
    std::string model;
    std::string image;
    std::string output = "latent_smoke.png";
    int image_channels = 4;
    int width          = 0;
    int height         = 0;
    bool sample        = false;
    bool decode        = true;
    bool split_decode_context = false;
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model <model> --image <image> [options]\n"
        << "Options:\n"
        << "  --output <path>          decoded output path (default: latent_smoke.png)\n"
        << "  --image-channels <3|4>   channel count to load and pass into sd_encode_image (default: 4)\n"
        << "  --width <int>            optional target width for sample/decode path\n"
        << "  --height <int>           optional target height for sample/decode path\n"
        << "  --sample                 run sd_sample_latent after encode\n"
        << "  --split-decode-context   decode with a separate vae_decode_only=true context\n"
        << "  --no-decode              skip sd_decode_latent\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--model") {
            const char* value = need_value("--model");
            if (value == nullptr) return false;
            args.model = value;
        } else if (arg == "--image") {
            const char* value = need_value("--image");
            if (value == nullptr) return false;
            args.image = value;
        } else if (arg == "--output") {
            const char* value = need_value("--output");
            if (value == nullptr) return false;
            args.output = value;
        } else if (arg == "--image-channels") {
            const char* value = need_value("--image-channels");
            if (value == nullptr) return false;
            args.image_channels = std::atoi(value);
        } else if (arg == "--width") {
            const char* value = need_value("--width");
            if (value == nullptr) return false;
            args.width = std::atoi(value);
        } else if (arg == "--height") {
            const char* value = need_value("--height");
            if (value == nullptr) return false;
            args.height = std::atoi(value);
        } else if (arg == "--sample") {
            args.sample = true;
        } else if (arg == "--split-decode-context") {
            args.split_decode_context = true;
        } else if (arg == "--no-decode") {
            args.decode = false;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (args.model.empty() || args.image.empty()) {
        return false;
    }
    if (args.image_channels != 3 && args.image_channels != 4) {
        std::cerr << "--image-channels must be 3 or 4\n";
        return false;
    }
    return true;
}

static sd_ctx_t* create_context(const Args& args, bool vae_decode_only) {
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path            = args.model.c_str();
    ctx_params.vae_decode_only       = vae_decode_only;
    ctx_params.diffusion_flash_attn  = true;
    ctx_params.vae_conv_direct       = true;
    ctx_params.offload_params_to_cpu = false;
    ctx_params.keep_clip_on_cpu      = false;
    ctx_params.keep_vae_on_cpu       = false;

    std::cout << "creating context vae_decode_only=" << (vae_decode_only ? "true" : "false") << "\n";
    return new_sd_ctx(&ctx_params);
}

static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    log_print(level, log, true, false);
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    sd_set_log_callback(sd_log_cb, nullptr);

    sd_ctx_t* ctx = create_context(args, false);
    if (ctx == nullptr) {
        std::cerr << "new_sd_ctx failed\n";
        return 1;
    }

    sd_image_t image{};
    if (!load_sd_image_from_file(&image, args.image.c_str(), 0, 0, args.image_channels)) {
        if (args.image_channels == 4 && load_sd_image_from_file(&image, args.image.c_str(), 0, 0, 3)) {
            std::vector<uint8_t> rgba(static_cast<size_t>(image.width) * image.height * 4);
            for (uint32_t y = 0; y < image.height; ++y) {
                for (uint32_t x = 0; x < image.width; ++x) {
                    const size_t src = (static_cast<size_t>(y) * image.width + x) * 3;
                    const size_t dst = (static_cast<size_t>(y) * image.width + x) * 4;
                    rgba[dst + 0]    = image.data[src + 0];
                    rgba[dst + 1]    = image.data[src + 1];
                    rgba[dst + 2]    = image.data[src + 2];
                    rgba[dst + 3]    = 255;
                }
            }
            free(image.data);
            image.channel = 4;
            image.data    = static_cast<uint8_t*>(malloc(rgba.size()));
            if (image.data != nullptr) {
                std::memcpy(image.data, rgba.data(), rgba.size());
            }
        }
    }
    if (image.data == nullptr) {
        std::cerr << "failed to load image: " << args.image << "\n";
        free_sd_ctx(ctx);
        return 1;
    }
    std::cout << "loaded image " << image.width << "x" << image.height << "x" << image.channel << "\n";

    sd_tiling_params_t tiling{};
    tiling.enabled        = false;
    tiling.target_overlap = 0.5f;

    std::cout << "calling sd_encode_image\n";
    sd_latent_t* encoded = sd_encode_image(ctx, &image, &tiling);
    free(image.data);
    if (encoded == nullptr) {
        std::cerr << "sd_encode_image failed\n";
        free_sd_ctx(ctx);
        return 1;
    }
    std::cout << "encoded latent " << encoded->width << "x" << encoded->height << "x"
              << encoded->channel << " elements=" << encoded->element_count << "\n";

    sd_latent_t* latent_to_decode = encoded;
    if (args.sample) {
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        gen_params.prompt          = "a detailed fantasy orc portrait, high quality";
        gen_params.negative_prompt = "blurry, low quality, noisy";
        gen_params.width           = args.width > 0 ? args.width : static_cast<int>(image.width);
        gen_params.height          = args.height > 0 ? args.height : static_cast<int>(image.height);
        gen_params.seed            = 12345;
        gen_params.strength        = 0.65f;
        gen_params.sample_params.sample_steps          = 8;
        gen_params.sample_params.guidance.txt_cfg      = 1.2f;
        gen_params.sample_params.guidance.img_cfg      = 1.2f;
        gen_params.sample_params.sample_method         = EULER_SAMPLE_METHOD;
        gen_params.sample_params.scheduler             = KARRAS_SCHEDULER;
        gen_params.vae_tiling_params                   = tiling;

        std::cout << "calling sd_sample_latent\n";
        sd_latent_t* sampled = sd_sample_latent(ctx, &gen_params, encoded);
        if (sampled == nullptr) {
            std::cerr << "sd_sample_latent failed\n";
            free_sd_latent(encoded);
            free_sd_ctx(ctx);
            return 1;
        }
        std::cout << "sampled latent " << sampled->width << "x" << sampled->height << "x"
                  << sampled->channel << " elements=" << sampled->element_count << "\n";
        latent_to_decode = sampled;
    }

    if (args.decode) {
        sd_ctx_t* decode_ctx = ctx;
        if (args.split_decode_context) {
            decode_ctx = create_context(args, true);
            if (decode_ctx == nullptr) {
                std::cerr << "decode new_sd_ctx failed\n";
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
        }

        std::cout << "calling sd_decode_latent\n";
        sd_image_t* output = sd_decode_latent(decode_ctx, latent_to_decode, &tiling);
        if (output == nullptr) {
            std::cerr << "sd_decode_latent failed\n";
            if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
            if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
            free_sd_latent(encoded);
            free_sd_ctx(ctx);
            return 1;
        }
        if (!write_image_to_file(args.output, output->data, output->width, output->height, output->channel)) {
            std::cerr << "failed to write output: " << args.output << "\n";
            free_sd_image(output);
            if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
            if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
            free_sd_latent(encoded);
            free_sd_ctx(ctx);
            return 1;
        }
        std::cout << "wrote " << args.output << "\n";
        free_sd_image(output);
        if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
    }

    if (latent_to_decode != encoded) {
        free_sd_latent(latent_to_decode);
    }
    free_sd_latent(encoded);
    free_sd_ctx(ctx);
    return 0;
}
