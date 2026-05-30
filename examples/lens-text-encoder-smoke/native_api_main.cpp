#include "lens_gptoss_text_encoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLAS
#include <cuda_runtime.h>
#endif

struct NativeArgs {
    std::string text_encoder_dir;
    std::string bootstrap_oracle_dir;
    std::string output;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --text-encoder <Lens-Turbo/text_encoder> --bootstrap-oracle <oracle-dir> --output <lens_cond_v1.safetensors>\n",
                 argv0);
}

static bool parse_native_args(int argc, char** argv, NativeArgs& args) {
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--text-encoder") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.text_encoder_dir = value;
        } else if (std::strcmp(argv[i], "--bootstrap-oracle") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.bootstrap_oracle_dir = value;
        } else if (std::strcmp(argv[i], "--output") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.output = value;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return !args.text_encoder_dir.empty() && !args.bootstrap_oracle_dir.empty() && !args.output.empty();
}

static void print_vram_snapshot(const char* label) {
#ifdef SD_LENS_TEXT_ENCODER_USE_CUBLAS
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes > 0) {
        const double total_gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
        const double free_gib = static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "vram_snapshot stage=" << label
                  << " used_gib=" << (total_gib - free_gib)
                  << " free_gib=" << free_gib
                  << " total_gib=" << total_gib << "\n";
    }
#else
    (void)label;
#endif
}

int main(int argc, char** argv) {
    NativeArgs args;
    if (!parse_native_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    sd_lens_text_encoder_result result;
    print_vram_snapshot("before_text_encoder_create");
    sd_lens_text_encoder* encoder = sd_lens_text_encoder_create();
    sd_lens_text_encoder_load_options load_options;
    load_options.text_encoder_dir = args.text_encoder_dir;
    if (!sd_lens_text_encoder_load(encoder, load_options, &result)) {
        std::cerr << "sd_lens_text_encoder_load failed: " << result.error << "\n";
        sd_lens_text_encoder_free(encoder);
        return 1;
    }
    print_vram_snapshot("after_text_encoder_load");

    sd_lens_text_encoder_encode_options encode_options;
    encode_options.bootstrap_oracle_dir = args.bootstrap_oracle_dir;
    encode_options.output_safetensors = args.output;
    if (!sd_lens_text_encoder_encode(encoder, encode_options, &result)) {
        std::cerr << "sd_lens_text_encoder_encode failed: " << result.error << "\n";
        sd_lens_text_encoder_free(encoder);
        return 1;
    }
    print_vram_snapshot("after_text_encoder_encode");

    std::cout << "Lens native text_encoder API emitted " << args.output
              << " encode_seconds=" << result.encode_seconds
              << " router_mode=native-tolerant"
              << " tokenizer=bootstrap_input_ids_mask_from_oracle\n";
    sd_lens_text_encoder_free(encoder);
    print_vram_snapshot("after_text_encoder_free");
    std::cout << "Lens native text_encoder API released text_encoder runtime state\n";
    return 0;
}
