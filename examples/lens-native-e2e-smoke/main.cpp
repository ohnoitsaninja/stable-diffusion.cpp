#include "lens_gptoss_text_encoder.hpp"
#include "stable-diffusion.h"

#include "../common/media_io.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

using LensTransformerSmokeRunFn = int (*)(int, char**, const std::unordered_map<std::string, Tensor>*, Tensor*);

struct E2EArgs {
    std::string text_encoder_dir;
    std::string bootstrap_oracle_dir;
    std::string transformer_dir;
    std::string vae_path;
    std::string output_png;
    std::string optional_cond_out;
    int width = 256;
    int height = 256;
    int steps = 4;
    int seed = 42;
};

static void e2e_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --text-encoder <Lens-Turbo/text_encoder> --bootstrap-oracle <oracle-dir> "
                 "--transformer <Lens-Turbo/transformer> --vae <Lens-Turbo/vae/diffusion_pytorch_model.safetensors> "
                 "--height 256 --width 256 --steps 4 --seed 42 --output <out.png> "
                 "[--optional-cond-out <lens_cond_v1.safetensors>]\n",
                 argv0);
}

static bool parse_e2e_args(int argc, char** argv, E2EArgs& args) {
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
        } else if (std::strcmp(argv[i], "--transformer") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.transformer_dir = value;
        } else if (std::strcmp(argv[i], "--vae") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.vae_path = value;
        } else if (std::strcmp(argv[i], "--height") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.height = std::atoi(value);
        } else if (std::strcmp(argv[i], "--width") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.width = std::atoi(value);
        } else if (std::strcmp(argv[i], "--steps") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.steps = std::atoi(value);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.seed = std::atoi(value);
        } else if (std::strcmp(argv[i], "--output") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.output_png = value;
        } else if (std::strcmp(argv[i], "--optional-cond-out") == 0) {
            const char* value = need(argv[i]);
            if (value == nullptr) return false;
            args.optional_cond_out = value;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            e2e_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return !args.text_encoder_dir.empty() &&
           !args.bootstrap_oracle_dir.empty() &&
           !args.transformer_dir.empty() &&
           !args.vae_path.empty() &&
           !args.output_png.empty();
}

static void e2e_print_vram_snapshot(const char* stage) {
    std::cout << "vram_snapshot stage=" << stage << " source=external-only\n";
}

static void create_parent_dir(const std::string& path) {
    const std::filesystem::path fs_path(path);
    const std::filesystem::path parent = fs_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

static std::unordered_map<std::string, Tensor> make_transformer_condition_map(const sd_lens_cond_v1_native& cond) {
    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(5);
    for (int i = 0; i < 4; ++i) {
        Tensor t;
        t.shape = cond.feature_shapes[static_cast<size_t>(i)];
        t.data = cond.features[static_cast<size_t>(i)];
        if (t.shape.size() != 3 || t.shape[0] != 1 || t.shape[1] != cond.trimmed_seq_len || t.shape[2] != 2880) {
            throw std::runtime_error("in-memory Lens feature_" + std::to_string(i) + " has invalid shape");
        }
        tensors.emplace("feature_" + std::to_string(i), std::move(t));
    }
    Tensor mask;
    mask.shape = cond.attention_mask_shape;
    mask.data = cond.attention_mask;
    if (mask.shape.size() != 2 || mask.shape[0] != 1 || mask.shape[1] != cond.trimmed_seq_len) {
        throw std::runtime_error("in-memory Lens attention_mask has invalid shape");
    }
    tensors.emplace("attention_mask", std::move(mask));
    return tensors;
}

static void e2e_sd_log_cb(enum sd_log_level_t level, const char* log, void*) {
    if (log == nullptr) {
        return;
    }
    FILE* stream = level == SD_LOG_ERROR ? stderr : stdout;
    std::fputs(log, stream);
    std::fflush(stream);
}

static bool decode_lens_vae_to_png(const std::string& vae_path,
                                   const Tensor& latent,
                                   const std::string& output_png,
                                   double& decode_seconds) {
    if (latent.shape.size() != 4 || latent.shape[0] != 1 || latent.shape[1] != 32) {
        std::cerr << "in-memory Lens latent must have shape 1x32xHxW\n";
        return false;
    }
    create_parent_dir(output_png);
    sd_set_log_callback(e2e_sd_log_cb, nullptr);
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.vae_path = vae_path.c_str();
    params.vae_decode_only = true;
    params.free_params_immediately = true;
    params.keep_vae_on_cpu = true;
#ifdef _WIN32
    _putenv_s("SDCPP_MODEL_FAMILY_HINT", "lens");
#else
    setenv("SDCPP_MODEL_FAMILY_HINT", "lens", 1);
#endif
    e2e_print_vram_snapshot("before_vae_load");
    const auto vae_load_start = std::chrono::steady_clock::now();
    sd_ctx_t* ctx = new_sd_ctx(&params);
    const auto vae_load_end = std::chrono::steady_clock::now();
    if (ctx == nullptr) {
        std::cerr << "failed to create Lens VAE decode context\n";
        return false;
    }
    e2e_print_vram_snapshot("after_vae_load");

    const int64_t h = latent.shape[2];
    const int64_t w = latent.shape[3];
    sd_latent_t* sd_latent = sd_latent_import_f32(latent.data.data(),
                                                 static_cast<uint64_t>(latent.data.size()),
                                                 static_cast<uint32_t>(w),
                                                 static_cast<uint32_t>(h),
                                                 32);
    if (sd_latent == nullptr) {
        std::cerr << "sd_latent_import_f32 failed\n";
        free_sd_ctx(ctx);
        return false;
    }

    sd_vae_memory_report_t report;
    sd_vae_memory_report_init(&report);
    const auto decode_start = std::chrono::steady_clock::now();
    sd_image_t* image = sd_decode_latent_normal(ctx, sd_latent, nullptr, &report);
    const auto decode_end = std::chrono::steady_clock::now();
    free_sd_latent(sd_latent);
    if (image == nullptr) {
        std::cerr << "Lens VAE decode failed\n";
        free_sd_ctx(ctx);
        return false;
    }
    e2e_print_vram_snapshot("after_vae_decode");
    if (!write_image_to_file(output_png, image->data, image->width, image->height, image->channel)) {
        std::cerr << "failed to write output: " << output_png << "\n";
        free_sd_image(image);
        free_sd_ctx(ctx);
        return false;
    }
    decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
    std::cout << "Lens native e2e VAE wrote " << output_png
              << " image=" << image->width << "x" << image->height << "x" << image->channel
              << " latent=1x32x" << h << "x" << w
              << " vae_load_seconds=" << std::chrono::duration<double>(vae_load_end - vae_load_start).count()
              << " decode_seconds=" << decode_seconds
              << " decode_graph_ms=" << report.decode_graph_ms << "\n";
    free_sd_image(image);
    free_sd_ctx(ctx);
    e2e_print_vram_snapshot("final_cleanup");
    return true;
}

static LensTransformerSmokeRunFn load_transformer_smoke_api() {
#ifdef _WIN32
    wchar_t exe_path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        throw std::runtime_error("GetModuleFileNameW failed while resolving transformer smoke API DLL");
    }
    std::filesystem::path dll_path(exe_path);
    dll_path = dll_path.parent_path() / "sd-lens-transformer-smoke-api.dll";
    HMODULE module = LoadLibraryW(dll_path.wstring().c_str());
    if (module == nullptr) {
        throw std::runtime_error("failed to load " + dll_path.string());
    }
    FARPROC proc = GetProcAddress(module, "sd_lens_transformer_smoke_main_impl");
    if (proc == nullptr) {
        throw std::runtime_error("sd-lens-transformer-smoke-api.dll missing sd_lens_transformer_smoke_main_impl");
    }
    return reinterpret_cast<LensTransformerSmokeRunFn>(proc);
#else
    throw std::runtime_error("sd-lens-native-e2e-smoke transformer smoke API loader is currently Windows-only");
#endif
}

int main(int argc, char** argv) {
    E2EArgs args;
    if (!parse_e2e_args(argc, argv, args)) {
        e2e_usage(argv[0]);
        return 2;
    }
    if (args.steps != 4) {
        std::cerr << "B2.1 Lens native e2e smoke currently expects --steps 4\n";
        return 2;
    }
    if (args.width != args.height || (args.width != 256 && args.width != 512)) {
        std::cerr << "B2.1 Lens native e2e smoke supports square 256 or 512 only\n";
        return 2;
    }

    try {
        create_parent_dir(args.output_png);
        if (!args.optional_cond_out.empty()) {
            create_parent_dir(args.optional_cond_out);
        }

        std::cout << "Lens native e2e smoke start\n" << std::flush;
        e2e_print_vram_snapshot("process_start");
        e2e_print_vram_snapshot("before_text_encoder_create_load");
        sd_lens_text_encoder_result text_result;
        sd_lens_text_encoder* encoder = sd_lens_text_encoder_create();
        sd_lens_text_encoder_load_options load_options;
        load_options.text_encoder_dir = args.text_encoder_dir;
        const auto text_load_start = std::chrono::steady_clock::now();
        if (!sd_lens_text_encoder_load(encoder, load_options, &text_result)) {
            std::cerr << "sd_lens_text_encoder_load failed: " << text_result.error << "\n";
            sd_lens_text_encoder_free(encoder);
            return 1;
        }
        const auto text_load_end = std::chrono::steady_clock::now();
        e2e_print_vram_snapshot("after_text_encoder_load");

        sd_lens_cond_v1_native cond;
        sd_lens_text_encoder_encode_options encode_options;
        encode_options.bootstrap_oracle_dir = args.bootstrap_oracle_dir;
        encode_options.output_safetensors = args.optional_cond_out;
        encode_options.output_condition = &cond;
        const auto text_encode_start = std::chrono::steady_clock::now();
        if (!sd_lens_text_encoder_encode(encoder, encode_options, &text_result)) {
            std::cerr << "sd_lens_text_encoder_encode failed: " << text_result.error << "\n";
            sd_lens_text_encoder_free(encoder);
            return 1;
        }
        const auto text_encode_end = std::chrono::steady_clock::now();
        e2e_print_vram_snapshot("after_text_encoder_encode");

        sd_lens_text_encoder_free(encoder);
        e2e_print_vram_snapshot("after_text_encoder_free");
        std::cout << "Lens native e2e text_encoder stage complete"
                  << " load_seconds=" << std::chrono::duration<double>(text_load_end - text_load_start).count()
                  << " encode_seconds=" << text_result.encode_seconds
                  << " wrapper_wall_seconds=" << std::chrono::duration<double>(text_encode_end - text_encode_start).count()
                  << " condition_handoff=in_memory"
                  << " optional_cond_out=" << (args.optional_cond_out.empty() ? "<none>" : args.optional_cond_out)
                  << "\n";

        std::unordered_map<std::string, Tensor> cond_tensors = make_transformer_condition_map(cond);
        const std::filesystem::path output_path(args.output_png);
        const std::filesystem::path parent = output_path.parent_path().empty() ? std::filesystem::current_path() : output_path.parent_path();
        const std::string stem = output_path.stem().string();
        const std::string latent_npy = (parent / (stem + "_latent.npy")).string();
        const std::string packed_npy = (parent / (stem + "_packed.npy")).string();

        std::vector<std::string> transformer_args = {
            "sd-lens-transformer-smoke",
            "--real-block-transformer", args.transformer_dir,
            "--real-full-transformer",
            "--native-cuda-generate-256",
            "--use-transformer-context",
            "--lens-attention-mode", "regular-f32",
            "--tiny-flow-steps", std::to_string(args.steps),
            "--seed", std::to_string(args.seed),
            "--external-height", std::to_string(args.height),
            "--external-width", std::to_string(args.width),
            "--tiny-denoise-npy", latent_npy,
            "--packed-tokens-npy", packed_npy,
        };
        std::vector<char*> transformer_argv;
        transformer_argv.reserve(transformer_args.size());
        for (std::string& arg : transformer_args) {
            transformer_argv.push_back(arg.data());
        }

        Tensor latent;
        e2e_print_vram_snapshot("before_transformer_load");
        const auto transformer_start = std::chrono::steady_clock::now();
        LensTransformerSmokeRunFn transformer_run = load_transformer_smoke_api();
        const int transformer_rc = transformer_run(static_cast<int>(transformer_argv.size()),
                                                   transformer_argv.data(),
                                                   &cond_tensors,
                                                   &latent);
        const auto transformer_end = std::chrono::steady_clock::now();
        if (transformer_rc != 0) {
            std::cerr << "Lens native e2e transformer stage failed rc=" << transformer_rc << "\n";
            return transformer_rc;
        }
        if (latent.data.empty()) {
            std::cerr << "Lens native e2e transformer stage did not return an in-memory latent\n";
            return 1;
        }

        double vae_decode_seconds = 0.0;
        const auto vae_start = std::chrono::steady_clock::now();
        if (!decode_lens_vae_to_png(args.vae_path, latent, args.output_png, vae_decode_seconds)) {
            return 1;
        }
        const auto vae_end = std::chrono::steady_clock::now();

        std::cout << "Lens native e2e smoke passed"
                  << " output=" << args.output_png
                  << " text_encoder_seconds=" << text_result.encode_seconds
                  << " transformer_wall_seconds=" << std::chrono::duration<double>(transformer_end - transformer_start).count()
                  << " vae_wall_seconds=" << std::chrono::duration<double>(vae_end - vae_start).count()
                  << " vae_decode_seconds=" << vae_decode_seconds
                  << " condition_handoff=in_memory"
                  << " text_encoder_released_before_transformer=true"
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sd-lens-native-e2e-smoke failed: " << e.what() << "\n";
        return 1;
    }
}
