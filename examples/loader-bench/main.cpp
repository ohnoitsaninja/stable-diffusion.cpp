#include "stable-diffusion.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string bench = "all";
    std::string model;
    std::string vae;
    std::string control_net;
    std::string lora;
    std::string prompt = "a test image";
    std::string negative_prompt;
    int threads = 0;
    int width = 1024;
    int height = 1024;
    int steps = 1;
    uint32_t read_threads = 4;
    uint64_t pin_budget_mb = 1024;
    uint64_t max_staging_mb = 256;
    uint64_t min_tensor_kb = 4096;
    uint64_t max_run_mb = 0;
    bool disable_threaded_loader = false;
};

bool require_value(int& index, int argc, char** argv, std::string& out) {
    if (index + 1 >= argc) {
        return false;
    }
    out = argv[++index];
    return true;
}

bool require_int(int& index, int argc, char** argv, int& out) {
    std::string value;
    if (!require_value(index, argc, argv, value)) {
        return false;
    }
    out = std::atoi(value.c_str());
    return true;
}

bool require_u32(int& index, int argc, char** argv, uint32_t& out) {
    std::string value;
    if (!require_value(index, argc, argv, value)) {
        return false;
    }
    out = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
    return true;
}

bool require_u64(int& index, int argc, char** argv, uint64_t& out) {
    std::string value;
    if (!require_value(index, argc, argv, value)) {
        return false;
    }
    out = static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
    return true;
}

void usage() {
    std::cerr
        << "sd-loader-bench --model <path> [--bench cold|warm|lora|workflow|all]\n"
        << "  [--vae <path>] [--control-net <path>] [--lora <path>]\n"
        << "  [--prompt <text>] [--negative-prompt <text>] [--steps <n>]\n"
        << "  [--width <px>] [--height <px>] [--threads <n>] [--read-threads <n>]\n"
        << "  [--pin-budget-mb <n>] [--max-staging-mb <n>] [--min-tensor-kb <n>] [--max-run-mb <n>]\n"
        << "  [--disable-threaded-loader]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") {
            if (!require_value(i, argc, argv, args.bench)) return false;
        } else if (arg == "--model") {
            if (!require_value(i, argc, argv, args.model)) return false;
        } else if (arg == "--vae") {
            if (!require_value(i, argc, argv, args.vae)) return false;
        } else if (arg == "--control-net") {
            if (!require_value(i, argc, argv, args.control_net)) return false;
        } else if (arg == "--lora") {
            if (!require_value(i, argc, argv, args.lora)) return false;
        } else if (arg == "--prompt") {
            if (!require_value(i, argc, argv, args.prompt)) return false;
        } else if (arg == "--negative-prompt") {
            if (!require_value(i, argc, argv, args.negative_prompt)) return false;
        } else if (arg == "--threads") {
            if (!require_int(i, argc, argv, args.threads)) return false;
        } else if (arg == "--width") {
            if (!require_int(i, argc, argv, args.width)) return false;
        } else if (arg == "--height") {
            if (!require_int(i, argc, argv, args.height)) return false;
        } else if (arg == "--steps") {
            if (!require_int(i, argc, argv, args.steps)) return false;
        } else if (arg == "--read-threads") {
            if (!require_u32(i, argc, argv, args.read_threads)) return false;
        } else if (arg == "--pin-budget-mb") {
            if (!require_u64(i, argc, argv, args.pin_budget_mb)) return false;
        } else if (arg == "--max-staging-mb") {
            if (!require_u64(i, argc, argv, args.max_staging_mb)) return false;
        } else if (arg == "--min-tensor-kb") {
            if (!require_u64(i, argc, argv, args.min_tensor_kb)) return false;
        } else if (arg == "--max-run-mb") {
            if (!require_u64(i, argc, argv, args.max_run_mb)) return false;
        } else if (arg == "--disable-threaded-loader") {
            args.disable_threaded_loader = true;
        } else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return !args.model.empty();
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_stats(const std::string& label, double wall_ms) {
    sd_loader_stats_t stats;
    sd_loader_stats_init(&stats);
    const bool stats_ok = sd_get_loader_stats(&stats);
    std::cout << label
              << " wall_ms=" << wall_ms
              << " loader_stats=" << (stats_ok ? "true" : "false")
              << " disk_read_mb=" << (stats.disk_read_bytes / 1024.0 / 1024.0)
              << " h2d_mb=" << (stats.h2d_bytes / 1024.0 / 1024.0)
              << " pinned_peak_mb=" << (stats.pinned_bytes_peak / 1024.0 / 1024.0)
              << " disk_read_ms=" << stats.disk_read_ms
              << " h2d_sync_ms=" << stats.h2d_ms
              << " total_model_load_ms=" << stats.total_model_load_ms
              << " fallback_count=" << stats.fallback_count
              << "\n";
}

sd_ctx_t* create_context(const Args& args) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = args.model.c_str();
    params.vae_path = args.vae.empty() ? nullptr : args.vae.c_str();
    params.control_net_path = args.control_net.empty() ? nullptr : args.control_net.c_str();
    params.n_threads = args.threads;
    params.vae_decode_only = false;
    params.free_params_immediately = false;
    params.enable_mmap = true;
    params.offload_params_to_cpu = false;
    return new_sd_ctx(&params);
}

bool bench_load_only(const Args& args, const std::string& label) {
    sd_reset_loader_stats();
    const auto start = std::chrono::steady_clock::now();
    sd_ctx_t* ctx = create_context(args);
    const auto end = std::chrono::steady_clock::now();
    if (ctx == nullptr) {
        std::cerr << label << " failed to create context\n";
        return false;
    }
    print_stats(label, elapsed_ms(start, end));
    free_sd_ctx(ctx);
    return true;
}

bool bench_workflow(const Args& args, const std::string& label, bool with_lora) {
    sd_reset_loader_stats();
    const auto start = std::chrono::steady_clock::now();
    sd_ctx_t* ctx = create_context(args);
    if (ctx == nullptr) {
        std::cerr << label << " failed to create context\n";
        return false;
    }

    sd_lora_t lora{};
    std::vector<sd_lora_t> loras;
    if (with_lora && !args.lora.empty()) {
        lora.path = args.lora.c_str();
        lora.multiplier = 1.0f;
        loras.push_back(lora);
    }

    sd_img_gen_params_t gen;
    sd_img_gen_params_init(&gen);
    gen.prompt = args.prompt.c_str();
    gen.negative_prompt = args.negative_prompt.c_str();
    gen.width = args.width;
    gen.height = args.height;
    gen.sample_params.sample_steps = args.steps;
    gen.loras = loras.empty() ? nullptr : loras.data();
    gen.lora_count = static_cast<uint32_t>(loras.size());

    sd_image_t* image = generate_image(ctx, &gen);
    const auto end = std::chrono::steady_clock::now();
    if (image == nullptr) {
        std::cerr << label << " failed to generate image\n";
        free_sd_ctx(ctx);
        return false;
    }
    free_sd_image(image);
    print_stats(label, elapsed_ms(start, end));
    free_sd_ctx(ctx);
    return true;
}

bool wants(const Args& args, const char* bench) {
    return args.bench == "all" || args.bench == bench;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 2;
    }

    sd_loader_config_t loader_config;
    sd_loader_config_init(&loader_config);
    loader_config.enable_threaded_loader = !args.disable_threaded_loader;
    loader_config.enable_pinned_staging = !args.disable_threaded_loader;
    loader_config.read_threads = args.read_threads;
    loader_config.pin_budget_bytes = args.pin_budget_mb * 1024ull * 1024ull;
    loader_config.max_staging_bytes = args.max_staging_mb * 1024ull * 1024ull;
    loader_config.min_tensor_bytes = args.min_tensor_kb * 1024ull;
    loader_config.max_run_bytes = args.max_run_mb * 1024ull * 1024ull;
    sd_set_loader_config(&loader_config);

    bool ok = true;
    if (wants(args, "cold")) {
        std::cout << "cold label does not flush the OS file cache; restart or purge externally for a true cold run\n";
        ok = bench_load_only(args, "cold_load") && ok;
    }
    if (wants(args, "warm")) {
        ok = bench_load_only(args, "warm_os_cache_load") && ok;
    }
    if (wants(args, "lora")) {
        ok = bench_workflow(args, "lora_first_run", true) && ok;
    }
    if (wants(args, "workflow")) {
        ok = bench_workflow(args, "sdxl_full_workflow_first_run", false) && ok;
    }
    return ok ? 0 : 1;
}
