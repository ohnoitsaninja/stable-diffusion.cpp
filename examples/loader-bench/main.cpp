#include "stable-diffusion.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string bench = "all";
    std::string model;
    std::string sdxl_model;
    std::string sd15_model;
    std::string sd3_model;
    std::string flux_model;
    std::string flux2_model;
    std::string qwen_image_model;
    std::string anima_model;
    std::string wan_model;
    std::string z_image_model;
    std::string ovis_image_model;
    std::string ernie_image_model;
    std::string chroma_model;
    std::string diffusion_model;
    std::string high_noise_diffusion_model;
    std::string clip_l;
    std::string clip_g;
    std::string clip_vision;
    std::string t5xxl;
    std::string llm;
    std::string llm_vision;
    std::string vae;
    std::string taesd;
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
    bool disable_threaded_loader = false;
};

struct BenchCase {
    std::string label;
    Args args;
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
        << "sd-loader-bench [--model <path> | --sdxl-model <path> | --flux-model <path> ...]\n"
        << "  [--bench cold|warm|repeat|lora|workflow|all]\n"
        << "  [--diffusion-model <path>] [--high-noise-diffusion-model <path>]\n"
        << "  [--clip-l <path>] [--clip-g <path>] [--clip-vision <path>]\n"
        << "  [--t5xxl <path>] [--llm <path>] [--llm-vision <path>]\n"
        << "  [--vae <path>] [--taesd <path>] [--controlnet <path>] [--lora <path>]\n"
        << "  [--prompt <text>] [--negative-prompt <text>] [--steps <n>]\n"
        << "  [--width <px>] [--height <px>] [--threads <n>] [--read-threads <n>]\n"
        << "  [--pin-budget-mb <n>] [--max-staging-mb <n>] [--min-tensor-kb <n>]\n"
        << "  [--disable-threaded-loader]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") {
            if (!require_value(i, argc, argv, args.bench)) return false;
        } else if (arg == "--model") {
            if (!require_value(i, argc, argv, args.model)) return false;
        } else if (arg == "--sdxl-model") {
            if (!require_value(i, argc, argv, args.sdxl_model)) return false;
        } else if (arg == "--sd15-model") {
            if (!require_value(i, argc, argv, args.sd15_model)) return false;
        } else if (arg == "--sd3-model") {
            if (!require_value(i, argc, argv, args.sd3_model)) return false;
        } else if (arg == "--flux-model") {
            if (!require_value(i, argc, argv, args.flux_model)) return false;
        } else if (arg == "--flux2-model") {
            if (!require_value(i, argc, argv, args.flux2_model)) return false;
        } else if (arg == "--qwen-image-model") {
            if (!require_value(i, argc, argv, args.qwen_image_model)) return false;
        } else if (arg == "--anima-model") {
            if (!require_value(i, argc, argv, args.anima_model)) return false;
        } else if (arg == "--wan-model") {
            if (!require_value(i, argc, argv, args.wan_model)) return false;
        } else if (arg == "--z-image-model") {
            if (!require_value(i, argc, argv, args.z_image_model)) return false;
        } else if (arg == "--ovis-image-model") {
            if (!require_value(i, argc, argv, args.ovis_image_model)) return false;
        } else if (arg == "--ernie-image-model") {
            if (!require_value(i, argc, argv, args.ernie_image_model)) return false;
        } else if (arg == "--chroma-model") {
            if (!require_value(i, argc, argv, args.chroma_model)) return false;
        } else if (arg == "--diffusion-model") {
            if (!require_value(i, argc, argv, args.diffusion_model)) return false;
        } else if (arg == "--high-noise-diffusion-model") {
            if (!require_value(i, argc, argv, args.high_noise_diffusion_model)) return false;
        } else if (arg == "--clip-l") {
            if (!require_value(i, argc, argv, args.clip_l)) return false;
        } else if (arg == "--clip-g") {
            if (!require_value(i, argc, argv, args.clip_g)) return false;
        } else if (arg == "--clip-vision") {
            if (!require_value(i, argc, argv, args.clip_vision)) return false;
        } else if (arg == "--t5xxl") {
            if (!require_value(i, argc, argv, args.t5xxl)) return false;
        } else if (arg == "--llm") {
            if (!require_value(i, argc, argv, args.llm)) return false;
        } else if (arg == "--llm-vision") {
            if (!require_value(i, argc, argv, args.llm_vision)) return false;
        } else if (arg == "--vae") {
            if (!require_value(i, argc, argv, args.vae)) return false;
        } else if (arg == "--taesd") {
            if (!require_value(i, argc, argv, args.taesd)) return false;
        } else if (arg == "--control-net" || arg == "--controlnet") {
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
    return true;
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool path_exists(const std::string& path) {
    return !path.empty() && std::filesystem::exists(std::filesystem::path(path));
}

void clear_missing_optional_path(std::string& path, const char* label) {
    if (!path.empty() && !path_exists(path)) {
        std::cout << "SKIPPED " << label << " missing path=" << path << "\n";
        path.clear();
    }
}

void sanitize_optional_paths(Args& args) {
    clear_missing_optional_path(args.diffusion_model, "diffusion_model");
    clear_missing_optional_path(args.high_noise_diffusion_model, "high_noise_diffusion");
    clear_missing_optional_path(args.clip_l, "clip_l");
    clear_missing_optional_path(args.clip_g, "clip_g");
    clear_missing_optional_path(args.clip_vision, "clip_vision");
    clear_missing_optional_path(args.t5xxl, "t5xxl");
    clear_missing_optional_path(args.llm, "llm");
    clear_missing_optional_path(args.llm_vision, "llm_vision");
    clear_missing_optional_path(args.vae, "vae");
    clear_missing_optional_path(args.taesd, "tae");
    clear_missing_optional_path(args.control_net, "controlnet");
    clear_missing_optional_path(args.lora, "lora");
}

void add_model_case(std::vector<BenchCase>& cases, const Args& base, const std::string& label, const std::string& path) {
    if (path.empty()) {
        return;
    }
    if (!path_exists(path)) {
        std::cout << "SKIPPED " << label << " missing path=" << path << "\n";
        return;
    }
    Args case_args = base;
    case_args.model = path;
    sanitize_optional_paths(case_args);
    cases.push_back(BenchCase{label, case_args});
}

bool has_split_component_path(const Args& args) {
    return !args.diffusion_model.empty() ||
           !args.high_noise_diffusion_model.empty() ||
           !args.clip_l.empty() ||
           !args.clip_g.empty() ||
           !args.clip_vision.empty() ||
           !args.t5xxl.empty() ||
           !args.llm.empty() ||
           !args.llm_vision.empty() ||
           !args.vae.empty() ||
           !args.taesd.empty() ||
           !args.control_net.empty();
}

std::vector<BenchCase> build_cases(const Args& args) {
    std::vector<BenchCase> cases;
    add_model_case(cases, args, "model", args.model);
    add_model_case(cases, args, "sdxl", args.sdxl_model);
    add_model_case(cases, args, "sd15", args.sd15_model);
    add_model_case(cases, args, "sd3", args.sd3_model);
    add_model_case(cases, args, "flux", args.flux_model);
    add_model_case(cases, args, "flux2", args.flux2_model);
    add_model_case(cases, args, "qwen_image", args.qwen_image_model);
    add_model_case(cases, args, "anima", args.anima_model);
    add_model_case(cases, args, "wan", args.wan_model);
    add_model_case(cases, args, "z_image", args.z_image_model);
    add_model_case(cases, args, "ovis_image", args.ovis_image_model);
    add_model_case(cases, args, "ernie_image", args.ernie_image_model);
    add_model_case(cases, args, "chroma", args.chroma_model);

    if (cases.empty() && has_split_component_path(args)) {
        Args split_args = args;
        sanitize_optional_paths(split_args);
        if (has_split_component_path(split_args)) {
            cases.push_back(BenchCase{"split_components", split_args});
        }
    }
    return cases;
}

void print_stats(const std::string& label, double wall_ms) {
    sd_loader_stats_t stats;
    sd_loader_stats_init(&stats);
    const bool stats_ok = sd_get_loader_stats(&stats);
    const double fast_mb = (stats.fast_path_bytes != 0 ? stats.fast_path_bytes : stats.h2d_bytes) / 1024.0 / 1024.0;
    std::cout << label
              << " wall_ms=" << wall_ms
              << " loader_stats=" << (stats_ok ? "true" : "false")
              << " disk_read_mb=" << (stats.disk_read_bytes / 1024.0 / 1024.0)
              << " h2d_mb=" << (stats.h2d_bytes / 1024.0 / 1024.0)
              << " fast_path_mb=" << fast_mb
              << " fallback_mb=" << (stats.fallback_bytes / 1024.0 / 1024.0)
              << " pinned_peak_mb=" << (stats.pinned_bytes_peak / 1024.0 / 1024.0)
              << " disk_read_ms=" << stats.disk_read_ms
              << " disk_read_wall_ms=" << stats.disk_read_wall_ms
              << " h2d_sync_ms=" << stats.h2d_ms
              << " h2d_event_ms=" << stats.h2d_event_ms
              << " total_model_load_ms=" << stats.total_model_load_ms
              << " tensor_count=" << stats.tensor_count
              << " fast_tensor_count=" << stats.fast_path_tensor_count
              << " fallback_tensor_count=" << stats.fallback_tensor_count
              << " fallback_count=" << stats.fallback_count
              << " fallback_below_threshold=" << stats.fallback_below_threshold_count
              << " fallback_host_destination=" << stats.fallback_host_destination_count
              << " fallback_null_destination=" << stats.fallback_null_destination_count
              << " fallback_zip_or_indirect=" << stats.fallback_zip_or_indirect_count
              << " fallback_conversion_required=" << stats.fallback_conversion_required_count
              << " fallback_type_mismatch=" << stats.fallback_type_mismatch_count
              << " fallback_unsupported_backend=" << stats.fallback_unsupported_backend_count
              << " fallback_arena_unavailable=" << stats.fallback_arena_unavailable_count
              << " fallback_other=" << stats.fallback_other_count
              << " dry_run_tensor_count=" << stats.dry_run_tensor_count
              << " cudaHostRegister_calls=" << stats.cuda_host_register_count
              << " cudaHostUnregister_calls=" << stats.cuda_host_unregister_count
              << " cudaStreamSynchronize_calls=" << stats.cuda_stream_synchronize_count
              << " cudaDeviceSynchronize_calls=" << stats.cuda_device_synchronize_count
              << "\n";
}

sd_ctx_t* create_context(const Args& args) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = args.model.empty() ? nullptr : args.model.c_str();
    params.clip_l_path = args.clip_l.empty() ? nullptr : args.clip_l.c_str();
    params.clip_g_path = args.clip_g.empty() ? nullptr : args.clip_g.c_str();
    params.clip_vision_path = args.clip_vision.empty() ? nullptr : args.clip_vision.c_str();
    params.t5xxl_path = args.t5xxl.empty() ? nullptr : args.t5xxl.c_str();
    params.llm_path = args.llm.empty() ? nullptr : args.llm.c_str();
    params.llm_vision_path = args.llm_vision.empty() ? nullptr : args.llm_vision.c_str();
    params.diffusion_model_path = args.diffusion_model.empty() ? nullptr : args.diffusion_model.c_str();
    params.high_noise_diffusion_model_path = args.high_noise_diffusion_model.empty() ? nullptr : args.high_noise_diffusion_model.c_str();
    params.vae_path = args.vae.empty() ? nullptr : args.vae.c_str();
    params.taesd_path = args.taesd.empty() ? nullptr : args.taesd.c_str();
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
        if (!path_exists(args.lora)) {
            std::cout << "SKIPPED " << label << " missing lora path=" << args.lora << "\n";
            free_sd_ctx(ctx);
            return true;
        }
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

bool bench_same_process_repeat(const Args& args, const std::string& label) {
    bool ok = true;
    ok = bench_load_only(args, label + "_same_process_load_1") && ok;
    ok = bench_load_only(args, label + "_same_process_load_2") && ok;
    return ok;
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
    sd_set_loader_config(&loader_config);

    std::vector<BenchCase> cases = build_cases(args);
    if (cases.empty()) {
        std::cerr << "no existing model/component paths were provided\n";
        return 2;
    }

    bool ok = true;
    if (wants(args, "cold")) {
        std::cout << "cold label does not flush the OS file cache; restart or purge externally for a true cold run\n";
    }
    for (const BenchCase& bench_case : cases) {
        if (wants(args, "cold")) {
            ok = bench_load_only(bench_case.args, bench_case.label + "_coldish_load") && ok;
        }
        if (wants(args, "warm")) {
            ok = bench_load_only(bench_case.args, bench_case.label + "_warm_os_cache_load") && ok;
        }
        if (wants(args, "repeat")) {
            ok = bench_same_process_repeat(bench_case.args, bench_case.label) && ok;
        }
        if (wants(args, "lora")) {
            if (bench_case.args.lora.empty()) {
                std::cout << "SKIPPED " << bench_case.label << "_lora_first_run no lora path\n";
            } else {
                ok = bench_workflow(bench_case.args, bench_case.label + "_lora_first_run", true) && ok;
            }
        }
        if (wants(args, "workflow")) {
            ok = bench_workflow(bench_case.args, bench_case.label + "_one_step_workflow", false) && ok;
        }
    }
    return ok ? 0 : 1;
}
