#include "common/media_io.h"
#include "stable-diffusion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(SD_USE_CUDA) && defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SD_BONSAI_BENCH_HAS_NVTX 1
#endif
#endif

namespace {

struct Args {
    std::string diffusion_model;
    std::string llm;
    std::string vae;
    std::string output_dir = "bonsai-bench";
    std::string prompt = "A bonsai tree in a quiet ceramic studio, soft morning light";
    std::string negative_prompt;
    std::string sampler = "euler";
    std::string scheduler = "simple";
    std::string prediction = "flux2_flow";
    int width = 512;
    int height = 512;
    int steps = 1;
    int warmup = 1;
    int runs = 3;
    int threads = -1;
    int64_t seed = 42;
    float cfg = 1.0f;
    float guidance = 1.0f;
    float flow_shift = 3.0f;
    bool diffusion_flash_attn = true;
    bool profile_linears = false;
};

void bench_log_callback(enum sd_log_level_t level, const char* text, void*) {
    const char* prefix = "info";
    switch (level) {
        case SD_LOG_DEBUG: prefix = "debug"; break;
        case SD_LOG_INFO: prefix = "info"; break;
        case SD_LOG_WARN: prefix = "warn"; break;
        case SD_LOG_ERROR: prefix = "error"; break;
        default: break;
    }
    std::cerr << "[" << prefix << "] " << (text != nullptr ? text : "") << "\n";
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void usage() {
    std::cerr
        << "sd-bonsai-gemlite-int1-bench --diffusion-model <path> --llm <path> --vae <path> [options]\n"
        << "  --prompt <text>\n"
        << "  --negative-prompt <text>\n"
        << "  --width <px> --height <px> --steps <n> --seed <n>\n"
        << "  --sampler <name> --scheduler <name> --cfg-scale <v> --guidance <v> --flow-shift <v>\n"
        << "  --prediction <eps|v|edm_v|sd3_flow|flux_flow|flux2_flow>\n"
        << "  --warmup <n> --runs <n> --output-dir <dir> --profile-linears\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const char* v = nullptr;
        if (arg == "--diffusion-model" && (v = need(i))) args.diffusion_model = v;
        else if (arg == "--llm" && (v = need(i))) args.llm = v;
        else if (arg == "--vae" && (v = need(i))) args.vae = v;
        else if ((arg == "--prompt" || arg == "-p") && (v = need(i))) args.prompt = v;
        else if (arg == "--negative-prompt" && (v = need(i))) args.negative_prompt = v;
        else if ((arg == "--width" || arg == "-W") && (v = need(i))) args.width = std::atoi(v);
        else if ((arg == "--height" || arg == "-H") && (v = need(i))) args.height = std::atoi(v);
        else if (arg == "--steps" && (v = need(i))) args.steps = std::atoi(v);
        else if (arg == "--seed" && (v = need(i))) args.seed = std::strtoll(v, nullptr, 10);
        else if (arg == "--sampler" && (v = need(i))) args.sampler = v;
        else if (arg == "--scheduler" && (v = need(i))) args.scheduler = v;
        else if (arg == "--prediction" && (v = need(i))) args.prediction = v;
        else if (arg == "--cfg-scale" && (v = need(i))) args.cfg = static_cast<float>(std::atof(v));
        else if (arg == "--guidance" && (v = need(i))) args.guidance = static_cast<float>(std::atof(v));
        else if (arg == "--flow-shift" && (v = need(i))) args.flow_shift = static_cast<float>(std::atof(v));
        else if (arg == "--warmup" && (v = need(i))) args.warmup = std::atoi(v);
        else if (arg == "--runs" && (v = need(i))) args.runs = std::atoi(v);
        else if (arg == "--threads" && (v = need(i))) args.threads = std::atoi(v);
        else if (arg == "--output-dir" && (v = need(i))) args.output_dir = v;
        else if (arg == "--profile-linears") args.profile_linears = true;
        else if (arg == "--no-diffusion-fa") args.diffusion_flash_attn = false;
        else {
            std::cerr << "unknown or incomplete argument: " << arg << "\n";
            return false;
        }
    }
    if (args.diffusion_model.empty() || args.llm.empty() || args.vae.empty()) {
        return false;
    }
    if (args.width <= 0 || args.height <= 0 || args.steps <= 0 || args.runs <= 0 || args.warmup < 0) {
        return false;
    }
    return true;
}

bool profile_range_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_PROFILE_RANGE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

class ScopedBenchNvtxRange {
public:
    ScopedBenchNvtxRange(const char* name, bool active) : active_(active) {
        if (!active_) {
            return;
        }
#if defined(SD_BONSAI_BENCH_HAS_NVTX)
        nvtxRangePushA(name != nullptr ? name : "bonsai.bench");
#else
        (void)name;
#endif
    }

    ~ScopedBenchNvtxRange() {
#if defined(SD_BONSAI_BENCH_HAS_NVTX)
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

private:
    bool active_ = false;
};

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1) != 0) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

struct ImageStats {
    uint8_t min_value = 255;
    uint8_t max_value = 0;
    double mean = 0.0;
};

ImageStats image_stats(const sd_image_t& image) {
    ImageStats stats;
    const size_t count = static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * static_cast<size_t>(image.channel);
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const uint8_t v = image.data[i];
        stats.min_value = std::min(stats.min_value, v);
        stats.max_value = std::max(stats.max_value, v);
        sum += v;
    }
    stats.mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
    return stats;
}

sd_ctx_t* create_context(const Args& args) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.diffusion_model_path = args.diffusion_model.c_str();
    params.llm_path = args.llm.c_str();
    params.vae_path = args.vae.c_str();
    params.n_threads = args.threads;
    params.vae_decode_only = false;
    params.free_params_immediately = false;
    params.enable_mmap = true;
    params.diffusion_flash_attn = args.diffusion_flash_attn;
    params.prediction = str_to_prediction(args.prediction.c_str());
    params.model_path = "";
    params.clip_l_path = "";
    params.clip_g_path = "";
    params.clip_vision_path = "";
    params.t5xxl_path = "";
    params.llm_vision_path = "";
    params.high_noise_diffusion_model_path = "";
    params.taesd_path = "";
    params.control_net_path = "";
    params.photo_maker_path = "";
    params.tensor_type_rules = "";
    return new_sd_ctx(&params);
}

sd_img_gen_params_t make_generation_params(const Args& args, int64_t seed) {
    sd_img_gen_params_t gen;
    sd_img_gen_params_init(&gen);
    gen.prompt = args.prompt.c_str();
    gen.negative_prompt = args.negative_prompt.c_str();
    gen.width = args.width;
    gen.height = args.height;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.sample_params.sample_steps = args.steps;
    gen.sample_params.guidance.txt_cfg = args.cfg;
    gen.sample_params.guidance.distilled_guidance = args.guidance;
    gen.sample_params.flow_shift = args.flow_shift;
    gen.sample_params.sample_method = str_to_sample_method(args.sampler.c_str());
    gen.sample_params.scheduler = str_to_scheduler(args.scheduler.c_str());
    return gen;
}

struct RunResult {
    bool ok = false;
    double generate_ms = 0.0;
    double png_save_ms = 0.0;
    sd_bonsai_generation_timing_t timing{};
    std::string output_path;
};

RunResult run_once(sd_ctx_t* ctx, const Args& args, int index, bool save_output) {
    RunResult result;
    const bool profile_range = profile_range_enabled();
    const int64_t seed = args.seed + index;
    sd_img_gen_params_t gen = make_generation_params(args, seed);
    auto generate_start = std::chrono::steady_clock::now();
    sd_image_t* image = nullptr;
    {
        ScopedBenchNvtxRange generate_range("bonsai.generate_image", profile_range);
        image = generate_image(ctx, &gen);
    }
    auto generate_end = std::chrono::steady_clock::now();
    result.generate_ms = elapsed_ms(generate_start, generate_end);
    result.timing.struct_size = sizeof(sd_bonsai_generation_timing_t);
    sd_bonsai_get_last_generation_timing(ctx, &result.timing);
    if (image == nullptr || image->data == nullptr) {
        return result;
    }

    if (save_output) {
        std::ostringstream filename;
        filename << "bonsai_bench_run_" << std::setw(2) << std::setfill('0') << index << ".png";
        result.output_path = (std::filesystem::path(args.output_dir) / filename.str()).string();
        auto save_start = std::chrono::steady_clock::now();
        {
            ScopedBenchNvtxRange png_range("bonsai.png_save", profile_range);
            result.ok = write_image_to_file(result.output_path, image->data, image->width, image->height, image->channel);
        }
        auto save_end = std::chrono::steady_clock::now();
        result.png_save_ms = elapsed_ms(save_start, save_end);
        const ImageStats stats = image_stats(*image);
        std::cout << "[BonsaiBench] image path=" << result.output_path
                  << " min=" << static_cast<int>(stats.min_value)
                  << " max=" << static_cast<int>(stats.max_value)
                  << " mean=" << std::fixed << std::setprecision(3) << stats.mean
                  << " png_save_ms=" << result.png_save_ms << "\n";
    } else {
        result.ok = true;
    }
    free_sd_image(image);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 2;
    }

    set_env("SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1", "1");
    set_env("SDCPP_MODEL_FAMILY_HINT", "bonsai");
    set_env("SDCPP_PROFILE_BONSAI_GENERATION", "1");
    if (args.profile_linears) {
        set_env("SDCPP_PROFILE_BONSAI_INT1", "1");
    }

    sd_set_log_callback(bench_log_callback, nullptr);

    std::filesystem::create_directories(args.output_dir);

    std::cout << "[BonsaiBench] settings"
              << " resolution=" << args.width << "x" << args.height
              << " steps=" << args.steps
              << " sampler=" << args.sampler
              << " scheduler=" << args.scheduler
              << " prediction=" << args.prediction
              << " cfg=" << args.cfg
              << " guidance=" << args.guidance
              << " requested_shift=" << args.flow_shift
              << " seed=" << args.seed
              << " prompt=\"" << args.prompt << "\""
              << " elapsed_includes_model_load=false"
              << " warmup=" << args.warmup
              << " runs=" << args.runs
              << " profile_linears=" << (args.profile_linears ? "true" : "false")
              << "\n";

    const auto load_start = std::chrono::steady_clock::now();
    sd_ctx_t* ctx = create_context(args);
    const auto load_end = std::chrono::steady_clock::now();
    if (ctx == nullptr) {
        std::cerr << "failed to create sd context\n";
        return 1;
    }
    std::cout << "[BonsaiBench] model_load_ms=" << elapsed_ms(load_start, load_end)
              << " elapsed_includes_model_load=true\n";

    if (args.warmup > 0) {
        for (int i = 0; i < args.warmup; ++i) {
            RunResult warmup = run_once(ctx, args, i, false);
            std::cout << "[BonsaiBench] warmup index=" << i
                      << " ok=" << (warmup.ok ? "true" : "false")
                      << " generate_ms=" << warmup.generate_ms
                      << " denoise_ms=" << warmup.timing.transformer_denoise_ms
                      << " vae_decode_ms=" << warmup.timing.vae_decode_ms
                      << "\n";
            if (!warmup.ok) {
                free_sd_ctx(ctx);
                return 1;
            }
        }
    }

    std::vector<double> full_ms;
    std::vector<double> generate_ms;
    std::vector<double> denoise_ms;
    std::vector<double> text_ms;
    std::vector<double> vae_ms;
    std::vector<double> png_ms;

    bool ok = true;
    for (int i = 0; i < args.runs; ++i) {
        RunResult run = run_once(ctx, args, i, true);
        ok = ok && run.ok;
        const double full = run.generate_ms + run.png_save_ms;
        full_ms.push_back(full);
        generate_ms.push_back(run.generate_ms);
        denoise_ms.push_back(run.timing.transformer_denoise_ms);
        text_ms.push_back(run.timing.text_encode_ms);
        vae_ms.push_back(run.timing.vae_decode_ms);
        png_ms.push_back(run.png_save_ms);
        std::cout << "[BonsaiBench] run index=" << i
                  << " ok=" << (run.ok ? "true" : "false")
                  << " seed=" << (args.seed + i)
                  << " text_encode_ms=" << run.timing.text_encode_ms
                  << " denoise_ms=" << run.timing.transformer_denoise_ms
                  << " vae_decode_ms=" << run.timing.vae_decode_ms
                  << " png_save_ms=" << run.png_save_ms
                  << " generate_ms=" << run.generate_ms
                  << " full_image_ms=" << full
                  << " output=" << run.output_path
                  << "\n";
    }

    std::cout << "[BonsaiBench] summary"
              << " runs=" << full_ms.size()
              << " full_mean_ms=" << mean(full_ms)
              << " full_median_ms=" << median(full_ms)
              << " generate_mean_ms=" << mean(generate_ms)
              << " generate_median_ms=" << median(generate_ms)
              << " denoise_mean_ms=" << mean(denoise_ms)
              << " denoise_median_ms=" << median(denoise_ms)
              << " text_encode_mean_ms=" << mean(text_ms)
              << " vae_decode_mean_ms=" << mean(vae_ms)
              << " png_save_mean_ms=" << mean(png_ms)
              << " elapsed_includes_model_load=false"
              << "\n";

    free_sd_ctx(ctx);
    return ok ? 0 : 1;
}
