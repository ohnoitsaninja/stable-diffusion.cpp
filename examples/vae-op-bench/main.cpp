#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

struct Args {
    std::string out_json = "sdcpp_op_bench.json";
    std::string out_md   = "sdcpp_op_bench.md";
    int warmup           = 10;
    int iterations       = 50;
    int memory_cap_mb    = 8192;
    bool include_legacy  = false;
    bool correctness     = false;
};

struct BenchCase {
    std::string name;
    std::string op;
    int w       = 0;
    int h       = 0;
    int c       = 0;
    int out_c   = 0;
    int k       = 3;
    int groups  = 32;
    bool direct = true;
    bool implicit = false;
    bool weight_f16 = false;
    bool scaled_ref = false;
    bool fused_scale = false;
    float conv_scale = 1.0f;
};

struct BenchResult {
    BenchCase bench;
    std::string status = "ok";
    std::string error;
    double mean_ms   = 0.0;
    double median_ms = 0.0;
    double min_ms    = 0.0;
    double p95_ms    = 0.0;
    double max_ms    = 0.0;
    size_t buffer_bytes = 0;
    std::vector<double> times_ms;
    std::vector<float> output;
};

struct BuiltGraph {
    ggml_tensor* out = nullptr;
    std::vector<ggml_tensor*> inputs;
};

struct CorrectnessResult {
    BenchCase bench;
    std::string status = "ok";
    std::string error;
    BenchResult direct;
    BenchResult implicit;
    double mean_abs_diff = 0.0;
    double p95_abs_diff = 0.0;
    double p99_abs_diff = 0.0;
    double max_abs_diff = 0.0;
    uint64_t direct_hash = 0;
    uint64_t implicit_hash = 0;
    uint64_t direct_nan_inf = 0;
    uint64_t implicit_nan_inf = 0;
    float direct_min = 0.0f;
    float direct_max = 0.0f;
    double direct_mean = 0.0;
    float implicit_min = 0.0f;
    float implicit_max = 0.0f;
    double implicit_mean = 0.0;
};

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--out-json path] [--out-md path] [--warmup n] [--iterations n] [--memory-cap-mb n] [--include-legacy] [--correctness]\n";
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
        if (arg == "--out-json") {
            const char* v = need_value("--out-json");
            if (v == nullptr) return false;
            args.out_json = v;
        } else if (arg == "--out-md") {
            const char* v = need_value("--out-md");
            if (v == nullptr) return false;
            args.out_md = v;
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup");
            if (v == nullptr) return false;
            args.warmup = std::atoi(v);
        } else if (arg == "--iterations") {
            const char* v = need_value("--iterations");
            if (v == nullptr) return false;
            args.iterations = std::atoi(v);
        } else if (arg == "--memory-cap-mb") {
            const char* v = need_value("--memory-cap-mb");
            if (v == nullptr) return false;
            args.memory_cap_mb = std::atoi(v);
        } else if (arg == "--include-legacy") {
            args.include_legacy = true;
        } else if (arg == "--correctness") {
            args.correctness = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return args.warmup >= 0 && args.iterations > 0 && args.memory_cap_mb > 0;
}

static std::vector<BenchCase> make_cases(bool include_legacy) {
    std::vector<BenchCase> cases = {
        {"direct_conv_decode_stage1_4c_to_512_128", "conv2d_direct", 128, 128, 4, 512, 3, 32, true, false},
        {"implicit_conv_decode_stage1_4c_to_512_128", "conv2d_implicit", 128, 128, 4, 512, 3, 32, true, true},
        {"direct_conv_decode_1x1_512c_128", "conv2d_direct", 128, 128, 512, 512, 1, 32, true, false},
        {"implicit_conv_decode_1x1_512c_128", "conv2d_implicit", 128, 128, 512, 512, 1, 32, true, true},
        {"direct_conv_stage3_512c_512", "conv2d_direct", 512, 512, 512, 512, 3, 32, true, false},
        {"implicit_conv_stage3_512c_512", "conv2d_implicit", 512, 512, 512, 512, 3, 32, true, true},
        {"direct_conv_stage4_256c_1024", "conv2d_direct", 1024, 1024, 256, 256, 3, 32, true, false},
        {"implicit_conv_stage4_256c_1024", "conv2d_implicit", 1024, 1024, 256, 256, 3, 32, true, true},
        {"direct_conv_stage5_128c_1024", "conv2d_direct", 1024, 1024, 128, 128, 3, 32, true, false},
        {"implicit_conv_stage5_128c_1024", "conv2d_implicit", 1024, 1024, 128, 128, 3, 32, true, true},
        {"group_norm_stage5_256c_1024", "group_norm", 1024, 1024, 256, 0, 0, 32, true, false},
        {"upscale_stage3_512c_256_to_512", "upscale", 256, 256, 512, 0, 0, 32, true, false},
        {"upscale_stage4_256c_512_to_1024", "upscale", 512, 512, 256, 0, 0, 32, true, false},
        {"pointwise_stage4_256c_1024", "pointwise", 1024, 1024, 256, 0, 0, 32, true, false},
    };
    if (include_legacy) {
        cases.push_back({"legacy_conv_stage5_128c_1024", "conv2d_legacy", 1024, 1024, 128, 128, 3, 32, false, false});
    }
    return cases;
}

static void set_implicit_conv_env(bool enabled) {
#ifdef _WIN32
    _putenv_s("SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV", enabled ? "1" : "");
#else
    if (enabled) {
        setenv("SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV", "1", 1);
    } else {
        unsetenv("SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV");
    }
#endif
}

static double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t idx = static_cast<size_t>(std::min<double>(values.size() - 1, std::max<double>(0.0, (values.size() - 1) * q)));
    return values[idx];
}

static void fill_stats(BenchResult& result) {
    if (result.times_ms.empty()) return;
    result.mean_ms = std::accumulate(result.times_ms.begin(), result.times_ms.end(), 0.0) / static_cast<double>(result.times_ms.size());
    std::vector<double> sorted = result.times_ms;
    std::sort(sorted.begin(), sorted.end());
    result.median_ms = sorted[sorted.size() / 2];
    result.min_ms    = sorted.front();
    result.max_ms    = sorted.back();
    result.p95_ms    = percentile(sorted, 0.95);
}

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static BuiltGraph build_graph_for_case(ggml_context* ctx, const BenchCase& bench) {
    BuiltGraph built;
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bench.w, bench.h, bench.c, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    built.inputs.push_back(x);

    if (bench.op == "conv2d_direct" || bench.op == "conv2d_implicit" || bench.op == "conv2d_legacy") {
        const ggml_type weight_type = bench.weight_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor* w = ggml_new_tensor_4d(ctx, weight_type, bench.k, bench.k, bench.c, bench.out_c);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bench.out_c);
        ggml_set_name(w, "weight");
        ggml_set_name(b, "bias");
        ggml_set_input(w);
        ggml_set_input(b);
        built.inputs.push_back(w);
        built.inputs.push_back(b);
        ggml_tensor* conv_input = x;
        if (bench.scaled_ref && bench.conv_scale != 1.0f) {
            conv_input = ggml_scale(ctx, conv_input, bench.conv_scale);
        }
        ggml_tensor* y = bench.direct ? ggml_conv_2d_direct(ctx, w, conv_input, 1, 1, 1, 1, 1, 1)
                                      : ggml_conv_2d(ctx, w, conv_input, 1, 1, 1, 1, 1, 1);
        if (bench.fused_scale && bench.conv_scale != 1.0f) {
            reinterpret_cast<float*>(y->op_params)[6] = bench.conv_scale;
        }
        if (bench.scaled_ref && bench.conv_scale != 1.0f) {
            y = ggml_scale(ctx, y, 1.0f / bench.conv_scale);
        }
        b = ggml_reshape_4d(ctx, b, 1, 1, bench.out_c, 1);
        y = ggml_add_inplace(ctx, y, b);
        ggml_set_name(y, "output");
        ggml_set_output(y);
        built.out = y;
        return built;
    }

    if (bench.op == "group_norm") {
        ggml_tensor* y = ggml_group_norm(ctx, x, bench.groups, 1e-6f);
        ggml_set_name(y, "output");
        ggml_set_output(y);
        built.out = y;
        return built;
    }

    if (bench.op == "upscale") {
        ggml_tensor* y = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
        ggml_set_name(y, "output");
        ggml_set_output(y);
        built.out = y;
        return built;
    }

    if (bench.op == "pointwise") {
        ggml_tensor* residual = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, bench.w, bench.h, bench.c, 1);
        ggml_set_name(residual, "residual");
        ggml_set_input(residual);
        built.inputs.push_back(residual);
        ggml_tensor* y = ggml_scale(ctx, x, 0.18215f);
        y = ggml_silu(ctx, y);
        y = ggml_scale(ctx, y, 0.875f);
        y = ggml_add(ctx, y, residual);
        ggml_set_name(y, "output");
        ggml_set_output(y);
        built.out = y;
        return built;
    }

    return built;
}

static float deterministic_value(uint64_t i, uint64_t salt, float scale) {
    const uint64_t x = (i * 1103515245ull + 12345ull + salt * 2654435761ull) & 0x00ffffffull;
    const float centered = static_cast<float>(static_cast<int64_t>(x % 20001ull) - 10000) / 10000.0f;
    const float wave = std::sin(static_cast<float>((i + 1) * (salt + 3)) * 0.013f);
    return (0.75f * centered + 0.25f * wave) * scale;
}

static void fill_input_tensor(ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        return;
    }
    const int64_t n = ggml_nelements(tensor);
    std::vector<float> data(static_cast<size_t>(n));
    const std::string name = tensor->name;
    float scale = 0.25f;
    uint64_t salt = 1;
    if (name.find("weight") != std::string::npos) {
        scale = 0.015f;
        salt = 17;
    } else if (name.find("bias") != std::string::npos) {
        scale = 0.002f;
        salt = 29;
    } else if (name.find("residual") != std::string::npos) {
        scale = 0.125f;
        salt = 41;
    }
    for (int64_t i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = deterministic_value(static_cast<uint64_t>(i), salt, scale);
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data_f16(static_cast<size_t>(n));
        ggml_fp32_to_fp16_row(data.data(), data_f16.data(), n);
        ggml_backend_tensor_set(tensor, data_f16.data(), 0, data_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    }
}

static BenchResult run_case(ggml_backend_t backend, const BenchCase& bench, const Args& args, bool capture_output = false) {
    BenchResult result;
    result.bench = bench;
    set_implicit_conv_env(bench.implicit);

    ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        result.status = "error";
        result.error  = "ggml_init failed";
        set_implicit_conv_env(false);
        return result;
    }

    BuiltGraph built = build_graph_for_case(ctx, bench);
    if (built.out == nullptr) {
        result.status = "error";
        result.error  = "unsupported case";
        ggml_free(ctx);
        set_implicit_conv_env(false);
        return result;
    }

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, built.out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        result.status = "skipped";
        result.error  = "backend tensor allocation failed";
        ggml_free(ctx);
        set_implicit_conv_env(false);
        return result;
    }
    result.buffer_bytes = ggml_backend_buffer_get_size(buffer);
    const size_t cap_bytes = static_cast<size_t>(args.memory_cap_mb) * 1024ull * 1024ull;
    if (result.buffer_bytes > cap_bytes) {
        result.status = "skipped";
        std::ostringstream ss;
        ss << "buffer " << (result.buffer_bytes / 1024.0 / 1024.0) << " MB exceeds cap " << args.memory_cap_mb << " MB";
        result.error = ss.str();
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        set_implicit_conv_env(false);
        return result;
    }

    ggml_backend_buffer_clear(buffer, 0);
    for (ggml_tensor* input : built.inputs) {
        fill_input_tensor(input);
    }
    ggml_backend_synchronize(backend);

    auto compute_once = [&]() -> bool {
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        return status == GGML_STATUS_SUCCESS;
    };

    for (int i = 0; i < args.warmup; ++i) {
        if (!compute_once()) {
            result.status = "error";
            result.error  = "warmup compute failed";
            break;
        }
    }
    if (result.status == "ok") {
        for (int i = 0; i < args.iterations; ++i) {
            int64_t start = now_us();
            if (!compute_once()) {
                result.status = "error";
                result.error  = "compute failed";
                break;
            }
            int64_t end = now_us();
            result.times_ms.push_back(static_cast<double>(end - start) / 1000.0);
        }
    }
    fill_stats(result);
    if (result.status == "ok" && capture_output) {
        const int64_t n = ggml_nelements(built.out);
        if (built.out->type == GGML_TYPE_F32 && n > 0) {
            result.output.resize(static_cast<size_t>(n));
            ggml_backend_tensor_get(built.out, result.output.data(), 0, result.output.size() * sizeof(float));
        } else {
            result.status = "error";
            result.error = "capture only supports f32 output";
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    set_implicit_conv_env(false);
    return result;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

static uint64_t fnv1a_bytes(const std::vector<float>& values) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(values.data());
    const size_t size = values.size() * sizeof(float);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void fill_value_stats(const std::vector<float>& values, float& min_v, float& max_v, double& mean, uint64_t& nan_inf) {
    min_v = std::numeric_limits<float>::infinity();
    max_v = -std::numeric_limits<float>::infinity();
    long double sum = 0.0;
    nan_inf = 0;
    uint64_t finite_count = 0;
    for (float v : values) {
        if (!std::isfinite(v)) {
            ++nan_inf;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += v;
        ++finite_count;
    }
    if (finite_count == 0) {
        min_v = 0.0f;
        max_v = 0.0f;
        mean = 0.0;
    } else {
        mean = static_cast<double>(sum / finite_count);
    }
}

static CorrectnessResult compare_conv_case(ggml_backend_t backend, const BenchCase& direct_case, const Args& args) {
    CorrectnessResult result;
    result.bench = direct_case;
    BenchCase implicit_case = direct_case;
    implicit_case.name = "implicit_" + direct_case.name;
    implicit_case.op = "conv2d_implicit";
    implicit_case.direct = true;
    implicit_case.implicit = true;

    BenchCase direct = direct_case;
    direct.op = "conv2d_direct";
    direct.direct = true;
    direct.implicit = false;

    Args local_args = args;
    local_args.warmup = std::min(args.warmup, 1);
    local_args.iterations = std::min(args.iterations, 3);

    result.direct = run_case(backend, direct, local_args, true);
    result.implicit = run_case(backend, implicit_case, local_args, true);
    if (result.direct.status != "ok" || result.implicit.status != "ok") {
        result.status = "error";
        result.error = "direct or implicit run failed";
        return result;
    }
    if (result.direct.output.size() != result.implicit.output.size() || result.direct.output.empty()) {
        result.status = "error";
        result.error = "output size mismatch or empty output";
        return result;
    }

    result.direct_hash = fnv1a_bytes(result.direct.output);
    result.implicit_hash = fnv1a_bytes(result.implicit.output);
    fill_value_stats(result.direct.output, result.direct_min, result.direct_max, result.direct_mean, result.direct_nan_inf);
    fill_value_stats(result.implicit.output, result.implicit_min, result.implicit_max, result.implicit_mean, result.implicit_nan_inf);

    const size_t n = result.direct.output.size();
    const size_t max_samples = 1000000;
    const size_t stride = std::max<size_t>(1, n / max_samples);
    std::vector<double> sampled_diffs;
    sampled_diffs.reserve(std::min(max_samples, n));
    long double sum_abs = 0.0;
    double max_abs = 0.0;
    uint64_t valid_count = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = result.direct.output[i];
        const float b = result.implicit.output[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            continue;
        }
        const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
        sum_abs += diff;
        max_abs = std::max(max_abs, diff);
        ++valid_count;
        if ((i % stride) == 0) {
            sampled_diffs.push_back(diff);
        }
    }
    result.mean_abs_diff = valid_count == 0 ? 0.0 : static_cast<double>(sum_abs / valid_count);
    result.max_abs_diff = max_abs;
    result.p95_abs_diff = percentile(sampled_diffs, 0.95);
    result.p99_abs_diff = percentile(sampled_diffs, 0.99);
    return result;
}

static CorrectnessResult compare_scaled_conv_case(ggml_backend_t backend, const BenchCase& base_case, const Args& args) {
    CorrectnessResult result;
    result.bench = base_case;

    BenchCase reference = base_case;
    reference.name = "scaled_ref_" + base_case.name;
    reference.op = "conv2d_direct";
    reference.direct = true;
    reference.implicit = true;
    reference.weight_f16 = true;
    reference.scaled_ref = true;
    reference.fused_scale = false;

    BenchCase fused = base_case;
    fused.name = "fused_scale_" + base_case.name;
    fused.op = "conv2d_implicit";
    fused.direct = true;
    fused.implicit = true;
    fused.weight_f16 = true;
    fused.scaled_ref = false;
    fused.fused_scale = true;

    Args local_args = args;
    local_args.warmup = std::min(args.warmup, 1);
    local_args.iterations = std::min(args.iterations, 3);

    result.direct = run_case(backend, reference, local_args, true);
    result.implicit = run_case(backend, fused, local_args, true);
    if (result.direct.status != "ok" || result.implicit.status != "ok") {
        result.status = "error";
        result.error = "scaled reference or fused implicit run failed";
        return result;
    }
    if (result.direct.output.size() != result.implicit.output.size() || result.direct.output.empty()) {
        result.status = "error";
        result.error = "output size mismatch or empty output";
        return result;
    }

    result.direct_hash = fnv1a_bytes(result.direct.output);
    result.implicit_hash = fnv1a_bytes(result.implicit.output);
    fill_value_stats(result.direct.output, result.direct_min, result.direct_max, result.direct_mean, result.direct_nan_inf);
    fill_value_stats(result.implicit.output, result.implicit_min, result.implicit_max, result.implicit_mean, result.implicit_nan_inf);

    const size_t n = result.direct.output.size();
    const size_t max_samples = 1000000;
    const size_t stride = std::max<size_t>(1, n / max_samples);
    std::vector<double> sampled_diffs;
    sampled_diffs.reserve(std::min(max_samples, n));
    long double sum_abs = 0.0;
    double max_abs = 0.0;
    uint64_t valid_count = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = result.direct.output[i];
        const float b = result.implicit.output[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            continue;
        }
        const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
        sum_abs += diff;
        max_abs = std::max(max_abs, diff);
        ++valid_count;
        if ((i % stride) == 0) {
            sampled_diffs.push_back(diff);
        }
    }
    result.mean_abs_diff = valid_count == 0 ? 0.0 : static_cast<double>(sum_abs / valid_count);
    result.max_abs_diff = max_abs;
    result.p95_abs_diff = percentile(sampled_diffs, 0.95);
    result.p99_abs_diff = percentile(sampled_diffs, 0.99);
    return result;
}

static std::vector<BenchCase> make_conv_correctness_cases() {
    return {
        {"decode_stage1_4c_to_512_128", "conv2d_direct", 128, 128, 4, 512, 3, 32, true, false},
        {"decode_1x1_512c_128", "conv2d_direct", 128, 128, 512, 512, 1, 32, true, false},
        {"decode_mid_512c_256", "conv2d_direct", 256, 256, 512, 512, 3, 32, true, false},
        {"decode_stage3_512c_512", "conv2d_direct", 512, 512, 512, 512, 3, 32, true, false},
        {"decode_512c_to_256_512", "conv2d_direct", 512, 512, 512, 256, 3, 32, true, false},
        {"decode_stage4_256c_1024", "conv2d_direct", 1024, 1024, 256, 256, 3, 32, true, false},
        {"decode_256c_to_128_1024", "conv2d_direct", 1024, 1024, 256, 128, 3, 32, true, false},
        {"decode_stage5_128c_1024", "conv2d_direct", 1024, 1024, 128, 128, 3, 32, true, false},
    };
}

static std::vector<BenchCase> make_scaled_conv_correctness_cases() {
    constexpr float sdxl_vae_scale = 1.0f / 32.0f;
    return {
        {"scaled_decode_stage1_4c_to_512_128", "conv2d_direct", 128, 128, 4, 512, 3, 32, true, false, true, false, false, sdxl_vae_scale},
        {"scaled_decode_1x1_512c_128", "conv2d_direct", 128, 128, 512, 512, 1, 32, true, false, true, false, false, sdxl_vae_scale},
        {"scaled_decode_mid_512c_256", "conv2d_direct", 256, 256, 512, 512, 3, 32, true, false, true, false, false, sdxl_vae_scale},
        {"scaled_decode_512c_to_256_512", "conv2d_direct", 512, 512, 512, 256, 3, 32, true, false, true, false, false, sdxl_vae_scale},
        {"scaled_decode_256c_to_128_1024", "conv2d_direct", 1024, 1024, 256, 128, 3, 32, true, false, true, false, false, sdxl_vae_scale},
        {"scaled_decode_stage5_128c_1024", "conv2d_direct", 1024, 1024, 128, 128, 3, 32, true, false, true, false, false, sdxl_vae_scale},
    };
}

static void write_correctness_json(const std::string& path, const std::vector<CorrectnessResult>& results, const Args& args, const char* backend_name) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"backend\": \"" << json_escape(backend_name ? backend_name : "unknown") << "\",\n";
    out << "  \"warmup\": " << args.warmup << ",\n";
    out << "  \"iterations\": " << args.iterations << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"case\": \"" << json_escape(r.bench.name) << "\",\n";
        out << "      \"status\": \"" << json_escape(r.status) << "\",\n";
        out << "      \"error\": \"" << json_escape(r.error) << "\",\n";
        out << "      \"shape_whcn\": [" << r.bench.w << ", " << r.bench.h << ", " << r.bench.c << ", 1],\n";
        out << "      \"out_channels\": " << r.bench.out_c << ",\n";
        out << "      \"kernel\": " << r.bench.k << ",\n";
        out << "      \"direct_median_ms\": " << r.direct.median_ms << ",\n";
        out << "      \"implicit_median_ms\": " << r.implicit.median_ms << ",\n";
        out << "      \"direct_buffer_mb\": " << (r.direct.buffer_bytes / 1024.0 / 1024.0) << ",\n";
        out << "      \"implicit_buffer_mb\": " << (r.implicit.buffer_bytes / 1024.0 / 1024.0) << ",\n";
        out << "      \"mean_abs_diff\": " << r.mean_abs_diff << ",\n";
        out << "      \"p95_abs_diff\": " << r.p95_abs_diff << ",\n";
        out << "      \"p99_abs_diff\": " << r.p99_abs_diff << ",\n";
        out << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n";
        out << "      \"direct_nan_inf\": " << r.direct_nan_inf << ",\n";
        out << "      \"implicit_nan_inf\": " << r.implicit_nan_inf << ",\n";
        out << "      \"direct_min\": " << r.direct_min << ",\n";
        out << "      \"direct_max\": " << r.direct_max << ",\n";
        out << "      \"direct_mean\": " << r.direct_mean << ",\n";
        out << "      \"implicit_min\": " << r.implicit_min << ",\n";
        out << "      \"implicit_max\": " << r.implicit_max << ",\n";
        out << "      \"implicit_mean\": " << r.implicit_mean << ",\n";
        out << "      \"direct_hash\": \"" << std::hex << r.direct_hash << std::dec << "\",\n";
        out << "      \"implicit_hash\": \"" << std::hex << r.implicit_hash << std::dec << "\"\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

static void write_correctness_md(const std::string& path, const std::vector<CorrectnessResult>& results, const Args& args, const char* backend_name) {
    std::ofstream out(path);
    out << "# Implicit Conv Correctness\n\n";
    out << "Backend: `" << (backend_name ? backend_name : "unknown") << "`\n\n";
    out << "Warmup: `" << args.warmup << "`, iterations: `" << args.iterations << "`, memory cap: `" << args.memory_cap_mb << " MB`\n\n";
    out << "| Case | Shape | Direct | Implicit | Mean abs | p99 abs | Max abs | NaN/Inf | Buffers |\n";
    out << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& r : results) {
        out << "| `" << r.bench.name << "` | `[" << r.bench.w << "," << r.bench.h << "," << r.bench.c << "]->" << r.bench.out_c << " k" << r.bench.k << "` | "
            << std::fixed << std::setprecision(3) << r.direct.median_ms << " ms | "
            << r.implicit.median_ms << " ms | "
            << std::scientific << std::setprecision(3) << r.mean_abs_diff << " | "
            << r.p99_abs_diff << " | "
            << r.max_abs_diff << " | "
            << std::dec << (r.direct_nan_inf + r.implicit_nan_inf) << " | "
            << std::fixed << std::setprecision(1) << (r.direct.buffer_bytes / 1024.0 / 1024.0)
            << "/" << (r.implicit.buffer_bytes / 1024.0 / 1024.0) << " MB |\n";
        if (!r.error.empty()) {
            out << "\nError for `" << r.bench.name << "`: `" << r.error << "`\n\n";
        }
    }
}

static void write_json(const std::string& path, const std::vector<BenchResult>& results, const Args& args, const char* backend_name) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"backend\": \"" << json_escape(backend_name ? backend_name : "unknown") << "\",\n";
    out << "  \"warmup\": " << args.warmup << ",\n";
    out << "  \"iterations\": " << args.iterations << ",\n";
    out << "  \"memory_cap_mb\": " << args.memory_cap_mb << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"case\": \"" << json_escape(r.bench.name) << "\",\n";
        out << "      \"op\": \"" << json_escape(r.bench.op) << "\",\n";
        out << "      \"status\": \"" << json_escape(r.status) << "\",\n";
        out << "      \"error\": \"" << json_escape(r.error) << "\",\n";
        out << "      \"shape_whcn\": [" << r.bench.w << ", " << r.bench.h << ", " << r.bench.c << ", 1],\n";
        out << "      \"out_channels\": " << r.bench.out_c << ",\n";
        out << "      \"kernel\": " << r.bench.k << ",\n";
        out << "      \"buffer_mb\": " << (r.buffer_bytes / 1024.0 / 1024.0) << ",\n";
        out << "      \"mean_ms\": " << r.mean_ms << ",\n";
        out << "      \"median_ms\": " << r.median_ms << ",\n";
        out << "      \"min_ms\": " << r.min_ms << ",\n";
        out << "      \"p95_ms\": " << r.p95_ms << ",\n";
        out << "      \"max_ms\": " << r.max_ms << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

static void write_md(const std::string& path, const std::vector<BenchResult>& results, const Args& args, const char* backend_name) {
    std::ofstream out(path);
    out << "# SDCPP VAE Op Bench\n\n";
    out << "Backend: `" << (backend_name ? backend_name : "unknown") << "`\n\n";
    out << "Warmup: `" << args.warmup << "`, iterations: `" << args.iterations << "`, memory cap: `" << args.memory_cap_mb << " MB`\n\n";
    out << "| Case | Op | Shape WHCN | Status | Median | Mean | Min | p95 | Buffer |\n";
    out << "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n";
    for (const auto& r : results) {
        out << "| `" << r.bench.name << "` | `" << r.bench.op << "` | `["
            << r.bench.w << "," << r.bench.h << "," << r.bench.c << ",1]` | `" << r.status << "` | "
            << std::fixed << std::setprecision(3) << r.median_ms << " ms | "
            << r.mean_ms << " ms | " << r.min_ms << " ms | " << r.p95_ms << " ms | "
            << (r.buffer_bytes / 1024.0 / 1024.0) << " MB |\n";
        if (!r.error.empty()) {
            out << "\nError for `" << r.bench.name << "`: `" << r.error << "`\n\n";
        }
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        backend = ggml_backend_init_best();
    }
    if (backend == nullptr) {
        std::cerr << "failed to initialize backend\n";
        return 1;
    }

    if (args.correctness) {
        std::vector<CorrectnessResult> correctness_results;
        for (const auto& bench : make_conv_correctness_cases()) {
            std::cout << "checking " << bench.name << "\n";
            CorrectnessResult result = compare_conv_case(backend, bench, args);
            std::cout << "  status=" << result.status
                      << " direct_ms=" << result.direct.median_ms
                      << " implicit_ms=" << result.implicit.median_ms
                      << " mean_abs=" << result.mean_abs_diff
                      << " p99_abs=" << result.p99_abs_diff
                      << " max_abs=" << result.max_abs_diff
                      << " nan_inf=" << (result.direct_nan_inf + result.implicit_nan_inf)
                      << "\n";
            if (!result.error.empty()) {
                std::cout << "  error=" << result.error << "\n";
            }
            correctness_results.push_back(std::move(result));
        }
        for (const auto& bench : make_scaled_conv_correctness_cases()) {
            std::cout << "checking " << bench.name << "\n";
            CorrectnessResult result = compare_scaled_conv_case(backend, bench, args);
            std::cout << "  status=" << result.status
                      << " ref_ms=" << result.direct.median_ms
                      << " fused_ms=" << result.implicit.median_ms
                      << " mean_abs=" << result.mean_abs_diff
                      << " p99_abs=" << result.p99_abs_diff
                      << " max_abs=" << result.max_abs_diff
                      << " nan_inf=" << (result.direct_nan_inf + result.implicit_nan_inf)
                      << "\n";
            if (!result.error.empty()) {
                std::cout << "  error=" << result.error << "\n";
            }
            correctness_results.push_back(std::move(result));
        }
        const char* backend_name = ggml_backend_name(backend);
        write_correctness_json(args.out_json, correctness_results, args, backend_name);
        write_correctness_md(args.out_md, correctness_results, args, backend_name);
        ggml_backend_free(backend);
        return 0;
    }

    std::vector<BenchResult> results;
    for (const auto& bench : make_cases(args.include_legacy)) {
        std::cout << "running " << bench.name << "\n";
        BenchResult result = run_case(backend, bench, args);
        std::cout << "  status=" << result.status << " median_ms=" << result.median_ms
                  << " buffer_mb=" << (result.buffer_bytes / 1024.0 / 1024.0) << "\n";
        if (!result.error.empty()) {
            std::cout << "  error=" << result.error << "\n";
        }
        results.push_back(std::move(result));
    }

    const char* backend_name = ggml_backend_name(backend);
    write_json(args.out_json, results, args, backend_name);
    write_md(args.out_md, results, args, backend_name);
    ggml_backend_free(backend);
    return 0;
}
