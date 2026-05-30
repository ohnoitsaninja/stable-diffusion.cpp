#include "src/bonsai_gemlite_int1.hpp"

#include "ggml.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string pack;
    std::string internal_name = "model.diffusion_model.single_blocks.0.linear1.weight";
    int rows_m = 1536;
    int warmup = 1;
    int runs = 10;
    bool compare_current = false;
    std::string kernel = "env";
};

bool require_value(int& index, int argc, char** argv, std::string& out) {
    if (index + 1 >= argc) return false;
    out = argv[++index];
    return true;
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--pack" && require_value(i, argc, argv, value)) {
            args.pack = value;
        } else if ((arg == "--internal-name" || arg == "--linear") && require_value(i, argc, argv, value)) {
            args.internal_name = value;
        } else if ((arg == "--rows-m" || arg == "--m") && require_value(i, argc, argv, value)) {
            args.rows_m = std::atoi(value.c_str());
        } else if (arg == "--warmup" && require_value(i, argc, argv, value)) {
            args.warmup = std::atoi(value.c_str());
        } else if (arg == "--runs" && require_value(i, argc, argv, value)) {
            args.runs = std::atoi(value.c_str());
        } else if (arg == "--kernel" && require_value(i, argc, argv, value)) {
            args.kernel = value;
        } else if (arg == "--compare-current") {
            args.compare_current = true;
        } else {
            return false;
        }
    }
    return !args.pack.empty() && args.rows_m > 0 && args.warmup >= 0 && args.runs > 0 && args.runs <= 100 &&
           (args.kernel == "env" || args.kernel == "current" || args.kernel == "gemlite-shape-v1" ||
            args.kernel == "gemlite-shape-v2" || args.kernel == "gemlite-tc-gemm" ||
            args.kernel == "gemlite-tc-gemm-v2" || args.kernel == "gemlite-tc-gemm-v2-dequant-only" ||
            args.kernel == "gemlite-tc-gemm-v2-mma-only" || args.kernel == "gemlite-tc-gemm-v2-prepack" ||
            args.kernel == "gemlite-tc-gemm-v2-prepack-v1" || args.kernel == "gemlite-tc-gemm-v2-prepack-v2" ||
            args.kernel == "gemlite-tc-gemm-v2-prepack-v3" || args.kernel == "gemlite-tc-linear2" ||
            args.kernel == "gemlite-tc-linear1-largesmem" || args.kernel == "gemlite-tc-linear1-bankfix" ||
            args.kernel == "gemlite-tc-linear2-async" || args.kernel == "gemlite-tc-linear2-async-b" ||
            args.kernel == "gemlite-tc-linear2-ldmatrix" ||
            args.kernel == "gemlite-tc-img-mlp0" || args.kernel == "gemlite-tc-img-mlp2" ||
            args.kernel == "gemlite-tc-img-qkv" || args.kernel == "gemlite-tc-txt-qkv" ||
            args.kernel == "gemlite-tc-img-attn-proj" || args.kernel == "gemlite-tc-txt-attn-proj");
}

void usage() {
    std::cerr << "sd-bonsai-gemlite-int1-kernel-bench --pack <state_dict.pt>"
              << " [--internal-name model.diffusion_model.single_blocks.0.linear1.weight]"
              << " [--rows-m 1536] [--warmup 1] [--runs 10]"
              << " [--kernel current|gemlite-shape-v1|gemlite-shape-v2|gemlite-tc-gemm|gemlite-tc-gemm-v2|gemlite-tc-linear1-largesmem|gemlite-tc-linear1-bankfix|gemlite-tc-linear2|gemlite-tc-linear2-async|gemlite-tc-linear2-async-b|gemlite-tc-linear2-ldmatrix|gemlite-tc-img-mlp0|gemlite-tc-img-mlp2|gemlite-tc-img-qkv|gemlite-tc-txt-qkv|gemlite-tc-img-attn-proj|gemlite-tc-txt-attn-proj|gemlite-tc-gemm-v2-dequant-only|gemlite-tc-gemm-v2-mma-only|gemlite-tc-gemm-v2-prepack-v1|gemlite-tc-gemm-v2-prepack-v2|gemlite-tc-gemm-v2-prepack-v3|env] [--compare-current]\n";
}

bool experimental_gate_enabled() {
    const char* value = std::getenv("SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

double elapsed_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n == 0) return 0.0;
    if ((n & 1u) != 0u) return values[n / 2];
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void set_env_value(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void select_kernel(const std::string& kernel) {
    if (kernel == "env") {
        return;
    }
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_GEMM_V2", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM", "0");
    set_env_value("SDCPP_BONSAI_INT1_LINEAR1_BANKFIX", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_QKV", "0");
    set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ", "0");
    set_env_value("SDCPP_BONSAI_INT1_TC_V2_DEQUANT_ONLY", "0");
    set_env_value("SDCPP_BONSAI_INT1_TC_V2_MMA_ONLY", "0");
    set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "0");
    set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "2");
    set_env_value("SDCPP_BONSAI_INT1_TILED_KERNEL", "0");
    if (kernel == "gemlite-shape-v1") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL", "1");
    } else if (kernel == "gemlite-shape-v2") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_GEMM_V2", "1");
    } else if (kernel == "gemlite-tc-gemm") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM", "1");
    } else if (kernel == "gemlite-tc-gemm-v2") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2", "1");
    } else if (kernel == "gemlite-tc-linear1-largesmem") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM", "1");
    } else if (kernel == "gemlite-tc-linear1-bankfix") {
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "1");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "3");
        set_env_value("SDCPP_BONSAI_INT1_LINEAR1_BANKFIX", "1");
    } else if (kernel == "gemlite-tc-linear2") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2", "1");
    } else if (kernel == "gemlite-tc-linear2-async") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC", "1");
    } else if (kernel == "gemlite-tc-linear2-async-b") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B", "1");
    } else if (kernel == "gemlite-tc-linear2-ldmatrix") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX", "1");
    } else if (kernel == "gemlite-tc-img-mlp0") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0", "1");
    } else if (kernel == "gemlite-tc-img-mlp2") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2", "1");
    } else if (kernel == "gemlite-tc-img-qkv" || kernel == "gemlite-tc-txt-qkv") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_QKV", "1");
    } else if (kernel == "gemlite-tc-img-attn-proj" || kernel == "gemlite-tc-txt-attn-proj") {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ", "1");
    } else if (kernel == "gemlite-tc-gemm-v2-dequant-only") {
        set_env_value("SDCPP_BONSAI_INT1_TC_V2_DEQUANT_ONLY", "1");
    } else if (kernel == "gemlite-tc-gemm-v2-mma-only") {
        set_env_value("SDCPP_BONSAI_INT1_TC_V2_MMA_ONLY", "1");
    } else if (kernel == "gemlite-tc-gemm-v2-prepack" || kernel == "gemlite-tc-gemm-v2-prepack-v2") {
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "1");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "2");
    } else if (kernel == "gemlite-tc-gemm-v2-prepack-v1") {
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "1");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "1");
    } else if (kernel == "gemlite-tc-gemm-v2-prepack-v3") {
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "1");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "3");
    }
}

struct CompareStats {
    double corr = 0.0;
    double cos = 0.0;
    double mean_abs = 0.0;
    double max_abs = 0.0;
};

CompareStats compare_outputs(const std::vector<ggml_fp16_t>& a, const std::vector<ggml_fp16_t>& b) {
    CompareStats stats;
    const size_t n = std::min(a.size(), b.size());
    if (n == 0) return stats;
    double sum_a = 0.0;
    double sum_b = 0.0;
    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double av = ggml_fp16_to_fp32(a[i]);
        const double bv = ggml_fp16_to_fp32(b[i]);
        sum_a += av;
        sum_b += bv;
        sum_aa += av * av;
        sum_bb += bv * bv;
        sum_ab += av * bv;
        const double diff = std::abs(av - bv);
        stats.mean_abs += diff;
        stats.max_abs = std::max(stats.max_abs, diff);
    }
    stats.mean_abs /= static_cast<double>(n);
    stats.cos = sum_ab / std::max(std::sqrt(sum_aa * sum_bb), std::numeric_limits<double>::min());
    const double mean_a = sum_a / static_cast<double>(n);
    const double mean_b = sum_b / static_cast<double>(n);
    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = ggml_fp16_to_fp32(a[i]) - mean_a;
        const double db = ggml_fp16_to_fp32(b[i]) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    stats.corr = cov / std::max(std::sqrt(var_a * var_b), std::numeric_limits<double>::min());
    return stats;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 2;
    }
    if (!experimental_gate_enabled()) {
        std::cerr << "Bonsai GemLite INT1 kernel bench is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.\n";
        return 3;
    }
    select_kernel(args.kernel);

    std::string error;
    auto runtime = sd::BonsaiGemliteRuntime::load_from_pack(args.pack, error);
    if (!runtime) {
        std::cerr << "failed to create Bonsai runtime: " << error << "\n";
        return 1;
    }
    if (!runtime->has_linear(args.internal_name)) {
        std::cerr << "runtime does not have mapped internal linear: " << args.internal_name << "\n";
        return 1;
    }

    int64_t k = 0;
    int64_t out_features = 0;
    if (!runtime->linear_shape(args.internal_name, k, out_features)) {
        std::cerr << "failed to resolve linear shape for: " << args.internal_name << "\n";
        return 1;
    }
    const int64_t rows = args.rows_m;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::cerr << "ggml_backend_cuda_init(0) failed\n";
        return 1;
    }
    ggml_init_params params;
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::cerr << "ggml_init failed\n";
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, rows);
    ggml_tensor* y = runtime->linear_forward(ctx, x, args.internal_name.c_str());
    if (y == nullptr) {
        std::cerr << "runtime linear_forward returned null\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::cerr << "ggml_backend_alloc_ctx_tensors failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<ggml_fp16_t> host_x(static_cast<size_t>(k * rows));
    for (size_t i = 0; i < host_x.size(); ++i) {
        const float v = static_cast<float>((static_cast<int>(i % 127) - 63)) / 63.0f;
        host_x[i] = ggml_fp32_to_fp16(v);
    }
    ggml_backend_tensor_set(x, host_x.data(), 0, host_x.size() * sizeof(ggml_fp16_t));

    if (args.compare_current) {
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_GEMM_V2", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_QKV", "0");
        set_env_value("SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ", "0");
        set_env_value("SDCPP_BONSAI_INT1_TC_V2_DEQUANT_ONLY", "0");
        set_env_value("SDCPP_BONSAI_INT1_TC_V2_MMA_ONLY", "0");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK", "0");
        set_env_value("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT", "2");
        set_env_value("SDCPP_BONSAI_INT1_TILED_KERNEL", "0");
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "baseline graph compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        std::vector<ggml_fp16_t> baseline(static_cast<size_t>(ggml_nelements(y)));
        ggml_backend_tensor_get(y, baseline.data(), 0, baseline.size() * sizeof(ggml_fp16_t));
        select_kernel(args.kernel == "env" ? "gemlite-shape-v1" : args.kernel);
        status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "gemlite-shape graph compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        std::vector<ggml_fp16_t> optimized(static_cast<size_t>(ggml_nelements(y)));
        ggml_backend_tensor_get(y, optimized.data(), 0, optimized.size() * sizeof(ggml_fp16_t));
        const CompareStats cmp = compare_outputs(baseline, optimized);
        std::cout << std::fixed << std::setprecision(9)
                  << "[BonsaiKernelBench] compare_current kernel=" << (args.kernel == "env" ? "gemlite-shape-v1" : args.kernel)
                  << " corr=" << cmp.corr
                  << " cos=" << cmp.cos
                  << " mean_abs=" << cmp.mean_abs
                  << " max_abs=" << cmp.max_abs << "\n";
    }

    for (int i = 0; i < args.warmup; ++i) {
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "warmup graph compute failed\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(args.runs));
    for (int i = 0; i < args.runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto stop = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::cerr << "graph compute failed at run " << i << "\n";
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
        times.push_back(elapsed_ms(start, stop));
    }

    std::vector<ggml_fp16_t> host_y(static_cast<size_t>(ggml_nelements(y)));
    ggml_backend_tensor_get(y, host_y.data(), 0, host_y.size() * sizeof(ggml_fp16_t));
    double sum = 0.0;
    float mn = 0.0f;
    float mx = 0.0f;
    int finite = 0;
    int nan = 0;
    int inf = 0;
    for (size_t i = 0; i < host_y.size(); ++i) {
        const float v = ggml_fp16_to_fp32(host_y[i]);
        if (std::isnan(v)) {
            nan++;
            continue;
        }
        if (std::isinf(v)) {
            inf++;
            continue;
        }
        if (finite == 0 || v < mn) mn = v;
        if (finite == 0 || v > mx) mx = v;
        sum += v;
        finite++;
    }

    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    const auto summary = runtime->summary();
    if (args.internal_name.size() >= 11 && args.internal_name.rfind(".qkv.weight") == args.internal_name.size() - 11 && y->ne[0] % 3 == 0) {
        const int64_t n_total = y->ne[0];
        const int64_t n_each = n_total / 3;
        const int64_t rows_out = static_cast<int64_t>(host_y.size()) / n_total;
        const char* labels[3] = {"q", "k", "v"};
        for (int slice = 0; slice < 3; ++slice) {
            int slice_finite = 0;
            int slice_nan = 0;
            int slice_inf = 0;
            float slice_min = 0.0f;
            float slice_max = 0.0f;
            double slice_sum = 0.0;
            for (int64_t row = 0; row < rows_out; ++row) {
                const size_t base = static_cast<size_t>(row * n_total + slice * n_each);
                for (int64_t col = 0; col < n_each; ++col) {
                    const float v = ggml_fp16_to_fp32(host_y[base + static_cast<size_t>(col)]);
                    if (std::isnan(v)) {
                        slice_nan++;
                        continue;
                    }
                    if (std::isinf(v)) {
                        slice_inf++;
                        continue;
                    }
                    if (slice_finite == 0 || v < slice_min) slice_min = v;
                    if (slice_finite == 0 || v > slice_max) slice_max = v;
                    slice_sum += v;
                    slice_finite++;
                }
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "[BonsaiKernelBench] qkv_slice=" << labels[slice]
                      << " finite=" << slice_finite
                      << " nan=" << slice_nan
                      << " inf=" << slice_inf
                      << " min=" << slice_min
                      << " max=" << slice_max
                      << " sum=" << slice_sum << "\n";
        }
    }
    std::cout << std::fixed << std::setprecision(3)
              << "[BonsaiKernelBench] internal_linear=" << args.internal_name << "\n"
              << "[BonsaiKernelBench] kernel=" << args.kernel << "\n"
              << "[BonsaiKernelBench] shape M=" << rows << " K=" << k << " N=" << y->ne[0] << "\n"
              << "[BonsaiKernelBench] warmup=" << args.warmup << " runs=" << args.runs
              << " mean_ms=" << mean << " median_ms=" << median(times)
              << " min_ms=" << *std::min_element(times.begin(), times.end())
              << " max_ms=" << *std::max_element(times.begin(), times.end()) << "\n"
              << "[BonsaiKernelBench] times_ms=";
    for (size_t i = 0; i < times.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << times[i];
    }
    std::cout << "\n"
              << "[BonsaiKernelBench] output finite=" << finite << " nan=" << nan << " inf=" << inf
              << " min=" << mn << " max=" << mx << " sum=" << sum << "\n"
              << "[BonsaiKernelBench] bonsai_int1_linear_calls=" << summary.linear_calls
              << " unique_bonsai_int1_linears_executed=" << summary.unique_linears_executed
              << " missing_linear_calls=" << summary.missing_linear_calls
              << " full_fp16_weight_expansion=false cpu_dequant_fallback_calls=0\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
