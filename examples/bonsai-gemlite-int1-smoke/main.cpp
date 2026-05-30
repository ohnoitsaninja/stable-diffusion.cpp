#include "src/bonsai_gemlite_int1.hpp"
#include "src/model.h"

#include "ggml.h"
#include "ggml-cuda.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

namespace {

struct Args {
    std::string pack;
    std::string linear = "transformer_blocks.0.attn.to_q";
    int64_t rows_m = 16;
    bool summary_only = false;
    bool dump_names = false;
    bool graph_runtime = false;
    std::string internal_name = "model.diffusion_model.single_blocks.0.linear1.weight";
};

bool require_value(int& index, int argc, char** argv, std::string& out) {
    if (index + 1 >= argc) {
        return false;
    }
    out = argv[++index];
    return true;
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pack") {
            if (!require_value(i, argc, argv, args.pack)) return false;
        } else if (arg == "--linear") {
            if (!require_value(i, argc, argv, args.linear)) return false;
        } else if (arg == "--rows-m") {
            std::string value;
            if (!require_value(i, argc, argv, value)) return false;
            args.rows_m = std::strtoll(value.c_str(), nullptr, 10);
        } else if (arg == "--summary-only") {
            args.summary_only = true;
        } else if (arg == "--dump-names") {
            args.dump_names = true;
        } else if (arg == "--graph-runtime") {
            args.graph_runtime = true;
        } else if (arg == "--internal-name") {
            if (!require_value(i, argc, argv, args.internal_name)) return false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return !args.pack.empty();
}

void usage() {
    std::cerr
        << "sd-bonsai-gemlite-int1-smoke --pack <transformer-gemlite-int1/state_dict.pt>\n"
        << "  [--linear transformer_blocks.0.attn.to_q] [--rows-m 16] [--summary-only]\n"
        << "  [--dump-names]\n"
        << "  [--graph-runtime] [--internal-name model.diffusion_model.single_blocks.0.linear1.weight]\n"
        << "\n"
        << "Requires SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.\n";
}

bool experimental_gate_enabled() {
    const char* value = std::getenv("SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1");
    return value != nullptr && std::string(value) == "1";
}

bool run_graph_runtime_smoke(const Args& args) {
    std::string error;
    auto runtime = sd::BonsaiGemliteRuntime::load_from_pack(args.pack, error);
    if (!runtime) {
        std::cerr << "failed to create Bonsai runtime: " << error << "\n";
        return false;
    }
    if (!runtime->has_linear(args.internal_name)) {
        std::cerr << "runtime does not have mapped internal linear: " << args.internal_name << "\n";
        return false;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::cerr << "ggml_backend_cuda_init(0) failed\n";
        return false;
    }
    const int64_t k = 3072;
    const int64_t rows = args.rows_m;
    ggml_init_params params;
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::cerr << "ggml_init failed\n";
        ggml_backend_free(backend);
        return false;
    }
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, rows);
    ggml_tensor* y = runtime->linear_forward(ctx, x, args.internal_name.c_str());
    if (y == nullptr) {
        std::cerr << "runtime linear_forward returned null\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::cerr << "ggml_backend_alloc_ctx_tensors failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    std::vector<ggml_fp16_t> host_x(static_cast<size_t>(k * rows));
    for (size_t i = 0; i < host_x.size(); ++i) {
        const float v = static_cast<float>((static_cast<int>(i % 31) - 15)) / 31.0f;
        host_x[i] = ggml_fp32_to_fp16(v);
    }
    ggml_backend_tensor_set(x, host_x.data(), 0, host_x.size() * sizeof(ggml_fp16_t));
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "ggml graph compute failed\n";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    std::vector<ggml_fp16_t> host_y(static_cast<size_t>(ggml_nelements(y)));
    ggml_backend_tensor_get(y, host_y.data(), 0, host_y.size() * sizeof(ggml_fp16_t));
    double sum = 0.0;
    float mn = 0.0f;
    float mx = 0.0f;
    for (size_t i = 0; i < host_y.size(); ++i) {
        const float v = ggml_fp16_to_fp32(host_y[i]);
        if (i == 0 || v < mn) mn = v;
        if (i == 0 || v > mx) mx = v;
        sum += v;
    }
    const auto summary = runtime->summary();
    std::cout << "graph_runtime=ok\n"
              << "internal_linear=" << args.internal_name << "\n"
              << "output_elements=" << host_y.size() << "\n"
              << "output_min=" << mn << "\n"
              << "output_max=" << mx << "\n"
              << "output_sum=" << sum << "\n"
              << "bonsai_int1_linear_calls=" << summary.linear_calls << "\n"
              << "unique_bonsai_int1_linears_executed=" << summary.unique_linears_executed << "\n"
              << "full_fp16_weight_expansion=false\n";
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 2;
    }
    if (!experimental_gate_enabled()) {
        std::cerr << "Bonsai GemLite INT1 spike is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.\n";
        return 3;
    }
    if (args.graph_runtime) {
        return run_graph_runtime_smoke(args) ? 0 : 1;
    }

    std::string error;
    sd::BonsaiGemlitePackSummary summary;
    if (!sd::bonsai_gemlite_int1_summarize_pack(args.pack, summary, error)) {
        std::cerr << "failed to summarize Bonsai pack: " << error << "\n";
        return 1;
    }
    sd::bonsai_gemlite_int1_print_summary(std::cout, summary);
    if (args.dump_names) {
        ModelLoader loader;
        if (!loader.init_from_file(args.pack)) {
            std::cerr << "failed to index pack for name dump\n";
            return 1;
        }
        int count = 0;
        for (const auto& name : loader.get_tensor_names()) {
            std::cout << "tensor_name[" << count++ << "]=" << name << "\n";
            if (count >= 80) {
                break;
            }
        }
    }
    if (args.summary_only) {
        return 0;
    }

    sd::BonsaiGemliteLinear linear;
    if (!sd::bonsai_gemlite_int1_load_linear(args.pack, args.linear, linear, error)) {
        std::cerr << "failed to load Bonsai linear: " << error << "\n";
        return 1;
    }
    std::cout << "linear=" << linear.name << "\n"
              << "orig_shape_out_in=" << linear.out_features << "x" << linear.in_features << "\n"
              << "packed_wq_bytes=" << linear.wq.size() << "\n"
              << "scale_count=" << linear.scales.size() << "\n"
              << "zero_count=" << linear.zeros.size() << "\n";

    sd::BonsaiGemliteCudaProbeResult result = sd::bonsai_gemlite_int1_cuda_probe(linear, args.rows_m);
    if (!result.ok) {
        std::cerr << "CUDA Bonsai INT1 linear probe failed: " << result.error << "\n";
        return 1;
    }
    std::cout << "cuda_probe=ok\n"
              << "rows_m=" << args.rows_m << "\n"
              << "kernel_ms=" << result.elapsed_ms << "\n"
              << "device_bytes=" << result.device_bytes << "\n"
              << "output_min=" << result.output_min << "\n"
              << "output_max=" << result.output_max << "\n"
              << "output_sum=" << result.output_sum << "\n"
              << "full_fp16_weight_expansion=false\n";
    return 0;
}
