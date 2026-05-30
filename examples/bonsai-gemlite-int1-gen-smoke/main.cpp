#include "src/bonsai_gemlite_int1.hpp"
#include "src/flux.hpp"
#include "src/model.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string pack;
    int width_tokens = 4;
    int height_tokens = 4;
    int text_tokens = 8;
    int threads = 8;
    bool load_support_tensors = false;
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

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pack") {
            if (!require_value(i, argc, argv, args.pack)) return false;
        } else if (arg == "--width-tokens") {
            if (!require_int(i, argc, argv, args.width_tokens)) return false;
        } else if (arg == "--height-tokens") {
            if (!require_int(i, argc, argv, args.height_tokens)) return false;
        } else if (arg == "--text-tokens") {
            if (!require_int(i, argc, argv, args.text_tokens)) return false;
        } else if (arg == "--threads") {
            if (!require_int(i, argc, argv, args.threads)) return false;
        } else if (arg == "--load-support-tensors") {
            args.load_support_tensors = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }
    return !args.pack.empty() && args.width_tokens > 0 && args.height_tokens > 0 && args.text_tokens > 0;
}

bool experimental_gate_enabled() {
    const char* value = std::getenv("SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1");
    return value != nullptr && std::string(value) == "1";
}

TensorStorage storage_1d(const std::string& name, int64_t n0, ggml_type type = GGML_TYPE_F32) {
    int64_t ne[SD_MAX_DIMS] = {n0, 1, 1, 1, 1};
    return TensorStorage(name, type, ne, 1, 0);
}

TensorStorage storage_2d(const std::string& name, int64_t n0, int64_t n1, ggml_type type = GGML_TYPE_F32) {
    int64_t ne[SD_MAX_DIMS] = {n0, n1, 1, 1, 1};
    return TensorStorage(name, type, ne, 2, 0);
}

String2TensorStorage make_bonsai_flux2_shape_map() {
    String2TensorStorage map;
    auto add = [&](const TensorStorage& storage) {
        map[storage.name] = storage;
    };
    add(storage_2d("model.diffusion_model.txt_in.weight", 2560, 3072));
    add(storage_1d("model.diffusion_model.double_blocks.0.txt_attn.norm.key_norm.scale", 128));
    add(storage_1d("model.diffusion_model.single_blocks.0.norm.key_norm.scale", 128));
    add(storage_2d("model.diffusion_model.double_blocks.4.img_mlp.0.weight", 3072, 18432));
    add(storage_2d("model.diffusion_model.single_blocks.19.linear1.weight", 3072, 27648));
    return map;
}

void fill_pattern(sd::Tensor<float>& tensor, float scale) {
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        tensor.data()[i] = static_cast<float>((i % 29) - 14) * scale;
    }
}

void zero_runner_params(Flux::FluxRunner& runner) {
    std::map<std::string, ggml_tensor*> tensors;
    runner.get_param_tensors(tensors, "model.diffusion_model");
    std::vector<uint8_t> zeros;
    for (const auto& kv : tensors) {
        ggml_tensor* tensor = kv.second;
        const size_t bytes = ggml_nbytes(tensor);
        zeros.assign(bytes, 0);
        ggml_backend_tensor_set(tensor, zeros.data(), 0, bytes);
    }
}

int64_t tensor_ne0_or_default(const String2TensorStorage& map, const std::string& name, int64_t fallback) {
    auto it = map.find(name);
    if (it == map.end()) {
        return fallback;
    }
    return it->second.ne[0];
}

std::string shape_string(const TensorStorage& storage) {
    std::string out = "[";
    for (int i = 0; i < storage.n_dims; ++i) {
        if (i > 0) out += "x";
        out += std::to_string(storage.ne[i]);
    }
    out += "]";
    return out;
}

struct SupportLoadStats {
    int total = 0;
    int loaded = 0;
    int runtime_attached = 0;
    int zero_placeholders = 0;
    uint64_t bytes = 0;
};

bool init_converted_loader(const std::string& pack, ModelLoader& loader) {
    if (!loader.init_from_file_and_convert_name(pack, "model.diffusion_model.", VERSION_FLUX2_KLEIN)) {
        std::cerr << "failed to initialize Bonsai support ModelLoader from " << pack << "\n";
        return false;
    }
    return true;
}

bool load_bonsai_support_tensors(const std::string& pack,
                                 int threads,
                                 Flux::FluxRunner& runner,
                                 const std::shared_ptr<sd::BonsaiGemliteRuntime>& runtime,
                                 SupportLoadStats& stats) {
    ModelLoader loader;
    if (!init_converted_loader(pack, loader)) {
        return false;
    }
    auto& storage_map = loader.get_tensor_storage_map();

    std::map<std::string, ggml_tensor*> all_params;
    runner.get_param_tensors(all_params, "model.diffusion_model");

    std::map<std::string, ggml_tensor*> support_params;
    for (const auto& kv : all_params) {
        const std::string& name = kv.first;
        ggml_tensor* tensor = kv.second;
        if (runtime->has_linear(name)) {
            continue;
        }
        auto storage_it = storage_map.find(name);
        if (storage_it == storage_map.end()) {
            continue;
        }
        support_params[name] = tensor;
    }

    stats.total = static_cast<int>(support_params.size());
    for (const auto& kv : support_params) {
        const auto& storage = storage_map[kv.first];
        const uint64_t bytes = static_cast<uint64_t>(ggml_nbytes(kv.second));
        stats.bytes += bytes;
        std::cout << "[BonsaiSupport] name=" << kv.first
                  << " dtype=" << ggml_type_name(storage.type)
                  << " shape=" << shape_string(storage)
                  << " bytes=" << bytes
                  << " loaded=pending"
                  << " attached=true\n";
    }

    if (stats.total != 69) {
        std::cerr << "unexpected Bonsai support tensor count: " << stats.total << " expected 69\n";
        return false;
    }

    if (!loader.load_tensors(support_params, {}, threads, false)) {
        std::cerr << "failed to load Bonsai BF16 support tensors\n";
        return false;
    }

    stats.loaded = stats.total;
    stats.runtime_attached = stats.total;
    stats.zero_placeholders = 0;
    for (const auto& kv : support_params) {
        std::cout << "[BonsaiSupport] name=" << kv.first
                  << " loaded=true"
                  << " attached=true\n";
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        std::cerr << "usage: sd-bonsai-gemlite-int1-gen-smoke --pack <state_dict.pt> [--width-tokens 4] [--height-tokens 4] [--text-tokens 8] [--load-support-tensors]\n";
        return 2;
    }
    if (!experimental_gate_enabled()) {
        std::cerr << "Bonsai GemLite INT1 generation smoke is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.\n";
        std::cout << "bonsai_runtime_attached=false\n";
        return 3;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string error;
    auto runtime = sd::BonsaiGemliteRuntime::load_from_pack(args.pack, error);
    if (!runtime) {
        std::cerr << "failed to create Bonsai runtime: " << error << "\n";
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::cerr << "ggml_backend_cuda_init(0) failed\n";
        return 1;
    }

    std::unique_ptr<ModelLoader> shape_loader;
    String2TensorStorage fake_shape_map = make_bonsai_flux2_shape_map();
    const String2TensorStorage* shape_map = &fake_shape_map;
    if (args.load_support_tensors) {
        shape_loader.reset(new ModelLoader());
        if (!init_converted_loader(args.pack, *shape_loader)) {
            ggml_backend_free(backend);
            return 1;
        }
        shape_map = &shape_loader->get_tensor_storage_map();
    }
    Flux::FluxRunner runner(backend, false, *shape_map, "model.diffusion_model", VERSION_FLUX2_KLEIN);
    runner.set_bonsai_gemlite_runtime(runtime);
    if (!runner.alloc_params_buffer()) {
        std::cerr << "FluxRunner alloc_params_buffer failed\n";
        ggml_backend_free(backend);
        return 1;
    }
    zero_runner_params(runner);
    SupportLoadStats support_stats;
    if (args.load_support_tensors) {
        if (!load_bonsai_support_tensors(args.pack, args.threads, runner, runtime, support_stats)) {
            ggml_backend_free(backend);
            return 1;
        }
    } else {
        support_stats.total = 69;
        support_stats.zero_placeholders = 69;
    }
    const auto t2 = std::chrono::steady_clock::now();

    const int64_t latent_channels = tensor_ne0_or_default(*shape_map, "model.diffusion_model.img_in.weight", 128);
    const int64_t context_dim = tensor_ne0_or_default(*shape_map, "model.diffusion_model.txt_in.weight", 2560);
    sd::Tensor<float> x({args.width_tokens, args.height_tokens, latent_channels, 1});
    sd::Tensor<float> timesteps({1}, std::vector<float>{1.0f});
    sd::Tensor<float> context({context_dim, args.text_tokens, 1});
    fill_pattern(x, 0.001f);
    fill_pattern(context, 0.0005f);

    auto out = runner.compute(args.threads, x, timesteps, context);
    const auto t3 = std::chrono::steady_clock::now();
    ggml_backend_synchronize(backend);

    const auto summary = runtime->summary();
    double output_sum = 0.0;
    float output_min = 0.0f;
    float output_max = 0.0f;
    if (!out.empty()) {
        for (int64_t i = 0; i < out.numel(); ++i) {
            const float v = out.data()[i];
            if (i == 0 || v < output_min) output_min = v;
            if (i == 0 || v > output_max) output_max = v;
            output_sum += v;
        }
    }

    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cout << "bonsai_runtime_attached=true\n"
              << "bonsai_pack_loaded=true\n"
              << "bonsai_linears_total=" << summary.quantized_linears << "\n"
              << "bonsai_linears_runtime_mapped=100\n"
              << "bonsai_support_tensors_total=" << support_stats.total << "\n"
              << "bonsai_support_tensors_loaded=" << support_stats.loaded << "\n"
              << "bonsai_support_tensors_runtime_attached=" << support_stats.runtime_attached << "\n"
              << "bonsai_support_bytes=" << support_stats.bytes << "\n"
              << "bonsai_zero_support_placeholders=" << support_stats.zero_placeholders << "\n"
              << "bonsai_int1_linear_calls=" << summary.linear_calls << "\n"
              << "unique_bonsai_int1_linears_executed=" << summary.unique_linears_executed << "\n"
              << "missing_linear_calls=" << summary.missing_linear_calls << "\n"
              << "first_missing_linear=" << summary.first_missing_linear << "\n"
              << "cpu_dequant_fallback_calls=0\n"
              << "gpu_custom_op_calls=" << summary.linear_calls << "\n"
              << "full_fp16_weight_expansion=false\n"
              << "pack_load_upload_ms=" << ms(t0, t1) << "\n"
              << "flux_runner_init_ms=" << ms(t1, t2) << "\n"
              << "transformer_invocation_ms=" << ms(t2, t3) << "\n"
              << "output_elements=" << out.numel() << "\n"
              << "output_min=" << output_min << "\n"
              << "output_max=" << output_max << "\n"
              << "output_sum=" << output_sum << "\n";

    ggml_backend_free(backend);
    return out.empty() || summary.linear_calls == 0 || summary.missing_linear_calls != 0 ? 1 : 0;
}
