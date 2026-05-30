#include "stable-diffusion.h"
#include "src/ggml_extend.hpp"
#include "src/lens_transformer_runtime.hpp"
#include "src/loader/async_weight_loader.h"
#include "src/loader/loader_stats.h"
#include "src/loader/pinned_host_arena.h"
#include "src/rope.hpp"
#include "src/tensor_ggml.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <json.hpp>

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
#include <cuda_runtime.h>
#include <cublasLt.h>
#endif

#ifdef SD_LENS_TRANSFORMER_SMOKE_EXPORTS
#define SD_LENS_TRANSFORMER_SMOKE_API extern "C" __declspec(dllexport)
#elif defined(SD_LENS_TRANSFORMER_SMOKE_RUNTIME)
#define SD_LENS_TRANSFORMER_SMOKE_API extern "C"
#else
#define SD_LENS_TRANSFORMER_SMOKE_API static
#endif

#ifndef SD_LENS_TRANSFORMER_SMOKE_INSIDE_STABLE_DIFFUSION_LIB
void log_printf(sd_log_level_t, const char*, int, const char*, ...) {
}
#endif

static const std::unordered_map<std::string, Tensor>* g_lens_in_memory_cond_tensors = nullptr;
static Tensor* g_lens_native_cuda_output_latent = nullptr;

struct TensorInput {
    std::vector<int64_t> shape;
    const float* data = nullptr;
    const ggml_bf16_t* bf16_data = nullptr;
    sd_tensor_dtype_t dtype = SD_DTYPE_F32;
};

struct LensAttentionFixture {
    int b = 0;
    int h = 0;
    int s = 0;
    int d = 0;
    std::vector<int64_t> mask_shape;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> mask;
    std::vector<float> mask_expanded;
    Tensor expected;
};

static void read_exact(std::ifstream& in, void* dst, size_t bytes);

enum class LensAttentionMode {
    RegularF32,
    Flash,
};

static const char* lens_attention_mode_name(LensAttentionMode mode) {
    switch (mode) {
        case LensAttentionMode::RegularF32:
            return "regular-f32";
        case LensAttentionMode::Flash:
            return "flash";
    }
    return "unknown";
}

static std::vector<float> read_f32_vector(std::ifstream& in, uint64_t count) {
    std::vector<float> values(static_cast<size_t>(count));
    if (count > 0) {
        read_exact(in, values.data(), static_cast<size_t>(count) * sizeof(float));
    }
    return values;
}

static LensAttentionFixture load_attention_fixture(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open attention fixture: " + path);
    }
    char magic[9] = {};
    read_exact(in, magic, sizeof(magic));
    if (std::memcmp(magic, "LENSATTN1", 9) != 0) {
        throw std::runtime_error("invalid attention fixture magic");
    }
    int32_t header[8] = {};
    read_exact(in, header, sizeof(header));

    LensAttentionFixture fx;
    fx.b = header[0];
    fx.h = header[1];
    fx.s = header[2];
    fx.d = header[3];
    fx.mask_shape = {header[4], header[5], header[6], header[7]};
    if (fx.b <= 0 || fx.h <= 0 || fx.s <= 0 || fx.d <= 0) {
        throw std::runtime_error("invalid attention fixture dimensions");
    }
    const uint64_t bh = static_cast<uint64_t>(fx.b) * static_cast<uint64_t>(fx.h);
    fx.q = read_f32_vector(in, static_cast<uint64_t>(fx.d) * fx.s * bh);
    fx.k = read_f32_vector(in, static_cast<uint64_t>(fx.d) * fx.s * bh);
    fx.v = read_f32_vector(in, static_cast<uint64_t>(fx.d) * fx.h * fx.s * fx.b);
    uint64_t mask_count = 0;
    if (fx.mask_shape.size() == 4 && fx.mask_shape[0] > 0) {
        mask_count = static_cast<uint64_t>(fx.mask_shape[0]) *
                     static_cast<uint64_t>(fx.mask_shape[1]) *
                     static_cast<uint64_t>(fx.mask_shape[2]) *
                     static_cast<uint64_t>(fx.mask_shape[3]);
    }
    fx.mask = read_f32_vector(in, mask_count);
    fx.expected.shape = {static_cast<int64_t>(fx.d) * fx.h, fx.s, fx.b};
    fx.expected.data = read_f32_vector(in, static_cast<uint64_t>(fx.d) * fx.h * fx.s * fx.b);

    if (fx.mask_shape == std::vector<int64_t>{fx.b, 1, 1, fx.s}) {
        fx.mask_expanded.assign(static_cast<size_t>(fx.s) * fx.s * fx.b * fx.h, 0.0f);
        for (int batch = 0; batch < fx.b; ++batch) {
            for (int head = 0; head < fx.h; ++head) {
                const int bh_index = batch * fx.h + head;
                for (int query = 0; query < fx.s; ++query) {
                    for (int key = 0; key < fx.s; ++key) {
                        const float value = fx.mask[static_cast<size_t>(batch) * fx.s + key];
                        fx.mask_expanded[static_cast<size_t>(key) +
                                         static_cast<size_t>(fx.s) *
                                             (static_cast<size_t>(query) + static_cast<size_t>(fx.s) * bh_index)] = value;
                    }
                }
            }
        }
    }
    return fx;
}

static bool query_nvidia_smi_vram(double& used_gib, double& free_gib, double& total_gib) {
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits", "r");
#endif
    if (pipe == nullptr) {
        return false;
    }
    char line[256] = {};
    const bool ok = std::fgets(line, sizeof(line), pipe) != nullptr;
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    if (!ok) {
        return false;
    }
    for (char& ch : line) {
        if (ch == ',') {
            ch = ' ';
        }
    }
    double used_mib = 0.0;
    double free_mib = 0.0;
    double total_mib = 0.0;
    if (std::sscanf(line, "%lf %lf %lf", &used_mib, &free_mib, &total_mib) != 3 || total_mib <= 0.0) {
        return false;
    }
    used_gib = used_mib / 1024.0;
    free_gib = free_mib / 1024.0;
    total_gib = total_mib / 1024.0;
    return true;
}

static void print_transformer_vram_snapshot(const char* stage) {
    const char* enable = std::getenv("SD_LENS_ENABLE_IN_PROCESS_VRAM_SNAPSHOT");
    if (enable == nullptr || std::strcmp(enable, "1") != 0) {
        std::cout << "vram_snapshot stage=" << stage << " source=disabled\n";
        return;
    }
    double used_gib = 0.0;
    double free_gib = 0.0;
    double total_gib = 0.0;
    if (query_nvidia_smi_vram(used_gib, free_gib, total_gib)) {
        std::cout << "vram_snapshot stage=" << stage
                  << " used_gib=" << used_gib
                  << " free_gib=" << free_gib
                  << " total_gib=" << total_gib
                  << " source=nvidia-smi\n";
    }
}

struct DiffStats {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float max_ref = 0.0f;
    float mean_ref = 0.0f;
    float rel_max = 0.0f;
    float rel_mean = 0.0f;
    size_t max_index = 0;
    size_t finite_count = 0;
    size_t nonfinite_actual = 0;
    size_t nonfinite_ref = 0;
};

struct TensorRangeStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    size_t finite_count = 0;
    size_t nonfinite_count = 0;
};

static uint64_t elem_count(const std::vector<int64_t>& shape) {
    uint64_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("invalid tensor dimension");
        }
        count *= static_cast<uint64_t>(dim);
    }
    return count;
}

static int64_t dim(const Tensor& t, size_t i) {
    if (i >= t.shape.size()) {
        throw std::runtime_error("rank mismatch");
    }
    return t.shape[i];
}

static size_t idx2(const Tensor& t, int64_t a, int64_t b) {
    return static_cast<size_t>(a * dim(t, 1) + b);
}

static size_t idx3(const Tensor& t, int64_t a, int64_t b, int64_t c) {
    return static_cast<size_t>((a * dim(t, 1) + b) * dim(t, 2) + c);
}

static size_t idx4(const Tensor& t, int64_t a, int64_t b, int64_t c, int64_t d) {
    return static_cast<size_t>(((a * dim(t, 1) + b) * dim(t, 2) + c) * dim(t, 3) + d);
}

static void read_exact(std::ifstream& in, void* dst, size_t bytes) {
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) {
        throw std::runtime_error("unexpected end of fixture");
    }
}

static std::unordered_map<std::string, Tensor> load_fixture(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open fixture: " + path);
    }
    char magic[8] = {};
    read_exact(in, magic, sizeof(magic));
    if (std::memcmp(magic, "LENSBLK1", sizeof(magic)) != 0) {
        throw std::runtime_error("invalid Lens block fixture magic");
    }
    uint32_t tensor_count = 0;
    read_exact(in, &tensor_count, sizeof(tensor_count));
    std::unordered_map<std::string, Tensor> tensors;
    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint32_t name_len = 0;
        uint32_t rank = 0;
        read_exact(in, &name_len, sizeof(name_len));
        read_exact(in, &rank, sizeof(rank));
        std::string name(name_len, '\0');
        read_exact(in, name.data(), name.size());
        Tensor t;
        t.shape.resize(rank);
        for (uint32_t r = 0; r < rank; ++r) {
            read_exact(in, &t.shape[r], sizeof(int64_t));
        }
        const uint64_t count = elem_count(t.shape);
        t.data.resize(static_cast<size_t>(count));
        read_exact(in, t.data.data(), static_cast<size_t>(count * sizeof(float)));
        tensors.emplace(std::move(name), std::move(t));
    }
    return tensors;
}

static const Tensor& need(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("fixture missing tensor: " + name);
    }
    return it->second;
}

static uint64_t parse_u64(const char* text, const char* name) {
    if (text == nullptr || text[0] == '\0') {
        throw std::runtime_error(std::string(name) + " requires a value");
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer for ") + name);
    }
    return static_cast<uint64_t>(value);
}

static int parse_i32(const char* text, const char* name) {
    if (text == nullptr || text[0] == '\0') {
        throw std::runtime_error(std::string(name) + " requires a value");
    }
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 10000) {
        throw std::runtime_error(std::string("invalid integer for ") + name);
    }
    return static_cast<int>(value);
}

static uint64_t read_le_u64(std::ifstream& in) {
    uint8_t bytes[8] = {};
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!in) {
        throw std::runtime_error("failed to read safetensors header size");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

static std::vector<std::filesystem::path> safetensor_files(const std::filesystem::path& path) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_regular_file(path)) {
        if (path.extension() == ".safetensors") {
            files.push_back(path);
        }
        return files;
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error("real block transformer path is not a file or directory");
    }
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

struct SafetensorEntry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

static std::vector<SafetensorEntry> read_safetensors_header(const std::filesystem::path& file_path,
                                                            const std::string& prefix,
                                                            uint64_t& payload_offset,
                                                            bool allow_i32 = false) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors file: " + file_path.string());
    }
    const uint64_t header_size = read_le_u64(in);
    if (header_size == 0 || header_size > (256ull * 1024ull * 1024ull)) {
        throw std::runtime_error("safetensors header size is outside the guarded range");
    }
    std::string header(static_cast<size_t>(header_size), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in) {
        throw std::runtime_error("failed to read safetensors header: " + file_path.string());
    }
    payload_offset = 8 + header_size;

    nlohmann::json json = nlohmann::json::parse(header);
    std::vector<SafetensorEntry> entries;
    for (const auto& item : json.items()) {
        const std::string name = item.key();
        if (name == "__metadata__" || name.rfind(prefix, 0) != 0) {
            continue;
        }
        const auto& info = item.value();
        const std::string dtype = info.at("dtype").get<std::string>();
        if (dtype != "F32" && !(allow_i32 && dtype == "I32")) {
            throw std::runtime_error("real Lens block smoke currently supports only F32 tensors: " + name);
        }
        SafetensorEntry entry;
        entry.name = name;
        entry.dtype = dtype;
        for (const auto& dim : info.at("shape")) {
            const int64_t value = dim.get<int64_t>();
            if (value <= 0) {
                throw std::runtime_error("invalid tensor shape in " + name);
            }
            entry.shape.push_back(value);
        }
        entry.begin = info.at("data_offsets").at(0).get<uint64_t>();
        entry.end = info.at("data_offsets").at(1).get<uint64_t>();
        const uint64_t bytes = entry.end - entry.begin;
        if (bytes != elem_count(entry.shape) * sizeof(float)) {
            throw std::runtime_error("unexpected safetensors byte count in " + name);
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

static std::unordered_map<std::string, Tensor> load_real_lens_block(const std::string& transformer_path,
                                                                    int block_index,
                                                                    uint64_t max_block_bytes) {
    const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
    auto files = safetensor_files(std::filesystem::path(transformer_path));
    if (files.empty()) {
        throw std::runtime_error("no safetensors files found for real Lens block load");
    }

    struct PendingRead {
        std::filesystem::path file;
        SafetensorEntry entry;
        uint64_t payload_offset = 0;
    };
    std::vector<PendingRead> pending;
    uint64_t total_bytes = 0;
    for (const auto& file : files) {
        uint64_t payload_offset = 0;
        auto entries = read_safetensors_header(file, prefix, payload_offset);
        for (auto& entry : entries) {
            const uint64_t bytes = entry.end - entry.begin;
            if (total_bytes + bytes > max_block_bytes) {
                std::ostringstream oss;
                oss << "refusing real Lens block load: block " << block_index
                    << " exceeds byte cap " << max_block_bytes;
                throw std::runtime_error(oss.str());
            }
            total_bytes += bytes;
            pending.push_back(PendingRead{file, std::move(entry), payload_offset});
        }
    }
    if (pending.empty()) {
        throw std::runtime_error("requested Lens block was not found in transformer shards");
    }

    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(pending.size());
    for (const PendingRead& read : pending) {
        std::ifstream in(read.file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open safetensors file for tensor read: " + read.file.string());
        }
        Tensor tensor;
        tensor.shape = read.entry.shape;
        tensor.data.resize(static_cast<size_t>(elem_count(tensor.shape)));
        in.seekg(static_cast<std::streamoff>(read.payload_offset + read.entry.begin));
        in.read(reinterpret_cast<char*>(tensor.data.data()),
                static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
        if (!in) {
            throw std::runtime_error("failed to read tensor payload: " + read.entry.name);
        }
        tensors.emplace(read.entry.name, std::move(tensor));
    }
    return tensors;
}

static std::unordered_map<std::string, Tensor> load_real_lens_named_tensors(const std::string& transformer_path,
                                                                            const std::vector<std::string>& names,
                                                                            uint64_t max_bytes,
                                                                            bool allow_i32 = false) {
    const auto files = safetensor_files(std::filesystem::path(transformer_path));
    if (files.empty()) {
        throw std::runtime_error("no safetensors files found for real Lens named tensor load");
    }
    std::unordered_set<std::string> wanted(names.begin(), names.end());

    struct PendingRead {
        std::filesystem::path file;
        SafetensorEntry entry;
        uint64_t payload_offset = 0;
    };
    std::vector<PendingRead> pending;
    uint64_t total_bytes = 0;
    for (const auto& file : files) {
        uint64_t payload_offset = 0;
        auto entries = read_safetensors_header(file, "", payload_offset, allow_i32);
        for (auto& entry : entries) {
            if (wanted.find(entry.name) == wanted.end()) {
                continue;
            }
            const uint64_t bytes = entry.end - entry.begin;
            if (total_bytes + bytes > max_bytes) {
                throw std::runtime_error("refusing real Lens named tensor load: selected tensors exceed byte cap");
            }
            total_bytes += bytes;
            wanted.erase(entry.name);
            pending.push_back(PendingRead{file, std::move(entry), payload_offset});
        }
    }
    if (!wanted.empty()) {
        throw std::runtime_error("missing required real Lens tensor: " + *wanted.begin());
    }

    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(pending.size());
    for (const PendingRead& read : pending) {
        std::ifstream in(read.file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open safetensors file for tensor read: " + read.file.string());
        }
        Tensor tensor;
        tensor.shape = read.entry.shape;
        tensor.data.resize(static_cast<size_t>(elem_count(tensor.shape)));
        in.seekg(static_cast<std::streamoff>(read.payload_offset + read.entry.begin));
        if (read.entry.dtype == "I32") {
            std::vector<int32_t> values(tensor.data.size());
            in.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(int32_t)));
            if (!in) {
                throw std::runtime_error("failed to read tensor payload: " + read.entry.name);
            }
            for (size_t i = 0; i < values.size(); ++i) {
                tensor.data[i] = static_cast<float>(values[i]);
            }
        } else {
            in.read(reinterpret_cast<char*>(tensor.data.data()),
                    static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
            if (!in) {
                throw std::runtime_error("failed to read tensor payload: " + read.entry.name);
            }
        }
        tensors.emplace(read.entry.name, std::move(tensor));
    }
    return tensors;
}

static Tensor copy_lens_context_tensor(sd_ctx_t* ctx,
                                       sd_lens_transformer_handle_t handle,
                                       const std::string& name) {
    int64_t shape[4] = {};
    uint32_t rank = 0;
    uint64_t elements = 0;
    if (!sd_lens_transformer_copy_tensor_f32(ctx,
                                             handle,
                                             name.c_str(),
                                             shape,
                                             4,
                                             &rank,
                                             nullptr,
                                             0,
                                             &elements)) {
        throw std::runtime_error("failed to query cached Lens tensor: " + name);
    }
    Tensor tensor;
    tensor.shape.assign(shape, shape + rank);
    tensor.data.resize(static_cast<size_t>(elements));
    if (!sd_lens_transformer_copy_tensor_f32(ctx,
                                             handle,
                                             name.c_str(),
                                             shape,
                                             4,
                                             &rank,
                                             tensor.data.data(),
                                             elements,
                                             &elements)) {
        throw std::runtime_error("failed to copy cached Lens tensor: " + name);
    }
    return tensor;
}

static TensorInput borrow_lens_context_tensor(sd_ctx_t* ctx,
                                              sd_lens_transformer_handle_t handle,
                                              const std::string& name) {
    int64_t shape[4] = {};
    uint32_t rank = 0;
    uint64_t elements = 0;
    const void* data = nullptr;
    sd_tensor_dtype_t dtype = SD_DTYPE_F32;
    if (!sd_lens_transformer_borrow_tensor_data(ctx,
                                                handle,
                                                name.c_str(),
                                                shape,
                                                4,
                                                &rank,
                                                &data,
                                                &elements,
                                                &dtype)) {
        throw std::runtime_error("failed to borrow cached Lens tensor: " + name);
    }
    TensorInput input;
    input.shape.assign(shape, shape + rank);
    input.dtype = dtype;
    if (dtype == SD_DTYPE_BF16) {
        input.bf16_data = static_cast<const ggml_bf16_t*>(data);
    } else {
        input.data = static_cast<const float*>(data);
    }
    if (elem_count(input.shape) != elements || data == nullptr) {
        throw std::runtime_error("borrowed cached Lens tensor has invalid shape/data: " + name);
    }
    return input;
}

static std::unordered_map<std::string, Tensor> load_cached_lens_named_tensors(sd_ctx_t* ctx,
                                                                              sd_lens_transformer_handle_t handle,
                                                                              const std::vector<std::string>& names) {
    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(names.size());
    for (const std::string& name : names) {
        tensors.emplace(name, copy_lens_context_tensor(ctx, handle, name));
    }
    return tensors;
}

static std::unordered_map<std::string, TensorInput> borrow_cached_lens_named_tensors(sd_ctx_t* ctx,
                                                                                     sd_lens_transformer_handle_t handle,
                                                                                     const std::vector<std::string>& names) {
    std::unordered_map<std::string, TensorInput> tensors;
    tensors.reserve(names.size());
    for (const std::string& name : names) {
        tensors.emplace(name, borrow_lens_context_tensor(ctx, handle, name));
    }
    return tensors;
}

static std::unordered_map<std::string, Tensor> load_cached_lens_block(sd_ctx_t* ctx,
                                                                      sd_lens_transformer_handle_t handle,
                                                                      int block_index) {
    static const char* suffixes[] = {
        "img_norm1.weight", "img_norm2.weight", "img_mod.1.weight", "img_mod.1.bias",
        "txt_norm1.weight", "txt_norm2.weight", "txt_mod.1.weight", "txt_mod.1.bias",
        "attn.img_qkv.weight", "attn.img_qkv.bias", "attn.txt_qkv.weight", "attn.txt_qkv.bias",
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "attn.to_out.0.weight", "attn.to_out.0.bias", "attn.to_add_out.weight", "attn.to_add_out.bias",
        "img_mlp.w1.weight", "img_mlp.w2.weight", "img_mlp.w3.weight",
        "txt_mlp.w1.weight", "txt_mlp.w2.weight", "txt_mlp.w3.weight",
    };
    const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
    std::vector<std::string> names;
    names.reserve(sizeof(suffixes) / sizeof(suffixes[0]));
    for (const char* suffix : suffixes) {
        names.push_back(prefix + suffix);
    }
    return load_cached_lens_named_tensors(ctx, handle, names);
}

static std::unordered_map<std::string, TensorInput> borrow_cached_lens_block(sd_ctx_t* ctx,
                                                                             sd_lens_transformer_handle_t handle,
                                                                             int block_index) {
    static const char* suffixes[] = {
        "img_norm1.weight", "img_norm2.weight", "img_mod.1.weight", "img_mod.1.bias",
        "txt_norm1.weight", "txt_norm2.weight", "txt_mod.1.weight", "txt_mod.1.bias",
        "attn.img_qkv.weight", "attn.img_qkv.bias", "attn.txt_qkv.weight", "attn.txt_qkv.bias",
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "attn.to_out.0.weight", "attn.to_out.0.bias", "attn.to_add_out.weight", "attn.to_add_out.bias",
        "img_mlp.w1.weight", "img_mlp.w2.weight", "img_mlp.w3.weight",
        "txt_mlp.w1.weight", "txt_mlp.w2.weight", "txt_mlp.w3.weight",
    };
    const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
    std::vector<std::string> names;
    names.reserve(sizeof(suffixes) / sizeof(suffixes[0]));
    for (const char* suffix : suffixes) {
        names.push_back(prefix + suffix);
    }
    return borrow_cached_lens_named_tensors(ctx, handle, names);
}

static Tensor make_pattern3(int64_t bsz, int64_t seq, int64_t hidden, float scale) {
    Tensor t{{bsz, seq, hidden}, std::vector<float>(static_cast<size_t>(bsz * seq * hidden))};
    for (size_t i = 0; i < t.data.size(); ++i) {
        t.data[i] = std::sin(static_cast<float>(i % 97) * 0.17f) * scale;
    }
    return t;
}

static Tensor make_pattern2(int64_t bsz, int64_t hidden, float scale) {
    Tensor t{{bsz, hidden}, std::vector<float>(static_cast<size_t>(bsz * hidden))};
    for (size_t i = 0; i < t.data.size(); ++i) {
        t.data[i] = std::cos(static_cast<float>(i % 89) * 0.11f) * scale;
    }
    return t;
}

static Tensor make_identity_freqs(int64_t seq, int64_t head_dim) {
    Tensor t{{seq, head_dim / 2, 2}, std::vector<float>(static_cast<size_t>(seq * (head_dim / 2) * 2), 0.0f)};
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t p = 0; p < head_dim / 2; ++p) {
            t.data[idx3(t, s, p, 0)] = 1.0f;
            t.data[idx3(t, s, p, 1)] = 0.0f;
        }
    }
    return t;
}

static Tensor make_random_normal_packed_tokens(int64_t seq, int64_t channels, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Tensor t{{1, seq, channels}, std::vector<float>(static_cast<size_t>(seq * channels))};
    for (float& value : t.data) {
        value = normal(rng);
    }
    return t;
}

static Tensor make_lens_timestep_proj(float timestep) {
    constexpr int64_t embedding_dim = 256;
    constexpr int64_t half_dim = embedding_dim / 2;
    constexpr float max_period = 10000.0f;
    Tensor proj{{1, embedding_dim}, std::vector<float>(static_cast<size_t>(embedding_dim), 0.0f)};
    for (int64_t i = 0; i < half_dim; ++i) {
        const float exponent = -std::log(max_period) * static_cast<float>(i) / static_cast<float>(half_dim);
        const float angle = 1000.0f * timestep * std::exp(exponent);
        proj.data[static_cast<size_t>(i)] = std::cos(angle);
        proj.data[static_cast<size_t>(half_dim + i)] = std::sin(angle);
    }
    return proj;
}

static void fill_rope_axis(std::vector<float>& out, int64_t seq, int64_t offset, int64_t axis_complex_dim, int64_t index) {
    for (int64_t d = 0; d < axis_complex_dim; ++d) {
        const float inv = 1.0f / std::pow(10000.0f, static_cast<float>(2 * d) / static_cast<float>(axis_complex_dim * 2));
        const float angle = static_cast<float>(index) * inv;
        out[static_cast<size_t>((seq * 32 + offset + d) * 2 + 0)] = std::cos(angle);
        out[static_cast<size_t>((seq * 32 + offset + d) * 2 + 1)] = std::sin(angle);
    }
}

static Tensor make_lens_image_rope_freqs(int64_t latent_h, int64_t latent_w) {
    Tensor freqs{{latent_h * latent_w, 32, 2}, std::vector<float>(static_cast<size_t>(latent_h * latent_w * 32 * 2), 0.0f)};
    int64_t seq = 0;
    for (int64_t y = 0; y < latent_h; ++y) {
        const int64_t y_index = y < (latent_h - latent_h / 2) ? -(latent_h - latent_h / 2) + y : y - (latent_h - latent_h / 2);
        for (int64_t x = 0; x < latent_w; ++x) {
            const int64_t x_index = x < (latent_w - latent_w / 2) ? -(latent_w - latent_w / 2) + x : x - (latent_w - latent_w / 2);
            fill_rope_axis(freqs.data, seq, 0, 4, 0);
            fill_rope_axis(freqs.data, seq, 4, 14, y_index);
            fill_rope_axis(freqs.data, seq, 18, 14, x_index);
            ++seq;
        }
    }
    return freqs;
}

static Tensor make_lens_text_rope_freqs(int64_t text_seq_len, int64_t latent_h, int64_t latent_w) {
    const int64_t start_index = std::max(latent_h / 2, latent_w / 2);
    Tensor freqs{{text_seq_len, 32, 2}, std::vector<float>(static_cast<size_t>(text_seq_len * 32 * 2), 0.0f)};
    for (int64_t s = 0; s < text_seq_len; ++s) {
        const int64_t index = start_index + s;
        fill_rope_axis(freqs.data, s, 0, 4, index);
        fill_rope_axis(freqs.data, s, 4, 14, index);
        fill_rope_axis(freqs.data, s, 18, 14, index);
    }
    return freqs;
}

static Tensor make_attention_mask(int64_t bsz, int64_t seq) {
    return Tensor{{bsz, seq}, std::vector<float>(static_cast<size_t>(bsz * seq), 0.0f)};
}

static Tensor make_joint_attention_mask_from_lens_cond(const Tensor& cond_mask, int64_t img_seq) {
    if (dim(cond_mask, 0) != 1 || dim(cond_mask, 1) <= 0 || img_seq <= 0) {
        throw std::runtime_error("lens_cond_v1 attention_mask must have shape [1, txt_seq]");
    }
    const int64_t txt_seq = dim(cond_mask, 1);
    Tensor mask{{1, img_seq + txt_seq}, std::vector<float>(static_cast<size_t>(img_seq + txt_seq), 0.0f)};
    for (int64_t s = 0; s < txt_seq; ++s) {
        const float valid = cond_mask.data[idx2(cond_mask, 0, s)];
        mask.data[idx2(mask, 0, img_seq + s)] = valid != 0.0f ? 0.0f : -std::numeric_limits<float>::infinity();
    }
    return mask;
}

static std::string tensor_shape_string(const Tensor& tensor) {
    std::ostringstream out;
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << tensor.shape[i];
    }
    return out.str();
}

static bool write_f32_npy(const std::string& path, const Tensor& tensor) {
    if (tensor.shape.empty()) {
        return false;
    }
    std::ostringstream shape;
    shape << "(";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i > 0) {
            shape << ", ";
        }
        shape << tensor.shape[i];
    }
    if (tensor.shape.size() == 1) {
        shape << ",";
    }
    shape << ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape.str() + ", }";
    const size_t preamble = 10;
    size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    if (padding == 16) {
        padding = 0;
    }
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > 65535) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write("\x93NUMPY", 6);
    const uint8_t version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), sizeof(version));
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(tensor.data.data()),
              static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));
    return static_cast<bool>(out);
}

static Tensor apply_lens_flow_step_at(const Tensor& sample,
                                      const Tensor& model_output,
                                      const std::vector<float>& sigmas,
                                      int step_index,
                                      float& sigma0,
                                      float& sigma1,
                                      float& dt) {
    if (step_index < 0 ||
        static_cast<size_t>(step_index + 1) >= sigmas.size() ||
        sample.shape != model_output.shape) {
        throw std::runtime_error("invalid Lens scheduled flow step inputs");
    }
    sigma0 = sigmas[static_cast<size_t>(step_index)];
    sigma1 = sigmas[static_cast<size_t>(step_index + 1)];
    dt = sigma1 - sigma0;

    Tensor updated = sample;
    for (size_t i = 0; i < updated.data.size(); ++i) {
        updated.data[i] = sample.data[i] + dt * model_output.data[i];
    }
    return updated;
}

static Tensor lens_packed_tokens_to_vae_whcn(const Tensor& packed_tokens_bsh,
                                             int64_t packed_h,
                                             int64_t packed_w) {
    if (packed_tokens_bsh.shape.size() != 3 ||
        dim(packed_tokens_bsh, 0) != 1 ||
        dim(packed_tokens_bsh, 1) != packed_h * packed_w ||
        dim(packed_tokens_bsh, 2) != 128) {
        throw std::runtime_error("Lens packed VAE latent conversion expected BxSx128 tokens");
    }
    Tensor packed{{1, 128, packed_h, packed_w},
                  std::vector<float>(static_cast<size_t>(128 * packed_h * packed_w), 0.0f)};
    for (int64_t py = 0; py < packed_h; ++py) {
        for (int64_t px = 0; px < packed_w; ++px) {
            const int64_t token = py * packed_w + px;
            for (int64_t c = 0; c < 128; ++c) {
                const size_t src = static_cast<size_t>(token * 128 + c);
                const size_t dst = static_cast<size_t>(c * packed_h * packed_w + py * packed_w + px);
                packed.data[dst] = packed_tokens_bsh.data[src];
            }
        }
    }
    return packed;
}

static std::vector<float> build_lens_tiny_flow_sigmas(int steps, int image_seq_len) {
    if (steps <= 0 || image_seq_len <= 0) {
        throw std::runtime_error("invalid Lens tiny flow schedule inputs");
    }
    sd_lens_schedule_options_t options;
    sd_lens_schedule_options_init(&options);
    options.steps = steps;
    options.image_seq_len = image_seq_len;

    std::vector<float> sigmas(static_cast<size_t>(steps + 1), 0.0f);
    std::vector<float> timesteps(static_cast<size_t>(steps), 0.0f);
    sd_lens_schedule_desc_t desc;
    sd_lens_schedule_desc_init(&desc);
    if (!sd_lens_turbo_build_schedule(&options,
                                      sigmas.data(),
                                      static_cast<uint32_t>(sigmas.size()),
                                      timesteps.data(),
                                      static_cast<uint32_t>(timesteps.size()),
                                      &desc)) {
        throw std::runtime_error("sd_lens_turbo_build_schedule failed for tiny flow loop");
    }
    return sigmas;
}

static bool tensor_stats(const Tensor& t, float& max_abs, float& mean_abs) {
    double sum = 0.0;
    max_abs = 0.0f;
    for (float value : t.data) {
        if (!std::isfinite(value)) {
            return false;
        }
        const float abs_value = std::fabs(value);
        max_abs = std::max(max_abs, abs_value);
        sum += abs_value;
    }
    mean_abs = t.data.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(t.data.size()));
    return true;
}

static TensorRangeStats tensor_range_stats(const Tensor& t) {
    TensorRangeStats stats;
    double sum = 0.0;
    bool initialized = false;
    for (float value : t.data) {
        if (!std::isfinite(value)) {
            ++stats.nonfinite_count;
            continue;
        }
        if (!initialized) {
            stats.min = value;
            stats.max = value;
            initialized = true;
        } else {
            stats.min = std::min(stats.min, value);
            stats.max = std::max(stats.max, value);
        }
        sum += value;
        ++stats.finite_count;
    }
    if (stats.finite_count > 0) {
        stats.mean = static_cast<float>(sum / static_cast<double>(stats.finite_count));
    }
    return stats;
}

static void print_tensor_range_stats(const std::string& label, int block_index, const Tensor& t) {
    const TensorRangeStats stats = tensor_range_stats(t);
    std::cout << label;
    if (block_index >= 0) {
        std::cout << " block=" << block_index;
    }
    std::cout << " finite_count=" << stats.finite_count
              << " nonfinite_count=" << stats.nonfinite_count
              << " min=" << stats.min
              << " max=" << stats.max
              << " mean=" << stats.mean
              << "\n";
}

static bool tensor_has_nonfinite(const Tensor& t) {
    return tensor_range_stats(t).nonfinite_count != 0;
}

struct RunnerTimingTotals {
    double offload_params_seconds = 0.0;
    double alloc_compute_buffer_seconds = 0.0;
    double graph_build_seconds = 0.0;
    double graph_alloc_seconds = 0.0;
    double input_copy_seconds = 0.0;
    double input_copy_submit_seconds = 0.0;
    double input_sync_seconds = 0.0;
    double backend_compute_seconds = 0.0;
    double backend_compute_submit_seconds = 0.0;
    double backend_sync_seconds = 0.0;
    double output_copy_seconds = 0.0;
    double cleanup_seconds = 0.0;
    uint64_t input_copy_bytes = 0;
    uint64_t output_copy_bytes = 0;
    uint64_t runner_count = 0;
    std::map<std::string, uint64_t> copy_bytes_by_category;
    std::map<std::string, double> copy_seconds_by_category;
    std::map<std::string, uint64_t> copy_count_by_category;
};

struct BlockTimingTotals {
    double fetch_seconds = 0.0;
    double setup_seconds = 0.0;
    double upload_seconds = 0.0;
    double upload_submit_seconds = 0.0;
    double upload_sync_seconds = 0.0;
    double compute_seconds = 0.0;
    double compute_submit_seconds = 0.0;
    double sync_seconds = 0.0;
    uint64_t input_copy_bytes = 0;
    uint64_t calls = 0;
};

struct BlockStepTiming {
    double alloc_compute_buffer_seconds = 0.0;
    double graph_build_seconds = 0.0;
    double graph_alloc_seconds = 0.0;
    double input_copy_seconds = 0.0;
    double input_copy_submit_seconds = 0.0;
    double input_sync_seconds = 0.0;
    double compute_seconds = 0.0;
    double compute_submit_seconds = 0.0;
    double sync_seconds = 0.0;
    double output_copy_seconds = 0.0;
    double cleanup_seconds = 0.0;
    uint64_t input_copy_bytes = 0;
};

struct LensResidentBlockWeights {
    std::unordered_map<std::string, std::unique_ptr<GgmlBackendTensorResource>> tensors;
    uint64_t bytes = 0;

    LensResidentBlockWeights() = default;
    LensResidentBlockWeights(const LensResidentBlockWeights&) = delete;
    LensResidentBlockWeights& operator=(const LensResidentBlockWeights&) = delete;
    LensResidentBlockWeights(LensResidentBlockWeights&&) noexcept = default;
    LensResidentBlockWeights& operator=(LensResidentBlockWeights&&) noexcept = default;
};

struct LensResidentStaticTensors {
    std::unordered_map<std::string, std::unique_ptr<GgmlBackendTensorResource>> tensors;
    uint64_t bytes = 0;

    LensResidentStaticTensors() = default;
    LensResidentStaticTensors(const LensResidentStaticTensors&) = delete;
    LensResidentStaticTensors& operator=(const LensResidentStaticTensors&) = delete;
    LensResidentStaticTensors(LensResidentStaticTensors&&) noexcept = default;
    LensResidentStaticTensors& operator=(LensResidentStaticTensors&&) noexcept = default;

    void add(std::string name, std::unique_ptr<GgmlBackendTensorResource> resource) {
        if (resource != nullptr && resource->tensor != nullptr) {
            bytes += static_cast<uint64_t>(ggml_nbytes(resource->tensor));
        }
        tensors.emplace(std::move(name), std::move(resource));
    }

    std::unordered_map<std::string, const GgmlBackendTensorResource*> ptrs() const {
        std::unordered_map<std::string, const GgmlBackendTensorResource*> out;
        out.reserve(tensors.size());
        for (const auto& item : tensors) {
            if (item.second != nullptr && !item.second->empty()) {
                out.emplace(item.first, item.second.get());
            }
        }
        return out;
    }
};

struct LensWarmTransformerCache {
    bool valid = false;
    std::string key;
    sd_ctx_t* lens_ctx = nullptr;
    sd_lens_transformer_handle_t transformer_handle = 0;
    sd_lens_transformer_desc_t transformer_desc{};
    ggml_backend_t cuda_backend = nullptr;
    std::vector<LensResidentBlockWeights> resident_blocks;
    uint64_t resident_weight_bytes = 0;
    size_t resident_weight_tensors = 0;

    void clear() {
        resident_blocks.clear();
        if (cuda_backend != nullptr) {
            ggml_backend_free(cuda_backend);
            cuda_backend = nullptr;
        }
        if (lens_ctx != nullptr) {
            if (transformer_handle != 0) {
                sd_lens_transformer_release(lens_ctx, transformer_handle);
                transformer_handle = 0;
            }
            free_sd_ctx(lens_ctx);
            lens_ctx = nullptr;
        }
        transformer_desc = {};
        resident_weight_bytes = 0;
        resident_weight_tensors = 0;
        key.clear();
        valid = false;
    }
};

static LensWarmTransformerCache g_lens_warm_transformer_cache;

static std::string make_lens_warm_transformer_cache_key(const std::string& transformer_dir,
                                                        const std::string& transformer_residency) {
    return transformer_dir + "|residency=" + transformer_residency + "|weights=bf16-full";
}

extern "C" void sd_lens_transformer_smoke_warm_cache_clear() {
    g_lens_warm_transformer_cache.clear();
}

class LensBackendResourceUploader : public GGMLRunner {
public:
    explicit LensBackendResourceUploader(ggml_backend_t backend)
        : GGMLRunner(backend, false) {
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
        loader_config_ = sd::loader::get_config();
        if (loader_config_.enable_threaded_loader) {
            pinned_arena_ = std::make_unique<sd::loader::PinnedHostArena>(loader_config_);
            async_loader_ = std::make_unique<sd::loader::AsyncWeightLoader>();
            if (async_loader_ == nullptr || !async_loader_->available()) {
                async_loader_.reset();
            }
        }
#endif
    }

    std::string get_desc() override {
        return "LensBackendResourceUploader";
    }

    std::unique_ptr<GgmlBackendTensorResource> upload_f32(const TensorInput& input, const std::string& name) {
        if (input.data != nullptr) {
            return copy_data_to_resource_handle<float>(input.shape, input.data, name.c_str());
        }
        if (input.bf16_data == nullptr) {
            return nullptr;
        }
        std::vector<float> f32(static_cast<size_t>(elem_count(input.shape)));
        ggml_bf16_to_fp32_row(input.bf16_data, f32.data(), static_cast<int64_t>(f32.size()));
        return copy_data_to_resource_handle<float>(input.shape, f32.data(), name.c_str());
    }

    std::unique_ptr<GgmlBackendTensorResource> upload_bf16_from_f32(const TensorInput& input, const std::string& name) {
        if (input.shape.empty() || input.shape.size() > 5 || (input.data == nullptr && input.bf16_data == nullptr)) {
            return nullptr;
        }
        int n_dims = std::min(static_cast<int>(input.shape.size()), GGML_MAX_DIMS);
        std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
        for (int64_t i = 0; i < n_dims; ++i) {
            ne[static_cast<size_t>(i)] = input.shape[static_cast<size_t>(i)];
        }
        if (input.shape.size() == 5) {
            ne[3] *= input.shape[4];
        }
        const size_t count = elem_count(input.shape);

        auto handle = std::make_unique<GgmlBackendTensorResource>();
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;
        handle->ctx       = ggml_init(params);
        GGML_ASSERT(handle->ctx != nullptr);
        handle->tensor = ggml_new_tensor(handle->ctx, GGML_TYPE_BF16, n_dims, ne.data());
        ggml_set_name(handle->tensor, name.c_str());
        handle->buffer = ggml_backend_alloc_ctx_tensors(handle->ctx, runtime_backend);
        if (handle->buffer == nullptr) {
            LOG_ERROR("%s alloc BF16 backend tensor resource failed", get_desc().c_str());
            return nullptr;
        }
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
        if (loader_config_.enable_threaded_loader && pinned_arena_ != nullptr && async_loader_ != nullptr) {
            const size_t bytes = ggml_nbytes(handle->tensor);
            sd::loader::PinnedHostSpan span = pinned_arena_->acquire(bytes);
            if (span.data == nullptr) {
                async_loader_->synchronize();
                pinned_arena_->reset();
                span = pinned_arena_->acquire(bytes);
            }
            if (span.data != nullptr) {
                if (input.bf16_data != nullptr) {
                    std::memcpy(span.data, input.bf16_data, bytes);
                } else {
                    ggml_fp32_to_bf16_row(input.data, static_cast<ggml_bf16_t*>(span.data), static_cast<int64_t>(count));
                }
                if (async_loader_->upload(runtime_backend, handle->tensor, span.data, 0, bytes)) {
                    sd::loader::note_fast_path(static_cast<uint64_t>(bytes));
                    return handle;
                }
                return handle;
            }
            sd::loader::note_fallback(sd::loader::LoaderFallbackReason::arena_unavailable, static_cast<uint64_t>(bytes));
        }
#endif
        std::vector<ggml_bf16_t> bf16;
        const ggml_bf16_t* bf16_data = input.bf16_data;
        if (bf16_data == nullptr) {
            bf16.resize(count);
            ggml_fp32_to_bf16_row(input.data, bf16.data(), static_cast<int64_t>(count));
            bf16_data = bf16.data();
        }
        ggml_backend_tensor_set(handle->tensor, bf16_data, 0, ggml_nbytes(handle->tensor));
        ggml_backend_synchronize(runtime_backend);
        return handle;
    }

    void synchronize() {
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
        if (async_loader_ != nullptr) {
            async_loader_->synchronize();
        }
#endif
    }

private:
#if defined(SD_CUDA_THREADED_WEIGHT_LOADER) && defined(SD_USE_CUDA)
    sd::loader::LoaderConfig loader_config_{};
    std::unique_ptr<sd::loader::PinnedHostArena> pinned_arena_;
    std::unique_ptr<sd::loader::AsyncWeightLoader> async_loader_;
#endif
};

static void add_runner_timing(RunnerTimingTotals& totals, const GGMLRunnerTimingProfile& timing) {
    totals.offload_params_seconds += timing.offload_params_seconds;
    totals.alloc_compute_buffer_seconds += timing.alloc_compute_buffer_seconds;
    totals.graph_build_seconds += timing.graph_build_seconds;
    totals.graph_alloc_seconds += timing.graph_alloc_seconds;
    totals.input_copy_seconds += timing.input_copy_seconds;
    totals.input_copy_submit_seconds += timing.input_copy_submit_seconds;
    totals.input_sync_seconds += timing.input_sync_seconds;
    totals.backend_compute_seconds += timing.backend_compute_seconds;
    totals.backend_compute_submit_seconds += timing.backend_compute_submit_seconds;
    totals.backend_sync_seconds += timing.backend_sync_seconds;
    totals.output_copy_seconds += timing.output_copy_seconds;
    totals.cleanup_seconds += timing.cleanup_seconds;
    totals.input_copy_bytes += timing.input_copy_bytes;
    totals.output_copy_bytes += timing.output_copy_bytes;
    for (const GGMLRunnerCopyProfile& copy : timing.copy_breakdown) {
        const std::string key = copy.category + ":" + copy.direction;
        totals.copy_bytes_by_category[key] += copy.bytes;
        totals.copy_seconds_by_category[key] += copy.seconds;
        totals.copy_count_by_category[key] += 1;
    }
    totals.runner_count += 1;
}

static void add_runner_timing_totals(RunnerTimingTotals& totals, const RunnerTimingTotals& timing) {
    totals.offload_params_seconds += timing.offload_params_seconds;
    totals.alloc_compute_buffer_seconds += timing.alloc_compute_buffer_seconds;
    totals.graph_build_seconds += timing.graph_build_seconds;
    totals.graph_alloc_seconds += timing.graph_alloc_seconds;
    totals.input_copy_seconds += timing.input_copy_seconds;
    totals.input_copy_submit_seconds += timing.input_copy_submit_seconds;
    totals.input_sync_seconds += timing.input_sync_seconds;
    totals.backend_compute_seconds += timing.backend_compute_seconds;
    totals.backend_compute_submit_seconds += timing.backend_compute_submit_seconds;
    totals.backend_sync_seconds += timing.backend_sync_seconds;
    totals.output_copy_seconds += timing.output_copy_seconds;
    totals.cleanup_seconds += timing.cleanup_seconds;
    totals.input_copy_bytes += timing.input_copy_bytes;
    totals.output_copy_bytes += timing.output_copy_bytes;
    for (const auto& item : timing.copy_bytes_by_category) {
        totals.copy_bytes_by_category[item.first] += item.second;
    }
    for (const auto& item : timing.copy_seconds_by_category) {
        totals.copy_seconds_by_category[item.first] += item.second;
    }
    for (const auto& item : timing.copy_count_by_category) {
        totals.copy_count_by_category[item.first] += item.second;
    }
    totals.runner_count += timing.runner_count;
}

static void print_seconds_vector(const char* label, const std::vector<double>& values) {
    std::cout << " " << label << "=";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << values[i];
    }
}

static void print_top_block_timings(const char* label,
                                    const std::vector<BlockTimingTotals>& blocks,
                                    double BlockTimingTotals::*field) {
    std::vector<std::pair<int, double>> values;
    values.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        values.emplace_back(static_cast<int>(i), blocks[i].*field);
    }
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::cout << " " << label << "=";
    const size_t count = std::min<size_t>(10, values.size());
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << values[i].first << ":" << values[i].second;
    }
}

static void print_copy_breakdown(const char* label, const RunnerTimingTotals& totals) {
    std::vector<std::pair<std::string, uint64_t>> values(totals.copy_bytes_by_category.begin(),
                                                         totals.copy_bytes_by_category.end());
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::cout << " " << label << "=";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        const std::string& key = values[i].first;
        const double seconds = totals.copy_seconds_by_category.count(key) ? totals.copy_seconds_by_category.at(key) : 0.0;
        const uint64_t count = totals.copy_count_by_category.count(key) ? totals.copy_count_by_category.at(key) : 0;
        std::cout << key << ":" << values[i].second << ":" << seconds << ":" << count;
    }
}

static bool has_tensor(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name);
static float max_abs_diff(const Tensor& a, const Tensor& b, float* mean_abs);

static bool compare_tensor_with_tolerance(const Tensor& a,
                                          const Tensor& b,
                                          float tolerance,
                                          const std::string& label,
                                          float& max_diff,
                                          float& mean_diff) {
    max_diff = max_abs_diff(a, b, &mean_diff);
    std::cout << label << ": max_diff=" << max_diff
              << " mean_diff=" << mean_diff
              << " tolerance=" << tolerance << "\n";
    return max_diff <= tolerance;
}

static DiffStats diff_stats(const Tensor& a, const Tensor& ref) {
    if (a.shape != ref.shape || a.data.size() != ref.data.size()) {
        throw std::runtime_error("comparison shape mismatch");
    }
    DiffStats stats;
    double sum_diff = 0.0;
    double sum_ref = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        if (!std::isfinite(a.data[i])) {
            ++stats.nonfinite_actual;
        }
        if (!std::isfinite(ref.data[i])) {
            ++stats.nonfinite_ref;
        }
        if (!std::isfinite(a.data[i]) || !std::isfinite(ref.data[i])) {
            continue;
        }
        const float d = std::fabs(a.data[i] - ref.data[i]);
        const float r = std::fabs(ref.data[i]);
        if (d > stats.max_diff) {
            stats.max_diff = d;
            stats.max_index = i;
        }
        stats.max_ref = std::max(stats.max_ref, r);
        sum_diff += d;
        sum_ref += r;
        ++stats.finite_count;
    }
    if (stats.finite_count > 0) {
        stats.mean_diff = static_cast<float>(sum_diff / static_cast<double>(stats.finite_count));
        stats.mean_ref = static_cast<float>(sum_ref / static_cast<double>(stats.finite_count));
    } else if (!a.data.empty()) {
        stats.mean_diff = std::numeric_limits<float>::infinity();
        stats.mean_ref = std::numeric_limits<float>::infinity();
    }
    constexpr float eps = 1.0e-12f;
    stats.rel_max = stats.max_diff / std::max(stats.max_ref, eps);
    stats.rel_mean = stats.mean_diff / std::max(stats.mean_ref, eps);
    return stats;
}

static void print_diff_stats(const std::string& label, int block_index, const DiffStats& stats) {
    std::cout << label;
    if (block_index >= 0) {
        std::cout << " block=" << block_index;
    }
    std::cout << " max_diff=" << stats.max_diff
              << " mean_diff=" << stats.mean_diff
              << " max_ref=" << stats.max_ref
              << " mean_ref=" << stats.mean_ref
              << " rel_max=" << stats.rel_max
              << " rel_mean=" << stats.rel_mean
              << " max_index=" << stats.max_index
              << " finite_count=" << stats.finite_count
              << " nonfinite_actual=" << stats.nonfinite_actual
              << " nonfinite_ref=" << stats.nonfinite_ref
              << "\n";
}

static bool has_nonfinite(const DiffStats& stats) {
    return stats.nonfinite_actual != 0 || stats.nonfinite_ref != 0;
}

static sd::Tensor<float> to_sd_2d(const Tensor& t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("to_sd_2d rank mismatch");
    }
    return sd::Tensor<float>({dim(t, 1), dim(t, 0)}, t.data);
}

static sd::Tensor<float> to_sd_1d(const Tensor& t) {
    if (t.shape.size() != 1) {
        throw std::runtime_error("to_sd_1d rank mismatch");
    }
    return sd::Tensor<float>({dim(t, 0)}, t.data);
}

static TensorInput to_input_2d(const Tensor& t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("to_input_2d rank mismatch");
    }
    return TensorInput{{dim(t, 1), dim(t, 0)}, t.data.data()};
}

static TensorInput to_input_2d(const TensorInput& t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("to_input_2d borrowed rank mismatch");
    }
    return TensorInput{{t.shape[1], t.shape[0]}, t.data, t.bf16_data, t.dtype};
}

static TensorInput to_input_1d(const Tensor& t) {
    if (t.shape.size() != 1) {
        throw std::runtime_error("to_input_1d rank mismatch");
    }
    return TensorInput{{dim(t, 0)}, t.data.data()};
}

static TensorInput to_input_1d(const TensorInput& t) {
    if (t.shape.size() != 1) {
        throw std::runtime_error("to_input_1d borrowed rank mismatch");
    }
    return TensorInput{{t.shape[0]}, t.data, t.bf16_data, t.dtype};
}

static TensorInput to_input_bsh_as_csb_view(const Tensor& t) {
    if (t.shape.size() != 3) {
        throw std::runtime_error("to_input_bsh_as_csb_view rank mismatch");
    }
    return TensorInput{{dim(t, 2), dim(t, 1), dim(t, 0)}, t.data.data()};
}

static TensorInput to_input_bh_as_hb_view(const Tensor& t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("to_input_bh_as_hb_view rank mismatch");
    }
    return TensorInput{{dim(t, 1), dim(t, 0)}, t.data.data()};
}

static sd::Tensor<float> to_sd_bsh_as_csb(const Tensor& t) {
    if (t.shape.size() != 3) {
        throw std::runtime_error("to_sd_bsh_as_csb rank mismatch");
    }
    const int64_t bsz = dim(t, 0);
    const int64_t seq = dim(t, 1);
    const int64_t hidden = dim(t, 2);
    sd::Tensor<float> out({hidden, seq, bsz});
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t h = 0; h < hidden; ++h) {
                out.values()[static_cast<size_t>(h + hidden * (s + seq * b))] = t.data[idx3(t, b, s, h)];
            }
        }
    }
    return out;
}

static sd::Tensor<float> to_sd_bh_as_hb(const Tensor& t) {
    if (t.shape.size() != 2) {
        throw std::runtime_error("to_sd_bh_as_hb rank mismatch");
    }
    const int64_t bsz = dim(t, 0);
    const int64_t hidden = dim(t, 1);
    sd::Tensor<float> out({hidden, bsz});
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t h = 0; h < hidden; ++h) {
            out.values()[static_cast<size_t>(h + hidden * b)] = t.data[idx2(t, b, h)];
        }
    }
    return out;
}

static sd::Tensor<float> lens_freqs_to_rope_pe(const Tensor& freqs) {
    if (freqs.shape.size() != 3 || dim(freqs, 2) != 2) {
        throw std::runtime_error("Lens RoPE freqs must have shape [S,D/2,2]");
    }
    const int64_t seq = dim(freqs, 0);
    const int64_t half = dim(freqs, 1);
    sd::Tensor<float> pe({2, 2, half, seq});
    auto set = [&](int64_t s, int64_t d, int64_t a, int64_t b, float value) {
        pe.values()[static_cast<size_t>(a + 2 * (b + 2 * (d + half * s)))] = value;
    };
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t d = 0; d < half; ++d) {
            const float cos_v = freqs.data[static_cast<size_t>((s * half + d) * 2 + 0)];
            const float sin_v = freqs.data[static_cast<size_t>((s * half + d) * 2 + 1)];
            set(s, d, 0, 0, cos_v);
            set(s, d, 1, 0, -sin_v);
            set(s, d, 0, 1, sin_v);
            set(s, d, 1, 1, cos_v);
        }
    }
    return pe;
}

static Tensor from_sd_csb_as_bsh(const sd::Tensor<float>& t) {
    if (t.shape().size() < 2 || t.shape().size() > 4 ||
        (t.shape().size() == 4 && t.shape()[3] != 1)) {
        throw std::runtime_error("from_sd_csb_as_bsh rank mismatch shape=" + sd::tensor_shape_to_string(t.shape()));
    }
    const int64_t hidden = t.shape()[0];
    const int64_t seq = t.shape()[1];
    const int64_t bsz = t.shape().size() >= 3 ? t.shape()[2] : 1;
    Tensor out{{bsz, seq, hidden}, std::vector<float>(static_cast<size_t>(bsz * seq * hidden))};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t h = 0; h < hidden; ++h) {
                out.data[idx3(out, b, s, h)] = t.values()[static_cast<size_t>(h + hidden * (s + seq * b))];
            }
        }
    }
    return out;
}

static Tensor from_sd_hb_as_bh(const sd::Tensor<float>& t) {
    if (t.shape().size() == 1) {
        const int64_t hidden = t.shape()[0];
        Tensor out{{1, hidden}, std::vector<float>(static_cast<size_t>(hidden))};
        for (int64_t h = 0; h < hidden; ++h) {
            out.data[static_cast<size_t>(h)] = t.values()[static_cast<size_t>(h)];
        }
        return out;
    }
    if (t.shape().size() != 2) {
        throw std::runtime_error("from_sd_hb_as_bh rank mismatch shape=" + sd::tensor_shape_to_string(t.shape()));
    }
    const int64_t hidden = t.shape()[0];
    const int64_t bsz = t.shape()[1];
    Tensor out{{bsz, hidden}, std::vector<float>(static_cast<size_t>(bsz * hidden))};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t h = 0; h < hidden; ++h) {
            out.data[idx2(out, b, h)] = t.values()[static_cast<size_t>(h + hidden * b)];
        }
    }
    return out;
}

class LensTopCudaRunner : public GGMLRunner {
    std::unordered_map<std::string, sd::Tensor<float>> tensors;
    std::unordered_map<std::string, TensorInput> borrowed_tensors;
    std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors;
    bool alias_resident_tensors = false;

    ggml_tensor* input(const std::string& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return make_input(it->second);
        }
        auto borrowed = borrowed_tensors.find(name);
        if (borrowed != borrowed_tensors.end()) {
            return make_input_data_named<float>(borrowed->second.shape, borrowed->second.data, name);
        }
        auto resident = resident_tensors.find(name);
        if (resident != resident_tensors.end()) {
            return alias_resident_tensors ? make_backend_input_alias(*resident->second)
                                          : make_backend_input(*resident->second);
        }
        throw std::runtime_error("missing CUDA top runner tensor: " + name);
    }

    ggml_tensor* linear(ggml_tensor* x, const std::string& w_name, const std::string& b_name = "") {
        ggml_tensor* w = input(w_name);
        ggml_tensor* b = b_name.empty() ? nullptr : input(b_name);
        return ggml_ext_linear(compute_ctx, x, w, b);
    }

    ggml_tensor* rms_norm_weighted(ggml_tensor* x, const std::string& weight_name, float eps) {
        x = ggml_rms_norm(compute_ctx, x, eps);
        return ggml_mul(compute_ctx, x, input(weight_name));
    }

public:
    LensTopCudaRunner(ggml_backend_t backend,
                      std::unordered_map<std::string, sd::Tensor<float>> tensors)
        : GGMLRunner(backend, false),
          tensors(std::move(tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensTopCudaRunner(ggml_backend_t backend,
                      std::unordered_map<std::string, TensorInput> borrowed_tensors)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensTopCudaRunner(ggml_backend_t backend,
                      std::unordered_map<std::string, TensorInput> borrowed_tensors,
                      std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors,
                      bool alias_resident_tensors = false)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)),
          resident_tensors(std::move(resident_tensors)),
          alias_resident_tensors(alias_resident_tensors) {
        set_build_in_tensors_enabled(false);
    }

    std::string get_desc() override {
        return "LensTopCudaRunner";
    }

    std::optional<sd::Tensor<float>> compute_top(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(1024);
            ggml_tensor* hidden = linear(input("input.packed"), "img_in.weight", "img_in.bias");
            ggml_tensor* f0 = rms_norm_weighted(input("feature_0"), "txt_norm.0.weight", 1.0e-5f);
            ggml_tensor* f1 = rms_norm_weighted(input("feature_1"), "txt_norm.1.weight", 1.0e-5f);
            ggml_tensor* f2 = rms_norm_weighted(input("feature_2"), "txt_norm.2.weight", 1.0e-5f);
            ggml_tensor* f3 = rms_norm_weighted(input("feature_3"), "txt_norm.3.weight", 1.0e-5f);
            ggml_tensor* features01 = ggml_concat(compute_ctx, f0, f1, 0);
            ggml_tensor* features23 = ggml_concat(compute_ctx, f2, f3, 0);
            ggml_tensor* features = ggml_concat(compute_ctx, features01, features23, 0);
            ggml_tensor* encoder = linear(features, "txt_in.weight", "txt_in.bias");
            ggml_tensor* combined = ggml_concat(compute_ctx, hidden, encoder, 1);
            ggml_build_forward_expand(gf, combined);
            return gf;
        };
        return compute<float>(get_graph, n_threads, true);
    }

    std::unique_ptr<GgmlBackendTensorResource> compute_top_resource_alias(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(1024);
            ggml_tensor* hidden = linear(input("input.packed"), "img_in.weight", "img_in.bias");
            ggml_tensor* f0 = rms_norm_weighted(input("feature_0"), "txt_norm.0.weight", 1.0e-5f);
            ggml_tensor* f1 = rms_norm_weighted(input("feature_1"), "txt_norm.1.weight", 1.0e-5f);
            ggml_tensor* f2 = rms_norm_weighted(input("feature_2"), "txt_norm.2.weight", 1.0e-5f);
            ggml_tensor* f3 = rms_norm_weighted(input("feature_3"), "txt_norm.3.weight", 1.0e-5f);
            ggml_tensor* features01 = ggml_concat(compute_ctx, f0, f1, 0);
            ggml_tensor* features23 = ggml_concat(compute_ctx, f2, f3, 0);
            ggml_tensor* features = ggml_concat(compute_ctx, features01, features23, 0);
            ggml_tensor* encoder = linear(features, "txt_in.weight", "txt_in.bias");
            ggml_tensor* combined = ggml_concat(compute_ctx, hidden, encoder, 1);
            ggml_build_forward_expand(gf, combined);
            return gf;
        };
        return compute_to_backend_resource_alias(get_graph, n_threads, "lens_top_dynamic_combined");
    }
};

class LensTimestepCudaRunner : public GGMLRunner {
    std::unordered_map<std::string, sd::Tensor<float>> tensors;
    std::unordered_map<std::string, TensorInput> borrowed_tensors;
    std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors;
    bool alias_resident_tensors = false;

    ggml_tensor* input(const std::string& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return make_input(it->second);
        }
        auto borrowed = borrowed_tensors.find(name);
        if (borrowed != borrowed_tensors.end()) {
            return make_input_data_named<float>(borrowed->second.shape, borrowed->second.data, name);
        }
        auto resident = resident_tensors.find(name);
        if (resident != resident_tensors.end()) {
            return alias_resident_tensors ? make_backend_input_alias(*resident->second)
                                          : make_backend_input(*resident->second);
        }
        throw std::runtime_error("missing CUDA timestep runner tensor: " + name);
    }

    ggml_tensor* linear(ggml_tensor* x, const std::string& w_name, const std::string& b_name = "") {
        ggml_tensor* w = input(w_name);
        ggml_tensor* b = b_name.empty() ? nullptr : input(b_name);
        return ggml_ext_linear(compute_ctx, x, w, b);
    }

public:
    LensTimestepCudaRunner(ggml_backend_t backend,
                           std::unordered_map<std::string, sd::Tensor<float>> tensors)
        : GGMLRunner(backend, false),
          tensors(std::move(tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensTimestepCudaRunner(ggml_backend_t backend,
                           std::unordered_map<std::string, TensorInput> borrowed_tensors)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensTimestepCudaRunner(ggml_backend_t backend,
                           std::unordered_map<std::string, TensorInput> borrowed_tensors,
                           std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors,
                           bool alias_resident_tensors = false)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)),
          resident_tensors(std::move(resident_tensors)),
          alias_resident_tensors(alias_resident_tensors) {
        set_build_in_tensors_enabled(false);
    }

    std::string get_desc() override {
        return "LensTimestepCudaRunner";
    }

    std::optional<sd::Tensor<float>> compute_temb(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(512);
            ggml_tensor* x = linear(input("input.timestep_proj"),
                                    "time_text_embed.timestep_embedder.linear_1.weight",
                                    "time_text_embed.timestep_embedder.linear_1.bias");
            x = ggml_silu(compute_ctx, x);
            x = linear(x,
                       "time_text_embed.timestep_embedder.linear_2.weight",
                       "time_text_embed.timestep_embedder.linear_2.bias");
            ggml_build_forward_expand(gf, x);
            return gf;
        };
        return compute<float>(get_graph, n_threads, true);
    }

    std::unique_ptr<GgmlBackendTensorResource> compute_temb_resource_alias(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(512);
            ggml_tensor* x = linear(input("input.timestep_proj"),
                                    "time_text_embed.timestep_embedder.linear_1.weight",
                                    "time_text_embed.timestep_embedder.linear_1.bias");
            x = ggml_silu(compute_ctx, x);
            x = linear(x,
                       "time_text_embed.timestep_embedder.linear_2.weight",
                       "time_text_embed.timestep_embedder.linear_2.bias");
            ggml_build_forward_expand(gf, x);
            return gf;
        };
        return compute_to_backend_resource_alias(get_graph, n_threads, "lens_step_dynamic_temb");
    }
};

class LensBlockCudaRunner : public GGMLRunner {
    std::unordered_map<std::string, sd::Tensor<float>> tensors;
    std::unordered_map<std::string, TensorInput> borrowed_tensors;
    std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors;
    bool alias_resident_tensors = false;
    int64_t img_seq = 0;
    int64_t txt_seq = 0;
    int64_t hidden = 1536;
    int64_t heads = 24;
    int64_t head_dim = 64;
    LensAttentionMode attention_mode = LensAttentionMode::RegularF32;
    bool use_precomputed_modulation = false;

    ggml_tensor* input(const std::string& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return make_input(it->second);
        }
        auto borrowed = borrowed_tensors.find(name);
        if (borrowed != borrowed_tensors.end()) {
            return make_input_data_named<float>(borrowed->second.shape, borrowed->second.data, name);
        }
        auto resident = resident_tensors.find(name);
        if (resident != resident_tensors.end()) {
            return alias_resident_tensors ? make_backend_input_alias(*resident->second)
                                          : make_backend_input(*resident->second);
        }
        throw std::runtime_error("missing CUDA runner tensor: " + name);
    }

    bool has_resident_input(const std::string& name) const {
        return resident_tensors.find(name) != resident_tensors.end();
    }

    ggml_tensor* linear(ggml_tensor* x, const std::string& w_name, const std::string& b_name = "") {
        ggml_tensor* w = input(w_name);
        ggml_tensor* b = b_name.empty() ? nullptr : input(b_name);
        return ggml_ext_linear(compute_ctx, x, w, b);
    }

    ggml_tensor* rms_norm_weighted(ggml_tensor* x, const std::string& weight_name, float eps = 1.0e-6f) {
        x = ggml_rms_norm(compute_ctx, x, eps);
        return ggml_mul(compute_ctx, x, input(weight_name));
    }

    ggml_tensor* mod_slice(ggml_tensor* mod, int64_t offset) {
        return ggml_view_2d(compute_ctx,
                            mod,
                            hidden,
                            mod->ne[1],
                            mod->nb[1],
                            offset * mod->nb[0]);
    }

    ggml_tensor* modulate(ggml_tensor* x, ggml_tensor* mod, int64_t offset, ggml_tensor** gate_out) {
        ggml_tensor* shift = mod_slice(mod, offset + 0 * hidden);
        ggml_tensor* scale = mod_slice(mod, offset + 1 * hidden);
        ggml_tensor* gate = mod_slice(mod, offset + 2 * hidden);
        scale = ggml_reshape_3d(compute_ctx, scale, hidden, 1, scale->ne[1]);
        shift = ggml_reshape_3d(compute_ctx, shift, hidden, 1, shift->ne[1]);
        gate = ggml_reshape_3d(compute_ctx, gate, hidden, 1, gate->ne[1]);
        x = ggml_add(compute_ctx, x, ggml_mul(compute_ctx, x, scale));
        x = ggml_add(compute_ctx, x, shift);
        *gate_out = gate;
        return x;
    }

    ggml_tensor* modulate_precomputed(ggml_tensor* x,
                                      const std::string& shift_name,
                                      const std::string& scale_name,
                                      const std::string& gate_name,
                                      ggml_tensor** gate_out) {
        ggml_tensor* shift = input(shift_name);
        ggml_tensor* scale = input(scale_name);
        ggml_tensor* gate = input(gate_name);
        x = ggml_add(compute_ctx, x, ggml_mul(compute_ctx, x, scale));
        x = ggml_add(compute_ctx, x, shift);
        *gate_out = gate;
        return x;
    }

    std::vector<ggml_tensor*> split_qkv(ggml_tensor* qkv, int64_t seq) {
        ggml_tensor* q = ggml_view_4d(compute_ctx,
                                      qkv,
                                      head_dim,
                                      heads,
                                      seq,
                                      qkv->ne[2],
                                      qkv->nb[0] * head_dim,
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      0);
        ggml_tensor* k = ggml_view_4d(compute_ctx,
                                      qkv,
                                      head_dim,
                                      heads,
                                      seq,
                                      qkv->ne[2],
                                      qkv->nb[0] * head_dim,
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      hidden * qkv->nb[0]);
        ggml_tensor* v = ggml_view_4d(compute_ctx,
                                      qkv,
                                      head_dim,
                                      heads,
                                      seq,
                                      qkv->ne[2],
                                      qkv->nb[0] * head_dim,
                                      qkv->nb[1],
                                      qkv->nb[2],
                                      2 * hidden * qkv->nb[0]);
        return {q, k, v};
    }

    ggml_tensor* add_gated(ggml_tensor* x, ggml_tensor* update, ggml_tensor* gate) {
        return ggml_add(compute_ctx, x, ggml_mul(compute_ctx, update, gate));
    }

    ggml_tensor* gate_mlp(ggml_tensor* x,
                          const std::string& w1,
                          const std::string& w2,
                          const std::string& w3) {
        ggml_tensor* a = linear(x, w1);
        ggml_tensor* b = linear(x, w3);
        a = ggml_silu(compute_ctx, a);
        a = ggml_mul(compute_ctx, a, b);
        return linear(a, w2);
    }

public:
    LensBlockCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, sd::Tensor<float>> tensors,
                        int64_t img_seq,
                        int64_t txt_seq,
                        LensAttentionMode attention_mode,
                        bool use_precomputed_modulation = false)
        : GGMLRunner(backend, false),
          tensors(std::move(tensors)),
          img_seq(img_seq),
          txt_seq(txt_seq),
          attention_mode(attention_mode),
          use_precomputed_modulation(use_precomputed_modulation) {
        // Lens late-block activations can exceed F16 range; the current ggml
        // flash-attention helper casts K/V to F16, so keep attention on the
        // CUDA matmul/softmax path for parity.
        flash_attn_enabled = attention_mode == LensAttentionMode::Flash;
        set_build_in_tensors_enabled(attention_mode == LensAttentionMode::Flash);
    }

    LensBlockCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, TensorInput> borrowed_tensors,
                        int64_t img_seq,
                        int64_t txt_seq,
                        LensAttentionMode attention_mode,
                        bool use_precomputed_modulation = false)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)),
        img_seq(img_seq),
        txt_seq(txt_seq),
        attention_mode(attention_mode),
        use_precomputed_modulation(use_precomputed_modulation) {
        flash_attn_enabled = attention_mode == LensAttentionMode::Flash;
        set_build_in_tensors_enabled(attention_mode == LensAttentionMode::Flash);
    }

    LensBlockCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, TensorInput> borrowed_tensors,
                        std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors,
                        int64_t img_seq,
                        int64_t txt_seq,
                        LensAttentionMode attention_mode,
                        bool alias_resident_tensors = false,
                        bool use_precomputed_modulation = false)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)),
          resident_tensors(std::move(resident_tensors)),
          alias_resident_tensors(alias_resident_tensors),
        img_seq(img_seq),
        txt_seq(txt_seq),
        attention_mode(attention_mode),
        use_precomputed_modulation(use_precomputed_modulation) {
        flash_attn_enabled = attention_mode == LensAttentionMode::Flash;
        set_build_in_tensors_enabled(attention_mode == LensAttentionMode::Flash);
    }

    std::string get_desc() override {
        return "LensBlockCudaRunner";
    }

    void set_dynamic_inputs(std::unordered_map<std::string, TensorInput> dynamic_tensors) {
        for (auto& item : dynamic_tensors) {
            borrowed_tensors[item.first] = item.second;
        }
    }

    std::optional<sd::Tensor<float>> compute_block(int n_threads, bool keep_compute_buffer = false) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(4096);
            GGMLRunnerContext runner_ctx = get_context();
            ggml_tensor* hidden_state = nullptr;
            ggml_tensor* encoder_state = nullptr;
            if (has_resident_input("input.combined")) {
                ggml_tensor* combined_input = input("input.combined");
                hidden_state = ggml_view_3d(compute_ctx,
                                            combined_input,
                                            hidden,
                                            img_seq,
                                            combined_input->ne[2],
                                            combined_input->nb[1],
                                            combined_input->nb[2],
                                            0);
                encoder_state = ggml_view_3d(compute_ctx,
                                             combined_input,
                                             hidden,
                                             txt_seq,
                                             combined_input->ne[2],
                                             combined_input->nb[1],
                                             combined_input->nb[2],
                                             img_seq * combined_input->nb[1]);
            } else {
                hidden_state = input("input.hidden");
                encoder_state = input("input.encoder");
            }

            ggml_tensor* img_mod = nullptr;
            ggml_tensor* txt_mod = nullptr;
            if (!use_precomputed_modulation) {
                ggml_tensor* temb = input("input.temb");
                img_mod = linear(ggml_silu(compute_ctx, temb), "img_mod.1.weight", "img_mod.1.bias");
                txt_mod = linear(ggml_silu(compute_ctx, temb), "txt_mod.1.weight", "txt_mod.1.bias");
            }

            ggml_tensor* img_gate1 = nullptr;
            ggml_tensor* txt_gate1 = nullptr;
            ggml_tensor* img_modulated = use_precomputed_modulation
                                             ? modulate_precomputed(rms_norm_weighted(hidden_state, "img_norm1.weight"),
                                                                    "input.img_shift1",
                                                                    "input.img_scale1",
                                                                    "input.img_gate1",
                                                                    &img_gate1)
                                             : modulate(rms_norm_weighted(hidden_state, "img_norm1.weight"), img_mod, 0, &img_gate1);
            ggml_tensor* txt_modulated = use_precomputed_modulation
                                             ? modulate_precomputed(rms_norm_weighted(encoder_state, "txt_norm1.weight"),
                                                                    "input.txt_shift1",
                                                                    "input.txt_scale1",
                                                                    "input.txt_gate1",
                                                                    &txt_gate1)
                                             : modulate(rms_norm_weighted(encoder_state, "txt_norm1.weight"), txt_mod, 0, &txt_gate1);

            auto img_qkv = split_qkv(linear(img_modulated, "attn.img_qkv.weight", "attn.img_qkv.bias"), img_seq);
            auto txt_qkv = split_qkv(linear(txt_modulated, "attn.txt_qkv.weight", "attn.txt_qkv.bias"), txt_seq);
            ggml_tensor* img_q = rms_norm_weighted(img_qkv[0], "attn.norm_q.weight");
            ggml_tensor* img_k = rms_norm_weighted(img_qkv[1], "attn.norm_k.weight");
            ggml_tensor* txt_q = rms_norm_weighted(txt_qkv[0], "attn.norm_added_q.weight");
            ggml_tensor* txt_k = rms_norm_weighted(txt_qkv[1], "attn.norm_added_k.weight");

            ggml_tensor* q = ggml_concat(compute_ctx, img_q, txt_q, 2);
            ggml_tensor* k = ggml_concat(compute_ctx, img_k, txt_k, 2);
            ggml_tensor* v = ggml_concat(compute_ctx, img_qkv[2], txt_qkv[2], 2);
            ggml_tensor* pe = ggml_concat(compute_ctx, input("input.img_pe"), input("input.txt_pe"), 3);
            ggml_tensor* attn = Rope::attention(&runner_ctx, q, k, v, pe, nullptr, 1.0f, true);

            ggml_tensor* img_attn = ggml_view_3d(compute_ctx,
                                                 attn,
                                                 hidden,
                                                 img_seq,
                                                 attn->ne[2],
                                                 attn->nb[1],
                                                 attn->nb[2],
                                                 0);
            ggml_tensor* txt_attn = ggml_view_3d(compute_ctx,
                                                 attn,
                                                 hidden,
                                                 txt_seq,
                                                 attn->ne[2],
                                                 attn->nb[1],
                                                 attn->nb[2],
                                                 img_seq * attn->nb[1]);
            img_attn = linear(img_attn, "attn.to_out.0.weight", "attn.to_out.0.bias");
            txt_attn = linear(txt_attn, "attn.to_add_out.weight", "attn.to_add_out.bias");
            hidden_state = add_gated(hidden_state, img_attn, img_gate1);
            encoder_state = add_gated(encoder_state, txt_attn, txt_gate1);

            ggml_tensor* img_gate2 = nullptr;
            ggml_tensor* txt_gate2 = nullptr;
            ggml_tensor* img_mlp_in = use_precomputed_modulation
                                          ? modulate_precomputed(rms_norm_weighted(hidden_state, "img_norm2.weight"),
                                                                 "input.img_shift2",
                                                                 "input.img_scale2",
                                                                 "input.img_gate2",
                                                                 &img_gate2)
                                          : modulate(rms_norm_weighted(hidden_state, "img_norm2.weight"), img_mod, 3 * hidden, &img_gate2);
            hidden_state = add_gated(hidden_state,
                                     gate_mlp(img_mlp_in, "img_mlp.w1.weight", "img_mlp.w2.weight", "img_mlp.w3.weight"),
                                     img_gate2);
            ggml_tensor* txt_mlp_in = use_precomputed_modulation
                                          ? modulate_precomputed(rms_norm_weighted(encoder_state, "txt_norm2.weight"),
                                                                 "input.txt_shift2",
                                                                 "input.txt_scale2",
                                                                 "input.txt_gate2",
                                                                 &txt_gate2)
                                          : modulate(rms_norm_weighted(encoder_state, "txt_norm2.weight"), txt_mod, 3 * hidden, &txt_gate2);
            encoder_state = add_gated(encoder_state,
                                      gate_mlp(txt_mlp_in, "txt_mlp.w1.weight", "txt_mlp.w2.weight", "txt_mlp.w3.weight"),
                                      txt_gate2);

            ggml_tensor* combined = ggml_concat(compute_ctx, hidden_state, encoder_state, 1);
            ggml_build_forward_expand(gf, combined);
            return gf;
        };
        return compute<float>(get_graph, n_threads, !keep_compute_buffer);
    }

    std::unique_ptr<GgmlBackendTensorResource> compute_block_resource_alias(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(4096);
            GGMLRunnerContext runner_ctx = get_context();
            ggml_tensor* hidden_state = nullptr;
            ggml_tensor* encoder_state = nullptr;
            if (has_resident_input("input.combined")) {
                ggml_tensor* combined_input = input("input.combined");
                hidden_state = ggml_view_3d(compute_ctx,
                                            combined_input,
                                            hidden,
                                            img_seq,
                                            combined_input->ne[2],
                                            combined_input->nb[1],
                                            combined_input->nb[2],
                                            0);
                encoder_state = ggml_view_3d(compute_ctx,
                                             combined_input,
                                             hidden,
                                             txt_seq,
                                             combined_input->ne[2],
                                             combined_input->nb[1],
                                             combined_input->nb[2],
                                             img_seq * combined_input->nb[1]);
            } else {
                hidden_state = input("input.hidden");
                encoder_state = input("input.encoder");
            }

            ggml_tensor* img_mod = nullptr;
            ggml_tensor* txt_mod = nullptr;
            if (!use_precomputed_modulation) {
                ggml_tensor* temb = input("input.temb");
                img_mod = linear(ggml_silu(compute_ctx, temb), "img_mod.1.weight", "img_mod.1.bias");
                txt_mod = linear(ggml_silu(compute_ctx, temb), "txt_mod.1.weight", "txt_mod.1.bias");
            }

            ggml_tensor* img_gate1 = nullptr;
            ggml_tensor* txt_gate1 = nullptr;
            ggml_tensor* img_modulated = use_precomputed_modulation
                                             ? modulate_precomputed(rms_norm_weighted(hidden_state, "img_norm1.weight"),
                                                                    "input.img_shift1",
                                                                    "input.img_scale1",
                                                                    "input.img_gate1",
                                                                    &img_gate1)
                                             : modulate(rms_norm_weighted(hidden_state, "img_norm1.weight"), img_mod, 0, &img_gate1);
            ggml_tensor* txt_modulated = use_precomputed_modulation
                                             ? modulate_precomputed(rms_norm_weighted(encoder_state, "txt_norm1.weight"),
                                                                    "input.txt_shift1",
                                                                    "input.txt_scale1",
                                                                    "input.txt_gate1",
                                                                    &txt_gate1)
                                             : modulate(rms_norm_weighted(encoder_state, "txt_norm1.weight"), txt_mod, 0, &txt_gate1);

            auto img_qkv = split_qkv(linear(img_modulated, "attn.img_qkv.weight", "attn.img_qkv.bias"), img_seq);
            auto txt_qkv = split_qkv(linear(txt_modulated, "attn.txt_qkv.weight", "attn.txt_qkv.bias"), txt_seq);
            ggml_tensor* img_q = rms_norm_weighted(img_qkv[0], "attn.norm_q.weight");
            ggml_tensor* img_k = rms_norm_weighted(img_qkv[1], "attn.norm_k.weight");
            ggml_tensor* txt_q = rms_norm_weighted(txt_qkv[0], "attn.norm_added_q.weight");
            ggml_tensor* txt_k = rms_norm_weighted(txt_qkv[1], "attn.norm_added_k.weight");

            ggml_tensor* q = ggml_concat(compute_ctx, img_q, txt_q, 2);
            ggml_tensor* k = ggml_concat(compute_ctx, img_k, txt_k, 2);
            ggml_tensor* v = ggml_concat(compute_ctx, img_qkv[2], txt_qkv[2], 2);
            ggml_tensor* pe = ggml_concat(compute_ctx, input("input.img_pe"), input("input.txt_pe"), 3);
            ggml_tensor* attn = Rope::attention(&runner_ctx, q, k, v, pe, nullptr, 1.0f, true);

            ggml_tensor* img_attn = ggml_view_3d(compute_ctx,
                                                 attn,
                                                 hidden,
                                                 img_seq,
                                                 attn->ne[2],
                                                 attn->nb[1],
                                                 attn->nb[2],
                                                 0);
            ggml_tensor* txt_attn = ggml_view_3d(compute_ctx,
                                                 attn,
                                                 hidden,
                                                 txt_seq,
                                                 attn->ne[2],
                                                 attn->nb[1],
                                                 attn->nb[2],
                                                 img_seq * attn->nb[1]);
            img_attn = linear(img_attn, "attn.to_out.0.weight", "attn.to_out.0.bias");
            txt_attn = linear(txt_attn, "attn.to_add_out.weight", "attn.to_add_out.bias");
            hidden_state = add_gated(hidden_state, img_attn, img_gate1);
            encoder_state = add_gated(encoder_state, txt_attn, txt_gate1);

            ggml_tensor* img_gate2 = nullptr;
            ggml_tensor* txt_gate2 = nullptr;
            ggml_tensor* img_mlp_in = use_precomputed_modulation
                                          ? modulate_precomputed(rms_norm_weighted(hidden_state, "img_norm2.weight"),
                                                                 "input.img_shift2",
                                                                 "input.img_scale2",
                                                                 "input.img_gate2",
                                                                 &img_gate2)
                                          : modulate(rms_norm_weighted(hidden_state, "img_norm2.weight"), img_mod, 3 * hidden, &img_gate2);
            hidden_state = add_gated(hidden_state,
                                     gate_mlp(img_mlp_in, "img_mlp.w1.weight", "img_mlp.w2.weight", "img_mlp.w3.weight"),
                                     img_gate2);
            ggml_tensor* txt_mlp_in = use_precomputed_modulation
                                          ? modulate_precomputed(rms_norm_weighted(encoder_state, "txt_norm2.weight"),
                                                                 "input.txt_shift2",
                                                                 "input.txt_scale2",
                                                                 "input.txt_gate2",
                                                                 &txt_gate2)
                                          : modulate(rms_norm_weighted(encoder_state, "txt_norm2.weight"), txt_mod, 3 * hidden, &txt_gate2);
            encoder_state = add_gated(encoder_state,
                                      gate_mlp(txt_mlp_in, "txt_mlp.w1.weight", "txt_mlp.w2.weight", "txt_mlp.w3.weight"),
                                      txt_gate2);

            ggml_tensor* combined = ggml_concat(compute_ctx, hidden_state, encoder_state, 1);
            ggml_build_forward_expand(gf, combined);
            return gf;
        };
        return compute_to_backend_resource_alias(get_graph, n_threads, "lens_block_dynamic_combined");
    }
};

class LensAttentionMicrobenchRunner : public GGMLRunner {
    const LensAttentionFixture& fixture;
    LensAttentionMode attention_mode = LensAttentionMode::RegularF32;
    bool use_mask = true;

public:
    LensAttentionMicrobenchRunner(ggml_backend_t backend,
                                  const LensAttentionFixture& fixture,
                                  LensAttentionMode attention_mode,
                                  bool use_mask)
        : GGMLRunner(backend, false),
          fixture(fixture),
          attention_mode(attention_mode),
          use_mask(use_mask) {
        flash_attn_enabled = attention_mode == LensAttentionMode::Flash;
        set_build_in_tensors_enabled(attention_mode == LensAttentionMode::Flash);
    }

    std::string get_desc() override {
        return "LensAttentionMicrobenchRunner";
    }

    std::optional<sd::Tensor<float>> compute_attention(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(256);
            ggml_tensor* q = make_input_data_named<float>(
                {fixture.d, fixture.s, static_cast<int64_t>(fixture.b) * fixture.h},
                fixture.q.data(),
                "input.q");
            ggml_tensor* k = make_input_data_named<float>(
                {fixture.d, fixture.s, static_cast<int64_t>(fixture.b) * fixture.h},
                fixture.k.data(),
                "input.k");
            ggml_tensor* v = make_input_data_named<float>(
                {fixture.d, fixture.h, fixture.s, fixture.b},
                fixture.v.data(),
                "input.v");
            ggml_tensor* mask = nullptr;
            if (use_mask && !fixture.mask_expanded.empty()) {
                mask = make_input_data_named<float>(
                    {fixture.s, fixture.s, static_cast<int64_t>(fixture.b) * fixture.h},
                    fixture.mask_expanded.data(),
                    "input.attention_mask_expanded");
            }
            ggml_tensor* out = ggml_ext_attention_ext(compute_ctx,
                                                      runtime_backend,
                                                      q,
                                                      k,
                                                      v,
                                                      fixture.h,
                                                      mask,
                                                      true,
                                                      attention_mode == LensAttentionMode::Flash,
                                                      1.0f);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return compute<float>(get_graph, n_threads, true);
    }
};

class LensFinalCudaRunner : public GGMLRunner {
    std::unordered_map<std::string, sd::Tensor<float>> tensors;
    std::unordered_map<std::string, TensorInput> borrowed_tensors;
    std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors;
    bool alias_resident_tensors = false;
    int64_t hidden = 1536;
    int64_t img_seq = 0;

    ggml_tensor* input(const std::string& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return make_input(it->second);
        }
        auto borrowed = borrowed_tensors.find(name);
        if (borrowed != borrowed_tensors.end()) {
            return make_input_data_named<float>(borrowed->second.shape, borrowed->second.data, name);
        }
        auto resident = resident_tensors.find(name);
        if (resident != resident_tensors.end()) {
            return alias_resident_tensors ? make_backend_input_alias(*resident->second)
                                          : make_backend_input(*resident->second);
        }
        throw std::runtime_error("missing CUDA final runner tensor: " + name);
    }

    bool has_resident_input(const std::string& name) const {
        return resident_tensors.find(name) != resident_tensors.end();
    }

    ggml_tensor* linear(ggml_tensor* x, const std::string& w_name, const std::string& b_name = "") {
        ggml_tensor* w = input(w_name);
        ggml_tensor* b = b_name.empty() ? nullptr : input(b_name);
        return ggml_ext_linear(compute_ctx, x, w, b);
    }

    ggml_tensor* mod_slice(ggml_tensor* mod, int64_t offset) {
        return ggml_view_2d(compute_ctx,
                            mod,
                            hidden,
                            mod->ne[1],
                            mod->nb[1],
                            offset * mod->nb[0]);
    }

public:
    LensFinalCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, sd::Tensor<float>> tensors)
        : GGMLRunner(backend, false),
          tensors(std::move(tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensFinalCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, TensorInput> borrowed_tensors)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)) {
        set_build_in_tensors_enabled(false);
    }

    LensFinalCudaRunner(ggml_backend_t backend,
                        std::unordered_map<std::string, TensorInput> borrowed_tensors,
                        std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_tensors,
                        int64_t img_seq,
                        bool alias_resident_tensors = false)
        : GGMLRunner(backend, false),
          borrowed_tensors(std::move(borrowed_tensors)),
          resident_tensors(std::move(resident_tensors)),
          alias_resident_tensors(alias_resident_tensors),
          img_seq(img_seq) {
        set_build_in_tensors_enabled(false);
    }

    std::string get_desc() override {
        return "LensFinalCudaRunner";
    }

    std::optional<sd::Tensor<float>> compute_final(int n_threads) {
        auto get_graph = [&]() {
            ggml_cgraph* gf = new_graph_custom(1024);
            ggml_tensor* hidden_state = nullptr;
            if (has_resident_input("input.combined")) {
                ggml_tensor* combined_input = input("input.combined");
                hidden_state = ggml_view_3d(compute_ctx,
                                            combined_input,
                                            hidden,
                                            img_seq,
                                            combined_input->ne[2],
                                            combined_input->nb[1],
                                            combined_input->nb[2],
                                            0);
            } else {
                hidden_state = input("input.hidden");
            }
            ggml_tensor* temb = input("input.temb");
            ggml_tensor* emb = linear(ggml_silu(compute_ctx, temb), "norm_out.linear.weight", "norm_out.linear.bias");
            ggml_tensor* scale = ggml_reshape_3d(compute_ctx, mod_slice(emb, 0), hidden, 1, emb->ne[1]);
            ggml_tensor* shift = ggml_reshape_3d(compute_ctx, mod_slice(emb, hidden), hidden, 1, emb->ne[1]);
            hidden_state = ggml_norm(compute_ctx, hidden_state, 1.0e-6f);
            hidden_state = ggml_add(compute_ctx, hidden_state, ggml_mul(compute_ctx, hidden_state, scale));
            hidden_state = ggml_add(compute_ctx, hidden_state, shift);
            ggml_tensor* out = linear(hidden_state, "proj_out.weight", "proj_out.bias");
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return compute<float>(get_graph, n_threads, true);
    }
};

static bool verify_external_schedule_tensors(const std::unordered_map<std::string, Tensor>& tensors,
                                             int steps,
                                             int image_seq_len,
                                             float tolerance) {
    if (!has_tensor(tensors, "external.sigmas") ||
        !has_tensor(tensors, "external.timesteps") ||
        !has_tensor(tensors, "external.schedule_kind")) {
        std::cerr << "external flow fixture missing scheduler metadata tensors\n";
        return false;
    }
    const Tensor& kind = need(tensors, "external.schedule_kind");
    if (kind.data.empty() || static_cast<int>(std::llround(kind.data[0])) != 1) {
        std::cerr << "external flow fixture was not marked as Lens empirical-mu custom-Turbo schedule\n";
        return false;
    }
    const Tensor& sigmas_in = need(tensors, "external.sigmas");
    const Tensor& timesteps_in = need(tensors, "external.timesteps");
    if (sigmas_in.shape.size() != 1 || timesteps_in.shape.size() != 1 ||
        dim(sigmas_in, 0) != steps + 1 || dim(timesteps_in, 0) != steps) {
        std::cerr << "external flow fixture scheduler tensor shape mismatch\n";
        return false;
    }

    sd_lens_schedule_options_t options;
    sd_lens_schedule_options_init(&options);
    options.steps = steps;
    options.image_seq_len = image_seq_len;
    std::vector<float> sigmas(static_cast<size_t>(steps + 1), 0.0f);
    std::vector<float> timesteps(static_cast<size_t>(steps), 0.0f);
    sd_lens_schedule_desc_t desc;
    sd_lens_schedule_desc_init(&desc);
    if (!sd_lens_turbo_build_schedule(&options,
                                      sigmas.data(),
                                      static_cast<uint32_t>(sigmas.size()),
                                      timesteps.data(),
                                      static_cast<uint32_t>(timesteps.size()),
                                      &desc)) {
        std::cerr << "sd_lens_turbo_build_schedule failed during external fixture schedule validation\n";
        return false;
    }
    float max_sigma = 0.0f;
    float max_timestep = 0.0f;
    for (int i = 0; i < steps + 1; ++i) {
        max_sigma = std::max(max_sigma, std::fabs(sigmas_in.data[static_cast<size_t>(i)] - sigmas[static_cast<size_t>(i)]));
    }
    for (int i = 0; i < steps; ++i) {
        max_timestep = std::max(max_timestep, std::fabs(timesteps_in.data[static_cast<size_t>(i)] - timesteps[static_cast<size_t>(i)]));
    }
    std::cout << "Lens external schedule check: steps=" << steps
              << " image_seq_len=" << image_seq_len
              << " mu=" << desc.mu
              << " max_sigma_diff=" << max_sigma
              << " max_timestep_diff=" << max_timestep
              << " schedule_kind=lens_empirical_mu_custom_turbo\n";
    if (max_sigma > tolerance || max_timestep > tolerance) {
        std::cerr << "external flow fixture scheduler metadata does not match native Lens schedule\n";
        return false;
    }
    return true;
}

static bool has_tensor(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
    return tensors.find(name) != tensors.end();
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static Tensor round_tensor_bf16(const Tensor& x) {
    Tensor out = x;
    std::vector<ggml_bf16_t> tmp(out.data.size());
    ggml_fp32_to_bf16_row_ref(out.data.data(), tmp.data(), static_cast<int64_t>(tmp.size()));
    ggml_bf16_to_fp32_row(tmp.data(), out.data.data(), static_cast<int64_t>(tmp.size()));
    return out;
}

static Tensor linear3(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t in = dim(x, 2);
    const int64_t out = dim(weight, 0);
    if (dim(weight, 1) != in || (bias != nullptr && dim(*bias, 0) != out)) {
        throw std::runtime_error("linear3 shape mismatch");
    }
    Tensor y{{bsz, seq, out}, std::vector<float>(static_cast<size_t>(bsz * seq * out), 0.0f)};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t o = 0; o < out; ++o) {
                float acc = bias != nullptr ? bias->data[static_cast<size_t>(o)] : 0.0f;
                for (int64_t i = 0; i < in; ++i) {
                    acc += x.data[idx3(x, b, s, i)] * weight.data[idx2(weight, o, i)];
                }
                y.data[idx3(y, b, s, o)] = acc;
            }
        }
    }
    return y;
}

static Tensor linear2(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    const int64_t bsz = dim(x, 0);
    const int64_t in = dim(x, 1);
    const int64_t out = dim(weight, 0);
    if (dim(weight, 1) != in || (bias != nullptr && dim(*bias, 0) != out)) {
        throw std::runtime_error("linear2 shape mismatch");
    }
    Tensor y{{bsz, out}, std::vector<float>(static_cast<size_t>(bsz * out), 0.0f)};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t o = 0; o < out; ++o) {
            float acc = bias != nullptr ? bias->data[static_cast<size_t>(o)] : 0.0f;
            for (int64_t i = 0; i < in; ++i) {
                acc += x.data[idx2(x, b, i)] * weight.data[idx2(weight, o, i)];
            }
            y.data[idx2(y, b, o)] = acc;
        }
    }
    return y;
}

static Tensor rms_norm3(const Tensor& x, const Tensor& weight, float eps) {
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t hidden = dim(x, 2);
    if (dim(weight, 0) != hidden) {
        throw std::runtime_error("rms_norm3 shape mismatch");
    }
    Tensor y = x;
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            double sum = 0.0;
            for (int64_t h = 0; h < hidden; ++h) {
                const float v = x.data[idx3(x, b, s, h)];
                sum += static_cast<double>(v) * static_cast<double>(v);
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum / hidden) + eps);
            for (int64_t h = 0; h < hidden; ++h) {
                y.data[idx3(y, b, s, h)] = x.data[idx3(x, b, s, h)] * scale * weight.data[static_cast<size_t>(h)];
            }
        }
    }
    return y;
}

static Tensor layer_norm3(const Tensor& x, float eps) {
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t hidden = dim(x, 2);
    Tensor y = x;
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            double mean = 0.0;
            for (int64_t h = 0; h < hidden; ++h) {
                mean += x.data[idx3(x, b, s, h)];
            }
            mean /= static_cast<double>(hidden);
            double variance = 0.0;
            for (int64_t h = 0; h < hidden; ++h) {
                const double centered = static_cast<double>(x.data[idx3(x, b, s, h)]) - mean;
                variance += centered * centered;
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(variance / static_cast<double>(hidden)) + eps);
            for (int64_t h = 0; h < hidden; ++h) {
                y.data[idx3(y, b, s, h)] = (x.data[idx3(x, b, s, h)] - static_cast<float>(mean)) * scale;
            }
        }
    }
    return y;
}

static Tensor rms_norm4(const Tensor& x, const Tensor& weight, float eps) {
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t heads = dim(x, 2);
    const int64_t head_dim = dim(x, 3);
    if (dim(weight, 0) != head_dim) {
        throw std::runtime_error("rms_norm4 shape mismatch");
    }
    Tensor y = x;
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t h = 0; h < heads; ++h) {
                double sum = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    const float v = x.data[idx4(x, b, s, h, d)];
                    sum += static_cast<double>(v) * static_cast<double>(v);
                }
                const float scale = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
                for (int64_t d = 0; d < head_dim; ++d) {
                    y.data[idx4(y, b, s, h, d)] = x.data[idx4(x, b, s, h, d)] * scale * weight.data[static_cast<size_t>(d)];
                }
            }
        }
    }
    return y;
}

static Tensor mod_linear(const Tensor& temb, const Tensor& weight, const Tensor& bias) {
    Tensor activated = temb;
    for (float& v : activated.data) {
        v = silu(v);
    }
    return linear2(activated, weight, &bias);
}

static Tensor silu_tensor(const Tensor& x) {
    Tensor activated = x;
    for (float& v : activated.data) {
        v = silu(v);
    }
    return activated;
}

static Tensor modulation_chunk_to_bsh(const Tensor& mod, int64_t chunk_index) {
    if (mod.shape.size() != 2 || dim(mod, 0) != 1 || dim(mod, 1) % 6 != 0 ||
        chunk_index < 0 || chunk_index >= 6) {
        throw std::runtime_error("modulation_chunk_to_bsh shape mismatch");
    }
    const int64_t hidden = dim(mod, 1) / 6;
    Tensor out{{1, 1, hidden}, std::vector<float>(static_cast<size_t>(hidden))};
    const int64_t offset = chunk_index * hidden;
    for (int64_t h = 0; h < hidden; ++h) {
        out.data[idx3(out, 0, 0, h)] = mod.data[idx2(mod, 0, offset + h)];
    }
    return out;
}

#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
static Tensor linear2_cublaslt_bf16_optional_bias(const Tensor& input,
                                                  const Tensor& weight,
                                                  const Tensor* bias,
                                                  const std::string& label,
                                                  int* algo_id_out,
                                                  double* elapsed_ms) {
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TRANSFORMER_USE_CUBLASLT)
    (void)input;
    (void)weight;
    (void)bias;
    (void)label;
    (void)algo_id_out;
    (void)elapsed_ms;
    throw std::runtime_error("cuBLASLt modulation probe requires SD_USE_CUDA and CUDA::cublasLt");
#else
    if (input.shape.size() != 2 || weight.shape.size() != 2 ||
        dim(weight, 1) != dim(input, 1) ||
        (bias != nullptr && (bias->shape.size() != 1 || dim(*bias, 0) != dim(weight, 0)))) {
        throw std::runtime_error(label + " cuBLASLt BF16 linear shape mismatch");
    }
    const int64_t in_dim = dim(input, 1);
    const int64_t out_dim = dim(weight, 0);
    const int64_t seq = dim(input, 0);
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    std::vector<ggml_bf16_t> w_bf16(weight.data.size());
    std::vector<ggml_bf16_t> b_bf16;
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(x_bf16.size()));
    ggml_fp32_to_bf16_row_ref(weight.data.data(), w_bf16.data(), static_cast<int64_t>(w_bf16.size()));
    if (bias != nullptr) {
        b_bf16.resize(bias->data.size());
        ggml_fp32_to_bf16_row_ref(bias->data.data(), b_bf16.data(), static_cast<int64_t>(b_bf16.size()));
    }

    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_bias = nullptr;
    void* d_y = nullptr;
    void* workspace = nullptr;
    try {
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cublasLtCreate failed");
        }
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = bias != nullptr ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt descriptor setup failed");
        }
        if (cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt layout setup failed");
        }
        const size_t x_bytes = x_bf16.size() * sizeof(ggml_bf16_t);
        const size_t w_bytes = w_bf16.size() * sizeof(ggml_bf16_t);
        const size_t b_bytes = b_bf16.size() * sizeof(ggml_bf16_t);
        const size_t y_elems = static_cast<size_t>(seq * out_dim);
        if (cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
            cudaMalloc(&d_w, w_bytes) != cudaSuccess ||
            cudaMalloc(&d_y, y_elems * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMalloc failed");
        }
        if (bias != nullptr && cudaMalloc(&d_bias, b_bytes) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMalloc bias failed");
        }
        if (cudaMemcpy(d_x, x_bf16.data(), x_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_w, w_bf16.data(), w_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMemcpy H2D failed");
        }
        if (bias != nullptr) {
            if (cudaMemcpy(d_bias, b_bf16.data(), b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error(label + " cudaMemcpy bias H2D failed");
            }
            if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(label + " cuBLASLt bias pointer setup failed");
            }
        }
        size_t workspace_size = 32ull * 1024ull * 1024ull;
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            workspace = nullptr;
            workspace_size = 0;
        }
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt preference setup failed");
        }
        cublasLtMatmulHeuristicResult_t heuristic = {};
        int returned_results = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt,
                                           op_desc,
                                           a_desc,
                                           b_desc,
                                           c_desc,
                                           d_desc,
                                           pref,
                                           1,
                                           &heuristic,
                                           &returned_results) != CUBLAS_STATUS_SUCCESS ||
            returned_results <= 0) {
            throw std::runtime_error(label + " cuBLASLt heuristic lookup failed");
        }
        if (algo_id_out != nullptr) {
            int algo_id = -1;
            size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(&heuristic.algo,
                                                     CUBLASLT_ALGO_CONFIG_ID,
                                                     &algo_id,
                                                     sizeof(algo_id),
                                                     &written) == CUBLAS_STATUS_SUCCESS) {
                *algo_id_out = algo_id;
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasLtMatmul(lt,
                                                     op_desc,
                                                     &alpha,
                                                     d_w,
                                                     a_desc,
                                                     d_x,
                                                     b_desc,
                                                     &beta,
                                                     d_y,
                                                     c_desc,
                                                     d_y,
                                                     d_desc,
                                                     &heuristic.algo,
                                                     workspace,
                                                     workspace_size,
                                                     0);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cublasLtMatmul failed");
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error(label + " cudaDeviceSynchronize failed");
        }
        if (elapsed_ms != nullptr) {
            *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        std::vector<ggml_bf16_t> y_bf16(y_elems);
        if (cudaMemcpy(y_bf16.data(), d_y, y_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMemcpy D2H failed");
        }
        Tensor out{{seq, out_dim}, std::vector<float>(y_elems)};
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t o = 0; o < out_dim; ++o) {
                out.data[idx2(out, s, o)] = ggml_bf16_to_fp32(y_bf16[static_cast<size_t>(o + out_dim * s)]);
            }
        }
        if (workspace != nullptr) cudaFree(workspace);
        cudaFree(d_y);
        cudaFree(d_bias);
        cudaFree(d_w);
        cudaFree(d_x);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc);
        cublasLtDestroy(lt);
        return out;
    } catch (...) {
        if (workspace != nullptr) cudaFree(workspace);
        if (d_y != nullptr) cudaFree(d_y);
        if (d_bias != nullptr) cudaFree(d_bias);
        if (d_w != nullptr) cudaFree(d_w);
        if (d_x != nullptr) cudaFree(d_x);
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
        if (lt != nullptr) cublasLtDestroy(lt);
        throw;
    }
#endif
}

static Tensor transpose2(const Tensor& x) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("transpose2 rank mismatch");
    }
    Tensor y{{dim(x, 1), dim(x, 0)}, std::vector<float>(x.data.size())};
    for (int64_t r = 0; r < dim(x, 0); ++r) {
        for (int64_t c = 0; c < dim(x, 1); ++c) {
            y.data[idx2(y, c, r)] = x.data[idx2(x, r, c)];
        }
    }
    return y;
}

static Tensor linear2_cublaslt_bf16_optional_bias_algo(const Tensor& input,
                                                       const Tensor& weight,
                                                       const Tensor* bias,
                                                       const std::string& label,
                                                       const cublasLtMatmulAlgo_t* forced_algo,
                                                       int* algo_id_out,
                                                       double* elapsed_ms) {
    if (forced_algo == nullptr) {
        return linear2_cublaslt_bf16_optional_bias(input, weight, bias, label, algo_id_out, elapsed_ms);
    }
#if !defined(SD_USE_CUDA) || !defined(SD_LENS_TRANSFORMER_USE_CUBLASLT)
    (void)input; (void)weight; (void)bias; (void)label; (void)forced_algo; (void)algo_id_out; (void)elapsed_ms;
    throw std::runtime_error("cuBLASLt forced algorithm path requires CUDA");
#else
    if (input.shape.size() != 2 || weight.shape.size() != 2 ||
        dim(weight, 1) != dim(input, 1) ||
        (bias != nullptr && (bias->shape.size() != 1 || dim(*bias, 0) != dim(weight, 0)))) {
        throw std::runtime_error(label + " cuBLASLt BF16 linear forced algo shape mismatch");
    }
    const int64_t in_dim = dim(input, 1);
    const int64_t out_dim = dim(weight, 0);
    const int64_t seq = dim(input, 0);
    std::vector<ggml_bf16_t> x_bf16(input.data.size());
    std::vector<ggml_bf16_t> w_bf16(weight.data.size());
    std::vector<ggml_bf16_t> b_bf16;
    ggml_fp32_to_bf16_row_ref(input.data.data(), x_bf16.data(), static_cast<int64_t>(x_bf16.size()));
    ggml_fp32_to_bf16_row_ref(weight.data.data(), w_bf16.data(), static_cast<int64_t>(w_bf16.size()));
    if (bias != nullptr) {
        b_bf16.resize(bias->data.size());
        ggml_fp32_to_bf16_row_ref(bias->data.data(), b_bf16.data(), static_cast<int64_t>(b_bf16.size()));
    }
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_bias = nullptr;
    void* d_y = nullptr;
    void* workspace = nullptr;
    try {
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(label + " cublasLtCreate failed");
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        cublasLtEpilogue_t epilogue = bias != nullptr ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
        if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(out_dim), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, static_cast<uint64_t>(in_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(in_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16BF, static_cast<uint64_t>(out_dim), static_cast<uint64_t>(seq), static_cast<int64_t>(out_dim)) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(label + " cuBLASLt forced descriptor setup failed");
        }
        if (cudaMalloc(&d_x, x_bf16.size() * sizeof(ggml_bf16_t)) != cudaSuccess ||
            cudaMalloc(&d_w, w_bf16.size() * sizeof(ggml_bf16_t)) != cudaSuccess ||
            cudaMalloc(&d_y, static_cast<size_t>(seq * out_dim) * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMalloc failed");
        }
        if (bias != nullptr && cudaMalloc(&d_bias, b_bf16.size() * sizeof(ggml_bf16_t)) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMalloc bias failed");
        }
        if (cudaMemcpy(d_x, x_bf16.data(), x_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_w, w_bf16.data(), w_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error(label + " cudaMemcpy H2D failed");
        }
        if (bias != nullptr) {
            if (cudaMemcpy(d_bias, b_bf16.data(), b_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
                cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias)) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(label + " bias setup failed");
            }
        }
        size_t workspace_size = 32ull * 1024ull * 1024ull;
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
            workspace = nullptr;
            workspace_size = 0;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        const cublasStatus_t status = cublasLtMatmul(lt, op_desc, &alpha, d_w, a_desc, d_x, b_desc, &beta, d_y, c_desc, d_y, d_desc, forced_algo, workspace, workspace_size, 0);
        if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(label + " forced cublasLtMatmul failed");
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(label + " cudaDeviceSynchronize failed");
        if (elapsed_ms != nullptr) *elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (algo_id_out != nullptr) {
            int algo_id = -1;
            size_t written = 0;
            if (cublasLtMatmulAlgoConfigGetAttribute(forced_algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &written) == CUBLAS_STATUS_SUCCESS) {
                *algo_id_out = algo_id;
            }
        }
        std::vector<ggml_bf16_t> y_bf16(static_cast<size_t>(seq * out_dim));
        if (cudaMemcpy(y_bf16.data(), d_y, y_bf16.size() * sizeof(ggml_bf16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error(label + " D2H failed");
        }
        Tensor out{{seq, out_dim}, std::vector<float>(y_bf16.size())};
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t o = 0; o < out_dim; ++o) {
                out.data[idx2(out, s, o)] = ggml_bf16_to_fp32(y_bf16[static_cast<size_t>(o + out_dim * s)]);
            }
        }
        if (workspace != nullptr) cudaFree(workspace);
        cudaFree(d_y); if (d_bias != nullptr) cudaFree(d_bias); cudaFree(d_w); cudaFree(d_x);
        cublasLtMatrixLayoutDestroy(d_desc); cublasLtMatrixLayoutDestroy(c_desc); cublasLtMatrixLayoutDestroy(b_desc); cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc); cublasLtDestroy(lt);
        return out;
    } catch (...) {
        if (workspace != nullptr) cudaFree(workspace);
        if (d_y != nullptr) cudaFree(d_y);
        if (d_bias != nullptr) cudaFree(d_bias);
        if (d_w != nullptr) cudaFree(d_w);
        if (d_x != nullptr) cudaFree(d_x);
        if (d_desc != nullptr) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc != nullptr) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc != nullptr) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc != nullptr) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc != nullptr) cublasLtMatmulDescDestroy(op_desc);
        if (lt != nullptr) cublasLtDestroy(lt);
        throw;
    }
#endif
}

static Tensor linear2_cublaslt_bf16_bias(const Tensor& input,
                                         const Tensor& weight,
                                         const Tensor& bias,
                                         const std::string& label,
                                         int* algo_id_out,
                                         double* elapsed_ms) {
    return linear2_cublaslt_bf16_optional_bias(input, weight, &bias, label, algo_id_out, elapsed_ms);
}
#endif

#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
static void add_bf16_cublaslt_modulation_inputs(std::unordered_map<std::string, sd::Tensor<float>>& runner_tensors,
                                                const Tensor& temb,
                                                const Tensor& img_w,
                                                const Tensor& img_b,
                                                const Tensor& txt_w,
                                                const Tensor& txt_b,
                                                const std::string& label,
                                                int* img_algo_out = nullptr,
                                                int* txt_algo_out = nullptr,
                                                double* img_ms_out = nullptr,
                                                double* txt_ms_out = nullptr) {
    const Tensor silu_bf16 = round_tensor_bf16(silu_tensor(temb));
    int img_algo = -1;
    int txt_algo = -1;
    double img_ms = 0.0;
    double txt_ms = 0.0;
    const Tensor img_mod = linear2_cublaslt_bf16_bias(silu_bf16, img_w, img_b, label + "_img_mod", &img_algo, &img_ms);
    const Tensor txt_mod = linear2_cublaslt_bf16_bias(silu_bf16, txt_w, txt_b, label + "_txt_mod", &txt_algo, &txt_ms);
    runner_tensors["input.img_shift1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 0));
    runner_tensors["input.img_scale1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 1));
    runner_tensors["input.img_gate1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 2));
    runner_tensors["input.img_shift2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 3));
    runner_tensors["input.img_scale2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 4));
    runner_tensors["input.img_gate2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(img_mod, 5));
    runner_tensors["input.txt_shift1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 0));
    runner_tensors["input.txt_scale1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 1));
    runner_tensors["input.txt_gate1"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 2));
    runner_tensors["input.txt_shift2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 3));
    runner_tensors["input.txt_scale2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 4));
    runner_tensors["input.txt_gate2"] = to_sd_bsh_as_csb(modulation_chunk_to_bsh(txt_mod, 5));
    if (img_algo_out != nullptr) *img_algo_out = img_algo;
    if (txt_algo_out != nullptr) *txt_algo_out = txt_algo;
    if (img_ms_out != nullptr) *img_ms_out = img_ms;
    if (txt_ms_out != nullptr) *txt_ms_out = txt_ms;
}
#endif

static std::pair<Tensor, Tensor> modulate(const Tensor& normed, const Tensor& mod, int64_t offset) {
    const int64_t bsz = dim(normed, 0);
    const int64_t seq = dim(normed, 1);
    const int64_t hidden = dim(normed, 2);
    Tensor y{{bsz, seq, hidden}, std::vector<float>(normed.data.size())};
    Tensor gate{{bsz, 1, hidden}, std::vector<float>(static_cast<size_t>(bsz * hidden))};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t h = 0; h < hidden; ++h) {
            const float shift = mod.data[idx2(mod, b, offset + h)];
            const float scale = mod.data[idx2(mod, b, offset + hidden + h)];
            gate.data[idx3(gate, b, 0, h)] = mod.data[idx2(mod, b, offset + 2 * hidden + h)];
            for (int64_t s = 0; s < seq; ++s) {
                y.data[idx3(y, b, s, h)] = normed.data[idx3(normed, b, s, h)] * (1.0f + scale) + shift;
            }
        }
    }
    return {y, gate};
}

static std::vector<Tensor> split_qkv(const Tensor& qkv, int64_t heads, int64_t head_dim) {
    const int64_t bsz = dim(qkv, 0);
    const int64_t seq = dim(qkv, 1);
    Tensor q{{bsz, seq, heads, head_dim}, std::vector<float>(static_cast<size_t>(bsz * seq * heads * head_dim))};
    Tensor k = q;
    Tensor v = q;
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t part = 0; part < 3; ++part) {
                Tensor& dst = part == 0 ? q : (part == 1 ? k : v);
                for (int64_t h = 0; h < heads; ++h) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const int64_t src = part * heads * head_dim + h * head_dim + d;
                        dst.data[idx4(dst, b, s, h, d)] = qkv.data[idx3(qkv, b, s, src)];
                    }
                }
            }
        }
    }
    return {q, k, v};
}

static Tensor apply_rope(const Tensor& x, const Tensor& freqs) {
    Tensor y = x;
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t heads = dim(x, 2);
    const int64_t head_dim = dim(x, 3);
    if (dim(freqs, 0) < seq || dim(freqs, 1) * 2 != head_dim || dim(freqs, 2) != 2) {
        throw std::runtime_error("rope shape mismatch");
    }
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t p = 0; p < head_dim / 2; ++p) {
                    const float xr = x.data[idx4(x, b, s, h, 2 * p)];
                    const float xi = x.data[idx4(x, b, s, h, 2 * p + 1)];
                    const float fr = freqs.data[idx3(freqs, s, p, 0)];
                    const float fi = freqs.data[idx3(freqs, s, p, 1)];
                    y.data[idx4(y, b, s, h, 2 * p)] = xr * fr - xi * fi;
                    y.data[idx4(y, b, s, h, 2 * p + 1)] = xr * fi + xi * fr;
                }
            }
        }
    }
    return y;
}

static Tensor concat_seq4(const Tensor& a, const Tensor& b) {
    const int64_t bsz = dim(a, 0);
    const int64_t seq_a = dim(a, 1);
    const int64_t seq_b = dim(b, 1);
    const int64_t heads = dim(a, 2);
    const int64_t head_dim = dim(a, 3);
    Tensor y{{bsz, seq_a + seq_b, heads, head_dim},
             std::vector<float>(static_cast<size_t>(bsz * (seq_a + seq_b) * heads * head_dim))};
    for (int64_t bb = 0; bb < bsz; ++bb) {
        for (int64_t s = 0; s < seq_a + seq_b; ++s) {
            const Tensor& src = s < seq_a ? a : b;
            const int64_t ss = s < seq_a ? s : s - seq_a;
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    y.data[idx4(y, bb, s, h, d)] = src.data[idx4(src, bb, ss, h, d)];
                }
            }
        }
    }
    return y;
}

static Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& mask) {
    const int64_t bsz = dim(q, 0);
    const int64_t qseq = dim(q, 1);
    const int64_t heads = dim(q, 2);
    const int64_t head_dim = dim(q, 3);
    const int64_t kseq = dim(k, 1);
    Tensor y{{bsz, qseq, heads * head_dim}, std::vector<float>(static_cast<size_t>(bsz * qseq * heads * head_dim), 0.0f)};
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(kseq));
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t qs = 0; qs < qseq; ++qs) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (int64_t ks = 0; ks < kseq; ++ks) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        score += q.data[idx4(q, b, qs, h, d)] * k.data[idx4(k, b, ks, h, d)];
                    }
                    score = score * scale + mask.data[idx2(mask, b, ks)];
                    scores[static_cast<size_t>(ks)] = score;
                    max_score = std::max(max_score, score);
                }
                float denom = 0.0f;
                for (int64_t ks = 0; ks < kseq; ++ks) {
                    const float ex = std::exp(scores[static_cast<size_t>(ks)] - max_score);
                    scores[static_cast<size_t>(ks)] = ex;
                    denom += ex;
                }
                for (int64_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int64_t ks = 0; ks < kseq; ++ks) {
                        acc += (scores[static_cast<size_t>(ks)] / denom) * v.data[idx4(v, b, ks, h, d)];
                    }
                    y.data[idx3(y, b, qs, h * head_dim + d)] = acc;
                }
            }
        }
    }
    return y;
}

static Tensor slice_seq3(const Tensor& x, int64_t start, int64_t count) {
    Tensor y{{dim(x, 0), count, dim(x, 2)}, std::vector<float>(static_cast<size_t>(dim(x, 0) * count * dim(x, 2)))};
    for (int64_t b = 0; b < dim(y, 0); ++b) {
        for (int64_t s = 0; s < count; ++s) {
            for (int64_t h = 0; h < dim(y, 2); ++h) {
                y.data[idx3(y, b, s, h)] = x.data[idx3(x, b, start + s, h)];
            }
        }
    }
    return y;
}

static Tensor add_gated(const Tensor& x, const Tensor& update, const Tensor& gate) {
    Tensor y = x;
    for (int64_t b = 0; b < dim(x, 0); ++b) {
        for (int64_t s = 0; s < dim(x, 1); ++s) {
            for (int64_t h = 0; h < dim(x, 2); ++h) {
                y.data[idx3(y, b, s, h)] += update.data[idx3(update, b, s, h)] * gate.data[idx3(gate, b, 0, h)];
            }
        }
    }
    return y;
}

static Tensor gate_mlp(const Tensor& x, const Tensor& w1, const Tensor& w2, const Tensor& w3) {
    Tensor a = linear3(x, w1, nullptr);
    Tensor b = linear3(x, w3, nullptr);
    for (size_t i = 0; i < a.data.size(); ++i) {
        a.data[i] = silu(a.data[i]) * b.data[i];
    }
    return linear3(a, w2, nullptr);
}

#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
static Tensor flatten_bsh_to_2d(const Tensor& x) {
    if (x.shape.size() != 3) {
        throw std::runtime_error("flatten_bsh_to_2d rank mismatch");
    }
    const int64_t rows = dim(x, 0) * dim(x, 1);
    const int64_t hidden = dim(x, 2);
    Tensor y{{rows, hidden}, std::vector<float>(static_cast<size_t>(rows * hidden))};
    for (int64_t b = 0; b < dim(x, 0); ++b) {
        for (int64_t s = 0; s < dim(x, 1); ++s) {
            const int64_t row = b * dim(x, 1) + s;
            for (int64_t h = 0; h < hidden; ++h) {
                y.data[idx2(y, row, h)] = x.data[idx3(x, b, s, h)];
            }
        }
    }
    return y;
}

static Tensor unflatten_2d_to_bsh(const Tensor& x, int64_t bsz, int64_t seq) {
    if (x.shape.size() != 2 || dim(x, 0) != bsz * seq) {
        throw std::runtime_error("unflatten_2d_to_bsh shape mismatch");
    }
    Tensor y{{bsz, seq, dim(x, 1)}, std::vector<float>(x.data.size())};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const int64_t row = b * seq + s;
            for (int64_t h = 0; h < dim(x, 1); ++h) {
                y.data[idx3(y, b, s, h)] = x.data[idx2(x, row, h)];
            }
        }
    }
    return y;
}

static Tensor linear3_cublaslt_bf16(const Tensor& x,
                                    const Tensor& weight,
                                    const Tensor* bias,
                                    const std::string& label,
                                    int* algo_id_out,
                                    double* elapsed_ms) {
    const Tensor flat = flatten_bsh_to_2d(x);
    const Tensor y = linear2_cublaslt_bf16_optional_bias(flat, weight, bias, label, algo_id_out, elapsed_ms);
    return unflatten_2d_to_bsh(y, dim(x, 0), dim(x, 1));
}
#endif

static Tensor rms_norm3_bf16_step(const Tensor& x, const Tensor& weight, float eps) {
    const Tensor x_bf16 = round_tensor_bf16(x);
    const int64_t bsz = dim(x, 0);
    const int64_t seq = dim(x, 1);
    const int64_t hidden = dim(x, 2);
    Tensor y = x;
    Tensor weight_bf16 = round_tensor_bf16(weight);
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            double sum = 0.0;
            for (int64_t h = 0; h < hidden; ++h) {
                const float v = x_bf16.data[idx3(x_bf16, b, s, h)];
                sum += static_cast<double>(v) * static_cast<double>(v);
            }
            float inv = 1.0f / std::sqrt(static_cast<float>(sum / hidden) + eps);
            Tensor inv_tensor{{1}, {inv}};
            inv = round_tensor_bf16(inv_tensor).data[0];
            for (int64_t h = 0; h < hidden; ++h) {
                Tensor prod{{1}, {x_bf16.data[idx3(x_bf16, b, s, h)] * inv}};
                prod = round_tensor_bf16(prod);
                Tensor out{{1}, {prod.data[0] * weight_bf16.data[static_cast<size_t>(h)]}};
                y.data[idx3(y, b, s, h)] = round_tensor_bf16(out).data[0];
            }
        }
    }
    return y;
}

static std::pair<Tensor, Tensor> modulate_bf16_step(const Tensor& normed, const Tensor& mod) {
    const int64_t bsz = dim(normed, 0);
    const int64_t seq = dim(normed, 1);
    const int64_t hidden = dim(normed, 2);
    Tensor y{{bsz, seq, hidden}, std::vector<float>(normed.data.size())};
    Tensor gate{{bsz, 1, hidden}, std::vector<float>(static_cast<size_t>(bsz * hidden))};
    Tensor normed_bf16 = round_tensor_bf16(normed);
    Tensor mod_bf16 = round_tensor_bf16(mod);
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t h = 0; h < hidden; ++h) {
            const float shift = mod_bf16.data[idx2(mod_bf16, b, h)];
            const float scale = mod_bf16.data[idx2(mod_bf16, b, hidden + h)];
            gate.data[idx3(gate, b, 0, h)] = mod_bf16.data[idx2(mod_bf16, b, 2 * hidden + h)];
            Tensor one_plus{{1}, {1.0f + scale}};
            const float one_plus_bf16 = round_tensor_bf16(one_plus).data[0];
            for (int64_t s = 0; s < seq; ++s) {
                Tensor prod{{1}, {normed_bf16.data[idx3(normed_bf16, b, s, h)] * one_plus_bf16}};
                prod = round_tensor_bf16(prod);
                Tensor out{{1}, {prod.data[0] + shift}};
                y.data[idx3(y, b, s, h)] = round_tensor_bf16(out).data[0];
            }
        }
    }
    return {y, round_tensor_bf16(gate)};
}

static Tensor add_gated_bf16_step(const Tensor& x, const Tensor& update, const Tensor& gate) {
    Tensor y = x;
    Tensor x_bf16 = round_tensor_bf16(x);
    Tensor update_bf16 = round_tensor_bf16(update);
    Tensor gate_bf16 = round_tensor_bf16(gate);
    for (int64_t b = 0; b < dim(x, 0); ++b) {
        for (int64_t s = 0; s < dim(x, 1); ++s) {
            for (int64_t h = 0; h < dim(x, 2); ++h) {
                Tensor prod{{1}, {update_bf16.data[idx3(update_bf16, b, s, h)] * gate_bf16.data[idx3(gate_bf16, b, 0, h)]}};
                prod = round_tensor_bf16(prod);
                Tensor out{{1}, {x_bf16.data[idx3(x_bf16, b, s, h)] + prod.data[0]}};
                y.data[idx3(y, b, s, h)] = round_tensor_bf16(out).data[0];
            }
        }
    }
    return y;
}

#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
static Tensor gate_mlp_bf16_step(const Tensor& x,
                                 const Tensor& w1,
                                 const Tensor& w2,
                                 const Tensor& w3,
                                 const std::string& label,
                                 double* elapsed_ms) {
    double w1_ms = 0.0;
    double w3_ms = 0.0;
    double w2_ms = 0.0;
    Tensor a = linear3_cublaslt_bf16(x, w1, nullptr, label + ".w1", nullptr, &w1_ms);
    Tensor b = linear3_cublaslt_bf16(x, w3, nullptr, label + ".w3", nullptr, &w3_ms);
    a = round_tensor_bf16(a);
    b = round_tensor_bf16(b);
    for (size_t i = 0; i < a.data.size(); ++i) {
        Tensor silu_value{{1}, {silu(a.data[i])}};
        silu_value = round_tensor_bf16(silu_value);
        Tensor gated{{1}, {silu_value.data[0] * b.data[i]}};
        a.data[i] = round_tensor_bf16(gated).data[0];
    }
    Tensor out = linear3_cublaslt_bf16(a, w2, nullptr, label + ".w2", nullptr, &w2_ms);
    if (elapsed_ms != nullptr) {
        *elapsed_ms = w1_ms + w3_ms + w2_ms;
    }
    return out;
}
#endif

static std::pair<Tensor, Tensor> lens_block_forward_prefixed(const std::unordered_map<std::string, Tensor>& t,
                                                             Tensor hidden,
                                                             Tensor encoder,
                                                             const Tensor& temb,
                                                             const Tensor& img_freqs,
                                                             const Tensor& txt_freqs,
                                                             const Tensor& attention_mask,
                                                             const std::string& prefix) {
    const int64_t head_dim = dim(img_freqs, 1) * 2;
    const int64_t qkv_out = dim(need(t, prefix + "attn.img_qkv.weight"), 0);
    if (head_dim <= 0 || qkv_out <= 0 || (qkv_out % (3 * head_dim)) != 0) {
        throw std::runtime_error("cannot infer Lens attention head shape");
    }
    const int64_t heads = qkv_out / (3 * head_dim);
    constexpr float eps = 1.0e-6f;

    Tensor img_mod = mod_linear(temb, need(t, prefix + "img_mod.1.weight"), need(t, prefix + "img_mod.1.bias"));
    Tensor txt_mod = mod_linear(temb, need(t, prefix + "txt_mod.1.weight"), need(t, prefix + "txt_mod.1.bias"));
    auto img_m1 = modulate(rms_norm3(hidden, need(t, prefix + "img_norm1.weight"), eps), img_mod, 0);
    auto txt_m1 = modulate(rms_norm3(encoder, need(t, prefix + "txt_norm1.weight"), eps), txt_mod, 0);

    auto img_qkv = split_qkv(linear3(img_m1.first, need(t, prefix + "attn.img_qkv.weight"), &need(t, prefix + "attn.img_qkv.bias")), heads, head_dim);
    auto txt_qkv = split_qkv(linear3(txt_m1.first, need(t, prefix + "attn.txt_qkv.weight"), &need(t, prefix + "attn.txt_qkv.bias")), heads, head_dim);
    Tensor img_q = apply_rope(rms_norm4(img_qkv[0], need(t, prefix + "attn.norm_q.weight"), eps), img_freqs);
    Tensor img_k = apply_rope(rms_norm4(img_qkv[1], need(t, prefix + "attn.norm_k.weight"), eps), img_freqs);
    Tensor txt_q = apply_rope(rms_norm4(txt_qkv[0], need(t, prefix + "attn.norm_added_q.weight"), eps), txt_freqs);
    Tensor txt_k = apply_rope(rms_norm4(txt_qkv[1], need(t, prefix + "attn.norm_added_k.weight"), eps), txt_freqs);
    Tensor q = concat_seq4(img_q, txt_q);
    Tensor k = concat_seq4(img_k, txt_k);
    Tensor v = concat_seq4(img_qkv[2], txt_qkv[2]);
    Tensor out = attention(q, k, v, attention_mask);
    Tensor img_attn = linear3(slice_seq3(out, 0, dim(hidden, 1)), need(t, prefix + "attn.to_out.0.weight"), &need(t, prefix + "attn.to_out.0.bias"));
    Tensor txt_attn = linear3(slice_seq3(out, dim(hidden, 1), dim(encoder, 1)), need(t, prefix + "attn.to_add_out.weight"), &need(t, prefix + "attn.to_add_out.bias"));
    hidden = add_gated(hidden, img_attn, img_m1.second);
    encoder = add_gated(encoder, txt_attn, txt_m1.second);

    auto img_m2 = modulate(rms_norm3(hidden, need(t, prefix + "img_norm2.weight"), eps), img_mod, dim(hidden, 2) * 3);
    hidden = add_gated(hidden, gate_mlp(img_m2.first,
                                        need(t, prefix + "img_mlp.w1.weight"),
                                        need(t, prefix + "img_mlp.w2.weight"),
                                        need(t, prefix + "img_mlp.w3.weight")),
                       img_m2.second);

    auto txt_m2 = modulate(rms_norm3(encoder, need(t, prefix + "txt_norm2.weight"), eps), txt_mod, dim(encoder, 2) * 3);
    encoder = add_gated(encoder, gate_mlp(txt_m2.first,
                                          need(t, prefix + "txt_mlp.w1.weight"),
                                          need(t, prefix + "txt_mlp.w2.weight"),
                                          need(t, prefix + "txt_mlp.w3.weight")),
                        txt_m2.second);

    return {encoder, hidden};
}

static std::pair<Tensor, Tensor> lens_block_forward(const std::unordered_map<std::string, Tensor>& t) {
    return lens_block_forward_prefixed(t,
                                       need(t, "input.hidden"),
                                       need(t, "input.encoder"),
                                       need(t, "input.temb"),
                                       need(t, "input.img_freqs"),
                                       need(t, "input.txt_freqs"),
                                       need(t, "input.attention_mask"),
                                       "state.");
}

static Tensor concat_features(const std::vector<Tensor>& features) {
    if (features.empty()) {
        throw std::runtime_error("concat_features requires at least one tensor");
    }
    const int64_t bsz = dim(features[0], 0);
    const int64_t seq = dim(features[0], 1);
    int64_t hidden = 0;
    for (const Tensor& feature : features) {
        if (dim(feature, 0) != bsz || dim(feature, 1) != seq) {
            throw std::runtime_error("feature shape mismatch");
        }
        hidden += dim(feature, 2);
    }
    Tensor y{{bsz, seq, hidden}, std::vector<float>(static_cast<size_t>(bsz * seq * hidden))};
    for (int64_t b = 0; b < bsz; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            int64_t offset = 0;
            for (const Tensor& feature : features) {
                for (int64_t h = 0; h < dim(feature, 2); ++h) {
                    y.data[idx3(y, b, s, offset + h)] = feature.data[idx3(feature, b, s, h)];
                }
                offset += dim(feature, 2);
            }
        }
    }
    return y;
}

static Tensor final_ada_norm(const Tensor& x, const Tensor& temb, const Tensor& weight, const Tensor& bias) {
    Tensor activated = temb;
    for (float& v : activated.data) {
        v = silu(v);
    }
    Tensor emb = linear2(activated, weight, &bias);
    Tensor normed = layer_norm3(x, 1.0e-6f);
    Tensor y = normed;
    const int64_t hidden = dim(x, 2);
    for (int64_t b = 0; b < dim(x, 0); ++b) {
        for (int64_t s = 0; s < dim(x, 1); ++s) {
            for (int64_t h = 0; h < hidden; ++h) {
                const float scale = emb.data[idx2(emb, b, h)];
                const float shift = emb.data[idx2(emb, b, hidden + h)];
                y.data[idx3(y, b, s, h)] = normed.data[idx3(normed, b, s, h)] * (1.0f + scale) + shift;
            }
        }
    }
    return y;
}

static Tensor lens_full_forward(const std::unordered_map<std::string, Tensor>& t) {
    Tensor hidden = linear3(need(t, "full.input.hidden"), need(t, "full.state.img_in.weight"), &need(t, "full.state.img_in.bias"));
    std::vector<Tensor> features;
    for (int i = 0; i < 4; ++i) {
        const std::string idx = std::to_string(i);
        features.push_back(rms_norm3(need(t, "full.input.feature_" + idx), need(t, "full.state.txt_norm." + idx + ".weight"), 1.0e-5f));
    }
    Tensor encoder = linear3(concat_features(features), need(t, "full.state.txt_in.weight"), &need(t, "full.state.txt_in.bias"));
    const Tensor& temb = need(t, "full.input.temb");
    const Tensor& img_freqs = need(t, "full.input.img_freqs");
    const Tensor& txt_freqs = need(t, "full.input.txt_freqs");
    const Tensor& mask = need(t, "full.input.attention_mask");
    for (int i = 0; i < 2; ++i) {
        auto outputs = lens_block_forward_prefixed(t,
                                                   hidden,
                                                   encoder,
                                                   temb,
                                                   img_freqs,
                                                   txt_freqs,
                                                   mask,
                                                   "full.state.transformer_blocks." + std::to_string(i) + ".");
        encoder = outputs.first;
        hidden = outputs.second;
    }
    hidden = final_ada_norm(hidden, temb, need(t, "full.state.norm_out.linear.weight"), need(t, "full.state.norm_out.linear.bias"));
    return linear3(hidden, need(t, "full.state.proj_out.weight"), &need(t, "full.state.proj_out.bias"));
}

static float max_abs_diff(const Tensor& a, const Tensor& b, float* mean_abs) {
    if (a.shape != b.shape || a.data.size() != b.data.size()) {
        throw std::runtime_error("comparison shape mismatch");
    }
    double sum = 0.0;
    float max_value = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const float d = std::fabs(a.data[i] - b.data[i]);
        max_value = std::max(max_value, d);
        sum += d;
    }
    if (mean_abs != nullptr) {
        *mean_abs = static_cast<float>(sum / static_cast<double>(a.data.size()));
    }
    return max_value;
}

static bool run_external_flow_api_check(const std::string& lens_cond_path,
                                        const std::string& transformer_path,
                                        const Tensor& initial_tokens,
                                        const std::vector<float>& model_outputs,
                                        int steps,
                                        int image_seq_len,
                                        uint32_t expected_in_channels,
                                        uint32_t expected_enc_hidden_dim,
                                        Tensor& out_tokens,
                                        sd_lens_external_flow_loop_desc_t& out_desc) {
    if (lens_cond_path.empty() || transformer_path.empty()) {
        std::cerr << "--verify-external-flow-api requires --lens-cond and --real-block-transformer\n";
        return false;
    }
    if (dim(initial_tokens, 0) != 1 || dim(initial_tokens, 1) != image_seq_len ||
        dim(initial_tokens, 2) != static_cast<int64_t>(expected_in_channels) ||
        model_outputs.size() != initial_tokens.data.size() * static_cast<size_t>(steps)) {
        std::cerr << "external flow API input shape mismatch\n";
        return false;
    }

    sd_ctx_t* ctx = new_sd_ctx_lens_conditioning_only();
    if (ctx == nullptr) {
        std::cerr << "failed to create Lens conditioning-only context for external flow API check\n";
        return false;
    }

    sd_lens_conditioning_handle_t cond_handle = 0;
    sd_lens_transformer_handle_t transformer_handle = 0;
    auto cleanup = [&]() {
        if (transformer_handle != 0) {
            sd_lens_transformer_release(ctx, transformer_handle);
            transformer_handle = 0;
        }
        if (cond_handle != 0) {
            sd_lens_conditioning_release(ctx, cond_handle);
            cond_handle = 0;
        }
        free_sd_ctx(ctx);
    };

    sd_lens_conditioning_options_t cond_options;
    sd_lens_conditioning_options_init(&cond_options);
    cond_options.expected_schema_version = 1;
    cond_options.expected_batch = 1;
    cond_options.expected_hidden_size = static_cast<int64_t>(expected_enc_hidden_dim);
    cond_options.expected_selected_layer_count = 4;
    cond_options.expected_tensor_count = 5;
    cond_options.expected_storage_flags = SD_LENS_COND_STORAGE_FLAG_CPU;
    sd_lens_conditioning_desc_t cond_desc;
    sd_lens_conditioning_desc_init(&cond_desc);
    if (!sd_lens_conditioning_load(ctx, lens_cond_path.c_str(), &cond_options, &cond_handle, &cond_desc)) {
        std::cerr << "sd_lens_conditioning_load failed during external flow API check\n";
        cleanup();
        return false;
    }

    sd_lens_transformer_options_t transformer_options;
    sd_lens_transformer_options_init(&transformer_options);
    transformer_options.metadata_only = true;
    transformer_options.allow_unsafe_large_allocations = false;
    transformer_options.expected_in_channels = expected_in_channels;
    transformer_options.expected_enc_hidden_dim = expected_enc_hidden_dim;
    sd_lens_transformer_desc_t transformer_desc;
    sd_lens_transformer_desc_init(&transformer_desc);
    if (!sd_lens_transformer_load(ctx, transformer_path.c_str(), &transformer_options, &transformer_handle, &transformer_desc)) {
        std::cerr << "sd_lens_transformer_load failed during external flow API check\n";
        cleanup();
        return false;
    }

    sd_lens_schedule_options_t schedule_options;
    sd_lens_schedule_options_init(&schedule_options);
    schedule_options.steps = steps;
    schedule_options.image_seq_len = image_seq_len;

    out_tokens.shape = initial_tokens.shape;
    out_tokens.data.assign(initial_tokens.data.size(), 0.0f);
    sd_lens_external_flow_loop_desc_init(&out_desc);
    if (!sd_lens_run_external_flow_loop_f32(ctx,
                                           cond_handle,
                                           transformer_handle,
                                           &schedule_options,
                                           initial_tokens.data.data(),
                                           static_cast<uint64_t>(initial_tokens.data.size()),
                                           model_outputs.data(),
                                           static_cast<uint64_t>(model_outputs.size()),
                                           out_tokens.data.data(),
                                           static_cast<uint64_t>(out_tokens.data.size()),
                                           &out_desc)) {
        std::cerr << "sd_lens_run_external_flow_loop_f32 failed during external flow API check\n";
        cleanup();
        return false;
    }
    cleanup();

    return out_desc.steps == steps &&
           out_desc.image_seq_len == image_seq_len &&
           out_desc.in_channels == expected_in_channels &&
           out_desc.packed_token_elements == initial_tokens.data.size() &&
           out_desc.model_output_elements == model_outputs.size() &&
           out_desc.used_precomputed_conditioning &&
           out_desc.used_external_model_output &&
           !out_desc.native_transformer_forward &&
           out_desc.cpu_only;
}

SD_LENS_TRANSFORMER_SMOKE_API int sd_lens_transformer_smoke_main_impl_profiled(
    int argc,
    char** argv,
    const std::unordered_map<std::string, Tensor>* in_memory_cond_tensors,
    Tensor* out_native_cuda_latent,
    sd_lens_transformer_runtime_profile* out_profile,
    sd_lens_runtime_progress_cb_t progress_callback,
    void* progress_callback_data) {
    struct InMemoryScope {
        const std::unordered_map<std::string, Tensor>* previous_cond = nullptr;
        Tensor* previous_latent = nullptr;
        InMemoryScope(const std::unordered_map<std::string, Tensor>* cond, Tensor* latent)
            : previous_cond(g_lens_in_memory_cond_tensors),
              previous_latent(g_lens_native_cuda_output_latent) {
            g_lens_in_memory_cond_tensors = cond;
            g_lens_native_cuda_output_latent = latent;
        }
        ~InMemoryScope() {
            g_lens_in_memory_cond_tensors = previous_cond;
            g_lens_native_cuda_output_latent = previous_latent;
        }
    } in_memory_scope(in_memory_cond_tensors, out_native_cuda_latent);
    if (out_profile != nullptr) {
        *out_profile = {};
    }

    std::string fixture;
    std::string oracle_input_fixture;
    std::string real_block_oracle_fixture;
    std::string real_full_oracle_fixture;
    std::string real_block_cuda_oracle_fixture;
    std::string real_full_cuda_oracle_fixture;
    std::string external_flow_fixture;
    std::string inspect_transformer;
    std::string real_block_transformer;
    float tolerance = 1.0e-4f;
    int real_block_index = 0;
    int real_num_blocks = 1;
    int real_img_seq = 1;
    int real_txt_seq = 1;
    int external_height = 0;
    int external_width = 0;
    uint64_t max_real_block_bytes = 512ull * 1024ull * 1024ull;
    uint64_t max_real_top_bytes = 256ull * 1024ull * 1024ull;
    uint64_t max_real_cond_bytes = 256ull * 1024ull * 1024ull;
    LensAttentionMode lens_attention_mode = LensAttentionMode::RegularF32;
    bool real_full_transformer = false;
    bool emit_tiny_denoise_npy = false;
    bool apply_tiny_flow_step = false;
    bool verify_external_flow_api = false;
    bool native_cuda_generate_256 = false;
    bool use_transformer_context = false;
    bool keep_transformer_warm = false;
    bool output_packed_vae_latent = false;
    bool modulation_microbench = false;
    bool modulation_bf16_cublaslt = false;
    bool mlp_residual_microbench = false;
    bool txt_mlp_linear_microbench = false;
    bool attention_sdpa_microbench = false;
    std::string transformer_residency = "streaming";
    std::string transformer_runner_reuse = "none";
    std::string dynamic_residency = "none";
    int window_blocks = 0;
    int persistent_block = -1;
    int persistent_blocks = 0;
    uint64_t persistent_blocks_memory_mib = 4096;
    int tiny_flow_steps = 4;
    int seed = 42;
    int repeat_generations = 1;
    std::string lens_cond_path;
    std::string tiny_denoise_npy = "lens_tiny_denoise_latent.npy";
    std::string packed_tokens_npy = "lens_native_final_packed.npy";
    std::string initial_packed_tokens_npy;
    std::string modulation_out_prefix = "build/diagnostics/lens_b213_native_modulation";
    std::string mlp_residual_fixture = "build/diagnostics/lens_b217_mlp_residual_fixture.bin";
    std::string txt_mlp_linear_fixture = "build/diagnostics/lens_b218_txt_mlp_fixture.bin";
    std::string attention_fixture = "build/diagnostics/lens_b36_attention_fixture_512.bin";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) {
            fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--oracle-input-fixture") == 0 && i + 1 < argc) {
            oracle_input_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--real-block-oracle-fixture") == 0 && i + 1 < argc) {
            real_block_oracle_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--real-block-cuda-oracle-fixture") == 0 && i + 1 < argc) {
            real_block_cuda_oracle_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--real-full-cuda-oracle-fixture") == 0 && i + 1 < argc) {
            real_full_cuda_oracle_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--real-full-oracle-fixture") == 0 && i + 1 < argc) {
            real_full_oracle_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--external-flow-fixture") == 0 && i + 1 < argc) {
            external_flow_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--inspect-transformer") == 0 && i + 1 < argc) {
            inspect_transformer = argv[++i];
        } else if (std::strcmp(argv[i], "--real-block-transformer") == 0 && i + 1 < argc) {
            real_block_transformer = argv[++i];
        } else if (std::strcmp(argv[i], "--real-block-index") == 0 && i + 1 < argc) {
            real_block_index = parse_i32(argv[++i], "--real-block-index");
        } else if (std::strcmp(argv[i], "--real-num-blocks") == 0 && i + 1 < argc) {
            real_num_blocks = parse_i32(argv[++i], "--real-num-blocks");
        } else if (std::strcmp(argv[i], "--real-img-seq") == 0 && i + 1 < argc) {
            real_img_seq = parse_i32(argv[++i], "--real-img-seq");
        } else if (std::strcmp(argv[i], "--real-txt-seq") == 0 && i + 1 < argc) {
            real_txt_seq = parse_i32(argv[++i], "--real-txt-seq");
        } else if (std::strcmp(argv[i], "--external-height") == 0 && i + 1 < argc) {
            external_height = parse_i32(argv[++i], "--external-height");
        } else if (std::strcmp(argv[i], "--external-width") == 0 && i + 1 < argc) {
            external_width = parse_i32(argv[++i], "--external-width");
        } else if (std::strcmp(argv[i], "--max-real-block-bytes") == 0 && i + 1 < argc) {
            max_real_block_bytes = parse_u64(argv[++i], "--max-real-block-bytes");
        } else if (std::strcmp(argv[i], "--max-real-top-bytes") == 0 && i + 1 < argc) {
            max_real_top_bytes = parse_u64(argv[++i], "--max-real-top-bytes");
        } else if (std::strcmp(argv[i], "--max-real-cond-bytes") == 0 && i + 1 < argc) {
            max_real_cond_bytes = parse_u64(argv[++i], "--max-real-cond-bytes");
        } else if (std::strcmp(argv[i], "--lens-attention-mode") == 0 && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "regular-f32") {
                lens_attention_mode = LensAttentionMode::RegularF32;
            } else if (mode == "flash") {
                lens_attention_mode = LensAttentionMode::Flash;
            } else {
                std::cerr << "--lens-attention-mode must be regular-f32 or flash\n";
                return 2;
            }
        } else if (std::strcmp(argv[i], "--real-full-transformer") == 0) {
            real_full_transformer = true;
        } else if (std::strcmp(argv[i], "--lens-cond") == 0 && i + 1 < argc) {
            lens_cond_path = argv[++i];
        } else if (std::strcmp(argv[i], "--emit-tiny-denoise-npy") == 0) {
            emit_tiny_denoise_npy = true;
        } else if (std::strcmp(argv[i], "--apply-tiny-flow-step") == 0) {
            apply_tiny_flow_step = true;
        } else if (std::strcmp(argv[i], "--verify-external-flow-api") == 0) {
            verify_external_flow_api = true;
        } else if (std::strcmp(argv[i], "--native-cuda-generate-256") == 0) {
            native_cuda_generate_256 = true;
        } else if (std::strcmp(argv[i], "--use-transformer-context") == 0) {
            use_transformer_context = true;
        } else if (std::strcmp(argv[i], "--keep-transformer-warm") == 0) {
            keep_transformer_warm = true;
        } else if (std::strcmp(argv[i], "--output-packed-vae-latent") == 0) {
            output_packed_vae_latent = true;
        } else if (std::strcmp(argv[i], "--modulation-microbench") == 0) {
            modulation_microbench = true;
        } else if (std::strcmp(argv[i], "--modulation-bf16-cublaslt") == 0) {
            modulation_bf16_cublaslt = true;
        } else if (std::strcmp(argv[i], "--mlp-residual-microbench") == 0) {
            mlp_residual_microbench = true;
        } else if (std::strcmp(argv[i], "--mlp-residual-fixture") == 0 && i + 1 < argc) {
            mlp_residual_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--txt-mlp-linear-microbench") == 0) {
            txt_mlp_linear_microbench = true;
        } else if (std::strcmp(argv[i], "--txt-mlp-linear-fixture") == 0 && i + 1 < argc) {
            txt_mlp_linear_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--attention-sdpa-microbench") == 0) {
            attention_sdpa_microbench = true;
        } else if (std::strcmp(argv[i], "--attention-fixture") == 0 && i + 1 < argc) {
            attention_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--modulation-out-prefix") == 0 && i + 1 < argc) {
            modulation_out_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--transformer-residency") == 0 && i + 1 < argc) {
            transformer_residency = argv[++i];
        } else if (std::strcmp(argv[i], "--transformer-runner-reuse") == 0 && i + 1 < argc) {
            transformer_runner_reuse = argv[++i];
        } else if (std::strcmp(argv[i], "--dynamic-residency") == 0 && i + 1 < argc) {
            dynamic_residency = argv[++i];
        } else if (std::strcmp(argv[i], "--window-blocks") == 0 && i + 1 < argc) {
            window_blocks = parse_i32(argv[++i], "--window-blocks");
        } else if (std::strcmp(argv[i], "--persistent-block") == 0 && i + 1 < argc) {
            persistent_block = parse_i32(argv[++i], "--persistent-block");
        } else if (std::strcmp(argv[i], "--persistent-blocks") == 0 && i + 1 < argc) {
            persistent_blocks = parse_i32(argv[++i], "--persistent-blocks");
        } else if (std::strcmp(argv[i], "--persistent-blocks-memory-mib") == 0 && i + 1 < argc) {
            persistent_blocks_memory_mib = parse_u64(argv[++i], "--persistent-blocks-memory-mib");
        } else if (std::strcmp(argv[i], "--tiny-flow-steps") == 0 && i + 1 < argc) {
            tiny_flow_steps = parse_i32(argv[++i], "--tiny-flow-steps");
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = parse_i32(argv[++i], "--seed");
        } else if (std::strcmp(argv[i], "--repeat-generations") == 0 && i + 1 < argc) {
            repeat_generations = parse_i32(argv[++i], "--repeat-generations");
        } else if (std::strcmp(argv[i], "--tiny-denoise-npy") == 0 && i + 1 < argc) {
            tiny_denoise_npy = argv[++i];
        } else if (std::strcmp(argv[i], "--packed-tokens-npy") == 0 && i + 1 < argc) {
            packed_tokens_npy = argv[++i];
        } else if (std::strcmp(argv[i], "--initial-packed-tokens-npy") == 0 && i + 1 < argc) {
            initial_packed_tokens_npy = argv[++i];
        } else if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = std::strtof(argv[++i], nullptr);
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--fixture lens_block_fixture.bin]"
                      << " [--oracle-input-fixture lens_first_image_fixture.bin]"
                      << " [--real-block-oracle-fixture lens_block0_oracle.bin]"
                      << " [--real-block-cuda-oracle-fixture lens_block0_oracle.bin]"
                      << " [--real-full-cuda-oracle-fixture lens_full_oracle.bin]"
                      << " [--real-full-oracle-fixture lens_full_oracle.bin]"
                      << " [--external-flow-fixture lens_external_flow_fixture.bin]"
                      << " [--inspect-transformer transformer_dir]"
                      << " [--real-block-transformer transformer_dir]"
                      << " [--real-block-index 0]"
                      << " [--real-num-blocks 1]"
                      << " [--real-img-seq 1] [--real-txt-seq 1]"
                      << " [--external-height H --external-width W]"
                      << " [--max-real-block-bytes 536870912]"
                      << " [--max-real-top-bytes 268435456]"
                      << " [--max-real-cond-bytes 268435456]"
                      << " [--lens-attention-mode regular-f32|flash]"
                      << " [--real-full-transformer]"
                      << " [--lens-cond lens_cond_v1.safetensors]"
                      << " [--emit-tiny-denoise-npy]"
                      << " [--apply-tiny-flow-step]"
                      << " [--verify-external-flow-api]"
                      << " [--native-cuda-generate-256]"
                      << " [--use-transformer-context]"
                      << " [--keep-transformer-warm]"
                      << " [--output-packed-vae-latent]"
                      << " [--modulation-microbench --modulation-out-prefix prefix]"
                      << " [--modulation-bf16-cublaslt]"
                      << " [--mlp-residual-microbench --mlp-residual-fixture fixture.bin]"
                      << " [--txt-mlp-linear-microbench --txt-mlp-linear-fixture fixture.bin]"
                      << " [--attention-sdpa-microbench --attention-fixture fixture.bin]"
                      << " [--transformer-residency streaming|gpu-window|persistent-block|persistent-blocks|gpu-full-bf16 --window-blocks N --persistent-block 0 --persistent-blocks N]"
                      << " [--transformer-runner-reuse none|block0]"
                      << " [--dynamic-residency none|two-block-proof|gpu-streams]"
                      << " [--persistent-blocks-memory-mib 4096]"
                      << " [--tiny-flow-steps 4]"
                      << " [--seed 42]"
                      << " [--repeat-generations 1]"
                      << " [--tiny-denoise-npy output.npy]"
                      << " [--packed-tokens-npy output.npy]"
                      << " [--initial-packed-tokens-npy output.npy]"
                      << " [--tolerance 1e-4]\n";
            return 2;
        }
    }
    if (real_img_seq <= 0 || real_txt_seq <= 0 || real_num_blocks <= 0) {
        std::cerr << "--real-img-seq, --real-txt-seq, and --real-num-blocks must be positive\n";
        return 2;
    }
    if (verify_external_flow_api && !apply_tiny_flow_step) {
        std::cerr << "--verify-external-flow-api requires --apply-tiny-flow-step\n";
        return 2;
    }
    if (transformer_residency != "streaming" &&
        transformer_residency != "gpu-window" &&
        transformer_residency != "persistent-block" &&
        transformer_residency != "persistent-blocks" &&
        transformer_residency != "gpu-full-bf16") {
        std::cerr << "--transformer-residency must be streaming, gpu-window, persistent-block, persistent-blocks, or gpu-full-bf16\n";
        return 2;
    }
    if (transformer_residency == "gpu-window" && window_blocks <= 0) {
        std::cerr << "--transformer-residency gpu-window requires --window-blocks N\n";
        return 2;
    }
    if (transformer_residency == "persistent-block" && persistent_block != 0) {
        std::cerr << "--transformer-residency persistent-block currently requires --persistent-block 0\n";
        return 2;
    }
    if (transformer_residency == "persistent-blocks" && persistent_blocks <= 0) {
        std::cerr << "--transformer-residency persistent-blocks requires --persistent-blocks N\n";
        return 2;
    }
    if (transformer_runner_reuse != "none" && transformer_runner_reuse != "block0") {
        std::cerr << "--transformer-runner-reuse must be none or block0\n";
        return 2;
    }
    if (dynamic_residency != "none" &&
        dynamic_residency != "two-block-proof" &&
        dynamic_residency != "gpu-streams") {
        std::cerr << "--dynamic-residency must be none, two-block-proof, or gpu-streams\n";
        return 2;
    }
    if (dynamic_residency != "none" && transformer_residency != "gpu-full-bf16") {
        std::cerr << "--dynamic-residency currently requires --transformer-residency gpu-full-bf16\n";
        return 2;
    }
    if (transformer_runner_reuse == "block0" &&
        !(transformer_residency == "persistent-block" ||
          (transformer_residency == "persistent-blocks" && persistent_blocks >= 1))) {
        std::cerr << "--transformer-runner-reuse block0 requires persistent block 0 weights\n";
        return 2;
    }
    if (persistent_blocks > 48) {
        std::cerr << "--persistent-blocks must be <= 48\n";
        return 2;
    }
    if (persistent_blocks_memory_mib == 0) {
        std::cerr << "--persistent-blocks-memory-mib must be positive\n";
        return 2;
    }
    if (repeat_generations <= 0 || repeat_generations > 4) {
        std::cerr << "--repeat-generations must be in [1,4]\n";
        return 2;
    }
    if (window_blocks > 48) {
        std::cerr << "--window-blocks must be <= 48\n";
        return 2;
    }
    if (attention_sdpa_microbench) {
        try {
            LensAttentionFixture fx = load_attention_fixture(attention_fixture);
            std::cout << "attention_fixture path=" << attention_fixture
                      << " B=" << fx.b
                      << " H=" << fx.h
                      << " S=" << fx.s
                      << " D=" << fx.d
                      << " mask_shape=";
            for (size_t i = 0; i < fx.mask_shape.size(); ++i) {
                std::cout << (i == 0 ? "" : "x") << fx.mask_shape[i];
            }
            std::cout << " mask_expanded_bytes=" << static_cast<uint64_t>(fx.mask_expanded.size()) * sizeof(float)
                      << "\n";

            ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
            if (cuda_backend == nullptr) {
                std::cerr << "ggml_backend_cuda_init(0) failed\n";
                return 1;
            }
            struct AttentionCase {
                const char* name;
                LensAttentionMode mode;
                bool use_mask;
            };
            const std::vector<AttentionCase> cases = {
                {"regular_f32_with_mask", LensAttentionMode::RegularF32, true},
                {"regular_f32_no_mask", LensAttentionMode::RegularF32, false},
                {"ggml_flash_with_mask", LensAttentionMode::Flash, true},
                {"ggml_flash_no_mask", LensAttentionMode::Flash, false},
            };
            bool any_ok = false;
            for (const AttentionCase& c : cases) {
                std::cout << "attention_microbench_case name=" << c.name
                          << " mode=" << lens_attention_mode_name(c.mode)
                          << " use_mask=" << (c.use_mask ? "true" : "false")
                          << "\n";
                try {
                    LensAttentionMicrobenchRunner runner(cuda_backend, fx, c.mode, c.use_mask);
                    const int64_t t0 = ggml_time_us();
                    std::optional<sd::Tensor<float>> actual = runner.compute_attention(1);
                    const int64_t t1 = ggml_time_us();
                    if (!actual.has_value()) {
                        std::cout << "attention_microbench_result name=" << c.name << " status=failed_compute\n";
                        continue;
                    }
                    Tensor actual_tensor;
                    actual_tensor.shape = actual->shape();
                    actual_tensor.data = actual->values();
                    const DiffStats stats = diff_stats(actual_tensor, fx.expected);
                    const GGMLRunnerTimingProfile& profile = runner.get_last_timing_profile();
                    std::cout << "attention_microbench_result name=" << c.name
                              << " status=ok"
                              << " wall_s=" << static_cast<double>(t1 - t0) / 1000000.0
                              << " backend_compute_s=" << profile.backend_compute_seconds
                              << " input_copy_s=" << profile.input_copy_seconds
                              << " output_copy_s=" << profile.output_copy_seconds
                              << " graph_build_s=" << profile.graph_build_seconds
                              << " graph_alloc_s=" << profile.graph_alloc_seconds
                              << " input_copy_bytes=" << profile.input_copy_bytes
                              << " output_copy_bytes=" << profile.output_copy_bytes
                              << " max_diff=" << stats.max_diff
                              << " mean_diff=" << stats.mean_diff
                              << " nonfinite_actual=" << stats.nonfinite_actual
                              << " nonfinite_ref=" << stats.nonfinite_ref
                              << " extrapolated_192_backend_compute_s=" << profile.backend_compute_seconds * 192.0
                              << "\n";
                    any_ok = any_ok || stats.nonfinite_actual == 0;
                } catch (const std::exception& e) {
                    std::cout << "attention_microbench_result name=" << c.name
                              << " status=exception error=\"" << e.what() << "\"\n";
                }
            }
            ggml_backend_free(cuda_backend);
            return any_ok ? 0 : 1;
        } catch (const std::exception& e) {
            std::cerr << "attention microbench failed: " << e.what() << "\n";
            return 1;
        }
    }
    if (fixture.empty() && inspect_transformer.empty() && real_block_transformer.empty() && external_flow_fixture.empty()) {
        std::cerr << "usage: " << argv[0]
                  << " [--fixture lens_block_fixture.bin]"
                  << " [--oracle-input-fixture lens_first_image_fixture.bin]"
                  << " [--real-block-oracle-fixture lens_block0_oracle.bin]"
                  << " [--real-block-cuda-oracle-fixture lens_block0_oracle.bin]"
                  << " [--real-full-cuda-oracle-fixture lens_full_oracle.bin]"
                  << " [--real-full-oracle-fixture lens_full_oracle.bin]"
                  << " [--external-flow-fixture lens_external_flow_fixture.bin]"
                  << " [--inspect-transformer transformer_dir]"
                  << " [--real-block-transformer transformer_dir]\n";
        return 2;
    }

    try {
        if (!external_flow_fixture.empty()) {
            if (lens_cond_path.empty() || real_block_transformer.empty()) {
                std::cerr << "--external-flow-fixture requires --lens-cond and --real-block-transformer\n";
                return 2;
            }
            auto tensors = load_fixture(external_flow_fixture);
            const Tensor& initial = need(tensors, "external.initial");
            const Tensor& model_outputs = need(tensors, "external.model_outputs");
            if (dim(initial, 0) != 1 || dim(initial, 2) != 128 ||
                model_outputs.shape.size() != 4 || dim(model_outputs, 1) != 1 ||
                dim(model_outputs, 2) != dim(initial, 1) || dim(model_outputs, 3) != 128) {
                std::cerr << "external flow fixture must contain initial [1,S,128] and model_outputs [steps,1,S,128]\n";
                return 1;
            }
            const int external_steps = static_cast<int>(dim(model_outputs, 0));
            const int external_img_seq = static_cast<int>(dim(initial, 1));
            if (external_height > 0 || external_width > 0) {
                if (external_height <= 0 || external_width <= 0 ||
                    external_height % 16 != 0 || external_width % 16 != 0) {
                    std::cerr << "--external-height/--external-width must both be positive and divisible by 16\n";
                    return 1;
                }
                const int expected_seq = (external_height / 16) * (external_width / 16);
                if (external_img_seq != expected_seq) {
                    std::cerr << "external flow fixture image_seq_len " << external_img_seq
                              << " does not match requested resolution " << external_width
                              << "x" << external_height << " expected_seq=" << expected_seq << "\n";
                    return 1;
                }
                if (has_tensor(tensors, "external.height_width")) {
                    const Tensor& hw = need(tensors, "external.height_width");
                    if (hw.data.size() < 2 ||
                        static_cast<int>(std::llround(hw.data[0])) != external_height ||
                        static_cast<int>(std::llround(hw.data[1])) != external_width) {
                        std::cerr << "external flow fixture embedded height/width metadata does not match requested resolution\n";
                        return 1;
                    }
                }
            }
            if (!verify_external_schedule_tensors(tensors, external_steps, external_img_seq, 1.0e-4f)) {
                return 1;
            }
            Tensor api_tokens;
            sd_lens_external_flow_loop_desc_t flow_desc;
            if (!run_external_flow_api_check(lens_cond_path,
                                             real_block_transformer,
                                             initial,
                                             model_outputs.data,
                                             external_steps,
                                             external_img_seq,
                                             128,
                                             2880,
                                             api_tokens,
                                             flow_desc)) {
                return 1;
            }
            const int64_t packed_side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(external_img_seq))));
            if (packed_side * packed_side != external_img_seq) {
                std::cerr << "--external-flow-fixture requires square image token count for unpack\n";
                return 1;
            }
            const int64_t latent_h = packed_side * 2;
            const int64_t latent_w = packed_side * 2;
            if (external_height > 0 && (latent_h != external_height / 8 || latent_w != external_width / 8)) {
                std::cerr << "computed public Lens latent shape does not match requested output resolution\n";
                return 1;
            }
            Tensor latent{{1, 32, latent_h, latent_w},
                          std::vector<float>(static_cast<size_t>(32 * latent_h * latent_w), 0.0f)};
            sd_lens_vae_latent_desc_t unpack_desc;
            sd_lens_vae_latent_desc_init(&unpack_desc);
            if (!sd_lens_unpack_vae_latent_f32(api_tokens.data.data(),
                                               static_cast<uint64_t>(api_tokens.data.size()),
                                               1,
                                               32,
                                               latent_h,
                                               latent_w,
                                               latent.data.data(),
                                               static_cast<uint64_t>(latent.data.size()),
                                               &unpack_desc)) {
                std::cerr << "sd_lens_unpack_vae_latent_f32 failed for external flow fixture\n";
                return 1;
            }
            if (has_tensor(tensors, "external.final_packed_python")) {
                float max_diff = 0.0f;
                float mean_diff = 0.0f;
                compare_tensor_with_tolerance(api_tokens,
                                              need(tensors, "external.final_packed_python"),
                                              0.05f,
                                              "Lens external packed-token Python oracle",
                                              max_diff,
                                              mean_diff);
            }
            if (has_tensor(tensors, "external.public_latent_python")) {
                float max_diff = 0.0f;
                float mean_diff = 0.0f;
                compare_tensor_with_tolerance(latent,
                                              need(tensors, "external.public_latent_python"),
                                              0.05f,
                                              "Lens external public-latent Python oracle",
                                              max_diff,
                                              mean_diff);
            }
            float latent_max = 0.0f;
            float latent_mean = 0.0f;
            if (!tensor_stats(latent, latent_max, latent_mean) || !write_f32_npy(tiny_denoise_npy, latent)) {
                std::cerr << "failed to write external Lens flow latent NPY\n";
                return 1;
            }
            std::cout << "Lens external flow fixture wrote " << tiny_denoise_npy
                      << " shape=1x32x" << dim(latent, 2) << "x" << dim(latent, 3)
                      << " steps=" << external_steps
                      << " image_seq_len=" << external_img_seq
                      << " max_abs=" << latent_max
                      << " mean_abs=" << latent_mean
                      << " public_api_flow_loop=true external_trace=true\n";
            return 0;
        }
        if (modulation_microbench) {
            if (real_block_cuda_oracle_fixture.empty() || real_block_transformer.empty()) {
                std::cerr << "--modulation-microbench requires --real-block-cuda-oracle-fixture and --real-block-transformer\n";
                return 2;
            }
            if (real_block_index != 0) {
                std::cerr << "--modulation-microbench currently validates block 0 only\n";
                return 2;
            }
            auto block_oracle = load_fixture(real_block_cuda_oracle_fixture);
            const Tensor& input_temb = need(block_oracle, "input.temb");
            if (input_temb.shape.size() != 2 || dim(input_temb, 0) != 1 || dim(input_temb, 1) != 1536) {
                std::cerr << "modulation microbench input.temb must be [1,1536]\n";
                return 1;
            }
            auto tensors = load_real_lens_block(real_block_transformer, real_block_index, max_real_block_bytes);
            const std::string prefix = "transformer_blocks." + std::to_string(real_block_index) + ".";
            const Tensor& img_w = need(tensors, prefix + "img_mod.1.weight");
            const Tensor& img_b = need(tensors, prefix + "img_mod.1.bias");
            const Tensor& txt_w = need(tensors, prefix + "txt_mod.1.weight");
            const Tensor& txt_b = need(tensors, prefix + "txt_mod.1.bias");

            Tensor silu_f32 = silu_tensor(input_temb);
            Tensor silu_bf16 = round_tensor_bf16(silu_f32);
            Tensor img_current = linear2(silu_f32, img_w, &img_b);
            Tensor txt_current = linear2(silu_f32, txt_w, &txt_b);
            Tensor img_silu_bf16_f32_linear = linear2(silu_bf16, img_w, &img_b);
            Tensor txt_silu_bf16_f32_linear = linear2(silu_bf16, txt_w, &txt_b);

            std::filesystem::path out_prefix(modulation_out_prefix);
            if (out_prefix.has_parent_path()) {
                std::filesystem::create_directories(out_prefix.parent_path());
            }
            if (!write_f32_npy(modulation_out_prefix + "_silu_f32.npy", silu_f32) ||
                !write_f32_npy(modulation_out_prefix + "_silu_bf16.npy", silu_bf16) ||
                !write_f32_npy(modulation_out_prefix + "_img_current_f32.npy", img_current) ||
                !write_f32_npy(modulation_out_prefix + "_txt_current_f32.npy", txt_current) ||
                !write_f32_npy(modulation_out_prefix + "_img_silu_bf16_f32_linear.npy", img_silu_bf16_f32_linear) ||
                !write_f32_npy(modulation_out_prefix + "_txt_silu_bf16_f32_linear.npy", txt_silu_bf16_f32_linear)) {
                std::cerr << "failed to write modulation microbench NPY output\n";
                return 1;
            }
#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
            int img_algo = -1;
            int txt_algo = -1;
            double img_ms = 0.0;
            double txt_ms = 0.0;
            Tensor img_cublaslt = linear2_cublaslt_bf16_bias(silu_bf16, img_w, img_b, "img_mod", &img_algo, &img_ms);
            Tensor txt_cublaslt = linear2_cublaslt_bf16_bias(silu_bf16, txt_w, txt_b, "txt_mod", &txt_algo, &txt_ms);
            if (!write_f32_npy(modulation_out_prefix + "_img_cublaslt_bf16.npy", img_cublaslt) ||
                !write_f32_npy(modulation_out_prefix + "_txt_cublaslt_bf16.npy", txt_cublaslt)) {
                std::cerr << "failed to write cuBLASLt modulation microbench NPY output\n";
                return 1;
            }
            std::cout << "Lens modulation microbench: block=0"
                      << " out_prefix=" << modulation_out_prefix
                      << " cublaslt=true"
                      << " img_algo=" << img_algo
                      << " txt_algo=" << txt_algo
                      << " img_ms=" << img_ms
                      << " txt_ms=" << txt_ms
                      << " tensors_written=8\n";
#else
            std::cout << "Lens modulation microbench: block=0"
                      << " out_prefix=" << modulation_out_prefix
                      << " cublaslt=false"
                      << " tensors_written=6\n";
#endif
            return 0;
        }
        if (mlp_residual_microbench) {
            if (real_block_transformer.empty()) {
                std::cerr << "--mlp-residual-microbench requires --real-block-transformer\n";
                return 2;
            }
            if (real_block_index != 0) {
                std::cerr << "--mlp-residual-microbench currently validates block 0 only\n";
                return 2;
            }
#ifndef SD_LENS_TRANSFORMER_USE_CUBLASLT
            std::cerr << "--mlp-residual-microbench requires cuBLASLt support\n";
            return 2;
#else
            auto fixture_tensors = load_fixture(mlp_residual_fixture);
            auto block_weights = load_real_lens_block(real_block_transformer, real_block_index, max_real_block_bytes);
            const std::string prefix = "transformer_blocks." + std::to_string(real_block_index) + ".";
            const Tensor& hidden_after_attn = need(fixture_tensors, "input.hidden_after_attn");
            const Tensor& encoder_after_attn = need(fixture_tensors, "input.encoder_after_attn");
            const Tensor& img_mod2 = need(fixture_tensors, "input.img_mod2");
            const Tensor& txt_mod2 = need(fixture_tensors, "input.txt_mod2");
            constexpr float eps = 1.0e-6f;

            auto report_pair = [](const std::string& label, const Tensor& actual, const Tensor& expected) {
                const DiffStats stats = diff_stats(actual, expected);
                print_diff_stats(label, 0, stats);
            };

            Tensor img_norm2_current = rms_norm3(hidden_after_attn, need(block_weights, prefix + "img_norm2.weight"), eps);
            Tensor txt_norm2_current = rms_norm3(encoder_after_attn, need(block_weights, prefix + "txt_norm2.weight"), eps);
            auto img_modulated2_current = modulate(img_norm2_current, img_mod2, 0);
            auto txt_modulated2_current = modulate(txt_norm2_current, txt_mod2, 0);
            Tensor img_mlp_current = gate_mlp(img_modulated2_current.first,
                                              need(block_weights, prefix + "img_mlp.w1.weight"),
                                              need(block_weights, prefix + "img_mlp.w2.weight"),
                                              need(block_weights, prefix + "img_mlp.w3.weight"));
            Tensor txt_mlp_current = gate_mlp(txt_modulated2_current.first,
                                              need(block_weights, prefix + "txt_mlp.w1.weight"),
                                              need(block_weights, prefix + "txt_mlp.w2.weight"),
                                              need(block_weights, prefix + "txt_mlp.w3.weight"));
            Tensor hidden_current = add_gated(hidden_after_attn, img_mlp_current, img_modulated2_current.second);
            Tensor encoder_current = add_gated(encoder_after_attn, txt_mlp_current, txt_modulated2_current.second);

            double img_mlp_ms = 0.0;
            double txt_mlp_ms = 0.0;
            Tensor img_norm2_bf16 = rms_norm3_bf16_step(hidden_after_attn, need(block_weights, prefix + "img_norm2.weight"), eps);
            Tensor txt_norm2_bf16 = rms_norm3_bf16_step(encoder_after_attn, need(block_weights, prefix + "txt_norm2.weight"), eps);
            auto img_modulated2_bf16 = modulate_bf16_step(img_norm2_bf16, img_mod2);
            auto txt_modulated2_bf16 = modulate_bf16_step(txt_norm2_bf16, txt_mod2);
            Tensor img_mlp_bf16 = gate_mlp_bf16_step(img_modulated2_bf16.first,
                                                     need(block_weights, prefix + "img_mlp.w1.weight"),
                                                     need(block_weights, prefix + "img_mlp.w2.weight"),
                                                     need(block_weights, prefix + "img_mlp.w3.weight"),
                                                     "img_mlp",
                                                     &img_mlp_ms);
            Tensor txt_mlp_bf16 = gate_mlp_bf16_step(txt_modulated2_bf16.first,
                                                     need(block_weights, prefix + "txt_mlp.w1.weight"),
                                                     need(block_weights, prefix + "txt_mlp.w2.weight"),
                                                     need(block_weights, prefix + "txt_mlp.w3.weight"),
                                                     "txt_mlp",
                                                     &txt_mlp_ms);
            Tensor hidden_bf16 = add_gated_bf16_step(hidden_after_attn, img_mlp_bf16, img_modulated2_bf16.second);
            Tensor encoder_bf16 = add_gated_bf16_step(encoder_after_attn, txt_mlp_bf16, txt_modulated2_bf16.second);

            std::cout << "Lens MLP/residual microbench: block=0"
                      << " fixture=" << mlp_residual_fixture
                      << " bf16_mlp_ms_img=" << img_mlp_ms
                      << " bf16_mlp_ms_txt=" << txt_mlp_ms
                      << "\n";
            report_pair("Lens MLP current img_norm2:", img_norm2_current, need(fixture_tensors, "expected.img_norm2"));
            report_pair("Lens MLP bf16 img_norm2:", img_norm2_bf16, need(fixture_tensors, "expected.img_norm2"));
            report_pair("Lens MLP current txt_norm2:", txt_norm2_current, need(fixture_tensors, "expected.txt_norm2"));
            report_pair("Lens MLP bf16 txt_norm2:", txt_norm2_bf16, need(fixture_tensors, "expected.txt_norm2"));
            report_pair("Lens MLP current img_modulated2:", img_modulated2_current.first, need(fixture_tensors, "expected.img_modulated2"));
            report_pair("Lens MLP bf16 img_modulated2:", img_modulated2_bf16.first, need(fixture_tensors, "expected.img_modulated2"));
            report_pair("Lens MLP current txt_modulated2:", txt_modulated2_current.first, need(fixture_tensors, "expected.txt_modulated2"));
            report_pair("Lens MLP bf16 txt_modulated2:", txt_modulated2_bf16.first, need(fixture_tensors, "expected.txt_modulated2"));
            report_pair("Lens MLP current img_mlp:", img_mlp_current, need(fixture_tensors, "expected.img_mlp"));
            report_pair("Lens MLP bf16 img_mlp:", img_mlp_bf16, need(fixture_tensors, "expected.img_mlp"));
            report_pair("Lens MLP current txt_mlp:", txt_mlp_current, need(fixture_tensors, "expected.txt_mlp"));
            report_pair("Lens MLP bf16 txt_mlp:", txt_mlp_bf16, need(fixture_tensors, "expected.txt_mlp"));
            report_pair("Lens MLP current final hidden:", hidden_current, need(fixture_tensors, "expected.final.hidden"));
            report_pair("Lens MLP bf16 final hidden:", hidden_bf16, need(fixture_tensors, "expected.final.hidden"));
            report_pair("Lens MLP current final encoder:", encoder_current, need(fixture_tensors, "expected.final.encoder"));
            report_pair("Lens MLP bf16 final encoder:", encoder_bf16, need(fixture_tensors, "expected.final.encoder"));
            return 0;
#endif
        }
        if (txt_mlp_linear_microbench) {
#ifndef SD_LENS_TRANSFORMER_USE_CUBLASLT
            std::cerr << "--txt-mlp-linear-microbench requires cuBLASLt support\n";
            return 2;
#else
            auto tensors = load_fixture(txt_mlp_linear_fixture);
            const Tensor& txt_modulated2 = need(tensors, "input.txt_modulated2");
            const Tensor& w1 = need(tensors, "weight.txt_mlp.w1");
            const Tensor& w3 = need(tensors, "weight.txt_mlp.w3");
            const Tensor& w2 = need(tensors, "weight.txt_mlp.w2");

            auto report = [](const std::string& label, const Tensor& actual, const Tensor& expected) {
                const DiffStats stats = diff_stats(actual, expected);
                print_diff_stats(label, 0, stats);
            };

            Tensor current_w1 = linear3(txt_modulated2, w1, nullptr);
            Tensor current_w3 = linear3(txt_modulated2, w3, nullptr);
            Tensor current_silu = current_w1;
            for (float& v : current_silu.data) {
                v = silu(v);
            }
            Tensor current_gated = current_silu;
            for (size_t i = 0; i < current_gated.data.size(); ++i) {
                current_gated.data[i] *= current_w3.data[i];
            }
            Tensor current_mlp = linear3(current_gated, w2, nullptr);

            double w1_ms = 0.0;
            double w3_ms = 0.0;
            double w2_ms = 0.0;
            int w1_algo = -1;
            int w3_algo = -1;
            int w2_algo = -1;
            Tensor bf16_w1 = linear3_cublaslt_bf16(txt_modulated2, w1, nullptr, "txt_mlp.w1", &w1_algo, &w1_ms);
            Tensor bf16_w3 = linear3_cublaslt_bf16(txt_modulated2, w3, nullptr, "txt_mlp.w3", &w3_algo, &w3_ms);
            Tensor bf16_silu = round_tensor_bf16(bf16_w1);
            for (float& v : bf16_silu.data) {
                Tensor s{{1}, {silu(v)}};
                v = round_tensor_bf16(s).data[0];
            }
            Tensor bf16_gated = bf16_silu;
            Tensor bf16_w3_rounded = round_tensor_bf16(bf16_w3);
            for (size_t i = 0; i < bf16_gated.data.size(); ++i) {
                Tensor prod{{1}, {bf16_gated.data[i] * bf16_w3_rounded.data[i]}};
                bf16_gated.data[i] = round_tensor_bf16(prod).data[0];
            }
            Tensor bf16_mlp = linear3_cublaslt_bf16(bf16_gated, w2, nullptr, "txt_mlp.w2", &w2_algo, &w2_ms);

            Tensor packed_input = txt_modulated2;
            Tensor packed_w1 = w1;
            Tensor packed_w3 = w3;
            Tensor packed_w2 = w2;
            Tensor packed_w1_out = linear3_cublaslt_bf16(packed_input, packed_w1, nullptr, "txt_mlp.w1.packed", nullptr, nullptr);
            Tensor transposed_note{{1}, {0.0f}};
            bool transpose_applicable = false;
            try {
                Tensor transposed = transpose2(w1);
                (void)linear3_cublaslt_bf16(txt_modulated2, transposed, nullptr, "txt_mlp.w1.transposed", nullptr, nullptr);
                transpose_applicable = true;
                transposed_note.data[0] = 1.0f;
            } catch (...) {
                transpose_applicable = false;
            }

            std::cout << "Lens txt_mlp linear microbench: fixture=" << txt_mlp_linear_fixture
                      << " w1_algo=" << w1_algo
                      << " w3_algo=" << w3_algo
                      << " w2_algo=" << w2_algo
                      << " w1_ms=" << w1_ms
                      << " w3_ms=" << w3_ms
                      << " w2_ms=" << w2_ms
                      << " transpose_variant_applicable=" << (transpose_applicable ? "true" : "false")
                      << "\n";
            report("Lens txt_mlp current w1:", current_w1, need(tensors, "expected.txt_w1"));
            report("Lens txt_mlp bf16 w1:", bf16_w1, need(tensors, "expected.txt_w1"));
            report("Lens txt_mlp packed-copy w1:", packed_w1_out, need(tensors, "expected.txt_w1"));
            report("Lens txt_mlp current w3:", current_w3, need(tensors, "expected.txt_w3"));
            report("Lens txt_mlp bf16 w3:", bf16_w3, need(tensors, "expected.txt_w3"));
            report("Lens txt_mlp current silu_w1:", current_silu, need(tensors, "expected.txt_silu_w1"));
            report("Lens txt_mlp bf16 silu_w1:", bf16_silu, need(tensors, "expected.txt_silu_w1"));
            report("Lens txt_mlp current gated:", current_gated, need(tensors, "expected.txt_gated"));
            report("Lens txt_mlp bf16 gated:", bf16_gated, need(tensors, "expected.txt_gated"));
            report("Lens txt_mlp current w2/mlp:", current_mlp, need(tensors, "expected.txt_mlp"));
            report("Lens txt_mlp bf16 w2/mlp:", bf16_mlp, need(tensors, "expected.txt_mlp"));
            return 0;
#endif
        }
        if (!fixture.empty()) {
            auto tensors = load_fixture(fixture);
            auto outputs = lens_block_forward(tensors);
            float mean_encoder = 0.0f;
            float mean_hidden = 0.0f;
            const float max_encoder = max_abs_diff(outputs.first, need(tensors, "expected.encoder"), &mean_encoder);
            const float max_hidden = max_abs_diff(outputs.second, need(tensors, "expected.hidden"), &mean_hidden);
            std::cout << "Lens transformer block parity: max_encoder=" << max_encoder
                      << " mean_encoder=" << mean_encoder
                      << " max_hidden=" << max_hidden
                      << " mean_hidden=" << mean_hidden << "\n";
            if (max_encoder > tolerance || max_hidden > tolerance) {
                std::cerr << "Lens transformer block parity exceeded tolerance " << tolerance << "\n";
                return 1;
            }
            if (has_tensor(tensors, "full.expected.output")) {
                Tensor full_output = lens_full_forward(tensors);
                float mean_full = 0.0f;
                const float max_full = max_abs_diff(full_output, need(tensors, "full.expected.output"), &mean_full);
                std::cout << "Lens tiny full-transformer parity: max_output=" << max_full
                          << " mean_output=" << mean_full << "\n";
                if (max_full > tolerance) {
                    std::cerr << "Lens tiny full-transformer parity exceeded tolerance " << tolerance << "\n";
                    return 1;
                }
            }
        }
        if (!inspect_transformer.empty()) {
            sd_ctx_t* ctx = new_sd_ctx_lens_conditioning_only();
            if (ctx == nullptr) {
                std::cerr << "failed to create Lens metadata context\n";
                return 1;
            }
            sd_lens_transformer_options_t options;
            sd_lens_transformer_options_init(&options);
            sd_lens_transformer_handle_t handle = 0;
            sd_lens_transformer_desc_t desc;
            sd_lens_transformer_desc_init(&desc);
            if (!sd_lens_transformer_load(ctx, inspect_transformer.c_str(), &options, &handle, &desc)) {
                std::cerr << "sd_lens_transformer_load failed for " << inspect_transformer << "\n";
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "Lens transformer metadata: handle=" << handle
                      << " shards=" << desc.shard_count
                      << " tensors=" << desc.tensor_count
                      << " bytes=" << desc.estimated_bytes
                      << " layers=" << desc.layers
                      << " heads=" << desc.num_attention_heads
                      << " head_dim=" << desc.attention_head_dim
                      << " inner_dim=" << desc.inner_dim
                      << " metadata_only=" << (desc.metadata_only ? "true" : "false")
                      << " forward_supported=" << (desc.forward_supported ? "true" : "false")
                      << "\n";
            sd_lens_transformer_desc_t desc_again;
            sd_lens_transformer_desc_init(&desc_again);
            if (!sd_lens_transformer_get_desc(ctx, handle, &desc_again) ||
                desc_again.tensor_count != desc.tensor_count ||
                desc_again.estimated_bytes != desc.estimated_bytes) {
                std::cerr << "sd_lens_transformer_get_desc returned inconsistent metadata\n";
                free_sd_ctx(ctx);
                return 1;
            }
            if (!sd_lens_transformer_release(ctx, handle)) {
                std::cerr << "sd_lens_transformer_release failed\n";
                free_sd_ctx(ctx);
                return 1;
            }
            if (sd_lens_transformer_get_desc(ctx, handle, &desc_again)) {
                std::cerr << "released Lens transformer handle remained visible\n";
                free_sd_ctx(ctx);
                return 1;
            }
            free_sd_ctx(ctx);
        }
        if (!real_block_transformer.empty()) {
            if (native_cuda_generate_256) {
#ifndef SD_USE_CUDA
                std::cerr << "Lens native CUDA generation requires SD_USE_CUDA build\n";
                return 1;
#else
                const bool use_in_memory_conditioning = g_lens_in_memory_cond_tensors != nullptr;
                if (lens_cond_path.empty() && !use_in_memory_conditioning) {
                    std::cerr << "--native-cuda-generate-256 requires --lens-cond or in-memory conditioning\n";
                    return 2;
                }
                if (tiny_flow_steps <= 0) {
                    std::cerr << "--native-cuda-generate-256 requires positive --tiny-flow-steps\n";
                    return 2;
                }
                const int64_t image_h = external_height > 0 ? external_height : 256;
                const int64_t image_w = external_width > 0 ? external_width : 256;
                if (image_h <= 0 || image_w <= 0 || image_h % 16 != 0 || image_w % 16 != 0) {
                    std::cerr << "--native-cuda-generate-256 requires positive external dimensions divisible by 16\n";
                    return 2;
                }
                const int64_t packed_h = image_h / 16;
                const int64_t packed_w = image_w / 16;
                const int64_t image_seq_len = packed_h * packed_w;

                sd_ctx_t* lens_ctx = nullptr;
                sd_lens_transformer_handle_t transformer_handle = 0;
                sd_lens_transformer_desc_t transformer_desc;
                sd_lens_transformer_desc_init(&transformer_desc);
                double context_load_seconds = 0.0;
                const bool can_keep_warm_transformer =
                    keep_transformer_warm &&
                    use_transformer_context &&
                    transformer_residency == "gpu-full-bf16";
                const std::string warm_transformer_key =
                    make_lens_warm_transformer_cache_key(real_block_transformer, transformer_residency);
                bool warm_transformer_reused = false;
                if (use_transformer_context) {
                    print_transformer_vram_snapshot("before_transformer_load");
                    if (can_keep_warm_transformer &&
                        g_lens_warm_transformer_cache.valid &&
                        g_lens_warm_transformer_cache.key == warm_transformer_key &&
                        g_lens_warm_transformer_cache.lens_ctx != nullptr &&
                        g_lens_warm_transformer_cache.transformer_handle != 0 &&
                        g_lens_warm_transformer_cache.cuda_backend != nullptr &&
                        !g_lens_warm_transformer_cache.resident_blocks.empty()) {
                        lens_ctx = g_lens_warm_transformer_cache.lens_ctx;
                        transformer_handle = g_lens_warm_transformer_cache.transformer_handle;
                        transformer_desc = g_lens_warm_transformer_cache.transformer_desc;
                        warm_transformer_reused = true;
                        std::cout << "Lens transformer warm cache reused:"
                                  << " handle=" << transformer_handle
                                  << " host_cached=true"
                                  << " gpu_resident=partial"
                                  << " resident_weight_bytes=" << g_lens_warm_transformer_cache.resident_weight_bytes
                                  << " resident_weight_tensors=" << g_lens_warm_transformer_cache.resident_weight_tensors
                                  << " load_seconds=0\n";
                    } else {
                        if (can_keep_warm_transformer &&
                            g_lens_warm_transformer_cache.valid &&
                            g_lens_warm_transformer_cache.key != warm_transformer_key) {
                            g_lens_warm_transformer_cache.clear();
                        }
                        lens_ctx = new_sd_ctx_lens_conditioning_only();
                        if (lens_ctx == nullptr) {
                            std::cerr << "failed to create Lens transformer host-cache context\n";
                            return 1;
                        }
                        sd_lens_transformer_options_t transformer_options;
                        sd_lens_transformer_options_init(&transformer_options);
                        transformer_options.metadata_only = false;
                        transformer_options.allow_unsafe_large_allocations = true;
                        transformer_options.max_resident_weight_bytes = 0;
                        const auto ctx_load_start = std::chrono::steady_clock::now();
                        if (!sd_lens_transformer_load(lens_ctx,
                                                      real_block_transformer.c_str(),
                                                      &transformer_options,
                                                      &transformer_handle,
                                                      &transformer_desc)) {
                            free_sd_ctx(lens_ctx);
                            std::cerr << "sd_lens_transformer_load failed for host-cached A3 context\n";
                            return 1;
                        }
                        const auto ctx_load_end = std::chrono::steady_clock::now();
                        context_load_seconds = std::chrono::duration<double>(ctx_load_end - ctx_load_start).count();
                        std::cout << "Lens transformer context loaded:"
                                  << " handle=" << transformer_handle
                                  << " host_cached=" << (transformer_desc.host_cached ? "true" : "false")
                                  << " gpu_resident=" << (transformer_desc.gpu_resident ? "true" : "false")
                                  << " weights_loaded=" << (transformer_desc.weights_loaded ? "true" : "false")
                                  << " forward_supported=" << (transformer_desc.forward_supported ? "true" : "false")
                                  << " tensors=" << transformer_desc.tensor_count
                                  << " estimated_bytes=" << transformer_desc.estimated_bytes
                                  << " host_cached_bytes=" << transformer_desc.host_cached_bytes
                                  << " load_seconds=" << context_load_seconds << "\n";
                    }
                    print_transformer_vram_snapshot("after_transformer_load");
                }
                const std::vector<std::string> top_names = {
                    "img_in.weight",
                    "img_in.bias",
                    "txt_norm.0.weight",
                    "txt_norm.1.weight",
                    "txt_norm.2.weight",
                    "txt_norm.3.weight",
                    "txt_in.weight",
                    "txt_in.bias",
                    "time_text_embed.timestep_embedder.linear_1.weight",
                    "time_text_embed.timestep_embedder.linear_1.bias",
                    "time_text_embed.timestep_embedder.linear_2.weight",
                    "time_text_embed.timestep_embedder.linear_2.bias",
                    "norm_out.linear.weight",
                    "norm_out.linear.bias",
                    "proj_out.weight",
                    "proj_out.bias",
                };
                std::unordered_map<std::string, Tensor> top_tensors;
                std::unordered_map<std::string, TensorInput> top_borrowed;
                if (use_transformer_context) {
                    top_borrowed = borrow_cached_lens_named_tensors(lens_ctx, transformer_handle, top_names);
                } else {
                    top_tensors = load_real_lens_named_tensors(real_block_transformer, top_names, max_real_top_bytes);
                }
                auto top_dim = [&](const std::string& name, size_t index) -> int64_t {
                    if (use_transformer_context) {
                        auto it = top_borrowed.find(name);
                        if (it == top_borrowed.end() || index >= it->second.shape.size()) {
                            throw std::runtime_error("missing borrowed top tensor: " + name);
                        }
                        return it->second.shape[index];
                    }
                    return dim(need(top_tensors, name), index);
                };
                std::unordered_map<std::string, Tensor> cond_tensors_storage;
                const std::unordered_map<std::string, Tensor>* cond_tensors_ptr = nullptr;
                if (use_in_memory_conditioning) {
                    cond_tensors_ptr = g_lens_in_memory_cond_tensors;
                } else {
                    cond_tensors_storage = load_real_lens_named_tensors(lens_cond_path,
                                                                        {"feature_0", "feature_1", "feature_2", "feature_3", "attention_mask"},
                                                                        max_real_cond_bytes,
                                                                        true);
                    cond_tensors_ptr = &cond_tensors_storage;
                }
                const auto& cond_tensors = *cond_tensors_ptr;
                const Tensor& f0 = need(cond_tensors, "feature_0");
                if (dim(f0, 0) != 1 || dim(f0, 2) != top_dim("txt_norm.0.weight", 0)) {
                    std::cerr << "lens_cond_v1 feature_0 does not match Lens transformer text hidden size\n";
                    return 1;
                }
                const int64_t text_seq_len = dim(f0, 1);
                for (int i = 1; i < 4; ++i) {
                    const Tensor& f = need(cond_tensors, "feature_" + std::to_string(i));
                    if (f.shape != f0.shape) {
                        std::cerr << "lens_cond_v1 feature shapes do not match\n";
                        return 1;
                    }
                }
                const Tensor& cond_mask = need(cond_tensors, "attention_mask");
                if (dim(cond_mask, 0) != 1 || dim(cond_mask, 1) != text_seq_len) {
                    std::cerr << "lens_cond_v1 attention_mask shape does not match feature sequence length\n";
                    return 1;
                }

                std::vector<float> sigmas(static_cast<size_t>(tiny_flow_steps + 1), 0.0f);
                std::vector<float> timesteps(static_cast<size_t>(tiny_flow_steps), 0.0f);
                sd_lens_schedule_options_t schedule_options;
                sd_lens_schedule_options_init(&schedule_options);
                schedule_options.steps = tiny_flow_steps;
                schedule_options.image_seq_len = static_cast<int>(image_seq_len);
                sd_lens_schedule_desc_t schedule_desc;
                sd_lens_schedule_desc_init(&schedule_desc);
                if (!sd_lens_turbo_build_schedule(&schedule_options,
                                                  sigmas.data(),
                                                  static_cast<uint32_t>(sigmas.size()),
                                                  timesteps.data(),
                                                  static_cast<uint32_t>(timesteps.size()),
                                                  &schedule_desc)) {
                    std::cerr << "sd_lens_turbo_build_schedule failed for native CUDA generation\n";
                    return 1;
                }

                ggml_backend_t cuda_backend =
                    warm_transformer_reused ? g_lens_warm_transformer_cache.cuda_backend
                                            : ggml_backend_cuda_init(0);
                if (cuda_backend == nullptr) {
                    if (!warm_transformer_reused && lens_ctx != nullptr) {
                        if (transformer_handle != 0) {
                            sd_lens_transformer_release(lens_ctx, transformer_handle);
                        }
                        free_sd_ctx(lens_ctx);
                    }
                    std::cerr << "ggml_backend_cuda_init(0) failed\n";
                    return 1;
                }
                const auto total_start = std::chrono::steady_clock::now();
                double transformer_seconds = 0.0;
                double flow_seconds = 0.0;
                double unpack_seconds = 0.0;
                std::vector<double> per_step_transformer_seconds;
                per_step_transformer_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                std::vector<double> per_step_block_fetch_seconds;
                std::vector<double> per_step_runner_setup_seconds;
                std::vector<double> per_step_runner_input_copy_seconds;
                std::vector<double> per_step_runner_compute_seconds;
                std::vector<double> per_step_runner_graph_seconds;
                per_step_block_fetch_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                per_step_runner_setup_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                per_step_runner_input_copy_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                per_step_runner_compute_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                per_step_runner_graph_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                RunnerTimingTotals runner_timing_totals;
                std::vector<BlockTimingTotals> block_timing_totals(48);
                double block_fetch_seconds = 0.0;
                double runner_setup_seconds = 0.0;
                double resident_upload_seconds = 0.0;
                uint64_t resident_weight_bytes = 0;
                size_t resident_weight_tensors = 0;
                uint64_t total_streamed_bytes = 0;
                size_t total_streamed_tensors = 0;
                std::vector<LensResidentBlockWeights> local_resident_blocks;
                std::vector<LensResidentBlockWeights>& resident_blocks =
                    warm_transformer_reused ? g_lens_warm_transformer_cache.resident_blocks
                                            : local_resident_blocks;
                std::unique_ptr<LensBlockCudaRunner> reusable_block0_runner;
                std::vector<BlockStepTiming> block0_step_timings;
                block0_step_timings.reserve(static_cast<size_t>(tiny_flow_steps));
                const bool reuse_block0_runner = transformer_runner_reuse == "block0";
                const bool alias_resident_weights =
                    transformer_residency == "persistent-block" ||
                    transformer_residency == "persistent-blocks" ||
                    transformer_residency == "gpu-full-bf16";
                const bool resident_weights_bf16 = transformer_residency == "gpu-full-bf16";
                if (warm_transformer_reused) {
                    resident_upload_seconds = 0.0;
                    resident_weight_bytes = g_lens_warm_transformer_cache.resident_weight_bytes;
                    resident_weight_tensors = g_lens_warm_transformer_cache.resident_weight_tensors;
                } else if (transformer_residency == "gpu-window" ||
                           transformer_residency == "persistent-block" ||
                           transformer_residency == "persistent-blocks" ||
                           transformer_residency == "gpu-full-bf16") {
                    if (!use_transformer_context) {
                        std::cerr << "--transformer-residency " << transformer_residency << " requires --use-transformer-context\n";
                        if (!warm_transformer_reused && cuda_backend != nullptr) {
                            ggml_backend_free(cuda_backend);
                        }
                        if (!warm_transformer_reused && lens_ctx != nullptr) {
                            if (transformer_handle != 0) {
                                sd_lens_transformer_release(lens_ctx, transformer_handle);
                            }
                            free_sd_ctx(lens_ctx);
                        }
                        return 2;
                    }
                    int resident_count = 0;
                    if (transformer_residency == "gpu-window") {
                        resident_count = std::min(window_blocks, 48);
                    } else if (transformer_residency == "persistent-block") {
                        resident_count = 1;
                    } else if (transformer_residency == "gpu-full-bf16") {
                        resident_count = 48;
                    } else {
                        resident_count = std::min(persistent_blocks, 48);
                    }
                    if (transformer_residency == "persistent-blocks" || transformer_residency == "gpu-full-bf16") {
                        uint64_t estimated_bytes = 0;
                        size_t estimated_tensors = 0;
                        for (int block_index = 0; block_index < resident_count; ++block_index) {
                            const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                            auto block_borrowed = borrow_cached_lens_block(lens_ctx, transformer_handle, block_index);
                            for (const auto& item : block_borrowed) {
                                if (item.first.rfind(prefix, 0) != 0) {
                                    continue;
                                }
                                const TensorInput& borrowed = item.second;
                                const bool use_bf16_tensor = resident_weights_bf16 && borrowed.shape.size() == 2;
                                estimated_bytes += elem_count(borrowed.shape) * (use_bf16_tensor ? sizeof(ggml_bf16_t) : sizeof(float));
                                estimated_tensors += 1;
                            }
                        }
                        const uint64_t cap_bytes = persistent_blocks_memory_mib * 1024ull * 1024ull;
                        double used_gib = 0.0;
                        double free_gib = 0.0;
                        double total_gib = 0.0;
                        const bool have_vram = query_nvidia_smi_vram(used_gib, free_gib, total_gib);
                        std::cout << "Lens transformer resident estimate:"
                                  << " mode=" << transformer_residency
                                  << " blocks=" << resident_count
                                  << " tensors=" << estimated_tensors
                                  << " bytes=" << estimated_bytes
                                  << " dtype=" << (resident_weights_bf16 ? "bf16" : "f32");
                        if (transformer_residency == "persistent-blocks") {
                            std::cout << " cap_mib=" << persistent_blocks_memory_mib;
                        }
                        if (have_vram) {
                            std::cout << " vram_free_gib=" << free_gib
                                      << " vram_total_gib=" << total_gib;
                        } else {
                            std::cout << " vram_free_gib=<unavailable>";
                        }
                        std::cout << "\n";
                        if (transformer_residency == "persistent-blocks" && estimated_bytes > cap_bytes) {
                            std::cerr << "--persistent-blocks selected "
                                      << estimated_bytes
                                      << " bytes, exceeding cap "
                                      << cap_bytes
                                      << " bytes\n";
                            if (!warm_transformer_reused && cuda_backend != nullptr) {
                                ggml_backend_free(cuda_backend);
                            }
                            if (!warm_transformer_reused && lens_ctx != nullptr) {
                                if (transformer_handle != 0) {
                                    sd_lens_transformer_release(lens_ctx, transformer_handle);
                                }
                                free_sd_ctx(lens_ctx);
                            }
                            return 2;
                        }
                        if (have_vram && transformer_residency == "persistent-blocks") {
                            const double estimated_mib = static_cast<double>(estimated_bytes) / (1024.0 * 1024.0);
                            const double free_mib = free_gib * 1024.0;
                            const double kSafetyMarginMib = 1024.0;
                            if (free_mib < estimated_mib + kSafetyMarginMib) {
                                std::cerr << "--persistent-blocks needs about "
                                          << estimated_mib
                                          << " MiB plus "
                                          << kSafetyMarginMib
                                          << " MiB safety margin, but only "
                                          << free_mib
                                          << " MiB is free\n";
                                if (!warm_transformer_reused && cuda_backend != nullptr) {
                                    ggml_backend_free(cuda_backend);
                                }
                                if (!warm_transformer_reused && lens_ctx != nullptr) {
                                    if (transformer_handle != 0) {
                                        sd_lens_transformer_release(lens_ctx, transformer_handle);
                                    }
                                    free_sd_ctx(lens_ctx);
                                }
                                return 2;
                            }
                        }
                    }
                    resident_blocks.resize(static_cast<size_t>(resident_count));
                    LensBackendResourceUploader uploader(cuda_backend);
                    const auto resident_start = std::chrono::steady_clock::now();
                    for (int block_index = 0; block_index < resident_count; ++block_index) {
                        const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                        auto block_borrowed = borrow_cached_lens_block(lens_ctx, transformer_handle, block_index);
                        LensResidentBlockWeights& resident = resident_blocks[static_cast<size_t>(block_index)];
                        resident.tensors.reserve(block_borrowed.size());
                        for (const auto& item : block_borrowed) {
                            if (item.first.rfind(prefix, 0) != 0) {
                                continue;
                            }
                            const std::string name = item.first.substr(prefix.size());
                            const TensorInput& borrowed = item.second;
                            TensorInput runner_input = borrowed.shape.size() == 2 ? to_input_2d(borrowed) : to_input_1d(borrowed);
                            const bool use_bf16_tensor = resident_weights_bf16 && runner_input.shape.size() == 2;
                            const uint64_t bytes = elem_count(runner_input.shape) * (use_bf16_tensor ? sizeof(ggml_bf16_t) : sizeof(float));
                            auto resource = use_bf16_tensor
                                                ? uploader.upload_bf16_from_f32(runner_input, "lens_resident_block." + std::to_string(block_index) + "." + name)
                                                : uploader.upload_f32(runner_input, "lens_resident_block." + std::to_string(block_index) + "." + name);
                            if (resource == nullptr || resource->empty()) {
                                throw std::runtime_error("failed to upload resident Lens transformer block weight: " + name);
                            }
                            resident.bytes += bytes;
                            resident_weight_bytes += bytes;
                            resident_weight_tensors += 1;
                            resident.tensors.emplace(name, std::move(resource));
                        }
                    }
                    uploader.synchronize();
                    const auto resident_end = std::chrono::steady_clock::now();
                    resident_upload_seconds = std::chrono::duration<double>(resident_end - resident_start).count();
                    std::cout << "Lens transformer residency prepared:"
                              << " mode=" << transformer_residency
                              << " blocks=" << resident_blocks.size()
                              << " tensors=" << resident_weight_tensors
                              << " bytes=" << resident_weight_bytes
                              << " upload_seconds=" << resident_upload_seconds
                              << " alias_direct=" << (alias_resident_weights ? "true" : "false")
                              << " weight_dtype=" << (resident_weights_bf16 ? "bf16" : "f32")
                              << "\n";
                    if (reuse_block0_runner) {
                        if (resident_blocks.empty()) {
                            throw std::runtime_error("--transformer-runner-reuse block0 requires block0 resident weights");
                        }
                        const LensResidentBlockWeights& resident = resident_blocks[0];
                        std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_runner_tensors;
                        resident_runner_tensors.reserve(resident.tensors.size());
                        for (const auto& item : resident.tensors) {
                            resident_runner_tensors.emplace(item.first, item.second.get());
                        }
                        reusable_block0_runner = std::make_unique<LensBlockCudaRunner>(
                            cuda_backend,
                            std::unordered_map<std::string, TensorInput>{},
                            std::move(resident_runner_tensors),
                            image_seq_len,
                            text_seq_len,
                            lens_attention_mode,
                            true);
                        std::cout << "Lens transformer runner reuse prepared:"
                                  << " mode=block0"
                                  << " resident_tensors=" << resident.tensors.size()
                                  << "\n";
                    }
                }
                try {
                    Tensor img_freqs = make_lens_image_rope_freqs(packed_h, packed_w);
                    Tensor txt_freqs = make_lens_text_rope_freqs(text_seq_len, packed_h, packed_w);
                    sd::Tensor<float> img_pe_sd = lens_freqs_to_rope_pe(img_freqs);
                    sd::Tensor<float> txt_pe_sd = lens_freqs_to_rope_pe(txt_freqs);
                    const bool use_static_residency =
                        dynamic_residency == "gpu-streams" &&
                        transformer_residency == "gpu-full-bf16";
                    LensResidentStaticTensors resident_top_static;
                    LensResidentStaticTensors resident_temporal_static;
                    LensResidentStaticTensors resident_rope_static;
                    LensResidentStaticTensors resident_final_static;
                    uint64_t resident_static_bytes = 0;
                    size_t resident_static_tensors = 0;
                    if (use_static_residency) {
                        LensBackendResourceUploader static_uploader(cuda_backend);
                        auto upload_static = [&](LensResidentStaticTensors& dst,
                                                 const std::string& name,
                                                 const TensorInput& input) {
                            const bool use_bf16_tensor = resident_weights_bf16 && input.shape.size() == 2;
                            auto resource = use_bf16_tensor
                                                ? static_uploader.upload_bf16_from_f32(input, "lens_static." + name)
                                                : static_uploader.upload_f32(input, "lens_static." + name);
                            if (resource == nullptr || resource->empty()) {
                                throw std::runtime_error("failed to upload resident Lens static tensor: " + name);
                            }
                            dst.add(name, std::move(resource));
                            resident_static_tensors += 1;
                        };
                        auto top_input_for = [&](const std::string& name) -> TensorInput {
                            if (use_transformer_context) {
                                const TensorInput& t = top_borrowed.at(name);
                                return t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                            }
                            const Tensor& t = need(top_tensors, name);
                            return t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                        };

                        for (int i = 0; i < 4; ++i) {
                            upload_static(resident_top_static,
                                          "feature_" + std::to_string(i),
                                          to_input_bsh_as_csb_view(need(cond_tensors, "feature_" + std::to_string(i))));
                            const std::string norm_name = "txt_norm." + std::to_string(i) + ".weight";
                            upload_static(resident_top_static, norm_name, top_input_for(norm_name));
                        }
                        for (const char* name : {"img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias"}) {
                            upload_static(resident_top_static, name, top_input_for(name));
                        }
                        for (const char* name : {"time_text_embed.timestep_embedder.linear_1.weight",
                                                 "time_text_embed.timestep_embedder.linear_1.bias",
                                                 "time_text_embed.timestep_embedder.linear_2.weight",
                                                 "time_text_embed.timestep_embedder.linear_2.bias"}) {
                            upload_static(resident_temporal_static, name, top_input_for(name));
                        }
                        upload_static(resident_rope_static,
                                      "input.img_pe",
                                      TensorInput{img_pe_sd.shape(), img_pe_sd.data()});
                        upload_static(resident_rope_static,
                                      "input.txt_pe",
                                      TensorInput{txt_pe_sd.shape(), txt_pe_sd.data()});
                        for (const char* name : {"norm_out.linear.weight", "norm_out.linear.bias", "proj_out.weight", "proj_out.bias"}) {
                            upload_static(resident_final_static, name, top_input_for(name));
                        }
                        resident_static_bytes = resident_top_static.bytes +
                                                resident_temporal_static.bytes +
                                                resident_rope_static.bytes +
                                                resident_final_static.bytes;
                        std::cout << "Lens transformer static residency prepared:"
                                  << " mode=top-rope-temb-final"
                                  << " tensors=" << resident_static_tensors
                                  << " bytes=" << resident_static_bytes
                                  << " top_bytes=" << resident_top_static.bytes
                                  << " temporal_bytes=" << resident_temporal_static.bytes
                                  << " rope_bytes=" << resident_rope_static.bytes
                                  << " final_bytes=" << resident_final_static.bytes
                                  << "\n";
                    }
                    for (int generation_index = 0; generation_index < repeat_generations; ++generation_index) {
                        const auto generation_total_start = std::chrono::steady_clock::now();
                        double transformer_seconds = 0.0;
                        double flow_seconds = 0.0;
                        double unpack_seconds = 0.0;
                        std::vector<double> per_step_transformer_seconds;
                        per_step_transformer_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        std::vector<double> per_step_block_fetch_seconds;
                        std::vector<double> per_step_runner_setup_seconds;
                        std::vector<double> per_step_runner_input_copy_seconds;
                        std::vector<double> per_step_runner_input_copy_submit_seconds;
                        std::vector<double> per_step_runner_input_sync_seconds;
                        std::vector<double> per_step_runner_compute_seconds;
                        std::vector<double> per_step_runner_compute_submit_seconds;
                        std::vector<double> per_step_runner_sync_seconds;
                        std::vector<double> per_step_runner_graph_seconds;
                        per_step_block_fetch_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_setup_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_input_copy_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_input_copy_submit_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_input_sync_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_compute_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_compute_submit_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_sync_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        per_step_runner_graph_seconds.reserve(static_cast<size_t>(tiny_flow_steps));
                        RunnerTimingTotals runner_timing_totals;
                        std::vector<BlockTimingTotals> block_timing_totals(48);
                        double block_fetch_seconds = 0.0;
                        double runner_setup_seconds = 0.0;
                        uint64_t total_streamed_bytes = 0;
                        size_t total_streamed_tensors = 0;
                        std::vector<BlockStepTiming> block0_step_timings;
                        block0_step_timings.reserve(static_cast<size_t>(tiny_flow_steps));

                        Tensor current_tokens = make_random_normal_packed_tokens(image_seq_len, 128, static_cast<uint32_t>(seed));
                        Tensor initial_tokens = current_tokens;
                        if (generation_index == 0 &&
                            !initial_packed_tokens_npy.empty() &&
                            !write_f32_npy(initial_packed_tokens_npy, initial_tokens)) {
                            throw std::runtime_error("failed to write initial packed tokens NPY");
                        }
                        std::vector<float> external_model_outputs(current_tokens.data.size() * static_cast<size_t>(tiny_flow_steps), 0.0f);

                        std::cout << "Lens native CUDA generation start:"
                                  << " generation=" << generation_index
                                  << " repeat_generations=" << repeat_generations
                                  << " prompt_conditioning=" << (use_in_memory_conditioning ? "<in_memory_lens_cond_v1>" : lens_cond_path)
                                  << " image=" << image_w << "x" << image_h
                                  << " steps=" << tiny_flow_steps
                                  << " cfg=1.0 seed=" << seed
                                  << " attention_mode=" << lens_attention_mode_name(lens_attention_mode)
                                  << " conditioning_handoff=" << (use_in_memory_conditioning ? "in_memory" : "safetensors")
                                  << " scheduler=lens_compute_empirical_mu_custom_turbo_sigmas"
                                  << " mu=" << schedule_desc.mu
                                  << " python_owned=conditioning_only\n";

                        for (int step_index = 0; step_index < tiny_flow_steps; ++step_index) {
                        const auto step_transformer_start = std::chrono::steady_clock::now();
                        double step_block_fetch_seconds = 0.0;
                        double step_runner_setup_seconds = 0.0;
                        RunnerTimingTotals step_runner_timing_totals;
                        const auto top_setup_start = std::chrono::steady_clock::now();
                        std::unordered_map<std::string, TensorInput> top_runner_tensors;
                        top_runner_tensors["input.packed"] = to_input_bsh_as_csb_view(current_tokens);
                        std::unique_ptr<LensTopCudaRunner> top_runner_ptr;
                        std::unique_ptr<GgmlBackendTensorResource> top_output_resource;
                        Tensor hidden_state;
                        Tensor encoder_state;
                        if (use_static_residency) {
                            top_runner_ptr = std::make_unique<LensTopCudaRunner>(cuda_backend,
                                                                                 std::move(top_runner_tensors),
                                                                                 resident_top_static.ptrs(),
                                                                                 true);
                        } else {
                            for (int i = 0; i < 4; ++i) {
                                top_runner_tensors["feature_" + std::to_string(i)] = to_input_bsh_as_csb_view(need(cond_tensors, "feature_" + std::to_string(i)));
                                const std::string norm_name = "txt_norm." + std::to_string(i) + ".weight";
                                top_runner_tensors[norm_name] = use_transformer_context ? to_input_1d(top_borrowed.at(norm_name))
                                                                                         : to_input_1d(need(top_tensors, norm_name));
                            }
                            for (const char* name : {"img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias"}) {
                                if (use_transformer_context) {
                                    const TensorInput& t = top_borrowed.at(name);
                                    top_runner_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                } else {
                                    const Tensor& t = need(top_tensors, name);
                                    top_runner_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                }
                            }
                            top_runner_ptr = std::make_unique<LensTopCudaRunner>(cuda_backend, std::move(top_runner_tensors));
                        }
                        const auto top_setup_end = std::chrono::steady_clock::now();
                        step_runner_setup_seconds += std::chrono::duration<double>(top_setup_end - top_setup_start).count();
                        if (use_static_residency) {
                            top_output_resource = top_runner_ptr->compute_top_resource_alias(8);
                            if (top_output_resource == nullptr || top_output_resource->empty()) {
                                throw std::runtime_error("Lens native CUDA top-level projection alias failed at step " + std::to_string(step_index));
                            }
                        } else {
                            auto top_output_opt = top_runner_ptr->compute_top(8);
                            if (!top_output_opt.has_value()) {
                                throw std::runtime_error("Lens native CUDA top-level projection failed at step " + std::to_string(step_index));
                            }
                            Tensor combined_top = from_sd_csb_as_bsh(*top_output_opt);
                            hidden_state = slice_seq3(combined_top, 0, image_seq_len);
                            encoder_state = slice_seq3(combined_top, image_seq_len, text_seq_len);
                        }
                        add_runner_timing(step_runner_timing_totals, top_runner_ptr->get_last_timing_profile());

                        const auto temb_setup_start = std::chrono::steady_clock::now();
                        std::unordered_map<std::string, TensorInput> temb_tensors;
                        Tensor timestep_proj = make_lens_timestep_proj(timesteps[static_cast<size_t>(step_index)] / 1000.0f);
                        temb_tensors["input.timestep_proj"] = to_input_bh_as_hb_view(timestep_proj);
                        std::unique_ptr<LensTimestepCudaRunner> temb_runner_ptr;
                        std::unique_ptr<GgmlBackendTensorResource> temb_resource;
                        Tensor temb;
                        if (use_static_residency) {
                            temb_runner_ptr = std::make_unique<LensTimestepCudaRunner>(cuda_backend,
                                                                                       std::move(temb_tensors),
                                                                                       resident_temporal_static.ptrs(),
                                                                                       true);
                        } else {
                            for (const char* name : {"time_text_embed.timestep_embedder.linear_1.weight",
                                                     "time_text_embed.timestep_embedder.linear_1.bias",
                                                     "time_text_embed.timestep_embedder.linear_2.weight",
                                                     "time_text_embed.timestep_embedder.linear_2.bias"}) {
                                if (use_transformer_context) {
                                    const TensorInput& t = top_borrowed.at(name);
                                    temb_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                } else {
                                    const Tensor& t = need(top_tensors, name);
                                    temb_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                }
                            }
                            temb_runner_ptr = std::make_unique<LensTimestepCudaRunner>(cuda_backend, std::move(temb_tensors));
                        }
                        const auto temb_setup_end = std::chrono::steady_clock::now();
                        step_runner_setup_seconds += std::chrono::duration<double>(temb_setup_end - temb_setup_start).count();
                        if (use_static_residency) {
                            temb_resource = temb_runner_ptr->compute_temb_resource_alias(8);
                            if (temb_resource == nullptr || temb_resource->empty()) {
                                throw std::runtime_error("Lens native CUDA timestep embedding alias failed at step " + std::to_string(step_index));
                            }
                        } else {
                            auto temb_opt = temb_runner_ptr->compute_temb(8);
                            if (!temb_opt.has_value()) {
                                throw std::runtime_error("Lens native CUDA timestep embedding failed at step " + std::to_string(step_index));
                            }
                            temb = from_sd_hb_as_bh(*temb_opt);
                        }
                        add_runner_timing(step_runner_timing_totals, temb_runner_ptr->get_last_timing_profile());

                        std::unique_ptr<LensBlockCudaRunner> dynamic_prev_runner;
                        std::unique_ptr<GgmlBackendTensorResource> dynamic_prev_combined;
                        if (use_static_residency && top_output_resource != nullptr && !top_output_resource->empty()) {
                            dynamic_prev_combined = std::move(top_output_resource);
                        }
                        for (int block_index = 0; block_index < 48; ++block_index) {
                            const auto block_fetch_start = std::chrono::steady_clock::now();
                            std::unordered_map<std::string, Tensor> block_weights;
                            std::unordered_map<std::string, TensorInput> block_borrowed;
                            const bool use_resident_block = static_cast<size_t>(block_index) < resident_blocks.size();
                            if (use_resident_block) {
                                // Weight inputs for this block are already resident on the CUDA backend.
                            } else if (use_transformer_context) {
                                block_borrowed = borrow_cached_lens_block(lens_ctx, transformer_handle, block_index);
                            } else {
                                block_weights = load_real_lens_block(real_block_transformer, block_index, max_real_block_bytes);
                            }
                            const auto block_fetch_end = std::chrono::steady_clock::now();
                            const double block_fetch_elapsed = std::chrono::duration<double>(block_fetch_end - block_fetch_start).count();
                            step_block_fetch_seconds += block_fetch_elapsed;
                            const auto block_setup_start = std::chrono::steady_clock::now();
                            const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                            std::unordered_map<std::string, TensorInput> runner_tensors;
                            const bool use_dynamic_combined_input =
                                (dynamic_residency == "gpu-streams" ||
                                 (dynamic_residency == "two-block-proof" && block_index == 1)) &&
                                dynamic_prev_combined != nullptr &&
                                !dynamic_prev_combined->empty();
                            if (!use_dynamic_combined_input) {
                                runner_tensors["input.hidden"] = to_input_bsh_as_csb_view(hidden_state);
                                runner_tensors["input.encoder"] = to_input_bsh_as_csb_view(encoder_state);
                            }
                            if (!use_static_residency) {
                                runner_tensors["input.temb"] = to_input_bh_as_hb_view(temb);
                                runner_tensors["input.img_pe"] = TensorInput{img_pe_sd.shape(), img_pe_sd.data()};
                                runner_tensors["input.txt_pe"] = TensorInput{txt_pe_sd.shape(), txt_pe_sd.data()};
                            }
                            uint64_t block_bytes = 0;
                            std::unordered_map<std::string, const GgmlBackendTensorResource*> resident_runner_tensors;
                            if (use_resident_block) {
                                const LensResidentBlockWeights& resident = resident_blocks[static_cast<size_t>(block_index)];
                                resident_runner_tensors.reserve(resident.tensors.size() + 3);
                                for (const auto& item : resident.tensors) {
                                    block_bytes += item.second != nullptr && item.second->tensor != nullptr
                                                       ? static_cast<uint64_t>(ggml_nbytes(item.second->tensor))
                                                       : 0;
                                    resident_runner_tensors.emplace(item.first, item.second.get());
                                }
                            } else if (use_transformer_context) {
                                for (const auto& item : block_borrowed) {
                                    if (item.first.rfind(prefix, 0) != 0) {
                                        continue;
                                    }
                                    const std::string name = item.first.substr(prefix.size());
                                    const TensorInput& t = item.second;
                                    block_bytes += elem_count(t.shape) * sizeof(float);
                                    runner_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                }
                            } else {
                                for (const auto& item : block_weights) {
                                    if (item.first.rfind(prefix, 0) != 0) {
                                        continue;
                                    }
                                    const std::string name = item.first.substr(prefix.size());
                                    const Tensor& t = item.second;
                                    block_bytes += static_cast<uint64_t>(t.data.size() * sizeof(float));
                                    runner_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                }
                            }
                            if (use_dynamic_combined_input) {
                                resident_runner_tensors.emplace("input.combined", dynamic_prev_combined.get());
                            }
                            if (use_static_residency) {
                                if (temb_resource == nullptr || temb_resource->empty()) {
                                    throw std::runtime_error("Lens native CUDA missing resident temb at step " + std::to_string(step_index));
                                }
                                resident_runner_tensors.emplace("input.temb", temb_resource.get());
                                for (const auto& item : resident_rope_static.tensors) {
                                    resident_runner_tensors.emplace(item.first, item.second.get());
                                }
                            }
                            if (!use_resident_block) {
                                total_streamed_bytes += block_bytes;
                                total_streamed_tensors += use_transformer_context ? block_borrowed.size() : block_weights.size();
                            }
                            std::unique_ptr<LensBlockCudaRunner> runner;
                            LensBlockCudaRunner* runner_ptr = nullptr;
                            const bool reuse_this_block_runner = reuse_block0_runner && block_index == 0;
                            if (reuse_this_block_runner) {
                                reusable_block0_runner->set_dynamic_inputs(std::move(runner_tensors));
                                runner_ptr = reusable_block0_runner.get();
                            } else if (use_resident_block) {
                                runner = std::make_unique<LensBlockCudaRunner>(cuda_backend,
                                                                               std::move(runner_tensors),
                                                                               std::move(resident_runner_tensors),
                                                                               image_seq_len,
                                                                               text_seq_len,
                                                                               lens_attention_mode,
                                                                               alias_resident_weights);
                                runner_ptr = runner.get();
                            } else {
                                runner = std::make_unique<LensBlockCudaRunner>(cuda_backend,
                                                                               std::move(runner_tensors),
                                                                               image_seq_len,
                                                                               text_seq_len,
                                                                               lens_attention_mode);
                                runner_ptr = runner.get();
                            }
                            const auto block_setup_end = std::chrono::steady_clock::now();
                            const double block_setup_elapsed = std::chrono::duration<double>(block_setup_end - block_setup_start).count();
                            step_runner_setup_seconds += block_setup_elapsed;
                            const bool emit_dynamic_combined_output =
                                (dynamic_residency == "gpu-streams" ||
                                 (dynamic_residency == "two-block-proof" && block_index == 0)) &&
                                use_resident_block;
                            std::optional<sd::Tensor<float>> output_opt;
                            std::unique_ptr<GgmlBackendTensorResource> output_resource;
                            if (emit_dynamic_combined_output) {
                                output_resource = runner_ptr->compute_block_resource_alias(8);
                                if (output_resource == nullptr || output_resource->empty()) {
                                    throw std::runtime_error("Lens native CUDA dynamic block output alias failed at step " +
                                                             std::to_string(step_index) + " block " + std::to_string(block_index));
                                }
                            } else {
                                output_opt = runner_ptr->compute_block(8, reuse_this_block_runner);
                            }
                            if (!output_opt.has_value()) {
                                if (emit_dynamic_combined_output) {
                                    // block0 output intentionally remains GPU-resident and feeds block1.
                                } else {
                                throw std::runtime_error("Lens native CUDA block compute failed at step " + std::to_string(step_index) +
                                                         " block " + std::to_string(block_index));
                                }
                            }
                            const GGMLRunnerTimingProfile& block_timing = runner_ptr->get_last_timing_profile();
                            if (block_index == 0) {
                                block0_step_timings.push_back(BlockStepTiming{
                                    block_timing.alloc_compute_buffer_seconds,
                                    block_timing.graph_build_seconds,
                                    block_timing.graph_alloc_seconds,
                                    block_timing.input_copy_seconds,
                                    block_timing.input_copy_submit_seconds,
                                    block_timing.input_sync_seconds,
                                    block_timing.backend_compute_seconds,
                                    block_timing.backend_compute_submit_seconds,
                                    block_timing.backend_sync_seconds,
                                    block_timing.output_copy_seconds,
                                    block_timing.cleanup_seconds,
                                    block_timing.input_copy_bytes,
                                });
                            }
                            add_runner_timing(step_runner_timing_totals, block_timing);
                            BlockTimingTotals& block_totals = block_timing_totals[static_cast<size_t>(block_index)];
                            block_totals.fetch_seconds += block_fetch_elapsed;
                            block_totals.setup_seconds += block_setup_elapsed;
                            block_totals.upload_seconds += block_timing.input_copy_seconds;
                            block_totals.upload_submit_seconds += block_timing.input_copy_submit_seconds;
                            block_totals.upload_sync_seconds += block_timing.input_sync_seconds;
                            block_totals.compute_seconds += block_timing.backend_compute_seconds;
                            block_totals.compute_submit_seconds += block_timing.backend_compute_submit_seconds;
                            block_totals.sync_seconds += block_timing.backend_sync_seconds;
                            block_totals.input_copy_bytes += block_timing.input_copy_bytes;
                            block_totals.calls += 1;
                            if (emit_dynamic_combined_output) {
                                dynamic_prev_combined = std::move(output_resource);
                                dynamic_prev_runner = std::move(runner);
                                if (dynamic_residency == "gpu-streams") {
                                    continue;
                                }
                                GGML_ASSERT(block_index == 0);
                                continue;
                            }
                            Tensor combined = from_sd_csb_as_bsh(*output_opt);
                            hidden_state = slice_seq3(combined, 0, image_seq_len);
                            encoder_state = slice_seq3(combined, image_seq_len, text_seq_len);
                            if (tensor_has_nonfinite(hidden_state) || tensor_has_nonfinite(encoder_state)) {
                                throw std::runtime_error("Lens native CUDA produced nonfinite hidden/encoder at step " +
                                                         std::to_string(step_index) + " block " + std::to_string(block_index));
                            }
                        }

                        if (dynamic_residency == "gpu-streams") {
                            std::cout << "Lens native CUDA step hidden stats: block=" << step_index
                                      << " skipped=gpu_streams_dynamic_residency\n";
                            std::cout << "Lens native CUDA step encoder stats: block=" << step_index
                                      << " skipped=gpu_streams_dynamic_residency\n";
                        } else {
                            print_tensor_range_stats("Lens native CUDA step hidden stats:", step_index, hidden_state);
                            print_tensor_range_stats("Lens native CUDA step encoder stats:", step_index, encoder_state);
                        }
                        const auto final_setup_start = std::chrono::steady_clock::now();
                        std::unordered_map<std::string, TensorInput> final_tensors;
                        std::unordered_map<std::string, const GgmlBackendTensorResource*> final_resident_tensors;
                        const bool final_uses_dynamic_combined =
                            dynamic_residency == "gpu-streams" &&
                            dynamic_prev_combined != nullptr &&
                            !dynamic_prev_combined->empty();
                        if (final_uses_dynamic_combined) {
                            final_resident_tensors.emplace("input.combined", dynamic_prev_combined.get());
                        } else {
                            final_tensors["input.hidden"] = to_input_bsh_as_csb_view(hidden_state);
                        }
                        if (use_static_residency) {
                            if (temb_resource == nullptr || temb_resource->empty()) {
                                throw std::runtime_error("Lens native CUDA missing resident final temb at step " + std::to_string(step_index));
                            }
                            final_resident_tensors.emplace("input.temb", temb_resource.get());
                            for (const auto& item : resident_final_static.tensors) {
                                final_resident_tensors.emplace(item.first, item.second.get());
                            }
                        } else {
                            final_tensors["input.temb"] = to_input_bh_as_hb_view(temb);
                            for (const char* name : {"norm_out.linear.weight", "norm_out.linear.bias", "proj_out.weight", "proj_out.bias"}) {
                                if (use_transformer_context) {
                                    const TensorInput& t = top_borrowed.at(name);
                                    final_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                } else {
                                    const Tensor& t = need(top_tensors, name);
                                    final_tensors[name] = t.shape.size() == 2 ? to_input_2d(t) : to_input_1d(t);
                                }
                            }
                        }
                        std::unique_ptr<LensFinalCudaRunner> final_runner_ptr;
                        if (final_uses_dynamic_combined || use_static_residency) {
                            final_runner_ptr = std::make_unique<LensFinalCudaRunner>(cuda_backend,
                                                                                     std::move(final_tensors),
                                                                                     std::move(final_resident_tensors),
                                                                                     image_seq_len,
                                                                                     true);
                        } else {
                            final_runner_ptr = std::make_unique<LensFinalCudaRunner>(cuda_backend, std::move(final_tensors));
                        }
                        const auto final_setup_end = std::chrono::steady_clock::now();
                        step_runner_setup_seconds += std::chrono::duration<double>(final_setup_end - final_setup_start).count();
                        auto prediction_opt = final_runner_ptr->compute_final(8);
                        if (!prediction_opt.has_value()) {
                            throw std::runtime_error("Lens native CUDA final projection failed at step " + std::to_string(step_index));
                        }
                        add_runner_timing(step_runner_timing_totals, final_runner_ptr->get_last_timing_profile());
                        Tensor prediction = from_sd_csb_as_bsh(*prediction_opt);
                        if (prediction.shape != current_tokens.shape || tensor_has_nonfinite(prediction)) {
                            throw std::runtime_error("Lens native CUDA prediction shape/nonfinite failure at step " + std::to_string(step_index));
                        }
                        print_tensor_range_stats("Lens native CUDA step prediction stats:", step_index, prediction);
                        std::copy(prediction.data.begin(),
                                  prediction.data.end(),
                                  external_model_outputs.begin() + current_tokens.data.size() * static_cast<size_t>(step_index));
                        const auto step_transformer_end = std::chrono::steady_clock::now();
                        const double step_transformer_seconds =
                            std::chrono::duration<double>(step_transformer_end - step_transformer_start).count();
                        transformer_seconds += step_transformer_seconds;
                        per_step_transformer_seconds.push_back(step_transformer_seconds);
                        block_fetch_seconds += step_block_fetch_seconds;
                        runner_setup_seconds += step_runner_setup_seconds;
                        add_runner_timing_totals(runner_timing_totals, step_runner_timing_totals);
                        per_step_block_fetch_seconds.push_back(step_block_fetch_seconds);
                        per_step_runner_setup_seconds.push_back(step_runner_setup_seconds);
                        per_step_runner_input_copy_seconds.push_back(step_runner_timing_totals.input_copy_seconds);
                        per_step_runner_input_copy_submit_seconds.push_back(step_runner_timing_totals.input_copy_submit_seconds);
                        per_step_runner_input_sync_seconds.push_back(step_runner_timing_totals.input_sync_seconds);
                        per_step_runner_compute_seconds.push_back(step_runner_timing_totals.backend_compute_seconds);
                        per_step_runner_compute_submit_seconds.push_back(step_runner_timing_totals.backend_compute_submit_seconds);
                        per_step_runner_sync_seconds.push_back(step_runner_timing_totals.backend_sync_seconds);
                        per_step_runner_graph_seconds.push_back(step_runner_timing_totals.alloc_compute_buffer_seconds +
                                                                step_runner_timing_totals.graph_build_seconds +
                                                                step_runner_timing_totals.graph_alloc_seconds);

                        const auto flow_start = std::chrono::steady_clock::now();
                        float sigma0 = 0.0f;
                        float sigma1 = 0.0f;
                        float dt = 0.0f;
                        current_tokens = apply_lens_flow_step_at(current_tokens, prediction, sigmas, step_index, sigma0, sigma1, dt);
                        if (tensor_has_nonfinite(current_tokens)) {
                            throw std::runtime_error("Lens native CUDA flow update produced nonfinite packed tokens at step " + std::to_string(step_index));
                        }
                        const auto flow_end = std::chrono::steady_clock::now();
                        flow_seconds += std::chrono::duration<double>(flow_end - flow_start).count();
                        print_tensor_range_stats("Lens native CUDA step packed stats:", step_index, current_tokens);
                        if (progress_callback != nullptr) {
                            const float step_elapsed_seconds = static_cast<float>(
                                std::chrono::duration<double>(flow_end - step_transformer_start).count());
                            progress_callback(
                                step_index + 1,
                                tiny_flow_steps,
                                step_elapsed_seconds,
                                progress_callback_data);
                        }
                        std::cout << "Lens native CUDA generation step passed:"
                                  << " step=" << step_index
                                  << " transformer_seconds=" << step_transformer_seconds
                                  << " block_fetch_seconds=" << step_block_fetch_seconds
                                  << " runner_setup_seconds=" << step_runner_setup_seconds
                                  << " runner_input_copy_seconds=" << step_runner_timing_totals.input_copy_seconds
                                  << " runner_input_copy_submit_seconds=" << step_runner_timing_totals.input_copy_submit_seconds
                                  << " runner_input_sync_seconds=" << step_runner_timing_totals.input_sync_seconds
                                  << " runner_compute_seconds=" << step_runner_timing_totals.backend_compute_seconds
                                  << " runner_compute_submit_seconds=" << step_runner_timing_totals.backend_compute_submit_seconds
                                  << " runner_sync_seconds=" << step_runner_timing_totals.backend_sync_seconds
                                  << " runner_graph_seconds="
                                  << (step_runner_timing_totals.alloc_compute_buffer_seconds +
                                      step_runner_timing_totals.graph_build_seconds +
                                      step_runner_timing_totals.graph_alloc_seconds)
                                  << " sigma0=" << sigma0
                                  << " sigma1=" << sigma1
                                  << " dt=" << dt
                                  << " prediction_shape=1x" << image_seq_len << "x128\n";
                    }

                    float api_mean_diff = 0.0f;
                    float api_max_diff = 0.0f;
                    if (!lens_cond_path.empty()) {
                        Tensor api_tokens;
                        sd_lens_external_flow_loop_desc_t flow_desc;
                        const auto flow_api_start = std::chrono::steady_clock::now();
                        if (!run_external_flow_api_check(lens_cond_path,
                                                         real_block_transformer,
                                                         initial_tokens,
                                                         external_model_outputs,
                                                         tiny_flow_steps,
                                                         static_cast<int>(image_seq_len),
                                                         128,
                                                         static_cast<uint32_t>(dim(f0, 2)),
                                                         api_tokens,
                                                         flow_desc)) {
                            throw std::runtime_error("sd_lens_run_external_flow_loop_f32 check failed for native CUDA generation");
                        }
                        const auto flow_api_end = std::chrono::steady_clock::now();
                        flow_seconds += std::chrono::duration<double>(flow_api_end - flow_api_start).count();
                        api_max_diff = max_abs_diff(api_tokens, current_tokens, &api_mean_diff);
                        if (api_max_diff > 1.0e-6f) {
                            throw std::runtime_error("native CUDA manual flow and sd_lens_run_external_flow_loop_f32 diverged");
                        }
                        current_tokens = std::move(api_tokens);
                    } else {
                        std::cout << "Lens native CUDA external flow API check skipped: in_memory_conditioning_without_safetensors\n";
                    }
                    if (!write_f32_npy(packed_tokens_npy, current_tokens)) {
                        throw std::runtime_error("failed to write native CUDA final packed tokens NPY");
                    }

                    const auto unpack_start = std::chrono::steady_clock::now();
                    Tensor latent;
                    if (output_packed_vae_latent) {
                        latent = lens_packed_tokens_to_vae_whcn(current_tokens, packed_h, packed_w);
                    } else {
                        latent = Tensor{{1, 32, packed_h * 2, packed_w * 2},
                                        std::vector<float>(static_cast<size_t>(32 * packed_h * 2 * packed_w * 2), 0.0f)};
                        sd_lens_vae_latent_desc_t unpack_desc;
                        sd_lens_vae_latent_desc_init(&unpack_desc);
                        if (!sd_lens_unpack_vae_latent_f32(current_tokens.data.data(),
                                                           static_cast<uint64_t>(current_tokens.data.size()),
                                                           1,
                                                           32,
                                                           packed_h * 2,
                                                           packed_w * 2,
                                                           latent.data.data(),
                                                           static_cast<uint64_t>(latent.data.size()),
                                                           &unpack_desc)) {
                            throw std::runtime_error("sd_lens_unpack_vae_latent_f32 failed for native CUDA generation");
                        }
                    }
                    const auto unpack_end = std::chrono::steady_clock::now();
                    unpack_seconds += std::chrono::duration<double>(unpack_end - unpack_start).count();
                    if (g_lens_native_cuda_output_latent != nullptr) {
                        *g_lens_native_cuda_output_latent = latent;
                    }
                    if (tensor_has_nonfinite(latent) || !write_f32_npy(tiny_denoise_npy, latent)) {
                        throw std::runtime_error("native CUDA generation produced nonfinite latent or failed to write latent NPY");
                    }
                    print_transformer_vram_snapshot("after_denoise");
                    print_tensor_range_stats("Lens native CUDA final latent stats:", -1, latent);
                    const auto total_end = std::chrono::steady_clock::now();
                    const double total_generation_seconds =
                        std::chrono::duration<double>(total_end - generation_total_start).count();
                    if (out_profile != nullptr) {
                        out_profile->context_load_seconds = context_load_seconds;
                        out_profile->resident_upload_seconds =
                            generation_index == 0 ? resident_upload_seconds : 0.0;
                        out_profile->loop_seconds = transformer_seconds;
                        out_profile->total_generation_seconds = total_generation_seconds;
                        out_profile->runner_setup_seconds = runner_setup_seconds;
                        out_profile->runner_alloc_compute_buffer_seconds =
                            runner_timing_totals.alloc_compute_buffer_seconds;
                        out_profile->runner_graph_build_seconds = runner_timing_totals.graph_build_seconds;
                        out_profile->runner_graph_alloc_seconds = runner_timing_totals.graph_alloc_seconds;
                        out_profile->runner_input_copy_seconds = runner_timing_totals.input_copy_seconds;
                        out_profile->runner_compute_seconds = runner_timing_totals.backend_compute_seconds;
                        out_profile->runner_sync_seconds = runner_timing_totals.backend_sync_seconds;
                        out_profile->runner_output_copy_seconds = runner_timing_totals.output_copy_seconds;
                        out_profile->runner_cleanup_seconds = runner_timing_totals.cleanup_seconds;
                        out_profile->scheduler_flow_seconds = flow_seconds;
                        out_profile->unpack_seconds = unpack_seconds;
                        out_profile->runner_input_copy_bytes = runner_timing_totals.input_copy_bytes;
                        out_profile->runner_output_copy_bytes = runner_timing_totals.output_copy_bytes;
                        out_profile->streamed_bytes = total_streamed_bytes;
                        out_profile->disk_read_bytes =
                            use_transformer_context ? transformer_desc.host_cached_bytes : total_streamed_bytes;
                        out_profile->resident_weight_bytes = resident_weight_bytes;
                        out_profile->resident_static_bytes = resident_static_bytes;
                        out_profile->runner_count = runner_timing_totals.runner_count;
                    }
                    std::cout << "Lens native CUDA generation passed:"
                              << " generation=" << generation_index
                              << " repeat_generations=" << repeat_generations
                              << " steps=" << tiny_flow_steps
                              << " context_load_seconds=" << context_load_seconds
                              << " output_packed=" << packed_tokens_npy
                              << " output_latent=" << tiny_denoise_npy
                              << " latent_shape=" << tensor_shape_string(latent)
                              << " packed_vae_latent_output=" << (output_packed_vae_latent ? "true" : "false")
                              << " transformer_seconds=" << transformer_seconds
                              << " block_fetch_seconds=" << block_fetch_seconds
                              << " runner_setup_seconds=" << runner_setup_seconds
                              << " runner_offload_params_seconds=" << runner_timing_totals.offload_params_seconds
                              << " runner_alloc_compute_buffer_seconds=" << runner_timing_totals.alloc_compute_buffer_seconds
                              << " runner_graph_build_seconds=" << runner_timing_totals.graph_build_seconds
                              << " runner_graph_alloc_seconds=" << runner_timing_totals.graph_alloc_seconds
                              << " runner_input_copy_seconds=" << runner_timing_totals.input_copy_seconds
                              << " runner_input_copy_submit_seconds=" << runner_timing_totals.input_copy_submit_seconds
                              << " runner_input_sync_seconds=" << runner_timing_totals.input_sync_seconds
                              << " runner_compute_seconds=" << runner_timing_totals.backend_compute_seconds
                              << " runner_compute_submit_seconds=" << runner_timing_totals.backend_compute_submit_seconds
                              << " runner_sync_seconds=" << runner_timing_totals.backend_sync_seconds
                              << " runner_output_copy_seconds=" << runner_timing_totals.output_copy_seconds
                              << " runner_cleanup_seconds=" << runner_timing_totals.cleanup_seconds
                              << " runner_input_copy_bytes=" << runner_timing_totals.input_copy_bytes
                              << " runner_output_copy_bytes=" << runner_timing_totals.output_copy_bytes
                              << " runner_count=" << runner_timing_totals.runner_count;
                    print_copy_breakdown("runner_copy_breakdown", runner_timing_totals);
                    print_seconds_vector("per_step_transformer_seconds", per_step_transformer_seconds);
                    print_seconds_vector("per_step_block_fetch_seconds", per_step_block_fetch_seconds);
                    print_seconds_vector("per_step_runner_setup_seconds", per_step_runner_setup_seconds);
                    print_seconds_vector("per_step_runner_input_copy_seconds", per_step_runner_input_copy_seconds);
                    print_seconds_vector("per_step_runner_input_copy_submit_seconds", per_step_runner_input_copy_submit_seconds);
                    print_seconds_vector("per_step_runner_input_sync_seconds", per_step_runner_input_sync_seconds);
                    print_seconds_vector("per_step_runner_compute_seconds", per_step_runner_compute_seconds);
                    print_seconds_vector("per_step_runner_compute_submit_seconds", per_step_runner_compute_submit_seconds);
                    print_seconds_vector("per_step_runner_sync_seconds", per_step_runner_sync_seconds);
                    print_seconds_vector("per_step_runner_graph_seconds", per_step_runner_graph_seconds);
                    print_top_block_timings("top_block_fetch_seconds", block_timing_totals, &BlockTimingTotals::fetch_seconds);
                    print_top_block_timings("top_block_setup_seconds", block_timing_totals, &BlockTimingTotals::setup_seconds);
                    print_top_block_timings("top_block_upload_seconds", block_timing_totals, &BlockTimingTotals::upload_seconds);
                    print_top_block_timings("top_block_upload_submit_seconds", block_timing_totals, &BlockTimingTotals::upload_submit_seconds);
                    print_top_block_timings("top_block_upload_sync_seconds", block_timing_totals, &BlockTimingTotals::upload_sync_seconds);
                    print_top_block_timings("top_block_compute_seconds", block_timing_totals, &BlockTimingTotals::compute_seconds);
                    print_top_block_timings("top_block_compute_submit_seconds", block_timing_totals, &BlockTimingTotals::compute_submit_seconds);
                    print_top_block_timings("top_block_sync_seconds", block_timing_totals, &BlockTimingTotals::sync_seconds);
                    const BlockTimingTotals& block0_totals = block_timing_totals[0];
                    auto block0_seconds = [&](double BlockStepTiming::*field) {
                        std::vector<double> values;
                        values.reserve(block0_step_timings.size());
                        for (const BlockStepTiming& timing : block0_step_timings) {
                            values.push_back(timing.*field);
                        }
                        return values;
                    };
                    auto block0_bytes = [&]() {
                        std::cout << " block0_step_input_copy_bytes=";
                        for (size_t i = 0; i < block0_step_timings.size(); ++i) {
                            if (i != 0) {
                                std::cout << ",";
                            }
                            std::cout << block0_step_timings[i].input_copy_bytes;
                        }
                    };
                    print_seconds_vector("block0_step_alloc_compute_buffer_seconds",
                                         block0_seconds(&BlockStepTiming::alloc_compute_buffer_seconds));
                    print_seconds_vector("block0_step_graph_build_seconds",
                                         block0_seconds(&BlockStepTiming::graph_build_seconds));
                    print_seconds_vector("block0_step_graph_alloc_seconds",
                                         block0_seconds(&BlockStepTiming::graph_alloc_seconds));
                    print_seconds_vector("block0_step_input_copy_seconds",
                                         block0_seconds(&BlockStepTiming::input_copy_seconds));
                    print_seconds_vector("block0_step_input_copy_submit_seconds",
                                         block0_seconds(&BlockStepTiming::input_copy_submit_seconds));
                    print_seconds_vector("block0_step_input_sync_seconds",
                                         block0_seconds(&BlockStepTiming::input_sync_seconds));
                    print_seconds_vector("block0_step_compute_seconds",
                                         block0_seconds(&BlockStepTiming::compute_seconds));
                    print_seconds_vector("block0_step_compute_submit_seconds",
                                         block0_seconds(&BlockStepTiming::compute_submit_seconds));
                    print_seconds_vector("block0_step_sync_seconds",
                                         block0_seconds(&BlockStepTiming::sync_seconds));
                    print_seconds_vector("block0_step_output_copy_seconds",
                                         block0_seconds(&BlockStepTiming::output_copy_seconds));
                    print_seconds_vector("block0_step_cleanup_seconds",
                                         block0_seconds(&BlockStepTiming::cleanup_seconds));
                    block0_bytes();
                    std::cout
                              << " scheduler_flow_seconds=" << flow_seconds
                              << " unpack_seconds=" << unpack_seconds
                              << " total_seconds=" << total_generation_seconds
                              << " residency_mode=" << transformer_residency
                              << " runner_reuse=" << transformer_runner_reuse
                              << " dynamic_residency=" << dynamic_residency
                              << " alias_resident_weights=" << (alias_resident_weights ? "true" : "false")
                              << " resident_weight_dtype=" << (resident_weights_bf16 ? "bf16" : "f32")
                              << " resident_window_blocks=" << resident_blocks.size()
                              << " resident_upload_seconds=" << resident_upload_seconds
                              << " resident_upload_paid_this_generation="
                              << (generation_index == 0 ? resident_upload_seconds : 0.0)
                              << " resident_upload_reused="
                              << (generation_index > 0 && !resident_blocks.empty() ? "true" : "false")
                              << " resident_weight_bytes=" << resident_weight_bytes
                              << " resident_weight_tensors=" << resident_weight_tensors
                              << " resident_static_bytes=" << resident_static_bytes
                              << " resident_static_tensors=" << resident_static_tensors
                              << " block0_input_copy_bytes=" << block0_totals.input_copy_bytes
                              << " block0_input_copy_seconds=" << block0_totals.upload_seconds
                              << " block0_input_copy_submit_seconds=" << block0_totals.upload_submit_seconds
                              << " block0_input_sync_seconds=" << block0_totals.upload_sync_seconds
                              << " streamed_tensors=" << total_streamed_tensors
                              << " streamed_bytes=" << total_streamed_bytes
                              << " disk_read_bytes=" << (use_transformer_context ? transformer_desc.host_cached_bytes : total_streamed_bytes)
                              << " host_cached=" << (use_transformer_context ? "true" : "false")
                              << " gpu_resident=" << (resident_blocks.empty() ? "false" : "partial")
                              << " external_flow_api_max_diff=" << api_max_diff
                              << " external_flow_api_mean_diff=" << api_mean_diff
                              << " sd_cpp_owned=transformer_scheduler_flow_unpack"
                              << " python_owned=conditioning_only"
                              << " scalar_cpu_transformer=false\n";
                    }
                    if (can_keep_warm_transformer && !warm_transformer_reused && !local_resident_blocks.empty()) {
                        g_lens_warm_transformer_cache.clear();
                        g_lens_warm_transformer_cache.valid = true;
                        g_lens_warm_transformer_cache.key = warm_transformer_key;
                        g_lens_warm_transformer_cache.lens_ctx = lens_ctx;
                        g_lens_warm_transformer_cache.transformer_handle = transformer_handle;
                        g_lens_warm_transformer_cache.transformer_desc = transformer_desc;
                        g_lens_warm_transformer_cache.cuda_backend = cuda_backend;
                        g_lens_warm_transformer_cache.resident_blocks = std::move(local_resident_blocks);
                        g_lens_warm_transformer_cache.resident_weight_bytes = resident_weight_bytes;
                        g_lens_warm_transformer_cache.resident_weight_tensors = resident_weight_tensors;
                        lens_ctx = nullptr;
                        transformer_handle = 0;
                        cuda_backend = nullptr;
                        std::cout << "Lens transformer warm cache stored:"
                                  << " key=" << warm_transformer_key
                                  << " resident_weight_bytes=" << g_lens_warm_transformer_cache.resident_weight_bytes
                                  << " resident_weight_tensors=" << g_lens_warm_transformer_cache.resident_weight_tensors
                                  << "\n";
                    }
                    if (!warm_transformer_reused && cuda_backend != nullptr) {
                        ggml_backend_free(cuda_backend);
                    }
                    if (!warm_transformer_reused && lens_ctx != nullptr) {
                        if (transformer_handle != 0) {
                            sd_lens_transformer_release(lens_ctx, transformer_handle);
                        }
                        free_sd_ctx(lens_ctx);
                    }
                    print_transformer_vram_snapshot("after_transformer_free");
                    return 0;
                } catch (...) {
                    if (!warm_transformer_reused && cuda_backend != nullptr) {
                        ggml_backend_free(cuda_backend);
                    }
                    if (!warm_transformer_reused && lens_ctx != nullptr) {
                        if (transformer_handle != 0) {
                            sd_lens_transformer_release(lens_ctx, transformer_handle);
                        }
                        free_sd_ctx(lens_ctx);
                    }
                    throw;
                }
#endif
            }
            if (!real_full_cuda_oracle_fixture.empty()) {
#ifndef SD_USE_CUDA
                std::cerr << "Lens full CUDA smoke requires SD_USE_CUDA build\n";
                return 1;
#else
                auto full_oracle = load_fixture(real_full_cuda_oracle_fixture);
                for (const char* name : {"input.hidden",
                                         "input.encoder",
                                         "input.temb",
                                         "input.img_freqs",
                                         "input.txt_freqs",
                                         "expected.prediction"}) {
                    if (!has_tensor(full_oracle, name)) {
                        throw std::runtime_error(std::string("--real-full-cuda-oracle-fixture missing tensor: ") + name);
                    }
                }
                Tensor hidden_state = need(full_oracle, "input.hidden");
                Tensor encoder_state = need(full_oracle, "input.encoder");
                const Tensor& input_temb = need(full_oracle, "input.temb");
                const Tensor& input_img_freqs = need(full_oracle, "input.img_freqs");
                const Tensor& input_txt_freqs = need(full_oracle, "input.txt_freqs");
                if (dim(hidden_state, 0) != 1 || dim(hidden_state, 1) != 256 || dim(hidden_state, 2) != 1536 ||
                    dim(encoder_state, 0) != 1 || dim(encoder_state, 2) != 1536 ||
                    dim(input_temb, 0) != 1 || dim(input_temb, 1) != 1536 ||
                    dim(input_img_freqs, 0) != 256 || dim(input_img_freqs, 1) != 32 || dim(input_img_freqs, 2) != 2 ||
                    dim(input_txt_freqs, 0) != dim(encoder_state, 1) || dim(input_txt_freqs, 1) != 32 || dim(input_txt_freqs, 2) != 2) {
                    std::cerr << "full CUDA oracle input shape mismatch before native forward\n";
                    return 1;
                }
                ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
                if (cuda_backend == nullptr) {
                    std::cerr << "ggml_backend_cuda_init(0) failed\n";
                    return 1;
                }
                std::cout << "Lens CUDA full transformer attention_mode="
                          << lens_attention_mode_name(lens_attention_mode) << "\n";
                const auto start = std::chrono::steady_clock::now();
                uint64_t total_loaded_bytes = 0;
                size_t total_loaded_tensors = 0;
                int first_divergent_block = -1;
                float worst_rel = 0.0f;
                float worst_abs = 0.0f;
                try {
                    for (int block_index = 0; block_index < 48; ++block_index) {
                        auto block_weights = load_real_lens_block(real_block_transformer, block_index, max_real_block_bytes);
                        const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                        std::unordered_map<std::string, sd::Tensor<float>> runner_tensors;
                        runner_tensors["input.hidden"] = to_sd_bsh_as_csb(hidden_state);
                        runner_tensors["input.encoder"] = to_sd_bsh_as_csb(encoder_state);
                        runner_tensors["input.temb"] = to_sd_bh_as_hb(input_temb);
                        runner_tensors["input.img_pe"] = lens_freqs_to_rope_pe(input_img_freqs);
                        runner_tensors["input.txt_pe"] = lens_freqs_to_rope_pe(input_txt_freqs);
                        uint64_t block_bytes = 0;
                        for (const auto& item : block_weights) {
                            if (item.first.rfind(prefix, 0) != 0) {
                                continue;
                            }
                            const std::string name = item.first.substr(prefix.size());
                            const Tensor& t = item.second;
                            block_bytes += static_cast<uint64_t>(t.data.size() * sizeof(float));
                            if (t.shape.size() == 2) {
                                runner_tensors[name] = to_sd_2d(t);
                            } else if (t.shape.size() == 1) {
                                runner_tensors[name] = to_sd_1d(t);
                            } else {
                                throw std::runtime_error("unexpected Lens block weight rank for CUDA runner: " + item.first);
                            }
                        }
                        total_loaded_bytes += block_bytes;
                        total_loaded_tensors += block_weights.size();
                        LensBlockCudaRunner runner(cuda_backend,
                                                   std::move(runner_tensors),
                                                   dim(hidden_state, 1),
                                                   dim(encoder_state, 1),
                                                   lens_attention_mode);
                        auto output_opt = runner.compute_block(8);
                        if (!output_opt.has_value()) {
                            throw std::runtime_error("Lens full CUDA block compute failed at block " + std::to_string(block_index));
                        }
                        Tensor combined = from_sd_csb_as_bsh(*output_opt);
                        hidden_state = slice_seq3(combined, 0, 256);
                        encoder_state = slice_seq3(combined, 256, dim(encoder_state, 1));
                        const std::string checkpoint_prefix = "checkpoint.block_" + std::to_string(block_index) + ".";
                        if (has_tensor(full_oracle, checkpoint_prefix + "encoder")) {
                            print_tensor_range_stats("Lens CUDA full checkpoint encoder stats:", block_index, encoder_state);
                            const DiffStats stats = diff_stats(encoder_state, need(full_oracle, checkpoint_prefix + "encoder"));
                            print_diff_stats("Lens CUDA full checkpoint encoder parity:", block_index, stats);
                            worst_rel = std::max(worst_rel, stats.rel_max);
                            worst_abs = std::max(worst_abs, stats.max_diff);
                            if (first_divergent_block < 0 && (has_nonfinite(stats) || stats.rel_max > tolerance)) {
                                first_divergent_block = block_index;
                            }
                        }
                        if (has_tensor(full_oracle, checkpoint_prefix + "hidden")) {
                            print_tensor_range_stats("Lens CUDA full checkpoint hidden stats:", block_index, hidden_state);
                            const DiffStats stats = diff_stats(hidden_state, need(full_oracle, checkpoint_prefix + "hidden"));
                            print_diff_stats("Lens CUDA full checkpoint hidden parity:", block_index, stats);
                            worst_rel = std::max(worst_rel, stats.rel_max);
                            worst_abs = std::max(worst_abs, stats.max_diff);
                            if (first_divergent_block < 0 && (has_nonfinite(stats) || stats.rel_max > tolerance)) {
                                first_divergent_block = block_index;
                            }
                        }
                    }
                    std::vector<std::string> top_names = {
                        "norm_out.linear.weight",
                        "norm_out.linear.bias",
                        "proj_out.weight",
                        "proj_out.bias",
                    };
                    auto top_tensors = load_real_lens_named_tensors(real_block_transformer, top_names, max_real_top_bytes);
                    std::unordered_map<std::string, sd::Tensor<float>> final_tensors;
                    final_tensors["input.hidden"] = to_sd_bsh_as_csb(hidden_state);
                    final_tensors["input.temb"] = to_sd_bh_as_hb(input_temb);
                    for (const auto& item : top_tensors) {
                        total_loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                        total_loaded_tensors += 1;
                        if (item.second.shape.size() == 2) {
                            final_tensors[item.first] = to_sd_2d(item.second);
                        } else if (item.second.shape.size() == 1) {
                            final_tensors[item.first] = to_sd_1d(item.second);
                        }
                    }
                    LensFinalCudaRunner final_runner(cuda_backend, std::move(final_tensors));
                    auto prediction_opt = final_runner.compute_final(8);
                    if (!prediction_opt.has_value()) {
                        throw std::runtime_error("Lens CUDA final projection compute failed");
                    }
                    Tensor prediction = from_sd_csb_as_bsh(*prediction_opt);
                    if (dim(prediction, 0) != 1 || dim(prediction, 1) != 256 || dim(prediction, 2) != 128) {
                        throw std::runtime_error("Lens CUDA final prediction shape mismatch");
                    }
                    print_tensor_range_stats("Lens CUDA full prediction stats:", -1, prediction);
                    const DiffStats final_stats = diff_stats(prediction, need(full_oracle, "expected.prediction"));
                    print_diff_stats("Lens CUDA full prediction parity:", -1, final_stats);
                    worst_rel = std::max(worst_rel, final_stats.rel_max);
                    worst_abs = std::max(worst_abs, final_stats.max_diff);
                    const auto end = std::chrono::steady_clock::now();
                    std::cout << "Lens CUDA full transformer smoke passed: blocks=48"
                              << " total_tensors=" << total_loaded_tensors
                              << " total_streamed_bytes=" << total_loaded_bytes
                              << " loading=host_to_gpu_per_block"
                              << " output_shape=1x256x128"
                              << " runtime_seconds=" << std::chrono::duration<double>(end - start).count()
                              << " worst_abs=" << worst_abs
                              << " worst_rel=" << worst_rel
                              << " first_divergent_block=" << first_divergent_block
                              << " attention_mode=" << lens_attention_mode_name(lens_attention_mode)
                              << " backend=ggml_cuda scalar_cpu_matmul=false python_owned=conditioning_and_oracle_only\n";
                    ggml_backend_free(cuda_backend);
                    if (tensor_has_nonfinite(hidden_state) || tensor_has_nonfinite(encoder_state) || tensor_has_nonfinite(prediction)) {
                        std::cerr << "Lens CUDA full transformer produced nonfinite hidden, encoder, or prediction values\n";
                        return 1;
                    }
                    if (has_nonfinite(final_stats)) {
                        std::cerr << "Lens CUDA full prediction produced or compared against nonfinite values\n";
                        return 1;
                    }
                    if (final_stats.rel_max > tolerance) {
                        std::cerr << "Lens CUDA full prediction parity exceeded relative tolerance " << tolerance << "\n";
                        return 1;
                    }
                    return 0;
                } catch (...) {
                    ggml_backend_free(cuda_backend);
                    throw;
                }
#endif
            }
            if (!real_block_cuda_oracle_fixture.empty()) {
#ifndef SD_USE_CUDA
                std::cerr << "Lens CUDA block smoke requires SD_USE_CUDA build\n";
                return 1;
#else
                auto block_oracle = load_fixture(real_block_cuda_oracle_fixture);
                for (const char* name : {"input.hidden",
                                         "input.encoder",
                                         "input.temb",
                                         "input.img_freqs",
                                         "input.txt_freqs",
                                         "expected.encoder",
                                         "expected.hidden"}) {
                    if (!has_tensor(block_oracle, name)) {
                        throw std::runtime_error(std::string("--real-block-cuda-oracle-fixture missing tensor: ") + name);
                    }
                }
                if (real_block_index != 0) {
                    std::cerr << "--real-block-cuda-oracle-fixture currently validates block 0 only\n";
                    return 2;
                }
                const Tensor& input_hidden = need(block_oracle, "input.hidden");
                const Tensor& input_encoder = need(block_oracle, "input.encoder");
                const Tensor& input_temb = need(block_oracle, "input.temb");
                const Tensor& input_img_freqs = need(block_oracle, "input.img_freqs");
                const Tensor& input_txt_freqs = need(block_oracle, "input.txt_freqs");
                if (dim(input_hidden, 0) != 1 || dim(input_hidden, 1) != 256 || dim(input_hidden, 2) != 1536 ||
                    dim(input_encoder, 0) != 1 || dim(input_encoder, 2) != 1536 ||
                    dim(input_temb, 0) != 1 || dim(input_temb, 1) != 1536 ||
                    dim(input_img_freqs, 0) != 256 || dim(input_img_freqs, 1) != 32 || dim(input_img_freqs, 2) != 2 ||
                    dim(input_txt_freqs, 0) != dim(input_encoder, 1) || dim(input_txt_freqs, 1) != 32 || dim(input_txt_freqs, 2) != 2) {
                    std::cerr << "block0 CUDA oracle input shape mismatch before native forward\n";
                    return 1;
                }

                auto block_weights = load_real_lens_block(real_block_transformer, 0, max_real_block_bytes);
                const std::string prefix = "transformer_blocks.0.";
                std::unordered_map<std::string, sd::Tensor<float>> runner_tensors;
                runner_tensors["input.hidden"] = to_sd_bsh_as_csb(input_hidden);
                runner_tensors["input.encoder"] = to_sd_bsh_as_csb(input_encoder);
                runner_tensors["input.temb"] = to_sd_bh_as_hb(input_temb);
                runner_tensors["input.img_pe"] = lens_freqs_to_rope_pe(input_img_freqs);
                runner_tensors["input.txt_pe"] = lens_freqs_to_rope_pe(input_txt_freqs);
                uint64_t block_bytes = 0;
                for (const auto& item : block_weights) {
                    if (item.first.rfind(prefix, 0) != 0) {
                        continue;
                    }
                    const std::string name = item.first.substr(prefix.size());
                    const Tensor& t = item.second;
                    block_bytes += static_cast<uint64_t>(t.data.size() * sizeof(float));
                    if (t.shape.size() == 2) {
                        runner_tensors[name] = to_sd_2d(t);
                    } else if (t.shape.size() == 1) {
                        runner_tensors[name] = to_sd_1d(t);
                    } else {
                        throw std::runtime_error("unexpected Lens block weight rank for CUDA runner: " + item.first);
                    }
                }
#ifdef SD_LENS_TRANSFORMER_USE_CUBLASLT
                if (modulation_bf16_cublaslt) {
                    int img_algo = -1;
                    int txt_algo = -1;
                    double img_ms = 0.0;
                    double txt_ms = 0.0;
                    add_bf16_cublaslt_modulation_inputs(runner_tensors,
                                                        input_temb,
                                                        need(block_weights, prefix + "img_mod.1.weight"),
                                                        need(block_weights, prefix + "img_mod.1.bias"),
                                                        need(block_weights, prefix + "txt_mod.1.weight"),
                                                        need(block_weights, prefix + "txt_mod.1.bias"),
                                                        "block0",
                                                        &img_algo,
                                                        &txt_algo,
                                                        &img_ms,
                                                        &txt_ms);
                    std::cout << "Lens CUDA block0 modulation_bf16_cublaslt=true"
                              << " img_algo=" << img_algo
                              << " txt_algo=" << txt_algo
                              << " img_ms=" << img_ms
                              << " txt_ms=" << txt_ms
                              << "\n";
                }
#else
                if (modulation_bf16_cublaslt) {
                    std::cerr << "--modulation-bf16-cublaslt requires cuBLASLt support\n";
                    return 2;
                }
#endif
                ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
                if (cuda_backend == nullptr) {
                    std::cerr << "ggml_backend_cuda_init(0) failed\n";
                    return 1;
                }
                std::cout << "Lens CUDA block0 attention_mode="
                          << lens_attention_mode_name(lens_attention_mode) << "\n";
                const auto start = std::chrono::steady_clock::now();
                LensBlockCudaRunner runner(cuda_backend,
                                           std::move(runner_tensors),
                                           dim(input_hidden, 1),
                                           dim(input_encoder, 1),
                                           lens_attention_mode,
                                           modulation_bf16_cublaslt);
                auto output_opt = runner.compute_block(8);
                const auto end = std::chrono::steady_clock::now();
                ggml_backend_free(cuda_backend);
                if (!output_opt.has_value()) {
                    std::cerr << "Lens block0 CUDA ggml compute failed\n";
                    return 1;
                }
                Tensor combined = from_sd_csb_as_bsh(*output_opt);
                Tensor cuda_hidden = slice_seq3(combined, 0, dim(input_hidden, 1));
                Tensor cuda_encoder = slice_seq3(combined, dim(input_hidden, 1), dim(input_encoder, 1));
                print_tensor_range_stats("Lens CUDA block0 encoder stats:", 0, cuda_encoder);
                print_tensor_range_stats("Lens CUDA block0 hidden stats:", 0, cuda_hidden);
                const DiffStats encoder_stats = diff_stats(cuda_encoder, need(block_oracle, "expected.encoder"));
                const DiffStats hidden_stats = diff_stats(cuda_hidden, need(block_oracle, "expected.hidden"));
                print_diff_stats("Lens CUDA block0 encoder parity:", 0, encoder_stats);
                print_diff_stats("Lens CUDA block0 hidden parity:", 0, hidden_stats);
                if (has_nonfinite(encoder_stats) || has_nonfinite(hidden_stats)) {
                    std::cerr << "Lens CUDA block0 produced or compared against nonfinite values\n";
                    return 1;
                }
                std::cout << "Lens CUDA block0 smoke passed: block=0"
                          << " block_tensors=" << block_weights.size()
                          << " block_bytes=" << block_bytes
                          << " output_hidden_shape=1x" << dim(cuda_hidden, 1) << "x" << dim(cuda_hidden, 2)
                          << " output_encoder_shape=1x" << dim(cuda_encoder, 1) << "x" << dim(cuda_encoder, 2)
                          << " runtime_seconds=" << std::chrono::duration<double>(end - start).count()
                          << " attention_mode=" << lens_attention_mode_name(lens_attention_mode)
                          << " backend=ggml_cuda scalar_cpu_matmul=false\n";
                if (encoder_stats.rel_max > tolerance || hidden_stats.rel_max > tolerance) {
                    std::cerr << "Lens CUDA block0 parity exceeded relative tolerance " << tolerance << "\n";
                    return 1;
                }
                return 0;
#endif
            }
            if (!real_full_oracle_fixture.empty()) {
                auto full_oracle = load_fixture(real_full_oracle_fixture);
                for (const char* name : {"input.hidden",
                                         "input.encoder",
                                         "input.temb",
                                         "input.img_freqs",
                                         "input.txt_freqs",
                                         "input.attention_mask",
                                         "expected.prediction"}) {
                    if (!has_tensor(full_oracle, name)) {
                        throw std::runtime_error(std::string("--real-full-oracle-fixture missing tensor: ") + name);
                    }
                }
                const Tensor& input_hidden = need(full_oracle, "input.hidden");
                const Tensor& input_encoder = need(full_oracle, "input.encoder");
                const Tensor& input_temb = need(full_oracle, "input.temb");
                const Tensor& input_img_freqs = need(full_oracle, "input.img_freqs");
                const Tensor& input_txt_freqs = need(full_oracle, "input.txt_freqs");
                const Tensor& input_mask = need(full_oracle, "input.attention_mask");
                if (dim(input_hidden, 0) != 1 || dim(input_hidden, 1) != 256 || dim(input_hidden, 2) != 1536 ||
                    dim(input_encoder, 0) != 1 || dim(input_encoder, 2) != 1536 ||
                    dim(input_temb, 0) != 1 || dim(input_temb, 1) != 1536 ||
                    dim(input_img_freqs, 0) != 256 || dim(input_img_freqs, 1) != 32 || dim(input_img_freqs, 2) != 2 ||
                    dim(input_txt_freqs, 0) != dim(input_encoder, 1) || dim(input_txt_freqs, 1) != 32 || dim(input_txt_freqs, 2) != 2 ||
                    dim(input_mask, 0) != 1 || dim(input_mask, 1) != dim(input_hidden, 1) + dim(input_encoder, 1)) {
                    std::cerr << "full-transformer oracle input shape mismatch before native forward\n";
                    return 1;
                }
                std::vector<std::string> top_names = {
                    "norm_out.linear.weight",
                    "norm_out.linear.bias",
                    "proj_out.weight",
                    "proj_out.bias",
                };
                auto top_tensors = load_real_lens_named_tensors(real_block_transformer, top_names, max_real_top_bytes);
                uint64_t total_loaded_bytes = 0;
                size_t total_loaded_tensors = 0;
                for (const auto& item : top_tensors) {
                    total_loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                }
                total_loaded_tensors += top_tensors.size();
                Tensor hidden_state = input_hidden;
                Tensor encoder_state = input_encoder;
                const auto start_time = std::chrono::steady_clock::now();
                int first_divergent_block = -1;
                float worst_rel = 0.0f;
                float worst_abs = 0.0f;
                for (int block_index = 0; block_index < 48; ++block_index) {
                    auto tensors = load_real_lens_block(real_block_transformer, block_index, max_real_block_bytes);
                    const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                    const int64_t qkv_out = dim(need(tensors, prefix + "attn.img_qkv.weight"), 0);
                    const int64_t block_hidden = dim(need(tensors, prefix + "attn.img_qkv.weight"), 1);
                    const int64_t block_head_dim = dim(need(tensors, prefix + "attn.norm_q.weight"), 0);
                    const int64_t heads = qkv_out / (3 * block_head_dim);
                    if (qkv_out != 3 * block_hidden || block_hidden != 1536 || block_head_dim != 64 || heads != 24) {
                        std::cerr << "real Lens block metadata mismatch at block " << block_index << "\n";
                        return 1;
                    }
                    uint64_t loaded_bytes = 0;
                    for (const auto& item : tensors) {
                        loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                    }
                    total_loaded_bytes += loaded_bytes;
                    total_loaded_tensors += tensors.size();
                    auto outputs = lens_block_forward_prefixed(tensors,
                                                               hidden_state,
                                                               encoder_state,
                                                               input_temb,
                                                               input_img_freqs,
                                                               input_txt_freqs,
                                                               input_mask,
                                                               prefix);
                    encoder_state = std::move(outputs.first);
                    hidden_state = std::move(outputs.second);
                    tensors.clear();
                    float encoder_max = 0.0f;
                    float encoder_mean = 0.0f;
                    float hidden_max = 0.0f;
                    float hidden_mean = 0.0f;
                    if (!tensor_stats(encoder_state, encoder_max, encoder_mean) ||
                        !tensor_stats(hidden_state, hidden_max, hidden_mean)) {
                        std::cerr << "real Lens full-transformer produced non-finite output at block " << block_index << "\n";
                        return 1;
                    }
                    const std::string checkpoint_prefix = "checkpoint.block_" + std::to_string(block_index) + ".";
                    if (has_tensor(full_oracle, checkpoint_prefix + "encoder")) {
                        const DiffStats stats = diff_stats(encoder_state, need(full_oracle, checkpoint_prefix + "encoder"));
                        print_diff_stats("Lens full checkpoint encoder parity:", block_index, stats);
                        worst_rel = std::max(worst_rel, stats.rel_max);
                        worst_abs = std::max(worst_abs, stats.max_diff);
                        if (first_divergent_block < 0 && stats.rel_max > tolerance) {
                            first_divergent_block = block_index;
                        }
                    }
                    if (has_tensor(full_oracle, checkpoint_prefix + "hidden")) {
                        const DiffStats stats = diff_stats(hidden_state, need(full_oracle, checkpoint_prefix + "hidden"));
                        print_diff_stats("Lens full checkpoint hidden parity:", block_index, stats);
                        worst_rel = std::max(worst_rel, stats.rel_max);
                        worst_abs = std::max(worst_abs, stats.max_diff);
                        if (first_divergent_block < 0 && stats.rel_max > tolerance) {
                            first_divergent_block = block_index;
                        }
                    }
                }
                hidden_state = final_ada_norm(hidden_state,
                                              input_temb,
                                              need(top_tensors, "norm_out.linear.weight"),
                                              need(top_tensors, "norm_out.linear.bias"));
                Tensor projected = linear3(hidden_state,
                                           need(top_tensors, "proj_out.weight"),
                                           &need(top_tensors, "proj_out.bias"));
                if (dim(projected, 0) != 1 || dim(projected, 1) != 256 || dim(projected, 2) != 128) {
                    std::cerr << "native full-transformer prediction shape mismatch\n";
                    return 1;
                }
                const DiffStats final_stats = diff_stats(projected, need(full_oracle, "expected.prediction"));
                print_diff_stats("Lens full prediction parity:", -1, final_stats);
                worst_rel = std::max(worst_rel, final_stats.rel_max);
                worst_abs = std::max(worst_abs, final_stats.max_diff);
                const auto end_time = std::chrono::steady_clock::now();
                const double runtime_seconds = std::chrono::duration<double>(end_time - start_time).count();
                std::cout << "Lens full native transformer smoke passed: blocks=48"
                          << " total_tensors=" << total_loaded_tensors
                          << " total_streamed_bytes=" << total_loaded_bytes
                          << " loading=streamed"
                          << " output_shape=1x256x128"
                          << " runtime_seconds=" << runtime_seconds
                          << " worst_abs=" << worst_abs
                          << " worst_rel=" << worst_rel
                          << " first_divergent_block=" << first_divergent_block
                          << " cpu_reference=true ggml_graph=false cuda=false python_owned=conditioning_and_oracle_only\n";
                if (final_stats.rel_max > tolerance) {
                    std::cerr << "Lens full prediction parity exceeded relative tolerance " << tolerance << "\n";
                    return 1;
                }
                return 0;
            }
            if (!real_block_oracle_fixture.empty()) {
                auto block_oracle = load_fixture(real_block_oracle_fixture);
                for (const char* name : {"input.hidden",
                                         "input.encoder",
                                         "input.temb",
                                         "input.img_freqs",
                                         "input.txt_freqs",
                                         "input.attention_mask",
                                         "expected.encoder",
                                         "expected.hidden"}) {
                    if (!has_tensor(block_oracle, name)) {
                        throw std::runtime_error(std::string("--real-block-oracle-fixture missing tensor: ") + name);
                    }
                }
                if (real_block_index != 0) {
                    std::cerr << "--real-block-oracle-fixture currently validates block 0 only\n";
                    return 2;
                }
                auto tensors = load_real_lens_block(real_block_transformer, real_block_index, max_real_block_bytes);
                const std::string prefix = "transformer_blocks." + std::to_string(real_block_index) + ".";
                const int64_t qkv_out = dim(need(tensors, prefix + "attn.img_qkv.weight"), 0);
                const int64_t block_hidden = dim(need(tensors, prefix + "attn.img_qkv.weight"), 1);
                const int64_t block_head_dim = dim(need(tensors, prefix + "attn.norm_q.weight"), 0);
                const int64_t heads = qkv_out / (3 * block_head_dim);
                if (qkv_out != 3 * block_hidden ||
                    block_hidden != 1536 ||
                    block_head_dim != 64 ||
                    heads != 24) {
                    std::cerr << "real Lens block0 metadata mismatch: hidden=" << block_hidden
                              << " heads=" << heads
                              << " head_dim=" << block_head_dim
                              << " qkv_out=" << qkv_out << "\n";
                    return 1;
                }
                const Tensor& input_hidden = need(block_oracle, "input.hidden");
                const Tensor& input_encoder = need(block_oracle, "input.encoder");
                const Tensor& input_temb = need(block_oracle, "input.temb");
                const Tensor& input_img_freqs = need(block_oracle, "input.img_freqs");
                const Tensor& input_txt_freqs = need(block_oracle, "input.txt_freqs");
                const Tensor& input_mask = need(block_oracle, "input.attention_mask");
                if (dim(input_hidden, 0) != 1 || dim(input_hidden, 1) != 256 || dim(input_hidden, 2) != block_hidden) {
                    std::cerr << "block0 oracle input.hidden must be [1,256,1536]\n";
                    return 1;
                }
                if (dim(input_encoder, 0) != 1 || dim(input_encoder, 2) != block_hidden) {
                    std::cerr << "block0 oracle input.encoder must be [1,S_txt,1536]\n";
                    return 1;
                }
                const int64_t txt_seq = dim(input_encoder, 1);
                if (dim(input_temb, 0) != 1 || dim(input_temb, 1) != block_hidden ||
                    dim(input_img_freqs, 0) != 256 || dim(input_img_freqs, 1) * 2 != block_head_dim || dim(input_img_freqs, 2) != 2 ||
                    dim(input_txt_freqs, 0) != txt_seq || dim(input_txt_freqs, 1) * 2 != block_head_dim || dim(input_txt_freqs, 2) != 2 ||
                    dim(input_mask, 0) != 1 || dim(input_mask, 1) != 256 + txt_seq) {
                    std::cerr << "block0 oracle input shape mismatch before native forward\n";
                    return 1;
                }
                auto outputs = lens_block_forward_prefixed(tensors,
                                                           input_hidden,
                                                           input_encoder,
                                                           input_temb,
                                                           input_img_freqs,
                                                           input_txt_freqs,
                                                           input_mask,
                                                           prefix);
                float mean_encoder = 0.0f;
                float mean_hidden = 0.0f;
                float max_encoder = max_abs_diff(outputs.first, need(block_oracle, "expected.encoder"), &mean_encoder);
                float max_hidden = max_abs_diff(outputs.second, need(block_oracle, "expected.hidden"), &mean_hidden);
                std::cout << "Lens real block0 oracle parity: block=0"
                          << " input_hidden=1x256x" << block_hidden
                          << " input_encoder=1x" << txt_seq << "x" << block_hidden
                          << " temb=1x" << block_hidden
                          << " img_freqs=256x" << dim(input_img_freqs, 1) << "x2"
                          << " txt_freqs=" << txt_seq << "x" << dim(input_txt_freqs, 1) << "x2"
                          << " mask=1x" << dim(input_mask, 1)
                          << " max_encoder=" << max_encoder
                          << " mean_encoder=" << mean_encoder
                          << " max_hidden=" << max_hidden
                          << " mean_hidden=" << mean_hidden
                          << " native_real_block_forward=true python_owned=conditioning_and_oracle_only\n";
                if (max_encoder > tolerance || max_hidden > tolerance) {
                    float ignored = 0.0f;
                    compare_tensor_with_tolerance(outputs.first, need(block_oracle, "expected.encoder"), tolerance, "Lens block0 encoder oracle", max_encoder, ignored);
                    compare_tensor_with_tolerance(outputs.second, need(block_oracle, "expected.hidden"), tolerance, "Lens block0 hidden oracle", max_hidden, ignored);
                    std::cerr << "Lens real block0 oracle parity exceeded tolerance " << tolerance << "\n";
                    return 1;
                }
                return 0;
            }
            std::unordered_map<std::string, Tensor> oracle_inputs;
            if (!oracle_input_fixture.empty()) {
                oracle_inputs = load_fixture(oracle_input_fixture);
                for (const char* name : {"full.input.hidden",
                                         "full.input.temb",
                                         "full.input.img_freqs",
                                         "full.input.txt_freqs",
                                         "full.input.attention_mask"}) {
                    if (!has_tensor(oracle_inputs, name)) {
                        throw std::runtime_error(std::string("--oracle-input-fixture missing tensor: ") + name);
                    }
                }
            }
            std::unordered_map<std::string, Tensor> top_tensors;
            Tensor hidden_state;
            Tensor encoder_state;
            Tensor temb;
            Tensor img_freqs;
            Tensor txt_freqs;
            Tensor mask;
            Tensor base_encoder_state;
            int64_t hidden = 0;
            int64_t head_dim = 0;
            int64_t enc_hidden_dim = 0;
            uint64_t total_loaded_bytes = 0;
            size_t total_loaded_tensors = 0;
            bool using_precomputed_cond = false;
            Tensor precomputed_cond_mask;
            Tensor public_packed_tokens;
            if (real_full_transformer) {
                std::vector<std::string> top_names = {
                    "img_in.weight",
                    "img_in.bias",
                    "txt_norm.0.weight",
                    "txt_norm.1.weight",
                    "txt_norm.2.weight",
                    "txt_norm.3.weight",
                    "txt_in.weight",
                    "txt_in.bias",
                    "norm_out.linear.weight",
                    "norm_out.linear.bias",
                    "proj_out.weight",
                    "proj_out.bias",
                };
                top_tensors = load_real_lens_named_tensors(real_block_transformer, top_names, max_real_top_bytes);
                total_loaded_tensors += top_tensors.size();
                for (const auto& item : top_tensors) {
                    total_loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                }

                const int64_t in_channels = dim(need(top_tensors, "img_in.weight"), 1);
                const int64_t inner_dim = dim(need(top_tensors, "img_in.weight"), 0);
                enc_hidden_dim = dim(need(top_tensors, "txt_norm.0.weight"), 0);
                if (inner_dim <= 0 || in_channels <= 0 || enc_hidden_dim <= 0 ||
                    dim(need(top_tensors, "txt_in.weight"), 1) != enc_hidden_dim * 4 ||
                    dim(need(top_tensors, "txt_in.weight"), 0) != inner_dim) {
                    std::cerr << "real Lens top-level tensor shapes are inconsistent\n";
                    return 1;
                }
                if (!oracle_inputs.empty()) {
                    public_packed_tokens = need(oracle_inputs, "full.input.hidden");
                    if (dim(public_packed_tokens, 0) != 1 ||
                        dim(public_packed_tokens, 2) != in_channels) {
                        std::cerr << "oracle full.input.hidden must have shape [1,S," << in_channels << "]\n";
                        return 1;
                    }
                    real_img_seq = static_cast<int>(dim(public_packed_tokens, 1));
                } else {
                    public_packed_tokens = make_pattern3(1, real_img_seq, in_channels, 0.001f);
                }
                hidden_state = linear3(public_packed_tokens, need(top_tensors, "img_in.weight"), &need(top_tensors, "img_in.bias"));

                std::unordered_map<std::string, Tensor> cond_tensors;
                if (!lens_cond_path.empty()) {
                    cond_tensors = load_real_lens_named_tensors(lens_cond_path,
                                                                {"feature_0", "feature_1", "feature_2", "feature_3", "attention_mask"},
                                                                max_real_cond_bytes,
                                                                true);
                    const Tensor& f0 = need(cond_tensors, "feature_0");
                    if (dim(f0, 0) != 1 || dim(f0, 1) <= 0 || dim(f0, 2) != enc_hidden_dim) {
                        std::cerr << "lens_cond_v1 feature_0 shape does not match real Lens txt_norm hidden size\n";
                        return 1;
                    }
                    for (int i = 1; i < 4; ++i) {
                        const Tensor& f = need(cond_tensors, "feature_" + std::to_string(i));
                        if (dim(f, 0) != dim(f0, 0) || dim(f, 1) != dim(f0, 1) || dim(f, 2) != dim(f0, 2)) {
                            std::cerr << "lens_cond_v1 feature shapes do not match each other\n";
                            return 1;
                        }
                    }
                    const Tensor& cond_mask = need(cond_tensors, "attention_mask");
                    if (dim(cond_mask, 0) != 1 || dim(cond_mask, 1) != dim(f0, 1)) {
                        std::cerr << "lens_cond_v1 attention_mask shape does not match feature sequence length\n";
                        return 1;
                    }
                    real_txt_seq = static_cast<int>(dim(f0, 1));
                    precomputed_cond_mask = make_joint_attention_mask_from_lens_cond(cond_mask, real_img_seq);
                    using_precomputed_cond = true;
                    uint64_t cond_bytes = 0;
                    for (const auto& item : cond_tensors) {
                        cond_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                    }
                    total_loaded_tensors += cond_tensors.size();
                    total_loaded_bytes += cond_bytes;
                    std::cout << "Lens precomputed conditioning smoke: tensors=" << cond_tensors.size()
                              << " bytes=" << cond_bytes
                              << " seq=" << real_txt_seq
                              << " hidden=" << enc_hidden_dim
                              << " using_precomputed_cond=true\n";
                }

                std::vector<Tensor> features;
                features.reserve(4);
                for (int i = 0; i < 4; ++i) {
                    const std::string idx = std::to_string(i);
                    if (cond_tensors.empty()) {
                        Tensor feature = make_pattern3(1, real_txt_seq, enc_hidden_dim, 0.001f * static_cast<float>(i + 1));
                        features.push_back(rms_norm3(feature, need(top_tensors, "txt_norm." + idx + ".weight"), 1.0e-5f));
                    } else {
                        features.push_back(rms_norm3(need(cond_tensors, "feature_" + idx),
                                                     need(top_tensors, "txt_norm." + idx + ".weight"),
                                                     1.0e-5f));
                    }
                }
                encoder_state = linear3(concat_features(features), need(top_tensors, "txt_in.weight"), &need(top_tensors, "txt_in.bias"));
                base_encoder_state = encoder_state;
                if (!oracle_inputs.empty()) {
                    temb = has_tensor(oracle_inputs, "full.input.temb_0") ? need(oracle_inputs, "full.input.temb_0")
                                                                           : need(oracle_inputs, "full.input.temb");
                    if (dim(temb, 0) != 1 || dim(temb, 1) != inner_dim) {
                        std::cerr << "oracle full.input.temb must have shape [1," << inner_dim << "]\n";
                        return 1;
                    }
                } else {
                    temb = make_pattern2(1, inner_dim, 0.001f);
                }
                hidden = inner_dim;
                std::cout << "Lens real top-level smoke: tensors=" << top_tensors.size()
                          << " bytes=" << total_loaded_bytes
                          << " in_channels=" << in_channels
                          << " enc_hidden_dim=" << enc_hidden_dim
                          << " inner_dim=" << inner_dim
                          << " native_real_top_level=true"
                          << " source_backed_inputs=" << (!oracle_inputs.empty() ? "true" : "false") << "\n";
            }
            for (int block_offset = 0; block_offset < real_num_blocks; ++block_offset) {
                const int block_index = real_block_index + block_offset;
                auto tensors = load_real_lens_block(real_block_transformer, block_index, max_real_block_bytes);
                const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                const int64_t qkv_out = dim(need(tensors, prefix + "attn.img_qkv.weight"), 0);
                const int64_t block_hidden = dim(need(tensors, prefix + "attn.img_qkv.weight"), 1);
                const int64_t block_head_dim = dim(need(tensors, prefix + "attn.norm_q.weight"), 0);
                if (qkv_out != 3 * block_hidden || block_hidden <= 0 || block_head_dim <= 0 || block_hidden % block_head_dim != 0) {
                    std::cerr << "real Lens block tensor shapes are inconsistent for block " << block_index << "\n";
                    return 1;
                }
                if (block_offset == 0) {
                    if (!real_full_transformer) {
                        hidden = block_hidden;
                        hidden_state = make_pattern3(1, real_img_seq, hidden, 0.001f);
                        encoder_state = make_pattern3(1, real_txt_seq, hidden, 0.001f);
                        temb = make_pattern2(1, hidden, 0.001f);
                    } else if (block_hidden != hidden) {
                        std::cerr << "real Lens top-level hidden size does not match block hidden size\n";
                        return 1;
                    }
                    head_dim = block_head_dim;
                    if (!oracle_inputs.empty()) {
                        img_freqs = need(oracle_inputs, "full.input.img_freqs");
                        txt_freqs = need(oracle_inputs, "full.input.txt_freqs");
                        mask = need(oracle_inputs, "full.input.attention_mask");
                        if (dim(img_freqs, 0) < real_img_seq || dim(img_freqs, 1) * 2 != head_dim || dim(img_freqs, 2) != 2) {
                            std::cerr << "oracle full.input.img_freqs shape does not match Lens block RoPE requirements\n";
                            return 1;
                        }
                        if (dim(txt_freqs, 0) < real_txt_seq || dim(txt_freqs, 1) * 2 != head_dim || dim(txt_freqs, 2) != 2) {
                            std::cerr << "oracle full.input.txt_freqs shape does not match Lens block RoPE requirements\n";
                            return 1;
                        }
                        if (dim(mask, 0) != 1 || dim(mask, 1) != real_img_seq + real_txt_seq) {
                            std::cerr << "oracle full.input.attention_mask must have shape [1,img_seq+txt_seq]\n";
                            return 1;
                        }
                    } else {
                        img_freqs = make_identity_freqs(real_img_seq, head_dim);
                        txt_freqs = make_identity_freqs(real_txt_seq, head_dim);
                        mask = using_precomputed_cond ? precomputed_cond_mask : make_attention_mask(1, real_img_seq + real_txt_seq);
                    }
                } else if (block_hidden != hidden || block_head_dim != head_dim) {
                    std::cerr << "real Lens block shape changed across streamed blocks\n";
                    return 1;
                }

                uint64_t loaded_bytes = 0;
                for (const auto& item : tensors) {
                    loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                }
                if (loaded_bytes > max_real_block_bytes ||
                    total_loaded_bytes > std::numeric_limits<uint64_t>::max() - loaded_bytes) {
                    std::cerr << "real Lens streamed block byte accounting overflowed or exceeded the cap\n";
                    return 1;
                }
                total_loaded_bytes += loaded_bytes;
                total_loaded_tensors += tensors.size();

                auto outputs = lens_block_forward_prefixed(tensors,
                                                           hidden_state,
                                                           encoder_state,
                                                           temb,
                                                           img_freqs,
                                                           txt_freqs,
                                                           mask,
                                                           prefix);
                hidden_state = std::move(outputs.second);
                encoder_state = std::move(outputs.first);

                float encoder_max = 0.0f;
                float encoder_mean = 0.0f;
                float hidden_max = 0.0f;
                float hidden_mean = 0.0f;
                if (!tensor_stats(encoder_state, encoder_max, encoder_mean) ||
                    !tensor_stats(hidden_state, hidden_max, hidden_mean)) {
                    std::cerr << "real Lens block forward produced non-finite output at block " << block_index << "\n";
                    return 1;
                }
                std::cout << "Lens real block smoke: block=" << block_index
                          << " tensors=" << tensors.size()
                          << " bytes=" << loaded_bytes
                          << " hidden=" << hidden
                          << " heads=" << (hidden / head_dim)
                          << " head_dim=" << head_dim
                          << " img_seq=" << real_img_seq
                          << " txt_seq=" << real_txt_seq
                          << " max_encoder=" << encoder_max
                          << " mean_encoder=" << encoder_mean
                          << " max_hidden=" << hidden_max
                          << " mean_hidden=" << hidden_mean
                          << " native_real_block_forward=true full_model_forward=false\n";
                tensors.clear();
            }
            std::cout << "Lens streamed real block smoke passed: start_block=" << real_block_index
                      << " blocks=" << real_num_blocks
                      << " total_tensors=" << total_loaded_tensors
                      << " total_streamed_bytes=" << total_loaded_bytes
                      << " hidden=" << hidden
                      << " img_seq=" << real_img_seq
                      << " txt_seq=" << real_txt_seq
                      << " native_streamed_blocks=true full_model_forward=false\n";
            if (real_full_transformer) {
                hidden_state = final_ada_norm(hidden_state,
                                              temb,
                                              need(top_tensors, "norm_out.linear.weight"),
                                              need(top_tensors, "norm_out.linear.bias"));
                Tensor projected = linear3(hidden_state,
                                           need(top_tensors, "proj_out.weight"),
                                           &need(top_tensors, "proj_out.bias"));
                float projected_max = 0.0f;
                float projected_mean = 0.0f;
                if (!tensor_stats(projected, projected_max, projected_mean)) {
                    std::cerr << "real Lens full-transformer smoke produced non-finite output\n";
                    return 1;
                }
                std::cout << "Lens real tiny transformer smoke passed: blocks=" << real_num_blocks
                          << " output_shape=1x" << real_img_seq << "x" << dim(projected, 2)
                          << " max_output=" << projected_max
                          << " mean_output=" << projected_mean
                          << " native_real_tiny_transformer=true text_encoder_native=false gpu_inference=false\n";
                if (emit_tiny_denoise_npy) {
                    const int64_t packed_side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(real_img_seq))));
                    if (packed_side * packed_side != real_img_seq) {
                        std::cerr << "--emit-tiny-denoise-npy requires a square real image token count\n";
                        return 1;
                    }
                    Tensor denoise_tokens = projected;
                    if (apply_tiny_flow_step) {
                        if (tiny_flow_steps <= 0) {
                            std::cerr << "--tiny-flow-steps must be positive\n";
                            return 1;
                        }
                        const std::vector<float> sigmas = build_lens_tiny_flow_sigmas(tiny_flow_steps, real_img_seq);
                        const size_t token_elements = public_packed_tokens.data.size();
                        std::vector<float> external_model_outputs(token_elements * static_cast<size_t>(tiny_flow_steps), 0.0f);
                        if (base_encoder_state.data.empty()) {
                            throw std::runtime_error("Lens tiny flow loop missing base encoder state");
                        }
                        auto run_tiny_projector = [&](const Tensor& packed_tokens, int step_index) {
                            const Tensor* step_temb = &temb;
                            if (!oracle_inputs.empty()) {
                                const std::string step_temb_name = "full.input.temb_" + std::to_string(step_index);
                                if (has_tensor(oracle_inputs, step_temb_name)) {
                                    step_temb = &need(oracle_inputs, step_temb_name);
                                    if (dim(*step_temb, 0) != 1 || dim(*step_temb, 1) != hidden) {
                                        throw std::runtime_error("oracle per-step temb shape mismatch");
                                    }
                                }
                            }
                            Tensor step_hidden = linear3(packed_tokens,
                                                         need(top_tensors, "img_in.weight"),
                                                         &need(top_tensors, "img_in.bias"));
                            Tensor step_encoder = base_encoder_state;
                            uint64_t step_streamed_bytes = 0;
                            size_t step_streamed_tensors = 0;
                            for (int block_offset = 0; block_offset < real_num_blocks; ++block_offset) {
                                const int block_index = real_block_index + block_offset;
                                auto tensors = load_real_lens_block(real_block_transformer, block_index, max_real_block_bytes);
                                const std::string prefix = "transformer_blocks." + std::to_string(block_index) + ".";
                                const int64_t block_hidden = dim(need(tensors, prefix + "attn.img_qkv.weight"), 1);
                                const int64_t block_head_dim = dim(need(tensors, prefix + "attn.norm_q.weight"), 0);
                                if (block_hidden != hidden || block_head_dim != head_dim) {
                                    throw std::runtime_error("real Lens tiny flow loop block shape mismatch");
                                }
                                uint64_t loaded_bytes = 0;
                                for (const auto& item : tensors) {
                                    loaded_bytes += static_cast<uint64_t>(item.second.data.size() * sizeof(float));
                                }
                                step_streamed_bytes += loaded_bytes;
                                step_streamed_tensors += tensors.size();
                                auto outputs = lens_block_forward_prefixed(tensors,
                                                                           step_hidden,
                                                                           step_encoder,
                                                                           *step_temb,
                                                                           img_freqs,
                                                                           txt_freqs,
                                                                           mask,
                                                                           prefix);
                                step_hidden = std::move(outputs.second);
                                step_encoder = std::move(outputs.first);
                            }
                            step_hidden = final_ada_norm(step_hidden,
                                                         *step_temb,
                                                         need(top_tensors, "norm_out.linear.weight"),
                                                         need(top_tensors, "norm_out.linear.bias"));
                            Tensor step_projected = linear3(step_hidden,
                                                            need(top_tensors, "proj_out.weight"),
                                                            &need(top_tensors, "proj_out.bias"));
                            float step_projected_max = 0.0f;
                            float step_projected_mean = 0.0f;
                            if (!tensor_stats(step_projected, step_projected_max, step_projected_mean)) {
                                throw std::runtime_error("Lens tiny flow loop model pass produced non-finite output");
                            }
                            std::cout << "Lens tiny flow loop model pass: step=" << step_index
                                      << " tensors=" << step_streamed_tensors
                                      << " bytes=" << step_streamed_bytes
                                      << " max_output=" << step_projected_max
                                      << " mean_output=" << step_projected_mean
                                      << " source_backed_temb=" << ((!oracle_inputs.empty() && has_tensor(oracle_inputs, "full.input.temb_" + std::to_string(step_index))) ? "true" : "false")
                                      << "\n";
                            return step_projected;
                        };
                        Tensor current_tokens = public_packed_tokens;
                        Tensor model_output = projected;
                        for (int step_index = 0; step_index < tiny_flow_steps; ++step_index) {
                            if (step_index > 0) {
                                model_output = run_tiny_projector(current_tokens, step_index);
                            }
                            if (model_output.shape != public_packed_tokens.shape ||
                                model_output.data.size() != token_elements) {
                                std::cerr << "Lens tiny flow loop model output shape does not match packed tokens\n";
                                return 1;
                            }
                            std::copy(model_output.data.begin(),
                                      model_output.data.end(),
                                      external_model_outputs.begin() + static_cast<size_t>(step_index) * token_elements);
                            float sigma0 = 0.0f;
                            float sigma1 = 0.0f;
                            float dt = 0.0f;
                            current_tokens = apply_lens_flow_step_at(current_tokens,
                                                                     model_output,
                                                                     sigmas,
                                                                     step_index,
                                                                     sigma0,
                                                                     sigma1,
                                                                     dt);
                            float current_max = 0.0f;
                            float current_mean = 0.0f;
                            if (!tensor_stats(current_tokens, current_max, current_mean)) {
                                std::cerr << "Lens tiny flow loop produced non-finite sample at step " << step_index << "\n";
                                return 1;
                            }
                            std::cout << "Lens tiny flow loop step: step=" << step_index
                                      << " steps=" << tiny_flow_steps
                                      << " image_seq_len=" << real_img_seq
                                      << " sigma0=" << sigma0
                                      << " sigma1=" << sigma1
                                      << " dt=" << dt
                                      << " max_sample=" << current_max
                                      << " mean_sample=" << current_mean
                                      << " flow_update=true\n";
                        }
                        denoise_tokens = current_tokens;
                        if (verify_external_flow_api) {
                            Tensor api_tokens;
                            sd_lens_external_flow_loop_desc_t flow_desc;
                            if (!run_external_flow_api_check(lens_cond_path,
                                                             real_block_transformer,
                                                             public_packed_tokens,
                                                             external_model_outputs,
                                                             tiny_flow_steps,
                                                             real_img_seq,
                                                             static_cast<uint32_t>(dim(public_packed_tokens, 2)),
                                                             static_cast<uint32_t>(enc_hidden_dim),
                                                             api_tokens,
                                                             flow_desc)) {
                                return 1;
                            }
                            float api_mean_diff = 0.0f;
                            const float api_max_diff = max_abs_diff(api_tokens, current_tokens, &api_mean_diff);
                            if (api_max_diff > 1.0e-6f) {
                                std::cerr << "Lens external flow API output mismatch: max_diff=" << api_max_diff
                                          << " mean_diff=" << api_mean_diff << "\n";
                                return 1;
                            }
                            denoise_tokens = std::move(api_tokens);
                            std::cout << "Lens external flow API check passed: steps=" << flow_desc.steps
                                      << " image_seq_len=" << flow_desc.image_seq_len
                                      << " elements=" << flow_desc.packed_token_elements
                                      << " max_diff=" << api_max_diff
                                      << " mean_diff=" << api_mean_diff
                                      << " max_abs=" << flow_desc.max_abs
                                      << " mean_abs=" << flow_desc.mean_abs
                                      << " used_precomputed_conditioning="
                                      << (flow_desc.used_precomputed_conditioning ? "true" : "false")
                                      << " used_external_model_output="
                                      << (flow_desc.used_external_model_output ? "true" : "false")
                                      << " native_transformer_forward="
                                      << (flow_desc.native_transformer_forward ? "true" : "false")
                                      << " cpu_only=" << (flow_desc.cpu_only ? "true" : "false")
                                      << " public_api_flow_loop=true\n";
                        }
                        std::cout << "Lens tiny flow loop passed: steps=" << tiny_flow_steps
                                  << " image_seq_len=" << real_img_seq
                                  << " iterative_model_recompute=true\n";
                    }
                    const int64_t latent_h = packed_side * 2;
                    const int64_t latent_w = packed_side * 2;
                    Tensor latent{{1, 32, latent_h, latent_w},
                                  std::vector<float>(static_cast<size_t>(32 * latent_h * latent_w), 0.0f)};
                    sd_lens_vae_latent_desc_t unpack_desc;
                    sd_lens_vae_latent_desc_init(&unpack_desc);
                    if (!sd_lens_unpack_vae_latent_f32(denoise_tokens.data.data(),
                                                       static_cast<uint64_t>(denoise_tokens.data.size()),
                                                       1,
                                                       32,
                                                       latent_h,
                                                       latent_w,
                                                       latent.data.data(),
                                                       static_cast<uint64_t>(latent.data.size()),
                                                       &unpack_desc) ||
                        unpack_desc.output_n != 1 || unpack_desc.output_c != 32 ||
                        unpack_desc.output_h != latent_h || unpack_desc.output_w != latent_w) {
                        std::cerr << "sd_lens_unpack_vae_latent_f32 failed for Lens tiny denoise latent\n";
                        return 1;
                    }
                    float latent_max = 0.0f;
                    float latent_mean = 0.0f;
                    if (!tensor_stats(latent, latent_max, latent_mean) || !write_f32_npy(tiny_denoise_npy, latent)) {
                        std::cerr << "failed to write Lens tiny denoise latent NPY\n";
                        return 1;
                    }
                    std::cout << "Lens tiny denoise latent wrote " << tiny_denoise_npy
                              << " shape=1x32x" << dim(latent, 2) << "x" << dim(latent, 3)
                              << " max_abs=" << latent_max
                              << " mean_abs=" << latent_mean
                              << " denoise_artifact=true"
                              << " flow_step_applied=" << (apply_tiny_flow_step ? "true" : "false") << "\n";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "sd-lens-transformer-smoke failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

SD_LENS_TRANSFORMER_SMOKE_API int sd_lens_transformer_smoke_main_impl(
    int argc,
    char** argv,
    const std::unordered_map<std::string, Tensor>* in_memory_cond_tensors,
    Tensor* out_native_cuda_latent) {
    return sd_lens_transformer_smoke_main_impl_profiled(
        argc,
        argv,
        in_memory_cond_tensors,
        out_native_cuda_latent,
        nullptr,
        nullptr,
        nullptr);
}

#ifndef SD_LENS_TRANSFORMER_SMOKE_NO_MAIN
int main(int argc, char** argv) {
    return sd_lens_transformer_smoke_main_impl_profiled(argc, argv, nullptr, nullptr, nullptr, nullptr, nullptr);
}
#endif
