#include "bonsai_gemlite_int1.hpp"

#include "ggml.h"
#include "zip.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

uint32_t read_le32(const uint8_t* ptr) {
    return ((uint32_t)ptr[0]) | ((uint32_t)ptr[1] << 8) | ((uint32_t)ptr[2] << 16) | ((uint32_t)ptr[3] << 24);
}

uint64_t read_le64(const uint8_t* ptr) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= ((uint64_t)ptr[i]) << (8 * i);
    }
    return value;
}

bool is_digits(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    for (char c : value) {
        if (!std::isdigit((unsigned char)c)) {
            return false;
        }
    }
    return true;
}

bool looks_like_tensor_name(const std::string& value) {
    if (value == "storage" || value == "cuda:0" || value == "cpu" || is_digits(value)) {
        return false;
    }
    return value.find('.') != std::string::npos;
}

bool find_pickle_dir(zip_t* zip, std::string& dir, std::vector<uint8_t>& data, std::string& error) {
    const int n = (int)zip_entries_total(zip);
    for (int i = 0; i < n; ++i) {
        if (zip_entry_openbyindex(zip, i) != 0) {
            continue;
        }
        const std::string name = zip_entry_name(zip);
        const size_t pos = name.find("data.pkl");
        if (pos != std::string::npos) {
            dir = name.substr(0, pos);
            const size_t size = zip_entry_size(zip);
            data.resize(size);
            const size_t read = zip_entry_noallocread(zip, data.data(), size);
            zip_entry_close(zip);
            if (read != size) {
                error = "short read for data.pkl";
                return false;
            }
            return true;
        }
        zip_entry_close(zip);
    }
    error = "data.pkl not found in Bonsai pack";
    return false;
}

bool index_bonsai_state_dict(const std::string& pack_path,
                             std::string& data_dir,
                             std::map<std::string, std::string>& tensor_to_storage_id,
                             std::string& error) {
    zip_t* zip = zip_open(pack_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        error = "failed to open zip pack: " + pack_path;
        return false;
    }
    std::vector<uint8_t> pickle;
    if (!find_pickle_dir(zip, data_dir, pickle, error)) {
        zip_close(zip);
        return false;
    }
    zip_close(zip);

    std::string pending_tensor_name;
    const uint8_t* ptr = pickle.data();
    const uint8_t* end = pickle.data() + pickle.size();
    auto consume_string = [&](std::string value) {
        if (looks_like_tensor_name(value)) {
            pending_tensor_name = std::move(value);
        } else if (!pending_tensor_name.empty() && is_digits(value)) {
            tensor_to_storage_id[pending_tensor_name] = std::move(value);
            pending_tensor_name.clear();
        }
    };

    while (ptr < end) {
        const uint8_t opcode = *ptr++;
        switch (opcode) {
            case 'X': {
                if (ptr + 4 > end) return true;
                const uint32_t len = read_le32(ptr);
                ptr += 4;
                if (ptr + len > end) return true;
                consume_string(std::string((const char*)ptr, (size_t)len));
                ptr += len;
            } break;
            case 0x8C: {
                if (ptr >= end) return true;
                const uint8_t len = *ptr++;
                if (ptr + len > end) return true;
                consume_string(std::string((const char*)ptr, (size_t)len));
                ptr += len;
            } break;
            case 0x8D: {
                if (ptr + 8 > end) return true;
                const uint64_t len = read_le64(ptr);
                ptr += 8;
                if (ptr + len > end) return true;
                consume_string(std::string((const char*)ptr, (size_t)len));
                ptr += len;
            } break;
            case 'c': {
                while (ptr < end && *ptr++ != '\n') {}
                while (ptr < end && *ptr++ != '\n') {}
            } break;
            case 'q':
            case 'h':
            case 'K':
                ptr += 1;
                break;
            case 'M':
                ptr += 2;
                break;
            case 'J':
            case 'r':
                ptr += 4;
                break;
            case 0x95:
                ptr += 8;
                break;
            default:
                break;
        }
    }
    return true;
}

bool read_zip_entry_by_name(const std::string& pack_path,
                            const std::string& entry_name,
                            std::vector<uint8_t>& out,
                            std::string& error) {
    zip_t* zip = zip_open(pack_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        error = "failed to open zip pack: " + pack_path;
        return false;
    }
    if (zip_entry_open(zip, entry_name.c_str()) != 0) {
        zip_close(zip);
        error = "failed to open zip entry: " + entry_name;
        return false;
    }
    const size_t entry_size = zip_entry_size(zip);
    out.resize(entry_size);
    const size_t read = zip_entry_noallocread(zip, out.data(), entry_size);
    zip_entry_close(zip);
    zip_close(zip);
    if (read != entry_size) {
        error = "short zip read for " + entry_name + ": expected " + std::to_string(entry_size) + " got " + std::to_string(read);
        return false;
    }
    return true;
}

bool read_i32_tensor(const std::string& pack_path,
                     const std::string& entry_name,
                     std::vector<int32_t>& out,
                     std::string& error) {
    std::vector<uint8_t> bytes;
    if (!read_zip_entry_by_name(pack_path, entry_name, bytes, error)) {
        return false;
    }
    out.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

bool read_f32_tensor(const std::string& pack_path,
                     const std::string& entry_name,
                     std::vector<float>& out,
                     std::string& error) {
    std::vector<uint8_t> bytes;
    if (!read_zip_entry_by_name(pack_path, entry_name, bytes, error)) {
        return false;
    }
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

bool tensor_entry(const std::map<std::string, std::string>& index,
                  const std::string& data_dir,
                  const std::string& tensor_name,
                  std::string& out,
                  std::string& error,
                  bool required = true) {
    auto it = index.find(tensor_name);
    if (it == index.end()) {
        if (required) {
            error = "missing Bonsai tensor: " + tensor_name;
        }
        return false;
    }
    out = data_dir + "data/" + it->second;
    return true;
}

}  // namespace

extern "C" bool sd_bonsai_gemlite_int1_device_linear_create(const uint8_t* wq,
                                                             size_t wq_bytes,
                                                             const float* scales,
                                                             const float* zeros,
                                                             int k,
                                                             int n,
                                                             void** out,
                                                             uint64_t* device_bytes,
                                                             char* error,
                                                             size_t error_size);
extern "C" void sd_bonsai_gemlite_int1_device_linear_destroy(void* ptr);
extern "C" void sd_bonsai_gemlite_int1_ggml_custom_forward(ggml_tensor* dst, int ith, int nth, void* userdata);
extern "C" void sd_bonsai_gemlite_int1_ggml_custom_qkv_forward(ggml_tensor* dst, int ith, int nth, void* userdata);
extern "C" void sd_bonsai_gemlite_int1_ggml_debug_identity_forward(ggml_tensor* dst, int ith, int nth, void* userdata);

struct BonsaiGemliteRuntime::Impl {
    struct DeviceLinear {
        std::string hf_name;
        std::string internal_weight_name;
        int64_t out_features = 0;
        int64_t in_features = 0;
        uint64_t packed_bytes = 0;
        uint64_t scale_bytes = 0;
        uint64_t zero_bytes = 0;
        uint64_t device_bytes = 0;
        void* device = nullptr;
        uint64_t calls = 0;
    };

    std::unordered_map<std::string, DeviceLinear> by_internal_name;
    std::unordered_map<std::string, std::unique_ptr<BonsaiGemliteLinearCustomUserdata>> linear_userdata;
    std::unordered_map<std::string, std::unique_ptr<BonsaiGemliteQkvCustomUserdata>> qkv_userdata;
    BonsaiGemliteRuntimeSummary summary;

    ~Impl() {
        for (auto& kv : by_internal_name) {
            sd_bonsai_gemlite_int1_device_linear_destroy(kv.second.device);
            kv.second.device = nullptr;
        }
    }
};

static std::string internal_weight_name_for_bonsai_hf_linear(const std::string& hf_name) {
    auto block_index = [](const std::string& name, const std::string& prefix) -> int {
        if (name.rfind(prefix, 0) != 0) {
            return -1;
        }
        const size_t start = prefix.size();
        const size_t end = name.find('.', start);
        if (end == std::string::npos) {
            return -1;
        }
        return std::atoi(name.substr(start, end - start).c_str());
    };
    int i = block_index(hf_name, "single_transformer_blocks.");
    if (i >= 0) {
        const std::string tail = hf_name.substr(hf_name.find('.', std::string("single_transformer_blocks.").size()) + 1);
        if (tail == "attn.to_qkv_mlp_proj") {
            return "model.diffusion_model.single_blocks." + std::to_string(i) + ".linear1.weight";
        }
        if (tail == "attn.to_out") {
            return "model.diffusion_model.single_blocks." + std::to_string(i) + ".linear2.weight";
        }
        return {};
    }
    i = block_index(hf_name, "transformer_blocks.");
    if (i >= 0) {
        const std::string tail = hf_name.substr(hf_name.find('.', std::string("transformer_blocks.").size()) + 1);
        const std::string base = "model.diffusion_model.double_blocks." + std::to_string(i) + ".";
        if (tail == "ff.linear_in") return base + "img_mlp.0.weight";
        if (tail == "ff.linear_out") return base + "img_mlp.2.weight";
        if (tail == "ff_context.linear_in") return base + "txt_mlp.0.weight";
        if (tail == "ff_context.linear_out") return base + "txt_mlp.2.weight";
        if (tail == "attn.to_out.0") return base + "img_attn.proj.weight";
        if (tail == "attn.to_add_out") return base + "txt_attn.proj.weight";
        if (tail == "attn.to_q") return base + "img_attn.qkv.weight.q";
        if (tail == "attn.to_k") return base + "img_attn.qkv.weight.k";
        if (tail == "attn.to_v") return base + "img_attn.qkv.weight.v";
        if (tail == "attn.add_q_proj") return base + "txt_attn.qkv.weight.q";
        if (tail == "attn.add_k_proj") return base + "txt_attn.qkv.weight.k";
        if (tail == "attn.add_v_proj") return base + "txt_attn.qkv.weight.v";
        return {};
    }
    return {};
}

static bool bonsai_replaces_internal_weight_name(const std::string& weight_name) {
    const std::string prefix = "model.diffusion_model.";
    if (weight_name.rfind(prefix, 0) != 0) {
        return false;
    }
    if (weight_name.find(".single_blocks.") != std::string::npos) {
        return weight_name.find(".linear1.weight") != std::string::npos ||
               weight_name.find(".linear2.weight") != std::string::npos;
    }
    if (weight_name.find(".double_blocks.") == std::string::npos) {
        return false;
    }
    return weight_name.find(".img_mlp.0.weight") != std::string::npos ||
           weight_name.find(".img_mlp.2.weight") != std::string::npos ||
           weight_name.find(".txt_mlp.0.weight") != std::string::npos ||
           weight_name.find(".txt_mlp.2.weight") != std::string::npos ||
           weight_name.find(".img_attn.proj.weight") != std::string::npos ||
           weight_name.find(".txt_attn.proj.weight") != std::string::npos ||
           weight_name.find(".img_attn.qkv.weight") != std::string::npos ||
           weight_name.find(".txt_attn.qkv.weight") != std::string::npos;
}

static ggml_tensor* bonsai_custom_linear_node(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int64_t out_features,
                                              BonsaiGemliteLinearCustomUserdata* userdata) {
    ggml_tensor* args[] = {x};
    return ggml_custom_4d(ctx,
                          GGML_TYPE_F16,
                          out_features,
                          x->ne[1],
                          x->ne[2],
                          x->ne[3],
                          args,
                          1,
                          sd_bonsai_gemlite_int1_ggml_custom_forward,
                          1,
                          userdata);
}

static ggml_tensor* bonsai_custom_qkv_linear_node(ggml_context* ctx,
                                                  ggml_tensor* x,
                                                  int64_t out_features_each,
                                                  BonsaiGemliteQkvCustomUserdata* qkv_userdata) {
    ggml_tensor* args[] = {x};
    return ggml_custom_4d(ctx,
                          GGML_TYPE_F16,
                          3 * out_features_each,
                          x->ne[1],
                          x->ne[2],
                          x->ne[3],
                          args,
                          1,
                          sd_bonsai_gemlite_int1_ggml_custom_qkv_forward,
                          1,
                          qkv_userdata);
}

static bool bonsai_dump_tensors_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_DUMP_TENSORS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

ggml_tensor* bonsai_gemlite_debug_tensor(ggml_context* ctx, ggml_tensor* x, const char* label) {
    if (ctx == nullptr || x == nullptr || label == nullptr || !bonsai_dump_tensors_enabled()) {
        return x;
    }
    ggml_tensor* args[] = {x};
    return ggml_custom_4d(ctx,
                          x->type,
                          x->ne[0],
                          x->ne[1],
                          x->ne[2],
                          x->ne[3],
                          args,
                          1,
                          sd_bonsai_gemlite_int1_ggml_debug_identity_forward,
                          1,
                          const_cast<char*>(label));
}

BonsaiGemliteRuntime::BonsaiGemliteRuntime()
    : impl_(new Impl()) {
}

BonsaiGemliteRuntime::~BonsaiGemliteRuntime() = default;

std::shared_ptr<BonsaiGemliteRuntime> BonsaiGemliteRuntime::load_from_pack(const std::string& pack_path,
                                                                           std::string& error) {
    std::string data_dir;
    std::map<std::string, std::string> index;
    if (!index_bonsai_state_dict(pack_path, data_dir, index, error)) {
        return nullptr;
    }
    std::set<std::string> hf_linears;
    for (const auto& kv : index) {
        if (ends_with(kv.first, ".W_q")) {
            hf_linears.insert(kv.first.substr(0, kv.first.size() - 4));
        }
    }
    std::shared_ptr<BonsaiGemliteRuntime> runtime(new BonsaiGemliteRuntime());
    for (const std::string& hf_name : hf_linears) {
        const std::string internal_name = internal_weight_name_for_bonsai_hf_linear(hf_name);
        if (internal_name.empty()) {
            continue;
        }
        BonsaiGemliteLinear linear;
        if (!bonsai_gemlite_int1_load_linear(pack_path, hf_name, linear, error)) {
            return nullptr;
        }
        void* device = nullptr;
        uint64_t device_bytes = 0;
        std::vector<char> cuda_error(512, 0);
        if (!sd_bonsai_gemlite_int1_device_linear_create(linear.wq.data(),
                                                         linear.wq.size(),
                                                         linear.scales.data(),
                                                         linear.zeros.data(),
                                                         static_cast<int>(linear.in_features),
                                                         static_cast<int>(linear.out_features),
                                                         &device,
                                                         &device_bytes,
                                                         cuda_error.data(),
                                                         cuda_error.size())) {
            error = cuda_error.data();
            return nullptr;
        }
        Impl::DeviceLinear entry;
        entry.hf_name = hf_name;
        entry.internal_weight_name = internal_name;
        entry.out_features = linear.out_features;
        entry.in_features = linear.in_features;
        entry.packed_bytes = static_cast<uint64_t>(linear.wq.size());
        entry.scale_bytes = static_cast<uint64_t>(linear.scales.size() * sizeof(float));
        entry.zero_bytes = static_cast<uint64_t>(linear.zeros.size() * sizeof(float));
        entry.device_bytes = device_bytes;
        entry.device = device;
        runtime->impl_->summary.packed_weight_bytes += entry.packed_bytes;
        runtime->impl_->summary.scale_bytes += entry.scale_bytes;
        runtime->impl_->summary.zero_bytes += entry.zero_bytes;
        runtime->impl_->summary.device_bytes += entry.device_bytes;
        runtime->impl_->by_internal_name.emplace(internal_name, std::move(entry));
    }
    runtime->impl_->summary.quantized_linears = static_cast<int>(hf_linears.size());
    std::cout << "[BonsaiGemLiteINT1] quantized_linears=" << runtime->impl_->summary.quantized_linears
              << " mapped_runtime_linears=" << runtime->impl_->by_internal_name.size()
              << " packed_weight_mb=" << (double)runtime->impl_->summary.packed_weight_bytes / (1024.0 * 1024.0)
              << " scale_mb=" << (double)runtime->impl_->summary.scale_bytes / (1024.0 * 1024.0)
              << " zero_mb=" << (double)runtime->impl_->summary.zero_bytes / (1024.0 * 1024.0)
              << " device_mb=" << (double)runtime->impl_->summary.device_bytes / (1024.0 * 1024.0)
              << " full_fp16_weight_expansion=false\n";
    return runtime;
}

bool BonsaiGemliteRuntime::has_linear(const std::string& internal_weight_name) const {
    if (!impl_) {
        return false;
    }
    if (impl_->by_internal_name.find(internal_weight_name) != impl_->by_internal_name.end()) {
        return true;
    }
    if (ends_with(internal_weight_name, ".qkv.weight")) {
        return impl_->by_internal_name.find(internal_weight_name + ".q") != impl_->by_internal_name.end() &&
               impl_->by_internal_name.find(internal_weight_name + ".k") != impl_->by_internal_name.end() &&
               impl_->by_internal_name.find(internal_weight_name + ".v") != impl_->by_internal_name.end();
    }
    return false;
}

bool BonsaiGemliteRuntime::linear_shape(const std::string& internal_weight_name, int64_t& in_features, int64_t& out_features) const {
    if (!impl_) {
        return false;
    }
    auto it = impl_->by_internal_name.find(internal_weight_name);
    if (it != impl_->by_internal_name.end()) {
        in_features = it->second.in_features;
        out_features = it->second.out_features;
        return true;
    }
    if (ends_with(internal_weight_name, ".qkv.weight")) {
        auto q_it = impl_->by_internal_name.find(internal_weight_name + ".q");
        auto k_it = impl_->by_internal_name.find(internal_weight_name + ".k");
        auto v_it = impl_->by_internal_name.find(internal_weight_name + ".v");
        if (q_it != impl_->by_internal_name.end() &&
            k_it != impl_->by_internal_name.end() &&
            v_it != impl_->by_internal_name.end() &&
            q_it->second.in_features == k_it->second.in_features &&
            q_it->second.in_features == v_it->second.in_features) {
            in_features = q_it->second.in_features;
            out_features = q_it->second.out_features + k_it->second.out_features + v_it->second.out_features;
            return true;
        }
    }
    return false;
}

ggml_tensor* BonsaiGemliteRuntime::linear_forward(ggml_context* ctx, ggml_tensor* x, const char* internal_weight_name) {
    if (ctx == nullptr || x == nullptr || internal_weight_name == nullptr) {
        return nullptr;
    }
    ggml_tensor* linear_input = x;
    const bool cast_output_to_f32 = x->type == GGML_TYPE_F32;
    if (x->type == GGML_TYPE_F32) {
        linear_input = ggml_cast(ctx, x, GGML_TYPE_F16);
    }
    auto it = impl_->by_internal_name.find(internal_weight_name);
    if (it == impl_->by_internal_name.end()) {
        const std::string requested = internal_weight_name;
        if (ends_with(requested, ".qkv.weight")) {
            auto q_it = impl_->by_internal_name.find(requested + ".q");
            auto k_it = impl_->by_internal_name.find(requested + ".k");
            auto v_it = impl_->by_internal_name.find(requested + ".v");
            if (q_it != impl_->by_internal_name.end() &&
                k_it != impl_->by_internal_name.end() &&
                v_it != impl_->by_internal_name.end() &&
                linear_input->type == GGML_TYPE_F16 &&
                linear_input->ne[0] == q_it->second.in_features &&
                linear_input->ne[0] == k_it->second.in_features &&
                linear_input->ne[0] == v_it->second.in_features) {
                q_it->second.calls++;
                k_it->second.calls++;
                v_it->second.calls++;
                impl_->summary.linear_calls += 3;
                int unique = 0;
                for (const auto& kv : impl_->by_internal_name) {
                    if (kv.second.calls > 0) {
                        unique++;
                    }
                }
                impl_->summary.unique_linears_executed = unique;
                const int64_t m = ggml_nelements(linear_input) / linear_input->ne[0];
                std::cout << "[BonsaiGemLiteINT1] qkv_linear_call name=" << internal_weight_name
                          << " M=" << m
                          << " K=" << q_it->second.in_features
                          << " N_each=" << q_it->second.out_features
                          << " kernel=native_int1x3_concat"
                          << " no_fp16_weight_expansion=true\n";
                auto& qkv_ptr = impl_->qkv_userdata[requested];
                if (!qkv_ptr) {
                    qkv_ptr.reset(new BonsaiGemliteQkvCustomUserdata{q_it->second.device,
                                                                      k_it->second.device,
                                                                      v_it->second.device,
                                                                      q_it->second.internal_weight_name.c_str()});
                }
                ggml_tensor* out = bonsai_custom_qkv_linear_node(ctx, linear_input, q_it->second.out_features, qkv_ptr.get());
                return cast_output_to_f32 ? ggml_cast(ctx, out, GGML_TYPE_F32) : out;
            }
        }
    }
    if (it == impl_->by_internal_name.end()) {
        if (!bonsai_replaces_internal_weight_name(internal_weight_name)) {
            return nullptr;
        }
        impl_->summary.missing_linear_calls++;
        if (impl_->summary.first_missing_linear.empty()) {
            impl_->summary.first_missing_linear = internal_weight_name;
        }
        return nullptr;
    }
    Impl::DeviceLinear& linear = it->second;
    if (linear_input->type != GGML_TYPE_F16 || linear_input->ne[0] != linear.in_features) {
        return nullptr;
    }
    linear.calls++;
    impl_->summary.linear_calls++;
    int unique = 0;
    for (const auto& kv : impl_->by_internal_name) {
        if (kv.second.calls > 0) {
            unique++;
        }
    }
    impl_->summary.unique_linears_executed = unique;
    std::cout << "[BonsaiGemLiteINT1] linear_call name=" << internal_weight_name
              << " hf=" << linear.hf_name
              << " M=" << (ggml_nelements(linear_input) / linear_input->ne[0])
              << " K=" << linear.in_features
              << " N=" << linear.out_features
              << " kernel=native_int1"
              << " no_fp16_weight_expansion=true\n";
    auto& userdata = impl_->linear_userdata[internal_weight_name];
    if (!userdata) {
        userdata.reset(new BonsaiGemliteLinearCustomUserdata{linear.device, linear.internal_weight_name.c_str()});
    }
    ggml_tensor* out = bonsai_custom_linear_node(ctx, linear_input, linear.out_features, userdata.get());
    return cast_output_to_f32 ? ggml_cast(ctx, out, GGML_TYPE_F32) : out;
}

BonsaiGemliteRuntimeSummary BonsaiGemliteRuntime::summary() const {
    return impl_->summary;
}

ggml_tensor* bonsai_gemlite_int1_linear_forward_callback(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         void* runtime,
                                                         const char* internal_weight_name) {
    BonsaiGemliteRuntime* rt = reinterpret_cast<BonsaiGemliteRuntime*>(runtime);
    if (rt == nullptr) {
        return nullptr;
    }
    return rt->linear_forward(ctx, x, internal_weight_name);
}

bool bonsai_gemlite_int1_load_linear(const std::string& pack_path,
                                     const std::string& linear_name,
                                     BonsaiGemliteLinear& out,
                                     std::string& error) {
    std::string data_dir;
    std::map<std::string, std::string> index;
    if (!index_bonsai_state_dict(pack_path, data_dir, index, error)) {
        return false;
    }
    const std::string wq_name       = linear_name + ".W_q";
    const std::string scales_name   = linear_name + ".scales";
    const std::string zeros_name    = linear_name + ".zeros";
    const std::string metadata_name = linear_name + ".metadata";
    const std::string shape_name    = linear_name + ".orig_shape";

    std::string wq_entry;
    std::string scales_entry;
    std::string zeros_entry;
    std::string metadata_entry;
    std::string shape_entry;
    if (!tensor_entry(index, data_dir, wq_name, wq_entry, error) ||
        !tensor_entry(index, data_dir, scales_name, scales_entry, error) ||
        !tensor_entry(index, data_dir, metadata_name, metadata_entry, error)) {
        return false;
    }
    if (!read_zip_entry_by_name(pack_path, wq_entry, out.wq, error)) {
        return false;
    }
    if (!read_f32_tensor(pack_path, scales_entry, out.scales, error)) {
        return false;
    }
    if (tensor_entry(index, data_dir, zeros_name, zeros_entry, error, false)) {
        if (!read_f32_tensor(pack_path, zeros_entry, out.zeros, error)) {
            return false;
        }
    } else {
        out.zeros.resize(out.scales.size());
        for (size_t i = 0; i < out.scales.size(); ++i) {
            out.zeros[i] = -0.5f * out.scales[i];
        }
    }
    std::vector<int32_t> metadata;
    if (!read_i32_tensor(pack_path, metadata_entry, metadata, error)) {
        return false;
    }
    std::vector<int32_t> orig_shape;
    if (tensor_entry(index, data_dir, shape_name, shape_entry, error, false)) {
        if (!read_i32_tensor(pack_path, shape_entry, orig_shape, error)) {
            return false;
        }
    }
    if (metadata.size() < 12 || metadata[1] != 1 || metadata[2] != 128 || metadata[4] != 8 || metadata[10] != 4) {
        error = "unsupported GemLite metadata for " + linear_name;
        return false;
    }

    out.name         = linear_name;
    out.out_features = orig_shape.size() == 2 ? orig_shape[0] : 0;
    out.in_features  = orig_shape.size() == 2 ? orig_shape[1] : 0;
    if (out.out_features == 0 || out.in_features == 0) {
        const int64_t scale_count = static_cast<int64_t>(out.scales.size());
        const int64_t wq_count    = static_cast<int64_t>(out.wq.size());
        // K = groups * 128. wq_count = (K / 8) * N and scale_count = groups * N.
        // Therefore wq_count / scale_count must be 16 for Bonsai g128 int1.
        if (scale_count <= 0 || wq_count != scale_count * 16) {
            error = "cannot infer Bonsai GemLite shape for " + linear_name;
            return false;
        }
        // Shape inference without orig_shape is ambiguous in general. The spike
        // requires orig_shape for all real integration paths.
        error = "missing orig_shape for " + linear_name;
        return false;
    }
    out.packed_rows = out.in_features / 8;
    out.packed_cols = out.out_features;
    out.scale_rows  = out.in_features / 128;
    out.scale_cols  = out.out_features;
    out.groups     = out.in_features / 128;
    if (out.packed_rows != out.in_features / 8 || out.packed_cols != out.out_features ||
        out.scale_rows != out.groups || out.scale_cols != out.out_features ||
        out.zeros.size() != out.scales.size()) {
        error = "GemLite shape mismatch for " + linear_name;
        return false;
    }
    return true;
}

bool bonsai_gemlite_int1_summarize_pack(const std::string& pack_path,
                                        BonsaiGemlitePackSummary& out,
                                        std::string& error) {
    std::string data_dir;
    std::map<std::string, std::string> index;
    if (!index_bonsai_state_dict(pack_path, data_dir, index, error)) {
        return false;
    }
    std::set<std::string> linears;
    for (const auto& kv : index) {
        const std::string& name = kv.first;
        std::vector<uint8_t> bytes;
        const std::string entry = data_dir + "data/" + kv.second;
        if (ends_with(name, ".W_q")) {
            linears.insert(name.substr(0, name.size() - 4));
            if (read_zip_entry_by_name(pack_path, entry, bytes, error)) {
                out.packed_weight_bytes += bytes.size();
            }
        } else if (ends_with(name, ".scales")) {
            if (read_zip_entry_by_name(pack_path, entry, bytes, error)) {
                out.scale_bytes += bytes.size();
            }
        } else if (ends_with(name, ".zeros")) {
            if (read_zip_entry_by_name(pack_path, entry, bytes, error)) {
                out.zero_bytes += bytes.size();
            }
        } else if (ends_with(name, ".weight")) {
            out.skipped_bf16_weights++;
        }
    }
    out.quantized_linears = static_cast<int>(linears.size());
    return true;
}

void bonsai_gemlite_int1_print_summary(std::ostream& os,
                                       const BonsaiGemlitePackSummary& summary) {
    os << "quantized_linears=" << summary.quantized_linears << "\n"
       << "skipped_bf16_weights=" << summary.skipped_bf16_weights << "\n"
       << "packed_weight_mb=" << (double)summary.packed_weight_bytes / (1024.0 * 1024.0) << "\n"
       << "scale_mb=" << (double)summary.scale_bytes / (1024.0 * 1024.0) << "\n"
       << "zero_mb=" << (double)summary.zero_bytes / (1024.0 * 1024.0) << "\n";
}

#ifndef SD_USE_CUDA
BonsaiGemliteCudaProbeResult bonsai_gemlite_int1_cuda_probe(const BonsaiGemliteLinear&, int64_t) {
    BonsaiGemliteCudaProbeResult result;
    result.error = "stable-diffusion.cpp was built without CUDA";
    return result;
}
#else
extern "C" bool sd_bonsai_gemlite_int1_cuda_probe_run(const uint8_t* wq,
                                                       size_t wq_bytes,
                                                       const float* scales,
                                                       const float* zeros,
                                                       int m,
                                                       int k,
                                                       int n,
                                                       float* elapsed_ms,
                                                       float* output_min,
                                                       float* output_max,
                                                       double* output_sum,
                                                       uint64_t* device_bytes,
                                                       char* error,
                                                       size_t error_size);

BonsaiGemliteCudaProbeResult bonsai_gemlite_int1_cuda_probe(const BonsaiGemliteLinear& linear,
                                                            int64_t rows_m) {
    BonsaiGemliteCudaProbeResult result;
    std::vector<char> error(512, 0);
    result.ok = sd_bonsai_gemlite_int1_cuda_probe_run(linear.wq.data(),
                                                      linear.wq.size(),
                                                      linear.scales.data(),
                                                      linear.zeros.data(),
                                                      static_cast<int>(rows_m),
                                                      static_cast<int>(linear.in_features),
                                                      static_cast<int>(linear.out_features),
                                                      &result.elapsed_ms,
                                                      &result.output_min,
                                                      &result.output_max,
                                                      &result.output_sum,
                                                      &result.device_bytes,
                                                      error.data(),
                                                      error.size());
    if (!result.ok) {
        result.error = error.data();
    }
    return result;
}
#endif

}  // namespace sd
