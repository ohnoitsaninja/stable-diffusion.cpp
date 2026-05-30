#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace sd {

struct BonsaiGemliteLinear {
    std::string name;
    int64_t out_features = 0;
    int64_t in_features  = 0;
    int64_t groups       = 0;
    int64_t packed_rows  = 0;
    int64_t packed_cols  = 0;
    int64_t scale_rows   = 0;
    int64_t scale_cols   = 0;
    std::vector<uint8_t> wq;
    std::vector<float> scales;
    std::vector<float> zeros;
};

struct BonsaiGemlitePackSummary {
    int quantized_linears = 0;
    int skipped_bf16_weights = 0;
    uint64_t packed_weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t zero_bytes = 0;
};

struct BonsaiGemliteCudaProbeResult {
    bool ok = false;
    std::string error;
    float elapsed_ms = 0.0f;
    float output_min = 0.0f;
    float output_max = 0.0f;
    double output_sum = 0.0;
    uint64_t device_bytes = 0;
};

struct BonsaiGemliteRuntimeSummary {
    int quantized_linears = 0;
    uint64_t packed_weight_bytes = 0;
    uint64_t scale_bytes = 0;
    uint64_t zero_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t linear_calls = 0;
    int unique_linears_executed = 0;
    int missing_linear_calls = 0;
    std::string first_missing_linear;
    bool cute_cache_built = false;
    int cute_blocks = 0;
    uint64_t cute_linear1_calls = 0;
    uint64_t cute_linear2_calls = 0;
    uint64_t cute_fallback_calls = 0;
    uint64_t cute_prepack_bytes = 0;
    bool full_fp16_weight_expansion = false;
};

struct BonsaiGemliteLinearCustomUserdata {
    void* device_linear = nullptr;
    const char* debug_name = nullptr;
};

struct BonsaiGemliteQkvCustomUserdata {
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    const char* debug_name = nullptr;
};

class BonsaiGemliteRuntime {
public:
    ~BonsaiGemliteRuntime();

    static std::shared_ptr<BonsaiGemliteRuntime> load_from_pack(const std::string& pack_path,
                                                                std::string& error);

    ggml_tensor* linear_forward(ggml_context* ctx, ggml_tensor* x, const char* internal_weight_name);
    BonsaiGemliteRuntimeSummary summary() const;
    bool has_linear(const std::string& internal_weight_name) const;
    bool linear_shape(const std::string& internal_weight_name, int64_t& in_features, int64_t& out_features) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    BonsaiGemliteRuntime();
};

bool bonsai_gemlite_int1_load_linear(const std::string& pack_path,
                                     const std::string& linear_name,
                                     BonsaiGemliteLinear& out,
                                     std::string& error);

bool bonsai_gemlite_int1_summarize_pack(const std::string& pack_path,
                                        BonsaiGemlitePackSummary& out,
                                        std::string& error);

BonsaiGemliteCudaProbeResult bonsai_gemlite_int1_cuda_probe(const BonsaiGemliteLinear& linear,
                                                            int64_t rows_m);

ggml_tensor* bonsai_gemlite_int1_linear_forward_callback(ggml_context* ctx,
                                                         ggml_tensor* x,
                                                         void* runtime,
                                                         const char* internal_weight_name);

ggml_tensor* bonsai_gemlite_debug_tensor(ggml_context* ctx,
                                         ggml_tensor* x,
                                         const char* label);

void bonsai_gemlite_int1_print_summary(std::ostream& os,
                                       const BonsaiGemlitePackSummary& summary);

}  // namespace sd
