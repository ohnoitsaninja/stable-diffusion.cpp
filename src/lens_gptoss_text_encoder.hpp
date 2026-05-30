#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct sd_lens_text_encoder;

struct sd_lens_cond_v1_native {
    std::array<std::vector<float>, 4> features;
    std::array<std::vector<int64_t>, 4> feature_shapes;
    std::vector<float> attention_mask;
    std::vector<int64_t> attention_mask_shape;
    int txt_offset = 97;
    int raw_seq_len = 128;
    int trimmed_seq_len = 31;
    std::string router_mode = "native-tolerant";
    std::string source = "native_gptoss";
    bool bootstrap_tokens = true;
    std::array<int, 4> layer_taps = {5, 11, 17, 23};
    std::string expert_set_mismatch_tokens_per_layer;
};

struct sd_lens_text_encoder_load_options {
    std::string text_encoder_dir;
};

struct sd_lens_text_encoder_encode_options {
    std::string bootstrap_oracle_dir;
    std::string output_safetensors;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::string prompt;
    std::string tokenizer_source = "bootstrap_input_ids_mask_from_oracle";
    bool bootstrap_tokens = true;
    int txt_offset = 97;
    int max_seq_len = 128;
    std::string router_mode = "native-tolerant";
    std::string moe_backend = "cuda-batched-expert-matmul";
    std::string moe_cache = "per-layer-dequant";
    std::string moe_cache_layout = "full-layer-resident";
    std::string cache_upload = "cuda-mxfp4-dequant";
    bool summary_only = true;
    bool no_oracle_compare = true;
    bool suppress_stdout = false;
    sd_lens_cond_v1_native* output_condition = nullptr;
};

struct sd_lens_text_encoder_result {
    double encode_seconds = 0.0;
    std::string error;
};

sd_lens_text_encoder* sd_lens_text_encoder_create();
bool sd_lens_text_encoder_load(sd_lens_text_encoder* encoder,
                               const sd_lens_text_encoder_load_options& options,
                               sd_lens_text_encoder_result* result = nullptr);
bool sd_lens_text_encoder_encode(sd_lens_text_encoder* encoder,
                                 const sd_lens_text_encoder_encode_options& options,
                                 sd_lens_text_encoder_result* result = nullptr);
void sd_lens_text_encoder_free(sd_lens_text_encoder* encoder);
