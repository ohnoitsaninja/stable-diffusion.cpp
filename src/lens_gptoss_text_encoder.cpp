#include "lens_gptoss_text_encoder.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <unordered_map>
#include <vector>

#define SD_LENS_GPTOSS_TEXT_ENCODER_NO_MAIN 1
#include "../examples/lens-text-encoder-smoke/main.cpp"

struct sd_lens_text_encoder {
    std::filesystem::path text_encoder_dir;
    std::vector<SafetensorEntry> entries;
    std::unordered_map<std::string, SafetensorEntry> index;
    bool loaded = false;
};

static void set_error(sd_lens_text_encoder_result* result, const std::string& error) {
    if (result != nullptr) {
        result->error = error;
    }
}

static void copy_internal_condition(const LensCondV1NativeInternal& internal,
                                    sd_lens_cond_v1_native& out) {
    for (size_t i = 0; i < internal.features.size(); ++i) {
        out.feature_shapes[i] = internal.features[i].shape;
        out.features[i] = internal.features[i].data;
    }
    out.attention_mask_shape = internal.attention_mask.shape;
    out.attention_mask = internal.attention_mask.data;
    out.txt_offset = internal.txt_offset;
    out.raw_seq_len = internal.raw_seq_len;
    out.trimmed_seq_len = internal.trimmed_seq_len;
    out.router_mode = internal.router_mode;
    out.source = internal.source;
    out.bootstrap_tokens = internal.bootstrap_tokens;
    out.layer_taps = internal.layer_taps;
    out.expert_set_mismatch_tokens_per_layer = internal.expert_set_mismatch_tokens_per_layer;
}

class sd_lens_null_streambuf : public std::streambuf {
protected:
    int overflow(int ch) override {
        return ch;
    }
};

class sd_lens_scoped_cout_silencer {
public:
    explicit sd_lens_scoped_cout_silencer(bool enabled) {
        if (enabled) {
            old_ = std::cout.rdbuf(&null_);
        }
    }

    ~sd_lens_scoped_cout_silencer() {
        if (old_ != nullptr) {
            std::cout.rdbuf(old_);
        }
    }

    sd_lens_scoped_cout_silencer(const sd_lens_scoped_cout_silencer&) = delete;
    sd_lens_scoped_cout_silencer& operator=(const sd_lens_scoped_cout_silencer&) = delete;

private:
    sd_lens_null_streambuf null_;
    std::streambuf* old_ = nullptr;
};

sd_lens_text_encoder* sd_lens_text_encoder_create() {
    return new sd_lens_text_encoder();
}

bool sd_lens_text_encoder_load(sd_lens_text_encoder* encoder,
                               const sd_lens_text_encoder_load_options& options,
                               sd_lens_text_encoder_result* result) {
    if (result != nullptr) {
        *result = {};
    }
    if (encoder == nullptr) {
        set_error(result, "sd_lens_text_encoder_load received null encoder");
        return false;
    }
    try {
        encoder->text_encoder_dir = options.text_encoder_dir;
        encoder->entries.clear();
        encoder->index = index_safetensors(encoder->text_encoder_dir, &encoder->entries);
        encoder->loaded = true;
        return true;
    } catch (const std::exception& e) {
        encoder->loaded = false;
        set_error(result, e.what());
        return false;
    }
}

bool sd_lens_text_encoder_encode(sd_lens_text_encoder* encoder,
                                 const sd_lens_text_encoder_encode_options& options,
                                 sd_lens_text_encoder_result* result) {
    if (result != nullptr) {
        *result = {};
    }
    if (encoder == nullptr || !encoder->loaded) {
        set_error(result, "sd_lens_text_encoder_encode requires a loaded encoder");
        return false;
    }
    if (options.txt_offset != 97 || options.max_seq_len != 128) {
        set_error(result, "Lens GPT-OSS bootstrap path currently requires txt_offset=97 and max_seq_len=128");
        return false;
    }
    const bool has_prompt_tokens = !options.input_ids.empty() || !options.attention_mask.empty();
    if (has_prompt_tokens) {
        if (options.input_ids.size() != 128 || options.attention_mask.size() != 128) {
            set_error(result, "Lens GPT-OSS prompt token path requires 128 input_ids and 128 attention_mask values");
            return false;
        }
    } else if (options.bootstrap_oracle_dir.empty()) {
        set_error(result, "sd_lens_text_encoder_encode requires bootstrap_oracle_dir or prompt token input");
        return false;
    }
    if (options.output_safetensors.empty() && options.output_condition == nullptr) {
        set_error(result, "sd_lens_text_encoder_encode requires output_safetensors or output_condition");
        return false;
    }
    try {
        g_moe_backend = options.moe_backend;
        g_moe_cache = options.moe_cache;
        g_moe_cache_layout = options.moe_cache_layout;
        g_cache_upload = options.cache_upload;
        const auto start = std::chrono::steady_clock::now();
        sd_lens_scoped_cout_silencer quiet(options.suppress_stdout);
        LensCondV1NativeInternal internal_condition;
        NpyI64 input_ids_override;
        NpyI64 attention_mask_trimmed_override;
        const NpyI64* input_ids_ptr = nullptr;
        const NpyI64* trimmed_mask_ptr = nullptr;
        if (has_prompt_tokens) {
            input_ids_override.shape = {1, 128};
            input_ids_override.data = options.input_ids;
            attention_mask_trimmed_override.shape = {1, 31};
            attention_mask_trimmed_override.data.assign(options.attention_mask.begin() + options.txt_offset,
                                                        options.attention_mask.end());
            input_ids_ptr = &input_ids_override;
            trimmed_mask_ptr = &attention_mask_trimmed_override;
        }
        run_layers_0_to_n(std::filesystem::path(options.bootstrap_oracle_dir),
                          encoder->text_encoder_dir,
                          encoder->index,
                          23,
                          options.summary_only,
                          options.router_mode,
                          std::filesystem::path(options.output_safetensors),
                          options.no_oracle_compare,
                          options.output_condition != nullptr ? &internal_condition : nullptr,
                          input_ids_ptr,
                          trimmed_mask_ptr,
                          options.prompt.empty() ? "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" : options.prompt,
                          options.tokenizer_source,
                          options.bootstrap_tokens);
        if (options.output_condition != nullptr) {
            copy_internal_condition(internal_condition, *options.output_condition);
        }
        if (result != nullptr) {
            result->encode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        return true;
    } catch (const std::exception& e) {
        set_error(result, e.what());
        return false;
    }
}

void sd_lens_text_encoder_free(sd_lens_text_encoder* encoder) {
    if (encoder == nullptr) {
        return;
    }
    encoder->index.clear();
    encoder->entries.clear();
    encoder->loaded = false;
    release_lens_text_encoder_runtime_state();
    delete encoder;
}
