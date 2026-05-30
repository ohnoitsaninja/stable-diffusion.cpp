#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct sd_lens_tokenized_prompt {
    std::string prompt;
    std::string rendered_prompt;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    int max_sequence_length = 128;
    int txt_offset = 97;
    int raw_seq_len = 0;
    int trimmed_seq_len = 0;
    int pad_token_id = 199999;
};

struct sd_lens_gptoss_tokenizer {
    bool load(const std::string& tokenizer_dir, std::string* error = nullptr);
    bool encode_prompt(const std::string& prompt,
                       int max_sequence_length,
                       const std::string& current_date,
                       sd_lens_tokenized_prompt* out,
                       std::string* error = nullptr) const;

private:
    struct Impl;
    Impl* impl = nullptr;

public:
    sd_lens_gptoss_tokenizer();
    ~sd_lens_gptoss_tokenizer();
    sd_lens_gptoss_tokenizer(const sd_lens_gptoss_tokenizer&) = delete;
    sd_lens_gptoss_tokenizer& operator=(const sd_lens_gptoss_tokenizer&) = delete;
};

std::string sd_lens_current_date_yyyy_mm_dd();
