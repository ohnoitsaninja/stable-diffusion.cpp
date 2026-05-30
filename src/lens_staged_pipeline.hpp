#pragma once

#include "lens_gptoss_text_encoder.hpp"
#include "lens_gptoss_tokenizer.hpp"
#include "lens_transformer_runtime.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct sd_lens_pipeline_ctx {
    bool text_encoder_loaded = false;
    bool text_encoder_released = false;
    bool transformer_ran = false;
    bool warm_runtime = false;
    bool same_condition_reuse = false;
    bool transformer_loaded = false;
    bool transformer_released = false;
};

struct sd_lens_text_encoder_stage {
    sd_lens_text_encoder* encoder = nullptr;
    bool loaded = false;

    bool load(const std::string& text_encoder_dir, sd_lens_text_encoder_result* result);
    bool encode(const sd_lens_text_encoder_encode_options& options, sd_lens_text_encoder_result* result);
    void free();
};

struct sd_lens_transformer_stage {
    bool ran = false;

    bool run(const sd_lens_transformer_runtime_options& options,
             const std::unordered_map<std::string, Tensor>& condition_tensors,
             sd_lens_transformer_runtime_result* result);
};

struct sd_lens_vae_stage {
    std::string vae_path;
    bool decoded = false;
};

struct sd_lens_staged_pipeline_options {
    std::string text_encoder_dir;
    std::string bootstrap_oracle_dir;
    std::string prompt;
    std::string tokenizer_dir;
    std::string chat_current_date;
    std::string transformer_dir;
    std::string vae_path;
    std::string optional_cond_out;
    std::string latent_npy;
    std::string packed_tokens_npy;
    int width = 256;
    int height = 256;
    int steps = 4;
    int seed = 42;
    int repeat_generations = 1;
    std::string transformer_speed_mode;
    std::string transformer_residency = "streaming";
    std::string dynamic_residency = "none";
    int transformer_window_blocks = 0;
    int transformer_persistent_blocks = 0;
    uint64_t transformer_persistent_blocks_memory_mib = 4096;
};

struct sd_lens_staged_pipeline_result {
    sd_lens_pipeline_ctx lifecycle;
    sd_lens_cond_v1_native condition;
    Tensor latent;
    std::vector<uint8_t> image_rgb;
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    double text_encoder_load_seconds = 0.0;
    double text_encoder_encode_seconds = 0.0;
    double text_encoder_wrapper_seconds = 0.0;
    double transformer_wall_seconds = 0.0;
    double vae_wall_seconds = 0.0;
    double vae_decode_seconds = 0.0;
    std::string error;
};

struct LensPromptRequest {
    std::string text_encoder_dir;
    std::string transformer_dir;
    std::string vae_path;
    std::string bootstrap_oracle_dir;
    std::string prompt;
    std::string tokenizer_dir;
    std::string chat_current_date;
    std::string optional_cond_out;
    std::string latent_npy;
    std::string packed_tokens_npy;
    int width = 256;
    int height = 256;
    int steps = 4;
    float cfg = 1.0f;
    int seed = 42;
    int repeat_generations = 1;
    std::string speed_mode;
    std::string quality_mode = "reference";
    std::string transformer_residency = "streaming";
    std::string dynamic_residency = "none";
    int transformer_window_blocks = 0;
    int transformer_persistent_blocks = 0;
    uint64_t transformer_persistent_blocks_memory_mib = 4096;
};

struct LensPromptResult {
    sd_lens_pipeline_ctx lifecycle;
    sd_lens_cond_v1_native condition;
    Tensor latent;
    std::vector<uint8_t> image_rgb;
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    double text_encoder_load_seconds = 0.0;
    double text_encoder_encode_seconds = 0.0;
    double text_encoder_wrapper_seconds = 0.0;
    double transformer_wall_seconds = 0.0;
    double vae_wall_seconds = 0.0;
    double vae_decode_seconds = 0.0;
    std::string speed_mode;
    std::string quality_mode;
    bool hidden_parity_exact = true;
    bool prompt_tokenizer_used = false;
    bool bootstrap_oracle_used = false;
    bool transformer_upload_reused = false;
    std::vector<std::string> warnings;
    std::string error;
};

struct sd_lens_staged_runtime {
    sd_lens_staged_pipeline_options options;
    sd_lens_pipeline_ctx lifecycle;
    sd_lens_text_encoder_stage text_encoder_stage;
    sd_lens_transformer_stage transformer_stage;
    sd_lens_cond_v1_native condition;
    std::unordered_map<std::string, Tensor> condition_tensors;
    Tensor latent;
    bool condition_ready = false;
    bool transformer_ready = false;
    double text_encoder_load_seconds = 0.0;
    double text_encoder_encode_seconds = 0.0;
    double text_encoder_wrapper_seconds = 0.0;
    double transformer_wall_seconds = 0.0;
    std::string error;

    explicit sd_lens_staged_runtime(sd_lens_staged_pipeline_options runtime_options = {});
    bool configure(const LensPromptRequest& request);
    bool load_text_encoder();
    bool encode_condition();
    bool encode_prompt(LensPromptResult* result);
    void unload_text_encoder();
    bool load_transformer();
    bool prepare_transformer(LensPromptResult* result);
    bool run_transformer_generation();
    bool generate(LensPromptResult* result);
    bool decode_image(const std::string& vae_path, LensPromptResult* result);
    void unload_transformer();
    void destroy();
};

sd_lens_staged_runtime* sd_lens_staged_runtime_create(const LensPromptRequest& request);
void sd_lens_staged_runtime_destroy(sd_lens_staged_runtime* runtime);
bool sd_lens_staged_runtime_encode_prompt(sd_lens_staged_runtime* runtime, LensPromptResult* result);
bool sd_lens_staged_runtime_prepare_transformer(sd_lens_staged_runtime* runtime, LensPromptResult* result);
bool sd_lens_staged_runtime_generate(sd_lens_staged_runtime* runtime, LensPromptResult* result);
bool sd_lens_staged_runtime_decode(sd_lens_staged_runtime* runtime, LensPromptResult* result);

std::unordered_map<std::string, Tensor> sd_lens_make_transformer_condition_map(
    const sd_lens_cond_v1_native& condition);

bool sd_lens_run_staged_text_encoder_and_transformer(
    const sd_lens_staged_pipeline_options& options,
    sd_lens_staged_pipeline_result* result);

bool sd_lens_run_warm_staged_runtime(
    const sd_lens_staged_pipeline_options& options,
    sd_lens_staged_pipeline_result* result);

bool sd_lens_run_prompt_request(
    const LensPromptRequest& request,
    LensPromptResult* result);
