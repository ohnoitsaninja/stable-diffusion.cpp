#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

using sd_lens_runtime_progress_cb_t = void (*)(int step, int steps, float time, void* data);

struct sd_lens_transformer_runtime_options {
    std::string transformer_dir;
    std::string latent_npy;
    std::string packed_tokens_npy;
    std::string attention_mode = "regular-f32";
    int width = 256;
    int height = 256;
    int steps = 4;
    int seed = 42;
    int repeat_generations = 1;
    bool use_transformer_context = true;
    std::string transformer_residency = "streaming";
    std::string dynamic_residency = "none";
    int window_blocks = 0;
    int persistent_blocks = 0;
    uint64_t persistent_blocks_memory_mib = 4096;
    bool keep_transformer_warm = false;
    bool output_packed_vae_latent = false;
    sd_lens_runtime_progress_cb_t progress_callback = nullptr;
    void* progress_callback_data = nullptr;
};

struct sd_lens_transformer_runtime_result {
    Tensor latent;
    Tensor packed_vae_latent;
    double wall_seconds = 0.0;
    double context_load_seconds = 0.0;
    double resident_upload_seconds = 0.0;
    double loop_seconds = 0.0;
    double total_generation_seconds = 0.0;
    double runner_setup_seconds = 0.0;
    double runner_alloc_compute_buffer_seconds = 0.0;
    double runner_graph_build_seconds = 0.0;
    double runner_graph_alloc_seconds = 0.0;
    double runner_input_copy_seconds = 0.0;
    double runner_compute_seconds = 0.0;
    double runner_sync_seconds = 0.0;
    double runner_output_copy_seconds = 0.0;
    double runner_cleanup_seconds = 0.0;
    double scheduler_flow_seconds = 0.0;
    double unpack_seconds = 0.0;
    uint64_t runner_input_copy_bytes = 0;
    uint64_t runner_output_copy_bytes = 0;
    uint64_t streamed_bytes = 0;
    uint64_t disk_read_bytes = 0;
    uint64_t resident_weight_bytes = 0;
    uint64_t resident_static_bytes = 0;
    uint64_t runner_count = 0;
    int return_code = 0;
    std::string error;
};

struct sd_lens_transformer_runtime_profile {
    double context_load_seconds = 0.0;
    double resident_upload_seconds = 0.0;
    double loop_seconds = 0.0;
    double total_generation_seconds = 0.0;
    double runner_setup_seconds = 0.0;
    double runner_alloc_compute_buffer_seconds = 0.0;
    double runner_graph_build_seconds = 0.0;
    double runner_graph_alloc_seconds = 0.0;
    double runner_input_copy_seconds = 0.0;
    double runner_compute_seconds = 0.0;
    double runner_sync_seconds = 0.0;
    double runner_output_copy_seconds = 0.0;
    double runner_cleanup_seconds = 0.0;
    double scheduler_flow_seconds = 0.0;
    double unpack_seconds = 0.0;
    uint64_t runner_input_copy_bytes = 0;
    uint64_t runner_output_copy_bytes = 0;
    uint64_t streamed_bytes = 0;
    uint64_t disk_read_bytes = 0;
    uint64_t resident_weight_bytes = 0;
    uint64_t resident_static_bytes = 0;
    uint64_t runner_count = 0;
};

bool sd_lens_transformer_runtime_run_native_cuda(
    const sd_lens_transformer_runtime_options& options,
    const std::unordered_map<std::string, Tensor>& condition_tensors,
    sd_lens_transformer_runtime_result* result);

void sd_lens_transformer_runtime_clear_warm_cache();
