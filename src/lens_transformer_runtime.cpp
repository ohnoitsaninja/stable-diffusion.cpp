#include "lens_transformer_runtime.hpp"

#ifndef SD_LENS_STAGED_PIPELINE_ENABLE_IMPL

bool sd_lens_transformer_runtime_run_native_cuda(
    const sd_lens_transformer_runtime_options&,
    const std::unordered_map<std::string, Tensor>&,
    sd_lens_transformer_runtime_result* result) {
    if (result != nullptr) {
        *result = {};
        result->return_code = 2;
        result->error = "Lens transformer runtime implementation is not linked in this build target";
    }
    return false;
}

void sd_lens_transformer_runtime_clear_warm_cache() {}

#else

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int sd_lens_transformer_smoke_main_impl_profiled(
    int argc,
    char** argv,
    const std::unordered_map<std::string, Tensor>* in_memory_cond_tensors,
    Tensor* out_native_cuda_latent,
    sd_lens_transformer_runtime_profile* out_profile,
    sd_lens_runtime_progress_cb_t progress_callback,
    void* progress_callback_data);

extern "C" void sd_lens_transformer_smoke_warm_cache_clear();

bool sd_lens_transformer_runtime_run_native_cuda(
    const sd_lens_transformer_runtime_options& options,
    const std::unordered_map<std::string, Tensor>& condition_tensors,
    sd_lens_transformer_runtime_result* result) {
    if (result != nullptr) {
        *result = {};
    }
    if (options.transformer_dir.empty()) {
        if (result != nullptr) {
            result->error = "Lens transformer runtime requires transformer_dir";
            result->return_code = 2;
        }
        return false;
    }
    if (options.width <= 0 || options.height <= 0 || options.steps <= 0) {
        if (result != nullptr) {
            result->error = "Lens transformer runtime requires positive width/height/steps";
            result->return_code = 2;
        }
        return false;
    }

    std::vector<std::string> args = {
        "sd-lens-transformer-runtime",
        "--real-block-transformer", options.transformer_dir,
        "--real-full-transformer",
        "--native-cuda-generate-256",
        "--lens-attention-mode", options.attention_mode,
        "--tiny-flow-steps", std::to_string(options.steps),
        "--seed", std::to_string(options.seed),
        "--repeat-generations", std::to_string(options.repeat_generations),
        "--external-height", std::to_string(options.height),
        "--external-width", std::to_string(options.width),
    };
    if (options.use_transformer_context) {
        args.push_back("--use-transformer-context");
    }
    if (options.keep_transformer_warm) {
        args.push_back("--keep-transformer-warm");
    }
    if (!options.transformer_residency.empty() && options.transformer_residency != "streaming") {
        args.push_back("--transformer-residency");
        args.push_back(options.transformer_residency);
        if (options.window_blocks > 0) {
            args.push_back("--window-blocks");
            args.push_back(std::to_string(options.window_blocks));
        }
        if (options.persistent_blocks > 0) {
            args.push_back("--persistent-blocks");
            args.push_back(std::to_string(options.persistent_blocks));
        }
        if (options.persistent_blocks_memory_mib > 0) {
            args.push_back("--persistent-blocks-memory-mib");
            args.push_back(std::to_string(options.persistent_blocks_memory_mib));
        }
    }
    if (!options.dynamic_residency.empty() && options.dynamic_residency != "none") {
        args.push_back("--dynamic-residency");
        args.push_back(options.dynamic_residency);
    }
    if (options.output_packed_vae_latent) {
        args.push_back("--output-packed-vae-latent");
    }
    if (!options.latent_npy.empty()) {
        args.push_back("--tiny-denoise-npy");
        args.push_back(options.latent_npy);
    }
    if (!options.packed_tokens_npy.empty()) {
        args.push_back("--packed-tokens-npy");
        args.push_back(options.packed_tokens_npy);
    }

    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) {
        argv.push_back(arg.data());
    }

    Tensor latent;
    const auto start = std::chrono::steady_clock::now();
    sd_lens_transformer_runtime_profile profile;
    std::ostringstream captured_stderr;
    std::streambuf* previous_stderr = std::cerr.rdbuf(captured_stderr.rdbuf());
    int rc = 1;
    try {
        rc = sd_lens_transformer_smoke_main_impl_profiled(
            static_cast<int>(argv.size()),
            argv.data(),
            &condition_tensors,
            &latent,
            &profile,
            options.progress_callback,
            options.progress_callback_data);
    } catch (const std::exception& e) {
        captured_stderr << "Lens transformer runtime threw: " << e.what() << "\n";
        rc = 1;
    } catch (...) {
        captured_stderr << "Lens transformer runtime threw an unknown exception\n";
        rc = 1;
    }
    std::cerr.rdbuf(previous_stderr);
    const auto end = std::chrono::steady_clock::now();

    if (result != nullptr) {
        result->wall_seconds = std::chrono::duration<double>(end - start).count();
        result->context_load_seconds = profile.context_load_seconds;
        result->resident_upload_seconds = profile.resident_upload_seconds;
        result->loop_seconds = profile.loop_seconds;
        result->total_generation_seconds = profile.total_generation_seconds;
        result->runner_setup_seconds = profile.runner_setup_seconds;
        result->runner_alloc_compute_buffer_seconds = profile.runner_alloc_compute_buffer_seconds;
        result->runner_graph_build_seconds = profile.runner_graph_build_seconds;
        result->runner_graph_alloc_seconds = profile.runner_graph_alloc_seconds;
        result->runner_input_copy_seconds = profile.runner_input_copy_seconds;
        result->runner_compute_seconds = profile.runner_compute_seconds;
        result->runner_sync_seconds = profile.runner_sync_seconds;
        result->runner_output_copy_seconds = profile.runner_output_copy_seconds;
        result->runner_cleanup_seconds = profile.runner_cleanup_seconds;
        result->scheduler_flow_seconds = profile.scheduler_flow_seconds;
        result->unpack_seconds = profile.unpack_seconds;
        result->runner_input_copy_bytes = profile.runner_input_copy_bytes;
        result->runner_output_copy_bytes = profile.runner_output_copy_bytes;
        result->streamed_bytes = profile.streamed_bytes;
        result->disk_read_bytes = profile.disk_read_bytes;
        result->resident_weight_bytes = profile.resident_weight_bytes;
        result->resident_static_bytes = profile.resident_static_bytes;
        result->runner_count = profile.runner_count;
        result->return_code = rc;
        if (options.output_packed_vae_latent) {
            result->packed_vae_latent = std::move(latent);
        } else {
            result->latent = std::move(latent);
        }
        if (rc != 0) {
            result->error = "Lens transformer runtime returned " + std::to_string(rc);
            const std::string stderr_text = captured_stderr.str();
            if (!stderr_text.empty()) {
                result->error += ": " + stderr_text;
            }
        } else if ((!options.output_packed_vae_latent && result->latent.data.empty()) ||
                   (options.output_packed_vae_latent && result->packed_vae_latent.data.empty())) {
            result->error = options.output_packed_vae_latent
                                ? "Lens transformer runtime did not return an in-memory packed VAE latent"
                                : "Lens transformer runtime did not return an in-memory latent";
            result->return_code = 1;
        }
    }
    return rc == 0 &&
           (result == nullptr ||
            (!options.output_packed_vae_latent && !result->latent.data.empty()) ||
            (options.output_packed_vae_latent && !result->packed_vae_latent.data.empty()));
}

void sd_lens_transformer_runtime_clear_warm_cache() {
    sd_lens_transformer_smoke_warm_cache_clear();
}

#endif
