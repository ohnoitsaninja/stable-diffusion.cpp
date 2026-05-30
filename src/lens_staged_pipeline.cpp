#include "lens_staged_pipeline.hpp"

#include <utility>

#ifndef SD_LENS_STAGED_PIPELINE_ENABLE_IMPL

bool sd_lens_text_encoder_stage::load(const std::string&, sd_lens_text_encoder_result* result) {
    if (result != nullptr) {
        result->error = "Lens staged text_encoder implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_text_encoder_stage::encode(const sd_lens_text_encoder_encode_options&,
                                        sd_lens_text_encoder_result* result) {
    if (result != nullptr) {
        result->error = "Lens staged text_encoder implementation is not linked in this build target";
    }
    return false;
}

void sd_lens_text_encoder_stage::free() {
    encoder = nullptr;
    loaded = false;
}

bool sd_lens_transformer_stage::run(const sd_lens_transformer_runtime_options&,
                                    const std::unordered_map<std::string, Tensor>&,
                                    sd_lens_transformer_runtime_result* result) {
    if (result != nullptr) {
        result->error = "Lens staged transformer implementation is not linked in this build target";
        result->return_code = 2;
    }
    return false;
}

std::unordered_map<std::string, Tensor> sd_lens_make_transformer_condition_map(
    const sd_lens_cond_v1_native&) {
    return {};
}

bool sd_lens_run_staged_text_encoder_and_transformer(
    const sd_lens_staged_pipeline_options&,
    sd_lens_staged_pipeline_result* result) {
    if (result != nullptr) {
        *result = {};
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

sd_lens_staged_runtime::sd_lens_staged_runtime(sd_lens_staged_pipeline_options runtime_options)
    : options(std::move(runtime_options)) {}

bool sd_lens_staged_runtime::configure(const LensPromptRequest&) {
    error = "Lens staged pipeline implementation is not linked in this build target";
    return false;
}

bool sd_lens_staged_runtime::load_text_encoder() {
    error = "Lens staged pipeline implementation is not linked in this build target";
    return false;
}

bool sd_lens_staged_runtime::encode_condition() {
    error = "Lens staged pipeline implementation is not linked in this build target";
    return false;
}

bool sd_lens_staged_runtime::encode_prompt(LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

void sd_lens_staged_runtime::unload_text_encoder() {
    text_encoder_stage.free();
    lifecycle.text_encoder_released = true;
}

bool sd_lens_staged_runtime::load_transformer() {
    error = "Lens staged pipeline implementation is not linked in this build target";
    return false;
}

bool sd_lens_staged_runtime::prepare_transformer(LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_staged_runtime::run_transformer_generation() {
    error = "Lens staged pipeline implementation is not linked in this build target";
    return false;
}

bool sd_lens_staged_runtime::generate(LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_staged_runtime::decode_image(const std::string&, LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

void sd_lens_staged_runtime::unload_transformer() {
    transformer_ready = false;
    lifecycle.transformer_released = true;
}

void sd_lens_staged_runtime::destroy() {
    unload_text_encoder();
    unload_transformer();
    condition_tensors.clear();
    latent = {};
    condition_ready = false;
}

bool sd_lens_run_warm_staged_runtime(
    const sd_lens_staged_pipeline_options&,
    sd_lens_staged_pipeline_result* result) {
    if (result != nullptr) {
        *result = {};
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

sd_lens_staged_runtime* sd_lens_staged_runtime_create(const LensPromptRequest&) {
    return nullptr;
}

void sd_lens_staged_runtime_destroy(sd_lens_staged_runtime*) {}

bool sd_lens_staged_runtime_encode_prompt(sd_lens_staged_runtime*, LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_staged_runtime_prepare_transformer(sd_lens_staged_runtime*, LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_staged_runtime_generate(sd_lens_staged_runtime*, LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_staged_runtime_decode(sd_lens_staged_runtime*, LensPromptResult* result) {
    if (result != nullptr) {
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

bool sd_lens_run_prompt_request(const LensPromptRequest&, LensPromptResult* result) {
    if (result != nullptr) {
        *result = {};
        result->error = "Lens staged pipeline implementation is not linked in this build target";
    }
    return false;
}

#else

#include "stable-diffusion.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

bool sd_lens_text_encoder_stage::load(const std::string& text_encoder_dir, sd_lens_text_encoder_result* result) {
    free();
    encoder = sd_lens_text_encoder_create();
    sd_lens_text_encoder_load_options options;
    options.text_encoder_dir = text_encoder_dir;
    loaded = sd_lens_text_encoder_load(encoder, options, result);
    if (!loaded) {
        free();
    }
    return loaded;
}

bool sd_lens_text_encoder_stage::encode(const sd_lens_text_encoder_encode_options& options,
                                        sd_lens_text_encoder_result* result) {
    if (encoder == nullptr || !loaded) {
        if (result != nullptr) {
            result->error = "Lens text_encoder stage is not loaded";
        }
        return false;
    }
    return sd_lens_text_encoder_encode(encoder, options, result);
}

void sd_lens_text_encoder_stage::free() {
    if (encoder != nullptr) {
        sd_lens_text_encoder_free(encoder);
        encoder = nullptr;
    }
    loaded = false;
}

bool sd_lens_transformer_stage::run(const sd_lens_transformer_runtime_options& options,
                                    const std::unordered_map<std::string, Tensor>& condition_tensors,
                                    sd_lens_transformer_runtime_result* result) {
    ran = sd_lens_transformer_runtime_run_native_cuda(options, condition_tensors, result);
    return ran;
}

std::unordered_map<std::string, Tensor> sd_lens_make_transformer_condition_map(
    const sd_lens_cond_v1_native& condition) {
    std::unordered_map<std::string, Tensor> tensors;
    tensors.reserve(5);
    for (int i = 0; i < 4; ++i) {
        Tensor tensor;
        tensor.shape = condition.feature_shapes[static_cast<size_t>(i)];
        tensor.data = condition.features[static_cast<size_t>(i)];
        if (tensor.shape.size() != 3 ||
            tensor.shape[0] != 1 ||
            tensor.shape[1] != condition.trimmed_seq_len ||
            tensor.shape[2] != 2880) {
            throw std::runtime_error("Lens condition feature_" + std::to_string(i) + " has invalid shape");
        }
        tensors.emplace("feature_" + std::to_string(i), std::move(tensor));
    }

    Tensor mask;
    mask.shape = condition.attention_mask_shape;
    mask.data = condition.attention_mask;
    if (mask.shape.size() != 2 || mask.shape[0] != 1 || mask.shape[1] != condition.trimmed_seq_len) {
        throw std::runtime_error("Lens condition attention_mask has invalid shape");
    }
    tensors.emplace("attention_mask", std::move(mask));
    return tensors;
}

sd_lens_staged_runtime::sd_lens_staged_runtime(sd_lens_staged_pipeline_options runtime_options)
    : options(std::move(runtime_options)) {}

static void sd_lens_staged_log_cb(enum sd_log_level_t level, const char* log, void*) {
    if (log == nullptr) {
        return;
    }
    FILE* stream = level == SD_LOG_ERROR ? stderr : stdout;
    std::fputs(log, stream);
    std::fflush(stream);
}

static sd_lens_staged_pipeline_options sd_lens_options_from_prompt_request(
    const LensPromptRequest& request,
    std::string* error) {
    sd_lens_staged_pipeline_options options;
    options.text_encoder_dir = request.text_encoder_dir;
    options.bootstrap_oracle_dir = request.bootstrap_oracle_dir;
    options.prompt = request.prompt;
    options.tokenizer_dir = request.tokenizer_dir;
    options.chat_current_date = request.chat_current_date;
    options.transformer_dir = request.transformer_dir;
    options.vae_path = request.vae_path;
    options.optional_cond_out = request.optional_cond_out;
    options.latent_npy = request.latent_npy;
    options.packed_tokens_npy = request.packed_tokens_npy;
    options.width = request.width;
    options.height = request.height;
    options.steps = request.steps;
    options.seed = request.seed;
    options.repeat_generations = request.repeat_generations;
    options.transformer_speed_mode = request.speed_mode;
    options.transformer_residency = request.transformer_residency;
    options.dynamic_residency = request.dynamic_residency;
    options.transformer_window_blocks = request.transformer_window_blocks;
    options.transformer_persistent_blocks = request.transformer_persistent_blocks;
    options.transformer_persistent_blocks_memory_mib = request.transformer_persistent_blocks_memory_mib;
    if (std::fabs(request.cfg - 1.0f) > 0.000001f) {
        if (error != nullptr) {
            *error = "Lens staged runtime currently supports cfg=1.0 only";
        }
        return {};
    }
    if (request.speed_mode == "bf16-resident") {
        options.transformer_residency = "gpu-full-bf16";
    } else if (!request.speed_mode.empty()) {
        if (error != nullptr) {
            *error = "Lens staged runtime speed_mode must be bf16-resident when set";
        }
        return {};
    }
    if (error != nullptr) {
        error->clear();
    }
    return options;
}

static void sd_lens_copy_runtime_to_prompt_result(
    const sd_lens_staged_runtime& runtime,
    LensPromptResult* result) {
    if (result == nullptr) {
        return;
    }
    result->lifecycle = runtime.lifecycle;
    result->condition = runtime.condition;
    result->latent = runtime.latent;
    result->text_encoder_load_seconds = runtime.text_encoder_load_seconds;
    result->text_encoder_encode_seconds = runtime.text_encoder_encode_seconds;
    result->text_encoder_wrapper_seconds = runtime.text_encoder_wrapper_seconds;
    result->transformer_wall_seconds = runtime.transformer_wall_seconds;
    result->speed_mode = runtime.options.transformer_speed_mode;
    result->quality_mode = runtime.options.transformer_speed_mode == "bf16-resident" ? "speed" : "reference";
    result->hidden_parity_exact = runtime.options.transformer_speed_mode != "bf16-resident";
    result->prompt_tokenizer_used = !runtime.options.prompt.empty();
    result->bootstrap_oracle_used = !runtime.options.bootstrap_oracle_dir.empty();
}

bool sd_lens_staged_runtime::configure(const LensPromptRequest& request) {
    std::string configure_error;
    sd_lens_staged_pipeline_options configured = sd_lens_options_from_prompt_request(request, &configure_error);
    if (!configure_error.empty()) {
        error = configure_error;
        return false;
    }
    options = std::move(configured);
    lifecycle = {};
    condition = {};
    condition_tensors.clear();
    latent = {};
    condition_ready = false;
    transformer_ready = false;
    text_encoder_load_seconds = 0.0;
    text_encoder_encode_seconds = 0.0;
    text_encoder_wrapper_seconds = 0.0;
    transformer_wall_seconds = 0.0;
    error.clear();
    return true;
}

bool sd_lens_staged_runtime::load_text_encoder() {
    error.clear();
    if (options.text_encoder_dir.empty()) {
        error = "Lens staged runtime requires text_encoder_dir";
        return false;
    }

    sd_lens_text_encoder_result text_result;
    const auto text_load_start = std::chrono::steady_clock::now();
    if (!text_encoder_stage.load(options.text_encoder_dir, &text_result)) {
        error = "sd_lens_text_encoder_load failed: " + text_result.error;
        return false;
    }
    const auto text_load_end = std::chrono::steady_clock::now();
    text_encoder_load_seconds = std::chrono::duration<double>(text_load_end - text_load_start).count();
    lifecycle.text_encoder_loaded = true;
    lifecycle.text_encoder_released = false;
    std::cout << "vram_snapshot stage=after_text_encoder_load source=external-only\n";
    return true;
}

bool sd_lens_staged_runtime::encode_condition() {
    error.clear();
    if (!text_encoder_stage.loaded) {
        error = "Lens staged runtime text_encoder is not loaded";
        return false;
    }
    const bool use_prompt_tokenizer = !options.prompt.empty();
    if (options.bootstrap_oracle_dir.empty() && !use_prompt_tokenizer) {
        error = "Lens staged runtime requires bootstrap_oracle_dir or prompt";
        return false;
    }

    condition = {};
    sd_lens_tokenized_prompt tokenized;
    if (use_prompt_tokenizer) {
        const std::filesystem::path tokenizer_dir =
            options.tokenizer_dir.empty()
                ? (std::filesystem::path(options.text_encoder_dir).parent_path() / "tokenizer")
                : std::filesystem::path(options.tokenizer_dir);
        sd_lens_gptoss_tokenizer tokenizer;
        std::string tokenizer_error;
        if (!tokenizer.load(tokenizer_dir.string(), &tokenizer_error)) {
            error = "Lens GPT-OSS tokenizer load failed: " + tokenizer_error;
            return false;
        }
        if (!tokenizer.encode_prompt(options.prompt,
                                     128,
                                     options.chat_current_date,
                                     &tokenized,
                                     &tokenizer_error)) {
            error = "Lens GPT-OSS prompt tokenization failed: " + tokenizer_error;
            return false;
        }
        std::cout << "Lens staged prompt tokenized:"
                  << " tokenizer=native_gptoss_tokenizer_json"
                  << " prompt_chars=" << options.prompt.size()
                  << " rendered_chars=" << tokenized.rendered_prompt.size()
                  << " raw_seq_len=" << tokenized.raw_seq_len
                  << " trimmed_seq_len=" << tokenized.trimmed_seq_len
                  << " txt_offset=" << tokenized.txt_offset
                  << " chat_current_date="
                  << (options.chat_current_date.empty() ? sd_lens_current_date_yyyy_mm_dd() : options.chat_current_date)
                  << "\n";
    }
    sd_lens_text_encoder_result text_result;
    sd_lens_text_encoder_encode_options encode_options;
    encode_options.bootstrap_oracle_dir = options.bootstrap_oracle_dir;
    encode_options.output_safetensors = options.optional_cond_out;
    encode_options.output_condition = &condition;
    if (use_prompt_tokenizer) {
        encode_options.input_ids = tokenized.input_ids;
        encode_options.attention_mask = tokenized.attention_mask;
        encode_options.prompt = options.prompt;
        encode_options.tokenizer_source = "native_gptoss_tokenizer_json";
        encode_options.bootstrap_tokens = false;
    }

    const auto text_encode_start = std::chrono::steady_clock::now();
    if (!text_encoder_stage.encode(encode_options, &text_result)) {
        error = "sd_lens_text_encoder_encode failed: " + text_result.error;
        return false;
    }
    const auto text_encode_end = std::chrono::steady_clock::now();
    text_encoder_encode_seconds = text_result.encode_seconds;
    text_encoder_wrapper_seconds = std::chrono::duration<double>(text_encode_end - text_encode_start).count();
    condition_tensors = sd_lens_make_transformer_condition_map(condition);
    condition_ready = true;
    std::cout << "vram_snapshot stage=after_text_encoder_encode source=external-only\n";
    return true;
}

bool sd_lens_staged_runtime::encode_prompt(LensPromptResult* result) {
    if (!load_text_encoder()) {
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    if (!encode_condition()) {
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    unload_text_encoder();
    sd_lens_copy_runtime_to_prompt_result(*this, result);
    return true;
}

void sd_lens_staged_runtime::unload_text_encoder() {
    text_encoder_stage.free();
    lifecycle.text_encoder_released = true;
    std::cout << "vram_snapshot stage=after_text_encoder_free source=external-only\n";
}

bool sd_lens_staged_runtime::load_transformer() {
    error.clear();
    if (options.transformer_dir.empty()) {
        error = "Lens staged runtime requires transformer_dir";
        return false;
    }
    if (!condition_ready) {
        error = "Lens staged runtime requires condition before transformer load";
        return false;
    }
    if (text_encoder_stage.loaded) {
        error = "Lens staged runtime refuses transformer load while text_encoder is resident";
        return false;
    }
    transformer_ready = true;
    lifecycle.transformer_loaded = true;
    lifecycle.warm_runtime = options.repeat_generations > 1;
    lifecycle.same_condition_reuse = options.repeat_generations > 1;
    std::cout << "vram_snapshot stage=before_transformer_load source=external-only\n";
    std::cout << "Lens staged warm runtime transformer stage prepared:"
              << " warm_runtime=" << (lifecycle.warm_runtime ? "true" : "false")
              << " same_condition_reuse=" << (lifecycle.same_condition_reuse ? "true" : "false")
              << " speed_mode=" << (options.transformer_speed_mode.empty() ? "<none>" : options.transformer_speed_mode)
              << " residency=" << options.transformer_residency
              << " repeat_generations=" << options.repeat_generations
              << " text_encoder_resident=false\n";
    return true;
}

bool sd_lens_staged_runtime::prepare_transformer(LensPromptResult* result) {
    if (!load_transformer()) {
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    sd_lens_copy_runtime_to_prompt_result(*this, result);
    return true;
}

bool sd_lens_staged_runtime::run_transformer_generation() {
    error.clear();
    if (!transformer_ready) {
        error = "Lens staged runtime transformer is not prepared";
        return false;
    }
    if (!condition_ready) {
        error = "Lens staged runtime condition is not ready";
        return false;
    }

    sd_lens_transformer_runtime_options transformer_options;
    transformer_options.transformer_dir = options.transformer_dir;
    transformer_options.latent_npy = options.latent_npy;
    transformer_options.packed_tokens_npy = options.packed_tokens_npy;
    transformer_options.width = options.width;
    transformer_options.height = options.height;
    transformer_options.steps = options.steps;
    transformer_options.seed = options.seed;
    transformer_options.repeat_generations = options.repeat_generations;
    transformer_options.attention_mode = "regular-f32";
    transformer_options.use_transformer_context = true;
    transformer_options.transformer_residency = options.transformer_residency;
    transformer_options.dynamic_residency = options.dynamic_residency;
    transformer_options.window_blocks = options.transformer_window_blocks;
    transformer_options.persistent_blocks = options.transformer_persistent_blocks;
    transformer_options.persistent_blocks_memory_mib = options.transformer_persistent_blocks_memory_mib;

    sd_lens_transformer_runtime_result transformer_result;
    if (!transformer_stage.run(transformer_options, condition_tensors, &transformer_result)) {
        error = transformer_result.error.empty() ? "Lens transformer stage failed" : transformer_result.error;
        return false;
    }
    lifecycle.transformer_ran = true;
    transformer_wall_seconds = transformer_result.wall_seconds;
    latent = std::move(transformer_result.latent);
    return true;
}

bool sd_lens_staged_runtime::generate(LensPromptResult* result) {
    if (!run_transformer_generation()) {
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    sd_lens_copy_runtime_to_prompt_result(*this, result);
    return true;
}

bool sd_lens_staged_runtime::decode_image(const std::string& vae_path, LensPromptResult* result) {
    error.clear();
    if (latent.shape.size() != 4 || latent.shape[0] != 1 || latent.shape[1] != 32) {
        error = "in-memory Lens latent must have shape 1x32xHxW";
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    sd_set_log_callback(sd_lens_staged_log_cb, nullptr);
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.vae_path = vae_path.c_str();
    params.vae_decode_only = true;
    params.free_params_immediately = true;
    params.keep_vae_on_cpu = false;
#ifdef _WIN32
    _putenv_s("SDCPP_MODEL_FAMILY_HINT", "lens");
#else
    setenv("SDCPP_MODEL_FAMILY_HINT", "lens", 1);
#endif
    std::cout << "vram_snapshot stage=after_transformer_free source=external-only\n";
    std::cout << "vram_snapshot stage=before_vae_load source=external-only\n";
    const auto vae_wall_start = std::chrono::steady_clock::now();
    sd_ctx_t* ctx = new_sd_ctx(&params);
    if (ctx == nullptr) {
        error = "failed to create Lens VAE decode context";
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    std::cout << "vram_snapshot stage=after_vae_load source=external-only\n";

    const int64_t h = latent.shape[2];
    const int64_t w = latent.shape[3];
    sd_latent_t* sd_latent = sd_latent_import_f32(latent.data.data(),
                                                 static_cast<uint64_t>(latent.data.size()),
                                                 static_cast<uint32_t>(w),
                                                 static_cast<uint32_t>(h),
                                                 32);
    if (sd_latent == nullptr) {
        error = "sd_latent_import_f32 failed";
        free_sd_ctx(ctx);
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }

    sd_vae_memory_report_t report;
    sd_vae_memory_report_init(&report);
    const auto decode_start = std::chrono::steady_clock::now();
    sd_image_t* image = sd_decode_latent_normal(ctx, sd_latent, nullptr, &report);
    const auto decode_end = std::chrono::steady_clock::now();
    free_sd_latent(sd_latent);
    if (image == nullptr) {
        error = "Lens VAE decode failed";
        free_sd_ctx(ctx);
        if (result != nullptr) {
            result->error = error;
        }
        return false;
    }
    std::cout << "vram_snapshot stage=after_vae_decode source=external-only\n";

    const auto vae_wall_end = std::chrono::steady_clock::now();
    const double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
    const double vae_wall_seconds = std::chrono::duration<double>(vae_wall_end - vae_wall_start).count();
    if (result != nullptr) {
        sd_lens_copy_runtime_to_prompt_result(*this, result);
        result->vae_decode_seconds = decode_seconds;
        result->vae_wall_seconds = vae_wall_seconds;
        result->image_width = image->width;
        result->image_height = image->height;
        result->image_channels = image->channel;
        const size_t image_bytes =
            static_cast<size_t>(image->width) *
            static_cast<size_t>(image->height) *
            static_cast<size_t>(image->channel);
        result->image_rgb.assign(image->data, image->data + image_bytes);
    }
    std::cout << "Lens staged native VAE decoded"
              << " image=" << image->width << "x" << image->height << "x" << image->channel
              << " latent=1x32x" << h << "x" << w
              << " vae_wall_seconds=" << vae_wall_seconds
              << " decode_seconds=" << decode_seconds
              << " decode_graph_ms=" << report.decode_graph_ms << "\n";
    free_sd_image(image);
    free_sd_ctx(ctx);
    std::cout << "vram_snapshot stage=final_cleanup source=external-only\n";
    return true;
}

void sd_lens_staged_runtime::unload_transformer() {
    transformer_ready = false;
    lifecycle.transformer_released = true;
    std::cout << "vram_snapshot stage=after_transformer_runtime_release source=external-only\n";
}

void sd_lens_staged_runtime::destroy() {
    unload_text_encoder();
    unload_transformer();
    condition_tensors.clear();
    latent = {};
    condition_ready = false;
}

sd_lens_staged_runtime* sd_lens_staged_runtime_create(const LensPromptRequest& request) {
    auto* runtime = new sd_lens_staged_runtime();
    if (!runtime->configure(request)) {
        delete runtime;
        return nullptr;
    }
    return runtime;
}

void sd_lens_staged_runtime_destroy(sd_lens_staged_runtime* runtime) {
    if (runtime != nullptr) {
        runtime->destroy();
        delete runtime;
    }
}

bool sd_lens_staged_runtime_encode_prompt(sd_lens_staged_runtime* runtime, LensPromptResult* result) {
    if (runtime == nullptr) {
        if (result != nullptr) {
            result->error = "Lens staged runtime is null";
        }
        return false;
    }
    return runtime->encode_prompt(result);
}

bool sd_lens_staged_runtime_prepare_transformer(sd_lens_staged_runtime* runtime, LensPromptResult* result) {
    if (runtime == nullptr) {
        if (result != nullptr) {
            result->error = "Lens staged runtime is null";
        }
        return false;
    }
    return runtime->prepare_transformer(result);
}

bool sd_lens_staged_runtime_generate(sd_lens_staged_runtime* runtime, LensPromptResult* result) {
    if (runtime == nullptr) {
        if (result != nullptr) {
            result->error = "Lens staged runtime is null";
        }
        return false;
    }
    return runtime->generate(result);
}

bool sd_lens_staged_runtime_decode(sd_lens_staged_runtime* runtime, LensPromptResult* result) {
    if (runtime == nullptr) {
        if (result != nullptr) {
            result->error = "Lens staged runtime is null";
        }
        return false;
    }
    return runtime->decode_image(runtime->options.vae_path, result);
}

bool sd_lens_run_staged_text_encoder_and_transformer(
    const sd_lens_staged_pipeline_options& options,
    sd_lens_staged_pipeline_result* result) {
    if (result != nullptr) {
        *result = {};
    }
    auto fail = [&](const std::string& message) {
        if (result != nullptr) {
            result->error = message;
        }
        return false;
    };
    if (options.text_encoder_dir.empty() ||
        options.transformer_dir.empty()) {
        return fail("Lens staged pipeline requires text_encoder_dir and transformer_dir");
    }
    if (options.bootstrap_oracle_dir.empty() && options.prompt.empty()) {
        return fail("Lens staged pipeline requires bootstrap_oracle_dir or prompt");
    }

    sd_lens_text_encoder_stage text_stage;
    sd_lens_text_encoder_result text_result;
    const auto text_load_start = std::chrono::steady_clock::now();
    if (!text_stage.load(options.text_encoder_dir, &text_result)) {
        return fail("sd_lens_text_encoder_load failed: " + text_result.error);
    }
    const auto text_load_end = std::chrono::steady_clock::now();
    if (result != nullptr) {
        result->lifecycle.text_encoder_loaded = true;
        result->text_encoder_load_seconds = std::chrono::duration<double>(text_load_end - text_load_start).count();
    }
    std::cout << "vram_snapshot stage=after_text_encoder_load source=external-only\n";

    sd_lens_cond_v1_native condition;
    sd_lens_text_encoder_encode_options encode_options;
    encode_options.bootstrap_oracle_dir = options.bootstrap_oracle_dir;
    encode_options.output_safetensors = options.optional_cond_out;
    encode_options.output_condition = &condition;
    sd_lens_tokenized_prompt tokenized;
    if (!options.prompt.empty()) {
        const std::filesystem::path tokenizer_dir =
            options.tokenizer_dir.empty()
                ? (std::filesystem::path(options.text_encoder_dir).parent_path() / "tokenizer")
                : std::filesystem::path(options.tokenizer_dir);
        sd_lens_gptoss_tokenizer tokenizer;
        std::string tokenizer_error;
        if (!tokenizer.load(tokenizer_dir.string(), &tokenizer_error) ||
            !tokenizer.encode_prompt(options.prompt, 128, options.chat_current_date, &tokenized, &tokenizer_error)) {
            text_stage.free();
            return fail("Lens GPT-OSS prompt tokenization failed: " + tokenizer_error);
        }
        encode_options.input_ids = tokenized.input_ids;
        encode_options.attention_mask = tokenized.attention_mask;
        encode_options.prompt = options.prompt;
        encode_options.tokenizer_source = "native_gptoss_tokenizer_json";
        encode_options.bootstrap_tokens = false;
        std::cout << "Lens staged prompt tokenized:"
                  << " tokenizer=native_gptoss_tokenizer_json"
                  << " prompt_chars=" << options.prompt.size()
                  << " rendered_chars=" << tokenized.rendered_prompt.size()
                  << " raw_seq_len=" << tokenized.raw_seq_len
                  << " trimmed_seq_len=" << tokenized.trimmed_seq_len
                  << " txt_offset=" << tokenized.txt_offset
                  << " chat_current_date="
                  << (options.chat_current_date.empty() ? sd_lens_current_date_yyyy_mm_dd() : options.chat_current_date)
                  << "\n";
    }
    const auto text_encode_start = std::chrono::steady_clock::now();
    if (!text_stage.encode(encode_options, &text_result)) {
        text_stage.free();
        return fail("sd_lens_text_encoder_encode failed: " + text_result.error);
    }
    const auto text_encode_end = std::chrono::steady_clock::now();
    if (result != nullptr) {
        result->text_encoder_encode_seconds = text_result.encode_seconds;
        result->text_encoder_wrapper_seconds =
            std::chrono::duration<double>(text_encode_end - text_encode_start).count();
    }
    std::cout << "vram_snapshot stage=after_text_encoder_encode source=external-only\n";

    text_stage.free();
    if (result != nullptr) {
        result->lifecycle.text_encoder_released = true;
        result->condition = condition;
    }
    std::cout << "vram_snapshot stage=after_text_encoder_free source=external-only\n";

    std::unordered_map<std::string, Tensor> condition_tensors = sd_lens_make_transformer_condition_map(condition);

    sd_lens_transformer_runtime_options transformer_options;
    transformer_options.transformer_dir = options.transformer_dir;
    transformer_options.latent_npy = options.latent_npy;
    transformer_options.packed_tokens_npy = options.packed_tokens_npy;
    transformer_options.width = options.width;
    transformer_options.height = options.height;
    transformer_options.steps = options.steps;
    transformer_options.seed = options.seed;
    transformer_options.repeat_generations = options.repeat_generations;
    transformer_options.attention_mode = "regular-f32";
    transformer_options.use_transformer_context = true;
    transformer_options.transformer_residency = options.transformer_residency;
    transformer_options.dynamic_residency = options.dynamic_residency;
    transformer_options.window_blocks = options.transformer_window_blocks;
    transformer_options.persistent_blocks = options.transformer_persistent_blocks;
    transformer_options.persistent_blocks_memory_mib = options.transformer_persistent_blocks_memory_mib;

    sd_lens_transformer_stage transformer_stage;
    sd_lens_transformer_runtime_result transformer_result;
    std::cout << "vram_snapshot stage=before_transformer_load source=external-only\n";
    if (!transformer_stage.run(transformer_options, condition_tensors, &transformer_result)) {
        return fail(transformer_result.error.empty() ? "Lens transformer stage failed" : transformer_result.error);
    }
    if (result != nullptr) {
        result->lifecycle.transformer_ran = true;
        result->transformer_wall_seconds = transformer_result.wall_seconds;
        result->latent = std::move(transformer_result.latent);
    }
    return true;
}

bool sd_lens_run_warm_staged_runtime(
    const sd_lens_staged_pipeline_options& options,
    sd_lens_staged_pipeline_result* result) {
    if (result != nullptr) {
        *result = {};
    }
    auto fail = [&](const std::string& message) {
        if (result != nullptr) {
            result->error = message;
        }
        return false;
    };
    if (options.text_encoder_dir.empty() ||
        options.transformer_dir.empty()) {
        return fail("Lens warm staged runtime requires text_encoder_dir and transformer_dir");
    }
    if (options.bootstrap_oracle_dir.empty() && options.prompt.empty()) {
        return fail("Lens warm staged runtime requires bootstrap_oracle_dir or prompt");
    }

    sd_lens_staged_runtime runtime(options);
    runtime.lifecycle.warm_runtime = true;
    runtime.lifecycle.same_condition_reuse = options.repeat_generations > 1;

    if (!runtime.load_text_encoder()) {
        return fail(runtime.error);
    }
    if (!runtime.encode_condition()) {
        runtime.destroy();
        return fail(runtime.error);
    }
    runtime.unload_text_encoder();

    if (!runtime.load_transformer()) {
        runtime.destroy();
        return fail(runtime.error);
    }
    if (!runtime.run_transformer_generation()) {
        runtime.destroy();
        return fail(runtime.error);
    }
    runtime.unload_transformer();

    if (result != nullptr) {
        result->lifecycle = runtime.lifecycle;
        result->condition = std::move(runtime.condition);
        result->latent = std::move(runtime.latent);
        result->text_encoder_load_seconds = runtime.text_encoder_load_seconds;
        result->text_encoder_encode_seconds = runtime.text_encoder_encode_seconds;
        result->text_encoder_wrapper_seconds = runtime.text_encoder_wrapper_seconds;
        result->transformer_wall_seconds = runtime.transformer_wall_seconds;
    }
    runtime.condition_tensors.clear();
    runtime.condition_ready = false;
    return true;
}

bool sd_lens_run_prompt_request(
    const LensPromptRequest& request,
    LensPromptResult* result) {
    if (result != nullptr) {
        *result = {};
        result->speed_mode = request.speed_mode;
        result->quality_mode = request.quality_mode;
        result->hidden_parity_exact = request.speed_mode != "bf16-resident";
        result->prompt_tokenizer_used = !request.prompt.empty();
        result->bootstrap_oracle_used = !request.bootstrap_oracle_dir.empty();
        if (request.speed_mode == "bf16-resident") {
            result->warnings.push_back("bf16-resident speed mode is image-validated but not hidden-parity exact");
        }
    }
    auto fail = [&](const std::string& message) {
        if (result != nullptr) {
            result->error = message;
        }
        return false;
    };
    if (request.text_encoder_dir.empty() ||
        request.transformer_dir.empty() ||
        request.vae_path.empty()) {
        return fail("Lens prompt request requires text_encoder_dir, transformer_dir, and vae_path");
    }
    if (request.bootstrap_oracle_dir.empty() && request.prompt.empty()) {
        return fail("Lens prompt request requires bootstrap_oracle_dir or prompt");
    }

    sd_lens_staged_runtime runtime;
    if (!runtime.configure(request)) {
        return fail(runtime.error);
    }
    if (!runtime.encode_prompt(result)) {
        runtime.destroy();
        return fail(runtime.error);
    }
    if (!runtime.prepare_transformer(result)) {
        runtime.destroy();
        return fail(runtime.error);
    }
    if (!runtime.generate(result)) {
        runtime.destroy();
        return fail(runtime.error);
    }
    runtime.unload_transformer();
    if (!runtime.decode_image(request.vae_path, result)) {
        runtime.destroy();
        return fail(runtime.error);
    }
    sd_lens_copy_runtime_to_prompt_result(runtime, result);
    runtime.condition_tensors.clear();
    runtime.condition_ready = false;
    return true;
}

#endif
