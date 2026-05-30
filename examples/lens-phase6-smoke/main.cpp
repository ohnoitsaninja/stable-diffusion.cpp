#include "stable-diffusion.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Args {
    std::string cond;
    std::string transformer;
    int steps = 4;
    int image_seq_len = 4096;
    int latent_h = 8;
    int latent_w = 8;
    bool check_real_load_refusal = false;
};

static void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --cond lens_cond_v1.safetensors --transformer <Lens-Turbo/transformer> [options]\n"
                 "options:\n"
                 "  --steps N                       Lens-Turbo schedule steps, default 4\n"
                 "  --image-seq-len N               Lens image sequence length, default 4096\n"
                 "  --latent-h N --latent-w N       Tiny public Lens latent size, default 8x8\n"
                 "  --check-real-load-refusal       Verify non-metadata transformer load fails closed\n",
                 argv0);
}

static bool parse_int(const char* text, int& out) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < -2147483647L || value > 2147483647L) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (std::strcmp(arg, "--cond") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.cond = value;
        } else if (std::strcmp(arg, "--transformer") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr) return false;
            args.transformer = value;
        } else if (std::strcmp(arg, "--steps") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr || !parse_int(value, args.steps)) return false;
        } else if (std::strcmp(arg, "--image-seq-len") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr || !parse_int(value, args.image_seq_len)) return false;
        } else if (std::strcmp(arg, "--latent-h") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr || !parse_int(value, args.latent_h)) return false;
        } else if (std::strcmp(arg, "--latent-w") == 0) {
            const char* value = need_value(arg);
            if (value == nullptr || !parse_int(value, args.latent_w)) return false;
        } else if (std::strcmp(arg, "--check-real-load-refusal") == 0) {
            args.check_real_load_refusal = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            return false;
        }
    }
    if (args.cond.empty() || args.transformer.empty()) {
        return false;
    }
    if (args.steps <= 0 || args.image_seq_len <= 0 || args.latent_h <= 0 || args.latent_w <= 0 ||
        (args.latent_h % 2) != 0 || (args.latent_w % 2) != 0) {
        std::fprintf(stderr, "steps/image_seq_len must be positive and latent H/W must be positive even values\n");
        return false;
    }
    return true;
}

static bool nearly_equal(float a, float b, float eps = 1.0e-5f) {
    return std::fabs(a - b) <= eps;
}

static bool build_and_check_schedule(int steps,
                                     int image_seq_len,
                                     std::vector<float>& sigmas,
                                     std::vector<float>& timesteps,
                                     sd_lens_schedule_desc_t& desc) {
    sd_lens_schedule_options_t options;
    sd_lens_schedule_options_init(&options);
    options.steps = steps;
    options.image_seq_len = image_seq_len;

    sigmas.assign(static_cast<size_t>(steps + 1), 0.0f);
    timesteps.assign(static_cast<size_t>(steps), 0.0f);
    sd_lens_schedule_desc_init(&desc);
    if (!sd_lens_turbo_build_schedule(&options,
                                      sigmas.data(),
                                      static_cast<uint32_t>(sigmas.size()),
                                      timesteps.data(),
                                      static_cast<uint32_t>(timesteps.size()),
                                      &desc)) {
        std::fprintf(stderr, "sd_lens_turbo_build_schedule failed\n");
        return false;
    }
    if (desc.steps != steps || desc.sigma_count != steps + 1 || desc.timestep_count != steps ||
        desc.image_seq_len != image_seq_len || !desc.use_dynamic_shifting || !nearly_equal(sigmas.back(), 0.0f)) {
        std::fprintf(stderr, "unexpected Lens schedule descriptor\n");
        return false;
    }
    for (int i = 1; i < steps; ++i) {
        if (!(sigmas[static_cast<size_t>(i - 1)] > sigmas[static_cast<size_t>(i)])) {
            std::fprintf(stderr, "Lens schedule sigmas are not strictly descending\n");
            return false;
        }
    }
    return true;
}

static bool pack_tiny_latent(int h, int w, sd_lens_vae_latent_desc_t& desc) {
    const uint64_t input_elements = static_cast<uint64_t>(1) * 32u * static_cast<uint64_t>(h) * static_cast<uint64_t>(w);
    const uint64_t output_elements = static_cast<uint64_t>(1) * 128u * static_cast<uint64_t>(h / 2) * static_cast<uint64_t>(w / 2);
    std::vector<float> lens_latent(static_cast<size_t>(input_elements), 0.0f);
    std::vector<float> packed(static_cast<size_t>(output_elements), -1.0f);
    for (size_t i = 0; i < lens_latent.size(); ++i) {
        lens_latent[i] = static_cast<float>(i % 257) / 257.0f;
    }
    sd_lens_vae_latent_desc_init(&desc);
    if (!sd_lens_pack_vae_latent_f32(lens_latent.data(),
                                    input_elements,
                                    1,
                                    32,
                                    h,
                                    w,
                                    packed.data(),
                                    output_elements,
                                    &desc)) {
        std::fprintf(stderr, "sd_lens_pack_vae_latent_f32 failed\n");
        return false;
    }
    if (desc.input_n != 1 || desc.input_c != 32 || desc.input_h != h || desc.input_w != w ||
        desc.output_n != 1 || desc.output_c != 128 || desc.output_h != h / 2 || desc.output_w != w / 2 ||
        desc.input_elements != input_elements || desc.output_elements != output_elements) {
        std::fprintf(stderr, "unexpected Lens VAE pack descriptor\n");
        return false;
    }
    std::vector<float> packed_bsc(static_cast<size_t>(output_elements), 0.0f);
    for (int py = 0; py < h / 2; ++py) {
        for (int px = 0; px < w / 2; ++px) {
            const int token = py * (w / 2) + px;
            for (int pc = 0; pc < 128; ++pc) {
                const size_t nchw_idx = static_cast<size_t>(pc * (h / 2) * (w / 2) + py * (w / 2) + px);
                const size_t bsc_idx = static_cast<size_t>(token * 128 + pc);
                packed_bsc[bsc_idx] = packed[nchw_idx];
            }
        }
    }
    std::vector<float> unpacked(static_cast<size_t>(input_elements), -1.0f);
    sd_lens_vae_latent_desc_t unpack_desc;
    sd_lens_vae_latent_desc_init(&unpack_desc);
    if (!sd_lens_unpack_vae_latent_f32(packed_bsc.data(),
                                       output_elements,
                                       1,
                                       32,
                                       h,
                                       w,
                                       unpacked.data(),
                                       input_elements,
                                       &unpack_desc)) {
        std::fprintf(stderr, "sd_lens_unpack_vae_latent_f32 failed\n");
        return false;
    }
    if (unpack_desc.input_n != 1 || unpack_desc.input_c != 128 || unpack_desc.input_h != h / 2 ||
        unpack_desc.input_w != w / 2 || unpack_desc.output_n != 1 || unpack_desc.output_c != 32 ||
        unpack_desc.output_h != h || unpack_desc.output_w != w ||
        unpack_desc.input_elements != output_elements || unpack_desc.output_elements != input_elements) {
        std::fprintf(stderr, "unexpected Lens VAE unpack descriptor\n");
        return false;
    }
    for (size_t i = 0; i < lens_latent.size(); ++i) {
        if (!nearly_equal(lens_latent[i], unpacked[i], 0.0f)) {
            std::fprintf(stderr, "Lens VAE pack/unpack roundtrip mismatch\n");
            return false;
        }
    }
    return true;
}

static bool run_external_flow_loop_check(sd_ctx_t* ctx,
                                         sd_lens_conditioning_handle_t cond_handle,
                                         sd_lens_transformer_handle_t transformer_handle,
                                         int steps,
                                         int image_seq_len,
                                         uint32_t channels,
                                         sd_lens_external_flow_loop_desc_t& desc) {
    if (channels == 0 || steps <= 0 || image_seq_len <= 0) {
        std::fprintf(stderr, "invalid Lens external flow-loop dimensions\n");
        return false;
    }
    const uint64_t token_elements = static_cast<uint64_t>(image_seq_len) * static_cast<uint64_t>(channels);
    std::vector<float> initial(static_cast<size_t>(token_elements), 0.0f);
    std::vector<float> model_outputs(static_cast<size_t>(token_elements) * static_cast<size_t>(steps), 0.0f);
    std::vector<float> output(static_cast<size_t>(token_elements), 0.0f);
    std::vector<float> expected(static_cast<size_t>(token_elements), 0.0f);
    for (uint64_t i = 0; i < token_elements; ++i) {
        initial[static_cast<size_t>(i)] = static_cast<float>(i % 257) * 0.0001f;
        expected[static_cast<size_t>(i)] = initial[static_cast<size_t>(i)];
    }
    for (int step = 0; step < steps; ++step) {
        const uint64_t step_offset = token_elements * static_cast<uint64_t>(step);
        for (uint64_t i = 0; i < token_elements; ++i) {
            model_outputs[static_cast<size_t>(step_offset + i)] =
                static_cast<float>((i + static_cast<uint64_t>(step) * 17u) % 251u) * 0.0002f;
        }
    }

    sd_lens_schedule_options_t options;
    sd_lens_schedule_options_init(&options);
    options.steps = steps;
    options.image_seq_len = image_seq_len;
    std::vector<float> sigmas(static_cast<size_t>(steps + 1), 0.0f);
    std::vector<float> timesteps(static_cast<size_t>(steps), 0.0f);
    sd_lens_schedule_desc_t schedule_desc;
    sd_lens_schedule_desc_init(&schedule_desc);
    if (!sd_lens_turbo_build_schedule(&options,
                                      sigmas.data(),
                                      static_cast<uint32_t>(sigmas.size()),
                                      timesteps.data(),
                                      static_cast<uint32_t>(timesteps.size()),
                                      &schedule_desc)) {
        std::fprintf(stderr, "sd_lens_turbo_build_schedule failed for external flow-loop check\n");
        return false;
    }
    for (int step = 0; step < steps; ++step) {
        const float dt = sigmas[static_cast<size_t>(step + 1)] - sigmas[static_cast<size_t>(step)];
        const uint64_t step_offset = token_elements * static_cast<uint64_t>(step);
        for (uint64_t i = 0; i < token_elements; ++i) {
            expected[static_cast<size_t>(i)] += dt * model_outputs[static_cast<size_t>(step_offset + i)];
        }
    }
    float expected_max_abs = 0.0f;
    double expected_sum_abs = 0.0;
    for (float value : expected) {
        const float abs_value = std::fabs(value);
        expected_max_abs = std::max(expected_max_abs, abs_value);
        expected_sum_abs += static_cast<double>(abs_value);
    }
    const float expected_mean_abs = expected.empty()
                                        ? 0.0f
                                        : static_cast<float>(expected_sum_abs / static_cast<double>(expected.size()));

    sd_lens_external_flow_loop_desc_init(&desc);
    if (!sd_lens_run_external_flow_loop_f32(ctx,
                                           cond_handle,
                                           transformer_handle,
                                           &options,
                                           initial.data(),
                                           token_elements,
                                           model_outputs.data(),
                                           static_cast<uint64_t>(model_outputs.size()),
                                           output.data(),
                                           static_cast<uint64_t>(output.size()),
                                           &desc)) {
        std::fprintf(stderr, "sd_lens_run_external_flow_loop_f32 failed\n");
        return false;
    }
    if (desc.steps != steps || desc.image_seq_len != image_seq_len || desc.in_channels != channels ||
        desc.packed_token_elements != token_elements ||
        desc.model_output_elements != static_cast<uint64_t>(model_outputs.size()) ||
        !desc.used_precomputed_conditioning || !desc.used_external_model_output ||
        desc.native_transformer_forward || !desc.cpu_only ||
        !nearly_equal(desc.max_abs, expected_max_abs, 1.0e-6f) ||
        !nearly_equal(desc.mean_abs, expected_mean_abs, 1.0e-6f)) {
        std::fprintf(stderr, "unexpected Lens external flow-loop descriptor\n");
        return false;
    }
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < token_elements; ++i) {
        if (!std::isfinite(output[static_cast<size_t>(i)])) {
            std::fprintf(stderr, "Lens external flow-loop produced non-finite output\n");
            return false;
        }
        max_diff = std::max(max_diff, std::fabs(output[static_cast<size_t>(i)] - expected[static_cast<size_t>(i)]));
    }
    if (max_diff > 1.0e-6f) {
        std::fprintf(stderr, "Lens external flow-loop max diff %.9g exceeded tolerance\n", max_diff);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }

    sd_ctx_t* ctx = new_sd_ctx_lens_conditioning_only();
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create Lens conditioning-only context\n");
        return 1;
    }

    sd_lens_conditioning_options_t cond_options;
    sd_lens_conditioning_options_init(&cond_options);
    cond_options.expected_schema_version = 1;
    cond_options.expected_selected_layer_count = 4;
    cond_options.expected_tensor_count = 5;
    cond_options.expected_storage_flags = SD_LENS_COND_STORAGE_FLAG_CPU;

    sd_lens_conditioning_handle_t cond_handle = 0;
    sd_lens_conditioning_desc_t cond_desc;
    sd_lens_conditioning_desc_init(&cond_desc);
    if (!sd_lens_conditioning_load(ctx, args.cond.c_str(), &cond_options, &cond_handle, &cond_desc)) {
        std::fprintf(stderr, "sd_lens_conditioning_load failed for %s\n", args.cond.c_str());
        free_sd_ctx(ctx);
        return 1;
    }
    if (cond_handle == 0 || cond_desc.handle != cond_handle || cond_desc.tensor_count != 5 ||
        cond_desc.selected_layer_count != 4 || cond_desc.backend != SD_BACKEND_CPU || cond_desc.device_resident) {
        std::fprintf(stderr, "unexpected Lens conditioning descriptor\n");
        free_sd_ctx(ctx);
        return 1;
    }

    sd_lens_transformer_options_t transformer_options;
    sd_lens_transformer_options_init(&transformer_options);
    transformer_options.metadata_only = true;
    transformer_options.allow_unsafe_large_allocations = false;
    sd_lens_transformer_handle_t transformer_handle = 0;
    sd_lens_transformer_desc_t transformer_desc;
    sd_lens_transformer_desc_init(&transformer_desc);
    if (!sd_lens_transformer_load(ctx, args.transformer.c_str(), &transformer_options, &transformer_handle, &transformer_desc)) {
        std::fprintf(stderr, "sd_lens_transformer_load metadata-only failed for %s\n", args.transformer.c_str());
        free_sd_ctx(ctx);
        return 1;
    }
    if (transformer_handle == 0 || transformer_desc.handle != transformer_handle ||
        !transformer_desc.metadata_only || transformer_desc.weights_loaded || transformer_desc.forward_supported ||
        transformer_desc.layers != 48 || transformer_desc.in_channels != 128 || transformer_desc.out_channels != 32) {
        std::fprintf(stderr, "unexpected Lens transformer metadata descriptor\n");
        free_sd_ctx(ctx);
        return 1;
    }

    if (args.check_real_load_refusal) {
        sd_lens_transformer_options_t unsafe_options;
        sd_lens_transformer_options_init(&unsafe_options);
        unsafe_options.metadata_only = false;
        unsafe_options.allow_unsafe_large_allocations = false;
        sd_lens_transformer_handle_t refused_handle = 0;
        if (sd_lens_transformer_load(ctx, args.transformer.c_str(), &unsafe_options, &refused_handle, nullptr) ||
            refused_handle != 0) {
            std::fprintf(stderr, "Lens transformer non-metadata load did not fail closed\n");
            free_sd_ctx(ctx);
            return 1;
        }
    }

    std::vector<float> sigmas;
    std::vector<float> timesteps;
    sd_lens_schedule_desc_t schedule_desc;
    if (!build_and_check_schedule(args.steps, args.image_seq_len, sigmas, timesteps, schedule_desc)) {
        free_sd_ctx(ctx);
        return 1;
    }

    sd_lens_vae_latent_desc_t latent_desc;
    if (!pack_tiny_latent(args.latent_h, args.latent_w, latent_desc)) {
        free_sd_ctx(ctx);
        return 1;
    }
    const int flow_image_seq_len = static_cast<int>(latent_desc.output_h * latent_desc.output_w);
    sd_lens_external_flow_loop_desc_t flow_desc;
    if (!run_external_flow_loop_check(ctx,
                                      cond_handle,
                                      transformer_handle,
                                      args.steps,
                                      flow_image_seq_len,
                                      transformer_desc.in_channels,
                                      flow_desc)) {
        free_sd_ctx(ctx);
        return 1;
    }

    if (!sd_lens_transformer_release(ctx, transformer_handle)) {
        std::fprintf(stderr, "sd_lens_transformer_release failed\n");
        free_sd_ctx(ctx);
        return 1;
    }
    if (!sd_lens_conditioning_release(ctx, cond_handle)) {
        std::fprintf(stderr, "sd_lens_conditioning_release failed\n");
        free_sd_ctx(ctx);
        return 1;
    }

    std::printf("Lens Phase 6 scaffold passed: cond_batch=%" PRId64
                " cond_seq=%" PRId64
                " cond_hidden=%" PRId64
                " transformer_tensors=%u transformer_bytes=%" PRIu64
                " schedule_steps=%d image_seq_len=%d first_sigma=%.8f last_sigma=%.8f"
                " packed_latent=1x128x%" PRId64 "x%" PRId64
                " external_flow_steps=%d external_flow_seq=%d external_flow_max_abs=%.8f"
                " external_transformer_required=true native_real_forward=false native_flow_loop=true\n",
                cond_desc.batch,
                cond_desc.seq_len,
                cond_desc.hidden_size,
                transformer_desc.tensor_count,
                transformer_desc.estimated_bytes,
                schedule_desc.steps,
                schedule_desc.image_seq_len,
                sigmas.front(),
                sigmas.back(),
                latent_desc.output_h,
                latent_desc.output_w,
                flow_desc.steps,
                flow_desc.image_seq_len,
                flow_desc.max_abs);

    free_sd_ctx(ctx);
    return 0;
}
