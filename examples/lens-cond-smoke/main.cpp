#include "stable-diffusion.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s --cond lens_cond_v1.safetensors [--scheduler-check]\n", argv0);
}

static bool nearly_equal(float a, float b, float eps = 1.0e-5f) {
    return std::fabs(a - b) <= eps;
}

static float expected_shift(int image_seq_len, int steps) {
    const float a1 = 8.73809524e-05f;
    const float b1 = 1.89833333f;
    const float a2 = 0.00016927f;
    const float b2 = 0.45666666f;
    if (image_seq_len > 4300) {
        return a2 * static_cast<float>(image_seq_len) + b2;
    }
    const float m_200 = a2 * static_cast<float>(image_seq_len) + b2;
    const float m_10 = a1 * static_cast<float>(image_seq_len) + b1;
    const float a = (m_200 - m_10) / 190.0f;
    const float b = m_200 - 200.0f * a;
    return a * static_cast<float>(steps) + b;
}

static float expected_dynamic_shift(float mu, float t) {
    if (t <= 0.0f) {
        return 0.0f;
    }
    if (t >= 1.0f) {
        return 1.0f;
    }
    const float exp_mu = std::exp(mu);
    return exp_mu / (exp_mu + std::pow((1.0f / t - 1.0f), 1.0f));
}

static bool check_lens_schedule(int steps, int image_seq_len) {
    sd_lens_schedule_options_t schedule_options;
    sd_lens_schedule_options_init(&schedule_options);
    schedule_options.steps = steps;
    schedule_options.image_seq_len = image_seq_len;

    float sigmas[16] = {};
    float timesteps[16] = {};
    sd_lens_schedule_desc_t schedule_desc;
    sd_lens_schedule_desc_init(&schedule_desc);
    if (!sd_lens_turbo_build_schedule(&schedule_options,
                                      sigmas,
                                      static_cast<uint32_t>(sizeof(sigmas) / sizeof(sigmas[0])),
                                      timesteps,
                                      static_cast<uint32_t>(sizeof(timesteps) / sizeof(timesteps[0])),
                                      &schedule_desc)) {
        return false;
    }
    const float mu = expected_shift(image_seq_len, steps);
    if (schedule_desc.steps != steps ||
        schedule_desc.sigma_count != steps + 1 ||
        schedule_desc.timestep_count != steps ||
        schedule_desc.image_seq_len != image_seq_len ||
        !schedule_desc.use_dynamic_shifting ||
        !nearly_equal(schedule_desc.mu, mu)) {
        return false;
    }
    for (int i = 0; i < steps; ++i) {
        float sigma = 1.0f;
        if (steps > 1) {
            const float t = static_cast<float>(i) / static_cast<float>(steps - 1);
            sigma = 1.0f + (1.0f / static_cast<float>(steps) - 1.0f) * t;
        }
        const float expected_sigma = expected_dynamic_shift(mu, sigma);
        if (!nearly_equal(sigmas[i], expected_sigma) ||
            !nearly_equal(timesteps[i], expected_sigma * 1000.0f)) {
            return false;
        }
        if (i > 0 && !(sigmas[i - 1] > sigmas[i])) {
            return false;
        }
    }
    return nearly_equal(sigmas[steps], 0.0f);
}

int main(int argc, char** argv) {
    const char* cond_path = nullptr;
    bool scheduler_check = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cond") == 0 && i + 1 < argc) {
            cond_path = argv[++i];
        } else if (std::strcmp(argv[i], "--scheduler-check") == 0) {
            scheduler_check = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (cond_path == nullptr || cond_path[0] == '\0') {
        usage(argv[0]);
        return 2;
    }

    sd_ctx_t* ctx = new_sd_ctx_lens_conditioning_only();
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create Lens conditioning-only context\n");
        return 1;
    }

    sd_lens_conditioning_options_t options;
    sd_lens_conditioning_options_init(&options);
    options.expected_batch = 1;
    options.expected_seq_len = 3;
    options.expected_hidden_size = 4;
    options.expected_selected_layer_count = 4;
    options.expected_tensor_count = 5;
    options.expected_dtype = SD_DTYPE_F32;

    sd_lens_conditioning_handle_t handle = 0;
    sd_lens_conditioning_desc_t desc;
    sd_lens_conditioning_desc_init(&desc);
    if (!sd_lens_conditioning_load(ctx, cond_path, &options, &handle, &desc)) {
        std::fprintf(stderr, "sd_lens_conditioning_load failed for %s\n", cond_path);
        free_sd_ctx(ctx);
        return 1;
    }
    if (handle == 0 || desc.handle != handle || desc.batch != 1 || desc.seq_len != 3 ||
        desc.hidden_size != 4 || desc.selected_layer_count != 4 || desc.tensor_count != 5 ||
        desc.backend != SD_BACKEND_CPU || desc.device_resident) {
        std::fprintf(stderr, "unexpected Lens conditioning descriptor\n");
        free_sd_ctx(ctx);
        return 1;
    }

    sd_lens_conditioning_desc_t desc_again;
    sd_lens_conditioning_desc_init(&desc_again);
    if (!sd_lens_conditioning_get_desc(ctx, handle, &desc_again) || desc_again.shape_hash != desc.shape_hash ||
        desc_again.dtype_hash != desc.dtype_hash) {
        std::fprintf(stderr, "sd_lens_conditioning_get_desc failed or changed descriptor hashes\n");
        free_sd_ctx(ctx);
        return 1;
    }

    sd_lens_conditioning_options_t bad_options = options;
    bad_options.expected_hidden_size = 5;
    sd_lens_conditioning_handle_t bad_handle = 0;
    if (sd_lens_conditioning_load(ctx, cond_path, &bad_options, &bad_handle, nullptr) || bad_handle != 0) {
        std::fprintf(stderr, "Lens conditioning load accepted a hidden-size mismatch\n");
        free_sd_ctx(ctx);
        return 1;
    }
    bad_options = options;
    bad_options.expected_storage_flags = SD_LENS_COND_STORAGE_FLAG_CUDA;
    if (sd_lens_conditioning_load(ctx, cond_path, &bad_options, &bad_handle, nullptr) || bad_handle != 0) {
        std::fprintf(stderr, "Lens conditioning load accepted unsupported CUDA storage\n");
        free_sd_ctx(ctx);
        return 1;
    }
    bad_options = options;
    bad_options.expected_shape_hash = desc.shape_hash + 1;
    if (sd_lens_conditioning_load(ctx, cond_path, &bad_options, &bad_handle, nullptr) || bad_handle != 0) {
        std::fprintf(stderr, "Lens conditioning load accepted a shape-hash mismatch\n");
        free_sd_ctx(ctx);
        return 1;
    }

    if (!sd_lens_conditioning_release(ctx, handle)) {
        std::fprintf(stderr, "sd_lens_conditioning_release failed\n");
        free_sd_ctx(ctx);
        return 1;
    }
    if (sd_lens_conditioning_get_desc(ctx, handle, &desc_again)) {
        std::fprintf(stderr, "released Lens conditioning handle remained visible\n");
        free_sd_ctx(ctx);
        return 1;
    }

    if (scheduler_check) {
        if (!check_lens_schedule(1, 4096) ||
            !check_lens_schedule(2, 4096) ||
            !check_lens_schedule(4, 4096) ||
            !check_lens_schedule(8, 4096) ||
            !check_lens_schedule(4, 1024) ||
            !check_lens_schedule(4, 256)) {
            std::fprintf(stderr, "unexpected Lens scheduler trace\n");
            free_sd_ctx(ctx);
            return 1;
        }
    }

    std::printf("Lens conditioning smoke passed: handle=%" PRIu64 " batch=%" PRId64 " seq=%" PRId64 " hidden=%" PRId64 " bytes=%" PRIu64 "\n",
                handle,
                desc.batch,
                desc.seq_len,
                desc.hidden_size,
                desc.estimated_bytes);
    free_sd_ctx(ctx);
    return 0;
}
