#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/media_io.h"
#include "stable-diffusion.h"

struct Args {
    std::string model;
    std::string diffusion_model;
    std::string vae;
    std::string taesd;
    std::string clip_l;
    std::string t5xxl;
    std::string llm;
    std::string image;
    std::string ref_image;
    std::string output = "latent_smoke.png";
    std::string prompt = "a detailed fantasy orc portrait, high quality";
    std::string negative_prompt = "blurry, low quality, noisy";
    std::string sampling_method;
    int image_channels = 4;
    int width          = 0;
    int height         = 0;
    int steps          = 0;
    float cfg_scale    = 0.0f;
    bool sample        = false;
    bool sample_without_init = false;
    bool decode        = true;
    bool skip_estimate = false;
    bool split_decode_context = false;
    bool vae_conv_direct = false;
    bool diffusion_conv_direct = false;
    bool disable_default_vae_conv_direct = false;
    bool type_f16 = false;
    bool gpu_sample_output = false;
    bool true_gpu_sampler_spike = false;
    bool gpu_sampler_backend_euler = false;
    bool compare_gpu_sampler_backend_euler = false;
    bool gpu_init_sample_input = false;
    bool gpu_encode_output = false;
    bool gpu_upload_latent_decode_input = false;
    bool gpu_latent_decode_input = false;
    bool download_gpu_latent = false;
    bool gpu_decode_output = false;
    bool download_gpu_output = false;
    bool download_gpu_output_buffer = false;
    bool condition_handles = false;
    bool condition_handles_reuse = false;
    bool strict_gpu_resident = false;
    bool dump_gpu_handle_desc = false;
    bool expect_gpu_encode_refusal = false;
    bool expect_decode_refusal = false;
    bool preview_tae = false;
    bool marigold_iid = false;
    int preview_every = 0;
    float preview_percent_interval = 0.0f;
    std::vector<float> preview_percents;
    std::string preview_prefix;
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " (--model <model> | --diffusion-model <path> --vae <path> ...) --image <image> [options]\n"
        << "Options:\n"
        << "  --diffusion-model <path> split diffusion model path for Flux/Z/Wan style contexts\n"
        << "  --vae <path>             external VAE/AE path for split model contexts\n"
        << "  --taesd <path>           TAESD path for sampling previews\n"
        << "  --clip-l <path>          CLIP-L text encoder path\n"
        << "  --t5xxl <path>           T5XXL text encoder path\n"
        << "  --llm <path>             LLM text encoder path\n"
        << "  --output <path>          decoded output path (default: latent_smoke.png)\n"
        << "  --prompt <text>          prompt for sampler smoke\n"
        << "  --negative-prompt <text> negative prompt for sampler smoke\n"
        << "  --steps <int>            sampler steps override\n"
        << "  --cfg-scale <float>      text/image CFG override\n"
        << "  --sampling-method <name> sampler method override\n"
        << "  --image-channels <3|4>   channel count to load and pass into sd_encode_image (default: 4)\n"
        << "  --ref-image <path>       reference image for edit/reference-conditioning smoke\n"
        << "  --width <int>            optional target width for sample/decode path\n"
        << "  --height <int>           optional target height for sample/decode path\n"
        << "  --vae-conv-direct        enable direct VAE convolution\n"
        << "  --diffusion-conv-direct  enable direct diffusion-model convolution\n"
        << "  --disable-default-vae-conv-direct disable CUDA SDXL default direct VAE convolution\n"
        << "  --type-f16               request f16 model tensor conversion\n"
        << "  --sample                 run sd_sample_latent after encode\n"
        << "  --sample-without-init    run sampler with no init latent and skip VAE encode\n"
        << "  --gpu-sample-output      run sd_sample_latent_gpu and keep sampled latent as a GPU handle\n"
        << "  --true-gpu-sampler-spike run experimental backend-resident Euler sampler spike\n"
        << "  --gpu-sampler-backend-euler use experimental Euler backend sampler through sd_sample_latent_gpu\n"
        << "  --compare-gpu-sampler-backend-euler compare CPU sampler latent vs experimental Euler backend latent\n"
        << "  --gpu-init-sample-input  pass a GPU latent handle into the sampler init-latent bridge API\n"
        << "  --gpu-encode-output      call sd_encode_image_normal_gpu and keep encoded latent as a GPU handle\n"
        << "  --gpu-upload-latent-decode-input upload the CPU latent, then decode from the GPU latent handle\n"
        << "  --gpu-latent-decode-input decode from a GPU latent handle with sd_decode_gpu_latent_normal_gpu\n"
        << "  --download-gpu-latent    explicitly download the sampled GPU latent handle\n"
        << "  --skip-estimate          skip sd_estimate_vae_normal_memory smoke checks\n"
        << "  --split-decode-context   decode with a separate vae_decode_only=true context\n"
        << "  --gpu-decode-output      call sd_decode_latent_normal_gpu and keep decode output as a GPU handle\n"
        << "  --download-gpu-output    explicitly download the GPU image handle and write it\n"
        << "  --download-gpu-output-buffer download GPU image directly into caller-owned RGBA8 memory\n"
        << "  --condition-handles      encode prompt/negative into resident conditioning handles and sample with them\n"
        << "  --condition-handles-reuse sample twice with the same conditioning handles to prove CLIP encode is not rerun\n"
        << "  --strict-gpu-resident    set SDCPP_STRICT_GPU_RESIDENT=1 for GPU-output checks\n"
        << "  --dump-gpu-handle-desc   print GPU handle descriptor after decode\n"
        << "  --expect-gpu-encode-refusal treat sd_encode_image_normal_gpu refusal as a passing smoke result\n"
        << "  --expect-decode-refusal  treat GPU decode refusal as a passing smoke result\n"
        << "  --preview-tae            enable TAESD sampling previews\n"
        << "  --preview-every <int>    emit preview every N denoise steps\n"
        << "  --preview-percent-interval <float> emit preview at percentage increments, e.g. 0.25\n"
        << "  --preview-percent <float> add an explicit preview percentage, repeatable\n"
        << "  --preview-prefix <path>  preview output path prefix\n"
        << "  --marigold-iid          run Marigold intrinsic decomposition smoke and write target images\n"
        << "  --no-decode              skip sd_decode_latent\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--model") {
            const char* value = need_value("--model");
            if (value == nullptr) return false;
            args.model = value;
        } else if (arg == "--diffusion-model") {
            const char* value = need_value("--diffusion-model");
            if (value == nullptr) return false;
            args.diffusion_model = value;
        } else if (arg == "--vae") {
            const char* value = need_value("--vae");
            if (value == nullptr) return false;
            args.vae = value;
        } else if (arg == "--taesd") {
            const char* value = need_value("--taesd");
            if (value == nullptr) return false;
            args.taesd = value;
        } else if (arg == "--clip-l") {
            const char* value = need_value("--clip-l");
            if (value == nullptr) return false;
            args.clip_l = value;
        } else if (arg == "--t5xxl") {
            const char* value = need_value("--t5xxl");
            if (value == nullptr) return false;
            args.t5xxl = value;
        } else if (arg == "--llm") {
            const char* value = need_value("--llm");
            if (value == nullptr) return false;
            args.llm = value;
        } else if (arg == "--image") {
            const char* value = need_value("--image");
            if (value == nullptr) return false;
            args.image = value;
        } else if (arg == "--ref-image") {
            const char* value = need_value("--ref-image");
            if (value == nullptr) return false;
            args.ref_image = value;
        } else if (arg == "--output") {
            const char* value = need_value("--output");
            if (value == nullptr) return false;
            args.output = value;
        } else if (arg == "--prompt") {
            const char* value = need_value("--prompt");
            if (value == nullptr) return false;
            args.prompt = value;
        } else if (arg == "--negative-prompt") {
            const char* value = need_value("--negative-prompt");
            if (value == nullptr) return false;
            args.negative_prompt = value;
        } else if (arg == "--steps") {
            const char* value = need_value("--steps");
            if (value == nullptr) return false;
            args.steps = std::atoi(value);
        } else if (arg == "--cfg-scale") {
            const char* value = need_value("--cfg-scale");
            if (value == nullptr) return false;
            args.cfg_scale = static_cast<float>(std::atof(value));
        } else if (arg == "--sampling-method") {
            const char* value = need_value("--sampling-method");
            if (value == nullptr) return false;
            args.sampling_method = value;
        } else if (arg == "--image-channels") {
            const char* value = need_value("--image-channels");
            if (value == nullptr) return false;
            args.image_channels = std::atoi(value);
        } else if (arg == "--width") {
            const char* value = need_value("--width");
            if (value == nullptr) return false;
            args.width = std::atoi(value);
        } else if (arg == "--height") {
            const char* value = need_value("--height");
            if (value == nullptr) return false;
            args.height = std::atoi(value);
        } else if (arg == "--vae-conv-direct") {
            args.vae_conv_direct = true;
        } else if (arg == "--diffusion-conv-direct") {
            args.diffusion_conv_direct = true;
        } else if (arg == "--disable-default-vae-conv-direct") {
            args.disable_default_vae_conv_direct = true;
        } else if (arg == "--type-f16") {
            args.type_f16 = true;
        } else if (arg == "--sample") {
            args.sample = true;
        } else if (arg == "--sample-without-init") {
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--gpu-sample-output") {
            args.gpu_sample_output = true;
            args.sample = true;
        } else if (arg == "--true-gpu-sampler-spike") {
            args.true_gpu_sampler_spike = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--gpu-sampler-backend-euler") {
            args.gpu_sampler_backend_euler = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--compare-gpu-sampler-backend-euler") {
            args.compare_gpu_sampler_backend_euler = true;
            args.gpu_sampler_backend_euler = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
            args.decode = false;
        } else if (arg == "--gpu-init-sample-input") {
            args.gpu_init_sample_input = true;
            args.gpu_sample_output = true;
            args.sample = true;
        } else if (arg == "--gpu-encode-output") {
            args.gpu_encode_output = true;
        } else if (arg == "--gpu-upload-latent-decode-input") {
            args.gpu_upload_latent_decode_input = true;
            args.gpu_latent_decode_input = true;
            args.gpu_decode_output = true;
        } else if (arg == "--gpu-latent-decode-input") {
            args.gpu_latent_decode_input = true;
            args.gpu_decode_output = true;
        } else if (arg == "--download-gpu-latent") {
            args.download_gpu_latent = true;
        } else if (arg == "--skip-estimate") {
            args.skip_estimate = true;
        } else if (arg == "--split-decode-context") {
            args.split_decode_context = true;
        } else if (arg == "--gpu-decode-output") {
            args.gpu_decode_output = true;
        } else if (arg == "--download-gpu-output") {
            args.download_gpu_output = true;
        } else if (arg == "--download-gpu-output-buffer") {
            args.download_gpu_output_buffer = true;
        } else if (arg == "--condition-handles") {
            args.condition_handles = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--condition-handles-reuse") {
            args.condition_handles = true;
            args.condition_handles_reuse = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--strict-gpu-resident") {
            args.strict_gpu_resident = true;
        } else if (arg == "--dump-gpu-handle-desc") {
            args.dump_gpu_handle_desc = true;
        } else if (arg == "--expect-gpu-encode-refusal") {
            args.expect_gpu_encode_refusal = true;
        } else if (arg == "--expect-decode-refusal") {
            args.expect_decode_refusal = true;
        } else if (arg == "--preview-tae") {
            args.preview_tae = true;
        } else if (arg == "--preview-every") {
            const char* value = need_value("--preview-every");
            if (value == nullptr) return false;
            args.preview_every = std::atoi(value);
        } else if (arg == "--preview-percent-interval") {
            const char* value = need_value("--preview-percent-interval");
            if (value == nullptr) return false;
            args.preview_percent_interval = static_cast<float>(std::atof(value));
        } else if (arg == "--preview-percent") {
            const char* value = need_value("--preview-percent");
            if (value == nullptr) return false;
            args.preview_percents.push_back(static_cast<float>(std::atof(value)));
        } else if (arg == "--preview-prefix") {
            const char* value = need_value("--preview-prefix");
            if (value == nullptr) return false;
            args.preview_prefix = value;
        } else if (arg == "--marigold-iid") {
            args.marigold_iid = true;
        } else if (arg == "--no-decode") {
            args.decode = false;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return false;
        }
    }

    if ((args.model.empty() && args.diffusion_model.empty()) || args.image.empty()) {
        return false;
    }
    if (args.image_channels != 3 && args.image_channels != 4) {
        std::cerr << "--image-channels must be 3 or 4\n";
        return false;
    }
    return true;
}

static void set_env_value(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static sd_ctx_t* create_context(const Args& args, bool vae_decode_only) {
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path            = args.model.empty() ? nullptr : args.model.c_str();
    ctx_params.diffusion_model_path  = args.diffusion_model.empty() ? nullptr : args.diffusion_model.c_str();
    ctx_params.vae_path              = args.vae.empty() ? nullptr : args.vae.c_str();
    ctx_params.taesd_path            = args.taesd.empty() ? nullptr : args.taesd.c_str();
    ctx_params.clip_l_path           = args.clip_l.empty() ? nullptr : args.clip_l.c_str();
    ctx_params.t5xxl_path            = args.t5xxl.empty() ? nullptr : args.t5xxl.c_str();
    ctx_params.llm_path              = args.llm.empty() ? nullptr : args.llm.c_str();
    ctx_params.vae_decode_only       = vae_decode_only;
    ctx_params.diffusion_flash_attn  = true;
    ctx_params.diffusion_conv_direct = args.diffusion_conv_direct;
    ctx_params.vae_conv_direct       = args.vae_conv_direct;
    ctx_params.tae_preview_only      = args.preview_tae && !args.taesd.empty();
    ctx_params.wtype                 = args.type_f16 ? SD_TYPE_F16 : SD_TYPE_COUNT;
    ctx_params.offload_params_to_cpu = false;
    ctx_params.keep_clip_on_cpu      = false;
    ctx_params.keep_vae_on_cpu       = false;
    if (args.condition_handles_reuse) {
        ctx_params.free_params_immediately = false;
    }

    std::cout << "creating context vae_decode_only=" << (vae_decode_only ? "true" : "false") << "\n";
    return new_sd_ctx(&ctx_params);
}

static const char* vae_exec_mode_name(sd_vae_exec_mode_t mode) {
    switch (mode) {
        case SD_VAE_EXEC_LEGACY_GGML_GRAPH:
            return "legacy_ggml_graph";
        case SD_VAE_EXEC_DIRECT_GRAPH:
            return "direct_graph";
        case SD_VAE_EXEC_COMFY_NORMAL:
            return "comfy_normal";
        case SD_VAE_EXEC_AUTO:
            return "auto";
        default:
            return "unknown";
    }
}

static const char* vae_dtype_name(sd_vae_dtype_t dtype) {
    switch (dtype) {
        case SD_VAE_DTYPE_BF16:
            return "bf16";
        case SD_VAE_DTYPE_F16:
            return "f16";
        case SD_VAE_DTYPE_F32:
            return "f32";
        case SD_VAE_DTYPE_AUTO:
        default:
            return "auto";
    }
}

static void print_vae_report(const char* label, const sd_vae_memory_report_t& report) {
    std::cout << label
              << " requested=" << vae_exec_mode_name(report.requested_mode)
              << " resolved=" << vae_exec_mode_name(report.resolved_mode)
              << " dtype_requested=" << vae_dtype_name(report.requested_storage_dtype)
              << " dtype_resolved=" << vae_dtype_name(report.resolved_storage_dtype)
              << " planned_mb=" << (static_cast<double>(report.planned_workspace_bytes) / 1024.0 / 1024.0)
              << " largest_mb=" << (static_cast<double>(report.largest_tensor_bytes) / 1024.0 / 1024.0)
              << " largest_op=" << report.largest_tensor_op
              << " largest_type=" << report.largest_tensor_type
              << " largest_shape=" << report.largest_tensor_shape
              << " graphs=" << report.graph_count
              << " stages=" << report.stage_count
              << " host_copies=" << report.stage_boundary_host_copies
              << " device_copies=" << report.stage_boundary_device_copies
              << " dtype_promotions=" << report.stage_boundary_dtype_promotions
              << " im2col=" << (report.used_im2col ? "true" : "false")
              << " used_im2col=" << (report.used_im2col ? "true" : "false")
              << " used_tiling=" << (report.used_tiling ? "true" : "false")
              << " used_taesd=" << (report.used_taesd ? "true" : "false")
              << " direct=" << (report.used_direct_conv ? "true" : "false")
              << " direct_conv=" << (report.used_direct_conv ? "true" : "false")
              << " device_resident=" << (report.device_resident_stages ? "true" : "false")
              << " compact=" << (report.compact_activation_storage ? "true" : "false")
              << " math_policy=\"" << report.math_dtype_policy << "\""
              << " fallback=\"" << report.fallback_reason << "\""
              << "\n";
    for (uint32_t i = 0; i < report.stage_count && i < 16; ++i) {
        std::cout << label << "_stage[" << i << "] dtype=" << report.stage_output_dtype[i]
                  << " backend=" << report.stage_output_backend[i] << "\n";
    }
}

static void print_gpu_desc(sd_ctx_t* ctx, const char* label, sd_gpu_handle_t handle) {
    sd_gpu_tensor_desc_t desc{};
    if (!sd_gpu_handle_get_desc(ctx, handle, &desc)) {
        std::cout << label << "_desc unavailable handle=" << handle << "\n";
        return;
    }
    std::cout << label << "_desc handle=" << desc.handle
              << " backend=" << desc.backend
              << " kind=" << desc.kind
              << " dtype=" << desc.dtype
              << " layout=" << desc.layout
              << " shape_nchw=" << desc.n << "x" << desc.c << "x" << desc.h << "x" << desc.w
              << " strides=" << desc.stride_n << "," << desc.stride_c << "," << desc.stride_h << "," << desc.stride_w
              << " bytes=" << desc.byte_size
              << " flags=" << desc.flags
              << " refcount=" << desc.refcount << "\n";
}

static void print_conditioning_desc(sd_ctx_t* ctx, const char* label, sd_conditioning_handle_t handle) {
    sd_conditioning_desc_t desc{};
    if (!sd_conditioning_get_desc(ctx, handle, &desc)) {
        std::cout << label << "_desc unavailable handle=" << handle << "\n";
        return;
    }
    std::cout << label << "_desc handle=" << desc.handle
              << " backend=" << desc.backend
              << " dtype=" << desc.dtype
              << " device_resident=" << (desc.device_resident ? "true" : "false")
              << " batch=" << desc.batch
              << " tokens=" << desc.token_count
              << " crossattn_dim=" << desc.crossattn_dim
              << " vector_dim=" << desc.vector_dim
              << " bytes=" << desc.estimated_bytes
              << " clip_skip=" << desc.clip_skip
              << " size=" << desc.width << "x" << desc.height
              << " zero_out_masked=" << (desc.zero_out_masked ? "true" : "false")
              << " refcount=" << desc.refcount
              << " name=\"" << desc.debug_name << "\"\n";
}

struct LatentDiffStats {
    double mean_abs = 0.0;
    float p95_abs = 0.0f;
    float p99_abs = 0.0f;
    float max_abs = 0.0f;
    uint64_t compared = 0;
    uint64_t nan_or_inf = 0;
};

static bool download_gpu_float_tensor(sd_ctx_t* ctx,
                                      sd_gpu_handle_t handle,
                                      std::vector<float>& out,
                                      sd_gpu_tensor_desc_t* out_desc = nullptr) {
    sd_gpu_tensor_desc_t desc{};
    if (!sd_gpu_handle_get_desc(ctx, handle, &desc)) {
        std::cerr << "sd_gpu_handle_get_desc failed for handle=" << handle << "\n";
        return false;
    }
    if (desc.dtype != SD_DTYPE_F32 || desc.byte_size % sizeof(float) != 0) {
        std::cerr << "expected f32 tensor handle=" << handle
                  << " dtype=" << desc.dtype
                  << " bytes=" << desc.byte_size << "\n";
        return false;
    }
    out.resize(static_cast<size_t>(desc.byte_size / sizeof(float)));
    if (!sd_gpu_tensor_download(ctx, handle, out.data(), desc.byte_size, nullptr)) {
        std::cerr << "sd_gpu_tensor_download failed for handle=" << handle << "\n";
        return false;
    }
    if (out_desc != nullptr) {
        *out_desc = desc;
    }
    return true;
}

static bool download_cpu_latent_as_float_tensor(sd_ctx_t* ctx,
                                                const sd_latent_t* latent,
                                                std::vector<float>& out,
                                                sd_gpu_tensor_desc_t* out_desc = nullptr) {
    sd_gpu_handle_t uploaded = 0;
    if (!sd_cpu_latent_upload(ctx, latent, &uploaded, nullptr)) {
        std::cerr << "sd_cpu_latent_upload failed for comparison\n";
        return false;
    }
    bool ok = download_gpu_float_tensor(ctx, uploaded, out, out_desc);
    sd_gpu_handle_release(ctx, uploaded);
    return ok;
}

static LatentDiffStats compare_float_tensors(const std::vector<float>& a,
                                             const std::vector<float>& b) {
    LatentDiffStats stats{};
    const size_t count = std::min(a.size(), b.size());
    if (count == 0) {
        return stats;
    }

    std::vector<float> diffs;
    diffs.reserve(count);
    double sum = 0.0;
    float max_abs = 0.0f;
    uint64_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
        const float av = a[i];
        const float bv = b[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            ++bad;
            continue;
        }
        const float diff = std::fabs(av - bv);
        diffs.push_back(diff);
        sum += static_cast<double>(diff);
        max_abs = std::max(max_abs, diff);
    }

    stats.compared = static_cast<uint64_t>(diffs.size());
    stats.nan_or_inf = bad;
    if (diffs.empty()) {
        return stats;
    }
    auto percentile = [&diffs](double q) -> float {
        const size_t index = static_cast<size_t>(std::min<double>(
            static_cast<double>(diffs.size() - 1),
            std::ceil(q * static_cast<double>(diffs.size())) - 1.0));
        std::nth_element(diffs.begin(), diffs.begin() + index, diffs.end());
        return diffs[index];
    };
    stats.mean_abs = sum / static_cast<double>(diffs.size());
    stats.p95_abs = percentile(0.95);
    stats.p99_abs = percentile(0.99);
    stats.max_abs = max_abs;
    return stats;
}

static bool same_latent_desc_shape(const sd_gpu_tensor_desc_t& a,
                                   const sd_gpu_tensor_desc_t& b) {
    return a.dtype == b.dtype &&
           a.layout == b.layout &&
           a.n == b.n &&
           a.c == b.c &&
           a.h == b.h &&
           a.w == b.w &&
           a.byte_size == b.byte_size;
}

static void print_model_capabilities(sd_ctx_t* ctx) {
    sd_model_pipeline_capabilities_t caps{};
    if (!sd_get_model_pipeline_capabilities(ctx, &caps)) {
        std::cout << "model_capabilities unavailable\n";
        return;
    }
    std::cout << "model_capabilities family=" << caps.family_name
              << " family_id=" << caps.family
              << " latent_channels=" << caps.latent_channels
              << " vae_scale=" << caps.vae_scale_factor
              << " default_sample=" << sd_sample_method_name(caps.default_sample_method)
              << " default_scheduler=" << sd_scheduler_name(caps.default_scheduler)
              << " default_cfg=" << caps.default_cfg_scale
              << " default_steps=" << caps.default_steps
              << " default_flow_shift=" << caps.default_flow_shift
              << " requires_clip_l=" << (caps.requires_clip_l ? "true" : "false")
              << " requires_t5xxl=" << (caps.requires_t5xxl ? "true" : "false")
              << " requires_llm=" << (caps.requires_llm ? "true" : "false")
              << " gpu_sample_bridge=" << (caps.supports_gpu_sample_bridge_output ? "true" : "false")
              << " gpu_latent_decode=" << (caps.supports_gpu_latent_decode ? "true" : "false")
              << " gpu_image_output=" << (caps.supports_gpu_image_output ? "true" : "false")
              << " gpu_vae_encode=" << (caps.supports_vae_encode_gpu_output ? "true" : "false")
              << " reference_images=" << (caps.supports_reference_images ? "true" : "false")
              << " edit_mode=" << (caps.supports_edit_mode ? "true" : "false")
              << " edit_reference_conditioning=" << (caps.supports_edit_reference_conditioning ? "true" : "false")
              << " comfy_reference_vae_encode=" << (caps.supports_comfy_reference_vae_encode ? "true" : "false")
              << " strict_sampler_true_resident=" << (caps.strict_gpu_sample_is_true_resident ? "true" : "false")
              << "\n";
}

static void print_gpu_capabilities(sd_ctx_t* ctx) {
    sd_gpu_capabilities_t caps{};
    if (!sd_get_gpu_capabilities(ctx, &caps)) {
        std::cout << "gpu_capabilities unavailable\n";
        return;
    }
    std::cout << "gpu_capabilities handles=" << (caps.supports_gpu_handles ? "true" : "false")
              << " cuda=" << (caps.supports_cuda_gpu_handles ? "true" : "false")
              << " sampler_gpu_output=" << (caps.supports_sampler_gpu_latent_output ? "true" : "false")
              << " sampler_gpu_output_bridge=" << (caps.supports_sampler_gpu_latent_bridge_output ? "true" : "false")
              << " sampler_gpu_init=" << (caps.supports_sampler_gpu_init_latent_input ? "true" : "false")
              << " sampler_gpu_init_bridge=" << (caps.supports_sampler_gpu_init_latent_bridge_input ? "true" : "false")
              << " vae_gpu_latent_input=" << (caps.supports_vae_gpu_latent_input ? "true" : "false")
              << " gpu_download=" << (caps.supports_gpu_download ? "true" : "false")
              << "\n";
}

static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    log_print(level, log, true, false);
}

struct PreviewCapture {
    std::string prefix;
    int count = 0;
};

static void preview_cb(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    PreviewCapture* capture = static_cast<PreviewCapture*>(data);
    if (capture == nullptr || frames == nullptr || frame_count <= 0) {
        return;
    }
    capture->count++;
    std::cout << "preview_event step=" << step
              << " frames=" << frame_count
              << " noisy=" << (is_noisy ? "true" : "false")
              << " count=" << capture->count << "\n";
    if (capture->prefix.empty()) {
        return;
    }
    for (int i = 0; i < frame_count; ++i) {
        std::string path = capture->prefix + "_step" + std::to_string(std::abs(step)) +
                           (is_noisy ? "_noisy" : "_denoised") +
                           "_frame" + std::to_string(i) + ".png";
        if (!write_image_to_file(path, frames[i].data, frames[i].width, frames[i].height, frames[i].channel)) {
            std::cerr << "failed to write preview: " << path << "\n";
        } else {
            std::cout << "preview_wrote " << path << "\n";
        }
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    if (args.disable_default_vae_conv_direct) {
        set_env_value("SDCPP_DISABLE_DEFAULT_VAE_CONV_DIRECT", "1");
    }
    if (args.strict_gpu_resident) {
        set_env_value("SDCPP_STRICT_GPU_RESIDENT", "1");
    }
    if (args.true_gpu_sampler_spike) {
        set_env_value("SDCPP_EXPERIMENTAL_TRUE_GPU_SAMPLER", "1");
    }
    if (args.gpu_sampler_backend_euler) {
        set_env_value("SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER", "1");
    }
    sd_set_log_callback(sd_log_cb, nullptr);

    PreviewCapture preview_capture;
    if (args.preview_tae) {
        if (args.taesd.empty()) {
            std::cerr << "--preview-tae requires --taesd\n";
            return 2;
        }
        preview_capture.prefix = args.preview_prefix;
        sd_preview_options_t preview_options;
        sd_preview_options_init(&preview_options);
        preview_options.mode = PREVIEW_TAE;
        preview_options.denoised = true;
        preview_options.noisy = false;
        if (args.preview_percent_interval > 0.0f) {
            preview_options.schedule_mode = SD_PREVIEW_SCHEDULE_PERCENT_INTERVAL;
            preview_options.percent_interval = args.preview_percent_interval;
        } else if (!args.preview_percents.empty()) {
            preview_options.schedule_mode = SD_PREVIEW_SCHEDULE_EXPLICIT_PERCENTS;
            preview_options.percent_point_count = static_cast<uint32_t>(std::min<size_t>(args.preview_percents.size(), SD_PREVIEW_MAX_PERCENT_POINTS));
            for (uint32_t i = 0; i < preview_options.percent_point_count; ++i) {
                preview_options.percent_points[i] = args.preview_percents[i];
            }
        } else {
            preview_options.schedule_mode = SD_PREVIEW_SCHEDULE_EVERY_N_STEPS;
            preview_options.step_interval = args.preview_every > 0 ? args.preview_every : 1;
        }
        sd_set_preview_callback_v2(preview_cb, &preview_options, &preview_capture);
    }

    sd_ctx_t* ctx = create_context(args, false);
    if (ctx == nullptr) {
        std::cerr << "new_sd_ctx failed\n";
        return 1;
    }
    print_model_capabilities(ctx);
    print_gpu_capabilities(ctx);

    sd_image_t image{};
    if (!load_sd_image_from_file(&image, args.image.c_str(), 0, 0, args.image_channels)) {
        if (args.image_channels == 4 && load_sd_image_from_file(&image, args.image.c_str(), 0, 0, 3)) {
            std::vector<uint8_t> rgba(static_cast<size_t>(image.width) * image.height * 4);
            for (uint32_t y = 0; y < image.height; ++y) {
                for (uint32_t x = 0; x < image.width; ++x) {
                    const size_t src = (static_cast<size_t>(y) * image.width + x) * 3;
                    const size_t dst = (static_cast<size_t>(y) * image.width + x) * 4;
                    rgba[dst + 0]    = image.data[src + 0];
                    rgba[dst + 1]    = image.data[src + 1];
                    rgba[dst + 2]    = image.data[src + 2];
                    rgba[dst + 3]    = 255;
                }
            }
            free(image.data);
            image.channel = 4;
            image.data    = static_cast<uint8_t*>(malloc(rgba.size()));
            if (image.data != nullptr) {
                std::memcpy(image.data, rgba.data(), rgba.size());
            }
        }
    }
    if (image.data == nullptr) {
        std::cerr << "failed to load image: " << args.image << "\n";
        free_sd_ctx(ctx);
        return 1;
    }
    std::cout << "loaded image " << image.width << "x" << image.height << "x" << image.channel << "\n";

    if (args.marigold_iid) {
        sd_marigold_iid_options_t iid_options;
        sd_marigold_iid_options_init(&iid_options);
        iid_options.processing_width = args.width > 0 ? static_cast<uint32_t>(args.width) : 0;
        iid_options.processing_height = args.height > 0 ? static_cast<uint32_t>(args.height) : 0;
        iid_options.steps = args.steps > 0 ? static_cast<uint32_t>(args.steps) : 4;
        iid_options.seed = 12345;
        iid_options.match_input_resolution = true;

        std::cout << "calling sd_marigold_iid_predict\n";
        sd_marigold_iid_result_t* iid = sd_marigold_iid_predict(ctx, &image, &iid_options);
        if (iid == nullptr) {
            std::cerr << "sd_marigold_iid_predict failed\n";
            free(image.data);
            free_sd_ctx(ctx);
            return 1;
        }
        std::cout << "marigold_iid_result targets=" << iid->target_count
                  << " latent=" << (iid->latent != nullptr ? "true" : "false") << "\n";
        std::string stem = args.output;
        size_t dot = stem.find_last_of('.');
        std::string ext = ".png";
        if (dot != std::string::npos) {
            ext = stem.substr(dot);
            stem = stem.substr(0, dot);
        }
        for (uint32_t i = 0; i < iid->target_count; ++i) {
            const char* target_name = (iid->target_names != nullptr && iid->target_names[i] != nullptr) ? iid->target_names[i] : "target";
            std::string path = stem + "_" + target_name + ext;
            if (!write_image_to_file(path, iid->targets[i].data, iid->targets[i].width, iid->targets[i].height, iid->targets[i].channel)) {
                std::cerr << "failed to write " << path << "\n";
                free_sd_marigold_iid_result(iid);
                free(image.data);
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "wrote " << path << " " << iid->targets[i].width << "x" << iid->targets[i].height
                      << "x" << iid->targets[i].channel << "\n";
        }
        free_sd_marigold_iid_result(iid);
        free(image.data);
        free_sd_ctx(ctx);
        return 0;
    }

    sd_tiling_params_t tiling{};
    tiling.enabled        = false;
    tiling.target_overlap = 0.5f;
    sd_vae_run_options_t vae_options;
    sd_vae_run_options_init(&vae_options);
    if (args.disable_default_vae_conv_direct) {
        vae_options.mode = SD_VAE_EXEC_LEGACY_GGML_GRAPH;
        vae_options.fail_on_large_im2col = false;
    }

    if (!args.skip_estimate) {
        sd_vae_memory_report_t encode_estimate;
        if (sd_estimate_vae_normal_memory(ctx, image.width, image.height, false, &vae_options, &encode_estimate)) {
            print_vae_report("encode_estimate", encode_estimate);
        }
    }

    sd_latent_t* encoded = nullptr;
    sd_gpu_handle_t encoded_gpu_latent = 0;
    if (!args.sample_without_init) {
        std::cout << "calling sd_encode_image\n";
        sd_vae_memory_report_t encode_report;
        if (args.gpu_encode_output) {
            if (!sd_encode_image_normal_gpu(ctx, &image, &vae_options, &encoded_gpu_latent, &encode_report)) {
                if (args.expect_gpu_encode_refusal) {
                    std::cout << "expected_gpu_encode_refusal=true\n";
                    free(image.data);
                    free_sd_ctx(ctx);
                    return 0;
                }
                std::cerr << "sd_encode_image_normal_gpu failed\n";
                free(image.data);
                free_sd_ctx(ctx);
                return 1;
            }
            print_vae_report("encode_report", encode_report);
            print_gpu_desc(ctx, "gpu_encoded_latent", encoded_gpu_latent);
            if (!args.gpu_latent_decode_input && !args.gpu_upload_latent_decode_input) {
                encoded = sd_gpu_latent_download(ctx, encoded_gpu_latent, nullptr);
                if (encoded == nullptr) {
                    std::cerr << "sd_gpu_latent_download encoded failed\n";
                    sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    free(image.data);
                    free_sd_ctx(ctx);
                    return 1;
                }
            }
        } else {
            encoded = sd_encode_image_normal(ctx, &image, &vae_options, &encode_report);
            if (encoded == nullptr) {
                std::cerr << "sd_encode_image failed\n";
                free(image.data);
                free_sd_ctx(ctx);
                return 1;
            }
            print_vae_report("encode_report", encode_report);
        }
        if (encoded != nullptr) {
            std::cout << "encoded latent " << encoded->width << "x" << encoded->height << "x"
                      << encoded->channel << " elements=" << encoded->element_count << "\n";
        }
    } else {
        std::cout << "skipping sd_encode_image; sampler init latent is null\n";
    }
    free(image.data);

    sd_latent_t* latent_to_decode = encoded;
    sd_gpu_handle_t sampled_gpu_latent = 0;
    sd_gpu_handle_t sample_init_gpu_latent = 0;
    bool sample_init_gpu_latent_owned = false;
    if (args.sample) {
        sd_image_t ref_image{};
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        sd_model_pipeline_capabilities_t caps{};
        sd_get_model_pipeline_capabilities(ctx, &caps);
        gen_params.prompt          = args.prompt.c_str();
        gen_params.negative_prompt = args.negative_prompt.c_str();
        gen_params.width           = args.width > 0 ? args.width : static_cast<int>(image.width);
        gen_params.height          = args.height > 0 ? args.height : static_cast<int>(image.height);
        gen_params.seed            = 12345;
        gen_params.strength        = 0.65f;
        gen_params.sample_params.sample_steps          = args.steps > 0 ? args.steps : (caps.default_steps > 0 ? caps.default_steps : 8);
        gen_params.sample_params.guidance.txt_cfg      = args.cfg_scale > 0.0f ? args.cfg_scale : (caps.default_cfg_scale > 0.0f ? caps.default_cfg_scale : 1.2f);
        gen_params.sample_params.guidance.img_cfg      = gen_params.sample_params.guidance.txt_cfg;
        gen_params.sample_params.sample_method         = caps.default_sample_method;
        if (!args.sampling_method.empty()) {
            gen_params.sample_params.sample_method = str_to_sample_method(args.sampling_method.c_str());
        }
        gen_params.sample_params.scheduler             = caps.default_scheduler;
        gen_params.sample_params.flow_shift            = caps.default_flow_shift;
        gen_params.vae_tiling_params                   = tiling;
        if (!args.ref_image.empty()) {
            if (!load_sd_image_from_file(&ref_image, args.ref_image.c_str(), 0, 0, 3)) {
                std::cerr << "failed to load reference image: " << args.ref_image << "\n";
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            gen_params.ref_images = &ref_image;
            gen_params.ref_images_count = 1;
            gen_params.auto_resize_ref_image = true;
            std::cout << "loaded reference image " << ref_image.width << "x" << ref_image.height << "x" << ref_image.channel << "\n";
        }

        sd_conditioning_handle_t positive_condition = 0;
        sd_conditioning_handle_t negative_condition = 0;
        if (args.condition_handles) {
            sd_conditioning_capabilities_t condition_caps{};
            if (!sd_get_conditioning_capabilities(ctx, &condition_caps) ||
                !condition_caps.supports_text_conditioning_encode ||
                !condition_caps.supports_conditioning_handles ||
                !condition_caps.supports_sampler_conditioning_handle_input) {
                std::cerr << "conditioning handle capabilities are not available\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "conditioning_capabilities"
                      << " text_encode=" << (condition_caps.supports_text_conditioning_encode ? "true" : "false")
                      << " handles=" << (condition_caps.supports_conditioning_handles ? "true" : "false")
                      << " gpu_resident=" << (condition_caps.supports_conditioning_gpu_resident ? "true" : "false")
                      << " sampler_input=" << (condition_caps.supports_sampler_conditioning_handle_input ? "true" : "false")
                      << " reuse=" << (condition_caps.supports_conditioning_handle_reuse ? "true" : "false")
                      << "\n";

            sd_conditioning_encode_options_t condition_options;
            sd_conditioning_encode_options_init(&condition_options);
            condition_options.clip_skip = gen_params.clip_skip;
            condition_options.width = gen_params.width;
            condition_options.height = gen_params.height;

            sd_conditioning_desc_t positive_desc{};
            condition_options.cache_key_hint = "conditioning_positive";
            if (!sd_conditioning_encode_text(ctx, args.prompt.c_str(), &condition_options, &positive_condition, &positive_desc)) {
                std::cerr << "sd_conditioning_encode_text positive failed\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_conditioning_desc(ctx, "positive_condition", positive_condition);

            condition_options.force_zero_uncond = (caps.family == SD_MODEL_FAMILY_SDXL && args.negative_prompt.empty());
            condition_options.cache_key_hint = "conditioning_negative";
            sd_conditioning_desc_t negative_desc{};
            if (!sd_conditioning_encode_text(ctx, args.negative_prompt.c_str(), &condition_options, &negative_condition, &negative_desc)) {
                std::cerr << "sd_conditioning_encode_text negative failed\n";
                sd_conditioning_release(ctx, positive_condition);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_conditioning_desc(ctx, "negative_condition", negative_condition);
        }

        if (args.compare_gpu_sampler_backend_euler) {
            std::cout << "comparing CPU sampler vs experimental Euler backend sampler\n";
            sd_latent_t* cpu_sampled = sd_sample_latent(ctx, &gen_params, nullptr);
            if (cpu_sampled == nullptr) {
                std::cerr << "sd_sample_latent failed during comparison\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            std::vector<float> cpu_values;
            sd_gpu_tensor_desc_t cpu_desc{};
            if (!download_cpu_latent_as_float_tensor(ctx, cpu_sampled, cpu_values, &cpu_desc)) {
                free_sd_latent(cpu_sampled);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }

            sd_ctx_t* gpu_compare_ctx = create_context(args, false);
            if (gpu_compare_ctx == nullptr) {
                std::cerr << "gpu comparison new_sd_ctx failed\n";
                free_sd_latent(cpu_sampled);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }

            sd_gpu_handle_t gpu_sampled = 0;
            if (!sd_sample_latent_gpu(gpu_compare_ctx, &gen_params, nullptr, &gpu_sampled)) {
                std::cerr << "sd_sample_latent_gpu backend sampler failed during comparison\n";
                free_sd_ctx(gpu_compare_ctx);
                free_sd_latent(cpu_sampled);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_gpu_desc(gpu_compare_ctx, "gpu_backend_sample_latent", gpu_sampled);
            std::vector<float> gpu_values;
            sd_gpu_tensor_desc_t gpu_desc{};
            if (!download_gpu_float_tensor(gpu_compare_ctx, gpu_sampled, gpu_values, &gpu_desc)) {
                sd_gpu_handle_release(gpu_compare_ctx, gpu_sampled);
                free_sd_ctx(gpu_compare_ctx);
                free_sd_latent(cpu_sampled);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }

            const bool shape_match = same_latent_desc_shape(cpu_desc, gpu_desc);
            const bool count_match = cpu_values.size() == gpu_values.size();
            LatentDiffStats stats = compare_float_tensors(cpu_values, gpu_values);
            std::cout << "gpu_sampler_backend_euler_compare"
                      << " shape_match=" << (shape_match ? "true" : "false")
                      << " count_match=" << (count_match ? "true" : "false")
                      << " elements_cpu=" << cpu_values.size()
                      << " elements_gpu=" << gpu_values.size()
                      << " compared=" << stats.compared
                      << " nan_or_inf=" << stats.nan_or_inf
                      << " mean_abs=" << stats.mean_abs
                      << " p95_abs=" << stats.p95_abs
                      << " p99_abs=" << stats.p99_abs
                      << " max_abs=" << stats.max_abs
                      << "\n";

            sd_gpu_handle_release(gpu_compare_ctx, gpu_sampled);
            free_sd_ctx(gpu_compare_ctx);
            free_sd_latent(cpu_sampled);
            if (ref_image.data != nullptr) {
                free(ref_image.data);
            }
            if (encoded != nullptr) {
                free_sd_latent(encoded);
            }
            free_sd_ctx(ctx);

            if (!shape_match || !count_match || stats.nan_or_inf != 0 || stats.compared == 0 ||
                stats.mean_abs > 5.0e-2 || stats.p99_abs > 5.0e-1f || stats.max_abs > 5.0f) {
                std::cerr << "GPU sampler backend comparison exceeded tolerance\n";
                return 1;
            }
            return 0;
        }

        if (args.gpu_sample_output) {
            if (args.condition_handles) {
                std::cout << "calling sd_sample_latent_gpu_with_conditioning\n";
                if (!sd_sample_latent_gpu_with_conditioning(ctx, &gen_params, nullptr, positive_condition, negative_condition, &sampled_gpu_latent)) {
                    std::cerr << "sd_sample_latent_gpu_with_conditioning failed\n";
                    sd_conditioning_release(ctx, positive_condition);
                    sd_conditioning_release(ctx, negative_condition);
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                if (args.condition_handles_reuse) {
                    sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    sampled_gpu_latent = 0;

                    sd_gpu_handle_t reused_gpu_latent = 0;
                    std::cout << "calling sd_sample_latent_gpu_with_conditioning reuse\n";
                    if (!sd_sample_latent_gpu_with_conditioning(ctx, &gen_params, nullptr, positive_condition, negative_condition, &reused_gpu_latent)) {
                        std::cerr << "sd_sample_latent_gpu_with_conditioning reuse failed\n";
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                    sampled_gpu_latent = reused_gpu_latent;
                    std::cout << "conditioning_handle_reuse=true\n";
                }
            } else if (args.true_gpu_sampler_spike) {
                std::cout << "calling sd_sample_latent_gpu_true_euler_spike\n";
                if (!sd_sample_latent_gpu_true_euler_spike(ctx, &gen_params, &sampled_gpu_latent)) {
                    std::cerr << "sd_sample_latent_gpu_true_euler_spike failed\n";
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
            } else if (args.gpu_init_sample_input && !args.sample_without_init) {
                if (encoded_gpu_latent != 0) {
                    sample_init_gpu_latent = encoded_gpu_latent;
                } else if (encoded != nullptr) {
                    if (!sd_cpu_latent_upload(ctx, encoded, &sample_init_gpu_latent, nullptr)) {
                        std::cerr << "sd_cpu_latent_upload for sampler init failed\n";
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                    sample_init_gpu_latent_owned = true;
                    print_gpu_desc(ctx, "gpu_sample_init_uploaded_latent", sample_init_gpu_latent);
                }
                if (sample_init_gpu_latent == 0) {
                    std::cerr << "--gpu-init-sample-input requires an encoded or uploaded GPU latent\n";
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                std::cout << "calling sd_sample_latent_gpu_with_init_gpu\n";
                if (!sd_sample_latent_gpu_with_init_gpu(ctx, &gen_params, sample_init_gpu_latent, &sampled_gpu_latent)) {
                    std::cerr << "sd_sample_latent_gpu_with_init_gpu failed\n";
                    if (sample_init_gpu_latent_owned) sd_gpu_handle_release(ctx, sample_init_gpu_latent);
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
            } else {
                std::cout << "calling sd_sample_latent_gpu\n";
                if (!sd_sample_latent_gpu(ctx, &gen_params, args.sample_without_init ? nullptr : encoded, &sampled_gpu_latent)) {
                    std::cerr << "sd_sample_latent_gpu failed\n";
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
            }
            sd_gpu_tensor_desc_t desc{};
            if (sd_gpu_handle_get_desc(ctx, sampled_gpu_latent, &desc)) {
                print_gpu_desc(ctx, "gpu_sample_latent", sampled_gpu_latent);
            }
            if (args.download_gpu_latent || !args.gpu_latent_decode_input) {
                sd_latent_t* sampled = sd_gpu_latent_download(ctx, sampled_gpu_latent, nullptr);
                if (sampled == nullptr) {
                    std::cerr << "sd_gpu_latent_download failed\n";
                    sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (sample_init_gpu_latent_owned) sd_gpu_handle_release(ctx, sample_init_gpu_latent);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                std::cout << "downloaded sampled latent " << sampled->width << "x" << sampled->height << "x"
                          << sampled->channel << " elements=" << sampled->element_count << "\n";
                latent_to_decode = sampled;
            }
        } else {
            std::cout << "calling sd_sample_latent\n";
            sd_latent_t* sampled = sd_sample_latent(ctx, &gen_params, args.sample_without_init ? nullptr : encoded);
            if (sampled == nullptr) {
                std::cerr << "sd_sample_latent failed\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "sampled latent " << sampled->width << "x" << sampled->height << "x"
                      << sampled->channel << " elements=" << sampled->element_count << "\n";
            latent_to_decode = sampled;
        }
        if (ref_image.data != nullptr) {
            free(ref_image.data);
        }
        if (positive_condition != 0) {
            sd_conditioning_release(ctx, positive_condition);
        }
        if (negative_condition != 0) {
            sd_conditioning_release(ctx, negative_condition);
        }
    }
    if (sample_init_gpu_latent_owned) {
        sd_gpu_handle_release(ctx, sample_init_gpu_latent);
        sample_init_gpu_latent = 0;
    }

    if (args.decode) {
        sd_ctx_t* decode_ctx = ctx;
        if (args.split_decode_context) {
            decode_ctx = create_context(args, true);
            if (decode_ctx == nullptr) {
                std::cerr << "decode new_sd_ctx failed\n";
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
        }

        if (!args.skip_estimate) {
            sd_vae_memory_report_t decode_estimate;
            uint32_t decode_width = latent_to_decode != nullptr ? latent_to_decode->width * 8 : static_cast<uint32_t>(args.width > 0 ? args.width : 1024);
            uint32_t decode_height = latent_to_decode != nullptr ? latent_to_decode->height * 8 : static_cast<uint32_t>(args.height > 0 ? args.height : 1024);
            if (sd_estimate_vae_normal_memory(decode_ctx, decode_width, decode_height, true, &vae_options, &decode_estimate)) {
                print_vae_report("decode_estimate", decode_estimate);
            }
        }

        std::cout << "calling sd_decode_latent\n";
        sd_vae_memory_report_t decode_report;
        if (args.gpu_decode_output) {
            sd_gpu_handle_t gpu_image = 0;
            sd_gpu_handle_t uploaded_gpu_latent = 0;
            bool decode_ok = false;
            if (args.gpu_latent_decode_input) {
                sd_gpu_handle_t decode_gpu_latent = sampled_gpu_latent != 0 ? sampled_gpu_latent : encoded_gpu_latent;
                if (decode_gpu_latent == 0 && args.gpu_upload_latent_decode_input && latent_to_decode != nullptr) {
                    if (!sd_cpu_latent_upload(ctx, latent_to_decode, &uploaded_gpu_latent, nullptr)) {
                        std::cerr << "sd_cpu_latent_upload failed\n";
                    } else {
                        decode_gpu_latent = uploaded_gpu_latent;
                        print_gpu_desc(ctx, "gpu_uploaded_latent", uploaded_gpu_latent);
                    }
                }
                if (decode_gpu_latent == 0) {
                    std::cerr << "--gpu-latent-decode-input requires a GPU sampled, encoded, or uploaded latent in this smoke\n";
                } else {
                    print_gpu_desc(decode_ctx, "gpu_decode_input_latent", decode_gpu_latent);
                    decode_ok = sd_decode_gpu_latent_normal_gpu(decode_ctx, decode_gpu_latent, &vae_options, &gpu_image, &decode_report);
                }
            } else {
                decode_ok = sd_decode_latent_normal_gpu(decode_ctx, latent_to_decode, &vae_options, &gpu_image, &decode_report);
            }
            if (!decode_ok) {
                if (args.expect_decode_refusal) {
                    std::cout << "expected_decode_refusal=true\n";
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 0;
                }
                std::cerr << "sd_decode_latent_normal_gpu failed\n";
                if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_vae_report("decode_report", decode_report);
            std::cout << "gpu_decode_handle=" << gpu_image << "\n";
            if (args.dump_gpu_handle_desc) {
                sd_gpu_tensor_desc_t desc{};
                if (!sd_gpu_handle_get_desc(decode_ctx, gpu_image, &desc)) {
                    std::cerr << "sd_gpu_handle_get_desc failed\n";
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                print_gpu_desc(decode_ctx, "gpu_image", gpu_image);
            }
            sd_cuda_borrowed_ptr_t borrowed{};
            if (sd_gpu_handle_borrow_cuda_ptr(decode_ctx, gpu_image, &borrowed)) {
                std::cout << "gpu_borrow device_ptr=" << borrowed.device_ptr
                          << " bytes=" << borrowed.byte_size
                          << " dtype=" << borrowed.dtype
                          << " layout=" << borrowed.layout << "\n";
                sd_gpu_handle_end_cuda_borrow(decode_ctx, gpu_image);
            }
            if (args.download_gpu_output) {
                sd_image_t output{};
                if (!sd_gpu_image_download(decode_ctx, gpu_image, &output, nullptr)) {
                    std::cerr << "sd_gpu_image_download failed\n";
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                if (!write_image_to_file(args.output, output.data, output.width, output.height, output.channel)) {
                    std::cerr << "failed to write output: " << args.output << "\n";
                    sd_free_downloaded_image(output.data);
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                std::cout << "explicit_download_wrote " << args.output << "\n";
                sd_free_downloaded_image(output.data);
            }
            if (args.download_gpu_output_buffer) {
                sd_gpu_tensor_desc_t desc{};
                if (!sd_gpu_handle_get_desc(decode_ctx, gpu_image, &desc) || desc.w <= 0 || desc.h <= 0) {
                    std::cerr << "sd_gpu_handle_get_desc for buffer download failed\n";
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                const uint64_t stride = static_cast<uint64_t>(desc.w) * 4u;
                const uint64_t bytes = stride * static_cast<uint64_t>(desc.h);
                std::vector<uint8_t> rgba(static_cast<size_t>(bytes));
                if (!sd_gpu_image_download_to_buffer(decode_ctx, gpu_image, rgba.data(), bytes, stride, nullptr)) {
                    std::cerr << "sd_gpu_image_download_to_buffer failed\n";
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                if (!write_image_to_file(args.output, rgba.data(), static_cast<uint32_t>(desc.w), static_cast<uint32_t>(desc.h), 4)) {
                    std::cerr << "failed to write caller-owned output: " << args.output << "\n";
                    sd_gpu_handle_release(decode_ctx, gpu_image);
                    if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                    if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                    if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                    if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                std::cout << "caller_owned_download_wrote " << args.output
                          << " bytes=" << bytes
                          << " stride=" << stride << "\n";
            }
            if (!sd_gpu_handle_release(decode_ctx, gpu_image)) {
                std::cerr << "sd_gpu_handle_release failed\n";
                if (uploaded_gpu_latent != 0) sd_gpu_handle_release(ctx, uploaded_gpu_latent);
                if (sampled_gpu_latent != 0) sd_gpu_handle_release(ctx, sampled_gpu_latent);
                if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            if (uploaded_gpu_latent != 0) {
                sd_gpu_handle_release(ctx, uploaded_gpu_latent);
            }
        } else {
            sd_image_t* output = sd_decode_latent_normal(decode_ctx, latent_to_decode, &vae_options, &decode_report);
            if (output == nullptr) {
                std::cerr << "sd_decode_latent failed\n";
                if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_vae_report("decode_report", decode_report);
            if (!write_image_to_file(args.output, output->data, output->width, output->height, output->channel)) {
                std::cerr << "failed to write output: " << args.output << "\n";
                free_sd_image(output);
                if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
                if (latent_to_decode != encoded) free_sd_latent(latent_to_decode);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "wrote " << args.output << "\n";
            free_sd_image(output);
        }
        if (decode_ctx != ctx) free_sd_ctx(decode_ctx);
    }

    if (args.preview_tae) {
        std::cout << "preview_count=" << preview_capture.count << "\n";
        sd_preview_options_t disabled;
        sd_preview_options_init(&disabled);
        sd_set_preview_callback_v2(nullptr, &disabled, nullptr);
    }

    if (latent_to_decode != encoded) {
        free_sd_latent(latent_to_decode);
    }
    if (sampled_gpu_latent != 0) {
        sd_gpu_handle_release(ctx, sampled_gpu_latent);
    }
    if (encoded_gpu_latent != 0) {
        sd_gpu_handle_release(ctx, encoded_gpu_latent);
    }
    if (encoded != nullptr) {
        free_sd_latent(encoded);
    }
    free_sd_ctx(ctx);
    return 0;
}
