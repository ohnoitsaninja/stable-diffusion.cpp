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
    std::string clip_l;
    std::string t5xxl;
    std::string llm;
    std::string image;
    std::string ref_image;
    std::string output = "latent_smoke.png";
    std::string prompt = "a detailed fantasy orc portrait, high quality";
    std::string negative_prompt = "blurry, low quality, noisy";
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
    bool disable_default_vae_conv_direct = false;
    bool type_f16 = false;
    bool gpu_sample_output = false;
    bool gpu_encode_output = false;
    bool gpu_upload_latent_decode_input = false;
    bool gpu_latent_decode_input = false;
    bool download_gpu_latent = false;
    bool gpu_decode_output = false;
    bool download_gpu_output = false;
    bool download_gpu_output_buffer = false;
    bool strict_gpu_resident = false;
    bool dump_gpu_handle_desc = false;
    bool expect_gpu_encode_refusal = false;
    bool expect_decode_refusal = false;
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " (--model <model> | --diffusion-model <path> --vae <path> ...) --image <image> [options]\n"
        << "Options:\n"
        << "  --diffusion-model <path> split diffusion model path for Flux/Z/Wan style contexts\n"
        << "  --vae <path>             external VAE/AE path for split model contexts\n"
        << "  --clip-l <path>          CLIP-L text encoder path\n"
        << "  --t5xxl <path>           T5XXL text encoder path\n"
        << "  --llm <path>             LLM text encoder path\n"
        << "  --output <path>          decoded output path (default: latent_smoke.png)\n"
        << "  --prompt <text>          prompt for sampler smoke\n"
        << "  --negative-prompt <text> negative prompt for sampler smoke\n"
        << "  --steps <int>            sampler steps override\n"
        << "  --cfg-scale <float>      text/image CFG override\n"
        << "  --image-channels <3|4>   channel count to load and pass into sd_encode_image (default: 4)\n"
        << "  --ref-image <path>       reference image for edit/reference-conditioning smoke\n"
        << "  --width <int>            optional target width for sample/decode path\n"
        << "  --height <int>           optional target height for sample/decode path\n"
        << "  --vae-conv-direct        enable direct VAE convolution\n"
        << "  --disable-default-vae-conv-direct disable CUDA SDXL default direct VAE convolution\n"
        << "  --type-f16               request f16 model tensor conversion\n"
        << "  --sample                 run sd_sample_latent after encode\n"
        << "  --sample-without-init    run sampler with no init latent and skip VAE encode\n"
        << "  --gpu-sample-output      run sd_sample_latent_gpu and keep sampled latent as a GPU handle\n"
        << "  --gpu-encode-output      call sd_encode_image_normal_gpu and keep encoded latent as a GPU handle\n"
        << "  --gpu-upload-latent-decode-input upload the CPU latent, then decode from the GPU latent handle\n"
        << "  --gpu-latent-decode-input decode from a GPU latent handle with sd_decode_gpu_latent_normal_gpu\n"
        << "  --download-gpu-latent    explicitly download the sampled GPU latent handle\n"
        << "  --skip-estimate          skip sd_estimate_vae_normal_memory smoke checks\n"
        << "  --split-decode-context   decode with a separate vae_decode_only=true context\n"
        << "  --gpu-decode-output      call sd_decode_latent_normal_gpu and keep decode output as a GPU handle\n"
        << "  --download-gpu-output    explicitly download the GPU image handle and write it\n"
        << "  --download-gpu-output-buffer download GPU image directly into caller-owned RGBA8 memory\n"
        << "  --strict-gpu-resident    set SDCPP_STRICT_GPU_RESIDENT=1 for GPU-output checks\n"
        << "  --dump-gpu-handle-desc   print GPU handle descriptor after decode\n"
        << "  --expect-gpu-encode-refusal treat sd_encode_image_normal_gpu refusal as a passing smoke result\n"
        << "  --expect-decode-refusal  treat GPU decode refusal as a passing smoke result\n"
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
        } else if (arg == "--strict-gpu-resident") {
            args.strict_gpu_resident = true;
        } else if (arg == "--dump-gpu-handle-desc") {
            args.dump_gpu_handle_desc = true;
        } else if (arg == "--expect-gpu-encode-refusal") {
            args.expect_gpu_encode_refusal = true;
        } else if (arg == "--expect-decode-refusal") {
            args.expect_decode_refusal = true;
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
    ctx_params.clip_l_path           = args.clip_l.empty() ? nullptr : args.clip_l.c_str();
    ctx_params.t5xxl_path            = args.t5xxl.empty() ? nullptr : args.t5xxl.c_str();
    ctx_params.llm_path              = args.llm.empty() ? nullptr : args.llm.c_str();
    ctx_params.vae_decode_only       = vae_decode_only;
    ctx_params.diffusion_flash_attn  = true;
    ctx_params.vae_conv_direct       = args.vae_conv_direct;
    ctx_params.wtype                 = args.type_f16 ? SD_TYPE_F16 : SD_TYPE_COUNT;
    ctx_params.offload_params_to_cpu = false;
    ctx_params.keep_clip_on_cpu      = false;
    ctx_params.keep_vae_on_cpu       = false;

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

static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)data;
    log_print(level, log, true, false);
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
    sd_set_log_callback(sd_log_cb, nullptr);

    sd_ctx_t* ctx = create_context(args, false);
    if (ctx == nullptr) {
        std::cerr << "new_sd_ctx failed\n";
        return 1;
    }
    print_model_capabilities(ctx);

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

        if (args.gpu_sample_output) {
            std::cout << "calling sd_sample_latent_gpu\n";
            if (!sd_sample_latent_gpu(ctx, &gen_params, args.sample_without_init ? nullptr : encoded, &sampled_gpu_latent)) {
                std::cerr << "sd_sample_latent_gpu failed\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
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
