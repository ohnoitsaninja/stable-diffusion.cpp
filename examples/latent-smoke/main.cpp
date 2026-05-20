#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/media_io.h"
#include "stable-diffusion.h"
#include "../../src/rng_mt19937.hpp"

struct Args {
    std::string model;
    std::string diffusion_model;
    std::string vae;
    std::string taesd;
    std::string clip_l;
    std::string t5xxl;
    std::string llm;
    std::string llm_vision;
    std::string image;
    std::string ref_image;
    std::string conditioning_ref_image;
    std::string import_noise_npy;
    std::vector<std::string> import_step_noise_npys;
    std::string compare_comfy_cpu_noise_npy;
    std::string output = "latent_smoke.png";
    std::string prompt = "a detailed fantasy orc portrait, high quality";
    std::string negative_prompt = "blurry, low quality, noisy";
    std::string model_family;
    std::string sampling_method;
    int image_channels = 4;
    int width          = 0;
    int height         = 0;
    int steps          = 0;
    int64_t seed       = 12345;
    rng_type_t rng_type = RNG_TYPE_COUNT;
    rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
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
    bool gpu_flow_sampler = false;
    bool flux2_text_encoder_cpu_params = false;
    bool z_image_text_encoder_cpu_params = false;
    bool qwen_image_text_encoder_cpu_params = false;
    bool qwen_image_zero_cond_t = false;
    bool anima_text_encoder_cpu_params = false;
    bool capabilities_only = false;
    bool compare_gpu_sampler_backend_euler = false;
    bool gpu_init_sample_input = false;
    bool gpu_encode_output = false;
    bool prewarm_decode_bridge = false;
    bool compare_roundtrip_input = false;
    bool gpu_upload_latent_decode_input = false;
    bool gpu_latent_decode_input = false;
    bool download_gpu_latent = false;
    bool gpu_decode_output = false;
    bool decode_twice = false;
    bool download_gpu_output = false;
    bool download_gpu_output_buffer = false;
    bool condition_handles = false;
    bool condition_handles_reuse = false;
    bool condition_only = false;
    bool release_text_encoder_after_conditioning = false;
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
        << "  --llm-vision <path>      LLM vision/mmproj path for Qwen Image edit/reference models\n"
        << "  --output <path>          decoded output path (default: latent_smoke.png)\n"
        << "  --prompt <text>          prompt for sampler smoke\n"
        << "  --negative-prompt <text> negative prompt for sampler smoke\n"
        << "  --model-family <name>   optional expected family label for capability checks, e.g. flux2\n"
        << "  --steps <int>            sampler steps override\n"
        << "  --seed <int>             RNG seed override (default: 12345)\n"
        << "  --rng <name>             context RNG, e.g. cuda or cpu\n"
        << "  --sampler-rng <name>     sampler RNG; cpu selects Comfy-compatible CPU torch initial noise in GPU backend\n"
        << "  --cfg-scale <float>      text/image CFG override\n"
        << "  --sampling-method <name> sampler method override\n"
        << "  --image-channels <3|4>   channel count to load and pass into sd_encode_image (default: 4)\n"
        << "  --ref-image <path>       reference image for edit/reference-conditioning smoke\n"
        << "  --conditioning-ref-image <path> reference image used only for conditioning handle encoding\n"
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
        << "  --gpu-sampler-backend use experimental backend sampler through sd_sample_latent_gpu (SDXL/SD1 sampler subset)\n"
        << "  --gpu-flow-sampler   enable env-gated Flux2/Z-Image backend flow sampler lane\n"
        << "  --flux2-text-encoder-cpu-params keep Flux2 Qwen params in RAM and execute on GPU only during encode\n"
        << "  --z-image-text-encoder-cpu-params keep Z-Image Qwen params in RAM and execute on GPU only during encode\n"
        << "  --qwen-image-text-encoder-cpu-params keep Qwen-Image LLM params in RAM and execute on GPU only during encode\n"
        << "  --qwen-image-zero-cond-t enable Qwen Image Edit 2511 zero cond-t compatibility flag\n"
        << "  --anima-text-encoder-cpu-params keep Anima Qwen params in RAM and execute on GPU only during encode\n"
        << "  --capabilities-only  create the context, print capabilities, then exit before image load/sample/decode\n"
        << "  --gpu-sampler-backend-euler legacy alias for --gpu-sampler-backend\n"
        << "  --compare-gpu-sampler-backend-euler compare CPU sampler latent vs experimental Euler backend latent\n"
        << "  --gpu-init-sample-input  pass a GPU latent handle into the sampler init-latent bridge API\n"
        << "  --gpu-encode-output      call sd_encode_image_normal_gpu and keep encoded latent as a GPU handle\n"
        << "  --prewarm-decode-bridge  create/cache isolated VAE decode context before decode\n"
        << "  --compare-roundtrip-input compare downloaded output RGB against the input image\n"
        << "  --gpu-upload-latent-decode-input upload the CPU latent, then decode from the GPU latent handle\n"
        << "  --gpu-latent-decode-input decode from a GPU latent handle with sd_decode_gpu_latent_normal_gpu\n"
        << "  --download-gpu-latent    explicitly download the sampled GPU latent handle\n"
        << "  --skip-estimate          skip sd_estimate_vae_normal_memory smoke checks\n"
        << "  --split-decode-context   decode with a separate vae_decode_only=true context\n"
        << "  --gpu-decode-output      call sd_decode_latent_normal_gpu and keep decode output as a GPU handle\n"
        << "  --decode-twice           decode the same GPU latent twice to test cached decode context reuse\n"
        << "  --download-gpu-output    explicitly download the GPU image handle and write it\n"
        << "  --download-gpu-output-buffer download GPU image directly into caller-owned RGBA8 memory\n"
        << "  --condition-handles      encode prompt/negative into resident conditioning handles and sample with them\n"
        << "                           combine with --gpu-init-sample-input to test I2I conditioning-handle sampler bridge\n"
        << "  --condition-only         encode conditioning handles, print descriptors, then exit before sampling/decode\n"
        << "  --condition-handles-reuse sample twice with the same conditioning handles to prove CLIP encode is not rerun\n"
        << "  --release-text-encoder-after-conditioning release text encoder params after conditioning handles are resident\n"
        << "  --import-noise-npy <path> import f32 NCHW noise .npy as a GPU latent for parity/debug sampling\n"
        << "  --import-step-noise-npy <path> import one f32 NCHW per-step noise .npy as a GPU latent; repeat for SDE/Brownian parity\n"
        << "  --compare-comfy-cpu-noise-npy <path> compare sd.cpp MT19937/PyTorch CPU noise against an f32 NCHW .npy, then exit\n"
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
        } else if (arg == "--llm-vision" || arg == "--llm_vision" || arg == "--qwen2vl-vision" || arg == "--qwen2vl_vision") {
            const char* value = need_value(arg.c_str());
            if (value == nullptr) return false;
            args.llm_vision = value;
        } else if (arg == "--image") {
            const char* value = need_value("--image");
            if (value == nullptr) return false;
            args.image = value;
        } else if (arg == "--ref-image") {
            const char* value = need_value("--ref-image");
            if (value == nullptr) return false;
            args.ref_image = value;
        } else if (arg == "--conditioning-ref-image") {
            const char* value = need_value("--conditioning-ref-image");
            if (value == nullptr) return false;
            args.conditioning_ref_image = value;
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
        } else if (arg == "--model-family") {
            const char* value = need_value("--model-family");
            if (value == nullptr) return false;
            args.model_family = value;
        } else if (arg == "--steps") {
            const char* value = need_value("--steps");
            if (value == nullptr) return false;
            args.steps = std::atoi(value);
        } else if (arg == "--seed") {
            const char* value = need_value("--seed");
            if (value == nullptr) return false;
            args.seed = std::strtoll(value, nullptr, 10);
        } else if (arg == "--rng") {
            const char* value = need_value("--rng");
            if (value == nullptr) return false;
            args.rng_type = str_to_rng_type(value);
            if (args.rng_type == RNG_TYPE_COUNT) {
                std::cerr << "unknown --rng value: " << value << "\n";
                return false;
            }
        } else if (arg == "--sampler-rng") {
            const char* value = need_value("--sampler-rng");
            if (value == nullptr) return false;
            args.sampler_rng_type = str_to_rng_type(value);
            if (args.sampler_rng_type == RNG_TYPE_COUNT) {
                std::cerr << "unknown --sampler-rng value: " << value << "\n";
                return false;
            }
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
        } else if (arg == "--gpu-sampler-backend" || arg == "--gpu-sampler-backend-euler") {
            args.gpu_sampler_backend_euler = true;
            args.gpu_sample_output = true;
            args.sample = true;
            args.sample_without_init = true;
        } else if (arg == "--gpu-flow-sampler") {
            args.gpu_flow_sampler = true;
            args.gpu_sample_output = true;
            args.sample = true;
        } else if (arg == "--flux2-text-encoder-cpu-params") {
            args.flux2_text_encoder_cpu_params = true;
        } else if (arg == "--z-image-text-encoder-cpu-params") {
            args.z_image_text_encoder_cpu_params = true;
        } else if (arg == "--qwen-image-text-encoder-cpu-params") {
            args.qwen_image_text_encoder_cpu_params = true;
        } else if (arg == "--qwen-image-zero-cond-t") {
            args.qwen_image_zero_cond_t = true;
        } else if (arg == "--anima-text-encoder-cpu-params") {
            args.anima_text_encoder_cpu_params = true;
        } else if (arg == "--capabilities-only") {
            args.capabilities_only = true;
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
        } else if (arg == "--prewarm-decode-bridge") {
            args.prewarm_decode_bridge = true;
        } else if (arg == "--compare-roundtrip-input") {
            args.compare_roundtrip_input = true;
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
        } else if (arg == "--decode-twice") {
            args.decode_twice = true;
        } else if (arg == "--download-gpu-output") {
            args.download_gpu_output = true;
        } else if (arg == "--download-gpu-output-buffer") {
            args.download_gpu_output_buffer = true;
        } else if (arg == "--condition-handles") {
            args.condition_handles = true;
            args.gpu_sample_output = true;
            args.sample = true;
        } else if (arg == "--condition-only") {
            args.condition_handles = true;
            args.condition_only = true;
            args.sample = true;
            args.sample_without_init = true;
            args.decode = false;
        } else if (arg == "--condition-handles-reuse") {
            args.condition_handles = true;
            args.condition_handles_reuse = true;
            args.gpu_sample_output = true;
            args.sample = true;
        } else if (arg == "--release-text-encoder-after-conditioning") {
            args.release_text_encoder_after_conditioning = true;
        } else if (arg == "--import-noise-npy") {
            const char* value = need_value("--import-noise-npy");
            if (value == nullptr) return false;
            args.import_noise_npy = value;
        } else if (arg == "--import-step-noise-npy") {
            const char* value = need_value("--import-step-noise-npy");
            if (value == nullptr) return false;
            args.import_step_noise_npys.push_back(value);
        } else if (arg == "--compare-comfy-cpu-noise-npy") {
            const char* value = need_value("--compare-comfy-cpu-noise-npy");
            if (value == nullptr) return false;
            args.compare_comfy_cpu_noise_npy = value;
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

    if (!args.compare_comfy_cpu_noise_npy.empty()) {
        return true;
    }
    if (args.model.empty() && args.diffusion_model.empty()) {
        return false;
    }
    if (!args.capabilities_only && args.image.empty()) {
        return false;
    }
    if (args.gpu_init_sample_input) {
        args.sample_without_init = false;
    }
    if (args.image_channels != 3 && args.image_channels != 4) {
        std::cerr << "--image-channels must be 3 or 4\n";
        return false;
    }
    if ((!args.import_noise_npy.empty() || !args.import_step_noise_npys.empty()) && !args.condition_handles) {
        std::cerr << "--import-noise-npy/--import-step-noise-npy currently require --condition-handles so the imported-noise sampler API can be exercised\n";
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

static bool load_f32_npy(const std::string& path, std::vector<float>& data, std::vector<int64_t>& shape) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "failed to open npy: " << path << "\n";
        return false;
    }
    char magic[6] = {};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "not an npy file: " << path << "\n";
        return false;
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    in.read(reinterpret_cast<char*>(&major), sizeof(major));
    in.read(reinterpret_cast<char*>(&minor), sizeof(minor));
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), sizeof(len16));
        header_len = len16;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        std::cerr << "unsupported npy version: " << static_cast<int>(major) << "." << static_cast<int>(minor) << "\n";
        return false;
    }
    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in.good()) {
        std::cerr << "failed to read npy header: " << path << "\n";
        return false;
    }
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos &&
        header.find("'descr': '|f4'") == std::string::npos &&
        header.find("\"descr\": \"|f4\"") == std::string::npos) {
        std::cerr << "npy must be f32 little-endian: " << header << "\n";
        return false;
    }
    if (header.find("True") != std::string::npos) {
        std::cerr << "Fortran-order npy is not supported: " << header << "\n";
        return false;
    }
    const size_t open = header.find('(');
    const size_t close = header.find(')', open == std::string::npos ? 0 : open + 1);
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        std::cerr << "npy shape missing: " << header << "\n";
        return false;
    }
    std::string shape_text = header.substr(open + 1, close - open - 1);
    shape.clear();
    std::stringstream ss(shape_text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t first = token.find_first_not_of(" \t\r\n");
        size_t last = token.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        int64_t dim = std::strtoll(token.substr(first, last - first + 1).c_str(), nullptr, 10);
        if (dim <= 0) {
            std::cerr << "invalid npy shape dim: " << token << "\n";
            return false;
        }
        shape.push_back(dim);
    }
    if (shape.empty()) {
        std::cerr << "empty npy shape: " << header << "\n";
        return false;
    }
    uint64_t elements = 1;
    for (int64_t dim : shape) {
        elements *= static_cast<uint64_t>(dim);
    }
    data.resize(static_cast<size_t>(elements));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(elements * sizeof(float)));
    if (!in.good()) {
        std::cerr << "failed to read npy data: " << path << "\n";
        return false;
    }
    return true;
}

static sd_gpu_handle_t upload_noise_npy_as_gpu_latent(sd_ctx_t* ctx, const std::string& path) {
    std::vector<float> data;
    std::vector<int64_t> shape;
    if (!load_f32_npy(path, data, shape)) {
        return 0;
    }
    if (shape.size() != 4 || shape[0] != 1) {
        std::cerr << "noise npy must have shape NCHW with N=1, got [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) std::cerr << ", ";
            std::cerr << shape[i];
        }
        std::cerr << "]\n";
        return 0;
    }
    const uint32_t c = static_cast<uint32_t>(shape[1]);
    const uint32_t h = static_cast<uint32_t>(shape[2]);
    const uint32_t w = static_cast<uint32_t>(shape[3]);
    sd_latent_t* latent = sd_latent_import_f32(data.data(), data.size(), w, h, c);
    if (latent == nullptr) {
        std::cerr << "sd_latent_import_f32 failed for noise npy\n";
        return 0;
    }
    sd_gpu_handle_t handle = 0;
    if (!sd_cpu_latent_upload(ctx, latent, &handle, nullptr)) {
        std::cerr << "sd_cpu_latent_upload failed for imported noise\n";
        free_sd_latent(latent);
        return 0;
    }
    free_sd_latent(latent);
    std::cout << "imported_noise_gpu_handle=" << handle
              << " source=\"" << path << "\""
              << " shape_nchw=1x" << c << "x" << h << "x" << w
              << " elements=" << data.size() << "\n";
    return handle;
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
    ctx_params.llm_vision_path       = args.llm_vision.empty() ? nullptr : args.llm_vision.c_str();
    ctx_params.vae_decode_only       = vae_decode_only;
    ctx_params.diffusion_flash_attn  = true;
    ctx_params.diffusion_conv_direct = args.diffusion_conv_direct;
    ctx_params.vae_conv_direct       = args.vae_conv_direct;
    ctx_params.tae_preview_only      = args.preview_tae && !args.taesd.empty();
    ctx_params.wtype                 = args.type_f16 ? SD_TYPE_F16 : SD_TYPE_COUNT;
    if (args.rng_type != RNG_TYPE_COUNT) {
        ctx_params.rng_type = args.rng_type;
    }
    if (args.sampler_rng_type != RNG_TYPE_COUNT) {
        ctx_params.sampler_rng_type = args.sampler_rng_type;
    }
    ctx_params.offload_params_to_cpu = false;
    ctx_params.keep_clip_on_cpu      = false;
    ctx_params.qwen_image_zero_cond_t = args.qwen_image_zero_cond_t;
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
              << " decode_setup_ms=" << report.decode_setup_ms
              << " decode_context_ms=" << report.decode_context_ms
              << " decode_latent_d2d_ms=" << report.decode_latent_d2d_ms
              << " decode_graph_ms=" << report.decode_graph_ms
              << " decode_image_d2d_ms=" << report.decode_image_d2d_ms
              << " decode_download_ms=" << report.decode_download_ms
              << " decode_context_reuse=" << report.decode_context_reuse
              << " decode_same_context_attempted=" << report.decode_same_context_attempted
              << " decode_same_context_succeeded=" << report.decode_same_context_succeeded
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

static void print_rgb_diff_against_input(const char* label,
                                         const std::vector<uint8_t>& reference,
                                         uint32_t reference_width,
                                         uint32_t reference_height,
                                         uint32_t reference_channels,
                                         const uint8_t* candidate,
                                         uint32_t candidate_width,
                                         uint32_t candidate_height,
                                         uint32_t candidate_channels) {
    if (reference.empty() || candidate == nullptr ||
        reference_width != candidate_width ||
        reference_height != candidate_height ||
        reference_channels < 3 ||
        candidate_channels < 3) {
        std::cout << label << " skipped=true reason=shape_or_channel_mismatch"
                  << " reference=" << reference_width << "x" << reference_height << "x" << reference_channels
                  << " candidate=" << candidate_width << "x" << candidate_height << "x" << candidate_channels << "\n";
        return;
    }

    const uint64_t pixels = static_cast<uint64_t>(reference_width) * reference_height;
    double abs_sum = 0.0;
    double mse_sum = 0.0;
    uint32_t max_abs = 0;
    for (uint64_t i = 0; i < pixels; ++i) {
        const uint64_t ref_base = i * reference_channels;
        const uint64_t cand_base = i * candidate_channels;
        for (uint32_t c = 0; c < 3; ++c) {
            const int diff = static_cast<int>(candidate[cand_base + c]) -
                             static_cast<int>(reference[ref_base + c]);
            const uint32_t abs_diff = static_cast<uint32_t>(std::abs(diff));
            abs_sum += static_cast<double>(abs_diff);
            mse_sum += static_cast<double>(diff * diff);
            max_abs = std::max(max_abs, abs_diff);
        }
    }

    const double samples = static_cast<double>(pixels) * 3.0;
    const double mean_abs = samples > 0.0 ? abs_sum / samples : 0.0;
    const double mse = samples > 0.0 ? mse_sum / samples : 0.0;
    const double psnr = mse > 0.0 ? 20.0 * std::log10(255.0 / std::sqrt(mse)) : 99.0;
    std::cout << label << " mean_abs=" << mean_abs
              << " psnr=" << psnr
              << " max_abs=" << max_abs << "\n";
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
    if (desc.kind == SD_GPU_RESOURCE_LATENT) {
        sd_latent_view_t view{};
        if (!sd_gpu_latent_export_f32_nchw_debug(ctx, handle, out.data(), out.size(), &view, nullptr)) {
            std::cerr << "sd_gpu_latent_export_f32_nchw_debug failed for handle=" << handle << "\n";
            return false;
        }
        std::cout << "debug_export_gpu_latent handle=" << handle
                  << " shape_nchw=" << view.n << "x" << view.c << "x" << view.h << "x" << view.w
                  << " elements=" << view.element_count
                  << " flags=" << view.flags << "\n";
    } else if (!sd_gpu_tensor_download(ctx, handle, out.data(), desc.byte_size, nullptr)) {
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
    sd_latent_view_t view{};
    if (!sd_latent_get_view(latent, &view)) {
        std::cerr << "sd_latent_get_view failed for comparison\n";
        return false;
    }
    out.resize(static_cast<size_t>(view.element_count));
    if (!sd_latent_export_f32(latent, out.data(), out.size(), &view)) {
        std::cerr << "sd_latent_export_f32 failed for comparison\n";
        return false;
    }
    if (out_desc != nullptr) {
        *out_desc = {};
        out_desc->struct_size = sizeof(sd_gpu_tensor_desc_t);
        out_desc->version = SD_VAE_API_VERSION;
        out_desc->kind = SD_GPU_RESOURCE_LATENT;
        out_desc->backend = SD_BACKEND_CPU;
        out_desc->dtype = SD_DTYPE_F32;
        out_desc->layout = SD_LAYOUT_NCHW;
        out_desc->n = view.n;
        out_desc->c = view.c;
        out_desc->h = view.h;
        out_desc->w = view.w;
        out_desc->stride_n = view.stride_n;
        out_desc->stride_c = view.stride_c;
        out_desc->stride_h = view.stride_h;
        out_desc->stride_w = view.stride_w;
        out_desc->byte_size = view.byte_size;
        out_desc->flags = view.flags;
    }
    std::cout << "debug_export_cpu_latent shape_nchw=" << view.n << "x" << view.c << "x" << view.h << "x" << view.w
              << " elements=" << view.element_count << "\n";
    return true;
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
              << " gpu_latent_decode_bridge=" << (caps.supports_gpu_latent_decode_bridge ? "true" : "false")
              << " gpu_image_output=" << (caps.supports_gpu_image_output ? "true" : "false")
              << " gpu_image_output_bridge=" << (caps.supports_gpu_image_output_bridge ? "true" : "false")
              << " gpu_vae_encode=" << (caps.supports_vae_encode_gpu_output ? "true" : "false")
              << " reference_images=" << (caps.supports_reference_images ? "true" : "false")
              << " edit_mode=" << (caps.supports_edit_mode ? "true" : "false")
              << " edit_reference_conditioning=" << (caps.supports_edit_reference_conditioning ? "true" : "false")
              << " comfy_reference_vae_encode=" << (caps.supports_comfy_reference_vae_encode ? "true" : "false")
              << " strict_sampler_true_resident=" << (caps.strict_gpu_sample_is_true_resident ? "true" : "false")
              << " flux2_model_load=" << (caps.supports_flux2_model_load ? "true" : "false")
              << " flux2_qwen_conditioning=" << (caps.supports_flux2_qwen_conditioning ? "true" : "false")
              << " flux2_qwen_conditioning_gpu=" << (caps.supports_flux2_qwen_conditioning_gpu_resident ? "true" : "false")
              << " flux2_flow_backend_sampler=" << (caps.supports_flux2_flow_backend_sampler ? "true" : "false")
              << " flux2_gpu_latent_output=" << (caps.supports_flux2_gpu_latent_output ? "true" : "false")
              << " flux2_vae_decode_gpu=" << (caps.supports_flux2_vae_decode_gpu ? "true" : "false")
              << " flux2_controlnet=" << (caps.supports_flux2_controlnet ? "true" : "false")
              << " flux2_masks=" << (caps.supports_flux2_masks ? "true" : "false")
              << " flux2_reference=" << (caps.supports_flux2_reference ? "true" : "false")
              << " flux2_edit=" << (caps.supports_flux2_edit ? "true" : "false")
              << " flux2_multibatch=" << (caps.supports_flux2_multibatch ? "true" : "false")
              << " z_image_model_load=" << (caps.supports_z_image_model_load ? "true" : "false")
              << " z_image_qwen_conditioning=" << (caps.supports_z_image_qwen_conditioning ? "true" : "false")
              << " z_image_qwen_conditioning_gpu=" << (caps.supports_z_image_qwen_conditioning_gpu_resident ? "true" : "false")
              << " z_image_flow_backend_sampler=" << (caps.supports_z_image_flow_backend_sampler ? "true" : "false")
              << " z_image_gpu_latent_output=" << (caps.supports_z_image_gpu_latent_output ? "true" : "false")
              << " z_image_vae_decode_gpu=" << (caps.supports_z_image_vae_decode_gpu ? "true" : "false")
              << " z_image_controlnet=" << (caps.supports_z_image_controlnet ? "true" : "false")
              << " z_image_masks=" << (caps.supports_z_image_masks ? "true" : "false")
              << " z_image_reference=" << (caps.supports_z_image_reference ? "true" : "false")
              << " z_image_edit=" << (caps.supports_z_image_edit ? "true" : "false")
              << " z_image_multibatch=" << (caps.supports_z_image_multibatch ? "true" : "false")
              << " qwen_image_model_load=" << (caps.supports_qwen_image_model_load ? "true" : "false")
              << " qwen_image_qwen_conditioning=" << (caps.supports_qwen_image_qwen_conditioning ? "true" : "false")
              << " qwen_image_qwen_conditioning_gpu=" << (caps.supports_qwen_image_qwen_conditioning_gpu_resident ? "true" : "false")
              << " qwen_image_flow_backend_sampler=" << (caps.supports_qwen_image_flow_backend_sampler ? "true" : "false")
              << " qwen_image_gpu_latent_output=" << (caps.supports_qwen_image_gpu_latent_output ? "true" : "false")
              << " qwen_image_vae_decode_gpu=" << (caps.supports_qwen_image_vae_decode_gpu ? "true" : "false")
              << " qwen_image_vae_decode_bridge=" << (caps.supports_qwen_image_vae_decode_bridge ? "true" : "false")
              << " anima_vae_decode_bridge=" << (caps.supports_anima_vae_decode_bridge ? "true" : "false")
              << " qwen_image_controlnet=" << (caps.supports_qwen_image_controlnet ? "true" : "false")
              << " qwen_image_masks=" << (caps.supports_qwen_image_masks ? "true" : "false")
              << " qwen_image_reference=" << (caps.supports_qwen_image_reference ? "true" : "false")
              << " qwen_image_edit=" << (caps.supports_qwen_image_edit ? "true" : "false")
              << " qwen_image_multibatch=" << (caps.supports_qwen_image_multibatch ? "true" : "false")
              << "\n";
}

static std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream out;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << "x";
        }
        out << shape[i];
    }
    return out.str();
}

static bool compare_comfy_cpu_noise_npy(const Args& args) {
    std::vector<float> expected;
    std::vector<int64_t> shape;
    if (!load_f32_npy(args.compare_comfy_cpu_noise_npy, expected, shape)) {
        return false;
    }
    if (expected.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        std::cerr << "noise npy too large for MT19937RNG smoke: elements=" << expected.size() << "\n";
        return false;
    }

    MT19937RNG rng(static_cast<uint64_t>(args.seed));
    std::vector<float> actual = rng.randn(static_cast<uint32_t>(expected.size()));
    LatentDiffStats diff = compare_float_tensors(actual, expected);

    double mean = 0.0;
    double sq = 0.0;
    for (float v : actual) {
        mean += static_cast<double>(v);
        sq += static_cast<double>(v) * static_cast<double>(v);
    }
    mean /= std::max<size_t>(actual.size(), 1);
    const double variance = sq / std::max<size_t>(actual.size(), 1) - mean * mean;
    const double stddev = std::sqrt(std::max(0.0, variance));

    std::cout << "comfy_cpu_noise_compare"
              << " seed=" << args.seed
              << " shape=" << shape_to_string(shape)
              << " elements=" << actual.size()
              << " actual_mean=" << mean
              << " actual_std=" << stddev
              << " compared=" << diff.compared
              << " mean_abs=" << diff.mean_abs
              << " p95_abs=" << diff.p95_abs
              << " p99_abs=" << diff.p99_abs
              << " max_abs=" << diff.max_abs
              << " nan_or_inf=" << diff.nan_or_inf
              << "\n";
    constexpr float comfy_cpu_noise_tolerance = 1.0e-6f;
    const bool pass = actual.size() == expected.size() &&
                      diff.nan_or_inf == 0 &&
                      diff.max_abs <= comfy_cpu_noise_tolerance;
    std::cout << "comfy_cpu_noise_compare_result="
              << (pass ? "pass" : "fail")
              << " tolerance=" << comfy_cpu_noise_tolerance
              << "\n";
    return pass;
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
              << " sampler_imported_initial_noise=" << (caps.supports_sampler_imported_initial_noise ? "true" : "false")
              << " sampler_imported_step_noise=" << (caps.supports_sampler_imported_step_noise_schedule ? "true" : "false")
              << " sampler_brownian_step_noise_import=" << (caps.supports_sampler_brownian_step_noise_import ? "true" : "false")
              << " sampler_step_noise_count_query=" << (caps.supports_sampler_step_noise_count_query ? "true" : "false")
              << " vae_gpu_latent_input=" << (caps.supports_vae_gpu_latent_input ? "true" : "false")
              << " vae_gpu_latent_decode_bridge=" << (caps.supports_vae_gpu_latent_decode_bridge ? "true" : "false")
              << " flux2_gpu_output=" << (caps.supports_flux2_gpu_latent_output ? "true" : "false")
              << " flux2_flow_sampler=" << (caps.supports_flux2_flow_backend_sampler ? "true" : "false")
              << " flux2_vae_decode_gpu=" << (caps.supports_flux2_vae_decode_gpu ? "true" : "false")
              << " flux2_qwen_conditioning_gpu=" << (caps.supports_flux2_qwen_conditioning_gpu_resident ? "true" : "false")
              << " z_image_gpu_output=" << (caps.supports_z_image_gpu_latent_output ? "true" : "false")
              << " z_image_flow_sampler=" << (caps.supports_z_image_flow_backend_sampler ? "true" : "false")
              << " z_image_vae_decode_gpu=" << (caps.supports_z_image_vae_decode_gpu ? "true" : "false")
              << " z_image_qwen_conditioning_gpu=" << (caps.supports_z_image_qwen_conditioning_gpu_resident ? "true" : "false")
              << " qwen_image_gpu_output=" << (caps.supports_qwen_image_gpu_latent_output ? "true" : "false")
              << " qwen_image_flow_sampler=" << (caps.supports_qwen_image_flow_backend_sampler ? "true" : "false")
              << " qwen_image_vae_decode_gpu=" << (caps.supports_qwen_image_vae_decode_gpu ? "true" : "false")
              << " qwen_image_vae_decode_bridge=" << (caps.supports_qwen_image_vae_decode_bridge ? "true" : "false")
              << " anima_vae_decode_bridge=" << (caps.supports_anima_vae_decode_bridge ? "true" : "false")
              << " qwen_image_qwen_conditioning_gpu=" << (caps.supports_qwen_image_qwen_conditioning_gpu_resident ? "true" : "false")
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
        set_env_value("SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND", "1");
    }
    if (args.gpu_flow_sampler) {
        set_env_value("SDCPP_EXPERIMENTAL_FLUX2_BACKEND", "1");
        set_env_value("SDCPP_EXPERIMENTAL_Z_IMAGE_BACKEND", "1");
        set_env_value("SDCPP_EXPERIMENTAL_QWEN_IMAGE_BACKEND", "1");
        set_env_value("SDCPP_EXPERIMENTAL_ANIMA_BACKEND", "1");
    }
    if (args.flux2_text_encoder_cpu_params) {
        set_env_value("SDCPP_FLUX2_TEXT_ENCODER_CPU_PARAMS", "1");
    }
    if (args.z_image_text_encoder_cpu_params) {
        set_env_value("SDCPP_Z_IMAGE_TEXT_ENCODER_CPU_PARAMS", "1");
    }
    if (args.qwen_image_text_encoder_cpu_params) {
        set_env_value("SDCPP_QWEN_IMAGE_TEXT_ENCODER_CPU_PARAMS", "1");
    }
    if (args.anima_text_encoder_cpu_params) {
        set_env_value("SDCPP_ANIMA_TEXT_ENCODER_CPU_PARAMS", "1");
    }
    sd_set_log_callback(sd_log_cb, nullptr);

    if (!args.compare_comfy_cpu_noise_npy.empty()) {
        return compare_comfy_cpu_noise_npy(args) ? 0 : 1;
    }

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
    if (!args.model_family.empty()) {
        sd_model_pipeline_capabilities_t expected_caps{};
        if (!sd_get_model_pipeline_capabilities(ctx, &expected_caps) ||
            args.model_family != expected_caps.family_name) {
            std::cerr << "--model-family expected " << args.model_family
                      << " but context reported "
                      << (expected_caps.family_name[0] != '\0' ? expected_caps.family_name : "<unknown>")
                      << "\n";
            free_sd_ctx(ctx);
            return 1;
        }
    }
    if (args.capabilities_only) {
        free_sd_ctx(ctx);
        return 0;
    }

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
    std::vector<uint8_t> roundtrip_reference;
    uint32_t roundtrip_reference_width = image.width;
    uint32_t roundtrip_reference_height = image.height;
    uint32_t roundtrip_reference_channels = image.channel;
    if (args.compare_roundtrip_input) {
        const size_t bytes = static_cast<size_t>(image.width) *
                             static_cast<size_t>(image.height) *
                             static_cast<size_t>(image.channel);
        roundtrip_reference.assign(image.data, image.data + bytes);
    }

    if (args.marigold_iid) {
        sd_marigold_iid_options_t iid_options;
        sd_marigold_iid_options_init(&iid_options);
        iid_options.processing_width = args.width > 0 ? static_cast<uint32_t>(args.width) : 0;
        iid_options.processing_height = args.height > 0 ? static_cast<uint32_t>(args.height) : 0;
        iid_options.steps = args.steps > 0 ? static_cast<uint32_t>(args.steps) : 4;
        iid_options.seed = args.seed;
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
    if (args.prewarm_decode_bridge) {
        sd_vae_memory_report_t prewarm_report;
        if (!sd_prewarm_vae_decode_bridge(ctx, &vae_options, &prewarm_report)) {
            std::cerr << "sd_prewarm_vae_decode_bridge failed\n";
            if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
            if (encoded != nullptr) free_sd_latent(encoded);
            free(image.data);
            free_sd_ctx(ctx);
            return 1;
        }
        print_vae_report("prewarm_decode_bridge_report", prewarm_report);
    }
    free(image.data);

    sd_latent_t* latent_to_decode = encoded;
    sd_gpu_handle_t sampled_gpu_latent = 0;
    sd_gpu_handle_t sample_init_gpu_latent = 0;
    bool sample_init_gpu_latent_owned = false;
    if (args.sample) {
        sd_image_t ref_image{};
        sd_image_t conditioning_ref_image{};
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        sd_model_pipeline_capabilities_t caps{};
        sd_get_model_pipeline_capabilities(ctx, &caps);
        gen_params.prompt          = args.prompt.c_str();
        gen_params.negative_prompt = args.negative_prompt.c_str();
        gen_params.width           = args.width > 0 ? args.width : static_cast<int>(image.width);
        gen_params.height          = args.height > 0 ? args.height : static_cast<int>(image.height);
        gen_params.seed            = args.seed;
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
        if (!args.conditioning_ref_image.empty()) {
            if (!load_sd_image_from_file(&conditioning_ref_image, args.conditioning_ref_image.c_str(), 0, 0, 3)) {
                std::cerr << "failed to load conditioning reference image: " << args.conditioning_ref_image << "\n";
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            std::cout << "loaded conditioning reference image "
                      << conditioning_ref_image.width << "x"
                      << conditioning_ref_image.height << "x"
                      << conditioning_ref_image.channel << "\n";
        }

        sd_conditioning_handle_t positive_condition = 0;
        sd_conditioning_handle_t negative_condition = 0;
        sd_gpu_handle_t imported_noise_gpu = 0;
        std::vector<sd_gpu_handle_t> imported_step_noise_gpu;
        auto release_imported_step_noise_gpu = [&]() {
            for (sd_gpu_handle_t handle : imported_step_noise_gpu) {
                sd_gpu_handle_release(ctx, handle);
            }
            imported_step_noise_gpu.clear();
        };
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
                      << " sampler_init=" << (condition_caps.supports_sampler_conditioning_init_latent_input ? "true" : "false")
                      << " gpu_init_bridge=" << (condition_caps.supports_sampler_conditioning_gpu_init_latent_bridge_input ? "true" : "false")
                      << " reuse=" << (condition_caps.supports_conditioning_handle_reuse ? "true" : "false")
                      << " qwen_image_qwen=" << (condition_caps.supports_qwen_image_qwen_conditioning ? "true" : "false")
                      << " qwen_image_qwen_gpu=" << (condition_caps.supports_qwen_image_qwen_conditioning_gpu_resident ? "true" : "false")
                      << "\n";

            sd_conditioning_encode_options_t condition_options;
            sd_conditioning_encode_options_init(&condition_options);
            condition_options.clip_skip = gen_params.clip_skip;
            condition_options.width = gen_params.width;
            condition_options.height = gen_params.height;

            sd_conditioning_desc_t positive_desc{};
            condition_options.cache_key_hint = "conditioning_positive";
            const sd_image_t* positive_ref_image = conditioning_ref_image.data != nullptr
                                                       ? &conditioning_ref_image
                                                       : (ref_image.data != nullptr ? &ref_image : nullptr);
            const bool positive_has_ref_images = positive_ref_image != nullptr;
            const bool positive_encoded = positive_has_ref_images
                                              ? sd_conditioning_encode_text_with_ref_images(ctx,
                                                                                           args.prompt.c_str(),
                                                                                           positive_ref_image,
                                                                                           1,
                                                                                           &condition_options,
                                                                                           &positive_condition,
                                                                                           &positive_desc)
                                              : sd_conditioning_encode_text(ctx,
                                                                            args.prompt.c_str(),
                                                                            &condition_options,
                                                                            &positive_condition,
                                                                            &positive_desc);
            if (!positive_encoded) {
                std::cerr << "sd_conditioning_encode_text positive failed\n";
                if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 1;
            }
            print_conditioning_desc(ctx, "positive_condition", positive_condition);

            const bool needs_negative_condition =
                gen_params.sample_params.guidance.txt_cfg != 1.0f ||
                gen_params.sample_params.guidance.img_cfg != gen_params.sample_params.guidance.txt_cfg;
            if (needs_negative_condition) {
                condition_options.force_zero_uncond = (caps.family == SD_MODEL_FAMILY_SDXL && args.negative_prompt.empty());
                condition_options.cache_key_hint = "conditioning_negative";
                sd_conditioning_desc_t negative_desc{};
                if (!sd_conditioning_encode_text(ctx, args.negative_prompt.c_str(), &condition_options, &negative_condition, &negative_desc)) {
                    std::cerr << "sd_conditioning_encode_text negative failed\n";
                    sd_conditioning_release(ctx, positive_condition);
                    if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                print_conditioning_desc(ctx, "negative_condition", negative_condition);
            } else {
                std::cout << "negative_condition skipped for cfg=1 distilled/CFG-free sampling\n";
            }

            if (args.release_text_encoder_after_conditioning) {
                const auto release_start = std::chrono::steady_clock::now();
                if (!sd_release_clip_model_params(ctx)) {
                    std::cerr << "sd_release_clip_model_params failed\n";
                    sd_conditioning_release(ctx, positive_condition);
                    sd_conditioning_release(ctx, negative_condition);
                    if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                const auto release_end = std::chrono::steady_clock::now();
                std::cout << "released_text_encoder_after_conditioning=true release_ms="
                          << std::chrono::duration_cast<std::chrono::milliseconds>(release_end - release_start).count()
                          << "\n";
            }

            if (!args.import_noise_npy.empty()) {
                imported_noise_gpu = upload_noise_npy_as_gpu_latent(ctx, args.import_noise_npy);
                if (imported_noise_gpu == 0) {
                    sd_conditioning_release(ctx, positive_condition);
                    sd_conditioning_release(ctx, negative_condition);
                    if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                    if (ref_image.data != nullptr) free(ref_image.data);
                    if (encoded != nullptr) free_sd_latent(encoded);
                    free_sd_ctx(ctx);
                    return 1;
                }
                print_gpu_desc(ctx, "imported_noise_gpu_latent", imported_noise_gpu);
            }
            if (!args.import_step_noise_npys.empty()) {
                imported_step_noise_gpu.reserve(args.import_step_noise_npys.size());
                for (size_t i = 0; i < args.import_step_noise_npys.size(); ++i) {
                    sd_gpu_handle_t step_noise = upload_noise_npy_as_gpu_latent(ctx, args.import_step_noise_npys[i]);
                    if (step_noise == 0) {
                        release_imported_step_noise_gpu();
                        if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
                        if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                    imported_step_noise_gpu.push_back(step_noise);
                    std::string label = "imported_step_noise_gpu_latent_" + std::to_string(i);
                    print_gpu_desc(ctx, label.c_str(), step_noise);
                }
            }
            if (args.condition_only) {
                std::cout << "condition_only completed\n";
                release_imported_step_noise_gpu();
                if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                sd_conditioning_release(ctx, positive_condition);
                sd_conditioning_release(ctx, negative_condition);
                if (conditioning_ref_image.data != nullptr) free(conditioning_ref_image.data);
                if (ref_image.data != nullptr) free(ref_image.data);
                if (encoded_gpu_latent != 0) sd_gpu_handle_release(ctx, encoded_gpu_latent);
                if (encoded != nullptr) free_sd_latent(encoded);
                free_sd_ctx(ctx);
                return 0;
            }
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
                if (imported_noise_gpu != 0 || !imported_step_noise_gpu.empty()) {
                    if (args.condition_handles_reuse) {
                        std::cerr << "--import-noise-npy/--import-step-noise-npy do not support --condition-handles-reuse in this smoke\n";
                        release_imported_step_noise_gpu();
                        if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                    if (args.gpu_init_sample_input && !args.sample_without_init) {
                        if (encoded_gpu_latent != 0) {
                            sample_init_gpu_latent = encoded_gpu_latent;
                        } else if (encoded != nullptr) {
                            if (!sd_cpu_latent_upload(ctx, encoded, &sample_init_gpu_latent, nullptr)) {
                                std::cerr << "sd_cpu_latent_upload for imported-noise sampler init failed\n";
                                release_imported_step_noise_gpu();
                                if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                                sd_conditioning_release(ctx, positive_condition);
                                sd_conditioning_release(ctx, negative_condition);
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
                            release_imported_step_noise_gpu();
                            if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                            sd_conditioning_release(ctx, positive_condition);
                            sd_conditioning_release(ctx, negative_condition);
                            if (ref_image.data != nullptr) free(ref_image.data);
                            if (encoded != nullptr) free_sd_latent(encoded);
                            free_sd_ctx(ctx);
                            return 1;
                        }
                    }
                    const uint32_t expected_step_noise_count =
                        sd_sampler_step_noise_count(gen_params.sample_params.sample_method,
                                                    static_cast<uint32_t>(gen_params.sample_params.sample_steps + 1));
                    std::cout << "calling sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_schedule_gpu step_noise_count="
                              << imported_step_noise_gpu.size()
                              << " expected_for_standard_schedule=" << expected_step_noise_count
                              << " sampler_uses_step_noise="
                              << (sd_sampler_uses_step_noise(gen_params.sample_params.sample_method) ? "true" : "false")
                              << " sampler_uses_brownian_step_noise="
                              << (sd_sampler_uses_brownian_step_noise(gen_params.sample_params.sample_method) ? "true" : "false")
                              << "\n";
                    if (!sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_schedule_gpu(ctx,
                                                                                                    &gen_params,
                                                                                                    sample_init_gpu_latent,
                                                                                                    imported_noise_gpu,
                                                                                                    imported_step_noise_gpu.empty() ? nullptr : imported_step_noise_gpu.data(),
                                                                                                    static_cast<uint32_t>(imported_step_noise_gpu.size()),
                                                                                                    positive_condition,
                                                                                                    negative_condition,
                                                                                                    &sampled_gpu_latent)) {
                        std::cerr << "sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_schedule_gpu failed\n";
                        if (sample_init_gpu_latent_owned) sd_gpu_handle_release(ctx, sample_init_gpu_latent);
                        release_imported_step_noise_gpu();
                        if (imported_noise_gpu != 0) sd_gpu_handle_release(ctx, imported_noise_gpu);
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
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
                            std::cerr << "sd_cpu_latent_upload for conditioning sampler init failed\n";
                            sd_conditioning_release(ctx, positive_condition);
                            sd_conditioning_release(ctx, negative_condition);
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
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                    std::cout << "calling sd_sample_latent_gpu_with_init_gpu_and_conditioning\n";
                    if (!sd_sample_latent_gpu_with_init_gpu_and_conditioning(ctx, &gen_params, sample_init_gpu_latent, positive_condition, negative_condition, &sampled_gpu_latent)) {
                        std::cerr << "sd_sample_latent_gpu_with_init_gpu_and_conditioning failed\n";
                        if (sample_init_gpu_latent_owned) sd_gpu_handle_release(ctx, sample_init_gpu_latent);
                        sd_conditioning_release(ctx, positive_condition);
                        sd_conditioning_release(ctx, negative_condition);
                        if (ref_image.data != nullptr) free(ref_image.data);
                        if (encoded != nullptr) free_sd_latent(encoded);
                        free_sd_ctx(ctx);
                        return 1;
                    }
                } else {
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
                }
                if (args.condition_handles_reuse) {
                    sd_gpu_handle_release(ctx, sampled_gpu_latent);
                    sampled_gpu_latent = 0;

                    sd_gpu_handle_t reused_gpu_latent = 0;
                    std::cout << "calling conditioning sampler reuse\n";
                    bool reuse_ok = false;
                    if (args.gpu_init_sample_input && !args.sample_without_init) {
                        reuse_ok = sd_sample_latent_gpu_with_init_gpu_and_conditioning(ctx, &gen_params, sample_init_gpu_latent, positive_condition, negative_condition, &reused_gpu_latent);
                    } else {
                        reuse_ok = sd_sample_latent_gpu_with_conditioning(ctx, &gen_params, nullptr, positive_condition, negative_condition, &reused_gpu_latent);
                    }
                    if (!reuse_ok) {
                        std::cerr << "conditioning sampler reuse failed\n";
                        if (sample_init_gpu_latent_owned) sd_gpu_handle_release(ctx, sample_init_gpu_latent);
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
        if (conditioning_ref_image.data != nullptr) {
            free(conditioning_ref_image.data);
        }
        if (positive_condition != 0) {
            sd_conditioning_release(ctx, positive_condition);
        }
        if (negative_condition != 0) {
            sd_conditioning_release(ctx, negative_condition);
        }
        release_imported_step_noise_gpu();
        if (imported_noise_gpu != 0) {
            sd_gpu_handle_release(ctx, imported_noise_gpu);
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
            sd_gpu_handle_t decode_gpu_latent = 0;
            bool decode_ok = false;
            if (args.gpu_latent_decode_input) {
                decode_gpu_latent = sampled_gpu_latent != 0 ? sampled_gpu_latent : encoded_gpu_latent;
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
                const auto download_start = std::chrono::steady_clock::now();
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
                const auto download_end = std::chrono::steady_clock::now();
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
                std::cout << "explicit_download_ms="
                          << std::chrono::duration_cast<std::chrono::milliseconds>(download_end - download_start).count()
                          << "\n";
                if (args.compare_roundtrip_input) {
                    print_rgb_diff_against_input("roundtrip_input_diff",
                                                 roundtrip_reference,
                                                 roundtrip_reference_width,
                                                 roundtrip_reference_height,
                                                 roundtrip_reference_channels,
                                                 output.data,
                                                 output.width,
                                                 output.height,
                                                 output.channel);
                }
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
                const auto download_start = std::chrono::steady_clock::now();
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
                const auto download_end = std::chrono::steady_clock::now();
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
                std::cout << "caller_owned_download_ms="
                          << std::chrono::duration_cast<std::chrono::milliseconds>(download_end - download_start).count()
                          << "\n";
                if (args.compare_roundtrip_input) {
                    print_rgb_diff_against_input("roundtrip_input_diff",
                                                 roundtrip_reference,
                                                 roundtrip_reference_width,
                                                 roundtrip_reference_height,
                                                 roundtrip_reference_channels,
                                                 rgba.data(),
                                                 static_cast<uint32_t>(desc.w),
                                                 static_cast<uint32_t>(desc.h),
                                                 4);
                }
            }
            if (args.decode_twice) {
                if (!args.gpu_latent_decode_input || decode_gpu_latent == 0) {
                    std::cerr << "--decode-twice currently requires --gpu-latent-decode-input\n";
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
                sd_gpu_handle_t second_gpu_image = 0;
                sd_vae_memory_report_t second_decode_report;
                const auto second_start = std::chrono::steady_clock::now();
                if (!sd_decode_gpu_latent_normal_gpu(decode_ctx, decode_gpu_latent, &vae_options, &second_gpu_image, &second_decode_report)) {
                    std::cerr << "second sd_decode_gpu_latent_normal_gpu failed\n";
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
                const auto second_end = std::chrono::steady_clock::now();
                print_vae_report("decode_report_second", second_decode_report);
                std::cout << "second_gpu_decode_handle=" << second_gpu_image << "\n";
                std::cout << "second_decode_total_ms="
                          << std::chrono::duration_cast<std::chrono::milliseconds>(second_end - second_start).count()
                          << "\n";
                sd_gpu_handle_release(decode_ctx, second_gpu_image);
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
