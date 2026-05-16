#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "auto_encoder_kl.hpp"
#include "conditioner.hpp"
#include "control.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "esrgan.hpp"
#include "lora.hpp"
#include "pmid.hpp"
#include "sample-cache.h"
#include "tae.hpp"
#include "vae.hpp"

#include "latent-preview.h"
#include "name_conversion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_map>

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 1.x Tiny UNet",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SD 2.x Tiny UNet",
    "SDXS",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SDXL (Vega)",
    "SDXL (SSD1B)",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill",
    "Flux Control",
    "Flex.2",
    "Chroma Radiance",
    "Wan 2.x",
    "Wan 2.2 I2V",
    "Wan 2.2 TI2V",
    "Qwen Image",
    "Anima",
    "Flux.2",
    "Flux.2 klein",
    "Z-Image",
    "Ovis Image",
    "Marigold IID",
};

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
    "ER-SDE",
    "DPM++ SDE",
    "DPM++ SDE GPU",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE GPU",
    "DPM++ 2M SDE Heun",
    "DPM++ 2M SDE Heun GPU",
    "DPM++ 3M SDE",
    "DPM++ 3M SDE GPU",
};

/*================================================== Helper Functions ================================================*/

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120f,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = nullptr;  // general backend
    ggml_backend_t clip_backend        = nullptr;
    ggml_backend_t control_net_backend = nullptr;
    ggml_backend_t vae_backend         = nullptr;

    SDVersion version;
    bool vae_decode_only         = false;
    bool external_vae_is_invalid = false;
    bool free_params_immediately = false;

    bool circular_x = false;
    bool circular_y = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float default_flow_shift         = INFINITY;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd or wan2.1 i2v
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<DiffusionModel> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;
    std::shared_ptr<VAE> preview_vae;
    std::shared_ptr<ControlNet> control_net;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEmbed> pmid_id_embeds;
    std::vector<std::shared_ptr<LoraModel>> cond_stage_lora_models;
    std::vector<std::shared_ptr<LoraModel>> diffusion_lora_models;
    std::vector<std::shared_ptr<LoraModel>> first_stage_lora_models;
    bool apply_lora_immediately = false;

    std::string taesd_path;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
    bool offload_params_to_cpu           = false;
    bool use_pmid                        = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, ggml_tensor*> tensors;

    // lora_name => multiplier
    std::unordered_map<std::string, float> curr_lora_state;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    void init_backend() {
#ifdef SD_USE_CUDA
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        size_t device          = 0;
        const int device_count = ggml_backend_vk_get_device_count();
        if (device_count) {
            const char* SD_VK_DEVICE = getenv("SD_VK_DEVICE");
            if (SD_VK_DEVICE != nullptr) {
                std::string sd_vk_device_str = SD_VK_DEVICE;
                try {
                    device = std::stoull(sd_vk_device_str);
                } catch (const std::invalid_argument&) {
                    LOG_WARN("SD_VK_DEVICE environment variable is not a valid integer (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                } catch (const std::out_of_range&) {
                    LOG_WARN("SD_VK_DEVICE environment variable value is out of range for `unsigned long long` type (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                }
                if (device >= device_count) {
                    LOG_WARN("Cannot find targeted vulkan device (%llu). Falling back to device 0.", device);
                    device = 0;
                }
            }
            LOG_INFO("Vulkan: Using device %llu", device);
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif
#ifdef SD_USE_OPENCL
        LOG_DEBUG("Using OpenCL backend");
        // ggml_log_set(ggml_log_callback_default, nullptr); // Optional ggml logs
        backend = ggml_backend_opencl_init();
        if (!backend) {
            LOG_WARN("Failed to initialize OpenCL backend");
        }
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    static bool env_flag_enabled(const char* name) {
        const char* value = getenv(name);
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;

        bool use_tae = false;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        init_backend();

        ModelLoader model_loader;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
            LOG_INFO("loading high noise diffusion model from '%s'", sd_ctx_params->high_noise_diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->high_noise_diffusion_model_path, "model.high_noise_diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->high_noise_diffusion_model_path);
            }
        }

        bool is_unet = sd_version_is_unet(model_loader.get_sd_version());

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->t5xxl_path)) > 0) {
            LOG_INFO("loading t5xxl from '%s'", sd_ctx_params->t5xxl_path);
            if (!model_loader.init_from_file(sd_ctx_params->t5xxl_path, "text_encoders.t5xxl.transformer.")) {
                LOG_WARN("loading t5xxl from '%s' failed", sd_ctx_params->t5xxl_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
                external_vae_is_invalid = true;
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->taesd_path)) > 0) {
            LOG_INFO("loading tae from '%s'", sd_ctx_params->taesd_path);
            if (!model_loader.init_from_file(sd_ctx_params->taesd_path, "tae.")) {
                LOG_WARN("loading tae from '%s' failed", sd_ctx_params->taesd_path);
            }
            use_tae = true;
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        if (sd_ctx_params->lora_apply_mode == LORA_APPLY_AUTO) {
            bool have_quantized_weight = false;
            if (wtype != GGML_TYPE_COUNT && ggml_is_quantized(wtype)) {
                have_quantized_weight = true;
            } else {
                for (const auto& [type, _] : wtype_stat) {
                    if (ggml_is_quantized(type)) {
                        have_quantized_weight = true;
                        break;
                    }
                }
            }
            if (have_quantized_weight) {
                apply_lora_immediately = false;
            } else {
                apply_lora_immediately = true;
            }
        } else if (sd_ctx_params->lora_apply_mode == LORA_APPLY_IMMEDIATELY) {
            apply_lora_immediately = true;
        } else {
            apply_lora_immediately = false;
        }

        if (sd_version_is_control(version)) {
            // Might need vae encode for control cond
            vae_decode_only = false;
        }

        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS) {
            tae_preview_only = false;
            use_tae          = true;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        bool load_conditioner_and_diffusion = !vae_decode_only;
        if (!load_conditioner_and_diffusion) {
            LOG_INFO("vae_decode_only=true: skipping conditioner and diffusion model creation");
        }

        bool clip_on_cpu = sd_ctx_params->keep_clip_on_cpu;

        {
            clip_backend = backend;
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (load_conditioner_and_diffusion) {
                if (sd_version_is_sd3(version)) {
                    cond_stage_model = std::make_shared<SD3CLIPEmbedder>(clip_backend,
                                                                         offload_params_to_cpu,
                                                                         tensor_storage_map);
                    diffusion_model  = std::make_shared<MMDiTModel>(backend,
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map);
                } else if (sd_version_is_flux(version)) {
                    bool is_chroma = false;
                    for (auto pair : tensor_storage_map) {
                        if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
                            is_chroma = true;
                            break;
                        }
                    }
                    if (is_chroma) {
                        if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && sd_ctx_params->chroma_use_dit_mask) {
                            LOG_WARN(
                                "!!!It looks like you are using Chroma with flash attention. "
                                "This is currently unsupported. "
                                "If you find that the generated images are broken, "
                                "try either disabling flash attention or specifying "
                                "--chroma-disable-dit-mask as a workaround.");
                        }

                        cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                            offload_params_to_cpu,
                                                                            tensor_storage_map,
                                                                            sd_ctx_params->chroma_use_t5_mask,
                                                                            sd_ctx_params->chroma_t5_mask_pad);
                    } else if (version == VERSION_OVIS_IMAGE) {
                        cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                         offload_params_to_cpu,
                                                                         tensor_storage_map,
                                                                         version,
                                                                         "",
                                                                         false);
                    } else {
                        cond_stage_model = std::make_shared<FluxCLIPEmbedder>(clip_backend,
                                                                              offload_params_to_cpu,
                                                                              tensor_storage_map);
                    }
                    diffusion_model = std::make_shared<FluxModel>(backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  version,
                                                                  sd_ctx_params->chroma_use_dit_mask);
                } else if (sd_version_is_flux2(version)) {
                    bool is_chroma   = false;
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version);
                    diffusion_model  = std::make_shared<FluxModel>(backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  version,
                                                                  sd_ctx_params->chroma_use_dit_mask);
                } else if (sd_version_is_wan(version)) {
                    cond_stage_model = std::make_shared<T5CLIPEmbedder>(clip_backend,
                                                                        offload_params_to_cpu,
                                                                        tensor_storage_map,
                                                                        true,
                                                                        0,
                                                                        true);
                    diffusion_model  = std::make_shared<WanModel>(backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 "model.diffusion_model",
                                                                 version);
                    if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
                        high_noise_diffusion_model = std::make_shared<WanModel>(backend,
                                                                                offload_params_to_cpu,
                                                                                tensor_storage_map,
                                                                                "model.high_noise_diffusion_model",
                                                                                version);
                    }
                    if (diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
                        diffusion_model->get_desc() == "Wan2.1-FLF2V-14B" ||
                        diffusion_model->get_desc() == "Wan2.1-I2V-1.3B") {
                        clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(backend,
                                                                                 offload_params_to_cpu,
                                                                                 tensor_storage_map);
                        clip_vision->alloc_params_buffer();
                        clip_vision->get_param_tensors(tensors);
                    }
                } else if (sd_version_is_qwen_image(version)) {
                    bool enable_vision = true;
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "",
                                                                     enable_vision);
                    diffusion_model  = std::make_shared<QwenImageModel>(backend,
                                                                       offload_params_to_cpu,
                                                                       tensor_storage_map,
                                                                       "model.diffusion_model",
                                                                       version,
                                                                       sd_ctx_params->qwen_image_zero_cond_t);
                } else if (sd_version_is_anima(version)) {
                    cond_stage_model = std::make_shared<AnimaConditioner>(clip_backend,
                                                                          offload_params_to_cpu,
                                                                          tensor_storage_map);
                    diffusion_model  = std::make_shared<AnimaModel>(backend,
                                                                   offload_params_to_cpu,
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model");
                } else if (sd_version_is_z_image(version)) {
                    cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                     offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version);
                    diffusion_model  = std::make_shared<ZImageModel>(backend,
                                                                    offload_params_to_cpu,
                                                                    tensor_storage_map,
                                                                    "model.diffusion_model",
                                                                    version);
                } else {  // SD1.x SD2.x SDXL
                    std::map<std::string, std::string> embbeding_map;
                    for (uint32_t i = 0; i < sd_ctx_params->embedding_count; i++) {
                        embbeding_map.emplace(SAFE_STR(sd_ctx_params->embeddings[i].name), SAFE_STR(sd_ctx_params->embeddings[i].path));
                    }
                    if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                        cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                               offload_params_to_cpu,
                                                                                               tensor_storage_map,
                                                                                               embbeding_map,
                                                                                               version,
                                                                                               PM_VERSION_2);
                    } else {
                        cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend,
                                                                                               offload_params_to_cpu,
                                                                                               tensor_storage_map,
                                                                                               embbeding_map,
                                                                                               version);
                    }
                    diffusion_model = std::make_shared<UNetModel>(backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  version);
                    if (sd_ctx_params->diffusion_conv_direct) {
                        LOG_INFO("Using Conv2d direct in the diffusion model");
                        std::dynamic_pointer_cast<UNetModel>(diffusion_model)->unet.set_conv2d_direct_enabled(true);
                    }
                }

                cond_stage_model->alloc_params_buffer();
                cond_stage_model->get_param_tensors(tensors);

                diffusion_model->alloc_params_buffer();
                diffusion_model->get_param_tensors(tensors);

                if (sd_version_is_unet_edit(version)) {
                    vae_decode_only = false;
                }

                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->alloc_params_buffer();
                    high_noise_diffusion_model->get_param_tensors(tensors);
                }
            }

            if (sd_ctx_params->keep_vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("VAE Autoencoder: Using CPU backend");
                vae_backend = ggml_backend_cpu_init();
            } else {
                vae_backend = backend;
            }

            auto create_tae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<TinyVideoAutoEncoder>(vae_backend,
                                                                  offload_params_to_cpu,
                                                                  tensor_storage_map,
                                                                  "decoder",
                                                                  vae_decode_only,
                                                                  version);

                } else {
                    auto model = std::make_shared<TinyImageAutoEncoder>(vae_backend,
                                                                        offload_params_to_cpu,
                                                                        tensor_storage_map,
                                                                        "decoder.layers",
                                                                        vae_decode_only,
                                                                        version);
                    return model;
                }
            };

            auto create_vae = [&]() -> std::shared_ptr<VAE> {
                if (sd_version_is_wan(version) ||
                    sd_version_is_qwen_image(version) ||
                    sd_version_is_anima(version)) {
                    return std::make_shared<WAN::WanVAERunner>(vae_backend,
                                                               offload_params_to_cpu,
                                                               tensor_storage_map,
                                                               "first_stage_model",
                                                               vae_decode_only,
                                                               version);
                } else {
                    auto model = std::make_shared<AutoEncoderKL>(vae_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 vae_decode_only,
                                                                 false,
                                                                 version);
                    if (sd_version_is_sdxl(version) &&
                        (strlen(SAFE_STR(sd_ctx_params->vae_path)) == 0 || sd_ctx_params->force_sdxl_vae_conv_scale || external_vae_is_invalid)) {
                        float vae_conv_2d_scale = 1.f / 32.f;
                        LOG_WARN(
                            "No valid VAE specified with --vae or --force-sdxl-vae-conv-scale flag set, "
                            "using Conv2D scale %.3f",
                            vae_conv_2d_scale);
                        model->set_conv2d_scale(vae_conv_2d_scale);
                    }
                    return model;
                }
            };

            if (version == VERSION_CHROMA_RADIANCE) {
                LOG_INFO("using FakeVAE");
                first_stage_model = std::make_shared<FakeVAE>(version,
                                                              vae_backend,
                                                              offload_params_to_cpu);
            } else if (use_tae && !tae_preview_only) {
                LOG_INFO("using TAE for encoding / decoding");
                first_stage_model = create_tae();
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "tae");
            } else {
                LOG_INFO("using VAE for encoding / decoding");
                first_stage_model = create_vae();
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
                if (use_tae && tae_preview_only) {
                    LOG_INFO("using TAE for preview");
                    preview_vae = create_tae();
                    preview_vae->alloc_params_buffer();
                    preview_vae->get_param_tensors(tensors, "tae");
                }
            }

            bool use_vae_conv_direct = sd_ctx_params->vae_conv_direct;
#ifdef SD_USE_CUDA
            if (!use_vae_conv_direct &&
                sd_version_is_sdxl(version) &&
                !env_flag_enabled("SDCPP_DISABLE_DEFAULT_VAE_CONV_DIRECT")) {
                use_vae_conv_direct = true;
                LOG_INFO("Defaulting to Conv2d direct in the SDXL VAE on CUDA; set SDCPP_DISABLE_DEFAULT_VAE_CONV_DIRECT=1 to use the im2col path");
            }
#endif
            if (use_vae_conv_direct) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
                if (preview_vae) {
                    preview_vae->set_conv2d_direct_enabled(true);
                }
            }

            if (load_conditioner_and_diffusion && strlen(SAFE_STR(sd_ctx_params->control_net_path)) > 0) {
                ggml_backend_t controlnet_backend = nullptr;
                if (sd_ctx_params->keep_control_net_on_cpu && !ggml_backend_is_cpu(backend)) {
                    LOG_DEBUG("ControlNet: Using CPU backend");
                    controlnet_backend = ggml_backend_cpu_init();
                } else {
                    controlnet_backend = backend;
                }
                control_net = std::make_shared<ControlNet>(controlnet_backend,
                                                           offload_params_to_cpu,
                                                           version);
                control_net->set_wtype_override(wtype, tensor_type_rules);
                if (sd_ctx_params->diffusion_conv_direct) {
                    LOG_INFO("Using Conv2d direct in the control net");
                    control_net->set_conv2d_direct_enabled(true);
                }
            }

            if (load_conditioner_and_diffusion) {
                if (strstr(SAFE_STR(sd_ctx_params->photo_maker_path), "v2")) {
                    pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend,
                                                                       offload_params_to_cpu,
                                                                       tensor_storage_map,
                                                                       "pmid",
                                                                       version,
                                                                       PM_VERSION_2);
                    LOG_INFO("using PhotoMaker Version 2");
                } else {
                    pmid_model = std::make_shared<PhotoMakerIDEncoder>(backend,
                                                                       offload_params_to_cpu,
                                                                       tensor_storage_map,
                                                                       "pmid",
                                                                       version);
                }
            }
            if (load_conditioner_and_diffusion && strlen(SAFE_STR(sd_ctx_params->photo_maker_path)) > 0) {
                pmid_lora               = std::make_shared<LoraModel>("pmid", backend, sd_ctx_params->photo_maker_path, "", version);
                auto lora_tensor_filter = [&](const std::string& tensor_name) {
                    if (starts_with(tensor_name, "lora.model")) {
                        return true;
                    }
                    return false;
                };
                if (!pmid_lora->load_from_file(n_threads, lora_tensor_filter)) {
                    LOG_WARN("load photomaker lora tensors from %s failed", sd_ctx_params->photo_maker_path);
                    return false;
                }
                LOG_INFO("loading stacked ID embedding (PHOTOMAKER) model file from '%s'", sd_ctx_params->photo_maker_path);
                if (!model_loader.init_from_file_and_convert_name(sd_ctx_params->photo_maker_path, "pmid.")) {
                    LOG_WARN("loading stacked ID embedding from '%s' failed", sd_ctx_params->photo_maker_path);
                } else {
                    use_pmid = true;
                }
            }
            if (use_pmid) {
                if (!pmid_model->alloc_params_buffer()) {
                    LOG_ERROR(" pmid model params buffer allocation failed");
                    return false;
                }
                pmid_model->get_param_tensors(tensors, "pmid");
            }

            if (sd_ctx_params->flash_attn) {
                LOG_INFO("Using flash attention");
                if (cond_stage_model) {
                    cond_stage_model->set_flash_attention_enabled(true);
                }
                if (clip_vision) {
                    clip_vision->set_flash_attention_enabled(true);
                }
                if (first_stage_model) {
                    first_stage_model->set_flash_attention_enabled(true);
                }
                if (preview_vae) {
                    preview_vae->set_flash_attention_enabled(true);
                }
            }

            if ((sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) && diffusion_model) {
                LOG_INFO("Using flash attention in the diffusion model");
                diffusion_model->set_flash_attention_enabled(true);
                if (high_noise_diffusion_model) {
                    high_noise_diffusion_model->set_flash_attention_enabled(true);
                }
            }

            if (diffusion_model) {
                diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            if (control_net) {
                control_net->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
            circular_x = sd_ctx_params->circular_x;
            circular_y = sd_ctx_params->circular_y;
        }

        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        if (use_tae && !tae_preview_only) {
            ignore_tensors.insert("first_stage_model.");
        }
        if (use_pmid) {
            ignore_tensors.insert("pmid.unet.");
        }
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        if (vae_decode_only) {
            ignore_tensors.insert("cond_stage_model.");
            ignore_tensors.insert("conditioner.");
            ignore_tensors.insert("text_encoders.");
            ignore_tensors.insert("te.");
            ignore_tensors.insert("model.diffusion_model.");
            ignore_tensors.insert("model.high_noise_diffusion_model.");
            ignore_tensors.insert("unet.");
            ignore_tensors.insert("pmid.");
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("tae.encoder");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }
        if (version == VERSION_OVIS_IMAGE) {
            ignore_tensors.insert("text_encoders.llm.vision_model.");
            ignore_tensors.insert("text_encoders.llm.visual_tokenizer.");
            ignore_tensors.insert("text_encoders.llm.vte.");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size = cond_stage_model ? cond_stage_model->get_params_buffer_size() : 0;
            size_t unet_params_mem_size = diffusion_model ? diffusion_model->get_params_buffer_size() : 0;
            if (high_noise_diffusion_model) {
                unet_params_mem_size += high_noise_diffusion_model->get_params_buffer_size();
            }
            size_t vae_params_mem_size = 0;
            vae_params_mem_size        = first_stage_model->get_params_buffer_size();
            if (preview_vae) {
                vae_params_mem_size += preview_vae->get_params_buffer_size();
            }
            size_t control_net_params_mem_size = 0;
            if (control_net) {
                if (!control_net->load_from_file(SAFE_STR(sd_ctx_params->control_net_path), n_threads)) {
                    return false;
                }
                control_net_params_mem_size = control_net->get_params_buffer_size();
            }
            size_t pmid_params_mem_size = 0;
            if (use_pmid) {
                pmid_params_mem_size = pmid_model->get_params_buffer_size();
            }

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size + pmid_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size + pmid_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            if (control_net && ggml_backend_is_cpu(control_net_backend)) {
                total_params_ram_size += control_net_params_mem_size;
            } else if (control_net) {
                total_params_vram_size += control_net_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s), controlnet %.2fMB(%s), pmid %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                control_net_params_mem_size / 1024.0 / 1024.0,
                (control_net && ggml_backend_is_cpu(control_net_backend)) ? "RAM" : "VRAM",
                pmid_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        // init denoiser
        if (!vae_decode_only) {
            prediction_t pred_type = sd_ctx_params->prediction;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_sd2(version)) {
                    // check is_using_v_parameterization_for_sd2
                    if (is_using_v_parameterization_for_sd2(sd_version_is_inpaint(version))) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_marigold_iid(version)) {
                    pred_type = V_PRED;
                } else if (sd_version_is_sdxl(version)) {
                    if (tensor_storage_map.find("edm_vpred.sigma_max") != tensor_storage_map.end()) {
                        // CosXL models
                        // TODO: get sigma_min and sigma_max values from file
                        pred_type = EDM_V_PRED;
                    } else if (tensor_storage_map.find("v_pred") != tensor_storage_map.end()) {
                        pred_type = V_PRED;
                    } else {
                        pred_type = EPS_PRED;
                    }
                } else if (sd_version_is_sd3(version) ||
                           sd_version_is_wan(version) ||
                           sd_version_is_qwen_image(version) ||
                           sd_version_is_anima(version) ||
                           sd_version_is_z_image(version)) {
                    pred_type = FLOW_PRED;
                    if (sd_version_is_wan(version)) {
                        default_flow_shift = 5.f;
                    } else {
                        default_flow_shift = 3.f;
                    }
                } else if (sd_version_is_flux(version)) {
                    pred_type = FLUX_FLOW_PRED;

                    default_flow_shift = 1.0f;  // TODO: validate
                    for (const auto& [name, tensor_storage] : tensor_storage_map) {
                        if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight")) {
                            default_flow_shift = 1.15f;
                            break;
                        }
                    }
                } else if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
            }

            switch (pred_type) {
                case EPS_PRED:
                    LOG_INFO("running in eps-prediction mode");
                    break;
                case V_PRED:
                    LOG_INFO("running in v-prediction mode");
                    denoiser = std::make_shared<CompVisVDenoiser>();
                    break;
                case EDM_V_PRED:
                    LOG_INFO("running in v-prediction EDM mode");
                    denoiser = std::make_shared<EDMVDenoiser>();
                    break;
                case FLOW_PRED: {
                    LOG_INFO("running in FLOW mode");
                    denoiser = std::make_shared<DiscreteFlowDenoiser>();
                    break;
                }
                case FLUX_FLOW_PRED: {
                    LOG_INFO("running in Flux FLOW mode");
                    denoiser = std::make_shared<FluxFlowDenoiser>();
                    break;
                }
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    ggml_free(ctx);
                    return false;
                }
            }

            auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
            if (comp_vis_denoiser) {
                for (int i = 0; i < TIMESTEPS; i++) {
                    comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                    comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
                }
            }
        } else {
            LOG_INFO("vae_decode_only=true: skipping denoiser initialization");
        }

        ggml_free(ctx);
        return true;
    }

    bool is_using_v_parameterization_for_sd2(bool is_inpaint = false) {
        sd::Tensor<float> x_t   = sd::full<float>({8, 8, 4, 1}, 0.5f);
        sd::Tensor<float> c     = sd::full<float>({1024, 2, 1, 1}, 0.5f);
        sd::Tensor<float> steps = sd::full<float>({1}, 999.0f);
        sd::Tensor<float> concat;
        if (is_inpaint) {
            concat = sd::zeros<float>({8, 8, 5, 1});
        }

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> out;
        DiffusionParams diffusion_params;
        diffusion_params.x         = &x_t;
        diffusion_params.timesteps = &steps;
        diffusion_params.context   = &c;
        if (!concat.empty()) {
            diffusion_params.c_concat = &concat;
        }
        auto out_opt = diffusion_model->compute(n_threads, diffusion_params);
        GGML_ASSERT(!out_opt.empty());
        out = std::move(out_opt);
        diffusion_model->free_compute_buffer();

        double result = static_cast<double>((out - x_t).mean());
        int64_t t1    = ggml_time_ms();
        LOG_DEBUG("check is_using_v_parameterization_for_sd2, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result < -1;
    }

    std::shared_ptr<LoraModel> load_lora_model_from_file(const std::string& lora_id,
                                                         float multiplier,
                                                         ggml_backend_t backend,
                                                         LoraModel::filter_t lora_tensor_filter = nullptr) {
        std::string lora_path             = lora_id;
        static std::string high_noise_tag = "|high_noise|";
        bool is_high_noise                = false;
        if (starts_with(lora_path, high_noise_tag)) {
            lora_path     = lora_path.substr(high_noise_tag.size());
            is_high_noise = true;
            LOG_DEBUG("high noise lora: %s", lora_path.c_str());
        }
        auto lora = std::make_shared<LoraModel>(lora_id, backend, lora_path, is_high_noise ? "model.high_noise_" : "", version);
        if (!lora->load_from_file(n_threads, lora_tensor_filter)) {
            LOG_WARN("load lora tensors from %s failed", lora_path.c_str());
            return nullptr;
        }

        lora->multiplier = multiplier;
        return lora;
    }

    void apply_loras_immediately(const std::unordered_map<std::string, float>& lora_state) {
        std::unordered_map<std::string, float> lora_state_diff;
        for (auto& kv : lora_state) {
            const std::string& lora_name = kv.first;
            float multiplier             = kv.second;
            lora_state_diff[lora_name] += multiplier;
        }
        for (auto& kv : curr_lora_state) {
            const std::string& lora_name = kv.first;
            float curr_multiplier        = kv.second;
            lora_state_diff[lora_name] -= curr_multiplier;
        }

        if (lora_state_diff.empty()) {
            return;
        }

        LOG_INFO("apply lora immediately");

        size_t rm = lora_state_diff.size() - lora_state.size();
        if (rm != 0) {
            LOG_INFO("attempting to apply %lu LoRAs (removing %lu applied LoRAs)", lora_state.size(), rm);
        } else {
            LOG_INFO("attempting to apply %lu LoRAs", lora_state.size());
        }

        for (auto& kv : lora_state_diff) {
            int64_t t0 = ggml_time_ms();

            auto lora = load_lora_model_from_file(kv.first, kv.second, backend);
            if (!lora || lora->lora_tensors.empty()) {
                continue;
            }
            lora->apply(tensors, version, n_threads);
            lora->free_params_buffer();

            int64_t t1 = ggml_time_ms();

            LOG_INFO("lora '%s' applied, taking %.2fs", kv.first.c_str(), (t1 - t0) * 1.0f / 1000);
        }

        curr_lora_state = lora_state;
    }

    void apply_loras_at_runtime(const std::unordered_map<std::string, float>& lora_state) {
        cond_stage_lora_models.clear();
        diffusion_lora_models.clear();
        first_stage_lora_models.clear();
        if (cond_stage_model) {
            cond_stage_model->set_weight_adapter(nullptr);
        }
        if (diffusion_model) {
            diffusion_model->set_weight_adapter(nullptr);
        }
        if (high_noise_diffusion_model) {
            high_noise_diffusion_model->set_weight_adapter(nullptr);
        }
        if (first_stage_model) {
            first_stage_model->set_weight_adapter(nullptr);
        }
        if (lora_state.empty()) {
            return;
        }
        LOG_INFO("apply lora at runtime");
        if (cond_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : cond_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            cond_stage_lora_models  = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_cond_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_id = kv.first;
                float multiplier           = kv.second;

                auto lora = load_lora_model_from_file(lora_id, multiplier, clip_backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    cond_stage_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(cond_stage_lora_models);
            cond_stage_model->set_weight_adapter(multi_lora_adapter);
        }
        if (diffusion_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : diffusion_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            diffusion_lora_models   = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_diffusion_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    diffusion_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(diffusion_lora_models);
            diffusion_model->set_weight_adapter(multi_lora_adapter);
            if (high_noise_diffusion_model) {
                high_noise_diffusion_model->set_weight_adapter(multi_lora_adapter);
            }
        }

        if (first_stage_model) {
            std::vector<std::shared_ptr<LoraModel>> lora_models;
            auto lora_state_diff = lora_state;
            for (auto& lora_model : first_stage_lora_models) {
                auto iter = lora_state_diff.find(lora_model->lora_id);

                if (iter != lora_state_diff.end()) {
                    lora_model->multiplier = iter->second;
                    lora_models.push_back(lora_model);
                    lora_state_diff.erase(iter);
                }
            }
            first_stage_lora_models = lora_models;
            auto lora_tensor_filter = [&](const std::string& tensor_name) {
                if (is_first_stage_model_name(tensor_name)) {
                    return true;
                }
                return false;
            };
            for (auto& kv : lora_state_diff) {
                const std::string& lora_name = kv.first;
                float multiplier             = kv.second;

                auto lora = load_lora_model_from_file(lora_name, multiplier, vae_backend, lora_tensor_filter);
                if (lora && !lora->lora_tensors.empty()) {
                    lora->preprocess_lora_tensors(tensors);
                    first_stage_lora_models.push_back(lora);
                }
            }
            auto multi_lora_adapter = std::make_shared<MultiLoraAdapter>(first_stage_lora_models);
            first_stage_model->set_weight_adapter(multi_lora_adapter);
        }
    }

    void lora_stat() {
        if (!cond_stage_lora_models.empty()) {
            LOG_INFO("cond_stage_lora_models:");
            for (auto& lora_model : cond_stage_lora_models) {
                lora_model->stat();
            }
        }

        if (!diffusion_lora_models.empty()) {
            LOG_INFO("diffusion_lora_models:");
            for (auto& lora_model : diffusion_lora_models) {
                lora_model->stat();
            }
        }

        if (!first_stage_lora_models.empty()) {
            LOG_INFO("first_stage_lora_models:");
            for (auto& lora_model : first_stage_lora_models) {
                lora_model->stat();
            }
        }
    }

    void apply_loras(const sd_lora_t* loras, uint32_t lora_count) {
        std::unordered_map<std::string, float> lora_f2m;
        for (uint32_t i = 0; i < lora_count; i++) {
            std::string lora_id = SAFE_STR(loras[i].path);
            if (loras[i].is_high_noise) {
                lora_id = "|high_noise|" + lora_id;
            }
            lora_f2m[lora_id] = loras[i].multiplier;
            LOG_DEBUG("lora %s:%.2f", lora_id.c_str(), loras[i].multiplier);
        }
        int64_t t0 = ggml_time_ms();
        if (apply_lora_immediately) {
            apply_loras_immediately(lora_f2m);
        } else {
            apply_loras_at_runtime(lora_f2m);
        }
        int64_t t1 = ggml_time_ms();
        if (!lora_f2m.empty()) {
            LOG_INFO("apply_loras completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        }
    }

    SDCondition get_pmid_conditon(sd_pm_params_t pm_params,
                                  ConditionerParams& condition_params) {
        SDCondition id_cond;
        if (use_pmid) {
            if (!pmid_lora->applied) {
                int64_t t0 = ggml_time_ms();
                pmid_lora->apply(tensors, version, n_threads);
                int64_t t1         = ggml_time_ms();
                pmid_lora->applied = true;
                LOG_INFO("pmid_lora apply completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
                if (free_params_immediately) {
                    pmid_lora->free_params_buffer();
                }
            }
            // preprocess input id images
            bool pmv2 = pmid_model->get_version() == PM_VERSION_2;
            if (pm_params.id_images_count > 0) {
                int clip_image_size        = 224;
                pmid_model->style_strength = pm_params.style_strength;
                sd::Tensor<float> id_image_tensor;
                for (int i = 0; i < pm_params.id_images_count; i++) {
                    auto id_image           = sd_image_to_tensor(pm_params.id_images[i]);
                    auto processed_id_image = clip_preprocess(id_image, clip_image_size, clip_image_size);
                    if (id_image_tensor.empty()) {
                        id_image_tensor = processed_id_image;
                    } else {
                        id_image_tensor = sd::ops::concat(id_image_tensor, processed_id_image, 3);
                    }
                }

                int64_t t0                      = ggml_time_ms();
                condition_params.num_input_imgs = pm_params.id_images_count;
                auto cond_tup                   = cond_stage_model->get_learned_condition_with_trigger(n_threads,
                                                                                                       condition_params);
                id_cond                         = std::get<0>(cond_tup);
                auto class_tokens_mask          = std::get<1>(cond_tup);
                sd::Tensor<float> id_embeds;
                if (pmv2 && pm_params.id_embed_path != nullptr) {
                    try {
                        id_embeds = sd::load_tensor_from_file_as_tensor<float>(pm_params.id_embed_path);
                    } catch (const std::exception&) {
                        id_embeds = {};
                    }
                }
                if (pmv2 && id_embeds.empty()) {
                    LOG_WARN("Provided PhotoMaker images, but NO valid ID embeds file for PM v2");
                    LOG_WARN("Turn off PhotoMaker");
                    use_pmid = false;
                } else {
                    if (pmv2 && pm_params.id_images_count != id_embeds.shape()[1]) {
                        LOG_WARN("PhotoMaker image count (%d) does NOT match ID embeds (%d). You should run face_detect.py again.", pm_params.id_images_count, static_cast<int>(id_embeds.shape()[1]));
                        LOG_WARN("Turn off PhotoMaker");
                        use_pmid = false;
                    } else {
                        auto res = pmid_model->compute(n_threads,
                                                       id_image_tensor,
                                                       id_cond.c_crossattn,
                                                       id_embeds,
                                                       class_tokens_mask);
                        if (res.empty()) {
                            LOG_ERROR("Photomaker ID Stacking failed");
                            LOG_WARN("Turn off PhotoMaker");
                            use_pmid = false;
                        } else {
                            id_cond.c_crossattn = std::move(res);
                            int64_t t1          = ggml_time_ms();
                            LOG_INFO("Photomaker ID Stacking, taking %" PRId64 " ms", t1 - t0);
                            // Encode input prompt without the trigger word for delayed conditioning
                            condition_params.text = cond_stage_model->remove_trigger_from_prompt(condition_params.text);
                        }
                        if (free_params_immediately) {
                            pmid_model->free_params_buffer();
                        }
                    }
                }
            } else {
                LOG_WARN("Provided PhotoMaker model file, but NO input ID images");
                LOG_WARN("Turn off PhotoMaker");
                use_pmid = false;
            }
        }
        return id_cond;
    }

    sd::Tensor<float> get_clip_vision_output(const sd::Tensor<float>& image,
                                             bool return_pooled   = true,
                                             int clip_skip        = -1,
                                             bool zero_out_masked = false) {
        sd::Tensor<float> output;
        if (zero_out_masked) {
            if (return_pooled) {
                output = sd::zeros<float>({clip_vision->vision_model.projection_dim});
            } else {
                output = sd::zeros<float>({clip_vision->vision_model.hidden_size, 257});
            }
        } else {
            auto pixel_values = clip_preprocess(image, clip_vision->vision_model.image_size, clip_vision->vision_model.image_size);
            auto output_opt   = clip_vision->compute(n_threads, pixel_values, return_pooled, clip_skip);
            if (output_opt.empty()) {
                LOG_ERROR("clip_vision compute failed");
                return {};
            }
            output = std::move(output_opt);
        }
        return output;
    }

    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         const sd::Tensor<float>& init_latent,
                                         const sd::Tensor<float>& denoise_mask) {
        if (diffusion_model->get_desc() == "Wan2.2-TI2V-5B") {
            auto new_timesteps = std::vector<float>(static_cast<size_t>(init_latent.shape()[2]), timesteps[0]);

            if (!denoise_mask.empty()) {
                float value = denoise_mask.dim() == 5 ? denoise_mask.index(0, 0, 0, 0, 0) : denoise_mask.index(0, 0, 0, 0);
                if (value == 0.f) {
                    new_timesteps[0] = 0.f;
                }
            }
            return new_timesteps;
        } else {
            return timesteps;
        }
    }

    void preview_image(int step,
                       const sd::Tensor<float>& latents,
                       enum SDVersion version,
                       preview_t preview_mode,
                       std::function<void(int, int, sd_image_t*, bool, void*)> step_callback,
                       void* step_callback_data,
                       bool is_noisy) {
        if (preview_mode == PREVIEW_PROJ) {
            int patch_sz                     = 1;
            const float(*latent_rgb_proj)[3] = nullptr;
            float* latent_rgb_bias           = nullptr;
            bool is_video                    = preview_latent_tensor_is_video(latents);
            uint32_t dim                     = is_video ? static_cast<uint32_t>(latents.shape()[3]) : static_cast<uint32_t>(latents.shape()[2]);

            if (dim == 128) {
                if (sd_version_is_flux2(version)) {
                    latent_rgb_proj = flux2_latent_rgb_proj;
                    latent_rgb_bias = flux2_latent_rgb_bias;
                    patch_sz        = 2;
                }
            } else if (dim == 48) {
                if (sd_version_is_wan(version)) {
                    latent_rgb_proj = wan_22_latent_rgb_proj;
                    latent_rgb_bias = wan_22_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim == 16) {
                if (sd_version_is_sd3(version)) {
                    latent_rgb_proj = sd3_latent_rgb_proj;
                    latent_rgb_bias = sd3_latent_rgb_bias;
                } else if (sd_version_is_flux(version) || sd_version_is_z_image(version)) {
                    latent_rgb_proj = flux_latent_rgb_proj;
                    latent_rgb_bias = flux_latent_rgb_bias;
                } else if (sd_version_is_wan(version) || sd_version_is_qwen_image(version) || sd_version_is_anima(version)) {
                    latent_rgb_proj = wan_21_latent_rgb_proj;
                    latent_rgb_bias = wan_21_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim == 4) {
                if (sd_version_is_sdxl(version)) {
                    latent_rgb_proj = sdxl_latent_rgb_proj;
                    latent_rgb_bias = sdxl_latent_rgb_bias;
                } else if (sd_version_is_sd1(version) || sd_version_is_sd2(version)) {
                    latent_rgb_proj = sd_latent_rgb_proj;
                    latent_rgb_bias = sd_latent_rgb_bias;
                } else {
                    LOG_WARN("No latent to RGB projection known for this model");
                    return;
                }
            } else if (dim != 3) {
                LOG_WARN("No latent to RGB projection known for this model");
                return;
            }

            uint32_t frames     = is_video ? static_cast<uint32_t>(latents.shape()[2]) : 1;
            uint32_t img_width  = static_cast<uint32_t>(latents.shape()[0]) * patch_sz;
            uint32_t img_height = static_cast<uint32_t>(latents.shape()[1]) * patch_sz;

            uint8_t* data = (uint8_t*)malloc(frames * img_width * img_height * 3 * sizeof(uint8_t));
            GGML_ASSERT(data != nullptr);
            preview_latent_video(data, latents, latent_rgb_proj, latent_rgb_bias, patch_sz);
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; i++) {
                images[i] = {img_width, img_height, 3, data + i * img_width * img_height * 3};
            }
            step_callback(step, frames, images, is_noisy, step_callback_data);
            free(data);
            free(images);
            return;
        }

        if (preview_mode == PREVIEW_VAE || preview_mode == PREVIEW_TAE) {
            sd::Tensor<float> vae_latents;
            sd::Tensor<float> decoded;
            bool is_video = preview_latent_tensor_is_video(latents);
            if (preview_vae) {
                vae_latents = preview_vae->diffusion_to_vae_latents(latents);
                decoded     = preview_vae->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            } else {
                vae_latents = first_stage_model->diffusion_to_vae_latents(latents);
                decoded     = first_stage_model->decode(n_threads, vae_latents, vae_tiling_params, is_video, circular_x, circular_y, true);
            }
            if (decoded.empty()) {
                LOG_ERROR("preview decode failed at step %d", step);
                return;
            }

            is_video           = preview_latent_tensor_is_video(decoded);
            uint32_t frames    = is_video ? static_cast<uint32_t>(decoded.shape()[2]) : 1;
            sd_image_t* images = (sd_image_t*)malloc(frames * sizeof(sd_image_t));
            GGML_ASSERT(images != nullptr);
            for (uint32_t i = 0; i < frames; ++i) {
                images[i] = tensor_to_sd_image(decoded, static_cast<int>(i));
            }

            step_callback(step, frames, images, is_noisy, step_callback_data);
            for (uint32_t i = 0; i < frames; ++i) {
                free(images[i].data);
            }
            free(images);
            return;
        }

        if (preview_mode != PREVIEW_NONE) {
            LOG_WARN("Unsupported preview mode: %d", static_cast<int>(preview_mode));
        }
    }

    std::vector<float> prepare_sample_timesteps(float sigma,
                                                int shifted_timestep) {
        float t = denoiser->sigma_to_t(sigma);
        if (shifted_timestep > 0) {
            float shifted_t_float = t * (float(shifted_timestep) / float(TIMESTEPS));
            int64_t shifted_t     = static_cast<int64_t>(roundf(shifted_t_float));
            shifted_t             = std::max((int64_t)0, std::min((int64_t)(TIMESTEPS - 1), shifted_t));
            LOG_DEBUG("shifting timestep from %.2f to %" PRId64 " (sigma: %.4f)", t, shifted_t, sigma);
            return std::vector<float>{(float)shifted_t};
        }
        if (sd_version_is_anima(version)) {
            return std::vector<float>{t / static_cast<float>(TIMESTEPS)};
        }
        if (sd_version_is_z_image(version)) {
            return std::vector<float>{1000.f - t};
        }
        return std::vector<float>{t};
    }

    void adjust_sample_step_scalings(int shifted_timestep,
                                     const std::vector<float>& timesteps_vec,
                                     float c_in,
                                     float* c_skip,
                                     float* c_out) {
        GGML_ASSERT(c_skip != nullptr);
        GGML_ASSERT(c_out != nullptr);
        if (shifted_timestep <= 0) {
            return;
        }

        int64_t shifted_t_idx              = static_cast<int64_t>(roundf(timesteps_vec[0]));
        float shifted_sigma                = denoiser->t_to_sigma((float)shifted_t_idx);
        std::vector<float> shifted_scaling = denoiser->get_scalings(shifted_sigma);
        float shifted_c_skip               = shifted_scaling[0];
        float shifted_c_out                = shifted_scaling[1];
        float shifted_c_in                 = shifted_scaling[2];

        *c_skip = shifted_c_skip * c_in / shifted_c_in;
        *c_out  = shifted_c_out;
    }

    struct SamplePreviewContext {
        sd_preview_cb_t callback = nullptr;
        void* data               = nullptr;
        sd_preview_options_t options = {};
        int last_denoised_step       = -1;
        int last_noisy_step          = -1;
    };

    SamplePreviewContext prepare_sample_preview_context() {
        SamplePreviewContext preview;
        preview.callback = sd_get_preview_callback();
        preview.data     = sd_get_preview_callback_data();
        preview.options  = sd_get_preview_options();
        if (preview.options.struct_size != sizeof(sd_preview_options_t) ||
            preview.options.version != SD_PREVIEW_API_VERSION) {
            sd_preview_options_init(&preview.options);
            preview.options.mode          = sd_get_preview_mode();
            preview.options.step_interval = sd_get_preview_interval();
            preview.options.denoised      = sd_should_preview_denoised();
            preview.options.noisy         = sd_should_preview_noisy();
        }
        if (preview.options.step_interval <= 0) {
            preview.options.step_interval = 1;
        }
        if (preview.options.percent_point_count > SD_PREVIEW_MAX_PERCENT_POINTS) {
            preview.options.percent_point_count = SD_PREVIEW_MAX_PERCENT_POINTS;
        }
        return preview;
    }

    static int preview_percent_to_step(float percent, int total_steps) {
        if (total_steps <= 0) {
            return 0;
        }
        percent = std::max(0.0f, std::min(1.0f, percent));
        int step = static_cast<int>(std::ceil(percent * static_cast<float>(total_steps)));
        return std::max(1, std::min(total_steps, step));
    }

    static bool preview_step_matches_schedule(const sd_preview_options_t& options,
                                              int step,
                                              int total_steps) {
        if (step <= 0 || total_steps <= 0) {
            return false;
        }
        if (options.include_first_step && step == 1) {
            return true;
        }
        if (options.include_final_step && step == total_steps) {
            return true;
        }

        switch (options.schedule_mode) {
            case SD_PREVIEW_SCHEDULE_PERCENT_INTERVAL: {
                if (options.percent_interval <= 0.0f) {
                    return true;
                }
                float marker = options.percent_interval;
                while (marker <= 1.0f + 1e-6f) {
                    if (step == preview_percent_to_step(marker, total_steps)) {
                        return true;
                    }
                    marker += options.percent_interval;
                }
                return false;
            }
            case SD_PREVIEW_SCHEDULE_EXPLICIT_PERCENTS:
                for (uint32_t i = 0; i < options.percent_point_count; ++i) {
                    if (step == preview_percent_to_step(options.percent_points[i], total_steps)) {
                        return true;
                    }
                }
                return false;
            case SD_PREVIEW_SCHEDULE_EVERY_N_STEPS:
            default:
                return options.step_interval <= 1 || (step % options.step_interval) == 0;
        }
    }

    static bool should_emit_preview(SamplePreviewContext* preview,
                                    int step,
                                    size_t total_steps,
                                    bool is_noisy) {
        if (preview == nullptr || preview->callback == nullptr || preview->options.mode == PREVIEW_NONE) {
            return false;
        }
        if (is_noisy && !preview->options.noisy) {
            return false;
        }
        if (!is_noisy && !preview->options.denoised) {
            return false;
        }
        int logical_step = std::abs(step);
        if (logical_step <= 0) {
            return false;
        }
        int total = static_cast<int>(total_steps);
        if (!preview_step_matches_schedule(preview->options, logical_step, total)) {
            return false;
        }
        int& last_step = is_noisy ? preview->last_noisy_step : preview->last_denoised_step;
        if (last_step == logical_step) {
            return false;
        }
        last_step = logical_step;
        return true;
    }

    void report_sample_progress(int step, size_t total_steps, int64_t t0) {
        int64_t t1 = ggml_time_us();
        if (step > 0 || step == -(int)total_steps) {
            int showstep = std::abs(step);
            pretty_progress(showstep, (int)total_steps, (t1 - t0) / 1000000.f / showstep);
        }
    }

    void compute_sample_controls(const sd::Tensor<float>& control_image,
                                 const sd::Tensor<float>& noised_input,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const SDCondition& condition,
                                 std::vector<sd::Tensor<float>>* controls,
                                 std::vector<ggml_tensor*>* backend_controls,
                                 const char* pass_name) {
        GGML_ASSERT(controls != nullptr);
        GGML_ASSERT(backend_controls != nullptr);
        controls->clear();
        backend_controls->clear();
        if (control_image.empty() || control_net == nullptr) {
            return;
        }

        int64_t t0 = ggml_time_ms();
        auto backend_result = control_net->compute_backend(n_threads,
                                                           noised_input,
                                                           control_image,
                                                           timesteps_tensor,
                                                           condition.c_crossattn,
                                                           condition.c_vector);
        if (backend_result.has_value()) {
            *backend_controls = std::move(*backend_result);
            if (env_flag_enabled("SDCPP_TRACE_CONTROLNET")) {
                uint64_t control_bytes = 0;
                for (const auto* control : *backend_controls) {
                    if (control != nullptr) {
                        control_bytes += ggml_nbytes(control);
                    }
                }
                LOG_INFO("[ControlNet] pass=%s backend compute=%" PRId64 "ms controls=%zu gpu_bytes=%.2fMB host_materialize=0ms d2h=0",
                         pass_name ? pass_name : "unknown",
                         ggml_time_ms() - t0,
                         backend_controls->size(),
                         control_bytes / 1024.0 / 1024.0);
            }
            return;
        }

        auto control_result = control_net->compute(n_threads,
                                                   noised_input,
                                                   control_image,
                                                   timesteps_tensor,
                                                   condition.c_crossattn,
                                                   condition.c_vector);
        if (!control_result.has_value()) {
            LOG_ERROR("controlnet compute failed");
            return;
        }

        *controls = std::move(*control_result);
        if (env_flag_enabled("SDCPP_TRACE_CONTROLNET")) {
            LOG_INFO("[ControlNet] pass=%s host fallback compute=%" PRId64 "ms materialize=%" PRId64 "ms d2h=%.2fMB controls=%zu",
                     pass_name ? pass_name : "unknown",
                     ggml_time_ms() - t0,
                     control_net->get_last_materialize_ms(),
                     control_net->get_last_materialize_bytes() / 1024.0 / 1024.0,
                     controls->size());
        }
    }

    sd::Tensor<float> sample(const std::shared_ptr<DiffusionModel>& work_diffusion_model,
                             bool inverse_noise_scaling,
                             const sd::Tensor<float>& init_latent,
                             sd::Tensor<float> noise,
                             const SDCondition& cond,
                             const SDCondition& uncond,
                             const SDCondition& img_cond,
                             const SDCondition& id_cond,
                             const sd::Tensor<float>& control_image,
                             float control_strength,
                             const sd_guidance_params_t& guidance,
                             float eta,
                             float s_noise,
                             float dpmpp_sde_r,
                             dpmpp_sde_solver_t dpmpp_sde_solver,
                             int shifted_timestep,
                             sample_method_t method,
                             bool is_flow_denoiser,
                             const std::vector<float>& sigmas,
                             int start_merge_step,
                             const std::vector<sd::Tensor<float>>& ref_latents,
                             bool increase_ref_index,
                             const sd::Tensor<float>& denoise_mask,
                             const sd::Tensor<float>& vace_context,
                             float vace_strength,
                             const sd_cache_params_t* cache_params) {
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);
        float cfg_scale     = guidance.txt_cfg;
        float img_cfg_scale = guidance.img_cfg;
        float slg_scale     = guidance.slg.scale;

        sd_sample::SampleCacheRuntime cache_runtime = sd_sample::init_sample_cache_runtime(version,
                                                                                           cache_params,
                                                                                           denoiser.get(),
                                                                                           sigmas);
        size_t steps                                = sigmas.size() - 1;
        bool has_skiplayer                          = slg_scale != 0.0f && !skip_layers.empty();
        if (has_skiplayer && !sd_version_is_dit(version)) {
            has_skiplayer = false;
            LOG_WARN("SLG is incompatible with this model type");
        }

        int64_t t0                   = ggml_time_us();
        if (sd_version_is_marigold_iid(version) && method == DDIM_TRAILING_SAMPLE_METHOD && !noise.empty()) {
            auto build_zero_snr_alphas_cumprod = []() {
                constexpr double beta_start = 0.00085;
                constexpr double beta_end   = 0.0120;
                std::vector<double> alphas_bar_sqrt(TIMESTEPS);
                double product = 1.0;
                for (int i = 0; i < TIMESTEPS; ++i) {
                    const double beta_sqrt = std::sqrt(beta_start) +
                                             (std::sqrt(beta_end) - std::sqrt(beta_start)) *
                                                 (static_cast<double>(i) / static_cast<double>(TIMESTEPS - 1));
                    const double beta = beta_sqrt * beta_sqrt;
                    product *= (1.0 - beta);
                    alphas_bar_sqrt[i] = std::sqrt(product);
                }

                const double alpha0 = alphas_bar_sqrt.front();
                const double alphaT = alphas_bar_sqrt.back();
                const double denom  = alpha0 - alphaT;
                if (denom != 0.0) {
                    for (double& alpha : alphas_bar_sqrt) {
                        alpha = (alpha - alphaT) * alpha0 / denom;
                    }
                }

                std::vector<double> alphas_cumprod(TIMESTEPS);
                for (int i = 0; i < TIMESTEPS; ++i) {
                    alphas_cumprod[i] = alphas_bar_sqrt[i] * alphas_bar_sqrt[i];
                }
                return alphas_cumprod;
            };

            const std::vector<double> alphas_cumprod = build_zero_snr_alphas_cumprod();
            const int ddim_steps = std::max(1, static_cast<int>(steps));
            const int prev_step  = std::max(1, TIMESTEPS / ddim_steps);
            sd::Tensor<float> x  = std::move(noise);
            LOG_INFO("Marigold IID using direct Diffusers-compatible DDIM v-prediction sampler");

            for (int i = 0; i < ddim_steps; ++i) {
                if (i == 0) {
                    pretty_progress(0, ddim_steps, 0);
                }
                const int timestep = std::max(0,
                                              std::min(TIMESTEPS - 1,
                                                       static_cast<int>(std::round(TIMESTEPS - i * (static_cast<double>(TIMESTEPS) / ddim_steps))) - 1));
                const int prev_timestep = timestep - prev_step;

                sd::Tensor<float> timesteps_tensor({1}, std::vector<float>{static_cast<float>(timestep)});
                DiffusionParams diffusion_params;
                diffusion_params.x         = cond.c_concat.empty() ? &x : &cond.c_concat;
                diffusion_params.timesteps = &timesteps_tensor;
                diffusion_params.context   = cond.c_crossattn.empty() ? nullptr : &cond.c_crossattn;
                diffusion_params.c_concat  = cond.c_concat.empty() ? nullptr : &x;
                diffusion_params.y         = cond.c_vector.empty() ? nullptr : &cond.c_vector;

                auto model_output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                if (model_output_opt.empty()) {
                    LOG_ERROR("Marigold IID diffusion model compute failed");
                    if (work_diffusion_model) {
                        work_diffusion_model->free_compute_buffer();
                    }
                    return {};
                }
                sd::Tensor<float> model_output = std::move(model_output_opt);

                const double alpha_prod_t      = alphas_cumprod[timestep];
                const double alpha_prod_t_prev = prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0];
                const double beta_prod_t       = 1.0 - alpha_prod_t;
                sd::Tensor<float> pred_original_sample =
                    x * static_cast<float>(std::sqrt(alpha_prod_t)) -
                    model_output * static_cast<float>(std::sqrt(beta_prod_t));
                sd::Tensor<float> pred_epsilon =
                    model_output * static_cast<float>(std::sqrt(alpha_prod_t)) +
                    x * static_cast<float>(std::sqrt(beta_prod_t));

                const double variance =
                    ((1.0 - alpha_prod_t_prev) / beta_prod_t) *
                    (1.0 - alpha_prod_t / alpha_prod_t_prev);
                const double std_dev_t = eta * std::sqrt(std::max(variance, 0.0));
                const double direction_scale = std::sqrt(std::max(0.0, 1.0 - alpha_prod_t_prev - std_dev_t * std_dev_t));

                x = pred_original_sample * static_cast<float>(std::sqrt(alpha_prod_t_prev)) +
                    pred_epsilon * static_cast<float>(direction_scale);
                if (eta > 0.0f && std_dev_t > 0.0) {
                    x += sd::Tensor<float>::randn_like(x, sampler_rng) * static_cast<float>(std_dev_t);
                }

                const float avg_step_seconds = (ggml_time_us() - t0) / 1000000.f / static_cast<float>(i + 1);
                pretty_progress(i + 1, ddim_steps, avg_step_seconds);
                report_sample_progress(i + 1, ddim_steps, t0);
            }

            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return x;
        }

        sd::Tensor<float> x_t        = !noise.empty()
                                           ? denoiser->noise_scaling(sigmas[0], noise, init_latent)
                                           : init_latent;
        sd::Tensor<float> denoised   = x_t;
        SamplePreviewContext preview = prepare_sample_preview_context();

        auto denoise = [&](const sd::Tensor<float>& x, float sigma, int step) -> sd::Tensor<float> {
            const bool trace_controlnet = env_flag_enabled("SDCPP_TRACE_CONTROLNET");
            const int64_t step_start_ms = trace_controlnet ? ggml_time_ms() : 0;
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            std::vector<float> timesteps_vec = prepare_sample_timesteps(sigma, shifted_timestep);
            timesteps_vec                    = process_timesteps(timesteps_vec, init_latent, denoise_mask);
            adjust_sample_step_scalings(shifted_timestep, timesteps_vec, c_in, &c_skip, &c_out);

            sd::Tensor<float> timesteps_tensor({static_cast<int64_t>(timesteps_vec.size())}, timesteps_vec);
            sd::Tensor<float> guidance_tensor({1}, std::vector<float>{guidance.distilled_guidance});
            sd::Tensor<float> noised_input = x * c_in;
            if (!denoise_mask.empty() && version == VERSION_WAN2_2_TI2V) {
                noised_input = noised_input * denoise_mask + init_latent * (1.0f - denoise_mask);
            }

            if (cache_runtime.spectrum_enabled && cache_runtime.spectrum.should_predict()) {
                cache_runtime.spectrum.predict(&denoised);
                if (!denoise_mask.empty()) {
                    denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
                }
                if (should_emit_preview(&preview, step, steps, false)) {
                    preview_image(step, denoised, version, preview.options.mode, preview.callback, preview.data, false);
                }
                report_sample_progress(step, steps, t0);
                return denoised;
            }

            if (should_emit_preview(&preview, step, steps, true)) {
                preview_image(step, noised_input, version, preview.options.mode, preview.callback, preview.data, true);
            }

            sd::Tensor<float> cond_out;
            sd::Tensor<float> uncond_out;
            sd::Tensor<float> img_cond_out;
            sd::Tensor<float> skip_cond_out;
            sd_sample::SampleStepCacheDispatcher step_cache(cache_runtime, step, sigma);
            std::vector<sd::Tensor<float>> controls;
            std::vector<ggml_tensor*> backend_controls;
            DiffusionParams diffusion_params;
            diffusion_params.x                  = &noised_input;
            diffusion_params.timesteps          = &timesteps_tensor;
            diffusion_params.guidance           = &guidance_tensor;
            diffusion_params.ref_latents        = &ref_latents;
            diffusion_params.increase_ref_index = increase_ref_index;
            diffusion_params.controls           = &controls;
            diffusion_params.backend_controls   = &backend_controls;
            diffusion_params.control_strength   = control_strength;
            diffusion_params.vace_context       = vace_context.empty() ? nullptr : &vace_context;
            diffusion_params.vace_strength      = vace_strength;
            diffusion_params.skip_layers        = nullptr;

            compute_sample_controls(control_image,
                                    noised_input,
                                    timesteps_tensor,
                                    cond,
                                    &controls,
                                    &backend_controls,
                                    "cond");

            auto run_condition = [&](const SDCondition& condition,
                                     const char* pass_name,
                                     const sd::Tensor<float>* c_concat_override = nullptr,
                                     const std::vector<int>* local_skip_layers  = nullptr) -> sd::Tensor<float> {
                diffusion_params.context     = condition.c_crossattn.empty() ? nullptr : &condition.c_crossattn;
                diffusion_params.c_concat    = c_concat_override != nullptr ? c_concat_override : (condition.c_concat.empty() ? nullptr : &condition.c_concat);
                diffusion_params.y           = condition.c_vector.empty() ? nullptr : &condition.c_vector;
                diffusion_params.t5_ids      = condition.c_t5_ids.empty() ? nullptr : &condition.c_t5_ids;
                diffusion_params.t5_weights  = condition.c_t5_weights.empty() ? nullptr : &condition.c_t5_weights;
                diffusion_params.skip_layers = local_skip_layers;

                sd::Tensor<float> cached_output;
                if (step_cache.before_condition(&condition, noised_input, &cached_output)) {
                    if (trace_controlnet) {
                        LOG_INFO("[Denoise] step=%d pass=%s cache_hit=true backend_controls=%zu host_controls=%zu",
                                 step,
                                 pass_name ? pass_name : "unknown",
                                 backend_controls.size(),
                                 controls.size());
                    }
                    return std::move(cached_output);
                }

                int64_t unet_start_ms = trace_controlnet ? ggml_time_ms() : 0;
                auto output_opt = work_diffusion_model->compute(n_threads, diffusion_params);
                int64_t unet_ms = trace_controlnet ? ggml_time_ms() - unet_start_ms : 0;
                if (output_opt.empty()) {
                    LOG_ERROR("diffusion model compute failed");
                    return sd::Tensor<float>();
                }
                if (trace_controlnet) {
                    LOG_INFO("[Denoise] step=%d pass=%s unet_compute=%" PRId64 "ms backend_controls=%zu host_controls=%zu",
                             step,
                             pass_name ? pass_name : "unknown",
                             unet_ms,
                             backend_controls.size(),
                             controls.size());
                }

                step_cache.after_condition(&condition, noised_input, output_opt);
                return output_opt;
            };

            if (start_merge_step == -1 || step <= start_merge_step) {
                cond_out = run_condition(cond, "cond");
                if (cond_out.empty()) {
                    return {};
                }
            } else {
                GGML_ASSERT(!id_cond.empty());
                cond_out = run_condition(id_cond,
                                         "id_cond",
                                         cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (cond_out.empty()) {
                    return {};
                }
            }

            if (!uncond.empty()) {
                if (!step_cache.is_step_skipped()) {
                    compute_sample_controls(control_image,
                                            noised_input,
                                            timesteps_tensor,
                                            uncond,
                                            &controls,
                                            &backend_controls,
                                            "uncond");
                }
                uncond_out = run_condition(uncond, "uncond");
                if (uncond_out.empty()) {
                    return {};
                }
            }
            if (!img_cond.empty()) {
                img_cond_out = run_condition(img_cond,
                                             "img_cond",
                                             cond.c_concat.empty() ? nullptr : &cond.c_concat);
                if (img_cond_out.empty()) {
                    return {};
                }
            }
            bool is_skiplayer_step = has_skiplayer &&
                                     step > (int)(guidance.slg.layer_start * static_cast<int>(sigmas.size())) &&
                                     step < (int)(guidance.slg.layer_end * static_cast<int>(sigmas.size()));
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                if (!step_cache.is_step_skipped()) {
                    skip_cond_out = run_condition(cond,
                                                  "skip_cond",
                                                  cond.c_concat.empty() ? nullptr : &cond.c_concat,
                                                  &skip_layers);
                    if (skip_cond_out.empty()) {
                        return {};
                    }
                }
            }

            GGML_ASSERT(!cond_out.empty());
            sd::Tensor<float> latent_result = cond_out;
            if (!uncond_out.empty()) {
                if (!img_cond_out.empty()) {
                    latent_result = uncond_out +
                                    img_cfg_scale * (img_cond_out - uncond_out) +
                                    cfg_scale * (cond_out - img_cond_out);
                } else {
                    latent_result = uncond_out + cfg_scale * (cond_out - uncond_out);
                }
            } else if (!img_cond_out.empty()) {
                latent_result = img_cond_out + cfg_scale * (cond_out - img_cond_out);
            }

            if (is_skiplayer_step && !skip_cond_out.empty()) {
                latent_result += (cond_out - skip_cond_out) * slg_scale;
            }
            denoised = latent_result * c_out + x * c_skip;
            if (cache_runtime.spectrum_enabled) {
                cache_runtime.spectrum.update(denoised);
            }
            if (!denoise_mask.empty()) {
                denoised = denoised * denoise_mask + init_latent * (1.0f - denoise_mask);
            }
            if (should_emit_preview(&preview, step, steps, false)) {
                preview_image(step, denoised, version, preview.options.mode, preview.callback, preview.data, false);
            }
            if (trace_controlnet) {
                LOG_INFO("[Denoise] step=%d total=%" PRId64 "ms", step, ggml_time_ms() - step_start_ms);
            }
            report_sample_progress(step, steps, t0);
            return denoised;
        };

        auto x0_opt = sample_k_diffusion(method,
                                         denoise,
                                         x_t,
                                         sigmas,
                                         sampler_rng,
                                         eta,
                                         s_noise,
                                         dpmpp_sde_r,
                                         dpmpp_sde_solver,
                                         is_flow_denoiser);
        if (x0_opt.empty()) {
            LOG_ERROR("Diffusion model sampling failed");
            if (control_net) {
                control_net->free_control_ctx();
                control_net->free_compute_buffer();
            }
            if (work_diffusion_model) {
                work_diffusion_model->free_compute_buffer();
            }
            return {};
        }

        auto x0 = std::move(x0_opt);
        sd_sample::log_sample_cache_summary(cache_runtime, steps);
        if (inverse_noise_scaling) {
            x0 = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x0);
        }

        if (control_net) {
            control_net->free_control_ctx();
            control_net->free_compute_buffer();
        }
        if (work_diffusion_model) {
            work_diffusion_model->free_compute_buffer();
        }
        return x0;
    }

    int get_vae_scale_factor() {
        return first_stage_model->get_scale_factor();
    }

    int get_diffusion_model_down_factor() {
        int down_factor = 8;  // unet
        if (sd_version_is_dit(version)) {
            if (sd_version_is_wan(version)) {
                down_factor = 2;
            } else {
                down_factor = 1;
            }
        }
        return down_factor;
    }

    int get_latent_channel() {
        int latent_channel = 4;
        if (sd_version_is_marigold_iid(version)) {
            return 8;
        }
        if (sd_version_is_dit(version)) {
            if (version == VERSION_WAN2_2_TI2V) {
                latent_channel = 48;
            } else if (version == VERSION_CHROMA_RADIANCE) {
                latent_channel = 3;
            } else if (sd_version_is_flux2(version)) {
                latent_channel = 128;
            } else {
                latent_channel = 16;
            }
        }
        return latent_channel;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
    }

    sd::Tensor<float> generate_init_latent(int width,
                                           int height,
                                           int frames = 1,
                                           bool video = false) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;
        int T                = frames;
        if (sd_version_is_wan(version)) {
            T = ((T - 1) / 4) + 1;
        }
        int C = get_latent_channel();
        if (video) {
            return sd::zeros<float>({W, H, T, C, 1});
        }
        return sd::zeros<float>({W, H, C, 1});
    }

    sd::Tensor<float> encode_to_vae_latents(const sd::Tensor<float>& x) {
        auto latents = first_stage_model->encode(n_threads, x, vae_tiling_params, circular_x, circular_y);
        if (latents.empty()) {
            return {};
        }
        latents = first_stage_model->vae_output_to_latents(latents, rng);
        return latents;
    }

    sd::Tensor<float> encode_first_stage(const sd::Tensor<float>& x) {
        auto latents = encode_to_vae_latents(x);
        if (latents.empty()) {
            return {};
        }
        if (version != VERSION_SD1_PIX2PIX) {
            latents = first_stage_model->vae_to_diffusion_latents(latents);
        }
        return latents;
    }

    sd::Tensor<float> decode_first_stage(const sd::Tensor<float>& x, bool decode_video = false) {
        auto latents = first_stage_model->diffusion_to_vae_latents(x);
        return first_stage_model->decode(n_threads, latents, vae_tiling_params, decode_video, circular_x, circular_y);
    }

    void set_flow_shift(float flow_shift = INFINITY) {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        if (flow_denoiser) {
            if (flow_shift == INFINITY) {
                flow_shift = default_flow_shift;
            }
            flow_denoiser->set_shift(flow_shift);
        }
    }

    bool is_flow_denoiser() {
        auto flow_denoiser = std::dynamic_pointer_cast<DiscreteFlowDenoiser>(denoiser);
        return !!flow_denoiser;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    if (!strcmp(str, "dpm++sde")) {
        return DPMPP_SDE_SAMPLE_METHOD;
    }
    if (!strcmp(str, "dpm++2m_sde")) {
        return DPMPP2M_SDE_SAMPLE_METHOD;
    }
    if (!strcmp(str, "dpm++2m_sde_heun")) {
        return DPMPP2M_SDE_HEUN_SAMPLE_METHOD;
    }
    if (!strcmp(str, "dpm++3m_sde")) {
        return DPMPP3M_SDE_SAMPLE_METHOD;
    }
    return SAMPLE_METHOD_COUNT;
}

const char* dpmpp_sde_solver_to_str[] = {
    "midpoint",
    "heun",
};

static const char* dpmpp_sde_solver_name(enum dpmpp_sde_solver_t solver) {
    if (solver < DPMPP_SDE_SOLVER_COUNT) {
        return dpmpp_sde_solver_to_str[solver];
    }
    return "default";
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "beta",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "flux2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->s_noise                     = 1.0f;
    sample_params->dpmpp_sde_r                 = 0.5f;
    sample_params->dpmpp_sde_solver            = DPMPP_SDE_SOLVER_COUNT;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "s_noise: %.2f, "
             "dpmpp_sde_r: %.2f, "
             "dpmpp_sde_solver: %s, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->s_noise,
             sample_params->dpmpp_sde_r,
             dpmpp_sde_solver_name(sample_params->dpmpp_sde_solver),
             sample_params->shifted_timestep,
             sample_params->flow_shift);

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_img_gen_params->cache);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled));
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_gpu_resource_private_t {
    sd_gpu_handle_t handle = 0;
    enum sd_gpu_resource_kind_t kind = SD_GPU_RESOURCE_TENSOR;
    uint32_t refcount = 1;
    uint32_t flags = 0;
    int device_index = 0;
    enum sd_tensor_layout_t layout = SD_LAYOUT_WHCN_GGML;
    std::string debug_name;
    std::unique_ptr<GgmlBackendTensorResource> tensor;
};

struct sd_ctx_params_snapshot_t {
    sd_ctx_params_t params{};
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string control_net_path;
    std::string photo_maker_path;
    std::string tensor_type_rules;
    bool valid = false;
};

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
    sd_gpu_handle_t next_gpu_handle = 1;
    std::unordered_map<sd_gpu_handle_t, std::shared_ptr<sd_gpu_resource_private_t>> gpu_resources;
    sd_ctx_params_snapshot_t init_params;
    sd_ctx_t* vae_decode_bridge_ctx = nullptr;
};

static const char* snapshot_c_str(const std::string& value) {
    return value.empty() ? nullptr : value.c_str();
}

static void sd_ctx_snapshot_bind_strings(sd_ctx_params_snapshot_t* snapshot) {
    if (snapshot == nullptr) {
        return;
    }
    snapshot->params.model_path = snapshot_c_str(snapshot->model_path);
    snapshot->params.clip_l_path = snapshot_c_str(snapshot->clip_l_path);
    snapshot->params.clip_g_path = snapshot_c_str(snapshot->clip_g_path);
    snapshot->params.clip_vision_path = snapshot_c_str(snapshot->clip_vision_path);
    snapshot->params.t5xxl_path = snapshot_c_str(snapshot->t5xxl_path);
    snapshot->params.llm_path = snapshot_c_str(snapshot->llm_path);
    snapshot->params.llm_vision_path = snapshot_c_str(snapshot->llm_vision_path);
    snapshot->params.diffusion_model_path = snapshot_c_str(snapshot->diffusion_model_path);
    snapshot->params.high_noise_diffusion_model_path = snapshot_c_str(snapshot->high_noise_diffusion_model_path);
    snapshot->params.vae_path = snapshot_c_str(snapshot->vae_path);
    snapshot->params.taesd_path = snapshot_c_str(snapshot->taesd_path);
    snapshot->params.control_net_path = snapshot_c_str(snapshot->control_net_path);
    snapshot->params.photo_maker_path = snapshot_c_str(snapshot->photo_maker_path);
    snapshot->params.tensor_type_rules = snapshot_c_str(snapshot->tensor_type_rules);
}

static void sd_ctx_capture_init_params(sd_ctx_t* sd_ctx, const sd_ctx_params_t* params) {
    if (sd_ctx == nullptr || params == nullptr) {
        return;
    }
    auto& snapshot = sd_ctx->init_params;
    snapshot = {};
    snapshot.params = *params;
    snapshot.model_path = SAFE_STR(params->model_path);
    snapshot.clip_l_path = SAFE_STR(params->clip_l_path);
    snapshot.clip_g_path = SAFE_STR(params->clip_g_path);
    snapshot.clip_vision_path = SAFE_STR(params->clip_vision_path);
    snapshot.t5xxl_path = SAFE_STR(params->t5xxl_path);
    snapshot.llm_path = SAFE_STR(params->llm_path);
    snapshot.llm_vision_path = SAFE_STR(params->llm_vision_path);
    snapshot.diffusion_model_path = SAFE_STR(params->diffusion_model_path);
    snapshot.high_noise_diffusion_model_path = SAFE_STR(params->high_noise_diffusion_model_path);
    snapshot.vae_path = SAFE_STR(params->vae_path);
    snapshot.taesd_path = SAFE_STR(params->taesd_path);
    snapshot.control_net_path = SAFE_STR(params->control_net_path);
    snapshot.photo_maker_path = SAFE_STR(params->photo_maker_path);
    snapshot.tensor_type_rules = SAFE_STR(params->tensor_type_rules);
    snapshot.params.embeddings = nullptr;
    snapshot.params.embedding_count = 0;
    sd_ctx_snapshot_bind_strings(&snapshot);
    snapshot.valid = true;
}

static bool sd_ctx_make_vae_decode_bridge_params(sd_ctx_t* sd_ctx, sd_ctx_params_t* out) {
    if (sd_ctx == nullptr || out == nullptr || !sd_ctx->init_params.valid) {
        return false;
    }
    sd_ctx_snapshot_bind_strings(&sd_ctx->init_params);
    *out = sd_ctx->init_params.params;
    out->vae_decode_only = true;
    out->free_params_immediately = false;
    out->clip_l_path = nullptr;
    out->clip_g_path = nullptr;
    out->clip_vision_path = nullptr;
    out->t5xxl_path = nullptr;
    out->llm_path = nullptr;
    out->llm_vision_path = nullptr;
    out->high_noise_diffusion_model_path = nullptr;
    out->taesd_path = nullptr;
    out->control_net_path = nullptr;
    out->photo_maker_path = nullptr;
    out->embeddings = nullptr;
    out->embedding_count = 0;
    return true;
}

static sd_ctx_t* sd_ctx_get_vae_decode_bridge_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr) {
        return nullptr;
    }
    if (sd_ctx->vae_decode_bridge_ctx != nullptr) {
        return sd_ctx->vae_decode_bridge_ctx;
    }
    sd_ctx_params_t bridge_params;
    if (!sd_ctx_make_vae_decode_bridge_params(sd_ctx, &bridge_params)) {
        LOG_ERROR("VAE encoded-latent bridge could not recreate a decode-only context; original context parameters were not captured");
        return nullptr;
    }
    LOG_INFO("VAE encoded-latent bridge creating cached decode-only context");
    sd_ctx->vae_decode_bridge_ctx = new_sd_ctx(&bridge_params);
    if (sd_ctx->vae_decode_bridge_ctx == nullptr) {
        LOG_ERROR("VAE encoded-latent bridge failed to create decode-only context");
    }
    return sd_ctx->vae_decode_bridge_ctx;
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = new (std::nothrow) sd_ctx_t();
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        delete sd_ctx;
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        delete sd_ctx;
        return nullptr;
    }
    sd_ctx_capture_init_params(sd_ctx, sd_ctx_params);
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr) {
        return;
    }
    if (sd_ctx->vae_decode_bridge_ctx != nullptr) {
        free_sd_ctx(sd_ctx->vae_decode_bridge_ctx);
        sd_ctx->vae_decode_bridge_ctx = nullptr;
    }
    sd_ctx->gpu_resources.clear();
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    delete sd_ctx;
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_anima(sd_ctx->sd->version)) {
            return ER_SDE_SAMPLE_METHOD;
        }
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd_ctx->sd->denoiser);
        if (edm_v_denoiser) {
            return EXPONENTIAL_SCHEDULER;
        }
    }
    if (sample_method == LCM_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

static int64_t resolve_seed(int64_t seed) {
    if (seed >= 0) {
        return seed;
    }
    srand((int)time(nullptr));
    return rand();
}

static enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sample_method == SAMPLE_METHOD_COUNT) {
        return sd_get_default_sample_method(sd_ctx);
    }
    return sample_method;
}

static scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx,
                                     scheduler_t scheduler,
                                     enum sample_method_t sample_method) {
    if (scheduler == SCHEDULER_COUNT) {
        return sd_get_default_scheduler(sd_ctx, sample_method);
    }
    return scheduler;
}

static float resolve_eta(sd_ctx_t* sd_ctx,
                         float eta,
                         enum sample_method_t sample_method) {
    if (eta == INFINITY) {
        switch (sample_method) {
            case DDIM_TRAILING_SAMPLE_METHOD:
            case TCD_SAMPLE_METHOD:
            case RES_MULTISTEP_SAMPLE_METHOD:
            case RES_2S_SAMPLE_METHOD:
                return 0.0f;
            case ER_SDE_SAMPLE_METHOD:
            case EULER_A_SAMPLE_METHOD:
            case DPMPP2S_A_SAMPLE_METHOD:
            case DPMPP_SDE_SAMPLE_METHOD:
            case DPMPP_SDE_GPU_SAMPLE_METHOD:
            case DPMPP2M_SDE_SAMPLE_METHOD:
            case DPMPP2M_SDE_GPU_SAMPLE_METHOD:
            case DPMPP2M_SDE_HEUN_SAMPLE_METHOD:
            case DPMPP2M_SDE_HEUN_GPU_SAMPLE_METHOD:
            case DPMPP3M_SDE_SAMPLE_METHOD:
            case DPMPP3M_SDE_GPU_SAMPLE_METHOD:
                return 1.0f;
            default:;
        }
        return 0.0f;
    }
    return eta;
}

static enum dpmpp_sde_solver_t resolve_dpmpp_sde_solver(enum sample_method_t sample_method,
                                                         enum dpmpp_sde_solver_t solver) {
    if (solver != DPMPP_SDE_SOLVER_COUNT) {
        return solver;
    }
    switch (sample_method) {
        case DPMPP2M_SDE_HEUN_SAMPLE_METHOD:
        case DPMPP2M_SDE_HEUN_GPU_SAMPLE_METHOD:
            return DPMPP_SDE_SOLVER_HEUN;
        default:
            return DPMPP_SDE_SOLVER_MIDPOINT;
    }
}

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width                                = -1;
    int height                               = -1;
    int clip_skip                            = -1;
    int vae_scale_factor                     = -1;
    int diffusion_model_down_factor          = -1;
    int64_t seed                             = -1;
    bool use_uncond                          = false;
    bool use_img_cond                        = false;
    bool use_high_noise_uncond               = false;
    bool use_high_noise_img_cond             = false;
    const sd_cache_params_t* cache_params    = nullptr;
    int batch_count                          = 1;
    int shifted_timestep                     = 0;
    float strength                           = 1.f;
    float control_strength                   = 0.f;
    float eta                                = 0.f;
    bool increase_ref_index                  = false;
    bool auto_resize_ref_image               = false;
    sd_guidance_params_t guidance            = {};
    sd_guidance_params_t high_noise_guidance = {};
    sd_pm_params_t pm_params                 = {};
    int frames                               = -1;
    float vace_strength                      = 1.f;

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        increase_ref_index          = sd_img_gen_params->increase_ref_index;
        auto_resize_ref_image       = sd_img_gen_params->auto_resize_ref_image;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd_ctx);
    }

    GenerationRequest(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt                      = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_vid_gen_params->negative_prompt);
        width                       = sd_vid_gen_params->width;
        height                      = sd_vid_gen_params->height;
        frames                      = (sd_vid_gen_params->video_frames - 1) / 4 * 4 + 1;
        clip_skip                   = sd_vid_gen_params->clip_skip;
        vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
        seed                        = sd_vid_gen_params->seed;
        cache_params                = &sd_vid_gen_params->cache;
        vace_strength               = sd_vid_gen_params->vace_strength;
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        resolve(sd_ctx);
    }

    void align_generation_request_size() {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(width, spatial_multiple);
        int height_offset    = align_up_offset(height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = width;
        int original_height = height;

        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)",
                 original_width,
                 original_height,
                 width,
                 height,
                 spatial_multiple);
    }

    static void resolve_guidance(sd_ctx_t* sd_ctx,
                                 sd_guidance_params_t* guidance,
                                 bool* use_uncond,
                                 bool* use_img_cond,
                                 const char* stage_name = nullptr) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_cond != nullptr);
        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        // img_cfg == txt_cfg means that img_cfg is not used
        if (!std::isfinite(guidance->img_cfg)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (!sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version)) {
            guidance->img_cfg = guidance->txt_cfg;
        }

        if (guidance->txt_cfg != 1.f) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != guidance->txt_cfg) {
            *use_img_cond = true;
            *use_uncond   = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void resolve(sd_ctx_t* sd_ctx) {
        align_generation_request_size();
        seed = resolve_seed(seed);

        resolve_guidance(sd_ctx, &guidance, &use_uncond, &use_img_cond);
        if (sd_ctx->sd->high_noise_diffusion_model) {
            resolve_guidance(sd_ctx,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_cond,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd_ctx->sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }
};

struct SamplePlan {
    enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
    enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
    float eta                                     = 0.f;
    float high_noise_eta                          = 0.f;
    float s_noise                                 = 1.f;
    float high_noise_s_noise                      = 1.f;
    float dpmpp_sde_r                             = 0.5f;
    enum dpmpp_sde_solver_t dpmpp_sde_solver      = DPMPP_SDE_SOLVER_MIDPOINT;
    enum dpmpp_sde_solver_t high_noise_dpmpp_sde_solver = DPMPP_SDE_SOLVER_MIDPOINT;
    int sample_steps                              = 0;
    int high_noise_sample_steps                   = 0;
    int total_steps                               = 0;
    float moe_boundary                            = 0.f;
    int start_merge_step                          = -1;
    std::vector<float> sigmas;

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_img_gen_params_t* sd_img_gen_params,
               const GenerationRequest& request) {
        sample_method = sd_img_gen_params->sample_params.sample_method;
        eta           = sd_img_gen_params->sample_params.eta;
        s_noise       = sd_img_gen_params->sample_params.s_noise;
        dpmpp_sde_r   = sd_img_gen_params->sample_params.dpmpp_sde_r;
        dpmpp_sde_solver = sd_img_gen_params->sample_params.dpmpp_sde_solver;
        sample_steps  = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd_ctx, &request, &sd_img_gen_params->sample_params);
    }

    SamplePlan(sd_ctx_t* sd_ctx,
               const sd_vid_gen_params_t* sd_vid_gen_params,
               const GenerationRequest& request) {
        sample_method = sd_vid_gen_params->sample_params.sample_method;
        eta           = sd_vid_gen_params->sample_params.eta;
        s_noise       = sd_vid_gen_params->sample_params.s_noise;
        dpmpp_sde_solver = sd_vid_gen_params->sample_params.dpmpp_sde_solver;
        sample_steps  = sd_vid_gen_params->sample_params.sample_steps;
        if (sd_ctx->sd->high_noise_diffusion_model) {
            high_noise_sample_steps  = sd_vid_gen_params->high_noise_sample_params.sample_steps;
            high_noise_sample_method = sd_vid_gen_params->high_noise_sample_params.sample_method;
            high_noise_eta           = sd_vid_gen_params->high_noise_sample_params.eta;
            high_noise_s_noise       = sd_vid_gen_params->high_noise_sample_params.s_noise;
            high_noise_dpmpp_sde_solver = sd_vid_gen_params->high_noise_sample_params.dpmpp_sde_solver;
        }
        moe_boundary = sd_vid_gen_params->moe_boundary;
        resolve(sd_ctx, &request, &sd_vid_gen_params->sample_params);
    }

    void resolve(sd_ctx_t* sd_ctx,
                 const GenerationRequest* request,
                 const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd_ctx, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd_ctx,
                                                      sample_params->scheduler,
                                                      sample_method);
            sigmas                = sd_ctx->sd->denoiser->get_sigmas(total_steps,
                                                                     sd_ctx->sd->get_image_seq_len(request->height, request->width),
                                                                     scheduler,
                                                                     sd_ctx->sd->version);
        }

        eta = resolve_eta(sd_ctx, eta, sample_method);
        dpmpp_sde_solver = resolve_dpmpp_sde_solver(sample_method, dpmpp_sde_solver);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_DEBUG("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd_ctx,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd_ctx, high_noise_eta, high_noise_sample_method);
            high_noise_dpmpp_sde_solver = resolve_dpmpp_sde_solver(high_noise_sample_method, high_noise_dpmpp_sde_solver);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }

        if (sd_ctx->sd->use_pmid) {
            start_merge_step = int(sd_ctx->sd->pmid_model->style_strength / 100.f * total_steps);
            LOG_INFO("PHOTOMAKER: start_merge_step: %d", start_merge_step);
        }
    }
};

struct ImageGenerationLatents {
    sd::Tensor<float> init_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    sd::Tensor<float> control_image;
    std::vector<sd::Tensor<float>> ref_images;
    std::vector<sd::Tensor<float>> ref_latents;
    sd::Tensor<float> denoise_mask;
    sd::Tensor<float> clip_vision_output;
    sd::Tensor<float> vace_context;
    int64_t ref_image_num = 0;
};

struct ImageGenerationEmbeds {
    SDCondition cond;
    SDCondition uncond;
    SDCondition img_cond;
    SDCondition id_cond;
};

struct CircularAxesState {
    bool circular_x = false;
    bool circular_y = false;
};

static CircularAxesState configure_image_vae_axes(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  const GenerationRequest& request) {
    CircularAxesState original_axes = {sd_ctx->sd->circular_x, sd_ctx->sd->circular_y};

    if (!sd_img_gen_params->vae_tiling_params.enabled) {
        if (sd_ctx->sd->first_stage_model) {
            sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        if (sd_ctx->sd->preview_vae) {
            sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
        }
        return original_axes;
    }

    int tile_size_x, tile_size_y;
    float overlap;
    int latent_size_x = request.width / request.vae_scale_factor;
    int latent_size_y = request.height / request.vae_scale_factor;
    sd_ctx->sd->first_stage_model->get_tile_sizes(tile_size_x,
                                                  tile_size_y,
                                                  overlap,
                                                  sd_img_gen_params->vae_tiling_params,
                                                  latent_size_x,
                                                  latent_size_y);

    sd_ctx->sd->circular_x = sd_ctx->sd->circular_x && (tile_size_x >= latent_size_x);
    sd_ctx->sd->circular_y = sd_ctx->sd->circular_y && (tile_size_y >= latent_size_y);

    if (sd_ctx->sd->first_stage_model) {
        sd_ctx->sd->first_stage_model->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }
    if (sd_ctx->sd->preview_vae) {
        sd_ctx->sd->preview_vae->set_circular_axes(sd_ctx->sd->circular_x, sd_ctx->sd->circular_y);
    }

    sd_ctx->sd->circular_x = original_axes.circular_x && (tile_size_x < latent_size_x);
    sd_ctx->sd->circular_y = original_axes.circular_y && (tile_size_y < latent_size_y);

    return original_axes;
}

static void restore_image_vae_axes(sd_ctx_t* sd_ctx, const CircularAxesState& original_axes) {
    sd_ctx->sd->circular_x = original_axes.circular_x;
    sd_ctx->sd->circular_y = original_axes.circular_y;
}

class ImageVaeAxesGuard {
private:
    sd_ctx_t* sd_ctx = nullptr;
    CircularAxesState original_axes;

public:
    ImageVaeAxesGuard(sd_ctx_t* sd_ctx,
                      const sd_img_gen_params_t* sd_img_gen_params,
                      const GenerationRequest& request)
        : sd_ctx(sd_ctx),
          original_axes(configure_image_vae_axes(sd_ctx, sd_img_gen_params, request)) {}

    ~ImageVaeAxesGuard() {
        restore_image_vae_axes(sd_ctx, original_axes);
    }

    ImageVaeAxesGuard(const ImageVaeAxesGuard&)            = delete;
    ImageVaeAxesGuard& operator=(const ImageVaeAxesGuard&) = delete;
};

static sd::Tensor<float> encode_image_tensor_for_generation(sd_ctx_t* sd_ctx,
                                                            const sd::Tensor<float>& image_tensor,
                                                            const char* label);

static std::optional<ImageGenerationLatents> prepare_image_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_img_gen_params_t* sd_img_gen_params,
                                                                              GenerationRequest* request,
                                                                              SamplePlan* plan) {
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> init_image_tensor;
    sd::Tensor<float> control_image_tensor;
    sd::Tensor<float> mask_image_tensor;

    if (sd_img_gen_params->init_image.data != nullptr) {
        LOG_INFO("IMG2IMG");

        if (request->strength < 1.f) {
            size_t t_enc = static_cast<size_t>(plan->sample_steps * request->strength);
            if (t_enc == static_cast<size_t>(plan->sample_steps)) {
                t_enc--;
            }
            LOG_INFO("target t_enc is %zu steps", t_enc);
            std::vector<float> sigma_sched;
            sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
            plan->sigmas       = std::move(sigma_sched);
            plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
        }

        init_image_tensor = sd_image_to_tensor(sd_img_gen_params->init_image, request->width, request->height);
    }

    if (sd_img_gen_params->mask_image.data != nullptr) {
        mask_image_tensor = sd_image_to_tensor(sd_img_gen_params->mask_image, request->width, request->height);
        mask_image_tensor = sd::ops::round(mask_image_tensor);
    }

    if (sd_img_gen_params->control_image.data != nullptr) {
        int64_t control_convert_start = ggml_time_ms();
        control_image_tensor = sd_image_to_tensor(sd_img_gen_params->control_image, request->width, request->height);
        if (StableDiffusionGGML::env_flag_enabled("SDCPP_TRACE_CONTROLNET")) {
            LOG_INFO("[ControlNet] control image tensor conversion completed in %" PRId64 "ms shape=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
                     ggml_time_ms() - control_convert_start,
                     control_image_tensor.shape()[0],
                     control_image_tensor.shape()[1],
                     control_image_tensor.shape()[2],
                     control_image_tensor.shape()[3]);
        }
    }

    if (init_image_tensor.empty() || mask_image_tensor.empty()) {
        if (sd_version_is_inpaint(sd_ctx->sd->version)) {
            LOG_WARN("inpainting model requires both an init image and a mask image.");
        }
    }

    if (mask_image_tensor.empty()) {
        mask_image_tensor = sd::full<float>({request->width, request->height, 1, 1}, 1.f);
    }

    sd::Tensor<float> latent_mask = sd::ops::interpolate(mask_image_tensor,
                                                         {request->width / request->vae_scale_factor,
                                                          request->height / request->vae_scale_factor,
                                                          1,
                                                          1},
                                                         sd::ops::InterpolateMode::NearestMax);

    sd::Tensor<float> init_latent;
    sd::Tensor<float> control_latent;
    sd::Tensor<float> concat_latent;
    sd::Tensor<float> uncond_concat_latent;
    if (init_image_tensor.empty()) {
        init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height);
    } else {
        init_latent = encode_image_tensor_for_generation(sd_ctx, init_image_tensor, "init_image");
        if (init_latent.empty()) {
            LOG_ERROR("failed to encode init image");
            return std::nullopt;
        }
    }

    if (sd_version_is_marigold_iid(sd_ctx->sd->version)) {
        if (init_image_tensor.empty()) {
            LOG_ERROR("Marigold IID requires an init/control image; text-to-image is not supported");
            return std::nullopt;
        }
        sd::Tensor<float> image_condition_latent = init_latent;
        init_latent                              = sd_ctx->sd->generate_init_latent(request->width, request->height);
        concat_latent                           = std::move(image_condition_latent);
        uncond_concat_latent                    = sd::Tensor<float>::zeros_like(concat_latent);
        LOG_INFO("Marigold IID latents prepared: image_condition_channels=4 target_channels=8");
    }

    if (!control_image_tensor.empty() && !sd_ctx->sd->vae_decode_only && sd_version_is_control(sd_ctx->sd->version)) {
        int64_t control_encode_start = ggml_time_ms();
        control_latent = encode_image_tensor_for_generation(sd_ctx, control_image_tensor, "control_image");
        if (control_latent.empty()) {
            LOG_ERROR("failed to encode control image");
            return std::nullopt;
        }
        if (StableDiffusionGGML::env_flag_enabled("SDCPP_TRACE_CONTROLNET")) {
            LOG_INFO("[ControlNet] control image VAE encode completed in %" PRId64 "ms for built-in control model concat latent",
                     ggml_time_ms() - control_encode_start);
        }
    } else if (!control_image_tensor.empty() && StableDiffusionGGML::env_flag_enabled("SDCPP_TRACE_CONTROLNET")) {
        LOG_INFO("[ControlNet] skipped control image VAE encode for external ControlNet path version=%d", static_cast<int>(sd_ctx->sd->version));
    }

    std::vector<sd::Tensor<float>> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(sd_image_to_tensor(sd_img_gen_params->ref_images[i]));
    }

    if (ref_images.empty() && sd_version_is_unet_edit(sd_ctx->sd->version)) {
        LOG_WARN("This model needs at least one reference image; using an empty reference");
        ref_images.push_back(sd::zeros<float>({request->width, request->height, 3, 1}));
        request->guidance.img_cfg = request->guidance.txt_cfg;
    }

    if (!ref_images.empty()) {
        LOG_INFO("EDIT mode");
    }

    std::vector<sd::Tensor<float>> ref_latents;
    for (size_t i = 0; i < ref_images.size(); i++) {
        sd::Tensor<float> ref_latent;
        if (request->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            int vae_image_size = std::min(1024 * 1024, request->width * request->height);
            double vae_width   = sqrt(vae_image_size * ref_images[i].shape()[0] / ref_images[i].shape()[1]);
            double vae_height  = vae_width * ref_images[i].shape()[1] / ref_images[i].shape()[0];

            int factor = sd_version_is_qwen_image(sd_ctx->sd->version) ? 32 : 16;
            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            auto resized_ref_img = sd::ops::interpolate(ref_images[i],
                                                        {static_cast<int>(vae_width), static_cast<int>(vae_height), 3, 1});

            LOG_DEBUG("resize vae ref image %d from %" PRId64 "x%" PRId64 " to %" PRId64 "x%" PRId64,
                      static_cast<int>(i),
                      ref_images[i].shape()[1],
                      ref_images[i].shape()[0],
                      resized_ref_img.shape()[1],
                      resized_ref_img.shape()[0]);

            ref_latent = encode_image_tensor_for_generation(sd_ctx, resized_ref_img, "reference_image");
        } else {
            ref_latent = encode_image_tensor_for_generation(sd_ctx, ref_images[i], "reference_image");
        }
        if (ref_latent.empty()) {
            LOG_ERROR("failed to encode reference image %d", static_cast<int>(i));
            return std::nullopt;
        }

        ref_latents.push_back(std::move(ref_latent));
    }

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        sd::Tensor<float> masked_init_latent;

        if (sd_ctx->sd->version != VERSION_FLEX_2) {
            if (!init_image_tensor.empty()) {
                auto masked_image  = ((1.0f - mask_image_tensor) * (init_image_tensor - 0.5f)) + 0.5f;
                masked_init_latent = encode_image_tensor_for_generation(sd_ctx, masked_image, "masked_init_image");
                if (masked_init_latent.empty()) {
                    LOG_ERROR("failed to encode masked init image");
                    return std::nullopt;
                }
            } else {
                masked_init_latent = sd::Tensor<float>::zeros_like(init_latent);
            }
        } else {
            masked_init_latent = ((1.0f - latent_mask) * init_latent);
        }

        auto uncond_masked_init_latent = sd::Tensor<float>::zeros_like(masked_init_latent);

        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            auto mask = mask_image_tensor.reshape({request->vae_scale_factor,
                                                   request->width / request->vae_scale_factor,
                                                   request->vae_scale_factor,
                                                   request->height / request->vae_scale_factor});
            mask      = mask.permute({1, 3, 0, 2}).reshape({request->width / request->vae_scale_factor, request->height / request->vae_scale_factor, request->vae_scale_factor * request->vae_scale_factor, 1});

            concat_latent        = sd::ops::concat(masked_init_latent, mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, mask, 2);
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            concat_latent = sd::ops::concat(masked_init_latent, latent_mask, 2);
            if (!control_latent.empty()) {
                concat_latent = sd::ops::concat(concat_latent, control_latent, 2);
            } else {
                concat_latent = sd::ops::concat(concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
            }

            uncond_concat_latent = sd::ops::concat(uncond_masked_init_latent, latent_mask, 2);
            uncond_concat_latent = sd::ops::concat(uncond_concat_latent, sd::Tensor<float>::zeros_like(masked_init_latent), 2);
        } else {  // SD1.x SD2.x SDXL inpaint
            concat_latent        = sd::ops::concat(latent_mask, masked_init_latent, 2);
            uncond_concat_latent = sd::ops::concat(latent_mask, uncond_masked_init_latent, 2);
        }
    }
    if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        concat_latent        = sd::ops::interpolate<float>(ref_latents[0], init_latent.shape());
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }
    if (sd_version_is_control(sd_ctx->sd->version)) {
        if (!control_latent.empty()) {
            concat_latent = control_latent;
        } else {
            concat_latent = sd::Tensor<float>::zeros_like(init_latent);
        }
        uncond_concat_latent = sd::Tensor<float>::zeros_like(concat_latent);
    }

    if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0) {
        int64_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);
    }

    ImageGenerationLatents latents;
    latents.init_latent          = std::move(init_latent);
    latents.concat_latent        = std::move(concat_latent);
    latents.uncond_concat_latent = std::move(uncond_concat_latent);
    latents.control_image        = std::move(control_image_tensor);
    latents.ref_images           = std::move(ref_images);
    latents.ref_latents          = std::move(ref_latents);

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        latent_mask = sd::ops::max_pool_2d(latent_mask,
                                           {3, 3},
                                           {1, 1},
                                           {1, 1});
    }
    latents.denoise_mask = std::move(latent_mask);

    return latents;
}

static std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(sd_ctx_t* sd_ctx,
                                                                            const sd_img_gen_params_t* sd_img_gen_params,
                                                                            GenerationRequest* request,
                                                                            SamplePlan* plan,
                                                                            ImageGenerationLatents* latents) {
    ConditionerParams condition_params;
    condition_params.text            = request->prompt;
    condition_params.clip_skip       = request->clip_skip;
    condition_params.width           = request->width;
    condition_params.height          = request->height;
    condition_params.ref_images      = &latents->ref_images;
    condition_params.adm_in_channels = static_cast<int>(sd_ctx->sd->diffusion_model->get_adm_in_channels());

    auto id_cond                     = sd_ctx->sd->get_pmid_conditon(request->pm_params, condition_params);
    int64_t prepare_start_ms         = ggml_time_ms();
    condition_params.zero_out_masked = false;
    auto cond                        = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                           condition_params);
    if (cond.c_concat.empty()) {
        cond.c_concat = latents->concat_latent;  // TODO: optimize
    }

    SDCondition uncond;
    if (request->use_uncond || request->use_high_noise_uncond) {
        bool zero_out_masked = false;
        if (sd_version_is_sdxl(sd_ctx->sd->version) &&
            request->negative_prompt.empty() &&
            !sd_ctx->sd->is_using_edm_v_parameterization) {
            zero_out_masked = true;
        }
        condition_params.text            = request->negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                               condition_params);
        if (uncond.c_concat.empty()) {
            uncond.c_concat = latents->uncond_concat_latent;  // TODO: optimize
        }
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    ImageGenerationEmbeds embeds;
    if (request->use_img_cond) {
        embeds.img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    embeds.cond    = std::move(cond);
    embeds.uncond  = std::move(uncond);
    embeds.id_cond = std::move(id_cond);

    return embeds;
}

static sd_image_t* decode_image_outputs(sd_ctx_t* sd_ctx,
                                        const GenerationRequest& request,
                                        const std::vector<sd::Tensor<float>>& final_latents) {
    if (final_latents.size() != static_cast<size_t>(request.batch_count)) {
        LOG_ERROR("expected %d latents, got %zu", request.batch_count, final_latents.size());
        return nullptr;
    }
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<sd::Tensor<float>> decoded_images;
    int64_t t0 = ggml_time_ms();

    for (size_t i = 0; i < final_latents.size(); i++) {
        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(final_latents[i]);
        if (image.empty()) {
            LOG_ERROR("decode_first_stage failed for latent %" PRId64, i + 1);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->first_stage_model->free_params_buffer();
            }
            return nullptr;
        }
        decoded_images.push_back(std::move(image));
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t0) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(request.batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    memset(result_images, 0, request.batch_count * sizeof(sd_image_t));

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i] = tensor_to_sd_image(decoded_images[i]);
    }

    return result_images;
}

enum class sd_latent_source_t {
    unknown,
    vae_encode,
    sampler,
    gpu_download,
};

static const char* sd_latent_source_name(sd_latent_source_t source) {
    switch (source) {
        case sd_latent_source_t::vae_encode:
            return "vae_encode";
        case sd_latent_source_t::sampler:
            return "sampler";
        case sd_latent_source_t::gpu_download:
            return "gpu_download";
        case sd_latent_source_t::unknown:
        default:
            return "unknown";
    }
}

struct sd_latent_private_t {
    sd::Tensor<float> tensor;
    sd_latent_source_t source = sd_latent_source_t::unknown;
};

static const sd::Tensor<float>* sd_latent_tensor(const sd_latent_t* latent) {
    if (latent == nullptr || latent->opaque == nullptr) {
        return nullptr;
    }
    const auto* private_latent = static_cast<const sd_latent_private_t*>(latent->opaque);
    if (private_latent->tensor.empty()) {
        return nullptr;
    }
    return &private_latent->tensor;
}

static sd_latent_source_t sd_latent_source(const sd_latent_t* latent) {
    if (latent == nullptr || latent->opaque == nullptr) {
        return sd_latent_source_t::unknown;
    }
    const auto* private_latent = static_cast<const sd_latent_private_t*>(latent->opaque);
    return private_latent->source;
}

static sd_latent_t* make_sd_latent(sd::Tensor<float>&& tensor,
                                   sd_latent_source_t source = sd_latent_source_t::unknown) {
    if (tensor.empty()) {
        return nullptr;
    }

    sd_latent_t* latent = static_cast<sd_latent_t*>(calloc(1, sizeof(sd_latent_t)));
    if (latent == nullptr) {
        return nullptr;
    }

    sd_latent_private_t* private_latent = nullptr;
    try {
        private_latent = new sd_latent_private_t{std::move(tensor), source};
    } catch (...) {
        free(latent);
        return nullptr;
    }

    const auto& shape      = private_latent->tensor.shape();
    latent->width          = shape.size() > 0 ? static_cast<uint32_t>(shape[0]) : 0;
    latent->height         = shape.size() > 1 ? static_cast<uint32_t>(shape[1]) : 0;
    latent->channel        = shape.size() == 5 ? static_cast<uint32_t>(shape[3])
                                               : (shape.size() > 2 ? static_cast<uint32_t>(shape[2]) : 0);
    latent->element_count  = static_cast<uint64_t>(private_latent->tensor.numel());
    latent->opaque         = private_latent;
    return latent;
}

static bool trace_gpu_handles_enabled() {
    return StableDiffusionGGML::env_flag_enabled("SDCPP_TRACE_GPU_HANDLES");
}

static enum sd_backend_kind_t sd_backend_kind_from_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || ggml_backend_buffer_is_host(buffer)) {
        return SD_BACKEND_CPU;
    }
    return SD_BACKEND_CUDA;
}

static enum sd_tensor_dtype_t sd_dtype_from_ggml(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return SD_DTYPE_F32;
        case GGML_TYPE_F16:
            return SD_DTYPE_F16;
        case GGML_TYPE_BF16:
            return SD_DTYPE_BF16;
        default:
            return SD_DTYPE_F32;
    }
}

static std::shared_ptr<sd_gpu_resource_private_t> sd_gpu_resource_lookup(sd_ctx_t* sd_ctx,
                                                                         sd_gpu_handle_t handle) {
    if (sd_ctx == nullptr || handle == 0) {
        return nullptr;
    }
    auto it = sd_ctx->gpu_resources.find(handle);
    if (it == sd_ctx->gpu_resources.end()) {
        return nullptr;
    }
    return it->second;
}

static bool sd_gpu_fill_desc(const sd_gpu_resource_private_t& resource,
                             sd_gpu_tensor_desc_t* desc) {
    if (desc == nullptr || resource.tensor == nullptr || resource.tensor->empty()) {
        return false;
    }
    ggml_tensor* tensor = resource.tensor->tensor;
    *desc = {};
    desc->struct_size = sizeof(sd_gpu_tensor_desc_t);
    desc->version = SD_VAE_API_VERSION;
    desc->handle = resource.handle;
    desc->kind = resource.kind;
    desc->backend = sd_backend_kind_from_buffer(resource.tensor->buffer);
    desc->device_index = resource.device_index;
    desc->dtype = sd_dtype_from_ggml(tensor->type);
    desc->layout = resource.layout;
    desc->w = tensor->ne[0];
    desc->h = tensor->ne[1];
    desc->c = tensor->ne[2];
    desc->n = tensor->ne[3];
    const size_t elem_size = ggml_type_size(tensor->type);
    desc->stride_w = static_cast<int64_t>(tensor->nb[0] / elem_size);
    desc->stride_h = static_cast<int64_t>(tensor->nb[1] / elem_size);
    desc->stride_c = static_cast<int64_t>(tensor->nb[2] / elem_size);
    desc->stride_n = static_cast<int64_t>(tensor->nb[3] / elem_size);
    desc->byte_offset = 0;
    desc->byte_size = static_cast<uint64_t>(ggml_nbytes(tensor));
    desc->producer_stream_id = 0;
    desc->ready_event_id = 0;
    desc->flags = resource.flags;
    desc->refcount = resource.refcount;
    return true;
}

static sd_gpu_handle_t sd_gpu_register_resource(sd_ctx_t* sd_ctx,
                                                std::unique_ptr<GgmlBackendTensorResource> tensor,
                                                enum sd_gpu_resource_kind_t kind,
                                                enum sd_tensor_layout_t layout,
                                                uint32_t flags,
                                                const char* debug_name) {
    if (sd_ctx == nullptr || tensor == nullptr || tensor->empty()) {
        return 0;
    }
    auto resource = std::make_shared<sd_gpu_resource_private_t>();
    resource->handle = sd_ctx->next_gpu_handle++;
    resource->kind = kind;
    resource->layout = layout;
    resource->flags = flags;
    resource->debug_name = debug_name != nullptr ? debug_name : "";
    resource->tensor = std::move(tensor);
    sd_ctx->gpu_resources[resource->handle] = resource;
    if (trace_gpu_handles_enabled()) {
        sd_gpu_tensor_desc_t desc;
        if (sd_gpu_fill_desc(*resource, &desc)) {
            LOG_INFO("[GPU] handle created id=%" PRIu64 " kind=%d backend=%d dtype=%d layout=%d shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] bytes=%" PRIu64 " name=%s",
                     resource->handle,
                     static_cast<int>(resource->kind),
                     static_cast<int>(desc.backend),
                     static_cast<int>(desc.dtype),
                     static_cast<int>(desc.layout),
                     desc.n,
                     desc.c,
                     desc.h,
                     desc.w,
                     desc.byte_size,
                     resource->debug_name.c_str());
        }
    }
    return resource->handle;
}

static std::unique_ptr<GgmlBackendTensorResource> sd_upload_tensor_to_backend_resource(sd_ctx_t* sd_ctx,
                                                                                       const sd::Tensor<float>& tensor,
                                                                                       const char* name) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || tensor.empty() || tensor.dim() > 4) {
        return nullptr;
    }
    auto handle = std::make_unique<GgmlBackendTensorResource>();
    ggml_init_params params;
    params.mem_size = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    handle->ctx = ggml_init(params);
    if (handle->ctx == nullptr) {
        return nullptr;
    }
    const auto& shape = tensor.shape();
    int64_t ne0 = shape.size() > 0 ? shape[0] : 1;
    int64_t ne1 = shape.size() > 1 ? shape[1] : 1;
    int64_t ne2 = shape.size() > 2 ? shape[2] : 1;
    int64_t ne3 = shape.size() > 3 ? shape[3] : 1;
    handle->tensor = ggml_new_tensor_4d(handle->ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    if (name != nullptr) {
        ggml_set_name(handle->tensor, name);
    }
    ggml_backend_t resource_backend = sd_ctx->sd->vae_backend != nullptr ? sd_ctx->sd->vae_backend : sd_ctx->sd->backend;
    handle->buffer = ggml_backend_alloc_ctx_tensors(handle->ctx, resource_backend);
    if (handle->buffer == nullptr) {
        return nullptr;
    }
    ggml_backend_tensor_set(handle->tensor, tensor.data(), 0, ggml_nbytes(handle->tensor));
    ggml_backend_synchronize(resource_backend);
    return handle;
}

static std::unique_ptr<GgmlBackendTensorResource> sd_copy_gpu_resource_to_context(sd_ctx_t* dst_ctx,
                                                                                  const GgmlBackendTensorResource* src,
                                                                                  const char* name) {
    if (dst_ctx == nullptr || dst_ctx->sd == nullptr || src == nullptr || src->empty()) {
        return nullptr;
    }
    auto handle = std::make_unique<GgmlBackendTensorResource>();
    ggml_init_params params;
    params.mem_size = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    handle->ctx = ggml_init(params);
    if (handle->ctx == nullptr) {
        return nullptr;
    }
    handle->tensor = ggml_dup_tensor(handle->ctx, src->tensor);
    if (name != nullptr) {
        ggml_set_name(handle->tensor, name);
    }
    ggml_backend_t dst_backend = dst_ctx->sd->vae_backend != nullptr ? dst_ctx->sd->vae_backend : dst_ctx->sd->backend;
    handle->buffer = ggml_backend_alloc_ctx_tensors(handle->ctx, dst_backend);
    if (handle->buffer == nullptr) {
        return nullptr;
    }
    ggml_backend_tensor_copy(src->tensor, handle->tensor);
    ggml_backend_synchronize(dst_backend);
    return handle;
}

static bool sd_strict_gpu_resident_enabled() {
    return StableDiffusionGGML::env_flag_enabled("SDCPP_STRICT_GPU_RESIDENT");
}

static bool sd_gpu_resource_is_cuda(const sd_gpu_resource_private_t& resource) {
    return resource.tensor != nullptr &&
           !resource.tensor->empty() &&
           resource.tensor->buffer != nullptr &&
           !ggml_backend_buffer_is_host(resource.tensor->buffer);
}

static uint32_t expected_diffusion_latent_channels(SDVersion version) {
    if (sd_version_is_marigold_iid(version)) {
        return 8;
    }
    if (sd_version_is_flux2(version)) {
        return 128;
    }
    if (sd_version_is_flux(version) ||
        sd_version_is_z_image(version) ||
        sd_version_is_sd3(version) ||
        sd_version_is_qwen_image(version) ||
        sd_version_is_anima(version)) {
        return 16;
    }
    return 4;
}

static bool sd_model_supports_gpu_latent_decode(SDVersion version) {
    return sd_version_is_sdxl(version) ||
           sd_version_is_flux(version) ||
           sd_version_is_flux2(version) ||
           sd_version_is_z_image(version) ||
           sd_version_is_anima(version);
}

static bool sd_model_uses_gpu_latent_decode_bridge(SDVersion version) {
    return sd_version_is_anima(version);
}

static bool sd_gpu_latent_shape_is_supported(SDVersion version, const sd_gpu_resource_private_t& resource) {
    if (resource.tensor == nullptr || resource.tensor->empty()) {
        return false;
    }
    ggml_tensor* tensor = resource.tensor->tensor;
    const uint32_t expected_channels = expected_diffusion_latent_channels(version);
    return tensor != nullptr &&
           tensor->type == GGML_TYPE_F32 &&
           tensor->ne[2] == expected_channels &&
           tensor->ne[3] == 1;
}

static void scale_vae_decode_output_to_image_range(sd::Tensor<float>* tensor) {
    if (tensor == nullptr) {
        return;
    }
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        float value = ((*tensor)[i] + 1.0f) * 0.5f;
        (*tensor)[i] = std::max(0.0f, std::min(1.0f, value));
    }
}

static uint64_t default_im2col_warn_bytes() {
    return 512ull * 1024ull * 1024ull;
}

static sd_vae_run_options_t effective_vae_options(const sd_vae_run_options_t* options) {
    sd_vae_run_options_t effective;
    sd_vae_run_options_init(&effective);
    if (options != nullptr) {
        effective = *options;
    }
    if (effective.struct_size == 0) {
        effective.struct_size = sizeof(sd_vae_run_options_t);
    }
    if (effective.version == 0) {
        effective.version = SD_VAE_API_VERSION;
    }
    if (effective.im2col_warn_bytes == 0) {
        effective.im2col_warn_bytes = default_im2col_warn_bytes();
    }
    const char* dtype_env = getenv("SDCPP_VAE_DTYPE");
    if (dtype_env != nullptr && dtype_env[0] != '\0' && effective.storage_dtype == SD_VAE_DTYPE_AUTO) {
        std::string dtype_str(dtype_env);
        if (dtype_str == "bf16") {
            effective.storage_dtype = SD_VAE_DTYPE_BF16;
        } else if (dtype_str == "f16") {
            effective.storage_dtype = SD_VAE_DTYPE_F16;
        } else if (dtype_str == "f32") {
            effective.storage_dtype = SD_VAE_DTYPE_F32;
        } else {
            effective.storage_dtype = SD_VAE_DTYPE_AUTO;
        }
    }
    return effective;
}

static const char* sd_vae_exec_mode_name(enum sd_vae_exec_mode_t mode) {
    switch (mode) {
        case SD_VAE_EXEC_LEGACY_GGML_GRAPH:
            return "legacy_ggml_graph";
        case SD_VAE_EXEC_DIRECT_GRAPH:
            return "direct_graph";
        case SD_VAE_EXEC_COMFY_NORMAL:
            return "comfy_normal";
        case SD_VAE_EXEC_AUTO:
        default:
            return "auto";
    }
}

static const char* sd_vae_dtype_name(enum sd_vae_dtype_t dtype) {
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

static sd_vae_dtype_t resolve_vae_storage_dtype(const sd_vae_run_options_t& options,
                                                sd_vae_exec_mode_t resolved_mode,
                                                std::string* fallback_reason) {
    if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL) {
        if (fallback_reason != nullptr) {
            *fallback_reason = "dtype policy only applies to COMFY_NORMAL";
        }
        return SD_VAE_DTYPE_F32;
    }
    if (fallback_reason != nullptr) {
        *fallback_reason = "f32 fallback: CUDA group_norm/upscale are f32-only; direct conv follows f32 input; pointwise graph stores f32";
    }
    return SD_VAE_DTYPE_F32;
}

static sd_vae_exec_mode_t resolve_vae_exec_mode(StableDiffusionGGML* sd, sd_vae_exec_mode_t requested) {
    if (requested != SD_VAE_EXEC_AUTO) {
        return requested;
    }
    if (StableDiffusionGGML::env_flag_enabled("SDCPP_DISABLE_COMFY_NORMAL_VAE")) {
        return SD_VAE_EXEC_DIRECT_GRAPH;
    }
#ifdef SD_USE_CUDA
    if (sd != nullptr &&
        (sd_version_is_sdxl(sd->version) ||
         sd_version_is_flux(sd->version) ||
         sd_version_is_flux2(sd->version) ||
         sd_version_is_z_image(sd->version))) {
        return SD_VAE_EXEC_COMFY_NORMAL;
    }
#endif
    return SD_VAE_EXEC_DIRECT_GRAPH;
}

static void log_vae_report(const char* operation, const sd_vae_memory_report_t& report) {
    LOG_INFO("[VAE] %s report: requested_mode=%s resolved_mode=%s direct_conv=%s tiled=%s taesd=%s im2col=%s stages=%u graphs=%u host_copies=%u device_copies=%u planned=%.2fMB requested_dtype=%s resolved_dtype=%s fallback=\"%s\"",
             operation,
             sd_vae_exec_mode_name(report.requested_mode),
             sd_vae_exec_mode_name(report.resolved_mode),
             report.used_direct_conv ? "true" : "false",
             report.used_tiling ? "true" : "false",
             report.used_taesd ? "true" : "false",
             report.used_im2col ? "true" : "false",
             report.stage_count,
             report.graph_count,
             report.stage_boundary_host_copies,
             report.stage_boundary_device_copies,
             report.planned_workspace_bytes / 1024.0 / 1024.0,
             sd_vae_dtype_name(report.requested_storage_dtype),
             sd_vae_dtype_name(report.resolved_storage_dtype),
             report.fallback_reason);
}

static void copy_vae_report(sd_vae_memory_report_t* report,
                            const sd_vae_memory_report_t& graph_report,
                            const sd_vae_run_options_t& options,
                            sd_vae_exec_mode_t resolved_mode,
                            bool used_tiling,
                            bool used_taesd) {
    if (report == nullptr) {
        return;
    }
    *report = graph_report;
    report->struct_size = sizeof(sd_vae_memory_report_t);
    report->version = SD_VAE_API_VERSION;
    report->requested_mode = options.mode;
    report->resolved_mode = resolved_mode;
    report->requested_storage_dtype = options.storage_dtype;
    std::string fallback_reason;
    report->resolved_storage_dtype = resolve_vae_storage_dtype(options, resolved_mode, &fallback_reason);
    report->used_tiling = used_tiling;
    report->used_taesd = used_taesd;
    snprintf(report->math_dtype_policy,
             sizeof(report->math_dtype_policy),
             "storage=%s math=f32 reductions=f32",
             sd_vae_dtype_name(report->resolved_storage_dtype));
    snprintf(report->fallback_reason,
             sizeof(report->fallback_reason),
             "%s",
             fallback_reason.c_str());
    if (resolved_mode == SD_VAE_EXEC_COMFY_NORMAL) {
        report->used_direct_conv = true;
    }
}

static bool vae_report_large_im2col_disallowed(const sd_vae_memory_report_t& report,
                                               const sd_vae_run_options_t& options) {
    if (!options.fail_on_large_im2col || !report.used_im2col) {
        return false;
    }
    uint64_t threshold = options.im2col_warn_bytes == 0 ? default_im2col_warn_bytes() : options.im2col_warn_bytes;
    return report.largest_tensor_bytes > threshold;
}

static bool strict_comfy_normal_enabled() {
    return StableDiffusionGGML::env_flag_enabled("SDCPP_VAE_STRICT_COMFY_NORMAL");
}

static bool vae_report_comfy_guard_failed(const sd_vae_memory_report_t& report,
                                          sd_vae_exec_mode_t resolved_mode) {
    if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL) {
        return false;
    }
    bool failed = false;
    const bool strict = strict_comfy_normal_enabled();
    const uint64_t staged_decode_baseline = 3072ull * 1024ull * 1024ull;
    const uint64_t allowed_workspace = staged_decode_baseline + staged_decode_baseline / 10;
    auto guard_log = [&](const char* message) {
        if (strict) {
            LOG_ERROR("%s", message);
            failed = true;
        } else {
            LOG_WARN("%s", message);
        }
    };

    if (report.used_im2col) {
        LOG_ERROR("COMFY_NORMAL VAE created IM2COL; this path is forbidden");
        failed = true;
    }
    if (report.used_tiling) {
        LOG_ERROR("COMFY_NORMAL VAE entered tiled VAE path; this path is forbidden");
        failed = true;
    }
    if (report.used_taesd) {
        LOG_ERROR("COMFY_NORMAL VAE entered TAESD path; this path is forbidden");
        failed = true;
    }
    if (report.stage_boundary_host_copies > 0) {
        guard_log("COMFY_NORMAL VAE had device-host stage boundary copies");
    }
    if (!report.device_resident_stages) {
        guard_log("COMFY_NORMAL VAE stages were not device resident");
    }
    if (report.planned_workspace_bytes > allowed_workspace) {
        guard_log("COMFY_NORMAL VAE planned workspace regressed above staged baseline +10%");
    }
    if (report.requested_storage_dtype != SD_VAE_DTYPE_AUTO &&
        report.requested_storage_dtype != report.resolved_storage_dtype) {
        guard_log("COMFY_NORMAL VAE compact storage dtype request fell back; see fallback_reason in report");
    }
    return failed;
}

static bool prepare_normal_vae_run(sd_ctx_t* sd_ctx,
                                   const sd_vae_run_options_t& options,
                                   sd_vae_exec_mode_t* resolved_mode,
                                   bool* used_taesd) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->first_stage_model == nullptr) {
        LOG_ERROR("normal VAE requires a loaded VAE");
        return false;
    }
    *resolved_mode = resolve_vae_exec_mode(sd_ctx->sd, options.mode);
    *used_taesd = sd_ctx->sd->first_stage_model->get_desc() == "taesd" ||
                  sd_ctx->sd->first_stage_model->get_desc() == "taehv";
    if (*used_taesd && !options.allow_taesd) {
        LOG_ERROR("normal VAE run refuses TAESD/TAEHV; recreate the context with the full checkpoint VAE or set allow_taesd for diagnostics");
        return false;
    }
    sd_ctx->sd->vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
    if (*resolved_mode == SD_VAE_EXEC_LEGACY_GGML_GRAPH) {
        sd_ctx->sd->first_stage_model->set_comfy_normal_enabled(false);
        sd_ctx->sd->first_stage_model->set_conv2d_direct_enabled(false);
        LOG_WARN("normal VAE using legacy ggml graph compatibility mode; SDXL may allocate oversized IM2COL tensors");
    } else if (*resolved_mode == SD_VAE_EXEC_COMFY_NORMAL) {
        sd_ctx->sd->first_stage_model->set_comfy_normal_enabled(true);
        sd_ctx->sd->first_stage_model->set_conv2d_direct_enabled(true);
        std::string fallback_reason;
        sd_vae_dtype_t resolved_dtype = resolve_vae_storage_dtype(options, *resolved_mode, &fallback_reason);
        LOG_INFO("[VAE] COMFY_NORMAL dtype policy: requested=%s resolved=%s math=f32 fallback=\"%s\"",
                 sd_vae_dtype_name(options.storage_dtype),
                 sd_vae_dtype_name(resolved_dtype),
                 fallback_reason.c_str());
    } else {
        sd_ctx->sd->first_stage_model->set_comfy_normal_enabled(false);
        const bool use_direct_conv = !sd_version_is_anima(sd_ctx->sd->version);
        sd_ctx->sd->first_stage_model->set_conv2d_direct_enabled(use_direct_conv);
        if (!use_direct_conv) {
            LOG_INFO("[VAE] direct graph using legacy convolution for Anima Wan/Qwen VAE compatibility");
        }
    }
    return true;
}

class ScopedVaeImplicitGemmConv {
public:
    explicit ScopedVaeImplicitGemmConv(sd_vae_exec_mode_t resolved_mode) {
        if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL ||
            StableDiffusionGGML::env_flag_enabled("SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV")) {
            return;
        }
        const char* current = std::getenv(kEnvName);
        if (current != nullptr) {
            had_previous_ = true;
            previous_ = current;
        }
#ifdef _WIN32
        _putenv_s(kEnvName, "1");
#else
        setenv(kEnvName, "1", 1);
#endif
        active_ = true;
        LOG_INFO("[VAE] COMFY_NORMAL conv backend: implicit_gemm (set SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1 to force direct conv)");
    }

    ~ScopedVaeImplicitGemmConv() {
        if (!active_) {
            return;
        }
#ifdef _WIN32
        _putenv_s(kEnvName, had_previous_ ? previous_.c_str() : "");
#else
        if (had_previous_) {
            setenv(kEnvName, previous_.c_str(), 1);
        } else {
            unsetenv(kEnvName);
        }
#endif
    }

private:
    static constexpr const char* kEnvName = "SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV";
    bool active_ = false;
    bool had_previous_ = false;
    std::string previous_;
};

static bool should_use_normal_vae_for_generation_encode(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->first_stage_model == nullptr) {
        return false;
    }
#ifdef SD_USE_CUDA
    return sd_version_is_sdxl(sd_ctx->sd->version) ||
           sd_version_is_flux(sd_ctx->sd->version) ||
           sd_version_is_flux2(sd_ctx->sd->version) ||
           sd_version_is_z_image(sd_ctx->sd->version);
#else
    return false;
#endif
}

static sd::Tensor<float> encode_image_tensor_normal_internal(sd_ctx_t* sd_ctx,
                                                             const sd::Tensor<float>& image_tensor,
                                                             const sd_vae_run_options_t* options,
                                                             sd_vae_memory_report_t* report,
                                                             const char* label,
                                                             bool free_params_after) {
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image_tensor.empty()) {
        return {};
    }
    if (sd_ctx->sd->vae_decode_only) {
        LOG_ERROR("%s normal VAE encode requires a VAE encode-capable context; recreate with vae_decode_only=false",
                  label == nullptr ? "image" : label);
        return {};
    }
    if (sd_ctx->sd->first_stage_model == nullptr) {
        LOG_ERROR("%s normal VAE encode requires a loaded VAE", label == nullptr ? "image" : label);
        return {};
    }

    sd_vae_run_options_t effective = effective_vae_options(options);
    sd_vae_exec_mode_t resolved_mode = SD_VAE_EXEC_AUTO;
    bool used_taesd = false;
    if (!prepare_normal_vae_run(sd_ctx, effective, &resolved_mode, &used_taesd)) {
        return {};
    }
    ScopedVaeImplicitGemmConv implicit_conv_scope(resolved_mode);

    int64_t t0 = ggml_time_ms();
    auto vae_output = sd_ctx->sd->first_stage_model->encode(sd_ctx->sd->n_threads,
                                                            image_tensor,
                                                            sd_ctx->sd->vae_tiling_params,
                                                            sd_ctx->sd->circular_x,
                                                            sd_ctx->sd->circular_y);
    if (vae_output.empty()) {
        LOG_ERROR("%s normal VAE encode failed during VAE encode", label == nullptr ? "image" : label);
        return {};
    }

    sd_vae_memory_report_t graph_report = sd_ctx->sd->first_stage_model->get_last_graph_report();
    sd_vae_memory_report_t full_report;
    sd_vae_memory_report_init(&full_report);
    copy_vae_report(&full_report, graph_report, effective, resolved_mode, false, used_taesd);
    log_vae_report(label == nullptr ? "encode" : label, full_report);
    if (report != nullptr) {
        *report = full_report;
    }
    if (!sd_version_is_anima(sd_ctx->sd->version) &&
        vae_report_large_im2col_disallowed(graph_report, effective)) {
        LOG_ERROR("%s normal VAE encode refused oversized IM2COL tensor: largest=%" PRIu64 " threshold=%" PRIu64,
                  label == nullptr ? "image" : label,
                  graph_report.largest_tensor_bytes,
                  effective.im2col_warn_bytes);
        return {};
    }
    if (vae_report_comfy_guard_failed(full_report, resolved_mode)) {
        return {};
    }

    sd::Tensor<float> latent = sd_ctx->sd->first_stage_model->vae_output_to_latents(vae_output, sd_ctx->sd->rng);
    if (sd_ctx->sd->version != VERSION_SD1_PIX2PIX) {
        latent = sd_ctx->sd->first_stage_model->vae_to_diffusion_latents(latent);
    }
    if (free_params_after && sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (latent.empty()) {
        LOG_ERROR("%s normal VAE encode failed during latent conversion", label == nullptr ? "image" : label);
        return {};
    }
    int64_t t1 = ggml_time_ms();
    LOG_INFO("%s normal VAE encode completed, taking %.2fs",
             label == nullptr ? "image" : label,
             (t1 - t0) * 1.0f / 1000);
    return latent;
}

static sd::Tensor<float> encode_image_tensor_for_generation(sd_ctx_t* sd_ctx,
                                                            const sd::Tensor<float>& image_tensor,
                                                            const char* label) {
    if (!should_use_normal_vae_for_generation_encode(sd_ctx)) {
        return sd_ctx->sd->encode_first_stage(image_tensor);
    }

    sd_vae_run_options_t options;
    sd_vae_run_options_init(&options);
    options.mode = SD_VAE_EXEC_AUTO;
    options.allow_tiling = false;
    options.allow_taesd = false;
    options.fail_on_large_im2col = true;

    sd::Tensor<float> latent = encode_image_tensor_normal_internal(sd_ctx,
                                                                   image_tensor,
                                                                   &options,
                                                                   nullptr,
                                                                   label,
                                                                   false);
    if (latent.empty()) {
        LOG_ERROR("%s normal VAE encode failed; refusing legacy fallback for COMFY_NORMAL-capable model version %d",
                  label == nullptr ? "image" : label,
                  static_cast<int>(sd_ctx->sd->version));
    }
    return latent;
}

static bool validate_init_latent_shape(sd_ctx_t* sd_ctx,
                                       const GenerationRequest& request,
                                       const sd::Tensor<float>& init_latent) {
    sd::Tensor<float> expected = sd_ctx->sd->generate_init_latent(request.width, request.height);
    if (expected.shape() == init_latent.shape()) {
        return true;
    }
    LOG_ERROR("init latent shape does not match request: expected %s, got %s",
              sd::tensor_shape_to_string(expected.shape()).c_str(),
              sd::tensor_shape_to_string(init_latent.shape()).c_str());
    return false;
}

static void apply_latent_strength_to_plan(const GenerationRequest& request, SamplePlan* plan) {
    if (plan == nullptr || request.strength >= 1.f || plan->sample_steps <= 0 || plan->sigmas.empty()) {
        return;
    }
    size_t t_enc = static_cast<size_t>(plan->sample_steps * request.strength);
    if (t_enc == static_cast<size_t>(plan->sample_steps)) {
        t_enc--;
    }
    LOG_INFO("target t_enc is %zu steps", t_enc);
    std::vector<float> sigma_sched;
    sigma_sched.assign(plan->sigmas.begin() + plan->sample_steps - t_enc - 1, plan->sigmas.end());
    plan->sigmas       = std::move(sigma_sched);
    plan->sample_steps = static_cast<int>(plan->sigmas.size() - 1);
}

SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }
    if (sd_ctx->sd == nullptr ||
        sd_ctx->sd->vae_decode_only ||
        sd_ctx->sd->cond_stage_model == nullptr ||
        sd_ctx->sd->diffusion_model == nullptr ||
        sd_ctx->sd->denoiser == nullptr) {
        LOG_ERROR("generate_image requires a full SD context; recreate with vae_decode_only=false");
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    LOG_INFO("generate_image %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    std::vector<sd::Tensor<float>> final_latents;
    int64_t denoise_start = ggml_time_ms();
    for (int b = 0; b < request.batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = request.seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, request.batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

        sd::Tensor<float> x_0 = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                   true,
                                                   latents.init_latent,
                                                   std::move(noise),
                                                   embeds.cond,
                                                   embeds.uncond,
                                                   embeds.img_cond,
                                                   embeds.id_cond,
                                                   latents.control_image,
                                                   request.control_strength,
                                                   request.guidance,
                                                   plan.eta,
                                                   plan.s_noise,
                                                   plan.dpmpp_sde_r,
                                                   plan.dpmpp_sde_solver,
                                                   request.shifted_timestep,
                                                   plan.sample_method,
                                                   sd_ctx->sd->is_flow_denoiser(),
                                                   plan.sigmas,
                                                   plan.start_merge_step,
                                                   latents.ref_latents,
                                                   request.increase_ref_index,
                                                   latents.denoise_mask,
                                                   sd::Tensor<float>(),
                                                   1.f,
                                                   request.cache_params);
        int64_t sampling_end  = ggml_time_ms();
        if (!x_0.empty()) {
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(std::move(x_0));
            continue;
        }

        LOG_ERROR("sampling for image %d/%d failed after %.2fs",
                  b + 1,
                  request.batch_count,
                  (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->diffusion_model->free_params_buffer();
        }
        return nullptr;
    }
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t denoise_end = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs",
             final_latents.size(),
             (denoise_end - denoise_start) * 1.0f / 1000);

    auto result = decode_image_outputs(sd_ctx, request, final_latents);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_image completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}

SD_API sd_latent_t* sd_encode_image(sd_ctx_t* sd_ctx,
                                    const sd_image_t* image,
                                    const sd_tiling_params_t* vae_tiling_params) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image == nullptr || image->data == nullptr) {
        return nullptr;
    }
    if ((vae_tiling_params == nullptr || !vae_tiling_params->enabled) &&
        !sd_version_is_anima(sd_ctx->sd->version)) {
        return sd_encode_image_normal(sd_ctx, image, nullptr, nullptr);
    }
    if (sd_ctx->sd->vae_decode_only) {
        LOG_ERROR("sd_encode_image requires a VAE encode-capable context; recreate with vae_decode_only=false");
        return nullptr;
    }
    if (sd_ctx->sd->first_stage_model == nullptr) {
        LOG_ERROR("sd_encode_image requires a loaded VAE");
        return nullptr;
    }
    if (vae_tiling_params != nullptr) {
        sd_ctx->sd->vae_tiling_params = *vae_tiling_params;
    }

    sd::Tensor<float> image_tensor = sd_image_to_tensor(*image);
    sd::Tensor<float> latent       = sd_ctx->sd->encode_first_stage(image_tensor);
    if (sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (latent.empty()) {
        LOG_ERROR("sd_encode_image failed");
        return nullptr;
    }
    return make_sd_latent(std::move(latent), sd_latent_source_t::vae_encode);
}

SD_API void sd_vae_run_options_init(sd_vae_run_options_t* options) {
    if (options == nullptr) {
        return;
    }
    *options = {};
    options->struct_size = sizeof(sd_vae_run_options_t);
    options->version = SD_VAE_API_VERSION;
    options->mode = SD_VAE_EXEC_AUTO;
    options->storage_dtype = SD_VAE_DTYPE_AUTO;
    options->fail_on_large_im2col = true;
    options->allow_tiling = false;
    options->allow_taesd = false;
    options->im2col_warn_bytes = default_im2col_warn_bytes();
}

SD_API void sd_vae_memory_report_init(sd_vae_memory_report_t* report) {
    if (report == nullptr) {
        return;
    }
    *report = {};
    report->struct_size = sizeof(sd_vae_memory_report_t);
    report->version = SD_VAE_API_VERSION;
    report->requested_mode = SD_VAE_EXEC_AUTO;
    report->resolved_mode = SD_VAE_EXEC_AUTO;
    report->requested_storage_dtype = SD_VAE_DTYPE_AUTO;
    report->resolved_storage_dtype = SD_VAE_DTYPE_AUTO;
}

SD_API sd_latent_t* sd_encode_image_normal(sd_ctx_t* sd_ctx,
                                           const sd_image_t* image,
                                           const sd_vae_run_options_t* options,
                                           sd_vae_memory_report_t* report) {
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image == nullptr || image->data == nullptr) {
        return nullptr;
    }
    int64_t t0 = ggml_time_ms();
    sd::Tensor<float> image_tensor = sd_image_to_tensor(*image);
    sd::Tensor<float> latent = encode_image_tensor_normal_internal(sd_ctx,
                                                                   image_tensor,
                                                                   options,
                                                                   report,
                                                                   "encode",
                                                                   true);
    if (latent.empty()) {
        LOG_ERROR("sd_encode_image_normal failed");
        return nullptr;
    }
    int64_t t1 = ggml_time_ms();
    LOG_INFO("sd_encode_image_normal completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    return make_sd_latent(std::move(latent), sd_latent_source_t::vae_encode);
}

SD_API sd_latent_t* sd_sample_latent(sd_ctx_t* sd_ctx,
                                     const sd_img_gen_params_t* sd_img_gen_params,
                                     const sd_latent_t* init_latent) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }
    if (sd_ctx->sd->vae_decode_only ||
        sd_ctx->sd->cond_stage_model == nullptr ||
        sd_ctx->sd->diffusion_model == nullptr ||
        sd_ctx->sd->denoiser == nullptr) {
        LOG_ERROR("sd_sample_latent requires a full SD context; recreate with vae_decode_only=false");
        return nullptr;
    }

    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_img_gen_params);
    if (request.batch_count != 1) {
        LOG_ERROR("sd_sample_latent currently supports batch_count=1, got %d", request.batch_count);
        return nullptr;
    }
    LOG_INFO("sd_sample_latent %dx%d", request.width, request.height);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_img_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);

    ImageVaeAxesGuard axes_guard(sd_ctx, sd_img_gen_params, request);

    SamplePlan plan(sd_ctx, sd_img_gen_params, request);
    auto latents_opt = prepare_image_generation_latents(sd_ctx,
                                                        sd_img_gen_params,
                                                        &request,
                                                        &plan);
    if (!latents_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latents_opt);

    const sd::Tensor<float>* init_tensor = sd_latent_tensor(init_latent);
    if (init_latent != nullptr && init_tensor == nullptr) {
        LOG_ERROR("sd_sample_latent received an invalid init latent");
        return nullptr;
    }
    if (init_tensor != nullptr) {
        if (!validate_init_latent_shape(sd_ctx, request, *init_tensor)) {
            return nullptr;
        }
        latents.init_latent = *init_tensor;
        apply_latent_strength_to_plan(request, &plan);
    }

    auto embeds_opt = prepare_image_generation_embeds(sd_ctx,
                                                      sd_img_gen_params,
                                                      &request,
                                                      &plan,
                                                      &latents);
    if (!embeds_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationEmbeds embeds = std::move(*embeds_opt);

    int64_t sampling_start = ggml_time_ms();
    LOG_INFO("generating latent image - seed %" PRId64, request.seed);

    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

    sd::Tensor<float> final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                        true,
                                                        latents.init_latent,
                                                        std::move(noise),
                                                        embeds.cond,
                                                        embeds.uncond,
                                                        embeds.img_cond,
                                                        embeds.id_cond,
                                                        latents.control_image,
                                                        request.control_strength,
                                                        request.guidance,
                                                        plan.eta,
                                                        plan.s_noise,
                                                        plan.dpmpp_sde_r,
                                                        plan.dpmpp_sde_solver,
                                                        request.shifted_timestep,
                                                        plan.sample_method,
                                                        sd_ctx->sd->is_flow_denoiser(),
                                                        plan.sigmas,
                                                        plan.start_merge_step,
                                                        latents.ref_latents,
                                                        request.increase_ref_index,
                                                        latents.denoise_mask,
                                                        sd::Tensor<float>(),
                                                        1.f,
                                                        request.cache_params);
    int64_t sampling_end = ggml_time_ms();
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    if (final_latent.empty()) {
        LOG_ERROR("latent sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return nullptr;
    }
    LOG_INFO("latent sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("sd_sample_latent completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return make_sd_latent(std::move(final_latent), sd_latent_source_t::sampler);
}

SD_API bool sd_sample_latent_gpu(sd_ctx_t* sd_ctx,
                                 const sd_img_gen_params_t* sd_img_gen_params,
                                 const sd_latent_t* init_latent,
                                 sd_gpu_handle_t* out_gpu_latent) {
    if (out_gpu_latent != nullptr) {
        *out_gpu_latent = 0;
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_img_gen_params == nullptr || out_gpu_latent == nullptr) {
        return false;
    }
    if (sd_strict_gpu_resident_enabled()) {
        LOG_ERROR("sd_sample_latent_gpu strict mode refused CPU-backed sampler bridge; sampler math still materializes sd::Tensor<float>");
        return false;
    }

    sd_latent_t* latent = sd_sample_latent(sd_ctx, sd_img_gen_params, init_latent);
    if (latent == nullptr) {
        return false;
    }
    const sd::Tensor<float>* tensor = sd_latent_tensor(latent);
    if (tensor == nullptr) {
        free_sd_latent(latent);
        return false;
    }
    auto resource = sd_upload_tensor_to_backend_resource(sd_ctx, *tensor, "sampler_latent_f32");
    free_sd_latent(latent);
    sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                      std::move(resource),
                                                      SD_GPU_RESOURCE_LATENT,
                                                      SD_LAYOUT_WHCN_GGML,
                                                      SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT | SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD,
                                                      "sampler_latent_f32");
    if (handle == 0) {
        LOG_ERROR("sd_sample_latent_gpu failed to register GPU latent handle");
        return false;
    }
    *out_gpu_latent = handle;
    LOG_INFO("sd_sample_latent_gpu completed handle=%" PRIu64 " bridge_upload=true", handle);
    return true;
}

SD_API sd_image_t* sd_decode_latent(sd_ctx_t* sd_ctx,
                                    const sd_latent_t* latent,
                                    const sd_tiling_params_t* vae_tiling_params) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return nullptr;
    }
    if ((vae_tiling_params == nullptr || !vae_tiling_params->enabled) &&
        !sd_version_is_anima(sd_ctx->sd->version)) {
        return sd_decode_latent_normal(sd_ctx, latent, nullptr, nullptr);
    }
    if (sd_ctx->sd->first_stage_model == nullptr) {
        LOG_ERROR("sd_decode_latent requires a loaded VAE");
        return nullptr;
    }
    const sd::Tensor<float>* tensor = sd_latent_tensor(latent);
    if (tensor == nullptr) {
        LOG_ERROR("sd_decode_latent received an invalid latent");
        return nullptr;
    }
    if (vae_tiling_params != nullptr) {
        sd_ctx->sd->vae_tiling_params = *vae_tiling_params;
    }

    int64_t t0              = ggml_time_ms();
    sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(*tensor);
    int64_t t1              = ggml_time_ms();
    if (sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (image.empty()) {
        LOG_ERROR("sd_decode_latent failed after %.2fs", (t1 - t0) * 1.0f / 1000);
        return nullptr;
    }
    LOG_INFO("sd_decode_latent completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);

    sd_image_t output = tensor_to_sd_image(image);
    sd_image_t* result = static_cast<sd_image_t*>(calloc(1, sizeof(sd_image_t)));
    if (result == nullptr) {
        free(output.data);
        return nullptr;
    }
    *result = output;
    return result;
}

SD_API bool sd_decode_latent_normal_gpu(sd_ctx_t* sd_ctx,
                                        const sd_latent_t* latent,
                                        const sd_vae_run_options_t* options,
                                        sd_gpu_handle_t* out_gpu_image,
                                        sd_vae_memory_report_t* report) {
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (out_gpu_image != nullptr) {
        *out_gpu_image = 0;
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || out_gpu_image == nullptr) {
        return false;
    }
    const sd::Tensor<float>* tensor = sd_latent_tensor(latent);
    if (tensor == nullptr) {
        LOG_ERROR("sd_decode_latent_normal_gpu received an invalid latent");
        return false;
    }
    const sd_latent_source_t source = sd_latent_source(latent);
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] CPU latent decode upload source=%s elements=%" PRIu64,
                 sd_latent_source_name(source),
                 latent->element_count);
    }
    sd_gpu_handle_t gpu_latent = 0;
    if (!sd_cpu_latent_upload(sd_ctx, latent, &gpu_latent, nullptr)) {
        LOG_ERROR("sd_decode_latent_normal_gpu failed to upload CPU latent");
        return false;
    }
    bool ok = sd_decode_gpu_latent_normal_gpu(sd_ctx, gpu_latent, options, out_gpu_image, report);
    sd_gpu_handle_release(sd_ctx, gpu_latent);
    return ok;
}

SD_API sd_image_t* sd_decode_latent_normal(sd_ctx_t* sd_ctx,
                                           const sd_latent_t* latent,
                                           const sd_vae_run_options_t* options,
                                           sd_vae_memory_report_t* report) {
    sd_gpu_handle_t gpu_image = 0;
    if (!sd_decode_latent_normal_gpu(sd_ctx, latent, options, &gpu_image, report)) {
        return nullptr;
    }
    sd_image_t output{};
    if (!sd_gpu_image_download(sd_ctx, gpu_image, &output, nullptr)) {
        sd_gpu_handle_release(sd_ctx, gpu_image);
        return nullptr;
    }
    sd_gpu_handle_release(sd_ctx, gpu_image);
    sd_image_t* result = static_cast<sd_image_t*>(calloc(1, sizeof(sd_image_t)));
    if (result == nullptr) {
        free(output.data);
        return nullptr;
    }
    *result = output;
    return result;
}

SD_API bool sd_estimate_vae_normal_memory(sd_ctx_t* sd_ctx,
                                          uint32_t width,
                                          uint32_t height,
                                          bool decode,
                                          const sd_vae_run_options_t* options,
                                          sd_vae_memory_report_t* report) {
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || width == 0 || height == 0) {
        return false;
    }
    sd_vae_run_options_t effective = effective_vae_options(options);
    sd_vae_exec_mode_t resolved_mode = SD_VAE_EXEC_AUTO;
    bool used_taesd = false;
    if (!prepare_normal_vae_run(sd_ctx, effective, &resolved_mode, &used_taesd)) {
        return false;
    }
    ScopedVaeImplicitGemmConv implicit_conv_scope(resolved_mode);

    sd::Tensor<float> input;
    if (decode) {
        int scale_factor = sd_ctx->sd->first_stage_model->get_scale_factor();
        if (scale_factor <= 0 || width % static_cast<uint32_t>(scale_factor) != 0 || height % static_cast<uint32_t>(scale_factor) != 0) {
            LOG_ERROR("sd_estimate_vae_normal_memory decode dimensions must be divisible by VAE scale factor %d", scale_factor);
            return false;
        }
        const uint32_t latent_channels = expected_diffusion_latent_channels(sd_ctx->sd->version);
        input = sd::zeros<float>({static_cast<int64_t>(width / scale_factor),
                                  static_cast<int64_t>(height / scale_factor),
                                  static_cast<int64_t>(latent_channels),
                                  1});
        input = sd_ctx->sd->first_stage_model->diffusion_to_vae_latents(input);
    } else {
        input = sd::zeros<float>({static_cast<int64_t>(width),
                                  static_cast<int64_t>(height),
                                  3,
                                  1});
    }

    sd_vae_memory_report_t graph_report;
    sd_vae_memory_report_init(&graph_report);
    if (!sd_ctx->sd->first_stage_model->estimate_memory_report(input, decode, &graph_report)) {
        return false;
    }
    sd_vae_memory_report_t full_report;
    sd_vae_memory_report_init(&full_report);
    copy_vae_report(&full_report, graph_report, effective, resolved_mode, false, used_taesd);
    log_vae_report(decode ? "estimate_decode" : "estimate_encode", full_report);
    if (report != nullptr) {
        *report = full_report;
    }
    if (vae_report_large_im2col_disallowed(graph_report, effective)) {
        LOG_ERROR("sd_estimate_vae_normal_memory refused oversized IM2COL tensor: largest=%" PRIu64 " threshold=%" PRIu64,
                  graph_report.largest_tensor_bytes,
                  effective.im2col_warn_bytes);
        return false;
    }
    if (vae_report_comfy_guard_failed(full_report, resolved_mode)) {
        return false;
    }
    return true;
}

SD_API bool sd_get_vae_capabilities(sd_ctx_t* sd_ctx, sd_vae_capabilities_t* capabilities) {
    if (capabilities == nullptr) {
        return false;
    }
    *capabilities = {};
    capabilities->struct_size = sizeof(sd_vae_capabilities_t);
    capabilities->version = SD_VAE_API_VERSION;
    capabilities->supports_comfy_normal = true;
    capabilities->supports_device_resident_stages = true;
    capabilities->supports_bf16_storage = false;
    capabilities->supports_f16_storage = false;
    capabilities->supports_normal_encode = true;
    capabilities->supports_normal_decode = true;
    capabilities->supports_memory_report = true;
    capabilities->supports_no_im2col_guard = true;
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr &&
        !(sd_version_is_sdxl(sd_ctx->sd->version) ||
          sd_version_is_flux(sd_ctx->sd->version) ||
          sd_version_is_flux2(sd_ctx->sd->version) ||
          sd_version_is_z_image(sd_ctx->sd->version))) {
        capabilities->supports_comfy_normal = false;
    }
    return true;
}

SD_API bool sd_get_gpu_capabilities(sd_ctx_t* sd_ctx, sd_gpu_capabilities_t* capabilities) {
    if (capabilities == nullptr) {
        return false;
    }
    *capabilities = {};
    capabilities->struct_size = sizeof(sd_gpu_capabilities_t);
    capabilities->version = SD_VAE_API_VERSION;
    capabilities->supports_gpu_handles = true;
    capabilities->supports_cuda_gpu_handles = sd_ctx == nullptr || sd_ctx->sd == nullptr || !ggml_backend_is_cpu(sd_ctx->sd->backend);
    capabilities->supports_gpu_latent_output = true;
    capabilities->supports_gpu_latent_input = true;
    capabilities->supports_sampler_gpu_latent_output = false;
    capabilities->supports_vae_gpu_latent_input = true;
    const bool true_vae_encode_gpu =
        sd_ctx != nullptr &&
        sd_ctx->sd != nullptr &&
        should_use_normal_vae_for_generation_encode(sd_ctx) &&
        !sd_model_uses_gpu_latent_decode_bridge(sd_ctx->sd->version);
    capabilities->supports_vae_encode_gpu_latent_output = true_vae_encode_gpu;
    capabilities->supports_vae_encode_gpu_latent_bridge_output =
        sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_model_uses_gpu_latent_decode_bridge(sd_ctx->sd->version);
    capabilities->supports_gpu_image_output = true;
    capabilities->supports_gpu_image_to_rgba8 = false;
    capabilities->supports_gpu_download = true;
    capabilities->supports_gpu_latent_download = true;
    capabilities->supports_gpu_latent_upload = true;
    capabilities->supports_dlpack_export = false;
    capabilities->supports_cuda_pointer_borrow = true;
    capabilities->supports_cuda_ipc_export = false;
    capabilities->supports_external_memory_interop = false;
    return true;
}

static sd_model_family_t sd_model_family_from_version(SDVersion version) {
    if (sd_version_is_sd1(version)) {
        return SD_MODEL_FAMILY_SD1;
    }
    if (sd_version_is_sd2(version)) {
        return SD_MODEL_FAMILY_SD2;
    }
    if (sd_version_is_sdxl(version)) {
        return SD_MODEL_FAMILY_SDXL;
    }
    if (sd_version_is_sd3(version)) {
        return SD_MODEL_FAMILY_SD3;
    }
    if (sd_version_is_flux2(version)) {
        return SD_MODEL_FAMILY_FLUX2;
    }
    if (sd_version_is_flux(version)) {
        return SD_MODEL_FAMILY_FLUX;
    }
    if (sd_version_is_z_image(version)) {
        return SD_MODEL_FAMILY_Z_IMAGE;
    }
    if (sd_version_is_wan(version)) {
        return SD_MODEL_FAMILY_WAN;
    }
    if (sd_version_is_qwen_image(version)) {
        return SD_MODEL_FAMILY_QWEN_IMAGE;
    }
    if (sd_version_is_anima(version)) {
        return SD_MODEL_FAMILY_ANIMA;
    }
    if (sd_version_is_marigold_iid(version)) {
        return SD_MODEL_FAMILY_MARIGOLD_IID;
    }
    return SD_MODEL_FAMILY_UNKNOWN;
}

static const char* sd_model_family_name(sd_model_family_t family) {
    switch (family) {
        case SD_MODEL_FAMILY_SD1:
            return "sd1";
        case SD_MODEL_FAMILY_SD2:
            return "sd2";
        case SD_MODEL_FAMILY_SDXL:
            return "sdxl";
        case SD_MODEL_FAMILY_SD3:
            return "sd3";
        case SD_MODEL_FAMILY_FLUX:
            return "flux";
        case SD_MODEL_FAMILY_FLUX2:
            return "flux2";
        case SD_MODEL_FAMILY_Z_IMAGE:
            return "z_image";
        case SD_MODEL_FAMILY_WAN:
            return "wan";
        case SD_MODEL_FAMILY_QWEN_IMAGE:
            return "qwen_image";
        case SD_MODEL_FAMILY_ANIMA:
            return "anima";
        case SD_MODEL_FAMILY_MARIGOLD_IID:
            return "marigold_iid";
        case SD_MODEL_FAMILY_UNKNOWN:
        default:
            return "unknown";
    }
}

static bool sd_model_supports_reference_images(SDVersion version) {
    return sd_version_is_unet_edit(version) ||
           sd_version_is_flux2(version) ||
           sd_version_is_z_image(version) ||
           sd_version_is_qwen_image(version);
}

SD_API bool sd_get_model_pipeline_capabilities(sd_ctx_t* sd_ctx, sd_model_pipeline_capabilities_t* capabilities) {
    if (capabilities == nullptr) {
        return false;
    }
    *capabilities = {};
    capabilities->struct_size = sizeof(sd_model_pipeline_capabilities_t);
    capabilities->version = SD_VAE_API_VERSION;
    capabilities->family = SD_MODEL_FAMILY_UNKNOWN;
    std::snprintf(capabilities->family_name, sizeof(capabilities->family_name), "%s", "unknown");
    capabilities->latent_channels = 4;
    capabilities->vae_scale_factor = 8;
    capabilities->default_sample_method = EULER_A_SAMPLE_METHOD;
    capabilities->default_scheduler = DISCRETE_SCHEDULER;
    capabilities->default_cfg_scale = 7.0f;
    capabilities->default_steps = 20;
    capabilities->supports_text_to_image = true;
    capabilities->supports_image_to_image = true;
    capabilities->supports_gpu_sample_bridge_output = false;
    capabilities->supports_gpu_latent_decode = false;
    capabilities->supports_gpu_image_output = false;
    capabilities->supports_vae_encode = true;
    capabilities->supports_vae_encode_gpu_output = false;
    capabilities->supports_reference_images = false;
    capabilities->supports_edit_mode = false;
    capabilities->supports_edit_reference_conditioning = false;
    capabilities->supports_comfy_reference_vae_encode = false;
    capabilities->strict_gpu_sample_is_true_resident = false;

    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return true;
    }

    SDVersion version = sd_ctx->sd->version;
    capabilities->family = sd_model_family_from_version(version);
    std::snprintf(capabilities->family_name,
                  sizeof(capabilities->family_name),
                  "%s",
                  sd_model_family_name(capabilities->family));
    capabilities->latent_channels = expected_diffusion_latent_channels(version);
    if (sd_ctx->sd->first_stage_model != nullptr) {
        capabilities->vae_scale_factor = static_cast<uint32_t>(sd_ctx->sd->first_stage_model->get_scale_factor());
    }
    capabilities->default_sample_method = sd_get_default_sample_method(sd_ctx);
    capabilities->default_scheduler = sd_get_default_scheduler(sd_ctx, capabilities->default_sample_method);
    capabilities->default_flow_shift = std::isfinite(sd_ctx->sd->default_flow_shift) ? sd_ctx->sd->default_flow_shift : 0.0f;
    capabilities->supports_gpu_sample_bridge_output = true;
    capabilities->supports_gpu_latent_decode = sd_model_supports_gpu_latent_decode(version);
    capabilities->supports_gpu_image_output = capabilities->supports_gpu_latent_decode;
    capabilities->supports_vae_encode_gpu_output =
        capabilities->supports_vae_encode &&
        should_use_normal_vae_for_generation_encode(sd_ctx);
    capabilities->supports_reference_images = sd_model_supports_reference_images(version);
    capabilities->supports_edit_mode = sd_model_supports_reference_images(version);
    capabilities->supports_edit_reference_conditioning = sd_model_supports_reference_images(version);
    capabilities->supports_comfy_reference_vae_encode =
        capabilities->supports_edit_reference_conditioning &&
        should_use_normal_vae_for_generation_encode(sd_ctx);

    if (sd_version_is_dit(version)) {
        capabilities->default_cfg_scale = 1.0f;
        capabilities->default_steps = sd_version_is_z_image(version) ? 9 : 4;
        capabilities->requires_llm = sd_version_is_flux2(version) || sd_version_is_z_image(version);
        capabilities->requires_clip_l = sd_version_is_flux(version) && !sd_version_is_flux2(version);
        capabilities->requires_t5xxl = sd_version_is_flux(version) && !sd_version_is_flux2(version);
    }
    if (sd_version_is_z_image(version)) {
        capabilities->requires_llm = true;
        capabilities->requires_clip_l = false;
        capabilities->requires_t5xxl = false;
    }
    if (sd_version_is_flux2(version)) {
        capabilities->requires_llm = true;
    }
    if (sd_version_is_anima(version)) {
        capabilities->default_sample_method = ER_SDE_SAMPLE_METHOD;
        capabilities->default_scheduler = DISCRETE_SCHEDULER;
        capabilities->default_cfg_scale = 4.5f;
        capabilities->default_steps = 30;
        capabilities->requires_llm = true;
        capabilities->requires_clip_l = false;
        capabilities->requires_t5xxl = false;
        capabilities->supports_vae_encode = true;
        capabilities->supports_vae_encode_gpu_output = false;
    }
    if (sd_version_is_marigold_iid(version)) {
        capabilities->latent_channels = 8;
        capabilities->default_sample_method = DDIM_TRAILING_SAMPLE_METHOD;
        capabilities->default_scheduler = DISCRETE_SCHEDULER;
        capabilities->default_cfg_scale = 1.0f;
        capabilities->default_steps = 4;
        capabilities->supports_text_to_image = false;
        capabilities->supports_image_to_image = true;
        capabilities->supports_gpu_sample_bridge_output = false;
        capabilities->supports_gpu_latent_decode = false;
        capabilities->supports_gpu_image_output = false;
        capabilities->supports_intrinsic_image_decomposition = true;
        capabilities->intrinsic_target_count = 2;
    }
    return true;
}

SD_API void sd_marigold_iid_options_init(sd_marigold_iid_options_t* options) {
    if (options == nullptr) {
        return;
    }
    *options = {};
    options->struct_size = sizeof(sd_marigold_iid_options_t);
    options->version = SD_VAE_API_VERSION;
    options->processing_width = 0;
    options->processing_height = 0;
    options->steps = 4;
    options->seed = -1;
    options->match_input_resolution = true;
}

static void resolve_marigold_processing_size(const sd_image_t* image,
                                             const sd_marigold_iid_options_t* options,
                                             uint32_t* width,
                                             uint32_t* height) {
    uint32_t target_w = options != nullptr ? options->processing_width : 0;
    uint32_t target_h = options != nullptr ? options->processing_height : 0;
    if (target_w == 0 && target_h == 0) {
        const uint32_t max_side = 768;
        if (image->width >= image->height) {
            target_w = max_side;
            target_h = std::max<uint32_t>(8, static_cast<uint32_t>(std::llround(static_cast<double>(image->height) * max_side / image->width)));
        } else {
            target_h = max_side;
            target_w = std::max<uint32_t>(8, static_cast<uint32_t>(std::llround(static_cast<double>(image->width) * max_side / image->height)));
        }
    } else if (target_w == 0) {
        target_w = std::max<uint32_t>(8, static_cast<uint32_t>(std::llround(static_cast<double>(image->width) * target_h / image->height)));
    } else if (target_h == 0) {
        target_h = std::max<uint32_t>(8, static_cast<uint32_t>(std::llround(static_cast<double>(image->height) * target_w / image->width)));
    }
    auto align_up_u32 = [](uint32_t value, uint32_t multiple) -> uint32_t {
        return ((value + multiple - 1) / multiple) * multiple;
    };
    target_w = std::max<uint32_t>(8, align_up_u32(target_w, 8));
    target_h = std::max<uint32_t>(8, align_up_u32(target_h, 8));
    *width = target_w;
    *height = target_h;
}

SD_API sd_marigold_iid_result_t* sd_marigold_iid_predict(sd_ctx_t* sd_ctx,
                                                         const sd_image_t* image,
                                                         const sd_marigold_iid_options_t* options) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image == nullptr || image->data == nullptr) {
        return nullptr;
    }
    if (!sd_version_is_marigold_iid(sd_ctx->sd->version)) {
        LOG_ERROR("sd_marigold_iid_predict requires a Marigold IID context");
        return nullptr;
    }
    if (sd_ctx->sd->vae_decode_only ||
        sd_ctx->sd->cond_stage_model == nullptr ||
        sd_ctx->sd->diffusion_model == nullptr ||
        sd_ctx->sd->denoiser == nullptr ||
        sd_ctx->sd->first_stage_model == nullptr) {
        LOG_ERROR("sd_marigold_iid_predict requires a full Marigold IID context");
        return nullptr;
    }

    sd_marigold_iid_options_t effective;
    sd_marigold_iid_options_init(&effective);
    if (options != nullptr) {
        effective = *options;
        if (effective.struct_size == 0) {
            effective.struct_size = sizeof(sd_marigold_iid_options_t);
        }
        if (effective.version == 0) {
            effective.version = SD_VAE_API_VERSION;
        }
    }

    uint32_t width = 0;
    uint32_t height = 0;
    resolve_marigold_processing_size(image, &effective, &width, &height);

    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt = "";
    gen_params.negative_prompt = "";
    gen_params.init_image = *image;
    gen_params.width = static_cast<int>(width);
    gen_params.height = static_cast<int>(height);
    gen_params.seed = effective.seed;
    gen_params.strength = 1.0f;
    gen_params.batch_count = 1;
    gen_params.sample_params.sample_steps = effective.steps > 0 ? static_cast<int>(effective.steps) : 4;
    gen_params.sample_params.guidance.txt_cfg = 1.0f;
    gen_params.sample_params.guidance.img_cfg = 1.0f;
    gen_params.sample_params.sample_method = DDIM_TRAILING_SAMPLE_METHOD;
    gen_params.sample_params.scheduler = DISCRETE_SCHEDULER;
    gen_params.sample_params.eta = 0.0f;

    int64_t t0 = ggml_time_ms();
    sd_latent_t* latent = sd_sample_latent(sd_ctx, &gen_params, nullptr);
    if (latent == nullptr) {
        LOG_ERROR("sd_marigold_iid_predict failed during latent sampling");
        return nullptr;
    }
    const sd::Tensor<float>* pred_latent = sd_latent_tensor(latent);
    if (pred_latent == nullptr || pred_latent->empty() || pred_latent->shape().size() < 3 || pred_latent->shape()[2] != 8) {
        LOG_ERROR("sd_marigold_iid_predict expected an 8-channel prediction latent");
        free_sd_latent(latent);
        return nullptr;
    }

    std::vector<sd::Tensor<float>> target_latents = sd::ops::chunk(*pred_latent, 2, 2);
    sd_image_t* target_images = static_cast<sd_image_t*>(calloc(target_latents.size(), sizeof(sd_image_t)));
    if (target_images == nullptr) {
        free_sd_latent(latent);
        return nullptr;
    }

    for (size_t i = 0; i < target_latents.size(); ++i) {
        sd::Tensor<float> decoded = sd_ctx->sd->decode_first_stage(target_latents[i]);
        if (decoded.empty()) {
            LOG_ERROR("sd_marigold_iid_predict failed decoding target %zu", i);
            for (size_t j = 0; j < i; ++j) {
                free(target_images[j].data);
            }
            free(target_images);
            free_sd_latent(latent);
            return nullptr;
        }
        if (effective.match_input_resolution &&
            (decoded.shape()[0] != static_cast<int64_t>(image->width) ||
             decoded.shape()[1] != static_cast<int64_t>(image->height))) {
            decoded = sd::ops::interpolate(decoded,
                                           {static_cast<int64_t>(image->width),
                                            static_cast<int64_t>(image->height),
                                            decoded.shape()[2],
                                            decoded.shape().size() > 3 ? decoded.shape()[3] : 1});
        }
        target_images[i] = tensor_to_sd_image(decoded);
    }

    sd_marigold_iid_result_t* result = static_cast<sd_marigold_iid_result_t*>(calloc(1, sizeof(sd_marigold_iid_result_t)));
    if (result == nullptr) {
        for (size_t i = 0; i < target_latents.size(); ++i) {
            free(target_images[i].data);
        }
        free(target_images);
        free_sd_latent(latent);
        return nullptr;
    }
    static const char* kTargetNames[] = {"albedo", "material"};
    result->struct_size = sizeof(sd_marigold_iid_result_t);
    result->version = SD_VAE_API_VERSION;
    result->target_count = static_cast<uint32_t>(target_latents.size());
    result->targets = target_images;
    result->target_names = kTargetNames;
    result->latent = latent;
    LOG_INFO("sd_marigold_iid_predict completed targets=%u processing=%ux%u total=%.2fs",
             result->target_count,
             width,
             height,
             (ggml_time_ms() - t0) * 1.0f / 1000);
    return result;
}

SD_API void free_sd_marigold_iid_result(sd_marigold_iid_result_t* result) {
    if (result == nullptr) {
        return;
    }
    if (result->targets != nullptr) {
        for (uint32_t i = 0; i < result->target_count; ++i) {
            free(result->targets[i].data);
            result->targets[i].data = nullptr;
        }
        free(result->targets);
    }
    if (result->latent != nullptr) {
        free_sd_latent(result->latent);
    }
    free(result);
}

SD_API bool sd_gpu_handle_retain(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle) {
    auto resource = sd_gpu_resource_lookup(sd_ctx, handle);
    if (resource == nullptr) {
        return false;
    }
    resource->refcount += 1;
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] handle retain id=%" PRIu64 " refcount=%u", handle, resource->refcount);
    }
    return true;
}

SD_API bool sd_gpu_handle_release(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle) {
    auto resource = sd_gpu_resource_lookup(sd_ctx, handle);
    if (resource == nullptr || resource->refcount == 0) {
        return false;
    }
    resource->refcount -= 1;
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] handle release id=%" PRIu64 " refcount=%u", handle, resource->refcount);
    }
    if (resource->refcount == 0) {
        sd_ctx->gpu_resources.erase(handle);
    }
    return true;
}

SD_API bool sd_gpu_handle_get_desc(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, sd_gpu_tensor_desc_t* desc) {
    auto resource = sd_gpu_resource_lookup(sd_ctx, handle);
    if (resource == nullptr) {
        return false;
    }
    return sd_gpu_fill_desc(*resource, desc);
}

SD_API bool sd_gpu_handle_debug_name(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, const char* name) {
    auto resource = sd_gpu_resource_lookup(sd_ctx, handle);
    if (resource == nullptr) {
        return false;
    }
    resource->debug_name = name != nullptr ? name : "";
    return true;
}

SD_API bool sd_gpu_handle_borrow_cuda_ptr(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, sd_cuda_borrowed_ptr_t* out) {
    auto resource = sd_gpu_resource_lookup(sd_ctx, handle);
    if (resource == nullptr || out == nullptr || resource->tensor == nullptr || resource->tensor->empty()) {
        return false;
    }
    sd_gpu_tensor_desc_t desc;
    if (!sd_gpu_fill_desc(*resource, &desc) || desc.backend != SD_BACKEND_CUDA) {
        return false;
    }
    *out = {};
    out->struct_size = sizeof(sd_cuda_borrowed_ptr_t);
    out->version = SD_VAE_API_VERSION;
    out->device_ptr = resource->tensor->tensor->data;
    out->byte_size = desc.byte_size;
    out->device_index = desc.device_index;
    out->dtype = desc.dtype;
    out->layout = desc.layout;
    out->shape[0] = desc.n;
    out->shape[1] = desc.c;
    out->shape[2] = desc.h;
    out->shape[3] = desc.w;
    out->strides[0] = desc.stride_n;
    out->strides[1] = desc.stride_c;
    out->strides[2] = desc.stride_h;
    out->strides[3] = desc.stride_w;
    out->ready_event_id = desc.ready_event_id;
    out->producer_stream_id = desc.producer_stream_id;
    return out->device_ptr != nullptr;
}

SD_API bool sd_gpu_handle_end_cuda_borrow(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle) {
    return sd_gpu_resource_lookup(sd_ctx, handle) != nullptr;
}

SD_API bool sd_gpu_tensor_download(sd_ctx_t* sd_ctx,
                                   sd_gpu_handle_t gpu_tensor,
                                   void* dst,
                                   uint64_t dst_bytes,
                                   const sd_download_options_t* options) {
    SD_UNUSED(options);
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_tensor);
    if (resource == nullptr || resource->tensor == nullptr || resource->tensor->empty() || dst == nullptr) {
        return false;
    }
    const size_t bytes = ggml_nbytes(resource->tensor->tensor);
    if (dst_bytes < bytes) {
        LOG_ERROR("sd_gpu_tensor_download destination too small: dst=%" PRIu64 " required=%zu", dst_bytes, bytes);
        return false;
    }
    ggml_backend_tensor_get(resource->tensor->tensor, dst, 0, bytes);
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] tensor download id=%" PRIu64 " bytes=%zu", gpu_tensor, bytes);
    }
    return true;
}

SD_API sd_latent_t* sd_gpu_latent_download(sd_ctx_t* sd_ctx,
                                           sd_gpu_handle_t gpu_latent,
                                           const sd_download_options_t* options) {
    SD_UNUSED(options);
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_latent);
    if (resource == nullptr || resource->kind != SD_GPU_RESOURCE_LATENT ||
        resource->tensor == nullptr || resource->tensor->empty()) {
        return nullptr;
    }
    sd::Tensor<float> latent = sd::make_sd_tensor_from_ggml<float>(resource->tensor->tensor);
    if (latent.empty()) {
        return nullptr;
    }
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] latent download id=%" PRIu64 " elements=%" PRId64, gpu_latent, latent.numel());
    }
    sd_latent_source_t source = sd_latent_source_t::gpu_download;
    if ((resource->flags & SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT) != 0) {
        source = sd_latent_source_t::vae_encode;
    } else if ((resource->flags & SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT) != 0) {
        source = sd_latent_source_t::sampler;
    }
    return make_sd_latent(std::move(latent), source);
}

SD_API bool sd_cpu_latent_upload(sd_ctx_t* sd_ctx,
                                 const sd_latent_t* cpu_latent,
                                 sd_gpu_handle_t* out_gpu_latent,
                                 const sd_download_options_t* options) {
    SD_UNUSED(options);
    if (out_gpu_latent != nullptr) {
        *out_gpu_latent = 0;
    }
    if (sd_ctx == nullptr || out_gpu_latent == nullptr) {
        return false;
    }
    const sd::Tensor<float>* tensor = sd_latent_tensor(cpu_latent);
    if (tensor == nullptr) {
        return false;
    }
    const sd_latent_source_t source = sd_latent_source(cpu_latent);
    auto resource = sd_upload_tensor_to_backend_resource(sd_ctx, *tensor, "uploaded_latent_f32");
    sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                      std::move(resource),
                                                      SD_GPU_RESOURCE_LATENT,
                                                      SD_LAYOUT_WHCN_GGML,
                                                      SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD |
                                                          (source == sd_latent_source_t::vae_encode ? SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT : 0u) |
                                                          (source == sd_latent_source_t::sampler ? SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT : 0u),
                                                      "uploaded_latent_f32");
    if (handle == 0) {
        return false;
    }
    *out_gpu_latent = handle;
    return true;
}

SD_API bool sd_gpu_image_download(sd_ctx_t* sd_ctx,
                                  sd_gpu_handle_t gpu_image,
                                  sd_image_t* out_cpu_image,
                                  const sd_download_options_t* options) {
    SD_UNUSED(options);
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_image);
    if (resource == nullptr || out_cpu_image == nullptr || resource->kind != SD_GPU_RESOURCE_IMAGE ||
        resource->tensor == nullptr || resource->tensor->empty()) {
        return false;
    }
    sd::Tensor<float> image = sd::make_sd_tensor_from_ggml<float>(resource->tensor->tensor);
    if (image.empty()) {
        return false;
    }
    if (image.dim() == 3) {
        image.reshape_({image.shape()[0], image.shape()[1], image.shape()[2], 1});
    }
    if ((resource->flags & SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE) != 0) {
        scale_vae_decode_output_to_image_range(&image);
    }
    sd_image_t output = tensor_to_sd_image(image);
    *out_cpu_image = output;
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] image download id=%" PRIu64 " %ux%ux%u", gpu_image, output.width, output.height, output.channel);
    }
    return true;
}

SD_API bool sd_gpu_image_download_to_buffer(sd_ctx_t* sd_ctx,
                                            sd_gpu_handle_t gpu_image,
                                            void* dst_rgba8,
                                            uint64_t dst_bytes,
                                            uint64_t dst_stride_bytes,
                                            const sd_download_options_t* options) {
    SD_UNUSED(options);
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_image);
    if (resource == nullptr || dst_rgba8 == nullptr || resource->kind != SD_GPU_RESOURCE_IMAGE ||
        resource->tensor == nullptr || resource->tensor->empty()) {
        return false;
    }

    sd_gpu_tensor_desc_t desc;
    if (!sd_gpu_fill_desc(*resource, &desc)) {
        return false;
    }
    if (desc.layout != SD_LAYOUT_WHCN_GGML || desc.dtype != SD_DTYPE_F32 || desc.n != 1 ||
        (desc.c != 3 && desc.c != 4) || desc.w <= 0 || desc.h <= 0) {
        LOG_ERROR("sd_gpu_image_download_to_buffer expected f32 WHCN RGB/RGBA image, got layout=%d dtype=%d shape=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
                  static_cast<int>(desc.layout),
                  static_cast<int>(desc.dtype),
                  desc.n,
                  desc.c,
                  desc.h,
                  desc.w);
        return false;
    }

    const uint64_t width = static_cast<uint64_t>(desc.w);
    const uint64_t height = static_cast<uint64_t>(desc.h);
    const uint64_t channels = static_cast<uint64_t>(desc.c);
    if (width > std::numeric_limits<uint64_t>::max() / 4u) {
        LOG_ERROR("sd_gpu_image_download_to_buffer row size overflow");
        return false;
    }
    const uint64_t row_bytes = width * 4u;
    if (dst_stride_bytes < row_bytes) {
        LOG_ERROR("sd_gpu_image_download_to_buffer stride too small: stride=%" PRIu64 " required=%" PRIu64,
                  dst_stride_bytes,
                  row_bytes);
        return false;
    }
    uint64_t required_bytes = row_bytes;
    if (height > 1) {
        if ((height - 1) > (std::numeric_limits<uint64_t>::max() - row_bytes) / dst_stride_bytes) {
            LOG_ERROR("sd_gpu_image_download_to_buffer destination size overflow");
            return false;
        }
        required_bytes = (height - 1) * dst_stride_bytes + row_bytes;
    }
    if (dst_bytes < required_bytes) {
        LOG_ERROR("sd_gpu_image_download_to_buffer destination too small: dst=%" PRIu64 " required=%" PRIu64,
                  dst_bytes,
                  required_bytes);
        return false;
    }

    sd::Tensor<float> image = sd::make_sd_tensor_from_ggml<float>(resource->tensor->tensor);
    if (image.empty()) {
        return false;
    }
    if (image.dim() == 3) {
        image.reshape_({image.shape()[0], image.shape()[1], image.shape()[2], 1});
    }
    if (image.dim() != 4 || image.shape()[0] != desc.w || image.shape()[1] != desc.h ||
        image.shape()[2] != desc.c || image.shape()[3] != 1) {
        LOG_ERROR("sd_gpu_image_download_to_buffer downloaded tensor shape mismatch");
        return false;
    }
    if ((resource->flags & SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE) != 0) {
        scale_vae_decode_output_to_image_range(&image);
    }

    const float* src = image.data();
    uint8_t* dst = static_cast<uint8_t*>(dst_rgba8);
    const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
    const auto to_u8 = [](float value) -> uint8_t {
        value = std::clamp(value, 0.0f, 1.0f);
        return static_cast<uint8_t>(value * 255.0f + 0.5f);
    };
    for (uint64_t y = 0; y < height; ++y) {
        uint8_t* row = dst + y * dst_stride_bytes;
        const size_t row_base = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (uint64_t x = 0; x < width; ++x) {
            const size_t pixel_index = row_base + static_cast<size_t>(x);
            uint8_t* out = row + x * 4u;
            out[0] = to_u8(src[pixel_index + 0u * plane]);
            out[1] = to_u8(src[pixel_index + 1u * plane]);
            out[2] = to_u8(src[pixel_index + 2u * plane]);
            out[3] = channels == 4 ? to_u8(src[pixel_index + 3u * plane]) : 255;
        }
    }
    if (trace_gpu_handles_enabled()) {
        LOG_INFO("[GPU] image download to caller buffer id=%" PRIu64 " %" PRIu64 "x%" PRIu64 " stride=%" PRIu64 " bytes=%" PRIu64,
                 gpu_image,
                 width,
                 height,
                 dst_stride_bytes,
                 required_bytes);
    }
    return true;
}

static bool sd_decode_vae_encoded_gpu_latent_bridge(sd_ctx_t* sd_ctx,
                                                    sd_gpu_handle_t gpu_latent,
                                                    const std::shared_ptr<sd_gpu_resource_private_t>& resource,
                                                    const sd_vae_run_options_t* options,
                                                    sd_gpu_handle_t* out_gpu_image,
                                                    sd_vae_memory_report_t* report) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || resource == nullptr || out_gpu_image == nullptr) {
        return false;
    }
    const bool input_cpu_bridge = (resource->flags & SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD) != 0;
    if (sd_strict_gpu_resident_enabled() && input_cpu_bridge) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu strict mode refused CPU-bridge VAE-encoded latent; the input handle was already materialized through CPU memory");
        return false;
    }
    if (sd_version_is_anima(sd_ctx->sd->version)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu internal bridge mismatch: VAE-encoded Anima latent must route through the Wan/Qwen VAE bridge");
        return false;
    }

    sd_gpu_tensor_desc_t input_desc{};
    if (trace_gpu_handles_enabled() && sd_gpu_fill_desc(*resource, &input_desc)) {
        LOG_INFO("[GPU] VAE encoded-latent bridge input handle=%" PRIu64 " backend=%d dtype=%d layout=%d shape_nchw=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " bytes=%" PRIu64 " flags=%u refcount=%u",
                 gpu_latent,
                 static_cast<int>(input_desc.backend),
                 static_cast<int>(input_desc.dtype),
                 static_cast<int>(input_desc.layout),
                 input_desc.n,
                 input_desc.c,
                 input_desc.h,
                 input_desc.w,
                 input_desc.byte_size,
                 input_desc.flags,
                 input_desc.refcount);
    }

    int64_t t0 = ggml_time_ms();
    sd_ctx_t* decode_ctx = sd_ctx_get_vae_decode_bridge_ctx(sd_ctx);
    if (decode_ctx == nullptr || decode_ctx->sd == nullptr) {
        return false;
    }

    auto latent_copy = sd_copy_gpu_resource_to_context(decode_ctx,
                                                       resource->tensor.get(),
                                                       "vae_encode_latent_d2d_for_isolated_decode");
    int64_t t1 = ggml_time_ms();
    if (latent_copy == nullptr || latent_copy->empty()) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu failed to copy VAE-encoded latent to isolated decode context");
        return false;
    }

    sd_vae_run_options_t effective = effective_vae_options(options);
    sd_vae_exec_mode_t resolved_mode = SD_VAE_EXEC_AUTO;
    bool used_taesd = false;
    if (!prepare_normal_vae_run(decode_ctx, effective, &resolved_mode, &used_taesd)) {
        return false;
    }
    if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL) {
        LOG_ERROR("VAE encoded-latent bridge currently requires COMFY_NORMAL");
        return false;
    }
    ScopedVaeImplicitGemmConv implicit_conv_scope(resolved_mode);

    auto decoded_in_bridge_ctx = decode_ctx->sd->first_stage_model->decode_latent_resource_to_backend_resource(
        decode_ctx->sd->n_threads,
        latent_copy.get(),
        decode_ctx->sd->vae_tiling_params,
        decode_ctx->sd->circular_x,
        decode_ctx->sd->circular_y);
    int64_t t2 = ggml_time_ms();
    if (decoded_in_bridge_ctx == nullptr || decoded_in_bridge_ctx->empty()) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu VAE encoded-latent isolated GPU decode failed after %.2fs",
                  (t2 - t0) * 1.0f / 1000);
        return false;
    }

    sd_vae_memory_report_t graph_report = decode_ctx->sd->first_stage_model->get_last_graph_report();
    sd_vae_memory_report_t full_report;
    sd_vae_memory_report_init(&full_report);
    copy_vae_report(&full_report, graph_report, effective, resolved_mode, false, used_taesd);
    if (vae_report_large_im2col_disallowed(graph_report, effective)) {
        LOG_ERROR("VAE encoded-latent bridge refused oversized IM2COL tensor: largest=%" PRIu64 " threshold=%" PRIu64,
                  graph_report.largest_tensor_bytes,
                  effective.im2col_warn_bytes);
        return false;
    }
    if (vae_report_comfy_guard_failed(full_report, resolved_mode)) {
        return false;
    }

    auto gpu_image = sd_copy_gpu_resource_to_context(sd_ctx,
                                                     decoded_in_bridge_ctx.get(),
                                                     "vae_decode_rgb_f32_from_vae_encode_d2d");
    int64_t t3 = ggml_time_ms();
    if (gpu_image == nullptr || gpu_image->empty()) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu VAE encoded-latent isolated decode failed to copy decoded image to caller context");
        return false;
    }

    full_report.stage_boundary_device_copies += 2;
    full_report.device_resident_stages = full_report.stage_boundary_host_copies == 0 &&
                                         gpu_image->buffer != nullptr &&
                                         !ggml_backend_buffer_is_host(gpu_image->buffer);
    if (sd_strict_gpu_resident_enabled() &&
        (!full_report.device_resident_stages || full_report.stage_boundary_host_copies != 0)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu strict GPU resident check failed for isolated VAE-encoded latent decode: device_resident=%s host_copies=%u",
                  full_report.device_resident_stages ? "true" : "false",
                  full_report.stage_boundary_host_copies);
        return false;
    }
    std::snprintf(full_report.fallback_reason,
                  sizeof(full_report.fallback_reason),
                  "%s",
                  input_cpu_bridge
                      ? "Input VAE Encode handle was CPU-bridge-uploaded before this decode; decode used isolated VAE context with device-to-device handoff"
                      : "VAE Encode GPU latent decoded in isolated VAE context with device-to-device handoff; no CPU latent materialization");
    log_vae_report("decode_gpu_latent_vae_encode_d2d", full_report);
    if (report != nullptr) {
        *report = full_report;
    }

    sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                      std::move(gpu_image),
                                                      SD_GPU_RESOURCE_IMAGE,
                                                      SD_LAYOUT_WHCN_GGML,
                                                      SD_GPU_RESOURCE_FLAG_VAE_DECODE_OUTPUT |
                                                          SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE,
                                                      "vae_decode_rgb_f32_from_vae_encode_d2d");
    if (handle == 0) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu VAE encoded-latent isolated decode failed to register GPU image handle");
        return false;
    }
    *out_gpu_image = handle;
    LOG_INFO("sd_decode_gpu_latent_normal_gpu VAE encoded-latent D2D isolated decode completed, total=%.2fs latent_d2d=%.2fs decode=%.2fs image_d2d=%.2fs input_handle=%" PRIu64 " output_handle=%" PRIu64 " input_cpu_bridge=%s",
             (t3 - t0) * 1.0f / 1000,
             (t1 - t0) * 1.0f / 1000,
             (t2 - t1) * 1.0f / 1000,
             (t3 - t2) * 1.0f / 1000,
             gpu_latent,
             handle,
             input_cpu_bridge ? "true" : "false");
    return true;
}

SD_API bool sd_encode_image_normal_gpu(sd_ctx_t* sd_ctx,
                                       const sd_image_t* image,
                                       const sd_vae_run_options_t* options,
                                       sd_gpu_handle_t* out_gpu_latent,
                                       sd_vae_memory_report_t* report) {
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (out_gpu_latent != nullptr) {
        *out_gpu_latent = 0;
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || image == nullptr || image->data == nullptr || out_gpu_latent == nullptr) {
        return false;
    }

    const bool true_gpu_encode_supported =
        should_use_normal_vae_for_generation_encode(sd_ctx) &&
        !sd_model_uses_gpu_latent_decode_bridge(sd_ctx->sd->version);
    if (true_gpu_encode_supported) {
        if (sd_ctx->sd->vae_decode_only) {
            LOG_ERROR("sd_encode_image_normal_gpu requires a VAE encode-capable context; recreate with vae_decode_only=false");
            return false;
        }
        if (sd_ctx->sd->first_stage_model == nullptr) {
            LOG_ERROR("sd_encode_image_normal_gpu requires a loaded VAE");
            return false;
        }

        sd_vae_run_options_t effective = effective_vae_options(options);
        sd_vae_exec_mode_t resolved_mode = SD_VAE_EXEC_AUTO;
        bool used_taesd = false;
        if (!prepare_normal_vae_run(sd_ctx, effective, &resolved_mode, &used_taesd)) {
            return false;
        }
        if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL) {
            LOG_ERROR("sd_encode_image_normal_gpu true GPU latent output requires COMFY_NORMAL; resolved mode was %s",
                      sd_vae_exec_mode_name(resolved_mode));
            return false;
        }
        ScopedVaeImplicitGemmConv implicit_conv_scope(resolved_mode);

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> image_tensor = sd_image_to_tensor(*image);
        auto resource = sd_ctx->sd->first_stage_model->encode_to_backend_resource(sd_ctx->sd->n_threads,
                                                                                  image_tensor,
                                                                                  sd_ctx->sd->vae_tiling_params,
                                                                                  sd_ctx->sd->rng,
                                                                                  sd_ctx->sd->circular_x,
                                                                                  sd_ctx->sd->circular_y);
        int64_t t1 = ggml_time_ms();
        sd_vae_memory_report_t graph_report = sd_ctx->sd->first_stage_model->get_last_graph_report();
        sd_vae_memory_report_t full_report;
        sd_vae_memory_report_init(&full_report);
        copy_vae_report(&full_report, graph_report, effective, resolved_mode, false, used_taesd);
        log_vae_report("encode_gpu_latent", full_report);
        if (report != nullptr) {
            *report = full_report;
        }
        if (sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
            sd_ctx->sd->first_stage_model->free_params_buffer();
        }
        if (vae_report_large_im2col_disallowed(graph_report, effective)) {
            LOG_ERROR("sd_encode_image_normal_gpu refused oversized IM2COL tensor: largest=%" PRIu64 " threshold=%" PRIu64,
                      graph_report.largest_tensor_bytes,
                      effective.im2col_warn_bytes);
            return false;
        }
        if (vae_report_comfy_guard_failed(full_report, resolved_mode)) {
            return false;
        }
        if (sd_strict_gpu_resident_enabled() &&
            (!full_report.device_resident_stages || full_report.stage_boundary_host_copies != 0)) {
            LOG_ERROR("sd_encode_image_normal_gpu strict GPU resident check failed: device_resident=%s host_copies=%u",
                      full_report.device_resident_stages ? "true" : "false",
                      full_report.stage_boundary_host_copies);
            return false;
        }
        if (resource == nullptr || resource->empty()) {
            LOG_ERROR("sd_encode_image_normal_gpu failed after %.2fs", (t1 - t0) * 1.0f / 1000);
            return false;
        }

        sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                          std::move(resource),
                                                          SD_GPU_RESOURCE_LATENT,
                                                          SD_LAYOUT_WHCN_GGML,
                                                          SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT,
                                                          "vae_encode_diffusion_latent_gpu");
        if (handle == 0) {
            LOG_ERROR("sd_encode_image_normal_gpu failed to register GPU latent handle");
            return false;
        }
        *out_gpu_latent = handle;
        LOG_INFO("sd_encode_image_normal_gpu completed true GPU latent handle=%" PRIu64 " taking %.2fs",
                 handle,
                 (t1 - t0) * 1.0f / 1000);
        return true;
    }

    if (sd_strict_gpu_resident_enabled()) {
        LOG_ERROR("sd_encode_image_normal_gpu strict mode refused compatibility bridge output for model version %d; this path would materialize the latent on CPU before uploading a GPU handle",
                  static_cast<int>(sd_ctx->sd->version));
        return false;
    }
    LOG_WARN("sd_encode_image_normal_gpu using compatibility bridge for model version %d; handle will be flagged CPU_BRIDGE_UPLOAD and is not true all-GPU",
             static_cast<int>(sd_ctx->sd->version));

    sd_vae_memory_report_t encode_report;
    sd_latent_t* encoded = sd_encode_image_normal(sd_ctx, image, options, &encode_report);
    if (encoded == nullptr) {
        LOG_ERROR("sd_encode_image_normal_gpu failed during normal VAE encode");
        return false;
    }
    const sd::Tensor<float>* tensor = sd_latent_tensor(encoded);
    if (tensor == nullptr) {
        free_sd_latent(encoded);
        return false;
    }

    auto resource = sd_upload_tensor_to_backend_resource(sd_ctx, *tensor, "vae_encode_latent_f32_bridge");
    free_sd_latent(encoded);
    if (resource == nullptr || resource->empty()) {
        LOG_ERROR("sd_encode_image_normal_gpu failed to upload encoded latent");
        return false;
    }

    sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                      std::move(resource),
                                                      SD_GPU_RESOURCE_LATENT,
                                                      SD_LAYOUT_WHCN_GGML,
                                                      SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD |
                                                          SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT,
                                                      "vae_encode_latent_f32_bridge");
    if (handle == 0) {
        return false;
    }
    encode_report.stage_boundary_host_copies += 1;
    encode_report.stage_boundary_device_copies += 1;
    encode_report.device_resident_stages = false;
    std::snprintf(encode_report.fallback_reason,
                  sizeof(encode_report.fallback_reason),
                  "%s",
                  "VAE Encode GPU output is bridge-uploaded after CPU latent conversion");
    log_vae_report("encode_gpu_latent_bridge", encode_report);
    if (report != nullptr) {
        *report = encode_report;
    }
    *out_gpu_latent = handle;
    LOG_INFO("sd_encode_image_normal_gpu completed bridge-uploaded latent handle=%" PRIu64, handle);
    return true;
}

SD_API bool sd_decode_gpu_latent_normal_gpu(sd_ctx_t* sd_ctx,
                                            sd_gpu_handle_t gpu_latent,
                                            const sd_vae_run_options_t* options,
                                            sd_gpu_handle_t* out_gpu_image,
                                            sd_vae_memory_report_t* report) {
    if (out_gpu_image != nullptr) {
        *out_gpu_image = 0;
    }
    if (report != nullptr) {
        sd_vae_memory_report_init(report);
    }
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || out_gpu_image == nullptr) {
        return false;
    }
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_latent);
    if (resource == nullptr || resource->kind != SD_GPU_RESOURCE_LATENT || resource->tensor == nullptr || resource->tensor->empty()) {
        return false;
    }
    if (!sd_gpu_resource_is_cuda(*resource)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu requires a CUDA latent handle");
        return false;
    }
    if (!sd_model_supports_gpu_latent_decode(sd_ctx->sd->version)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu unsupported for model version %d; capability supports_gpu_latent_decode=false",
                  static_cast<int>(sd_ctx->sd->version));
        return false;
    }
    if (!sd_gpu_latent_shape_is_supported(sd_ctx->sd->version, *resource)) {
        sd_gpu_tensor_desc_t desc;
        if (sd_gpu_fill_desc(*resource, &desc)) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu unsupported latent shape/type for model version %d: dtype=%d shape_nchw=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " expected_channels=%u",
                      static_cast<int>(sd_ctx->sd->version),
                      static_cast<int>(desc.dtype),
                      desc.n,
                      desc.c,
                      desc.h,
                      desc.w,
                      expected_diffusion_latent_channels(sd_ctx->sd->version));
        } else {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu unsupported latent shape/type");
        }
        return false;
    }
    if ((resource->flags & SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT) != 0 &&
        !sd_model_uses_gpu_latent_decode_bridge(sd_ctx->sd->version)) {
        return sd_decode_vae_encoded_gpu_latent_bridge(sd_ctx,
                                                       gpu_latent,
                                                       resource,
                                                       options,
                                                       out_gpu_image,
                                                       report);
    }

    if (sd_model_uses_gpu_latent_decode_bridge(sd_ctx->sd->version)) {
        if (sd_strict_gpu_resident_enabled()) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu strict mode refused Anima bridge decode; this VAE path downloads the latent for legacy decode and re-uploads the image handle");
            return false;
        }

        LOG_INFO("sd_decode_gpu_latent_normal_gpu using Anima bridge for handle=%" PRIu64, gpu_latent);
        sd_vae_run_options_t effective = effective_vae_options(options);
        int64_t t0 = ggml_time_ms();
        if (trace_gpu_handles_enabled()) {
            LOG_INFO("sd_decode_gpu_latent_normal_gpu Anima bridge downloading latent");
        }
        sd::Tensor<float> latent = sd::make_sd_tensor_from_ggml<float>(resource->tensor->tensor);
        int64_t t1 = ggml_time_ms();
        if (latent.empty()) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu Anima bridge failed to download latent handle=%" PRIu64,
                      gpu_latent);
            return false;
        }
        if (latent.dim() == 3) {
            latent.reshape_({latent.shape()[0], latent.shape()[1], latent.shape()[2], 1});
        }
        if (trace_gpu_handles_enabled()) {
            LOG_INFO("sd_decode_gpu_latent_normal_gpu Anima bridge downloaded latent shape=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
                     latent.shape().size() > 0 ? latent.shape()[0] : 0,
                     latent.shape().size() > 1 ? latent.shape()[1] : 0,
                     latent.shape().size() > 2 ? latent.shape()[2] : 0,
                     latent.shape().size() > 3 ? latent.shape()[3] : 0);
        }

        sd_ctx->sd->vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
        sd_ctx->sd->first_stage_model->set_conv2d_direct_enabled(false);
        sd::Tensor<float> image = sd_ctx->sd->decode_first_stage(latent);
        int64_t t2 = ggml_time_ms();
        if (sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
            sd_ctx->sd->first_stage_model->free_params_buffer();
        }
        if (image.empty()) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu Anima bridge decode failed after %.2fs",
                      (t2 - t0) * 1.0f / 1000);
            return false;
        }

        if (trace_gpu_handles_enabled()) {
            LOG_INFO("sd_decode_gpu_latent_normal_gpu Anima bridge uploading decoded image shape=%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
                     image.shape().size() > 0 ? image.shape()[0] : 0,
                     image.shape().size() > 1 ? image.shape()[1] : 0,
                     image.shape().size() > 2 ? image.shape()[2] : 0,
                     image.shape().size() > 3 ? image.shape()[3] : 0);
        }
        auto gpu_image = sd_upload_tensor_to_backend_resource(sd_ctx, image, "vae_decode_rgb_f32_anima_bridge");
        int64_t t3 = ggml_time_ms();
        if (gpu_image == nullptr || gpu_image->empty()) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu Anima bridge failed to upload decoded image");
            return false;
        }

        sd_vae_memory_report_t graph_report = sd_ctx->sd->first_stage_model->get_last_graph_report();
        sd_vae_memory_report_t full_report;
        sd_vae_memory_report_init(&full_report);
        copy_vae_report(&full_report, graph_report, effective, SD_VAE_EXEC_DIRECT_GRAPH, false, false);
        full_report.stage_boundary_host_copies += 1;
        full_report.stage_boundary_device_copies += 1;
        full_report.device_resident_stages = false;
        std::snprintf(full_report.fallback_reason,
                      sizeof(full_report.fallback_reason),
                      "%s",
                      "Anima uses the Wan/Qwen VAE bridge: GPU latent is downloaded for legacy decode and decoded image is re-uploaded");
        log_vae_report("decode_gpu_latent_anima_bridge", full_report);
        if (report != nullptr) {
            *report = full_report;
        }

        sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                          std::move(gpu_image),
                                                          SD_GPU_RESOURCE_IMAGE,
                                                          SD_LAYOUT_WHCN_GGML,
                                                          SD_GPU_RESOURCE_FLAG_VAE_DECODE_OUTPUT |
                                                              SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_DOWNLOAD |
                                                              SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD,
                                                          "vae_decode_rgb_f32_anima_bridge");
        if (handle == 0) {
            LOG_ERROR("sd_decode_gpu_latent_normal_gpu Anima bridge failed to register GPU image handle");
            return false;
        }
        *out_gpu_image = handle;
        LOG_INFO("sd_decode_gpu_latent_normal_gpu Anima bridge completed, total=%.2fs download=%.2fs decode=%.2fs upload=%.2fs input_handle=%" PRIu64 " output_handle=%" PRIu64,
                 (t3 - t0) * 1.0f / 1000,
                 (t1 - t0) * 1.0f / 1000,
                 (t2 - t1) * 1.0f / 1000,
                 (t3 - t2) * 1.0f / 1000,
                 gpu_latent,
                 handle);
        return true;
    }

    sd_vae_run_options_t effective = effective_vae_options(options);
    sd_vae_exec_mode_t resolved_mode = SD_VAE_EXEC_AUTO;
    bool used_taesd = false;
    if (!prepare_normal_vae_run(sd_ctx, effective, &resolved_mode, &used_taesd)) {
        return false;
    }
    if (resolved_mode != SD_VAE_EXEC_COMFY_NORMAL) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu currently requires COMFY_NORMAL");
        return false;
    }
    ScopedVaeImplicitGemmConv implicit_conv_scope(resolved_mode);

    int64_t t0 = ggml_time_ms();
    auto gpu_image = sd_ctx->sd->first_stage_model->decode_latent_resource_to_backend_resource(sd_ctx->sd->n_threads,
                                                                                               resource->tensor.get(),
                                                                                               sd_ctx->sd->vae_tiling_params,
                                                                                               sd_ctx->sd->circular_x,
                                                                                               sd_ctx->sd->circular_y);
    int64_t t1 = ggml_time_ms();
    sd_vae_memory_report_t graph_report = sd_ctx->sd->first_stage_model->get_last_graph_report();
    sd_vae_memory_report_t full_report;
    sd_vae_memory_report_init(&full_report);
    copy_vae_report(&full_report, graph_report, effective, resolved_mode, false, used_taesd);
    log_vae_report("decode_gpu_latent", full_report);
    if (report != nullptr) {
        *report = full_report;
    }
    if (sd_ctx->sd->free_params_immediately && sd_ctx->sd->first_stage_model != nullptr) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (vae_report_large_im2col_disallowed(graph_report, effective)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu refused oversized IM2COL tensor: largest=%" PRIu64 " threshold=%" PRIu64,
                  graph_report.largest_tensor_bytes,
                  effective.im2col_warn_bytes);
        return false;
    }
    if (vae_report_comfy_guard_failed(full_report, resolved_mode)) {
        return false;
    }
    if (sd_strict_gpu_resident_enabled() &&
        (!full_report.device_resident_stages || full_report.stage_boundary_host_copies != 0)) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu strict GPU resident check failed: device_resident=%s host_copies=%u",
                  full_report.device_resident_stages ? "true" : "false",
                  full_report.stage_boundary_host_copies);
        return false;
    }
    if (gpu_image == nullptr || gpu_image->empty()) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu failed after %.2fs", (t1 - t0) * 1.0f / 1000);
        return false;
    }
    sd_gpu_handle_t handle = sd_gpu_register_resource(sd_ctx,
                                                      std::move(gpu_image),
                                                      SD_GPU_RESOURCE_IMAGE,
                                                      SD_LAYOUT_WHCN_GGML,
                                                      SD_GPU_RESOURCE_FLAG_VAE_DECODE_OUTPUT | SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE,
                                                      "vae_decode_rgb_f32_from_gpu_latent");
    if (handle == 0) {
        LOG_ERROR("sd_decode_gpu_latent_normal_gpu failed to register GPU image handle");
        return false;
    }
    *out_gpu_image = handle;
    LOG_INFO("sd_decode_gpu_latent_normal_gpu completed, taking %.2fs input_handle=%" PRIu64 " output_handle=%" PRIu64,
             (t1 - t0) * 1.0f / 1000,
             gpu_latent,
             handle);
    return true;
}

SD_API bool sd_encode_gpu_image_normal_gpu(sd_ctx_t* sd_ctx,
                                           sd_gpu_handle_t gpu_image,
                                           const sd_vae_run_options_t* options,
                                           sd_gpu_handle_t* out_gpu_latent,
                                           sd_vae_memory_report_t* report) {
    if (sd_strict_gpu_resident_enabled()) {
        LOG_ERROR("sd_encode_gpu_image_normal_gpu strict mode refused GPU image encode bridge; current path downloads the image before CPU VAE encode");
        if (out_gpu_latent != nullptr) {
            *out_gpu_latent = 0;
        }
        if (report != nullptr) {
            sd_vae_memory_report_init(report);
        }
        return false;
    }
    auto resource = sd_gpu_resource_lookup(sd_ctx, gpu_image);
    if (resource == nullptr || resource->kind != SD_GPU_RESOURCE_IMAGE || resource->tensor == nullptr || resource->tensor->empty()) {
        return false;
    }
    sd_image_t image{};
    if (!sd_gpu_image_download(sd_ctx, gpu_image, &image, nullptr)) {
        return false;
    }
    bool ok = sd_encode_image_normal_gpu(sd_ctx, &image, options, out_gpu_latent, report);
    free(image.data);
    return ok;
}

SD_API bool sd_release_clip_model_params(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }

    auto release_runner = [](auto& runner) {
        if (runner) {
            runner->free_params_buffer();
        }
    };

    release_runner(sd_ctx->sd->cond_stage_model);
    release_runner(sd_ctx->sd->clip_vision);
    release_runner(sd_ctx->sd->pmid_model);
    release_runner(sd_ctx->sd->pmid_lora);
    for (auto& lora : sd_ctx->sd->cond_stage_lora_models) {
        release_runner(lora);
    }
    return true;
}

SD_API bool sd_release_diffusion_model_params(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }

    auto release_runner = [](auto& runner) {
        if (runner) {
            runner->free_params_buffer();
        }
    };

    release_runner(sd_ctx->sd->diffusion_model);
    release_runner(sd_ctx->sd->high_noise_diffusion_model);
    release_runner(sd_ctx->sd->control_net);
    for (auto& lora : sd_ctx->sd->diffusion_lora_models) {
        release_runner(lora);
    }
    return true;
}

SD_API void free_sd_latent(sd_latent_t* latent) {
    if (latent == nullptr) {
        return;
    }
    auto* private_latent = static_cast<sd_latent_private_t*>(latent->opaque);
    delete private_latent;
    latent->opaque = nullptr;
    free(latent);
}

SD_API void free_sd_image(sd_image_t* image) {
    if (image == nullptr) {
        return;
    }
    free(image->data);
    image->data = nullptr;
    free(image);
}

SD_API void sd_free_downloaded_image(void* ptr) {
    free(ptr);
}

static std::optional<ImageGenerationLatents> prepare_video_generation_latents(sd_ctx_t* sd_ctx,
                                                                              const sd_vid_gen_params_t* sd_vid_gen_params,
                                                                              GenerationRequest* request) {
    ImageGenerationLatents latents;
    int64_t prepare_start_ms = ggml_time_ms();

    sd::Tensor<float> start_image;
    sd::Tensor<float> end_image;

    if (sd_vid_gen_params->init_image.data) {
        start_image = sd_image_to_tensor(sd_vid_gen_params->init_image, request->width, request->height);
    }

    if (sd_vid_gen_params->end_image.data) {
        end_image = sd_image_to_tensor(sd_vid_gen_params->end_image, request->width, request->height);
    }

    if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-I2V-14B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
        sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
        LOG_INFO("IMG2VID");

        if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-14B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-I2V-1.3B" ||
            sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
            if (!start_image.empty()) {
                auto clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2);
                if (clip_vision_output.empty()) {
                    LOG_ERROR("failed to compute clip vision output for init image");
                    return std::nullopt;
                }
                latents.clip_vision_output = std::move(clip_vision_output);
            } else {
                latents.clip_vision_output = sd_ctx->sd->get_clip_vision_output(start_image, false, -2, true);
            }

            if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-FLF2V-14B") {
                sd::Tensor<float> end_image_clip_vision_output;
                if (!end_image.empty()) {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2);
                    if (end_image_clip_vision_output.empty()) {
                        LOG_ERROR("failed to compute clip vision output for end image");
                        return std::nullopt;
                    }
                } else {
                    end_image_clip_vision_output = sd_ctx->sd->get_clip_vision_output(end_image, false, -2, true);
                }
                latents.clip_vision_output = sd::ops::concat(latents.clip_vision_output, end_image_clip_vision_output, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_INFO("get_clip_vision_output completed, taking %" PRId64 " ms", t1 - prepare_start_ms);
        }

        int64_t t1              = ggml_time_ms();
        sd::Tensor<float> image = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        if (!start_image.empty()) {
            sd::ops::slice_assign(&image, 2, 0, 1, start_image.unsqueeze(2));
        }
        if (!end_image.empty()) {
            sd::ops::slice_assign(&image, 2, request->frames - 1, request->frames, end_image.unsqueeze(2));
        }

        auto concat_latent = sd_ctx->sd->encode_first_stage(image);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (concat_latent.empty()) {
            LOG_ERROR("failed to encode video conditioning frames");
            return std::nullopt;
        }
        latents.concat_latent = std::move(concat_latent);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);

        sd::Tensor<float> concat_mask = sd::zeros<float>({latents.concat_latent.shape()[0],
                                                          latents.concat_latent.shape()[1],
                                                          latents.concat_latent.shape()[2],
                                                          4,
                                                          1});  // [b, 4, t, h/vae_scale_factor, w/vae_scale_factor]
        if (!start_image.empty()) {
            sd::ops::fill_slice(&concat_mask, 2, 0, 1, 1.0f);
        }
        if (!end_image.empty()) {
            auto last_channel = sd::ops::slice(concat_mask, 3, 3, 4);
            sd::ops::fill_slice(&last_channel, 2, last_channel.shape()[2] - 1, last_channel.shape()[2], 1.0f);
            sd::ops::slice_assign(&concat_mask, 3, 3, 4, last_channel);
        }
        latents.concat_latent = sd::ops::concat(concat_mask, latents.concat_latent, 3);  // [b, 4+c, t, h/vae_scale_factor, w/vae_scale_factor]
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.2-TI2V-5B" && !start_image.empty()) {
        LOG_INFO("IMG2VID");

        int64_t t1             = ggml_time_ms();
        auto init_img          = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
        auto init_image_latent = sd_ctx->sd->encode_first_stage(init_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
        if (init_image_latent.empty()) {
            LOG_ERROR("failed to encode init video frame");
            return std::nullopt;
        }

        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        sd::ops::slice_assign(&latents.init_latent, 2, 0, init_image_latent.shape()[2], init_image_latent);

        latents.denoise_mask = sd::full<float>({latents.init_latent.shape()[0], latents.init_latent.shape()[1], latents.init_latent.shape()[2], 1, 1}, 1.f);
        sd::ops::fill_slice(&latents.denoise_mask, 2, 0, init_image_latent.shape()[2], 0.0f);

        int64_t t2 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    } else if (sd_ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
               sd_ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B") {
        LOG_INFO("VACE");
        int64_t t1 = ggml_time_ms();
        sd::Tensor<float> ref_image_latent;
        if (!start_image.empty()) {
            auto ref_img     = start_image.reshape({start_image.shape()[0], start_image.shape()[1], 1, start_image.shape()[2], 1});
            auto encoded_ref = sd_ctx->sd->encode_first_stage(ref_img);  // [b, c, 1, h/vae_scale_factor, w/vae_scale_factor]
            if (encoded_ref.empty()) {
                LOG_ERROR("failed to encode VACE reference image");
                return std::nullopt;
            }
            ref_image_latent = sd::ops::concat(encoded_ref, sd::zeros<float>(encoded_ref.shape()), 3);  // [b, 2*c, 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        sd::Tensor<float> control_video = sd::full<float>({request->width, request->height, request->frames, 3, 1}, 0.5f);
        int64_t control_frame_count     = std::min<int64_t>(request->frames, sd_vid_gen_params->control_frames_size);
        for (int64_t i = 0; i < control_frame_count; ++i) {
            auto control_frame = sd_image_to_tensor(sd_vid_gen_params->control_frames[i], request->width, request->height);
            sd::ops::slice_assign(&control_video, 2, i, i + 1, control_frame.unsqueeze(2));
        }

        sd::Tensor<float> mask = sd::full<float>({request->width, request->height, request->frames, 1, 1}, 1.0f);

        control_video              = control_video - 0.5f;
        sd::Tensor<float> inactive = control_video * (1.0f - mask) + 0.5f;
        sd::Tensor<float> reactive = control_video * mask + 0.5f;

        inactive = sd_ctx->sd->encode_first_stage(inactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (inactive.empty()) {
            LOG_ERROR("failed to encode VACE inactive context");
            return std::nullopt;
        }

        reactive = sd_ctx->sd->encode_first_stage(reactive);  // [b, c, t, h/vae_scale_factor, w/vae_scale_factor]
        if (reactive.empty()) {
            LOG_ERROR("failed to encode VACE reactive context");
            return std::nullopt;
        }

        int64_t length = inactive.shape()[2];
        if (!ref_image_latent.empty()) {
            length += 1;
            request->frames       = static_cast<int>((length - 1) * 4 + 1);
            latents.ref_image_num = 1;
        }
        auto vace_context = sd::ops::concat(inactive, reactive, 3);  // [b, 2*c, t, h/vae_scale_factor, w/vae_scale_factor]

        mask              = sd::full<float>({request->width, request->height, inactive.shape()[2], 1, 1}, 1.0f);
        auto mask_context = mask.reshape({request->vae_scale_factor,
                                          inactive.shape()[0],
                                          request->vae_scale_factor,
                                          inactive.shape()[1],
                                          inactive.shape()[2]});   // [t, h/vae_scale_factor, vae_scale_factor, w/vae_scale_factor, vae_scale_factor]
        mask_context      = mask_context.permute({1, 3, 4, 0, 2})  // [vae_scale_factor, vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]
                           .reshape({inactive.shape()[0],
                                     inactive.shape()[1],
                                     inactive.shape()[2],
                                     request->vae_scale_factor * request->vae_scale_factor});  // [vae_scale_factor*vae_scale_factor, t, h/vae_scale_factor, w/vae_scale_factor]

        if (!ref_image_latent.empty()) {
            vace_context  = sd::ops::concat(ref_image_latent, vace_context, 2);  // [b, 2*c, t+1, h/vae_scale_factor, w/vae_scale_factor]
            auto mask_pad = sd::zeros<float>({mask_context.shape()[0],
                                              mask_context.shape()[1],
                                              1,
                                              mask_context.shape()[3]});  // [vae_scale_factor*vae_scale_factor, 1, h/vae_scale_factor, w/vae_scale_factor]
            mask_context  = sd::ops::concat(mask_pad, mask_context, 2);   // [vae_scale_factor*vae_scale_factor, t + 1, h/vae_scale_factor, w/vae_scale_factor]
        }

        mask_context.unsqueeze_(mask_context.dim());  // [b, vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]

        latents.vace_context = sd::ops::concat(vace_context, mask_context, 3);  // [b, 2*c + vae_scale_factor*vae_scale_factor, t + 1 or t, h/vae_scale_factor, w/vae_scale_factor]
        int64_t t2           = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %" PRId64 " ms", t2 - t1);
    }

    if (latents.init_latent.empty()) {
        latents.init_latent = sd_ctx->sd->generate_init_latent(request->width, request->height, request->frames, true);
    }

    return latents;
}

static ImageGenerationEmbeds prepare_video_generation_embeds(sd_ctx_t* sd_ctx,
                                                             const sd_vid_gen_params_t* sd_vid_gen_params,
                                                             const GenerationRequest& request,
                                                             const ImageGenerationLatents& latents) {
    ImageGenerationEmbeds embeds;
    ConditionerParams condition_params;
    condition_params.clip_skip       = request.clip_skip;
    condition_params.text            = request.prompt;
    condition_params.zero_out_masked = true;

    int64_t prepare_start_ms = ggml_time_ms();
    embeds.cond              = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                   condition_params);
    embeds.cond.c_concat     = latents.concat_latent;
    embeds.cond.c_vector     = latents.clip_vision_output;
    if (request.use_uncond) {
        condition_params.text  = request.negative_prompt;
        embeds.uncond          = sd_ctx->sd->cond_stage_model->get_learned_condition(sd_ctx->sd->n_threads,
                                                                                     condition_params);
        embeds.uncond.c_concat = latents.concat_latent;
        embeds.uncond.c_vector = latents.clip_vision_output;
    }

    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %.2fs", (t1 - prepare_start_ms) * 1.0f / 1000);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }
    return embeds;
}

static sd_image_t* decode_video_outputs(sd_ctx_t* sd_ctx,
                                        const sd::Tensor<float>& final_latent,
                                        int* num_frames_out) {
    if (final_latent.empty()) {
        LOG_ERROR("no latent video to decode");
        return nullptr;
    }
    int64_t t4            = ggml_time_ms();
    sd::Tensor<float> vid = sd_ctx->sd->decode_first_stage(final_latent, true);
    int64_t t5            = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t5 - t4) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }
    if (vid.empty()) {
        LOG_ERROR("decode_first_stage failed for video");
        return nullptr;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(vid.shape()[2], sizeof(sd_image_t));
    if (result_images == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = static_cast<int>(vid.shape()[2]);
    }

    for (int64_t i = 0; i < vid.shape()[2]; i++) {
        result_images[i] = tensor_to_sd_image(vid, static_cast<int>(i));
    }

    return result_images;
}

SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out) {
    if (sd_ctx == nullptr || sd_vid_gen_params == nullptr) {
        return nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }
    int64_t t0                    = ggml_time_ms();
    sd_ctx->sd->vae_tiling_params = sd_vid_gen_params->vae_tiling_params;
    GenerationRequest request(sd_ctx, sd_vid_gen_params);
    sd_ctx->sd->rng->manual_seed(request.seed);
    sd_ctx->sd->sampler_rng->manual_seed(request.seed);
    sd_ctx->sd->set_flow_shift(sd_vid_gen_params->sample_params.flow_shift);
    sd_ctx->sd->apply_loras(sd_vid_gen_params->loras, sd_vid_gen_params->lora_count);

    SamplePlan plan(sd_ctx, sd_vid_gen_params, request);
    auto latent_inputs_opt = prepare_video_generation_latents(sd_ctx, sd_vid_gen_params, &request);
    if (!latent_inputs_opt.has_value()) {
        return nullptr;
    }
    ImageGenerationLatents latents = std::move(*latent_inputs_opt);
    ImageGenerationEmbeds embeds   = prepare_video_generation_embeds(sd_ctx,
                                                                     sd_vid_gen_params,
                                                                     request,
                                                                     latents);
    LOG_INFO("generate_video %dx%dx%d",
             request.width,
             request.height,
             request.frames);

    int64_t latent_start = ggml_time_ms();
    int W                = request.width / request.vae_scale_factor;
    int H                = request.height / request.vae_scale_factor;
    int T                = static_cast<int>(latents.init_latent.shape()[2]);

    sd::Tensor<float> x_t   = latents.init_latent;
    sd::Tensor<float> noise = sd::Tensor<float>::randn_like(x_t, sd_ctx->sd->rng);

    if (plan.high_noise_sample_steps > 0) {
        LOG_DEBUG("sample(high noise) %dx%dx%d", W, H, T);

        int64_t sampling_start = ggml_time_ms();
        std::vector<float> high_noise_sigmas(plan.sigmas.begin(), plan.sigmas.begin() + plan.high_noise_sample_steps + 1);
        plan.sigmas = std::vector<float>(plan.sigmas.begin() + plan.high_noise_sample_steps, plan.sigmas.end());

        sd::Tensor<float> x_t_sampled = sd_ctx->sd->sample(sd_ctx->sd->high_noise_diffusion_model,
                                                           false,
                                                           x_t,
                                                           std::move(noise),
                                                           embeds.cond,
                                                           request.use_high_noise_uncond ? embeds.uncond : SDCondition(),
                                                           embeds.img_cond,
                                                           embeds.id_cond,
                                                           sd::Tensor<float>(),
                                                           0.f,
                                                           request.high_noise_guidance,
                                                           plan.high_noise_eta,
                                                           plan.high_noise_s_noise,
                                                           plan.dpmpp_sde_r,
                                                           plan.high_noise_dpmpp_sde_solver,
                                                           request.shifted_timestep,
                                                           plan.high_noise_sample_method,
                                                           sd_ctx->sd->is_flow_denoiser(),
                                                           high_noise_sigmas,
                                                           -1,
                                                           std::vector<sd::Tensor<float>>{},
                                                           false,
                                                           latents.denoise_mask,
                                                           latents.vace_context,
                                                           request.vace_strength,
                                                           request.cache_params);
        int64_t sampling_end          = ggml_time_ms();
        if (x_t_sampled.empty()) {
            LOG_ERROR("sampling(high noise) failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            if (sd_ctx->sd->free_params_immediately) {
                sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
            }
            return nullptr;
        }

        x_t   = std::move(x_t_sampled);
        noise = {};
        LOG_INFO("sampling(high noise) completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->high_noise_diffusion_model->free_params_buffer();
        }
    }

    LOG_DEBUG("sample %dx%dx%d", W, H, T);
    int64_t sampling_start         = ggml_time_ms();
    sd::Tensor<float> final_latent = sd_ctx->sd->sample(sd_ctx->sd->diffusion_model,
                                                        true,
                                                        x_t,
                                                        std::move(noise),
                                                        embeds.cond,
                                                        request.use_uncond ? embeds.uncond : SDCondition(),
                                                        embeds.img_cond,
                                                        embeds.id_cond,
                                                        sd::Tensor<float>(),
                                                        0.f,
                                                        sd_vid_gen_params->sample_params.guidance,
                                                        plan.eta,
                                                        plan.s_noise,
                                                        plan.dpmpp_sde_r,
                                                        plan.dpmpp_sde_solver,
                                                        sd_vid_gen_params->sample_params.shifted_timestep,
                                                        plan.sample_method,
                                                        sd_ctx->sd->is_flow_denoiser(),
                                                        plan.sigmas,
                                                        -1,
                                                        std::vector<sd::Tensor<float>>{},
                                                        false,
                                                        latents.denoise_mask,
                                                        latents.vace_context,
                                                        request.vace_strength,
                                                        request.cache_params);

    int64_t sampling_end = ggml_time_ms();
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    if (final_latent.empty()) {
        LOG_ERROR("sampling failed after %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
        return nullptr;
    }
    LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

    if (latents.ref_image_num > 0) {
        final_latent = sd::ops::slice(final_latent, 2, latents.ref_image_num, final_latent.shape()[2]);
    }

    int64_t latent_end = ggml_time_ms();
    LOG_INFO("generating latent video completed, taking %.2fs", (latent_end - latent_start) * 1.0f / 1000);

    auto result = decode_video_outputs(sd_ctx, final_latent, num_frames_out);
    if (result == nullptr) {
        return nullptr;
    }

    sd_ctx->sd->lora_stat();

    int64_t t1 = ggml_time_ms();
    LOG_INFO("generate_video completed in %.2fs", (t1 - t0) * 1.0f / 1000);
    return result;
}
