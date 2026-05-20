#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD,
    DPMPP2S_A_SAMPLE_METHOD,
    DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD,
    IPNDM_SAMPLE_METHOD,
    IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD,
    DDIM_TRAILING_SAMPLE_METHOD,
    TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD,
    RES_2S_SAMPLE_METHOD,
    ER_SDE_SAMPLE_METHOD,
    DPMPP_SDE_SAMPLE_METHOD,
    DPMPP_SDE_GPU_SAMPLE_METHOD,
    DPMPP2M_SDE_SAMPLE_METHOD,
    DPMPP2M_SDE_GPU_SAMPLE_METHOD,
    DPMPP2M_SDE_HEUN_SAMPLE_METHOD,
    DPMPP2M_SDE_HEUN_GPU_SAMPLE_METHOD,
    DPMPP3M_SDE_SAMPLE_METHOD,
    DPMPP3M_SDE_GPU_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum dpmpp_sde_solver_t {
    DPMPP_SDE_SOLVER_MIDPOINT,
    DPMPP_SDE_SOLVER_HEUN,
    DPMPP_SDE_SOLVER_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    BETA_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    FLUX2_FLOW_PRED,
    PREDICTION_COUNT
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
    SD_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale)
    SD_TYPE_COUNT = 41,
};

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};

enum sd_preview_schedule_mode_t {
    SD_PREVIEW_SCHEDULE_EVERY_N_STEPS = 0,
    SD_PREVIEW_SCHEDULE_PERCENT_INTERVAL = 1,
    SD_PREVIEW_SCHEDULE_EXPLICIT_PERCENTS = 2,
};

enum {
    SD_PREVIEW_API_VERSION = 1,
    SD_PREVIEW_MAX_PERCENT_POINTS = 16,
};

typedef struct sd_preview_options_t {
    uint32_t struct_size;
    uint32_t version;
    enum preview_t mode;
    enum sd_preview_schedule_mode_t schedule_mode;
    int step_interval;
    float percent_interval;
    float percent_points[SD_PREVIEW_MAX_PERCENT_POINTS];
    uint32_t percent_point_count;
    bool include_first_step;
    bool include_final_step;
    bool denoised;
    bool noisy;
    uint32_t reserved[8];
} sd_preview_options_t;

enum lora_apply_mode_t {
    LORA_APPLY_AUTO,
    LORA_APPLY_IMMEDIATELY,
    LORA_APPLY_AT_RUNTIME,
    LORA_APPLY_MODE_COUNT,
};

typedef struct {
    bool enabled;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
} sd_tiling_params_t;

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* clip_vision_path;
    const char* t5xxl_path;
    const char* llm_path;
    const char* llm_vision_path;
    const char* diffusion_model_path;
    const char* high_noise_diffusion_model_path;
    const char* vae_path;
    const char* taesd_path;
    const char* control_net_path;
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path;
    const char* tensor_type_rules;
    bool vae_decode_only;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool offload_params_to_cpu;
    bool enable_mmap;
    bool keep_clip_on_cpu;
    bool keep_control_net_on_cpu;
    bool keep_vae_on_cpu;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool tae_preview_only;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool circular_x;
    bool circular_y;
    bool force_sdxl_vae_conv_scale;
    bool chroma_use_dit_mask;
    bool chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    bool qwen_image_zero_cond_t;
} sd_ctx_params_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct sd_latent_t {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint64_t element_count;
    void* opaque;
} sd_latent_t;

enum sd_vae_exec_mode_t {
    SD_VAE_EXEC_LEGACY_GGML_GRAPH,
    SD_VAE_EXEC_DIRECT_GRAPH,
    SD_VAE_EXEC_COMFY_NORMAL,
    SD_VAE_EXEC_AUTO,
};

enum sd_vae_dtype_t {
    SD_VAE_DTYPE_AUTO,
    SD_VAE_DTYPE_BF16,
    SD_VAE_DTYPE_F16,
    SD_VAE_DTYPE_F32,
};

enum {
    SD_VAE_API_VERSION = 1,
};

typedef struct sd_vae_run_options_t {
    uint32_t struct_size;
    uint32_t version;
    enum sd_vae_exec_mode_t mode;
    enum sd_vae_dtype_t storage_dtype;
    bool fail_on_large_im2col;
    bool allow_tiling;
    bool allow_taesd;
    uint64_t im2col_warn_bytes;
    uint32_t reserved[8];
} sd_vae_run_options_t;

typedef struct sd_vae_memory_report_t {
    uint32_t struct_size;
    uint32_t version;
    enum sd_vae_exec_mode_t requested_mode;
    enum sd_vae_exec_mode_t resolved_mode;
    enum sd_vae_dtype_t requested_storage_dtype;
    enum sd_vae_dtype_t resolved_storage_dtype;
    uint64_t planned_workspace_bytes;
    uint64_t largest_tensor_bytes;
    uint64_t estimated_peak_bytes;
    uint64_t measured_peak_bytes;
    uint32_t graph_count;
    uint32_t stage_count;
    uint32_t stage_boundary_host_copies;
    uint32_t stage_boundary_device_copies;
    uint32_t stage_boundary_dtype_promotions;
    bool used_im2col;
    bool used_direct_conv;
    bool used_tiling;
    bool used_taesd;
    bool compact_activation_storage;
    bool device_resident_stages;
    char largest_tensor_op[32];
    char largest_tensor_type[16];
    char largest_tensor_shape[96];
    char stage_output_dtype[16][16];
    char stage_output_backend[16][32];
    char math_dtype_policy[96];
    char fallback_reason[192];
    uint32_t decode_setup_ms;
    uint32_t decode_context_ms;
    uint32_t decode_latent_d2d_ms;
    uint32_t decode_graph_ms;
    uint32_t decode_image_d2d_ms;
    uint32_t decode_download_ms;
    uint32_t decode_context_reuse;
    uint32_t decode_same_context_attempted;
    uint32_t decode_same_context_succeeded;
    uint32_t reserved[7];
} sd_vae_memory_report_t;

typedef struct sd_vae_capabilities_t {
    uint32_t struct_size;
    uint32_t version;
    bool supports_comfy_normal;
    bool supports_device_resident_stages;
    // Production-default compact BF16 storage support. Experimental env-gated
    // BF16 VAE paths intentionally keep this false until promoted.
    bool supports_bf16_storage;
    bool supports_f16_storage;
    bool supports_normal_encode;
    bool supports_normal_decode;
    bool supports_memory_report;
    bool supports_no_im2col_guard;
    uint32_t reserved[8];
} sd_vae_capabilities_t;

typedef uint64_t sd_gpu_handle_t;

enum sd_backend_kind_t {
    SD_BACKEND_CPU = 0,
    SD_BACKEND_CUDA = 1,
};

enum sd_gpu_resource_kind_t {
    SD_GPU_RESOURCE_TENSOR = 1,
    SD_GPU_RESOURCE_IMAGE = 2,
    SD_GPU_RESOURCE_LATENT = 3,
};

enum sd_tensor_dtype_t {
    SD_DTYPE_F32 = 0,
    SD_DTYPE_F16 = 1,
    SD_DTYPE_BF16 = 2,
    SD_DTYPE_U8 = 3,
};

enum sd_tensor_layout_t {
    SD_LAYOUT_NCHW = 0,
    SD_LAYOUT_NHWC = 1,
    SD_LAYOUT_WHCN_GGML = 2,
    SD_LAYOUT_PACKED_RGBA8 = 3,
};

typedef struct sd_latent_view_t {
    uint32_t struct_size;
    uint32_t version;
    enum sd_tensor_dtype_t dtype;
    enum sd_tensor_layout_t layout;
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t stride_n;
    int64_t stride_c;
    int64_t stride_h;
    int64_t stride_w;
    uint64_t element_count;
    uint64_t byte_size;
    const float* data;
    uint32_t flags;
    uint32_t reserved[8];
} sd_latent_view_t;

enum sd_image_color_space_t {
    SD_COLOR_LINEAR_RGB = 0,
    SD_COLOR_SRGB = 1,
};

enum sd_gpu_resource_flags_t {
    SD_GPU_RESOURCE_FLAG_VAE_DECODE_OUTPUT = 1u << 0,
    SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE = 1u << 1,
    SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT = 1u << 2,
    SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD = 1u << 3,
    SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_DOWNLOAD = 1u << 4,
    SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT = 1u << 5,
    SD_GPU_RESOURCE_FLAG_REQUIRES_ISOLATED_VAE_DECODE = 1u << 6,
};

typedef struct sd_gpu_device_info_t {
    uint32_t struct_size;
    uint32_t version;
    enum sd_backend_kind_t backend;
    int device_index;
    char device_name[128];
    uint64_t total_memory_bytes;
    uint32_t reserved[8];
} sd_gpu_device_info_t;

typedef struct sd_gpu_tensor_desc_t {
    uint32_t struct_size;
    uint32_t version;
    sd_gpu_handle_t handle;
    enum sd_gpu_resource_kind_t kind;
    enum sd_backend_kind_t backend;
    int device_index;
    enum sd_tensor_dtype_t dtype;
    enum sd_tensor_layout_t layout;
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t stride_n;
    int64_t stride_c;
    int64_t stride_h;
    int64_t stride_w;
    uint64_t byte_offset;
    uint64_t byte_size;
    uint64_t producer_stream_id;
    uint64_t ready_event_id;
    uint32_t flags;
    uint32_t refcount;
    uint32_t reserved[16];
} sd_gpu_tensor_desc_t;

typedef struct sd_gpu_capabilities_t {
    uint32_t struct_size;
    uint32_t version;
    bool supports_gpu_handles;
    bool supports_cuda_gpu_handles;
    bool supports_gpu_latent_output;
    bool supports_gpu_latent_input;
    bool supports_sampler_gpu_latent_output;
    bool supports_vae_gpu_latent_input;
    bool supports_vae_encode_gpu_latent_output;
    bool supports_vae_encode_gpu_latent_bridge_output;
    bool supports_gpu_image_output;
    bool supports_gpu_image_to_rgba8;
    bool supports_gpu_download;
    bool supports_gpu_latent_download;
    bool supports_gpu_latent_upload;
    bool supports_dlpack_export;
    bool supports_cuda_pointer_borrow;
    bool supports_cuda_ipc_export;
    bool supports_external_memory_interop;
    bool supports_sampler_gpu_init_latent_input;
    bool supports_sampler_gpu_init_latent_bridge_input;
    bool supports_sampler_gpu_latent_bridge_output;
    bool supports_sampler_imported_initial_noise;
    bool supports_sampler_imported_step_noise_schedule;
    bool supports_sampler_brownian_step_noise_import;
    bool supports_sampler_step_noise_count_query;
    bool supports_flux2_gpu_latent_output;
    bool supports_flux2_flow_backend_sampler;
    bool supports_flux2_vae_decode_gpu;
    bool supports_flux2_qwen_conditioning_gpu_resident;
    bool supports_z_image_gpu_latent_output;
    bool supports_z_image_flow_backend_sampler;
    bool supports_z_image_vae_decode_gpu;
    bool supports_z_image_qwen_conditioning_gpu_resident;
    bool supports_qwen_image_gpu_latent_output;
    bool supports_qwen_image_flow_backend_sampler;
    bool supports_qwen_image_vae_decode_gpu;
    bool supports_qwen_image_qwen_conditioning_gpu_resident;
    uint32_t reserved[4];
} sd_gpu_capabilities_t;

typedef uint64_t sd_conditioning_handle_t;

enum sd_conditioning_flags_t {
    SD_CONDITIONING_FLAG_HOST_TENSOR = 1u << 0,
    SD_CONDITIONING_FLAG_DEVICE_RESIDENT = 1u << 1,
    SD_CONDITIONING_FLAG_UPLOADED_BACKEND_TENSOR = 1u << 2,
    SD_CONDITIONING_FLAG_PER_STEP_UPLOAD_FALLBACK = 1u << 3,
};

typedef struct sd_conditioning_desc_t {
    uint32_t struct_size;
    uint32_t version;
    sd_conditioning_handle_t handle;
    uint32_t flags;
    uint32_t refcount;
    int64_t batch;
    int64_t token_count;
    int64_t crossattn_dim;
    int64_t vector_dim;
    int64_t concat_channels;
    int64_t t5_token_count;
    enum sd_tensor_dtype_t dtype;
    enum sd_backend_kind_t backend;
    uint64_t estimated_bytes;
    bool device_resident;
    bool has_crossattn;
    bool has_vector;
    bool has_concat;
    bool has_t5_ids;
    bool has_t5_weights;
    bool copy_safe;
    uint32_t extra_crossattn_count;
    int clip_skip;
    int width;
    int height;
    bool zero_out_masked;
    char debug_name[64];
    uint32_t reserved[16];
} sd_conditioning_desc_t;

typedef struct sd_conditioning_encode_options_t {
    uint32_t struct_size;
    uint32_t version;
    int clip_skip;
    int width;
    int height;
    bool force_zero_uncond;
    const char* cache_key_hint;
    uint32_t reserved[16];
} sd_conditioning_encode_options_t;

typedef struct sd_conditioning_capabilities_t {
    uint32_t struct_size;
    uint32_t version;
    bool supports_text_conditioning_encode;
    bool supports_conditioning_handles;
    bool supports_conditioning_gpu_resident;
    bool supports_sampler_conditioning_handle_input;
    bool supports_conditioning_handle_reuse;
    bool supports_conditioning_cpu_resident;
    bool supports_sampler_conditioning_init_latent_input;
    bool supports_sampler_conditioning_gpu_init_latent_bridge_input;
    bool supports_flux2_qwen_conditioning;
    bool supports_flux2_qwen_conditioning_gpu_resident;
    bool supports_z_image_qwen_conditioning;
    bool supports_z_image_qwen_conditioning_gpu_resident;
    bool supports_qwen_image_qwen_conditioning;
    bool supports_qwen_image_qwen_conditioning_gpu_resident;
    bool supports_conditioning_per_step_upload_fallback;
    uint32_t reserved[12];
} sd_conditioning_capabilities_t;

enum sd_model_family_t {
    SD_MODEL_FAMILY_UNKNOWN = 0,
    SD_MODEL_FAMILY_SD1 = 1,
    SD_MODEL_FAMILY_SD2 = 2,
    SD_MODEL_FAMILY_SDXL = 3,
    SD_MODEL_FAMILY_SD3 = 4,
    SD_MODEL_FAMILY_FLUX = 5,
    SD_MODEL_FAMILY_FLUX2 = 6,
    SD_MODEL_FAMILY_Z_IMAGE = 7,
    SD_MODEL_FAMILY_WAN = 8,
    SD_MODEL_FAMILY_QWEN_IMAGE = 9,
    SD_MODEL_FAMILY_ANIMA = 10,
    SD_MODEL_FAMILY_MARIGOLD_IID = 11,
};

typedef struct sd_model_pipeline_capabilities_t {
    uint32_t struct_size;
    uint32_t version;
    enum sd_model_family_t family;
    char family_name[32];
    uint32_t latent_channels;
    uint32_t vae_scale_factor;
    enum sample_method_t default_sample_method;
    enum scheduler_t default_scheduler;
    float default_cfg_scale;
    int default_steps;
    float default_flow_shift;
    bool requires_clip_l;
    bool requires_clip_g;
    bool requires_t5xxl;
    bool requires_llm;
    bool supports_text_to_image;
    bool supports_image_to_image;
    bool supports_gpu_sample_bridge_output;
    bool supports_gpu_latent_decode;
    bool supports_gpu_image_output;
    bool supports_vae_encode;
    bool supports_vae_encode_gpu_output;
    bool supports_reference_images;
    bool supports_edit_mode;
    bool supports_edit_reference_conditioning;
    bool supports_comfy_reference_vae_encode;
    bool strict_gpu_sample_is_true_resident;
    bool supports_intrinsic_image_decomposition;
    uint32_t intrinsic_target_count;
    bool supports_flux2_model_load;
    bool supports_flux2_qwen_conditioning;
    bool supports_flux2_qwen_conditioning_gpu_resident;
    bool supports_flux2_flow_backend_sampler;
    bool supports_flux2_gpu_latent_output;
    bool supports_flux2_vae_decode_gpu;
    bool supports_flux2_vae_bf16_or_compact_storage;
    bool supports_flux2_controlnet;
    bool supports_flux2_masks;
    bool supports_flux2_reference;
    bool supports_flux2_edit;
    bool supports_flux2_multibatch;
    bool supports_z_image_model_load;
    bool supports_z_image_qwen_conditioning;
    bool supports_z_image_qwen_conditioning_gpu_resident;
    bool supports_z_image_flow_backend_sampler;
    bool supports_z_image_gpu_latent_output;
    bool supports_z_image_vae_decode_gpu;
    bool supports_z_image_vae_bf16_or_compact_storage;
    bool supports_z_image_controlnet;
    bool supports_z_image_masks;
    bool supports_z_image_reference;
    bool supports_z_image_edit;
    bool supports_z_image_multibatch;
    bool supports_qwen_image_model_load;
    bool supports_qwen_image_qwen_conditioning;
    bool supports_qwen_image_qwen_conditioning_gpu_resident;
    bool supports_qwen_image_flow_backend_sampler;
    bool supports_qwen_image_gpu_latent_output;
    bool supports_qwen_image_vae_decode_gpu;
    bool supports_qwen_image_vae_bf16_or_compact_storage;
    bool supports_qwen_image_controlnet;
    bool supports_qwen_image_masks;
    bool supports_qwen_image_reference;
    bool supports_qwen_image_edit;
    bool supports_qwen_image_multibatch;
} sd_model_pipeline_capabilities_t;

typedef struct sd_marigold_iid_options_t {
    uint32_t struct_size;
    uint32_t version;
    uint32_t processing_width;
    uint32_t processing_height;
    uint32_t steps;
    int64_t seed;
    bool match_input_resolution;
    uint32_t reserved[8];
} sd_marigold_iid_options_t;

typedef struct sd_marigold_iid_result_t {
    uint32_t struct_size;
    uint32_t version;
    uint32_t target_count;
    sd_image_t* targets;
    const char** target_names;
    sd_latent_t* latent;
    uint32_t reserved[8];
} sd_marigold_iid_result_t;

typedef struct sd_download_options_t {
    uint32_t struct_size;
    uint32_t version;
    bool synchronize;
    uint32_t reserved[8];
} sd_download_options_t;

typedef struct sd_cuda_borrowed_ptr_t {
    uint32_t struct_size;
    uint32_t version;
    void* device_ptr;
    uint64_t byte_size;
    int device_index;
    enum sd_tensor_dtype_t dtype;
    enum sd_tensor_layout_t layout;
    int64_t shape[4];
    int64_t strides[4];
    uint64_t ready_event_id;
    uint64_t producer_stream_id;
    uint32_t reserved[8];
} sd_cuda_borrowed_ptr_t;

typedef struct {
    int* layers;
    size_t layer_count;
    float layer_start;
    float layer_end;
    float scale;
} sd_slg_params_t;

typedef struct {
    float txt_cfg;
    float img_cfg;
    float distilled_guidance;
    sd_slg_params_t slg;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    float s_noise;
    float dpmpp_sde_r;
    enum dpmpp_sde_solver_t dpmpp_sde_solver;
    int shifted_timestep;
    float* custom_sigmas;
    int custom_sigmas_count;
    float flow_shift;
} sd_sample_params_t;

typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;  // photo maker

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
    SD_CACHE_SPECTRUM,
};

typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const char* scm_mask;
    bool scm_policy_dynamic;
    float spectrum_w;
    int spectrum_m;
    float spectrum_lam;
    int spectrum_window_size;
    float spectrum_flex_window;
    int spectrum_warmup_steps;
    float spectrum_stop_percent;
} sd_cache_params_t;

typedef struct {
    bool is_high_noise;
    float multiplier;
    const char* path;
} sd_lora_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t* ref_images;
    int ref_images_count;
    bool auto_resize_ref_image;
    bool increase_ref_index;
    sd_image_t mask_image;
    int width;
    int height;
    sd_sample_params_t sample_params;
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_pm_params_t pm_params;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
} sd_img_gen_params_t;

typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t end_image;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float moe_boundary;
    float strength;
    int64_t seed;
    int video_frames;
    float vace_strength;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
} sd_vid_gen_params_t;

typedef struct sd_ctx_t sd_ctx_t;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
SD_API void sd_preview_options_init(sd_preview_options_t* options);
SD_API void sd_set_preview_callback_v2(sd_preview_cb_t cb, const sd_preview_options_t* options, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);
SD_API const char* sd_rng_type_name(enum rng_type_t rng_type);
SD_API enum rng_type_t str_to_rng_type(const char* str);
SD_API const char* sd_sample_method_name(enum sample_method_t sample_method);
SD_API enum sample_method_t str_to_sample_method(const char* str);
SD_API bool sd_sampler_uses_step_noise(enum sample_method_t sample_method);
SD_API bool sd_sampler_uses_brownian_step_noise(enum sample_method_t sample_method);
SD_API uint32_t sd_sampler_step_noise_count(enum sample_method_t sample_method, uint32_t sigma_count);
SD_API const char* sd_scheduler_name(enum scheduler_t scheduler);
SD_API enum scheduler_t str_to_scheduler(const char* str);
SD_API const char* sd_prediction_name(enum prediction_t prediction);
SD_API enum prediction_t str_to_prediction(const char* str);
SD_API const char* sd_preview_name(enum preview_t preview);
SD_API enum preview_t str_to_preview(const char* str);
SD_API const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode);
SD_API enum lora_apply_mode_t str_to_lora_apply_mode(const char* str);

SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);

SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
SD_API char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);
SD_API sd_latent_t* sd_encode_image(sd_ctx_t* sd_ctx,
                                    const sd_image_t* image,
                                    const sd_tiling_params_t* vae_tiling_params);
SD_API sd_latent_t* sd_sample_latent(sd_ctx_t* sd_ctx,
                                     const sd_img_gen_params_t* sd_img_gen_params,
                                     const sd_latent_t* init_latent);
SD_API bool sd_sample_latent_gpu(sd_ctx_t* sd_ctx,
                                 const sd_img_gen_params_t* sd_img_gen_params,
                                 const sd_latent_t* init_latent,
                                 sd_gpu_handle_t* out_gpu_latent);
SD_API bool sd_sample_latent_gpu_with_init_gpu(sd_ctx_t* sd_ctx,
                                               const sd_img_gen_params_t* sd_img_gen_params,
                                               sd_gpu_handle_t init_gpu_latent,
                                               sd_gpu_handle_t* out_gpu_latent);
SD_API bool sd_sample_latent_gpu_with_conditioning(sd_ctx_t* sd_ctx,
                                                   const sd_img_gen_params_t* sd_img_gen_params,
                                                   const sd_latent_t* init_latent,
                                                   sd_conditioning_handle_t positive,
                                                   sd_conditioning_handle_t negative,
                                                   sd_gpu_handle_t* out_gpu_latent);
SD_API bool sd_sample_latent_gpu_with_init_gpu_and_conditioning(sd_ctx_t* sd_ctx,
                                                               const sd_img_gen_params_t* sd_img_gen_params,
                                                               sd_gpu_handle_t init_gpu_latent,
                                                               sd_conditioning_handle_t positive,
                                                               sd_conditioning_handle_t negative,
                                                               sd_gpu_handle_t* out_gpu_latent);
SD_API bool sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_gpu(sd_ctx_t* sd_ctx,
                                                                             const sd_img_gen_params_t* sd_img_gen_params,
                                                                             sd_gpu_handle_t init_gpu_latent,
                                                                             sd_gpu_handle_t noise_gpu_latent,
                                                                             sd_conditioning_handle_t positive,
                                                                             sd_conditioning_handle_t negative,
                                                                             sd_gpu_handle_t* out_gpu_latent);
SD_API bool sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_schedule_gpu(sd_ctx_t* sd_ctx,
                                                                                       const sd_img_gen_params_t* sd_img_gen_params,
                                                                                       sd_gpu_handle_t init_gpu_latent,
                                                                                       sd_gpu_handle_t noise_gpu_latent,
                                                                                       const sd_gpu_handle_t* step_noise_gpu_latents,
                                                                                       uint32_t step_noise_count,
                                                                                       sd_conditioning_handle_t positive,
                                                                                       sd_conditioning_handle_t negative,
                                                                                       sd_gpu_handle_t* out_gpu_latent);
// Experimental SDXL/SD1 Euler proof path. Requires SDCPP_EXPERIMENTAL_TRUE_GPU_SAMPLER=1.
// This is not production sampling: initial noise is device-procedural rather than seed-compatible Philox.
SD_API bool sd_sample_latent_gpu_true_euler_spike(sd_ctx_t* sd_ctx,
                                                  const sd_img_gen_params_t* sd_img_gen_params,
                                                  sd_gpu_handle_t* out_gpu_latent);
SD_API sd_image_t* sd_decode_latent(sd_ctx_t* sd_ctx,
                                    const sd_latent_t* latent,
                                    const sd_tiling_params_t* vae_tiling_params);
SD_API void sd_vae_run_options_init(sd_vae_run_options_t* options);
SD_API void sd_vae_memory_report_init(sd_vae_memory_report_t* report);
SD_API sd_latent_t* sd_encode_image_normal(sd_ctx_t* sd_ctx,
                                           const sd_image_t* image,
                                           const sd_vae_run_options_t* options,
                                           sd_vae_memory_report_t* report);
SD_API sd_image_t* sd_decode_latent_normal(sd_ctx_t* sd_ctx,
                                           const sd_latent_t* latent,
                                           const sd_vae_run_options_t* options,
                                           sd_vae_memory_report_t* report);
SD_API bool sd_estimate_vae_normal_memory(sd_ctx_t* sd_ctx,
                                          uint32_t width,
                                          uint32_t height,
                                          bool decode,
                                          const sd_vae_run_options_t* options,
                                          sd_vae_memory_report_t* report);
SD_API bool sd_get_vae_capabilities(sd_ctx_t* sd_ctx, sd_vae_capabilities_t* capabilities);
SD_API bool sd_get_gpu_capabilities(sd_ctx_t* sd_ctx, sd_gpu_capabilities_t* capabilities);
SD_API bool sd_get_conditioning_capabilities(sd_ctx_t* sd_ctx, sd_conditioning_capabilities_t* capabilities);
SD_API bool sd_get_model_pipeline_capabilities(sd_ctx_t* sd_ctx, sd_model_pipeline_capabilities_t* capabilities);
SD_API void sd_marigold_iid_options_init(sd_marigold_iid_options_t* options);
SD_API sd_marigold_iid_result_t* sd_marigold_iid_predict(sd_ctx_t* sd_ctx,
                                                         const sd_image_t* image,
                                                         const sd_marigold_iid_options_t* options);
SD_API void free_sd_marigold_iid_result(sd_marigold_iid_result_t* result);
SD_API bool sd_gpu_handle_retain(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle);
SD_API bool sd_gpu_handle_release(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle);
SD_API bool sd_gpu_handle_get_desc(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, sd_gpu_tensor_desc_t* desc);
SD_API bool sd_gpu_handle_debug_name(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, const char* name);
SD_API bool sd_gpu_handle_borrow_cuda_ptr(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle, sd_cuda_borrowed_ptr_t* out);
SD_API bool sd_gpu_handle_end_cuda_borrow(sd_ctx_t* sd_ctx, sd_gpu_handle_t handle);
SD_API sd_latent_t* sd_gpu_latent_download(sd_ctx_t* sd_ctx,
                                           sd_gpu_handle_t gpu_latent,
                                           const sd_download_options_t* options);
SD_API bool sd_cpu_latent_upload(sd_ctx_t* sd_ctx,
                                 const sd_latent_t* cpu_latent,
                                 sd_gpu_handle_t* out_gpu_latent,
                                 const sd_download_options_t* options);
SD_API bool sd_decode_latent_normal_gpu(sd_ctx_t* sd_ctx,
                                        const sd_latent_t* latent,
                                        const sd_vae_run_options_t* options,
                                        sd_gpu_handle_t* out_gpu_image,
                                        sd_vae_memory_report_t* report);
SD_API bool sd_encode_image_normal_gpu(sd_ctx_t* sd_ctx,
                                       const sd_image_t* image,
                                       const sd_vae_run_options_t* options,
                                       sd_gpu_handle_t* out_gpu_latent,
                                       sd_vae_memory_report_t* report);
SD_API bool sd_prewarm_vae_decode_bridge(sd_ctx_t* sd_ctx,
                                          const sd_vae_run_options_t* options,
                                          sd_vae_memory_report_t* report);
SD_API bool sd_decode_gpu_latent_normal_gpu(sd_ctx_t* sd_ctx,
                                            sd_gpu_handle_t gpu_latent,
                                            const sd_vae_run_options_t* options,
                                            sd_gpu_handle_t* out_gpu_image,
                                            sd_vae_memory_report_t* report);
SD_API bool sd_encode_gpu_image_normal_gpu(sd_ctx_t* sd_ctx,
                                           sd_gpu_handle_t gpu_image,
                                           const sd_vae_run_options_t* options,
                                           sd_gpu_handle_t* out_gpu_latent,
                                           sd_vae_memory_report_t* report);
SD_API bool sd_gpu_image_download(sd_ctx_t* sd_ctx,
                                  sd_gpu_handle_t gpu_image,
                                  sd_image_t* out_cpu_image,
                                  const sd_download_options_t* options);
SD_API bool sd_gpu_image_download_to_buffer(sd_ctx_t* sd_ctx,
                                            sd_gpu_handle_t gpu_image,
                                            void* dst_rgba8,
                                            uint64_t dst_bytes,
                                            uint64_t dst_stride_bytes,
                                            const sd_download_options_t* options);
SD_API bool sd_gpu_tensor_download(sd_ctx_t* sd_ctx,
                                   sd_gpu_handle_t gpu_tensor,
                                   void* dst,
                                   uint64_t dst_bytes,
                                   const sd_download_options_t* options);
SD_API bool sd_latent_get_view(const sd_latent_t* latent,
                               sd_latent_view_t* out_view);
SD_API bool sd_latent_export_f32(const sd_latent_t* latent,
                                 float* dst,
                                 uint64_t dst_elements,
                                 sd_latent_view_t* out_view);
SD_API sd_latent_t* sd_latent_import_f32(const float* data,
                                         uint64_t element_count,
                                         uint32_t w,
                                         uint32_t h,
                                         uint32_t c);
SD_API bool sd_gpu_latent_export_f32_nchw_debug(sd_ctx_t* sd_ctx,
                                                sd_gpu_handle_t gpu_latent,
                                                float* dst,
                                                uint64_t dst_elements,
                                                sd_latent_view_t* out_view,
                                                const sd_download_options_t* options);
SD_API void sd_conditioning_encode_options_init(sd_conditioning_encode_options_t* options);
SD_API bool sd_conditioning_encode_text(sd_ctx_t* sd_ctx,
                                        const char* text,
                                        const sd_conditioning_encode_options_t* options,
                                        sd_conditioning_handle_t* out_handle,
                                        sd_conditioning_desc_t* out_desc);
SD_API bool sd_conditioning_encode_text_with_ref_images(sd_ctx_t* sd_ctx,
                                                        const char* text,
                                                        const sd_image_t* ref_images,
                                                        uint32_t ref_images_count,
                                                        const sd_conditioning_encode_options_t* options,
                                                        sd_conditioning_handle_t* out_handle,
                                                        sd_conditioning_desc_t* out_desc);
SD_API bool sd_conditioning_retain(sd_ctx_t* sd_ctx, sd_conditioning_handle_t handle);
SD_API bool sd_conditioning_release(sd_ctx_t* sd_ctx, sd_conditioning_handle_t handle);
SD_API bool sd_conditioning_get_desc(sd_ctx_t* sd_ctx,
                                     sd_conditioning_handle_t handle,
                                     sd_conditioning_desc_t* out_desc);
SD_API bool sd_conditioning_debug_name(sd_ctx_t* sd_ctx,
                                       sd_conditioning_handle_t handle,
                                       const char* name);
SD_API bool sd_release_clip_model_params(sd_ctx_t* sd_ctx);
SD_API bool sd_release_diffusion_model_params(sd_ctx_t* sd_ctx);
SD_API void free_sd_latent(sd_latent_t* latent);
SD_API void free_sd_image(sd_image_t* image);
SD_API void sd_free_downloaded_image(void* ptr);

SD_API void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params);
SD_API sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool offload_params_to_cpu,
                                        bool direct,
                                        int n_threads,
                                        int tile_size);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx,
                          sd_image_t input_image,
                          uint32_t upscale_factor);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);

SD_API bool preprocess_canny(sd_image_t image,
                             float high_threshold,
                             float low_threshold,
                             float weak,
                             float strong,
                             bool inverse);

SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
