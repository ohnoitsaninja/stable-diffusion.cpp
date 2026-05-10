# Paralol Flux and Z-Image Pipeline Capabilities

This fork exposes model-family capability metadata so Paralol can route SDXL,
Flux-family, Flux2, and Z-Image workflows without hard-coding latent shapes or
assuming SDXL semantics.

## Public API

Use `sd_get_model_pipeline_capabilities()` after creating the context.

Important fields:

- `family` / `family_name`: model family classification.
- `latent_channels`: diffusion latent channel count expected by the sampler and
  VAE handoff.
- `vae_scale_factor`: model VAE spatial scale.
- `default_sample_method`, `default_scheduler`, `default_cfg_scale`,
  `default_steps`, `default_flow_shift`: safe workflow defaults for this family.
- `requires_clip_l`, `requires_t5xxl`, `requires_llm`: required text encoder
  lanes for split-model loading.
- `supports_gpu_sample_bridge_output`: sampler can return a GPU latent handle,
  currently by bridge-uploading the final latent.
- `supports_gpu_latent_decode`: `sd_decode_gpu_latent_normal_gpu()` is supported
  for this family.
- `supports_gpu_image_output`: VAE decode can return a GPU image handle.
- `supports_reference_images`: generation params may carry reference images.
- `supports_edit_mode`: the model expects edit/reference conditioning rather
  than plain text-to-image only conditioning.
- `supports_edit_reference_conditioning`: reference images are consumed by the
  conditioning path and by the edit concat latent path.
- `supports_comfy_reference_vae_encode`: reference image VAE encoding is routed
  through COMFY_NORMAL/implicit-GEMM instead of the legacy IM2COL graph.
- `strict_gpu_sample_is_true_resident`: false until the sampler loop itself is
  fully GPU-resident.

## Current Family Status

| Family | Latent | VAE Scale | Text Encoder | GPU latent -> GPU image |
| --- | ---: | ---: | --- | --- |
| SDXL | 4 channels | 8 | CLIP stack | Supported |
| Flux / Flux1-style | 16 channels | model-reported | CLIP-L + T5XXL | Supported where the upstream VAE path matches Flux AE semantics |
| Z-Image | 16 channels | 8 | Qwen LLM | Supported |
| Flux2 | 128 channels | 16 | Qwen LLM | Supported |

## Z-Image Verification

Validated with:

- Diffusion model:
  `C:\tmp\stable-diffusion.cpp-paralol\build\hf-z-image-gguf\z_image_turbo-Q4_K.gguf`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\ae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf`
- Resolution: `1024x1024`
- Steps: `9`
- CFG: `1.0`
- Sampler/scheduler: `euler` / `discrete`

Observed API handoff:

- sampled GPU latent: `1x16x128x128`, f32, CUDA, 1,048,576 bytes
- VAE decode output: `1x3x1024x1024`, f32, CUDA
- COMFY_NORMAL VAE decode: supported
- implicit-GEMM conv: enabled
- tiled VAE: false
- TAESD: false
- IM2COL: false
- stage host copies: 0
- planned workspace: 2816 MB
- decode time: about 0.8 seconds on the local 4080 Super

The output image was coherent in the API smoke:

`C:\tmp\stable-diffusion.cpp-paralol\build\flux-z-api-smoke\z-image-api-1024-rerun.png`

## Flux2 Verification

Flux2 Klein 4B can render through the existing CLI path with split
`--diffusion-model`, `--vae`, and `--llm` arguments.

Flux2 is also exposed through the GPU latent -> GPU VAE decode path. It uses a
128-channel diffusion latent and a separate Flux2 VAE mean/std transform. The
fork represents that transform as a small backend graph before staged
COMFY_NORMAL decode, so the handoff remains device-resident.

Validated with:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\flux-2-klein-4b-fp8.safetensors`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf`
- Resolution: `1024x1024`
- Steps: `4`
- CFG: `1.0`
- Sampler/scheduler: `euler` / `discrete`

Observed API handoff:

- `family_name=flux2`
- `latent_channels=128`
- `vae_scale_factor=16`
- `supports_reference_images=true`
- `supports_edit_mode=true`
- `supports_edit_reference_conditioning=true`
- `supports_comfy_reference_vae_encode=true`
- `supports_gpu_sample_bridge_output=true`
- `supports_gpu_latent_decode=true`
- `supports_gpu_image_output=true`
- sampled GPU latent: `1x128x64x64`, f32, CUDA, 2,097,152 bytes
- VAE decode output: `1x3x1024x1024`, f32, CUDA
- COMFY_NORMAL VAE decode: supported
- implicit-GEMM conv: enabled
- tiled VAE: false
- TAESD: false
- IM2COL: false
- stage host copies: 0
- planned workspace: 2816 MB
- decode time: about 0.8 seconds on the local 4080 Super

The output image was coherent in the API smoke:

`C:\tmp\stable-diffusion.cpp-paralol\build\flux-z-verification\flux2-gpu-handoff-1024.png`

## Flux2 Edit / Reference Conditioning

Flux2 Klein edit mode is exposed through the same `sd_img_gen_params_t`
reference image fields used by the CLI `-r` path:

- set `ref_images` and `ref_images_count`
- keep `auto_resize_ref_image=true` unless the caller already matched the model
  target size/multiple
- use `sd_sample_latent_gpu()` to produce the sampled diffusion latent handle
- use `sd_decode_gpu_latent_normal_gpu()` to decode that handle into a GPU image

For CUDA Flux2, reference image VAE encoding now uses the normal
COMFY_NORMAL/implicit-GEMM path. The old `encode_first_stage()` path is refused
for COMFY_NORMAL-capable families if normal VAE encode fails, so edit workflows
do not quietly re-enter the oversized legacy IM2COL graph.

CLI edit mode was validated with a cat reference image and the prompt:

`put a black top hat, a monocle, and a cane on the cat`

The 1024 test produced a coherent edit output at:

`C:\tmp\stable-diffusion.cpp-paralol\build\flux-edit-verification\flux2-klein-edit-cat-1024.png`

## Paralol Integration Guidance

Paralol should choose node behavior from `sd_get_model_pipeline_capabilities()`
instead of filename checks.

For Z-Image:

1. Load split diffusion, AE/VAE, and Qwen LLM paths.
2. Use the reported sampler defaults unless the workflow overrides them.
3. Accept a `16`-channel latent shape for KSampler output.
4. Use `sd_sample_latent_gpu()` followed by
   `sd_decode_gpu_latent_normal_gpu()`.
5. Download image bytes only at preview/save/export boundaries.

For Flux2:

1. Load split diffusion, Flux2 VAE, and Qwen LLM paths.
2. Use the reported sampler defaults unless the workflow overrides them.
3. Accept a `128`-channel latent shape for KSampler output.
4. For edit mode, pass the source image through `ref_images` on
   `sd_img_gen_params_t`.
5. Use `sd_sample_latent_gpu()` followed by
   `sd_decode_gpu_latent_normal_gpu()`.
6. Download image bytes only at preview/save/export boundaries.
