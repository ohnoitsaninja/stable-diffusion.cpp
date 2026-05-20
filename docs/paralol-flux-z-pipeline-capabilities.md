# Paralol Flux and Z-Image Pipeline Capabilities

This fork exposes model-family capability metadata so Paralol can route SDXL,
Flux-family, Flux2, Z-Image, Qwen-Image, and Anima workflows without hard-coding
latent shapes or assuming SDXL semantics.

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
- `supports_gpu_latent_decode`: `sd_decode_gpu_latent_normal_gpu()` is a true
  device-resident decode for this family.
- `supports_gpu_image_output`: VAE decode can return a GPU image handle without
  an internal CPU bridge.
- `supports_gpu_latent_decode_bridge` / `supports_gpu_image_output_bridge`:
  `sd_decode_gpu_latent_normal_gpu()` can consume and return GPU handles, but
  the VAE path downloads the latent for legacy decode and re-uploads the image.
  Strict GPU-resident mode refuses this bridge.
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
| SD1 base | 4 channels | 8 | CLIP | Supported for base SD1; variants need their own smokes |
| SDXL | 4 channels | 8 | CLIP stack | Supported |
| Flux / Flux1-style | 16 channels | model-reported | CLIP-L + T5XXL | Supported |
| Z-Image | 16 channels | 8 | Qwen LLM | Supported |
| Flux2 | 128 channels | 16 | Qwen LLM | Supported |
| Qwen-Image | 16 channels | 8 | Qwen LLM | Sampler latent supported; VAE image decode remains experimental |
| Anima | 16 channels | 8 | Qwen LLM + T5 ids/weights | Sampler latent supported; env-gated direct GPU VAE decode |

## Shared Qwen Flow Backend Contract

Flux2, Z-Image, Qwen-Image, and Anima use the same narrow fork-side sampler
contract instead of one-off code paths:

- a model-family env gate enables the backend sampler lane;
- `sd_conditioning_encode_text()` produces reusable Qwen conditioning handles;
- the handles are uploaded once to backend tensors and consumed by reference;
- strict mode refuses host-backed conditioning or bridge sampler output;
- the sampler returns an `SD_GPU_RESOURCE_LATENT` handle with no CPU bridge
  flags;
- text-only GPU init latents are accepted by the same backend path for I2I
  smoke coverage;
- Flux2 Klein and Qwen-Image edit models also accept one or more reference
  images through the sampler `ref_images` path; those images are encoded to
  reference latents, uploaded once to backend tensors, and then consumed by
  reference during denoising;
- masks, ControlNet, multibatch, and image-CFG stay unsupported;
- `sd_decode_gpu_latent_normal_gpu()` consumes that handle and returns an
  `SD_GPU_RESOURCE_IMAGE` handle only for families whose VAE path is validated;
- CPU pixels are only materialized by explicit caller-owned download.

The shared helpers currently advertise only families that satisfy that exact
contract. A new Qwen/flow family should be added by wiring its model version
into the shared flow-backend predicates, then proving the same smoke checks.
Do not add a new family-specific capability bit or claim GPU residency until
the conditioning descriptor reports `device_resident=true`, sampler timing
reports `sampler_math_residency=gpu_backend_tensor`, init-latent timing reports
`init_bridge_download=false` when an init handle is supplied, and the sampled
latent has no bridge flags in `SDCPP_STRICT_GPU_RESIDENT=1`.

## Anima Verification

Anima has a strict fork-side sampler lane. Its Wan/Qwen VAE output path is now
available as an explicit experimental direct GPU decode lane. Enable it with:

```powershell
$env:SDCPP_EXPERIMENTAL_ANIMA_BACKEND=1
$env:SDCPP_ANIMA_TEXT_ENCODER_CPU_PARAMS=1
$env:SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1
```

The supported sampler lane is text-only T2I, batch 1, Qwen/T5 conditioning
handles, no ControlNet, no masks, no reference/edit images, and no multibatch.
Validated sampler methods are `euler`, `euler_a`, `er_sde`, and
`dpmpp_2m_sde_gpu`; CFG is allowed for Anima and the model capability defaults
remain `er_sde`, cfg `4.5`, 30 steps.

The fork applies ComfyUI's Wan21 latent mean/std transform for Anima and
Qwen-Image latents. The default path still keeps the older CPU compatibility
decode bridge. With `SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1`, the same transform
is applied inside the Wan/Qwen VAE graph and `sd_decode_gpu_latent_normal_gpu()`
returns a CUDA image handle without the latent download / image re-upload
bridge. This path is strict-mode clean for Anima T2I. For single-frame
Anima/Qwen image decode, CausalConv3d is reduced to an equivalent direct Conv2D
slice and uses the existing CUDA implicit-GEMM Conv2D backend. Set
`SDCPP_DISABLE_WAN_QWEN_VAE_DIRECT_CONV3D=1` to force the older Conv3D/IM2COL
graph for diagnostics.

Use realistic Anima settings for image-quality checks. The official Anima model
card recommends about 1MP output, 30-50 steps, CFG 4-5, and lists `er_sde`,
`euler_a`, and `dpmpp_2m_sde_gpu` as preferred sampler choices. Very short
Euler/cfg=1 smoke runs are useful for API and residency validation, but they
produce misleading low-quality images and must not be used as quality
acceptance.

Validated smoke:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\anima-base-v1.0.safetensors`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\qwen_image_vae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\qwen_3_06b_base.safetensors`
- Resolution: `512x512`
- Steps: `2` sampler-only strict smoke, `30` decoded image smoke
- CFG: `4.5`
- Sampler/scheduler: `er_sde` / `discrete`

Observed strict handoff without the experimental Wan/Qwen VAE GPU flag:

- conditioning handle: device-resident backend tensors, including Anima
  `t5_ids` and `t5_weights`
- conditioning per-step upload: false
- sampled GPU latent: `1x16x64x64`, f32, CUDA, 262,144 bytes
- sampler math residency: `gpu_backend_tensor`
- sampler bridge flags: none
- VAE decode/image output: bridge path; CPU compatibility decode applies Wan21
  latent scaling and re-uploads the image for GPU-handle compatibility

Observed strict handoff with `SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1`:

- `sd_decode_gpu_latent_normal_gpu()` accepts the Anima CUDA latent handle
- VAE decode bridge flags: none
- VAE host copies: 0
- VAE device copies: 0
- GPU image handle: `1x3x512x512`, f32, CUDA
- caller-owned explicit image download: supported
- strict GPU resident mode: passes
- VAE graph: `direct_graph`, direct conv true, IM2COL false
- planned workspace: about 676.2 MiB at 512x512
- largest tensor: about 290.3 MiB, `PAD`
- decode graph: about 129 ms on the local RTX 4080 SUPER

Optional compact Wan/Qwen VAE activation storage is available behind a second
experimental gate:

```powershell
$env:SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_BF16=1
```

This mirrors ComfyUI's practical Wan/Qwen VAE dtype choice on Ada-class CUDA
devices: Comfy's Wan VAE loader allows `torch.bfloat16`, `torch.float16`, and
`torch.float32`, and its model-management dtype resolver prefers bf16 on CUDA
devices with compute capability 8.x or newer. There is no separate Comfy
Wan/Qwen "small quant VAE" runtime path to mirror; the relevant memory lever is
activation dtype. The fork therefore does not introduce a quantized VAE format
here.

The BF16 path keeps public outputs f32 and keeps RMSNorm as an explicit f32
island because ggml CUDA RMS_NORM currently requires f32 input. On the local
RTX 4080 SUPER, the 512x512 30-step Anima decoded-image smoke measured:

- workspace: `676.2 MiB -> 338.3 MiB`
- largest tensor: `290.3 MiB f32 PAD -> 145.1 MiB bf16 PAD`
- decode graph: `129 ms -> 77 ms`
- BF16 vs F32 direct-conv output: mean abs `0.3879`, p99 `3`, PSNR
  `50.03 dB`

Keep this opt-in until more Qwen-Image/Wan-family VAE encode/decode cases are
validated. Capability fields continue to avoid promising BF16 as a default
production contract.

The 30-step ER-SDE Anima smoke is the meaningful image-quality acceptance here;
short 2-4 step samples are only API/residency smokes and can look misleadingly
bad. The direct-conv GPU VAE output matches the older bridge output closely
despite the different conv accumulation order, and it is slightly closer to a
ComfyUI Wan21 `process_out` decode of the same latent:

- direct-conv vs bridge: mean abs `0.2356`, p99 `2`, PSNR `52.95 dB`
- direct-conv vs Comfy Wan21: mean abs `0.4680`, p99 `3`, PSNR `48.50 dB`
- bridge vs Comfy Wan21: mean abs `0.4908`, p99 `4`, PSNR `48.09 dB`

## Z-Image Verification

Z-Image now has the same first-class T2I handoff shape as the Flux2 strict
backend lane, but remains explicitly gated. Enable it with:

```powershell
$env:SDCPP_EXPERIMENTAL_Z_IMAGE_BACKEND=1
```

The supported lane is intentionally narrow: Z-Image/Z-Image Turbo text T2I or
GPU-init I2I, batch 1, Euler, cfg `1.0`, Qwen text conditioning handles, no
ControlNet, no masks, no reference/edit images, and no multibatch. Unsupported requests fail closed in
`SDCPP_STRICT_GPU_RESIDENT=1`.

On 16 GB CUDA cards, use `SDCPP_Z_IMAGE_TEXT_ENCODER_CPU_PARAMS=1` for this
lane. The fork keeps Qwen parameters in RAM, executes Qwen on the GPU during
conditioning encode, then releases the text encoder before diffusion. Set
`SDCPP_DISABLE_Z_IMAGE_AUTO_RELEASE_TEXT_ENCODER=1` only for debugging. The
shared escape hatch `SDCPP_DISABLE_FLOW_BACKEND_AUTO_RELEASE_TEXT_ENCODER=1`
disables auto-release for both Flux2 and Z-Image.

The Z-Image backend reports these family-specific capability fields:

- `supports_z_image_model_load=true`
- `supports_z_image_qwen_conditioning=true`
- `supports_z_image_qwen_conditioning_gpu_resident=true`
- `supports_z_image_flow_backend_sampler=true`
- `supports_z_image_gpu_latent_output=true`
- `supports_z_image_vae_decode_gpu=true`
- `supports_z_image_controlnet=false`
- `supports_z_image_masks=false`
- `supports_z_image_reference=false`
- `supports_z_image_edit=false`
- `supports_z_image_multibatch=false`

The `supports_z_image_reference=false` and `supports_z_image_edit=false` flags
mean direct KSampler `ref_images` / VAE-reference-latent edit inputs are not
claimed. Z can still use a narrower reference-aware conditioning-handle path:

1. Encode the positive prompt with
   `sd_conditioning_encode_text_with_ref_images(...)`.
2. Pass only the resulting conditioning handle into KSampler.
3. Do not also pass `ref_images` to the sampler request.

In that mode the fork merges the Z `extra_c_crossattns` prompt chunks into the
primary cross-attention context, uploads that single context as a backend tensor,
and the flow sampler consumes it by reference with
`conditioning_per_step_upload=false`.

Validated target assets:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\z-image-turbo-q4_k_m.gguf`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\ae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf`
- Resolution: `1024x1024`
- Steps: `9`
- CFG: `1.0`
- Sampler/scheduler: `euler` / `discrete`

Expected API handoff:

- Qwen conditioning handle: CUDA device-resident cross-attention tensor
- conditioning per-step upload: false
- sampled GPU latent: `1x16x128x128`, f32, CUDA, 1,048,576 bytes
- sampler math residency: `gpu_backend_tensor`
- sampler bridge flags: none
- VAE decode output: `1x3x1024x1024`, f32, CUDA
- COMFY_NORMAL VAE decode: supported
- implicit-GEMM conv: enabled
- tiled VAE: false
- TAESD: false
- IM2COL: false
- stage host copies: 0

The smoke lane uses:

```powershell
$env:SDCPP_EXPERIMENTAL_Z_IMAGE_BACKEND=1
$env:SDCPP_STRICT_GPU_RESIDENT=1
build\codex\bin\sd-latent-smoke.exe `
  --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\z-image-turbo-q4_k_m.gguf `
  --vae F:\automatic1111\Stability\Models\VAE\ae.safetensors `
  --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf `
  --sample-without-init --gpu-flow-sampler --condition-handles `
  --z-image-text-encoder-cpu-params `
  --strict-gpu-resident --steps 9 --cfg-scale 1.0 `
  --sampling-method euler --width 1024 --height 1024 `
  --skip-estimate --gpu-latent-decode-input --gpu-decode-output `
  --download-gpu-output-buffer --dump-gpu-handle-desc
```

This proves the fork-side T2I handoff contract. It does not claim direct Z
KSampler reference/edit image inputs.

Current local validation caveat: the strict handoff smoke and the stock
`sd-cli` compatibility path both complete with the local
`z-image-turbo-q4_k_m.gguf` plus `Qwen3-4B-Q5_K_M.gguf` pairing, but both write
a blank/white image at 512px/9 steps. That means the GPU handoff path is not the
source of the blank output; the same model/text-encoder/VAE combination is not a
usable image-quality acceptance pair in this checkout. Keep Z-Image advertised
as API-handoff validated, not image-quality validated, until a known-good
Z/Qwen asset pair produces coherent pixels through the CLI compatibility path.

Validated narrow Z reference-aware conditioning-handle smoke:

- command shape: `--conditioning-ref-image <image> --condition-handles
  --sample-without-init --gpu-flow-sampler --strict-gpu-resident --no-decode`
- positive handle: `ref_images=1`, `device_resident=true`
- Z merge: `merged 2 extra cross-attention chunk(s) into primary context`
- sampler timing: `sampler_math_residency=gpu_backend_tensor`,
  `condition_input=handle`, `conditioning_storage=device_tensor`,
  `conditioning_per_step_upload=false`, `output_bridge_upload=false`
- sampled latent descriptor: `1x16x64x64` for the 512px smoke,
  `flags=SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`

Remaining Z reference/edit work is image-quality validation and any direct
KSampler ref-image/ref-latent path Paralol decides it needs. The currently
supported fork contract is the explicit conditioning-handle split above.

## Flux.1 Verification

Validated a Flux.1-style model with:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\flux1-kontext-dev-Q5_K_M.gguf`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\ae.safetensors`
- CLIP-L:
  `F:\automatic1111\Stability\Models\TextEncoders\clip_l.safetensors`
- T5XXL:
  `F:\automatic1111\Stability\Models\TextEncoders\t5-v1_1-xxl-encoder-Q3_K_L.gguf`
- Resolution: `512x512`
- Steps: `1`
- CFG: `1.0`

Observed T2I API handoff:

- `family_name=flux`
- `latent_channels=16`
- sampled GPU latent: `1x16x64x64`, f32, CUDA
- VAE decode output: `1x3x512x512`, f32, CUDA
- COMFY_NORMAL VAE decode: supported
- implicit-GEMM conv: enabled
- tiled VAE: false
- TAESD: false
- IM2COL: false
- stage host copies: 0
- planned workspace: 704 MB
- decode time: about 0.18 seconds on the local 4080 Super

The Flux.1 non-strict init-latent bridge path was also validated at 512:

- VAE Encode GPU latent: `1x16x64x64`, f32, CUDA
- `sd_sample_latent_gpu_with_init_gpu`: bridge-downloads init latent, samples,
  and bridge-uploads sampled latent
- isolated GPU VAE Decode: `im2col=false`, `tiled=false`, `taesd=false`,
  `host_copies=0`, planned workspace `704 MB`, decode about `0.17s`

## Flux2 Verification

Flux2 Klein 4B can render through the existing CLI path with split
`--diffusion-model`, `--vae`, and `--llm` arguments.

Flux2 is exposed through two different lanes:

- the existing broad compatibility lane, which may use older bridge/fallback
  behavior depending on the API call;
- the first strict GPU-resident T2I lane, gated by
  `SDCPP_EXPERIMENTAL_FLUX2_BACKEND=1`.

The strict lane is intentionally narrow: Flux2 Klein text T2I, GPU-init I2I, or
single-reference edit, batch 1, Euler, cfg `1.0`, Qwen conditioning handles, no
ControlNet, no masks, no multibatch, and no image-CFG. Unsupported requests
fail closed in `SDCPP_STRICT_GPU_RESIDENT=1`.

For 16 GB CUDA cards, use `SDCPP_FLUX2_TEXT_ENCODER_CPU_PARAMS=1` for this
lane. Flux2 Klein 4B plus Qwen 4B plus the Flux2 VAE can otherwise occupy about
15 GB before compute buffers, which leaves too little room for the Qwen and
Flux graphs and can trigger Windows GPU memory paging. This flag keeps Qwen
parameters in RAM and temporarily executes the text encoder on the GPU during
conditioning encode, while keeping the diffusion model, sampled latent, and VAE
handoff on the GPU.

When `sd_sample_latent_gpu_with_conditioning()` enters the Flux2 backend lane,
the fork automatically releases text encoder params after resident conditioning
handles are accepted and before the diffusion loop starts. Set
`SDCPP_DISABLE_FLUX2_AUTO_RELEASE_TEXT_ENCODER=1` only for debugging. The timing
line reports `text_encoder_released_before_diffusion` and
`text_encoder_release_ms`.

Flux2 uses a 128-channel diffusion latent and a separate Flux2 VAE mean/std
transform. The fork represents that transform as a small backend graph before
staged COMFY_NORMAL decode, so the VAE handoff remains device-resident.

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

Observed strict Flux2 backend handoff at 512x512:

- `family_name=flux2`
- `latent_channels=128`
- `vae_scale_factor=16`
- `supports_flux2_model_load=true`
- `supports_flux2_qwen_conditioning=true`
- `supports_flux2_qwen_conditioning_gpu_resident=true`
- `supports_flux2_flow_backend_sampler=true`
- `supports_flux2_gpu_latent_output=true`
- `supports_flux2_vae_decode_gpu=true`
- `supports_flux2_controlnet=false`
- `supports_flux2_masks=false`
- `supports_flux2_reference=true`
- `supports_flux2_edit=true`
- `supports_flux2_multibatch=false`
- `supports_gpu_latent_decode=true`
- `supports_gpu_image_output=true`
- conditioning handles: CUDA device-resident Qwen cross-attention tensors
- conditioning per-step upload: false
- sampled GPU latent: `1x128x32x32`, f32, CUDA, 524,288 bytes
- sampler math residency: `gpu_backend_tensor`
- sampler bridge flags: none
- VAE decode output: `1x3x512x512`, f32, CUDA
- COMFY_NORMAL VAE decode: supported
- implicit-GEMM conv: enabled
- tiled VAE: false
- TAESD: false
- IM2COL: false
- stage host copies: 0
- planned workspace: 352 MB
- explicit caller-owned image download: supported

Observed strict Flux2 GPU-init I2I handoff at 1024x1024:

- VAE Encode GPU latent: `1x128x64x64`, f32, CUDA, 2,097,152 bytes
- sampled GPU latent: `1x128x64x64`, f32, CUDA, no CPU bridge flags
- sampler timing reports `init_latent=gpu_handle`
- `init_bridge_download=false`
- `output_bridge_upload=false`
- conditioning handles are CUDA device-resident
- VAE Decode remains GPU-resident with explicit caller-owned image download

The strict backend smoke uses:

```powershell
$env:SDCPP_EXPERIMENTAL_FLUX2_BACKEND=1
$env:SDCPP_STRICT_GPU_RESIDENT=1
$env:SDCPP_FLUX2_TEXT_ENCODER_CPU_PARAMS=1
build\codex\bin\sd-latent-smoke.exe `
  --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\flux-2-klein-4b-fp8.safetensors `
  --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors `
  --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf `
  --image F:\Paralol\examples\orc.png `
  --sample-without-init --gpu-sample-output --gpu-flow-sampler `
  --flux2-text-encoder-cpu-params `
  --condition-handles --strict-gpu-resident `
  --steps 4 --cfg-scale 1.0 --sampling-method euler --width 512 --height 512 `
  --skip-estimate --gpu-latent-decode-input --gpu-decode-output `
  --download-gpu-output-buffer --dump-gpu-handle-desc
```

Current limitation: on this local GGUF Qwen/fp8 Klein setup, a 512x512 4-step
strict run is functionally correct but slow, roughly one minute per flow step.
That is backend Flux2 throughput work, not a CPU bridge or VAE handoff issue.
The structured timing logs identify `denoise_ms` separately from conditioning
and VAE decode.

## Flux2 Edit / Reference Conditioning

Flux2 Klein edit mode is exposed through the same `sd_img_gen_params_t`
reference image fields used by the CLI `-r` path:

- set `ref_images` and `ref_images_count`
- keep `auto_resize_ref_image=true` unless the caller already matched the model
  target size/multiple
- use resident Qwen conditioning handles for the positive/negative text inputs
- use `sd_sample_latent_gpu_with_conditioning()` or the matching init-latent
  variant to produce the sampled diffusion latent handle
- use `sd_decode_gpu_latent_normal_gpu()` to decode that handle into a GPU image

For the env-gated CUDA Flux2 Klein backend lane, reference image VAE encoding
still starts from a caller CPU image, but the encoded reference latent is
uploaded once into a backend tensor before denoising and then consumed by
reference by the Flux graph. The sampler timing line reports:

- `ref_latent_backend_upload_ms`
- `ref_latents`
- `ref_latents_backend=true`
- `ref_latents_per_step_upload=false`

This keeps the denoise loop from re-uploading reference latents every step.
Normal execution still fails closed for unsupported reference paths in strict
mode.

Do not use the Z-style `sd_conditioning_encode_text_with_ref_images(...)`
contract for Flux2 edit mode. Flux2 edit/reference support is the sampler
`ref_images` path: the image is VAE-encoded into a reference latent, then the
Flux graph consumes that backend reference latent alongside the sampled image
tokens. The text conditioning handle remains a Qwen text tensor.

Observed strict Flux2 Klein 512px single-reference edit smoke:

- `supports_flux2_reference=true`
- `supports_flux2_edit=true`
- Qwen conditioning handles: `device_resident=true`
- text encoder released before diffusion: `true`
- reference latents: `1`
- `ref_latents_backend=true`
- `ref_latents_per_step_upload=false`
- `ref_latent_backend_upload_ms=0`
- sampler residency: `gpu_backend_tensor`
- `init_bridge_download=false`
- `output_bridge_upload=false`
- `strict_gpu_resident=true`
- denoise: `382 ms` for a one-step 512px smoke
- decode graph: `147 ms`
- caller-owned image download: `3 ms`
- sampled latent descriptor: `1x128x32x32`, f32, CUDA, no bridge flags
- GPU image descriptor: `1x3x512x512`, f32, CUDA
- output:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux2-reference-edit-smoke\flux2-ref-edit-512-1step.png`
- logs:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux2-reference-edit-smoke\flux2-ref-edit-512-1step.stdout.log`

The matching no-decode sampler smoke is:
`F:\Paralol\local\stable-diffusion.cpp-speed\build\flux2-reference-edit-smoke\flux2-ref-sampler-1step.stdout.log`.

Observed 512px four-step edit smoke using the default Flux2 Klein step count
after the shared Qwen-flow reference/edit backend changes:

- sampler: Euler, cfg `1.0`, condition handles, strict GPU resident
- `sample_kdiffusion_gpu_backend`: `steps=4`, `unet_calls=4`,
  `backend_graph_calls=8`
- sampler timing: `latent_prepare_ms=95`, `condition_bind_ms=4`,
  `text_encoder_release_ms=0`, `denoise_ms=1160`, `total_ms=1314`
- reference latents: `1`, `ref_latents_backend=true`,
  `ref_latents_per_step_upload=false`
- bridges: `init_bridge_download=false`, `output_bridge_upload=false`
- decode graph: `159 ms`, planned workspace `704 MB`
- caller-owned image download: `3 ms`
- output:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux-qwen-ref-edit\flux2-klein-edit\flux2_edit.png`
- logs:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux-qwen-ref-edit\flux2-klein-edit\flux2_edit.stdout.log`

The 512px one-step smoke is a contract test, not a quality benchmark. It proves
the fork-side handoff for the full Flux2 edit/reference path:

1. Qwen positive/negative conditioning handles are encoded once.
2. Qwen text encoder params are released before diffusion on 16 GB cards.
3. The reference image is encoded by COMFY_NORMAL VAE.
4. The reference latent is uploaded once to a backend tensor.
5. The Flux2 backend sampler consumes text conditioning and reference latent by
   reference.
6. The sampled latent remains an `SD_GPU_RESOURCE_LATENT`.
7. VAE decode consumes the GPU latent and writes an `SD_GPU_RESOURCE_IMAGE`.
8. CPU pixels are produced only by explicit caller-owned download.

Z-Image reference/edit remains disabled. The Z/Qwen image-edit style path emits
resident extra cross-attention tensors from vision/text branches, and the
current strict backend sampler deliberately rejects `extra_c_crossattns` until
that tensor family is made device-resident and consumed by reference.

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
   `sd_img_gen_params_t`; this is currently advertised only for Flux2 Klein
   when `SDCPP_EXPERIMENTAL_FLUX2_BACKEND=1` is enabled.
5. Use `sd_sample_latent_gpu()` followed by
   `sd_decode_gpu_latent_normal_gpu()`.
6. Download image bytes only at preview/save/export boundaries.

For Qwen-Image and Anima:

1. Treat both as Qwen-family flow models, not SDXL variants.
2. Qwen-Image has a narrow env-gated strict sampler lane for T2I and
   single-reference edit:
   set `SDCPP_EXPERIMENTAL_QWEN_IMAGE_BACKEND=1` (or use
   `sd-latent-smoke --gpu-flow-sampler`) and keep the text encoder in RAM on
   16 GB cards with `SDCPP_QWEN_IMAGE_TEXT_ENCODER_CPU_PARAMS=1`.
3. Anima has the same narrow sampler contract behind
   `SDCPP_EXPERIMENTAL_ANIMA_BACKEND=1` with
   `SDCPP_ANIMA_TEXT_ENCODER_CPU_PARAMS=1`.
4. The supported strict sampler lane is batch 1, resident conditioning handles,
   no ControlNet, no masks, no multibatch, and no image-CFG. Qwen-Image
   supports Euler T2I and single-reference edit, including CFG/uncond for edit
   models. Anima has strict sampler smokes for `euler`, `euler_a`, `er_sde`, and
   `dpmpp_2m_sde_gpu` with cfg `4.5`. The sampler consumes resident
   conditioning tensors by reference and returns a CUDA `SD_GPU_RESOURCE_LATENT`
   with no sampler bridge flags.
5. Qwen-Image and Anima can use the experimental Wan/Qwen direct GPU VAE decode
   lane when `SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1` is set. In that mode
   `sd_decode_gpu_latent_normal_gpu()` consumes the CUDA latent handle and
   returns a CUDA image handle without bridge flags. Without that gate, older
   bridge-capable compatibility paths may still be available and must be
   reported as bridges.
6. Anima advertises workflow defaults (`er_sde`, cfg `4.5`, `30` steps). Use
   those settings for image-quality checks; short Euler/cfg=1 smokes are API
   validation only and produce misleading low-quality images.
7. Qwen-Image Edit 2511 requires `qwen_image_zero_cond_t`; use
   `--qwen-image-zero-cond-t` in smoke tooling and set
   `sd_ctx_params_t::qwen_image_zero_cond_t=true` from Paralol workflows that
   target the 2511 edit model.
8. On 16 GB CUDA cards, prefer keeping the Qwen-family text encoder params in
   RAM and running them on the GPU only during encode:
   `SDCPP_QWEN_IMAGE_TEXT_ENCODER_CPU_PARAMS=1` for Qwen-Image and
   `SDCPP_ANIMA_TEXT_ENCODER_CPU_PARAMS=1` for Anima. The smoke tool exposes
   matching flags:
   `--qwen-image-text-encoder-cpu-params` and
   `--anima-text-encoder-cpu-params`.
9. For capability and loader checks that should not launch a sampler, use
   `sd-latent-smoke --capabilities-only`. It creates the context, prints
   model/GPU capability fields, validates `--model-family` if supplied, and
   exits before image loading or diffusion.

Implementation note: Qwen-Image now has the same diffusion-model
`compute_to_backend_resource(...)` entry point shape as Flux and Z-Image, so the
model graph can consume a backend latent, backend context tensor, and optional
backend reference latents without forcing those tensors through host memory.
The strict Qwen-Image lane uses that path for both text T2I and single-reference
edit. Edit reference images are encoded to reference latents and uploaded once
before denoising; the timing line must report `ref_latents_backend=true` and
`ref_latents_per_step_upload=false`.

Validated Qwen-Image strict sampler smoke:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\qwen-image-2512-Q4_K_M.gguf`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\qwen_image_vae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`
- Resolution: `512x512`
- Steps: `1`
- CFG: `1.0`
- Sampler: `Euler`
- `family_name=qwen_image`
- `qwen_image_qwen_conditioning_gpu=true`
- `qwen_image_flow_backend_sampler=true`
- conditioning handle: `device_resident=true`
- sampled latent: CUDA `1x16x64x64`, `262144` bytes
- sampler bridge flags: `init_bridge_download=false`,
  `output_bridge_upload=false`
- `sampler_math_residency=gpu_backend_tensor`
- output:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\qwen-image-speed\qwen-image-t2i-512-1step.png`

Validated Qwen-Image Edit 2511 strict single-reference smoke:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\qwen-image-edit-2511-Q3_K_S.gguf`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\qwen_image_vae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`
- Resolution: `512x512`
- Steps: `2`
- CFG: `2.5`
- Sampler: `Euler`
- `qwen_image_zero_cond_t=true`
- `qwen_image_reference=true`
- `qwen_image_edit=true`
- positive and negative conditioning handles: `device_resident=true`
- CFG mode: `separate`, `unet_calls=4`, `backend_graph_calls=4`
- reference latents: `1`, `ref_latents_backend=true`,
  `ref_latents_per_step_upload=false`
- sampler residency: `gpu_backend_tensor`
- bridge flags: `init_bridge_download=false`, `output_bridge_upload=false`
- text encoder released before diffusion: `true`
- sampler timing: `latent_prepare_ms=6498`, `denoise_ms=2897`,
  `total_ms=9459`
- VAE decode: direct GPU Wan/Qwen path with `host_copies=0`,
  `direct_conv=true`, `used_im2col=false`, planned workspace about
  `676.2 MiB`, graph `96 ms`
- output:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux-qwen-ref-edit\qwen-image-edit\qwen_edit.png`
- logs:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\flux-qwen-ref-edit\qwen-image-edit\qwen_edit.stdout.log`

The local Qwen2.5-VL GGUF did not have a separate `--llm-vision` companion file
available in the model folders. The smoke still proves the fork-side strict
reference/edit handoff contract: resident conditioning handles, one-time
reference-latent upload, strict GPU sampler residency, GPU latent output, direct
GPU VAE decode, and explicit caller-owned image download. Use a full-quality
Qwen Image Edit asset pair and enough steps for visual acceptance; the local
two-step Q3 smoke is not a text-edit quality benchmark.

Validated Anima strict sampler smoke:

- Diffusion model:
  `F:\automatic1111\Stability\Models\DiffusionModels\anima-base-v1.0.safetensors`
- VAE:
  `F:\automatic1111\Stability\Models\VAE\qwen_image_vae.safetensors`
- LLM:
  `F:\automatic1111\Stability\Models\TextEncoders\qwen_3_06b_base.safetensors`
- Resolution: `512x512`
- Steps: `2` sampler-only strict smoke, `30` decoded bridge smoke
- CFG: `4.5`
- Samplers validated: `euler`, `euler_a`, `er_sde`, `dpmpp_2m_sde_gpu`

Observed handoff:

- conditioning handle is device-resident, including Anima `t5_ids` and
  `t5_weights`
- `conditioning_per_step_upload=false`
- `sampler_math_residency=gpu_backend_tensor`
- sampled latent is CUDA `1x16x64x64`, `262144` bytes
- `init_bridge_download=false`
- `output_bridge_upload=false`
- true GPU VAE/image output is not claimed
- non-strict VAE bridge is available and marks decoded image handles with
  CPU bridge provenance flags
- smoke log:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\anima-up-to-par\anima-strict-sampler.stdout.log`

The sampler path is strict GPU-resident. The VAE image path is currently a
recoverable compatibility bridge, not parity with the SDXL/Flux2 strict
GPU-resident VAE lane.

Anima VAE bridge correctness was checked against ComfyUI using the exact same
fork-exported `1x16x64x64` sampled latent. The fork bridge matches Comfy only
when the Comfy decode receives `Wan21.process_out(latent)` before
`vae.decode(...)`; raw latent decode does not match. This confirms the fork's
Wan21 latent mean/std transform is aligned with Comfy and that the bridge's
current limitation is residency/performance rather than VAE math.

- Comfy source behavior:
  - `comfy.supported_models.Anima.latent_format = latent_formats.Wan21`
  - `Wan21.process_out(latent) = latent * std + mean`
  - `nodes.VAEDecode.decode()` calls `vae.decode(latent)` after sampler/model
    code has produced VAE-space samples.
- Fork source behavior:
  - `AutoEncoderKL::uses_wan21_latent_format()` covers Qwen-Image and Anima.
  - `decode_first_stage(...)` applies `diffusion_to_vae_latents(...)` before
    VAE decode.
- Diagnostic harness:
  `scripts/anima_vae_bridge_parity.py`
- Local comparison artifacts:
  `F:\Paralol\local\stable-diffusion.cpp-speed\build\anima-up-to-par\vae-parity-bridge\comfy_compare`
- Same-latent results:
  - raw Comfy VAE decode: mean abs `17.9492`, p99 `116`, PSNR `18.20 dB`
  - Comfy `Wan21.process_out` decode: mean abs `0.2078`, p99 `2`,
    PSNR `53.77 dB`

Anima/Qwen VAE decode can be promoted only when the experimental Wan/Qwen GPU
VAE flag is set. Without that flag the capability split remains intentional:
`supports_gpu_latent_decode=false` and `supports_gpu_latent_decode_bridge=true`.
With `SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1`, the fork advertises true GPU
decode for the validated Anima/Qwen single-frame path and strict mode refuses
the old bridge.
