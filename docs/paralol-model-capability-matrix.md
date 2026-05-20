# Paralol Model Capability Matrix

This note records the fork-side capabilities Paralol should use when deciding
which stable-diffusion.cpp path to call. It is intentionally conservative: a
capability is marked supported only when the API shape is implemented and at
least one local smoke or existing artifact supports the claim.

## Common rules

- KSampler GPU latent output is family-specific. SD1/SDXL and the env-gated
  Flux2/Z text-flow lanes can return true backend-resident CUDA latent handles.
  Unsupported families still use the compatibility bridge that materializes the
  final latent on the host and uploads it into an owned CUDA
  `SD_GPU_RESOURCE_LATENT` handle.
- `sd_get_gpu_capabilities(...)` reports this split explicitly with
  `supports_sampler_gpu_latent_output` for true paths and
  `supports_sampler_gpu_latent_bridge_output` for bridge paths.
- `SDCPP_STRICT_GPU_RESIDENT=1` must refuse bridge paths and allow only the
  true backend-resident sampler lanes.
- GPU VAE Decode should prefer `sd_decode_gpu_latent_normal_gpu(...)` whenever
  `supports_gpu_latent_decode=true`.
- I2I KSampler init-latent handoff should use
  `sd_sample_latent_gpu_with_init_gpu(...)` only in non-strict mode. This API
  bridge-downloads the init latent internally, samples through the existing
  sampler, then returns an uploaded CUDA latent handle.
- `sd_sample_latent_gpu_with_init_gpu(...)` validates latent shape before the
  bridge download and marks the sampled output handle with
  `CPU_BRIDGE_DOWNLOAD | CPU_BRIDGE_UPLOAD` provenance flags.
- CPU image downloads are explicit. Paralol should prefer
  `sd_gpu_image_download_to_buffer(...)` into caller-owned RGBA8 memory.

## Family status

| Family | Latent | VAE scale | T2I sampled latent -> GPU decode | VAE encode -> KSampler init bridge | Reference/edit conditioning | TAE preview | Notes |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| SD1 base | 4 ch | 8 | Supported | Supported | No | Not claimed | Narrowly enabled only for `VERSION_SD1`, not SD1 inpaint/pix2pix/tiny variants. |
| SD2 | 4 ch | 8 | Not targeted | Not targeted | No | Not claimed | Intentionally out of scope for Paralol's current model set. |
| SDXL | 4 ch | 8 | Supported | Supported | No | TAESDXL verified | COMFY_NORMAL + implicit-GEMM VAE path is the production target. |
| Flux.1 | 16 ch | model-reported, normally 8 | Supported | Supported | No | TAEF1 verified | Uses CLIP-L + T5XXL. |
| Flux2 / Klein | 128 ch | 16 | Supported | Use reference images for edit mode, not SDXL-style init latent unless explicitly testing I2I | Yes, via `ref_images` | TAEF2 verified | Flux2 edit/reference conditioning is a different path than SDXL I2I init-latent sampling. |
| Z-Image / Z-Anime | 16 ch | 8 | Supported | Supported for Z-Image smoke path | Not claimed | Not claimed | Z T2I and init-latent bridge smokes pass. Reference/edit capability is intentionally not advertised yet. |
| Qwen-Image | 16 ch | 8 | Supported for text-only `cfg=1` strict sampler; VAE decode is not claimed | Compatibility bridge | Qwen edit path, not strict-resident | TAEHV compatible, not claimed here | `SDCPP_EXPERIMENTAL_QWEN_IMAGE_BACKEND=1` enables the strict sampler lane. The local Qwen VAE bridge produced a blurry image, so GPU image decode capability is deliberately false until that path is fixed. |
| Anima | 16 ch | 8 | Supported for text-only strict sampler, no GPU VAE decode claim | Compatibility bridge | No | Not claimed | `SDCPP_EXPERIMENTAL_ANIMA_BACKEND=1` enables the strict sampler lane. Validated sampler methods: `euler`, `euler_a`, `er_sde`, `dpmpp_2m_sde_gpu`. Qwen-image VAE GPU decode remains unclaimed. |
| Marigold IID | 8 ch | model-specific | Not supported | Not supported | N/A | N/A | Uses the dedicated intrinsic-image decomposition API. |

## Verification snapshot

Known verified paths at the time this matrix was written:

- SDXL T2I GPU sampled latent -> GPU VAE Decode -> caller-owned image download.
- SDXL I2I VAE Encode GPU latent -> non-strict KSampler init bridge -> isolated
  GPU VAE Decode.
- Z-Image T2I GPU sampled latent -> GPU VAE Decode at 512 and 1024 resolutions.
- Flux2/Klein T2I GPU sampled latent -> GPU VAE Decode at 1024.
- Flux2/Klein edit/reference conditioning through `ref_images`.
- Flux.1 T2I GPU sampled latent -> GPU VAE Decode at 512.
- Flux.1 VAE Encode GPU latent -> non-strict KSampler init bridge -> isolated
  GPU VAE Decode at 512.
- Z-Image VAE Encode GPU latent -> non-strict KSampler init bridge -> isolated
  GPU VAE Decode at 512.
- Anima text-only T2I strict sampler lane: resident Qwen/T5 conditioning handle
  -> backend flow sampler -> CUDA `1x16x64x64` latent at 512px, with no sampler
  bridge flags. Validated methods: `euler`, `euler_a`, `er_sde`,
  `dpmpp_2m_sde_gpu`; CFG `4.5` validated for the Anima lane.
- Anima/Qwen-image Wan21 latent mean/std transforms are applied for
  diffusion-latent <-> VAE-latent conversion. The CPU compatibility decode path
  can write a diagnostic image, but GPU VAE image output is still not advertised
  for Anima/Qwen-image.
- Qwen-Image text-only `cfg=1` T2I strict sampler lane: resident Qwen
  conditioning handle -> backend flow sampler -> CUDA `1x16x64x64` latent, with
  no sampler bridge flags.

Pending or deliberately unclaimed:

- Z reference/edit conditioning.
- Qwen-Image and Anima VAE decode image-quality validation/fix for the
  Qwen-image VAE. The Qwen/X bridge is not accepted because it produced a
  blurry output in local testing.
- Qwen-Image CFG, reference/edit, and vision conditioning.
- Anima Comfy/reference-quality image acceptance for the Qwen-image VAE path.
  The current fork can decode diagnostics through the CPU compatibility path,
  but `sd_decode_gpu_latent_normal_gpu()` remains intentionally disabled for
  Anima/Qwen-image until the VAE output is verified.
- TAE previews for SD1, Z-Image/Z-Anime, and Anima.
- True all-GPU sampler internals. See
  `docs/paralol-true-gpu-sampler-plan.md`.
