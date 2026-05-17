# Paralol Model Capability Matrix

This note records the fork-side capabilities Paralol should use when deciding
which stable-diffusion.cpp path to call. It is intentionally conservative: a
capability is marked supported only when the API shape is implemented and at
least one local smoke or existing artifact supports the claim.

## Common rules

- KSampler GPU latent output is still a non-strict bridge. The sampler loop
  materializes the final latent on the host and uploads it into an owned CUDA
  `SD_GPU_RESOURCE_LATENT` handle.
- `sd_get_gpu_capabilities(...)` reports this split explicitly:
  `supports_sampler_gpu_latent_output=false` and
  `supports_sampler_gpu_latent_bridge_output=true`.
- `SDCPP_STRICT_GPU_RESIDENT=1` must refuse sampler GPU output and sampler
  GPU init-latent bridge APIs until the sampler loop itself becomes
  GPU-resident.
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
| Anima | 16 ch | 8 | Compatibility bridge | Compatibility bridge | No | Not claimed | Wan/Qwen VAE path is safe but not COMFY_NORMAL: high IM2COL memory and host bridge copies are reported honestly. |
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
- Anima T2I and VAE Encode/Decode compatibility bridges with strict-mode
  refusal for the bridge-only paths.

Pending or deliberately unclaimed:

- Z reference/edit conditioning.
- TAE previews for SD1, Z-Image/Z-Anime, and Anima.
- True all-GPU sampler internals. See
  `docs/paralol-true-gpu-sampler-plan.md`.
