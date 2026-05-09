# Paralol COMFY_NORMAL VAE

## Problem

The legacy stable-diffusion.cpp VAE graph lowered normal SDXL 1024 VAE decode
through GGML IM2COL tensors. A full-frame decode planned a 7680.25 MB CUDA
workspace, which made Paralol's `KSampler -> Latent Decode` path look like it
was loading too much model state even when the latent shape was correct.

Tiled VAE and TAESD are useful fallback or preview options, but they are not the
correct default for Paralol's normal ComfyUI-style latent graph. Paralol needs a
full-frame VAE encode/decode path that preserves normal VAE semantics.

## Solution

`SD_VAE_EXEC_COMFY_NORMAL` is the production path for normal full-frame VAE
encode/decode in Paralol.

The mode uses:

- full-frame normal AutoencoderKL encode/decode
- no TAESD
- no tiled VAE
- no legacy IM2COL
- direct convolution
- stage-scoped graphs
- CUDA device-resident stage boundaries
- ABI-safe public C APIs with memory reports and capability discovery

`SD_VAE_EXEC_AUTO` selects `COMFY_NORMAL` for CUDA SDXL VAE contexts. Other
models can still use the direct graph path unless explicitly requested.

## Public API

Use these APIs from Paralol's native SD worker:

```c
sd_vae_run_options_t options;
sd_vae_run_options_init(&options);
options.mode = SD_VAE_EXEC_AUTO;

sd_vae_memory_report_t report;
sd_vae_memory_report_init(&report);

sd_latent_t* latent = sd_encode_image_normal(ctx, image, &options, &report);
sd_image_t* image = sd_decode_latent_normal(ctx, latent, &options, &report);
```

Capability discovery:

```c
sd_vae_capabilities_t capabilities;
capabilities.struct_size = sizeof(capabilities);
sd_get_vae_capabilities(ctx, &capabilities);
```

All VAE option, report, and capability structs include `struct_size`,
`version`, and reserved fields for ABI-safe extension.

## Environment Flags

- `SDCPP_VAE_NORMAL_MODE=auto|comfy_normal|direct_graph|legacy`
- `SDCPP_VAE_DTYPE=auto|bf16|f16|f32`
- `SDCPP_VAE_STRICT_COMFY_NORMAL=1`
- `SDCPP_TRACE_VAE_STAGES=1`
- `SDCPP_TRACE_GRAPH_ALLOC=1`
- `SDCPP_DISABLE_COMFY_NORMAL_VAE=1`

Strict mode fails or loudly reports if COMFY_NORMAL enters tiled VAE, TAESD,
IM2COL, host stage copies, or a workspace regression beyond the staged baseline.

## Current Numbers

Measured on SDXL 1024 with the embedded checkpoint VAE:

| Path | Planned Workspace |
| --- | ---: |
| Legacy IM2COL decode | 7680.25 MB |
| Direct monolithic decode | 3840.25 MB |
| COMFY_NORMAL staged decode | 2816 MB |
| COMFY_NORMAL staged encode | 1536 MB |
| Comfy normal reference | 2371.94 MiB allocated / 3328 MiB reserved |

Parity against Comfy normal VAE:

- mean abs diff: 0.00295
- p99 abs diff: 0.0196
- PSNR: 45.17 dB

COMFY_NORMAL reports:

- `used_taesd=false`
- `used_tiling=false`
- `used_im2col=false`
- `used_direct_conv=true`
- `stage_boundary_host_copies=0`
- `stage_boundary_device_copies=5`

## Deferred Work

Compact bf16/f16 VAE activations are deferred. The current path keeps storage
at f32 because CUDA group norm, nearest upscale, pointwise graph storage, and
direct conv dtype plumbing are not yet compact-storage complete.

The feasibility pass found that group norm and direct conv are the high-risk
pieces. The likely realistic memory win is not enough to block Paralol
integration, so Paralol should productionize COMFY_NORMAL first and revisit
compact activations on a separate branch.

## Paralol Recommendation

`Latent Decode` should call `sd_decode_latent_normal` by default.

`Latent Encode` should call `sd_encode_image_normal` by default.

The older tiled/direct/legacy paths should remain available only as explicit
debug or low-memory fallback modes.
