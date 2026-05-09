# VAE Implicit-GEMM Conv Stability

Branch: `paralol/comfy-normal-vae-cudnn`

Baseline commit: `cc1d2b955c13c017add26902215e75a1db5453b7`

Model: `F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors`

Upstream source: `bssrdf/llama.cpp` branch `conv2d-implicit`, commit `f1bb125740ee380b7899e035e51fbf5cbc1b13db`, originally discussed in `ggml-org/llama.cpp` PR `#15805` and stable-diffusion.cpp discussion `#1104`.

## Summary

The implicit-GEMM conv2d path is correct on deterministic nonzero synthetic VAE
conv shapes and now passes full COMFY_NORMAL SDXL 1024 encode/decode when Latent
Decode uses a VAE decode-only context.

The earlier full-decode hard stop was not caused by implicit-GEMM. It also
reproduced with implicit disabled when encode and decode shared the full
CLIP/UNet/VAE context. The verified production path is to keep normal VAE decode
in a VAE decode-only context, which also avoids keeping CLIP/UNet resident during
decode.

COMFY_NORMAL now scopes implicit-GEMM around VAE encode/decode/estimate by
default. Use `SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` to force the older direct
conv path for diagnostics.

## Files Changed

- `ggml/src/ggml-cuda/conv2d-implicit.cu`
- `ggml/src/ggml-cuda/conv2d-implicit.cuh`
- `ggml/src/ggml-cuda/cp-async.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `examples/vae-op-bench/main.cpp`
- `docs/vae-implicit-gemm-conv-port.md`
- `docs/vae-implicit-gemm-conv-stability.md`

The low-level CUDA dispatch remains controlled internally by:

```text
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV=1
```

Normal callers should not set it directly. `sd_encode_image_normal`,
`sd_decode_latent_normal`, and `sd_estimate_vae_normal_memory` scope it
automatically for `SD_VAE_EXEC_COMFY_NORMAL` unless
`SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` is set.

Extra per-conv sync/debug logging is behind:

```text
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_DEBUG=1
```

## Correctness

Command:

```powershell
F:\Paralol\build\sdcpp-latent\bin\sd-vae-op-bench.exe `
  --correctness `
  --out-json C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\implicit_conv_correctness.json `
  --out-md C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\implicit_conv_correctness.md `
  --warmup 0 `
  --iterations 1 `
  --memory-cap-mb 8192
```

All tested shapes used deterministic nonzero input and weight patterns, including positive and negative values.

| Shape | Direct | Implicit | Mean abs | p99 abs | Max abs | NaN/Inf | Buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128x128x4 -> 512, 3x3 | 4.219 ms | 2.176 ms | 0 | 0 | 0 | 0 | 32.3 MB |
| 128x128x512 -> 512, 1x1 | 22.260 ms | 0.550 ms | 0 | 0 | 0 | 0 | 66.0 MB |
| 256x256x512 -> 512, 3x3 | 307.930 ms | 15.508 ms | 0 | 0 | 0 | 0 | 265.0 MB |
| 512x512x512 -> 512, 3x3 | 1254.777 ms | 61.734 ms | 0 | 0 | 0 | 0 | 1033.0 MB |
| 512x512x512 -> 256, 3x3 | 628.165 ms | 31.666 ms | 0 | 0 | 0 | 0 | 772.5 MB |
| 1024x1024x256 -> 256, 3x3 | 1265.975 ms | 67.712 ms | 0 | 0 | 0 | 0 | 2050.3 MB |
| 1024x1024x256 -> 128, 3x3 | 634.823 ms | 36.902 ms | 0 | 0 | 0 | 0 | 1537.1 MB |
| 1024x1024x128 -> 128, 3x3 | 320.787 ms | 18.143 ms | 0 | 0 | 0 | 0 | 1024.6 MB |

Output artifacts:

- `build/vae-speed/implicit_conv_correctness.json`
- `build/vae-speed/implicit_conv_correctness.md`

No shape needed selective direct fallback based on op-level correctness.

## Full Decode Verification

Default implicit-GEMM, split VAE decode context:

```powershell
F:\Paralol\build\sdcpp-latent\bin\sd-latent-smoke.exe `
  --model F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors `
  --image F:\Paralol\examples\orc.png `
  --output C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\implicit_default_split.png `
  --image-channels 3 `
  --type-f16 `
  --split-decode-context
```

Result:

- exit code: 0
- encode: `sd_encode_image_normal completed, taking 0.74s`
- decode: `sd_decode_latent_normal completed, taking 0.81s`
- encode workspace: 1536 MB
- decode workspace: 2816 MB
- `used_im2col=false`
- `used_tiling=false`
- `used_taesd=false`
- `host_copies=0`
- `device_copies=5`

Direct conv escape hatch:

```text
SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1
```

Result:

- encode: 5.66s
- decode: 11.92s
- same 1536 MB encode workspace
- same 2816 MB decode workspace
- no IM2COL, tiled VAE, or TAESD

## Previous Same-Context Debug

Command shape:

```powershell
SDCPP_VAE_STRICT_COMFY_NORMAL=1
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV=1
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_DEBUG=1
SDCPP_TRACE_VAE_TIMING=1
SDCPP_TRACE_GRAPH_ALLOC=1
CUDA_LAUNCH_BLOCKING=1
sd-latent-smoke.exe --model <sdxl-model> --image F:\Paralol\examples\orc.png --image-channels 3 --type-f16
```

Result:

- encode completed
- `sd_encode_image_normal completed, taking 0.76s`
- first decode graph allocation started
- every implicit conv node in the first decode graph printed `before` and `after`
- the last conv in the first decode graph also completed:
  - `node_113`
  - input `[128,128,8,1]`
  - weight `[1,1,8,8]`
  - output `[128,128,8,1]`
- process stopped before a decode stage timing line or `decode_report`
- stderr had no CUDA error after the last conv

Logs:

- `build/vae-speed/implicit_decode_debug.stdout.log`
- `build/vae-speed/implicit_decode_debug.stderr.log`

## Direct Fallback Check

The same `sd-latent-smoke` decode stop also reproduced with `SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV` unset. That means the current hard stop is not sufficient evidence against implicit-GEMM.

Direct fallback log:

- `build/vae-speed/direct_min_decode_smoke.stdout.log`
- `build/vae-speed/direct_min_decode_smoke.stderr.log`

Sampled latent path with implicit-GEMM also stopped at first decode graph allocation, after sampling completed:

- `sd_sample_latent completed in 2.19s`
- first decode compute buffer: `1184.25 MB`
- no decode report

Logs:

- `build/vae-speed/implicit_sample_decode_smoke.stdout.log`
- `build/vae-speed/implicit_sample_decode_smoke.stderr.log`

## Memory

The implicit path did not increase op-level planned buffers. It used the same buffer sizes as direct conv in the tested shapes and avoided IM2COL-sized allocations.

Important examples:

- 1024x1024x256 3x3: direct `2050.3 MB`, implicit `2050.3 MB`
- 1024x1024x128 3x3: direct `1024.6 MB`, implicit `1024.6 MB`
- legacy/im2col 1024x1024x128 3x3: `6144.6 MB`

## Recommendation

Recommendation: **B. keep implicit-GEMM and continue**, but do not make it the default yet.

The op-level result is strong:

- deterministic nonzero correctness is exact versus direct for the tested f32 path
- no NaN/Inf
- large speedup on every significant VAE conv shape
- no IM2COL memory blowup

The next practical task should not be a cuDNN backend. It should be to isolate the decode-stage hard stop that reproduces even when implicit-GEMM is disabled. The most likely next target is the first decode graph outside conv2d, especially the attention `MUL_MAT` / `SOFT_MAX` region or the current `sd-latent-smoke` latent/decode handoff.

Once that independent full-decode issue is resolved, implicit-GEMM should be retested as the default CUDA `CONV_2D` backend for COMFY_NORMAL.
