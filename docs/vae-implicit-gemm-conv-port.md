# VAE Implicit-GEMM Conv2D Port

Branch: `paralol/comfy-normal-vae-cudnn`

Baseline commit: `cc1d2b955c13c017add26902215e75a1db5453b7`

Model: `F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors`

## Summary

The upstream CUDA implicit-GEMM conv2d work is a much better fit than a custom
cuDNN VAE decoder. The local port builds, deterministic VAE conv correctness
checks pass, and full COMFY_NORMAL SDXL 1024 encode/decode passes when Latent
Decode uses a VAE decode-only context.

COMFY_NORMAL now scopes implicit-GEMM around VAE encode/decode/estimate by
default. The direct conv path remains available with
`SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1`.

## Source

Upstream lead:

- `ggml-org/llama.cpp` PR `#15805`
- branch `bssrdf:conv2d-implicit`
- fetched commit: `f1bb125740ee380b7899e035e51fbf5cbc1b13db`
- stable-diffusion.cpp discussion: `#1104`

Imported files:

- `ggml/src/ggml-cuda/conv2d-implicit.cu`
- `ggml/src/ggml-cuda/conv2d-implicit.cuh`

Local compatibility patch:

- `ggml/src/ggml-cuda/cp-async.cuh`: made `ggml_cuda_cvta_generic_to_shared` accept `const void *`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: includes `conv2d-implicit.cuh` and selects `ggml_cuda_op_conv2d_implicit` when `SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV` is set.
- `src/stable-diffusion.cpp`: scopes `SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV=1` for COMFY_NORMAL VAE calls unless `SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` is set.
- `examples/vae-op-bench/main.cpp`: adds direct vs implicit benchmark cases.

## Compatibility Notes

The upstream work is best treated as a CUDA backend realization of `GGML_OP_CONV_2D`, not as a new long-term graph op. In this prototype it is selected inside CUDA dispatch behind:

```text
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV=1
```

Tensor layout remains the current `ggml_conv_2d_direct` layout at the graph boundary:

- input: `[W, H, C, N]`
- weight: `[KW, KH, IC, OC]`
- output: `[OW, OH, OC, N]`

Demonstrated dtype path in this fork is the current VAE path:

- activation input: `f32`
- VAE weights in this checkpoint: `f32`
- output: `f32`

The upstream branch also has an optimized f16-weight tensor-core path. BF16 was not validated here.

## Build

Configure was rerun so CMake would pick up the new CUDA source:

```powershell
cmake -S C:\tmp\stable-diffusion.cpp-paralol `
  -B F:\Paralol\build\sdcpp-latent `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSD_CUDA=ON `
  -DSD_BUILD_SHARED_LIBS=ON `
  -DSD_BUILD_EXAMPLES=ON `
  -DGGML_NATIVE=ON
```

Build target:

```powershell
cmake --build F:\Paralol\build\sdcpp-latent --config Release --target sd-vae-op-bench --parallel
```

Result: build passed.

## Op Benchmark

Command:

```powershell
F:\Paralol\build\sdcpp-latent\bin\sd-vae-op-bench.exe `
  --out-json C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\sdcpp_implicit_gemm_op_bench.json `
  --out-md C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\sdcpp_implicit_gemm_op_bench.md `
  --warmup 5 `
  --iterations 20 `
  --memory-cap-mb 8192 `
  --include-legacy
```

| Shape | Direct median | Implicit median | Buffer direct/implicit | PyTorch/cuDNN reference |
| --- | ---: | ---: | ---: | ---: |
| 512x512x512, 3x3 | 1272.647 ms | 65.344 ms | 1033.002 MB | not measured separately |
| 1024x1024x256, 3x3 | 1262.461 ms | 64.737 ms | 2050.251 MB | 15.100 ms BF16 channels-last |
| 1024x1024x128, 3x3 | 316.984 ms | 17.434 ms | 1024.563 MB | 4.482 ms BF16 channels-last |

Legacy/im2col one-shape comparison:

| Shape | Legacy/im2col median | Buffer |
| --- | ---: | ---: |
| 1024x1024x128, 3x3 | 29.126 ms | 6144.563 MB |

The implicit-GEMM path is slower than legacy/im2col for that one 128-channel shape, but avoids the unacceptable IM2COL-sized allocation. It is much faster than direct conv and uses the same planned buffer size as direct conv.

Additional short decode-shape coverage:

```powershell
F:\Paralol\build\sdcpp-latent\bin\sd-vae-op-bench.exe `
  --out-json C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\sdcpp_implicit_gemm_op_bench_1x1.json `
  --out-md C:\tmp\stable-diffusion.cpp-paralol\build\vae-speed\sdcpp_implicit_gemm_op_bench_1x1.md `
  --warmup 1 `
  --iterations 3 `
  --memory-cap-mb 8192
```

| Shape | Direct median | Implicit median | Buffer direct/implicit |
| --- | ---: | ---: | ---: |
| 128x128x4 -> 512, 3x3 | 1.023 ms | 0.093 ms | 32.322 MB |
| 128x128x512 -> 512, 1x1 | 19.638 ms | 0.509 ms | 66.010 MB |
| 512x512x512, 3x3 | 1198.329 ms | 57.703 ms | 1033.002 MB |
| 1024x1024x256, 3x3 | 1142.688 ms | 59.558 ms | 2050.251 MB |
| 1024x1024x128, 3x3 | 302.248 ms | 16.202 ms | 1024.563 MB |

## Full COMFY_NORMAL Smoke

Command:

```powershell
SDCPP_VAE_STRICT_COMFY_NORMAL=1
SDCPP_EXPERIMENTAL_VAE_IMPLICIT_GEMM_CONV=1
SDCPP_TRACE_VAE_TIMING=1
SDCPP_TRACE_GRAPH_ALLOC=1
sd-latent-smoke.exe --model <sdxl-model> --image F:\Paralol\examples\orc.png --image-channels 3 --type-f16
```

Result:

- encode completed
- `sd_encode_image_normal completed, taking 0.75s`
- encode report preserved COMFY_NORMAL invariants:
  - `im2col=false`
  - `used_tiling=false`
  - `used_taesd=false`
  - `host_copies=0`
  - `device_copies=5`
  - `planned_mb=1536`
- process hard-stopped during the first decode graph before `decode_report`
- stderr was empty

Captured logs:

- `build/vae-speed/implicit_decode_smoke.stdout.log`
- `build/vae-speed/implicit_decode_smoke.stderr.log`

Later stability and scale-fusion work resolved the full-decode issue. Current
default COMFY_NORMAL SDXL 1024 decode uses implicit-GEMM with fused SDXL VAE
conv scale, runs as a merged graph, and reports:

- decode: `sd_decode_gpu_latent_normal_gpu completed, taking 0.41s`
- workspace: 2816 MB
- graphs/stages: 1 / 1
- `used_im2col=false`
- `used_tiling=false`
- `used_taesd=false`
- `host_copies=0`
- `device_copies=0`

## Decision

Implicit-GEMM is worth continuing and should be preferred before returning to cuDNN/CUTLASS. It gets close enough to PyTorch/cuDNN at the op level to justify further work:

- 1024x1024x256: about 4.3x slower than PyTorch BF16 channels-last, about 1.9x slower than PyTorch F32/TF32 contiguous.
- 1024x1024x128: about 3.9x slower than PyTorch BF16 channels-last.
- Memory stays at the direct-conv footprint, not the IM2COL footprint.

Implicit-GEMM is now the production COMFY_NORMAL CUDA VAE conv backend, with
`SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` as the direct-conv escape hatch.

cuDNN is not needed yet. CUTLASS remains a backup if the upstream implicit-GEMM path cannot be made graph-stable.
