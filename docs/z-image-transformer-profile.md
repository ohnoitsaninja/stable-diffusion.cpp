# Z-Image Transformer Profile: Juggernaut Z Q6_K

Date: 2026-05-25

This pass profiles the existing local Juggernaut Z Q6_K model only. No alternate quant files were downloaded or required.

## Scope

- Model: `F:\automatic1111\Stability\Models\DiffusionModels\Juggernaut_Z_V1_by_RunDiffusion_q6_k-004.gguf`
- LLM: `F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
- VAE: `F:\automatic1111\Stability\Models\VAE\ae.safetensors`
- Resolution: `960x1440`
- Sampler: `res_2s`
- Scheduler: `beta`
- CFG: `3.5`
- Steps: `4`
- Profile flag: `SDCPP_PROFILE_Z_IMAGE_TRANSFORMER=1`

Output logs:

- `F:\Paralol\build\diagnostics\z-juggernaut-q6-transformer-profile\z_q6_profile_4step.stdout.log`
- `F:\Paralol\build\diagnostics\z-juggernaut-q6-transformer-profile\z_q6_profile_4step.stderr.log`
- `F:\Paralol\build\diagnostics\z-juggernaut-q6-transformer-profile\z_q6_profile_4step.png`

## Exact Command

The probe was launched with a hard 120 second timeout and separate stdout/stderr capture:

```powershell
$env:SDCPP_EXPERIMENTAL_Z_IMAGE_BACKEND='1'
$env:SDCPP_STRICT_GPU_RESIDENT='1'
$env:SDCPP_Z_IMAGE_TEXT_ENCODER_CPU_PARAMS='1'
$env:SDCPP_PROFILE_Z_IMAGE_TRANSFORMER='1'
$env:SDCPP_PROFILE_RUNNER_TIMINGS='1'

F:\Paralol\local\stable-diffusion.cpp-speed\build\codex\bin\sd-latent-smoke.exe `
  --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\Juggernaut_Z_V1_by_RunDiffusion_q6_k-004.gguf `
  --vae F:\automatic1111\Stability\Models\VAE\ae.safetensors `
  --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf `
  --image F:\Paralol\build\generated\sd_t2i_comfy_normal_test.bmp `
  --output F:\Paralol\build\diagnostics\z-juggernaut-q6-transformer-profile\z_q6_profile_4step.png `
  --prompt "A surreal double exposure portrait of a woman's profile, coastal ecosystem, cinematic sunset, turquoise waves, sharp details" `
  --negative-prompt "3D, ai generated, semi realistic, illustrated, drawing, comic, digital painting, 3D model, blender, video game screenshot, screenshot, render, high-fidelity, smooth textures, CGI, masterpiece, text, writing, subtitle, watermark, logo, blurry, low quality, jpeg, artifacts, grainy" `
  --width 960 --height 1440 --steps 4 --seed 42 --cfg-scale 3.5 `
  --sampling-method res_2s --scheduler beta `
  --sample-without-init --gpu-flow-sampler --gpu-sampler-backend `
  --gpu-decode-output --gpu-latent-decode-input --download-gpu-output-buffer `
  --condition-handles --release-text-encoder-after-conditioning --strict-gpu-resident
```

## Timing Summary

The sampler remained on the resident GPU path:

- `condition_input=handle`
- `conditioning_storage=device_tensor`
- `conditioning_per_step_upload=false`
- `sampler_math_residency=gpu_backend_tensor`
- `initial_noise_bridge_upload=false`
- `output_bridge_upload=false`

4-step result:

| Metric | Value |
|---|---:|
| `cfg_eval_mode` | `separate` |
| Z transformer invocations | 14 |
| backend graph calls | 21 |
| denoise time | 19,452 ms |
| total sampler call | 19,517 ms |

Per Z transformer invocation, from `[ZProfileRunner]`:

| Metric | Average | Min | Max |
|---|---:|---:|---:|
| Graph build | 24.68 ms | 22.96 ms | 31.29 ms |
| Graph alloc | ~0.30 ms | ~0.21 ms | ~0.39 ms |
| Input copy/sync | 0.90 ms | 0.78 ms | 1.10 ms |
| Backend compute | 1325.15 ms | 1303.99 ms | 1411.75 ms |
| Output copy | 1.12 ms | 0.82 ms | 1.46 ms |
| Cleanup | ~7 ms | ~6.5 ms | ~8.4 ms |

Conclusion: graph build, graph allocation, and tensor input/output copies are not the bottleneck. The runtime is dominated by backend graph compute inside the Z transformer.

## Per-Block Profile

`SDCPP_PROFILE_Z_IMAGE_TRANSFORMER=1` now emits `[ZProfileBlock]` lines during graph construction:

- embed
- context refiner blocks
- noise refiner blocks
- concat
- 30 main transformer layers
- final layer

These are graph-construction timings, not CUDA kernel timings. They are all tiny, mostly below `0.06 ms` per block. That means per-block graph construction is not the cause of the 18-19 second denoise time.

The key tensor shapes for the tested resolution:

| Stage | Positive context | Negative context |
|---|---:|---:|
| image latent tokens | 5400 | 5400 |
| context tokens before padding | 30 | 73 |
| context tokens after Z pad | 32 | 96 |
| transformer sequence length | 5440 | 5504 |
| hidden width | 3840 | 3840 |

The model has 30 main Z transformer layers plus 2 context refiner and 2 noise refiner blocks.

## Op/Category Profile

The ggml CUDA backend path used here does not expose reliable per-kernel CUDA timing per graph node. The profile therefore reports graph op/category counts and tensor byte volume, which is useful for locating likely pressure points but is not a substitute for kernel-level CUDA profiling.

Representative negative-prompt graph categories:

| Category | Nodes | Tensor bytes |
|---|---:|---:|
| layout/copy | 1744 | 99,851 MB |
| pointwise/norm | 1323 | 96,767 MB |
| matmul/projection/MLP | 208 | 26,743 MB |
| other | 275 | 19,396 MB |
| attention | 34 | 2,580 MB |

Top ops by tensor byte volume:

| Rank | Op | Nodes | Tensor bytes |
|---:|---|---:|---:|
| 1 | `MUL` | 470 | 36,190 MB |
| 2 | `VIEW` | 542 | 31,314 MB |
| 3 | `RESHAPE` | 445 | 28,389 MB |
| 4 | `MUL_MAT` | 208 | 26,743 MB |
| 5 | `SCALE` | 308 | 22,496 MB |
| 6 | `PERMUTE` | 242 | 18,408 MB |
| 7 | `ADD` | 239 | 15,638 MB |
| 8 | `RMS_NORM` | 205 | 15,481 MB |
| 9 | `REPEAT` | 206 | 14,101 MB |
| 10 | `CONT` | 376 | 13,262 MB |
| 11 | `GLU` | 34 | 6,880 MB |
| 12 | `PAD` | 68 | 5,295 MB |
| 13 | `CPY` | 102 | 4,538 MB |
| 14 | `CONCAT` | 37 | 3,941 MB |
| 15 | `FLASH_ATTN_EXT` | 34 | 2,580 MB |

Largest individual graph nodes were `MUL_MAT` and immediate `RESHAPE` nodes around:

- `MUL_MAT` f32 output shape `[11520,5504,1,1]`, about `241.88 MB`
- reshaped f32 output shape `[128,90,5504,1]`, about `241.88 MB`

This shape appears repeatedly across main transformer layers.

## Bottleneck Attribution

Most likely cause:

1. Large Z transformer matmul/projection/MLP work on long sequences:
   - ~5400 image tokens at `960x1440`.
   - 30 main transformer layers.
   - repeated `MUL_MAT` outputs around 240 MB.
   - model weights are quantized Q6_K, but activations and many intermediates are f32.

2. Heavy layout/reshape/pointwise churn around the transformer:
   - `VIEW`, `RESHAPE`, `PERMUTE`, `CONT`, `CPY`, `REPEAT`, `PAD`, `MUL`, `SCALE`, `ADD`, and `RMS_NORM` dominate tensor byte volume.
   - These are not necessarily all real allocations, but they indicate a lot of graph traffic and shape manipulation around attention/MLP blocks.

Less likely causes:

- Graph build: ~25 ms per Z model call, not the bottleneck.
- Input/output transfer: ~1 ms each per Z model call, not the bottleneck.
- Positional embedding generation/RoPE setup: about 20-29 ms per graph, not dominant.
- Sampler overhead outside transformer calls: denoise time is almost entirely explained by 14 backend model calls at ~1.3s each plus post-graph sampler ops.
- Attention alone: only 34 `FLASH_ATTN_EXT` nodes and comparatively low output byte volume; attention may still cost meaningful GPU time, but the graph evidence points more strongly at matmul/projection/MLP and layout traffic.

## Why CFG Batching Did Not Help

The CFG batching experiment reduced model calls from 14 to 7, but each batch-2 Z graph became roughly as expensive as two batch-1 graphs and added extra graph work:

- Default separate CFG, no decode: `unet_calls=14`, `backend_graph_calls=21`, `denoise_ms=17896`
- Explicit Z batched CFG, no decode: `unet_calls=7`, `backend_graph_calls=28`, `denoise_ms=18330`

So the current ggml Z backend does not get useful throughput scaling from batch-2 transformer evaluation at this resolution/model quant. It should stay opt-in until lower-level kernels/layout are improved.

## Recommended Next Work

1. Profile one Z graph with an external CUDA profiler if practical.
   The fork now has enough log markers to correlate a single backend graph call, but per-kernel timings require Nsight or CUDA event instrumentation below ggml graph nodes.

2. Investigate the `MUL_MAT` path for Q6_K Z weights at shapes around:
   - output `[11520,5504,1,1]`
   - reshaped `[128,90,5504,1]`
   The current evidence points to quantized matmul/dequant throughput and f32 activation traffic.

3. Reduce layout churn around Z transformer blocks:
   - `CONT`, `CPY`, `PERMUTE`, `RESHAPE`, `VIEW`, `REPEAT`, and `PAD` dominate graph traffic.
   - Look for avoidable `ggml_cont`, repeated pad-token expansion, repeated reshape/permute pairs, and opportunities to keep attention/MLP in a stable layout.

4. Keep CFG batching experimental.
   It is functionally working for Z with padded positive/negative contexts, but it is not a speed win on the Q6_K backend path tested here.

## Build Artifact

Build DLL:

- `F:\Paralol\local\stable-diffusion.cpp-speed\build\codex\bin\stable-diffusion.dll`
- SHA256: `56AEE1FD9594B19BEB2ACA97449A5FD3B71A25AD79C5083171DE6FDD00EF6787`

The staged Paralol DLL was not updated by this profiling-only pass.
