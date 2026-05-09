# VAE BF16/F16 Activation Feasibility

## Summary

Recommendation: do not start the compact bf16/f16 VAE activation kernel project before productionizing the current COMFY_NORMAL path.

The current COMFY_NORMAL work already removed the bad SDXL normal VAE failure mode:

- legacy IM2COL decode workspace: 7680.25 MB
- direct monolithic decode workspace: 3840.25 MB
- staged COMFY_NORMAL decode workspace: 2816 MB
- staged COMFY_NORMAL encode workspace: 1536 MB
- stage boundaries are CUDA device-resident: host_copies=0
- IM2COL is absent in traced encode/decode graphs

The remaining memory is worth understanding, but not worth blocking Paralol integration. The likely realistic win from compact activation storage is about 256-768 MB on decode and 128-512 MB on encode unless we also replace the f32-only direct conv and group norm paths. The highest upside path, compact conv + group norm + pointwise + upscale, is a larger CUDA/ggml backend project with real image-quality risk.

No prototype was implemented in this pass. The low-risk candidates are not isolated enough to be meaningful: pointwise outputs are immediately consumed by f32-only conv/group norm/upscale paths, nearest upscale is currently blocked by both ggml core and CUDA f32 asserts, and compact stage-boundary storage would mostly add casts without changing per-stage workspace.

## Baseline Run

Repository: `C:\tmp\stable-diffusion.cpp-paralol`

Model: `F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors`

Image: `F:\Paralol\examples\orc.png`

Trace files:

- `C:\tmp\stable-diffusion.cpp-paralol\build\vae-bf16-feasibility\encode_trace.out.log`
- `C:\tmp\stable-diffusion.cpp-paralol\build\vae-bf16-feasibility\decode_trace.out.log`
- stderr files were empty

The earlier stalled wrapper created `encode_nvml.csv`, but the stdout/stderr logs were empty and VRAM stayed around idle usage for about 899 seconds. Treat that NVML CSV as not useful. The successful bounded direct trace runs did not include NVML sampling, so measured NVML peak is unavailable for this report.

## Current Baseline Numbers

| Path | Planned Workspace | Graphs | Stages | Device Resident | IM2COL | Direct Conv | Storage |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Encode | 1536 MB | 6 | 6 | yes | false | true | f32 |
| Decode | 2816 MB | 6 | 6 | yes | false | true | f32 |
| Comfy normal decode reference | 2371.94 MiB allocated / 3328 MiB reserved | n/a | n/a | yes | no | torch/cuDNN path | bf16 |

Comfy parity harness result:

- Comfy selected `torch.bfloat16`
- mean abs diff vs Comfy: 0.0029508
- p99 abs diff vs Comfy: 0.0196078
- PSNR: 45.17 dB

## Remaining F32 Pressure

Encode peak planned workspace is stage 1 at 1536 MB:

| Encode Stage | Planned MB | F32 Tensor Bytes MB | Dominant Ops |
| ---: | ---: | ---: | --- |
| 0 | 524 | 1036 | SCALE 524, CONV_2D 512 |
| 1 | 1536 | 10498 | SCALE 4737, CONV_2D 2176, GROUP_NORM 2048, ADD 1024, PAD 513 |
| 2 | 896 | 5634 | SCALE 2625, CONV_2D 1344, GROUP_NORM 896, ADD 512, PAD 257 |
| 3 | 448 | 2818 | SCALE 1313, CONV_2D 672, GROUP_NORM 448, ADD 256, PAD 129 |
| 4 | 96 | 576 | SCALE 256, CONV_2D 128, GROUP_NORM 128, ADD 64 |
| 5 | 1184 | 2306.5 | MUL_MAT 1056, SCALE 545.5, CONV_2D 257, GROUP_NORM 192, CONT 160 |

Decode peak planned workspace is stage 3 at 2816 MB:

| Decode Stage | Planned MB | F32 Tensor Bytes MB | Dominant Ops |
| ---: | ---: | ---: | --- |
| 0 | 1184.25 | 2305 | MUL_MAT 1056, SCALE 544.75, CONV_2D 288.25, GROUP_NORM 160, CONT 160 |
| 1 | 288 | 1376 | SCALE 640, CONV_2D 320, GROUP_NORM 192, UPSCALE 128, ADD 96 |
| 2 | 1152 | 5504 | SCALE 2560, CONV_2D 1280, GROUP_NORM 768, UPSCALE 512, ADD 384 |
| 3 | 2816 | 12544 | SCALE 6144, CONV_2D 2816, GROUP_NORM 1792, UPSCALE 1024, ADD 768 |
| 4 | 2048 | 16896 | SCALE 8192, CONV_2D 3584, GROUP_NORM 3584, ADD 1536 |
| 5 | 1024 | 1048 | SCALE 524, GROUP_NORM 512, CONV_2D 12 |

Interpretation:

- The trace has no hidden IM2COL.
- The reported `UNARY` bytes are zero because those nodes are not independently allocating large tensors in this trace; their consumers, mostly SCALE/CONV/GROUP_NORM/ADD, carry the storage cost.
- `SCALE` is the largest aggregate f32 category, but many SCALE tensors are bias/scale wrappers around conv or normalization output. Changing SCALE alone will not halve workspace if downstream f32-only ops force promotion.
- Direct conv and group norm are the critical blockers for real compact storage.

## Memory Savings Estimate

These estimates are derived from the traced planned workspace and f32 op pressure, not from a prototype.

| Case | Decode Estimate | Encode Estimate | Notes |
| --- | ---: | ---: | --- |
| Theoretical: all non-required f32 activations become bf16 | save 1024-1400 MB, 36-50% | save 512-768 MB, 33-50% | Requires compact conv output, group norm output, upscale, pointwise, pad/cont, and attention-side tensors where safe. This is not a small patch. |
| Realistic: pointwise + nearest upscale only | save 256-512 MB, 9-18% | save 128-384 MB, 8-25% | Only useful if graph typing lets outputs stay compact across consumers. If conv/group norm remain f32, many savings disappear into casts. |
| Realistic: pointwise + upscale + group norm output compact, reductions fp32 | save 512-1024 MB, 18-36% | save 256-640 MB, 17-42% | Group norm output is a meaningful win, but it needs a new typed CUDA group norm kernel with fp32 accumulation. |
| Pessimistic: direct conv and group norm stay mostly f32 | save 0-256 MB, 0-9% | save 0-128 MB, 0-8% | This is the likely outcome of isolated pointwise/upscale changes without a broader dtype graph plan. |

On a 16 GB RTX 4080 Super, the current 2816 MB decode workspace is already acceptable. The remaining win matters for headroom when UNet/CLIP/VAE are all resident and for larger-than-1024 decode, but it is not the gating item for integrating COMFY_NORMAL into Paralol.

## Per-Op Difficulty

| Op Area | Current Source | Current Dtype Assumption | Existing Compact Support | Risk | Memory Win | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| SCALE | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/scale.cu` | CUDA asserts src/dst f32 | none for CUDA scale | moderate implementation, low correctness | medium aggregate, low alone | after graph dtype plan |
| ADD / bias add | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/binbcast.cu` | graph often f32; CUDA has some f16 mixed paths | partial f16 support | moderate implementation, medium correctness | medium | after conv/group norm |
| SiLU / unary pointwise | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/unary.cu` | CUDA supports f32/f16 for many unary ops, not bf16 | partial f16 support | easy/moderate, low/medium correctness | low in current trace | later |
| clamp / process output | `ggml/src/ggml-cuda/clamp.cu`, VAE output path | CUDA supports f32/f16 clamp | partial f16 support | easy, low correctness | low | later |
| PAD / CONT | `ggml/src/ggml-cuda/pad.cu`, graph cont/copy paths | f32-only in important CUDA paths | copy supports f16/bf16 in places | moderate, medium correctness | medium in encode | later |
| nearest UPSCALE | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/upscale.cu` | ggml core asserts input f32; CUDA asserts src/dst f32 | none | moderate implementation, low/medium correctness for nearest | high in decode stages | first kernel candidate only after graph dtype plumbing |
| GROUP_NORM | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/norm.cu` | CUDA asserts src/dst f32; reductions f32 | none for compact output | hard implementation, high correctness | high | highest-value but risky |
| direct CONV_2D | `ggml/src/ggml.c`, `ggml/src/ggml-cuda/conv2d.cu`, `src/ggml_extend.hpp` | input/output f32; kernel f16/f32; current VAE weights loaded f32 for this model | no compact activation output | hard implementation, medium/high correctness, unknown speed | high | defer or use cuDNN/CUTLASS path |

## Comfy Dtype Findings

The local ComfyUI install reports the VAE dtype as `torch.bfloat16` in the parity harness. Comfy code supports `--fp16-vae` and `--bf16-vae`, and `comfy.model_management.vae_dtype()` selects bf16 when supported. On this RTX 4080 Super/Ada machine, bf16 is the correct first target, not f16.

I did not force Comfy f16 in this pass because the requested bounded investigation already had enough evidence: local Comfy defaults to bf16, and stable-diffusion.cpp currently lacks bf16 activation kernels. Forcing f16 would answer a secondary question, but it would not change the implementation recommendation.

## RTX 4080 Super Relevance

The remaining bottleneck is graph workspace and memory traffic, not model parameter residency. The VAE params are small compared with the f32 activation graph.

BF16 storage would reduce VRAM even if speed does not improve, because the largest tensors are activation buffers. Hand-written kernels would mostly reduce memory traffic and workspace. They would not automatically use Tensor Cores. The current direct conv kernel is a scalar direct convolution path, so compact dtype alone is unlikely to deliver a Comfy/cuDNN-like speed profile.

For conv and possibly group norm fusion, cuDNN or CUTLASS is the better long-term direction than expanding the current hand-written direct conv kernel. A hand-written compact group norm is plausible, but it must keep fp32 accumulation and needs careful parity tests.

## Prototype Decision

No prototype was implemented.

Reasons:

- Pointwise-only compact output is not useful while direct conv and group norm force f32.
- Nearest upscale is tempting, but both ggml core and CUDA paths are f32-only, and the next conv currently wants f32 input/output.
- Stage-boundary compact conversion would not reduce per-stage planned workspace and would add conversion noise.
- Group norm and direct conv are explicitly out of scope for a low-risk exploratory prototype.

## Go / No-Go

No-go for compact bf16/f16 kernels before Paralol integration.

Do this first:

1. Commit and productionize the current COMFY_NORMAL path.
2. Wire Paralol to prefer `sd_decode_latent_normal` / `sd_encode_image_normal`.
3. Keep strict guards for no IM2COL, no tiled VAE, no TAESD, and host_copies=0.
4. Add the parity harness to release validation.

Smallest safe next implementation task if we continue later:

1. Add an experimental compact dtype graph planner for COMFY_NORMAL only.
2. Add typed nearest-upscale for f16/bf16, but only measure it after graph typing can keep adjacent tensors compact.
3. Then add group norm compact output with fp32 accumulation.
4. Defer direct conv replacement until deciding between hand-written, cuDNN, or CUTLASS.

Highest-risk part: direct conv and group norm correctness. These can change image color, contrast, banding, or introduce structured artifacts even when mean error looks acceptable.

Tests that must gate future implementation:

- strict COMFY_NORMAL smoke at 1024x1024 SDXL
- encode and decode parity against Comfy bf16 normal VAE
- mean, p95, p99, p999, max abs diff and PSNR
- visual artifact review
- no IM2COL guard
- no host stage-boundary copy guard
- explicit dtype fallback reporting
- memory regression guard against current 2816 MB decode and 1536 MB encode baselines

