# Lens GPT-OSS MoE Execution Design

Status: design checkpoint after B1.15 native-tolerant conditioning.

The native Lens text encoder can emit all four GPT-OSS feature tensors and drive
the native Lens denoiser to a coherent 256x256 image. The remaining blocker is
MoE execution time, not tokenizer, denoiser, or routing correctness.

## Current Measurements

B1.14 native-tolerant full pass:

- total text encoder time: 1500.98 s
- output: coherent 256x256 Lens image

B1.15 `cpu-parallel-expert` full pass:

- total text encoder time: 727.62 s
- output conditioning: bit-identical to B1.14
- total MoE wall time: 698.68 s
- average MoE wall time: 29.11 s/layer
- average active experts: 27.83/layer
- worker-summed dequant time: 344.35 s
- worker-summed gate_up matmul time: 919.73 s
- worker-summed down matmul time: 460.60 s
- SWiGLU, routing, and reduce are negligible

B1.15 naive `cuda-expert-matmul` layer0 probe:

- layer0 MoE wall time: 63.44 s
- layer0 parity: same bounded output as CPU path
- failure reason: serial per-expert setup, upload, GEMM, and readback dominate

## Model Dimensions

- sequence length: 128
- hidden size: 2880
- intermediate size: 2880
- gate_up output: 5760
- experts: 32
- experts per token: 4

Per expert:

- gate_up weights: 5760 x 2880 = 16,588,800 elements
- down weights: 2880 x 2880 = 8,294,400 elements
- BF16 bytes: 49,766,400 bytes, about 47.5 MiB
- F32 bytes: 99,532,800 bytes, about 94.9 MiB

One layer, all 32 experts:

- BF16 weights: 1,592,524,800 bytes, about 1.48 GiB
- F32 weights: 3,185,049,600 bytes, about 2.97 GiB

One layer of BF16 dequantized expert weights fits on a 16 GiB card when staged
separately from the Lens denoiser. F32 also appears feasible for a diagnostic
layer, but BF16 is the only plausible production direction.

## Backend Options

### A. Per-layer dequantized host cache

This is effectively the current `cpu-parallel-expert` path. It reduces repeated
work and preserves output, but CPU gate_up/down matmul remains the dominant cost.
It is not sufficient.

### B. GPU-resident per-layer dequantized weights

For each layer, dequantize expert weights to BF16, upload all active or all 32
experts once, run MoE compute on GPU, then free the layer cache. This avoids
per-token and repeated per-expert uploads. Memory is feasible.

This must not be implemented as many independent per-expert cuBLASLt calls; that
shape was tested and is slower than CPU because setup and small GEMM overhead
dominate.

### C. Batched/grouped GPU GEMM

Pack routed token groups and run batched or grouped GEMMs for gate_up and down:

- gate_up: `[experts, tokens_for_expert, 2880] @ [experts, 2880, 5760]`
- down: `[experts, tokens_for_expert, 2880] @ [experts, 2880, 2880]`

This is the smallest native path that can plausibly beat CPU. A first prototype
can pad token groups to a per-layer max token count and use strided batched
BF16 GEMM. A production path can later replace padding with grouped GEMM or a
custom MoE kernel.

### D. Hybrid host cache plus batched upload

Upload only active experts, reuse descriptors/workspace, and batch transfers.
This is smaller than a full resident path, but still risks being PCIe-bound.
Given one layer BF16 fits, D should be secondary to B/C.

## Decision

Choose B/C together: per-layer BF16 GPU-resident expert weights with batched
expert GEMM. Do not continue the per-expert cuBLASLt path.

Next prototype should be layer0 only:

1. Dequantize all active layer0 experts to BF16 host buffers.
2. Upload gate_up/down weights once for the layer.
3. Pack routed token rows into a padded `[expert, max_tokens, hidden]` BF16
   input buffer.
4. Run gate_up as strided batched BF16 GEMM.
5. Copy gate_up output once, apply the already-matched BF16 stepwise SWiGLU
   semantics.
6. Upload gated rows once.
7. Run down as strided batched BF16 GEMM.
8. Apply routing/index_add with the existing BF16 rounding order.
9. Compare layer0 output against the B1.15 CPU path.

Gate for extending to all layers:

- layer0 MoE wall time below 5 s, or clearly trending there after removing
  diagnostic readbacks
- output matches the current native-tolerant CPU path within the existing BF16
  tolerance

Stop condition:

- if padded strided batched GEMM remains above 15 s for layer0, native sd.cpp
  text_encoder work should pause until a real grouped-GEMM/custom MoE kernel is
  available.

## B1.17 Layer0 Batched Prototype Result

Implemented a layer0-only padded strided batched GEMM backend:

- backend flag: `--moe-backend cuda-batched-expert-matmul`
- input layout: `[active_experts, max_tokens_per_expert, hidden]`
- gate_up weights: `[active_experts, 5760, 2880]`, row-major host storage,
  interpreted by cuBLAS as batched `op(A)=T`
- gate_up output: `[active_experts, max_tokens_per_expert, 5760]`
- down input: `[active_experts, max_tokens_per_expert, 2880]`
- down weights: `[active_experts, 2880, 2880]`
- down output: `[active_experts, max_tokens_per_expert, 2880]`

Layer0 timing:

- MoE wall time: 16.61 s
- dequant time: 15.79 s
- upload time: 0.234 s
- gate_up batched GEMM: 0.003 s
- BF16 stepwise SWiGLU: 0.068 s
- down batched GEMM: 0.003 s
- routing/index_add: 0.023 s

Layer0 parity against the native-tolerant CPU path remained bounded:

- expert-set mismatches: 0
- MoE max/mean diff: 0.25 / 0.000220541
- final hidden max/mean diff: 0.5 / 0.000225454

Conclusion: padded strided batched GEMM fixes the matmul bottleneck. The new
dominant cost is CPU MXFP4 dequantization into BF16 layer weights. Because the
prototype remains above the 15 s stop gate, the next native step is not another
GEMM variant; it is MXFP4 dequant acceleration or caching.

## B1.18 Layer0 BF16 Weight Cache Result

Implemented `--moe-cache layer-bf16` with `--moe-bf16-cache-dir <dir>`.

Behavior:

- cold run: dequantize layer weights to BF16 in the exact batched GEMM layout
  and write cache files
- warm run: load BF16 gate_up/down cache and skip MXFP4 decode in the MoE hot
  path

Layer0 cache files:

- `layer0_gate_up_all32.bf16`: 1,061,683,200 bytes
- `layer0_down_all32.bf16`: 530,841,600 bytes
- total: 1,592,524,800 bytes, about 1.48 GiB

Cold layer0 run:

- MoE wall time: 18.16 s
- dequant/cache-write time: 17.37 s
- upload time: 0.240 s
- gate_up batched GEMM: 0.002 s
- down batched GEMM: 0.001 s

Warm layer0 run:

- MoE wall time: 2.81 s
- BF16 cache load/copy time reported in dequant bucket: 1.89 s
- upload time: 0.248 s
- gate_up batched GEMM: 0.003 s
- BF16 stepwise SWiGLU: 0.040 s
- down batched GEMM: 0.001 s
- routing/index_add: 0.023 s

Parity stayed bounded against the current native-tolerant CPU path:

- expert-set mismatches: 0
- MoE max/mean diff: 0.25 / 0.000220541
- final hidden max/mean diff: 0.5 / 0.000225454

Conclusion: prepacked BF16 layer cache proves the native path can be fast enough
once MXFP4 decode is removed from the hot path. The remaining layer0 hot time is
mostly BF16 cache file load/copy plus upload, not matmul.

Next step:

- extend the BF16 cache path to all layers and run one full native-tolerant
  conditioning pass using the cache strategy
- compare against the B1.14/B1.15 native-tolerant conditioning artifact
- run one 256 image regression if the emitted conditioning matches
- later replace the BF16 cache prototype with CUDA MXFP4 dequant-to-BF16 or a
  proper runtime cache if the storage tradeoff is unacceptable

## Full Cache-Populating Pass

Ran one full native-tolerant pass with `--moe-cache layer-bf16` and
`--moe-backend cuda-batched-expert-matmul`.

This was a cache-populating pass: layer0 hit the existing cache, while layers
1-23 wrote their BF16 cache files.

- wall time: 447.55 s
- MoE sum: 417.55 s
- dequant/cache-write sum: 400.89 s
- upload sum: 4.46 s
- gate_up GEMM sum: 0.046 s
- down GEMM sum: 0.024 s
- SWiGLU sum: 0.953 s
- routing/index_add sum: 0.577 s
- BF16 cache files: 48
- BF16 cache total size: 38,220,595,200 bytes

The emitted conditioning did not match the B1.15 native-tolerant artifact:

- feature_0 max/mean diff: 3.25 / 0.0349789
- feature_1 max/mean diff: 20 / 0.276451
- feature_2 max/mean diff: 96 / 1.77377
- feature_3 max/mean diff: 800 / 9.60784
- mask equal: true

No image regression should be treated as passed from this artifact until the
batched backend's numeric drift is accepted or reduced. The current batched
implementation uses strided batched cuBLAS BF16 GEMM without the exact cuBLASLt
fused-bias path used by the projection parity work; this likely explains the
conditioning drift.
