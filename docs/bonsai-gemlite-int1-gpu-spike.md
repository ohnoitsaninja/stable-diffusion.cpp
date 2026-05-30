# Bonsai GemLite INT1 GPU Spike

Branch: `experiment/bonsai-gemlite-int1-gpu-spike`

This branch is an experimental path toward native stable-diffusion.cpp / ggml execution of the Bonsai Image 4B GemLite INT1 transformer. It is guarded by:

`SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1`

Normal stable-diffusion.cpp execution remains unchanged when the flag is not set.

## Sources Inspected

- Bonsai model: `prism-ml/bonsai-image-binary-4B-gemlite-1bit`
- Local pack: `F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt`
- GemLite reference: `F:\Paralol\local\gemlite-ref`
- GemLite files inspected:
  - `gemlite/bitpack.py`
  - `gemlite/core.py`
  - `gemlite/triton_kernels/utils.py`
  - `gemlite/triton_kernels/gemm_kernels.py`
  - `gemlite/triton_kernels/gemv_kernels.py`

## Pack Format Found

Bonsai uses `gemlite-int1-g128` for the Flux2/Klein-style transformer linears.

Observed from `transformer-gemlite-int1/quantization_config.json` and the real `state_dict.pt`:

- `bits=1`
- `group_size=128`
- `packing_bitwidth=8`
- `input_dtype=fp16`
- `output_dtype=fp16`
- quantized linears: `100`
- skipped support tensors: `9` major BF16 linear weights, plus norm/modulation weights

Each quantized linear has:

- `<name>.W_q`: packed `uint8`, logical shape `[K / 8, N]`
- `<name>.scales`: `float32`, logical shape `[K / 128, N]`
- `<name>.zeros`: `float32`, logical shape `[K / 128, N]`
- `<name>.metadata`: `int32[12]`
- `<name>.orig_shape`: `int32[2]`, `[N, K]`

GemLite dequantizes with:

`weight = bit * scale + zero`

For this Bonsai pack, inspected tensors have:

`zero ~= -scale / 2`

So the effective binary mapping is:

- bit `0`: `-scale / 2`
- bit `1`: `+scale / 2`

This is equivalent to binary sign weights with the scale represented in GemLite's `scale/zero` form.

## Code Added

- `src/bonsai_gemlite_int1.hpp`
- `src/bonsai_gemlite_int1.cpp`
- `src/bonsai_gemlite_int1_cuda.cu`
- `examples/bonsai-gemlite-int1-smoke/main.cpp`

The smoke executable:

- indexes the actual Bonsai `state_dict.pt`
- maps all `state_dict/data/<id>` entries for the pack with a narrow Bonsai state_dict indexer
- loads a real packed linear
- uploads packed INT1 weights and scales to CUDA
- launches a native CUDA fused packed-bit dequant/matmul kernel
- avoids full FP16 weight expansion

The kernel is intentionally simple and correctness-oriented. It is not yet the optimized production kernel.

## Validation Command

PowerShell:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake -S . -B build-bonsai-int1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_LENS_EXPERIMENTAL_RUNTIME=OFF
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-smoke --parallel 8
F:\Paralol\scripts\currentVRAM.ps1 -MinFreeGiB 10
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear transformer_blocks.0.attn.to_q --rows-m 8
```

Observed result:

```text
quantized_linears=100
skipped_bf16_weights=69
packed_weight_mb=438.75
scale_mb=109.688
zero_mb=109.688
linear=single_transformer_blocks.0.attn.to_qkv_mlp_proj
orig_shape_out_in=27648x3072
packed_wq_bytes=10616832
scale_count=663552
zero_count=663552
cuda_probe=ok
rows_m=4
kernel_ms=0.402752
device_bytes=16171008
output_min=-1.6084
output_max=1.66016
output_sum=-72.7251
full_fp16_weight_expansion=false
```

This proves the actual Bonsai GemLite pack can be indexed, all 100 packed transformer linears can be discovered, and one large real Bonsai transformer linear can run through a native CUDA packed INT1 path without expanding that linear to FP16.

The executable also fails closed without the experimental env gate:

```text
Bonsai GemLite INT1 spike is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.
```

Smoke executable SHA256:

```text
30E5D27B521480F697025C7F50E74AE1ABFE6EB23CFE561292A57392E051A521
```

## Current Blockers

This is not end-to-end Bonsai image generation yet.

The exact current blockers are:

1. The CUDA kernel is not yet wired into the real Flux2 transformer block dispatch.

   The current proof runs a real Bonsai linear outside the Flux graph. To enter real transformer inference, Flux linears matching the Bonsai quantized FQNs need a `BonsaiGemliteLinear` block or equivalent backend op that consumes activation tensors and packed sidecar tensors inside the runner.

2. The first kernel is correctness-oriented and scalar.

   It should be replaced by a tiled kernel using vectorized packed-byte loads, better activation tiling, and split-K or GEMV variants where Bonsai execution needs them.

3. End-to-end Bonsai text conditioning is not implemented.

   The HF repo uses a separate Qwen HQQ 4-bit text encoder. The first transformer spike should not become an HQQ research branch, but full image generation needs either a compatible existing text encoder path or a minimal bridge.

4. The Bonsai state_dict indexer is intentionally narrow.

   It maps tensor names to `state_dict/data/<id>` entries for this pack and does not try to become a generic PyTorch checkpoint loader.

## Next Work

1. Add a Flux2/Bonsai linear block that dispatches matching linears to the packed CUDA kernel.
2. Execute one real Flux2 transformer block with Bonsai packed linears.
3. Replace the scalar kernel with a tiled CUDA kernel for actual inference throughput.
4. Add the minimum viable text-conditioning bridge and attempt full Bonsai image generation.

## Next Pass: Transformer Runtime Wiring

This pass promoted the smoke-only linear loader into a guarded runtime hook.

### Files Changed

- `src/bonsai_gemlite_int1.hpp`
- `src/bonsai_gemlite_int1.cpp`
- `src/bonsai_gemlite_int1_cuda.cu`
- `src/ggml_extend.hpp`
- `src/flux.hpp`
- `src/diffusion_model.hpp`
- `src/stable-diffusion.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `examples/bonsai-gemlite-int1-smoke/main.cpp`

### Runtime Representation

`sd::BonsaiGemliteRuntime` now owns a CUDA-resident table of mapped Bonsai linears.

Each mapped runtime linear holds:

- original Bonsai HF linear name
- internal stable-diffusion.cpp Flux weight name
- output/input shape
- CUDA `W_q`
- CUDA `scales`
- CUDA `zeros`
- byte accounting
- call counters

The runtime keeps packed INT1 weights and GemLite scale/zero tensors resident on CUDA. It does not allocate a full FP16 transformer copy.

Runtime load summary from the current pack:

```text
[BonsaiGemLiteINT1] quantized_linears=100 mapped_runtime_linears=70 packed_weight_mb=405 scale_mb=101.25 zero_mb=101.25 device_mb=607.5 full_fp16_weight_expansion=false
```

The 70 mapped linears are the linears that match this fork's current Flux2 `Linear` nodes directly:

- all single-block `to_qkv_mlp_proj` and `to_out`
- double-block MLP in/out linears
- double-block attention output projections

The double-block q/k/v projections are now represented as virtual combined `qkv` runtime linears. The Bonsai pack stores these as separate `to_q`, `to_k`, and `to_v` GemLite tensors, while this fork's Flux path asks for one combined internal `qkv.weight`. The runtime now detects that virtual internal name and emits one ggml custom op that launches the three packed INT1 kernels into a single combined QKV output tensor.

### CUDA Kernel Status

The runtime path now uses a ggml CUDA `GGML_OP_CUSTOM` hook that launches a native CUDA packed INT1 matmul during backend graph execution.

Current kernel:

- `kernel=tiled_mn_int1`
- activations: FP16 on CUDA
- packed weights: INT1 `uint8` on CUDA
- scales/zeros: FP32 on CUDA
- output: FP16 on CUDA
- no full unpack buffer
- math: `w = bit * scale + zero`

This is still a first practical tiled kernel, not an optimized GemLite-equivalent kernel.

### Integration Seam

`GGMLRunnerContext` now has an optional Bonsai GemLite callback:

```text
bonsai_gemlite_runtime
bonsai_gemlite_linear_forward
```

`Linear::forward()` checks this callback first. If the current internal weight name maps to a Bonsai packed linear, it emits the custom CUDA op. If not, it falls through to the existing stable-diffusion.cpp behavior.

`FluxRunner::get_context()` attaches the runtime when present.

`StableDiffusionGGML` attaches the runtime for Flux2 when:

```text
SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1
```

and `diffusion_model_path` points at the Bonsai `state_dict.pt`.

### Runtime Graph Smoke

PowerShell:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target stable-diffusion sd-bonsai-gemlite-int1-smoke --parallel 8
F:\Paralol\scripts\currentVRAM.ps1 -MinFreeGiB 10
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --graph-runtime --internal-name model.diffusion_model.single_blocks.0.linear1.weight --rows-m 4
```

Observed:

```text
[BonsaiGemLiteINT1] quantized_linears=100 mapped_runtime_linears=100 packed_weight_mb=438.75 scale_mb=109.688 zero_mb=109.688 device_mb=658.125 full_fp16_weight_expansion=false
[BonsaiGemLiteINT1] linear_call name=model.diffusion_model.single_blocks.0.linear1.weight hf=single_transformer_blocks.0.attn.to_qkv_mlp_proj M=4 K=3072 N=27648 kernel=tiled_mn_int1 no_fp16_weight_expansion=true
graph_runtime=ok
bonsai_int1_linear_calls=1
unique_bonsai_int1_linears_executed=1
full_fp16_weight_expansion=false
```

Double-block image QKV graph smoke:

```powershell
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --graph-runtime --internal-name model.diffusion_model.double_blocks.0.img_attn.qkv.weight --rows-m 4
```

Observed:

```text
[BonsaiGemLiteINT1] quantized_linears=100 mapped_runtime_linears=100 packed_weight_mb=438.75 scale_mb=109.688 zero_mb=109.688 device_mb=658.125 full_fp16_weight_expansion=false
[BonsaiGemLiteINT1] qkv_linear_call name=model.diffusion_model.double_blocks.0.img_attn.qkv.weight M=4 K=3072 N_each=3072 kernel=tiled_mn_int1x3_concat no_fp16_weight_expansion=true
graph_runtime=ok
output_elements=36864
bonsai_int1_linear_calls=3
unique_bonsai_int1_linears_executed=3
full_fp16_weight_expansion=false
```

Double-block text QKV graph smoke:

```powershell
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --graph-runtime --internal-name model.diffusion_model.double_blocks.0.txt_attn.qkv.weight --rows-m 4
```

Observed:

```text
[BonsaiGemLiteINT1] qkv_linear_call name=model.diffusion_model.double_blocks.0.txt_attn.qkv.weight M=4 K=3072 N_each=3072 kernel=tiled_mn_int1x3_concat no_fp16_weight_expansion=true
graph_runtime=ok
output_elements=36864
bonsai_int1_linear_calls=3
unique_bonsai_int1_linears_executed=3
full_fp16_weight_expansion=false
```

Double-block MLP graph smoke:

```powershell
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --graph-runtime --internal-name model.diffusion_model.double_blocks.0.img_mlp.0.weight --rows-m 4
```

Observed:

```text
[BonsaiGemLiteINT1] linear_call name=model.diffusion_model.double_blocks.0.img_mlp.0.weight hf=transformer_blocks.0.ff.linear_in M=4 K=3072 N=18432 kernel=tiled_mn_int1 no_fp16_weight_expansion=true
graph_runtime=ok
bonsai_int1_linear_calls=1
unique_bonsai_int1_linears_executed=1
full_fp16_weight_expansion=false
```

This proves the packed Bonsai linears are now callable from a ggml backend graph through the same optional `Linear::forward` runtime seam that Flux2 will use.

### End-to-End Transformer Invocation Pass

Snapshot:

```text
branch=experiment/bonsai-gemlite-int1-gpu-spike
head=d4ff8206245444058c3a1800ec6bc52aff853e90
build_dir=build-bonsai-int1
gpu=NVIDIA GeForce RTX 4080 SUPER
driver=596.49
cuda_toolkit=13.2
pack=F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt
env=SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1
```

The previous `sd-cli.exe` blocker was link isolation, not Bonsai. Reconfiguring this spike build with existing Lens sources enabled lets the CLI link without modifying Lens semantics:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake -S . -B build-bonsai-int1 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_LENS_EXPERIMENTAL_RUNTIME=ON -DSD_BONSAI_GEMLITE_INT1_SPIKE=ON
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-gen-smoke sd-bonsai-gemlite-int1-smoke sd-cli --parallel 8
```

Built executable hashes:

```text
sd-bonsai-gemlite-int1-smoke.exe=62F3E10989DFBADF07AD5D1A51B9706B83A841044FFACFBD1C418D9D88BB71FE
sd-bonsai-gemlite-int1-gen-smoke.exe=55FB4440BCC0EBFC1CAAF67963E2403E6622CA7DFF9DF4387B5865D55D6AB17A
sd-cli.exe=31EFA8136BD789E9B0B27BDC5F1BC2CB8FBCCA5DDFBA3899F8A3592DA598D2A6
```

The normal CLI can now be invoked far enough to prove model-family forcing and Bonsai pack loading. It is not yet an image-generation path:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --sampling-method euler -p "a small bonsai tree on a desk" -o build-bonsai-int1\bonsai-run\bonsai-cli-probe.png -v --diffusion-fa
```

Current CLI boundary:

```text
Version: Flux.2 klein
Weight type stat: f32:1 | i8:1 | i32:1 | bf16:69
```

With no LLM path it fails during Flux2/Qwen conditioner construction. With a Qwen path it still exits before denoising. This is now a text-conditioning/model-load bridge blocker, not an INT1 transformer-linears blocker.

### Bonsai Generation Harness

Added a Bonsai-specific executable:

```text
examples/bonsai-gemlite-int1-gen-smoke
```

This harness does not write a final image. It constructs a tiny Flux.2 Klein transformer invocation with synthetic latent and text tensors, attaches the real Bonsai GemLite INT1 runtime, and executes all Bonsai transformer linear sites through the ggml CUDA custom op. It is the deepest current real-model execution path without solving Bonsai text conditioning and VAE.

Important implementation details:

- `Linear::init_params()` allocates a one-element placeholder for Bonsai-replaced linears when `SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1`, avoiding full FP weight allocation.
- `Linear::forward()` fails loudly if a Bonsai-replaced linear is reached without a runtime mapping.
- F32 activations are cast to F16 for the packed INT1 CUDA kernel, then cast back to F32 for graph compatibility.
- QKV is emitted as one custom op writing the combined output tensor directly; no CPU QKV assembly and no full unpack buffer.

Run:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-gen-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --width-tokens 4 --height-tokens 4 --text-tokens 8
```

Observed:

```text
[BonsaiGemLiteINT1] quantized_linears=100 mapped_runtime_linears=100 packed_weight_mb=438.75 scale_mb=109.688 zero_mb=109.688 device_mb=658.125 full_fp16_weight_expansion=false
bonsai_runtime_attached=true
bonsai_pack_loaded=true
bonsai_linears_total=100
bonsai_linears_runtime_mapped=100
bonsai_int1_linear_calls=200
unique_bonsai_int1_linears_executed=100
missing_linear_calls=0
cpu_dequant_fallback_calls=0
gpu_custom_op_calls=200
full_fp16_weight_expansion=false
pack_load_upload_ms=1906.9
flux_runner_init_ms=163.187
transformer_invocation_ms=167.45
output_elements=2048
```

The runner builds the graph twice internally, so `bonsai_int1_linear_calls=200` means all 100 unique linears were emitted in both graph builds. The important coverage field is `unique_bonsai_int1_linears_executed=100`.

Gate-off check:

```powershell
Remove-Item Env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1 -ErrorAction SilentlyContinue
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-gen-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --width-tokens 4 --height-tokens 4 --text-tokens 8
```

Observed:

```text
exit=3
bonsai_runtime_attached=false
Bonsai GemLite INT1 generation smoke is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.
```

### Remaining Tasks

1. Add a real Bonsai text-conditioning bridge or cached-conditioning input.
2. Connect the real Flux2/Bonsai sampler and VAE path, then attempt an actual image.
3. Optimize the tiled INT1 kernel after the real image path is working.

## Real Support Tensor Loading Pass

This pass replaces the harness' zeroed non-INT1 Flux parameters with real BF16 Bonsai support tensors from the same PyTorch zip pack.

Files changed in this pass:

- `src/name_conversion.cpp`
- `examples/bonsai-gemlite-int1-gen-smoke/main.cpp`
- `docs/bonsai-gemlite-int1-gpu-spike.md`

The Flux name conversion now recognizes Bonsai-specific support tensor names:

- `time_guidance_embed.timestep_embedder.linear_1.weight -> time_in.in_layer.weight`
- `time_guidance_embed.timestep_embedder.linear_2.weight -> time_in.out_layer.weight`
- `double_stream_modulation_img.linear.weight -> double_stream_modulation_img.lin.weight`
- `double_stream_modulation_txt.linear.weight -> double_stream_modulation_txt.lin.weight`
- `single_stream_modulation.linear.weight -> single_stream_modulation.lin.weight`

The generation smoke now supports:

```powershell
--load-support-tensors
```

When this flag is present, the smoke uses the existing `ModelLoader` to parse and convert the Bonsai `state_dict.pt`, constructs the Flux runner from the converted tensor metadata, filters out the 100 INT1 linears, and loads the remaining BF16 support tensors into the runner's GPU params buffer. The INT1 transformer weights still come only from the Bonsai GemLite runtime and are not expanded to FP16.

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-gen-smoke sd-bonsai-gemlite-int1-smoke sd-cli --parallel 8
```

Built executable hashes after this pass:

```text
sd-bonsai-gemlite-int1-gen-smoke.exe=BE687799BB05E7E61B6CEBB9FB84EB72241C78085857309FB20662F72DCF0DC3
sd-bonsai-gemlite-int1-smoke.exe=7DCEF8AF7342DA22430F35CD3C80F8CCC764370EBB1455B4E687A0C6A72EA201
sd-cli.exe=F959DFBD0BE034B0E4E7299FFF511824DF8A26F96786C807D7482434B4AD3549
```

Support tensor smoke command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-gen-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --width-tokens 4 --height-tokens 4 --text-tokens 8 --load-support-tensors
```

Output summary:

```text
bonsai_runtime_attached=true
bonsai_pack_loaded=true
bonsai_linears_total=100
bonsai_linears_runtime_mapped=100
bonsai_support_tensors_total=69
bonsai_support_tensors_loaded=69
bonsai_support_tensors_runtime_attached=69
bonsai_support_bytes=390100992
bonsai_zero_support_placeholders=0
bonsai_int1_linear_calls=200
unique_bonsai_int1_linears_executed=100
missing_linear_calls=0
cpu_dequant_fallback_calls=0
gpu_custom_op_calls=200
full_fp16_weight_expansion=false
pack_load_upload_ms=1726.97
flux_runner_init_ms=1144.13
transformer_invocation_ms=147.338
output_elements=2048
output_min=nan
output_max=nan
output_sum=nan
```

The support tensors are real and attached, and all reached Bonsai INT1 linears still route through the native CUDA custom op. The synthetic-conditioning output remains NaN, so this is still not a meaningful image-generation path. The likely next boundary is real Bonsai/Qwen conditioning or a cached-conditioning replay path with the expected Bonsai text tensor shape. In this pack, the Flux runner expects `txt_in.weight` input width `7680`; the previous synthetic smoke had been using `2560`, which is now corrected by reading the shape from the converted pack metadata.

Gate-off check still passes:

```text
exit=3
bonsai_runtime_attached=false
Bonsai GemLite INT1 generation smoke is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.
```

CLI generation attempt:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --sampling-method euler -p "a small bonsai tree on a desk" -o build-bonsai-int1\bonsai-run\bonsai-cli-qwen-probe.png -v --diffusion-fa --offload-to-cpu
```

Current CLI boundary after this pass:

```text
Version: Flux.2 klein
Weight type stat: f32:146 | q4_K:216 | q6_K:37 | i8:1 | i32:1 | bf16:69
llm: num_layers = 36, vocab_size = 151936, hidden_size = 2560, intermediate_size = 9728
process exits before denoising
```

This confirms the normal CLI is still blocked in the text-conditioning/model-load bridge before it can reach Bonsai denoising. The Bonsai INT1 transformer runtime and real support tensor path are now past the previous zero-placeholder blocker.

### Remaining Tasks After Support Loading

1. Add a real Bonsai text-conditioning bridge or cached-conditioning replay that produces the expected 7680-wide context tensor.
2. Diagnose and eliminate the NaN output from synthetic conditioning, or prove it disappears with real cached conditioning.
3. Connect the real Flux2/Bonsai sampler and VAE path, then attempt an actual image.
4. Optimize the tiled INT1 kernel after a real image path exists.

## End-to-End CLI Attempt: White Output / NaN Boundary

The normal `sd-cli.exe` path now reaches end-to-end image save with the Bonsai INT1 runtime enabled. It does not yet produce a valid Bonsai image.

Command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_TRACE_EULER_PARITY_STATS='1'
$env:SDCPP_TRACE_BONSAI_INT1_STATS='1'
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --prediction flux2_flow --width 512 --height 512 --steps 4 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampling-method euler -p "A bonsai tree in a quiet ceramic studio, soft morning light" -o build-bonsai-int1\bonsai-run\bonsai-cli-qwen-fluxvae-proper-settings.png -v --diffusion-fa --offload-to-cpu
```

Output path:

```text
build-bonsai-int1\bonsai-run\bonsai-cli-qwen-fluxvae-proper-settings.png
```

Result:

```text
save result image 0 ... success
image min=(255,255,255) max=(255,255,255) unique_colors=1
```

The all-white image is not a VAE or PNG writer failure. Debug stats show the first denoise step already returns NaN from the diffusion model:

```text
generate_image_cond_crossattn shape=[7680,512] min=-1910.51294 max=7202.58643 std=6.64073926
cpu_sampler_step1_noised_input min=-4.39577579 max=4.24012709 std=1.00000264
cpu_sampler_step1_cond_model_output min=inf max=-inf mean=nan
cpu_sampler_final_latent_before_decode min=inf max=-inf mean=nan
```

The first Bonsai image-token QKV custom op is finite, but the first text-token QKV custom op already contains NaNs/Infs:

```text
qkv output_stats count=4718592 finite=4718592 nan=0 inf=0 min=-61.2812 max=70.6875
qkv output_stats count=9437184 finite=5143665 nan=4234630 inf=58889 min=-65504 max=65504
```

Using the full `qwen_3_4b.safetensors` text encoder instead of the Q4_K_M GGUF does not fix the boundary; the generated 7680-wide conditioning is even larger:

```text
generate_image_cond_crossattn shape=[7680,512] min=-4574.54346 max=16345.9434 std=14.9163456
cpu_sampler_step1_cond_model_output mean=nan
```

Current interpretation:

- The Bonsai INT1 runtime attaches and real inference reaches all 100 INT1 linears.
- The transformer weights remain packed; `full_fp16_weight_expansion=false`.
- The saved image is invalid because model output becomes NaN during the first denoise call.
- The first observed NaN boundary is the text stream inside the Bonsai transformer.
- The most likely missing component is the exact Bonsai HQQ/Qwen conditioning path from the model package, not the generic Flux2 Klein Qwen bridge currently reused by sd.cpp.
- A secondary risk remains in the packed INT1 kernel: it has not yet been validated against GemLite on a real activation tensor, so bit polarity/scale semantics and output ranges still need a source-parity test.

Next concrete work:

1. Load or replay the exact Bonsai text-conditioning tensors expected by the reference runtime.
2. Validate the Bonsai INT1 CUDA kernel against GemLite for one real activation/weight pair, including text QKV.
3. Add a fail-fast guard when any Bonsai INT1 custom op emits NaN/Inf so bad images are not silently saved.
4. Only after the first-step model output is finite should the VAE/image path be considered again.

## Oracle parity / first bad tensor pass

This pass stops treating white PNG output as a success condition. The Bonsai CLI path now fails closed when a model output or final latent contains NaN/Inf unless explicitly overridden with:

```powershell
$env:SDCPP_BONSAI_ALLOW_NAN_CONTINUE='1'
```

The fail-fast path is active only when:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
```

New SDCPP dump flags:

```powershell
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_DIR='build-bonsai-int1\bonsai-dumps\failfast'
```

These dump:

- CPU-side f32 conditioning and sampler tensors as `.npy` plus JSON metadata.
- Bonsai CUDA custom-op f16 inputs/outputs as `.f16.bin` plus JSON metadata.

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-gen-smoke sd-bonsai-gemlite-int1-smoke sd-cli --parallel 8
```

Oracle script:

```text
tools/diagnostics/export_bonsai_oracle.py
```

Oracle command:

```powershell
py -3.12 tools\diagnostics\export_bonsai_oracle.py --out-dir build-bonsai-int1\bonsai-oracle
```

Oracle result:

```text
status=blocked
torch=2.6.0+cu118
cuda_device=NVIDIA GeForce RTX 4080 SUPER
pack_keys=569
pack_int1_linears=100
diffusers=missing
transformers=missing
gemlite=missing
hqq=missing
blocker=diffusers import failed
```

The script still records pack-level tensor stats for key Bonsai tensors, but the full reference pipeline cannot run in the current Python environment without installing the reference dependencies/configuration. This is an external oracle environment blocker, not an SDCPP build blocker.

SDCPP dump/fail-fast command:

```powershell
$dump='build-bonsai-int1\bonsai-dumps\failfast'
Remove-Item $dump -Recurse -Force -ErrorAction SilentlyContinue
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_TRACE_EULER_PARITY_STATS='1'
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_DIR=$dump
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampling-method euler -p "A bonsai tree in a quiet ceramic studio, soft morning light" -o build-bonsai-int1\bonsai-run\bonsai-failfast-should-not-save.png -v --diffusion-fa --offload-to-cpu
```

Result:

```text
exit=1
generate_image_cond_crossattn finite=3932160 nan=0 inf=0 min=-1910.51294 max=7202.58643 std=6.64073926
cpu_sampler_step1_noised_input finite=131072 nan=0 inf=0 min=-4.39577579 max=4.24012709 std=1.00000264
[BonsaiFailFast] tensor=cond step=1 shape=[32,32,128,1] count=131072 finite=0 nan=131072 inf=0; aborting before VAE decode
generate failed
output_png_saved=false
```

Dump files are under:

```text
build-bonsai-int1\bonsai-dumps\failfast
```

First bad tensor now identified:

```text
0000_qkv_input  shape=[512,3072]  finite=1572864 nan=0 inf=0
0001_qkv_output shape=[512,9216]  finite=4718592 nan=0 inf=0
0002_qkv_input  shape=[1024,3072] finite=3036274 nan=0 inf=109454
0003_qkv_output shape=[1024,9216] finite=5143667 nan=4233522 inf=59995
```

This narrows the failure beyond "image is white": the second first-block QKV input is already non-finite before the INT1 QKV matmul runs. That means the first bad tensor is upstream of that custom op, likely in first double-block image-stream preprocessing/modulation rather than in VAE decode or PNG save. The current dump labels are operation-order labels; the shape strongly indicates the bad `1024x3072` tensor is the first image-token stream QKV input, while the finite `512x3072` tensor is the text-token stream.

Support smoke after these changes still reaches all INT1 linears:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-gen-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --width-tokens 4 --height-tokens 4 --text-tokens 8 --load-support-tensors
```

Result summary:

```text
bonsai_support_tensors_loaded=69
bonsai_zero_support_placeholders=0
bonsai_int1_linear_calls=200
unique_bonsai_int1_linears_executed=100
full_fp16_weight_expansion=false
output_min=nan
```

Current remaining blocker:

1. Add named dump labels for the first Flux double block tensors so the `512x3072`/`1024x3072` operation-order inference becomes explicit.
2. Dump the first block modulation inputs/outputs (`vec`, `img_mod`, `txt_mod`, `img_norm`, `txt_norm`) to find why the `1024x3072` QKV input has infinities.
3. Run the external oracle in an environment with Diffusers, Transformers, GemLite, and HQQ, or capture cached reference conditioning tensors elsewhere and replay them in SDCPP.
4. Validate the INT1 CUDA kernel against GemLite on the first finite real activation; the first observed non-finite tensor now appears before one QKV call, but kernel parity is still unproven.

## First Bad Op Before Second QKV

This pass replaced operation-order QKV labels with names that include the internal Flux/Bonsai weight name. It also added debug-only block-0 taps behind:

```powershell
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_DIR='build-bonsai-int1\bonsai-dumps\first-bad-op-sync'
```

The new custom-op dump labels include names such as:

```text
model.diffusion_model.double_blocks.0.txt_attn.qkv.weight.q.input
model.diffusion_model.double_blocks.0.img_attn.qkv.weight.q.input
model.diffusion_model.double_blocks.0.txt_mlp.2.weight.output
```

The new block-0 debug taps include:

```text
block00.double.txt.stream.input
block00.double.txt.norm1.output
block00.double.txt.qkv.input.after_modulate
block00.double.img.stream.input
block00.double.img.norm1.output
block00.double.img.qkv.input.after_modulate
block00.double.img/txt.mod{1,2}.{shift,scale,gate}
block00.double.pe.input
```

Initial named-dump result without custom-op synchronization:

```text
block00.double.txt.qkv.input.after_modulate: finite
model.diffusion_model.double_blocks.0.txt_attn.qkv.weight.q.output: finite
block00.double.img.qkv.input.after_modulate: finite
model.diffusion_model.double_blocks.0.img_attn.qkv.weight.q.output: finite
model.diffusion_model.double_blocks.0.txt_attn.proj.weight.output: finite
model.diffusion_model.double_blocks.0.txt_mlp.0.weight.output: finite
model.diffusion_model.double_blocks.0.txt_mlp.2.weight.input: finite
model.diffusion_model.double_blocks.0.txt_mlp.2.weight.output: first bad tensor
```

The bad `txt_mlp.2` output pattern was row-based: rows 0-341 were all NaN and rows 342-511 were finite. The input activation and the device-resident GemLite metadata for that exact linear were finite:

```text
txt_mlp.2 input: finite, max_abs about 9.125
txt_mlp.2 scales: finite, min=0.0168457 max=0.103027
txt_mlp.2 zeros: finite, min=-0.0515137 max=-0.00842285
```

A small external diagnostic using the real activation dump and PyTorch-loaded Bonsai tensors computed finite reference values for sampled rows, including rows that the CUDA custom op had returned as NaN. That ruled out obvious bad scales/zeros and made the INT1 custom-op scheduling boundary suspicious.

The decisive finding: adding a device synchronization before and after the experimental Bonsai custom kernels makes the first bad op disappear. With synchronization enabled:

```text
model.diffusion_model.double_blocks.0.txt_mlp.2.weight.output: finite=1572864 nan=0 inf=0
model.diffusion_model.double_blocks.1.txt_mlp.2.weight.output: finite=1572864 nan=0 inf=0
model.diffusion_model.double_blocks.2.txt_mlp.2.weight.output: finite=1572864 nan=0 inf=0
model.diffusion_model.double_blocks.3.txt_mlp.2.weight.output: finite=1572864 nan=0 inf=0
```

The one-step CLI run now completes denoise/decode/save instead of failing at `BonsaiFailFast`.

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-gen-smoke sd-bonsai-gemlite-int1-smoke sd-cli --parallel 8
```

Fail-fast diagnostic command:

```powershell
$dump='build-bonsai-int1\bonsai-dumps\first-bad-op-sync'
Remove-Item $dump -Recurse -Force -ErrorAction SilentlyContinue
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_TRACE_EULER_PARITY_STATS='1'
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_DIR=$dump
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampling-method euler -p "A bonsai tree in a quiet ceramic studio, soft morning light" -o build-bonsai-int1\bonsai-run\bonsai-first-bad-op-sync.png -v --diffusion-fa
```

Result:

```text
exit=0
output=build-bonsai-int1\bonsai-run\bonsai-first-bad-op-sync.png
png_sha256=EAF7ED823B9AD86D48D4A8C878CF67485347B93DD66B08913919F6A4D1384A8B
image_min=0 image_max=255 image_mean=117.32 image_std=54.27 unique_rgb=190440
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

No-dump generation also completes:

```text
output=build-bonsai-int1\bonsai-run\bonsai-sync-nodump.png
png_sha256=72A58352D43486915F0DAF12DE955EE9D48C523FC81DCEA495CE46DF7DF340C3
image_min=0 image_max=255 image_mean=133.24 image_std=50.68 unique_rgb=139982
```

Current interpretation:

The first real failure was not VAE, support tensor decode, missing INT1 linears, or Qwen conditioning. It was CUDA stream ordering. The Bonsai custom kernels launch on the default stream, while ggml CUDA uses a nonblocking backend stream. Without explicit synchronization or a proper ggml-stream dispatch hook, the custom op can read partially-written upstream tensors and produce NaNs. The conservative `cudaDeviceSynchronize()` guard proves correctness but is not the final performance design.

Next concrete task:

Replace the coarse device-wide synchronization with a proper ggml CUDA stream integration for `GGML_OP_CUSTOM`, or pass the active ggml CUDA stream into the Bonsai custom kernels. Keep the current synchronize guard as the experimental correctness baseline until that is done.

## CUDA Stream Integration Pass

Root cause recap:

- ggml CUDA stores its active stream in `ggml_backend_cuda_context::stream()`.
- Existing ggml CUDA kernels launch on that stream, usually through calls such as `kernel<<<..., ctx.stream()>>>`.
- `GGML_OP_CUSTOM` previously unpacked `ggml_custom_op_params` and called `p.fun(dst, 0, 1, p.userdata)` without exposing the active CUDA stream to the custom op.
- Bonsai INT1 custom kernels therefore launched on CUDA's default stream and could race ggml's nonblocking backend stream.

Implementation:

- Added a tiny CUDA-backend scoped thread-local stream bridge in `ggml/src/ggml-cuda/ggml-cuda.cu`.
- Around `GGML_OP_CUSTOM`, ggml now sets the active custom-op stream to `ctx.stream()` for the duration of the callback.
- Bonsai custom kernels read that stream through `ggml_cuda_get_current_custom_op_stream()` and launch with:

```text
kernel<<<grid, block, 0, stream>>>(...)
```

- Removed the unconditional device-wide sync from the default Bonsai path.
- Kept a debug fallback:

```powershell
$env:SDCPP_BONSAI_FORCE_DEVICE_SYNC='1'
```

- Added stream diagnostics behind:

```powershell
$env:SDCPP_TRACE_BONSAI_STREAMS='1'
```

Diagnostic counters:

```text
bonsai_custom_kernel_launches
bonsai_launches_on_backend_stream
bonsai_device_sync_calls
bonsai_event_sync_calls
bonsai_default_stream_launches
```

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-gen-smoke sd-bonsai-gemlite-int1-smoke sd-cli --parallel 8
```

Executable hashes:

```text
sd-cli.exe SHA256=BDFA8D5D91EA16DA17E22DD2526963B405E241690128BCDFC272475C0AA6B313
sd-bonsai-gemlite-int1-gen-smoke.exe SHA256=E09FE73EB320D1CE99736B768E90D573389BE1B40946616E5F14C3B9DF3A4DD4
sd-bonsai-gemlite-int1-smoke.exe SHA256=59F062D25BEB4D1A0DCF47E11B67D3A4D3440563FB314B1A141CD06E083D0E15
```

Default stream-correct smoke command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_TRACE_BONSAI_STREAMS='1'
Remove-Item Env:SDCPP_BONSAI_FORCE_DEVICE_SYNC -ErrorAction SilentlyContinue
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampling-method euler -p "A bonsai tree in a quiet ceramic studio, soft morning light" -o build-bonsai-int1\bonsai-run\bonsai-stream-default-final.png -v --diffusion-fa
```

Default result:

```text
exit=0
elapsed_s=9.159
output=build-bonsai-int1\bonsai-run\bonsai-stream-default-final.png
image_min=0 image_max=255 image_mean=133.242228 image_std=50.676087 unique_rgb=139982
png_sha256=72A58352D43486915F0DAF12DE955EE9D48C523FC81DCEA495CE46DF7DF340C3
bonsai_custom_kernel_launches=100
bonsai_launches_on_backend_stream=100
bonsai_device_sync_calls=0
bonsai_event_sync_calls=0
bonsai_default_stream_launches=0
force_device_sync=false
```

Forced-sync comparison:

```powershell
$env:SDCPP_BONSAI_FORCE_DEVICE_SYNC='1'
build-bonsai-int1\bin\sd-cli.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --prediction flux2_flow --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampling-method euler -p "A bonsai tree in a quiet ceramic studio, soft morning light" -o build-bonsai-int1\bonsai-run\bonsai-stream-force-sync.png -v --diffusion-fa
```

Forced-sync result:

```text
exit=0
elapsed_s=9.411
output=build-bonsai-int1\bonsai-run\bonsai-stream-force-sync.png
image_min=0 image_max=255 image_mean=133.242228 image_std=50.676087 unique_rgb=139982
png_sha256=72A58352D43486915F0DAF12DE955EE9D48C523FC81DCEA495CE46DF7DF340C3
bonsai_custom_kernel_launches=100
bonsai_launches_on_backend_stream=100
bonsai_device_sync_calls=199
bonsai_event_sync_calls=0
bonsai_default_stream_launches=0
force_device_sync=true
```

The stream-correct path is now the default and no longer depends on a per-custom-op `cudaDeviceSynchronize()`. The image is byte-identical to the forced-sync comparison in this one-step smoke, and fail-fast did not trigger.

Remaining performance work:

- Replace the current simple INT1 kernel with a tiled Tensor Core-friendly or warp-specialized packed-bit matmul.
- Reduce per-linear launch overhead once correctness remains stable beyond the one-step smoke.
- Keep the stream bridge narrow; if upstream ggml later adds a custom-op context ABI, migrate Bonsai off the thread-local accessor.

## Warm benchmark and INT1 timing pass

This pass added a Bonsai-specific benchmark executable:

```text
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe
```

It keeps one `sd_ctx_t` alive, runs configurable warmup iterations, generates 3-5 measured images, saves each PNG, and reports:

- model load time separately from measured generation;
- text encode time;
- transformer denoise time;
- VAE decode time;
- PNG save time;
- full image time excluding model load;
- optional CUDA-event INT1 custom-op aggregate and top-20 linear ranking.

The executable sets `SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1`, `SDCPP_MODEL_FAMILY_HINT=bonsai`, and `SDCPP_PROFILE_BONSAI_GENERATION=1` internally. Linear timing is enabled only with `--profile-linears`, because CUDA event timing synchronizes the backend stream and intentionally perturbs normal timing.

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench sd-bonsai-gemlite-int1-gen-smoke sd-cli --parallel 8
```

Benchmark command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_TRACE_BONSAI_STREAMS='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampler euler --scheduler simple --seed 42 --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --warmup 1 --runs 3 --output-dir build-bonsai-int1\bonsai-bench-final
```

Exact generation settings:

```text
resolution=512x512
steps=1
sampler=euler
scheduler=simple
prediction=flux2_flow
cfg=1.0
guidance=1.0
requested_flow_shift=3.0
seed=42,43,44 for measured runs
prompt="A bonsai tree in a quiet ceramic studio, soft morning light"
elapsed_includes_model_load=false for run/full-image timings
model_load_ms=4930.49
```

Measured warm result:

```text
runs=3
full_mean_ms=4875.480
full_median_ms=4900.013
generate_mean_ms=4819.301
generate_median_ms=4842.016
denoise_mean_ms=4468.333
denoise_median_ms=4487.000
text_encode_mean_ms=125.000
vae_decode_mean_ms=222.000
png_save_mean_ms=56.179
stream_custom_kernel_launches=400 for warmup + 3 measured runs
stream_backend_launches=400
stream_device_sync_calls=0
stream_default_stream_launches=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Generated image surfaced:

```text
path=build-bonsai-int1\bonsai-bench-final\bonsai_bench_run_00.png
sha256=9E1A1BB246730AFC5FA2FFBFA3F4DD194F0E22584498FC9432CC9299C4F2E3BD
min=0 max=255 mean=133.242
```

Linear profile command:

```powershell
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors --width 512 --height 512 --steps 1 --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --sampler euler --scheduler simple --seed 42 --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --warmup 0 --runs 1 --profile-linears --output-dir build-bonsai-int1\bonsai-bench-profile
```

Profile result:

```text
custom_op_aggregate calls=100 total_ms=4586.892 unique_linears=100
rank 1: model.diffusion_model.single_blocks.12.linear1.weight 144.972 ms shape=1536x3072x27648
rank 2: model.diffusion_model.single_blocks.14.linear1.weight 139.000 ms shape=1536x3072x27648
rank 3: model.diffusion_model.single_blocks.11.linear1.weight 138.002 ms shape=1536x3072x27648
rank 4: model.diffusion_model.single_blocks.17.linear1.weight 136.532 ms shape=1536x3072x27648
rank 5: model.diffusion_model.single_blocks.8.linear1.weight 135.388 ms shape=1536x3072x27648
```

The dominant cost is the `single_blocks.*.linear1` family at `M=1536 K=3072 N=27648`. That is the first shape to optimize further.

Tiled kernel status:

- Added a first experimental shared-activation tiled INT1 kernel for normal and strided QKV outputs.
- It is correct on the image smoke, but slower than the original scalar CUDA kernel on the one-step 512x512 benchmark.
- The slower tiled path is therefore opt-in with `SDCPP_BONSAI_INT1_TILED_KERNEL=1`.
- Final default keeps the faster native scalar INT1 path to avoid a regression.

Comparison:

```text
default native scalar: denoise_ms=4250-4487, full_image_ms=4768-4900
experimental tiled 16x16x128: denoise_ms=4922, full_image_ms=5423
experimental tiled 8x16x128: denoise median about 4889, full median about 5279
```

Next kernel work should not continue with simple shared-A tiling. It needs a real GemLite-style warp-specialized packed-bit dot path for the dominant `1536x3072x27648` linears, likely reducing across K cooperatively instead of assigning a full K loop to one output element.

## Python oracle / first quality divergence pass

The SDCPP Bonsai path was producing finite, prompt-shaped but heavily corrupted images. Blind scheduler, VAE, and shift tweaks did not resolve quality, so this pass established Prism's real Python GemLite/HQQ runtime as the oracle and compared tensors.

Python oracle setup:

```text
venv=F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv
demo=F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo
model=F:\Paralol\local\bonsai-python-oracle\models\bonsai-image-binary-4B-gemlite-1bit
transformer=transformer-gemlite-int1
text_encoder=text_encoder-hqq-4bit
vae=vae
torch=2.11.0+cu128
diffusers=0.38.0
transformers=5.9.0
gemlite=0.5.1.post1
hqq=0.2.8.post1
```

The local transformer pack initially only had `state_dict.pt`; Prism's runtime also required `config.json`, `quantization_config.json`, and `gemlite_autotune.json`. Those metadata files were fetched from `prism-ml/bonsai-image-binary-4B-gemlite-1bit` without fetching another large weight file.

Reference generation command:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH='F:\Paralol\local\bonsai-python-oracle\models\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1'
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe scripts\generate.py --force-gpu-run --model binary-gemlite --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --seed 42 --steps 4 --size 1024x1024 --output F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_reference_1024_seed42.png
```

Reference result:

```text
output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_reference_1024_seed42.png
sha256=409536FA114F145C81D96A6DDE0F910430624472141BF694C464C4FA14453026
size=1024x1024
steps=4
seed=42
setup_s=38.85
diffusion_s=38.24
wall_s=77.15
min=0 max=255 mean=99.5915 std=69.6003 unique_rgb=287482
quality=clean prompt-following bonsai image
```

Added diagnostics:

- `tools\diagnostics\export_bonsai_oracle.py`
- `tools\diagnostics\compare_bonsai_oracle.py`

The oracle exporter uses Prism's `backend_gpu.GpuPipeline`, not stock Diffusers fallback. It captures the actual Klein/Qwen3 contract:

- rendered chat-template prompt;
- tokenizer ids and mask;
- hidden layers 9/18/27 stacked into `[1,512,7680]`;
- CPU f32 initial noise and packed latent;
- empirical-mu FlowMatch Euler timesteps;
- first double-block text/image stream inputs, modulation/norm/QKV boundaries;
- selected single block linear inputs/outputs;
- step 0 model output and final VAE input/output.

Oracle tensor capture command:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\export_bonsai_oracle.py --width 512 --height 512 --steps 4 --seed 42 --guidance 1.0 --out-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512
```

Oracle 512 result:

```text
output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512\oracle_reference_512x512_seed42.png
sha256=99C8DDB838C82406EE6915656D20AED5AAAAF80A83C14F5ACF308C003674C454
min=0 max=255 mean=115.2731 std=65.9522 unique_rgb=116393
scheduler_mu=2.0306897079499455
input_sigmas=[1.0,0.75,0.5,0.25]
timesteps=[1000.0,958.0853881835938,883.9818725585938,717.49658203125]
```

Matching SDCPP dump command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_TRACE_BONSAI_INT1_STATS='1'
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_DIR='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_compare_512\dumps'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 0 --runs 1 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_compare_512
```

Comparison command:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\compare_bonsai_oracle.py --python-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512 --sdcpp-dump-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_compare_512\dumps --out F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\compare_512.json
```

Key tensor comparisons:

```text
conditioning.prompt_embeds [512,7680]:
  mean_abs=1.0711 max_abs=8830.01 corr=0.97795 cosine=0.978229
  python std=14.6626, sd.cpp std=6.6407

block00.text_stream_input [512,3072]:
  mean_abs=2.11063 max_abs=1882 corr=0.967937 cosine=0.967991

block00.text_qkv_input_after_modulate [512,3072]:
  mean_abs=1.30866 max_abs=13.923 corr=0.54472 cosine=0.544942

step0.model_output [1024,128]:
  mean_abs=1.059 max_abs=6.05469 corr=0.00528199 cosine=0.0111399
```

Conclusion:

The first major quality divergence is the text-conditioning contract, before transformer math. Prism's reference path uses `text_encoder-hqq-4bit/qmodel.pt`, Qwen chat template with `add_generation_prompt=True` and `enable_thinking=False`, and hidden layers `(9,18,27)` stacked to `[1,512,7680]`. The current SDCPP path uses `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` through the generic Qwen bridge. With the correct `[512,7680]` comparison layout, SDCPP conditioning is correlated with the oracle (`corr≈0.978`) but materially different in value distribution and scale (`std≈6.64` vs `14.66`, `mean_abs≈1.07`, large outlier differences). By the first block's modulated text-QKV input, correlation has fallen to `≈0.545`, and the step-0 model output is effectively uncorrelated (`corr≈0.0053`).

This rules out VAE/decode as the primary cause for the corrupted Bonsai images. It also makes further scheduler or INT1 kernel tweaking premature until SDCPP can consume Bonsai-compatible conditioning. Initial noise is also not identical in this comparison, but it is not the explanation for the severe quality corruption; the conditioning boundary is already wrong.

Next concrete work:

- Add a Bonsai-specific text-conditioning lane that matches the reference HQQ/Qwen contract, or add a diagnostic cached-conditioning import path using `conditioning.prompt_embeds_bsz_seq_7680.npy`.
- Re-run SDCPP with oracle conditioning. If quality fixes, implement the HQQ/Qwen path properly; if not, continue comparison at block00 QKV and scheduler/latent boundaries.
- Keep the existing INT1 runtime, support tensor loading, stream integration, and fail-fast checks unchanged while this is resolved.

## Final-layer parity pass

This pass resumed from the 512x512 oracle replay after block and single-block parity had already been proven. The diagnostic run used:

```text
resolution=512x512
steps=4
seed=42
prompt=A bonsai tree in a quiet ceramic studio, soft morning light
conditioning=oracle replay from conditioning.prompt_embeds_bsz_seq_7680.npy
initial_noise=oracle replay from latents.initial_noise_cpu_f32_nchw.npy
dump_sync=active ggml backend stream
line_profiling=false
```

The run directory is:

```text
F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_final_compare_512_fixed
```

The first real divergence after `single_blocks.12` was in the final layer, not the combined stream order:

```text
final_layer.input_image_slice vs python norm_out.input_arg0:
  mean_abs=0.10199621 max_abs=6.5854492 corr=0.999998131 cosine=0.999995828

final_layer.output vs python proj_out.output, before fix:
  corr=0.395214 mean_abs=0.826630
```

The image-token slice was therefore correct enough to continue; the final AdaLayerNorm/projection path was not. The concrete bug was the final modulation chunk order in `Flux::LastLayer::forward`. Diffusers `AdaLayerNormContinuous` chunks the final modulation as:

```text
scale, shift = torch.chunk(emb, 2, dim=1)
x = norm(x) * (1 + scale) + shift
```

The fork had consumed the chunks as `shift, scale`. The fix changes only the final layer chunk assignment to `scale = m_vec[0]`, `shift = m_vec[1]`; the existing `Flux::modulate(ctx, x, shift, scale)` helper remains unchanged.

Post-fix final-layer comparison:

```text
final norm/proj input vs python proj_out.input_arg0:
  mean_abs=0.00070316647 max_abs=0.039858937 corr=0.999993044 cosine=0.999989033

final proj output vs python proj_out.output:
  mean_abs=0.003267251 max_abs=0.02734375 corr=0.999990463 cosine=0.999990940
```

The comparison also exposed one harness/layout trap: SDCPP's dumped sampler tensor is WHCN backing storage. For the 512 Bonsai packed output, the correct comparison against Python `[1024,128]` packed output is:

```python
sd_packed = np.load(sdcpp_npy).reshape(-1).reshape(128, 1024).T
```

Using that layout, step-0 model output and the first scheduler update are aligned with the Python oracle:

```text
step0 model output vs python proj_out.output:
  mean_abs=0.003267251 corr=0.999990463 cosine=0.999990940

step0 scheduler update vs python step0.latents_after_scheduler:
  mean_abs=0.00024135859 max_abs=0.0023293495 corr=0.999999942 cosine=0.999998450
```

The scheduler update comparison must use the shifted next sigma `0.9580853881835938`; using the raw input sigma `0.75` gives a false mismatch.

After the fix, one normal 512x512 4-step smoke was run without oracle conditioning/noise overrides and without tensor dumps:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 0 --runs 1 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_normal_512_after_final_fix
```

Result:

```text
output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_normal_512_after_final_fix\bonsai_bench_run_00.png
sha256=D79256C413F5416004457766D41131923ED2FC0CE618C2AF665C0A0CA794EC7F
min=0 max=255 mean=116.229116 std=54.882233 unique_rgb=127989
quality=visually plausible bonsai image, no white/NaN failure
text_encode_ms=198
denoise_ms=15984
vae_decode_ms=225
full_image_ms=16466.329
elapsed_includes_model_load=false
```

Remaining boundary:

The final 4-step latent is still not aligned with the Python oracle:

```text
python final.latents_packed_before_decode vs SDCPP final_latent_before_decode, WHCN-corrected:
  mean_abs=0.40351945 max_abs=5.323741 corr=0.84881655 cosine=0.849050581
```

That is no longer a final-layer step-0 problem. The next smallest pass should capture Python and SDCPP model outputs and post-update latents for steps 1, 2, and 3, then stop at the first later-step divergence. Do not revisit VAE, text encoder, stream ordering, or final AdaLayerNorm until that per-step boundary says to.

## Multi-step denoise parity after final-layer fix

This pass added per-step dumps for the Python Prism oracle and the SDCPP Bonsai CPU sampler wrapper. The run kept the same diagnostic inputs:

```text
resolution=512x512
steps=4
seed=42
prompt=A bonsai tree in a quiet ceramic studio, soft morning light
conditioning=oracle replay
initial_noise=oracle replay
line_profiling=false
tiled_kernel=false
```

Build:

```powershell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench sd-cli --parallel 8
```

Python oracle command:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH='F:\Paralol\local\bonsai-python-oracle\models\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1'
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\export_bonsai_oracle.py --width 512 --height 512 --steps 4 --seed 42 --guidance 1.0 --out-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512_multistep
```

SDCPP replay command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_TRACE_BONSAI_INT1_STATS='1'
$env:SDCPP_BONSAI_DUMP_TENSORS='1'
$env:SDCPP_BONSAI_DUMP_TENSOR_LIMIT='220'
$env:SDCPP_BONSAI_CONDITIONING_NPY='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512_multistep\conditioning.prompt_embeds_bsz_seq_7680.npy'
$env:SDCPP_BONSAI_NOISE_NPY='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\python_capture_512_multistep\latents.initial_noise_cpu_f32_nchw.npy'
$env:SDCPP_BONSAI_DUMP_DIR='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_multistep_compare_512_schedfix\dumps'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 0 --runs 1 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_multistep_compare_512_schedfix
```

First divergent boundary before the fix:

```text
Python shifted sigmas: [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0]
SDCPP shifted sigmas: [1.0, 0.857692361, 0.602150559, 0.00892859139, 0.0]
```

The step-0 model output and update matched because both schedules start at `1.0`. Step 1 diverged before the transformer call because the loop-carried latent was advanced to the wrong next sigma. Root cause category: timestep/sigma progression.

The fix is narrow and Bonsai-gated in `Flux2FlowDenoiser::get_sigmas`: when `SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1` and the discrete scheduler is used, SDCPP now matches Prism's `FlowMatchEulerDiscreteScheduler` path by building base sigmas `[1.0, 0.75, 0.5, 0.25]`, applying the empirical dynamic `mu`, and appending `0.0`.

Post-fix schedule:

```text
Python timesteps: [1000.0, 958.0853881835938, 883.9818725585938, 717.49658203125]
Python sigmas:    [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0]
SDCPP timesteps:  [1.0, 0.958085358, 0.883981824, 0.717496574]
SDCPP sigmas:     [1.0, 0.958085358, 0.883981824, 0.717496574, 0.0]
```

The SDCPP timestep values are already divided by 1000 before entering the transformer, so they compare to Python's `timestep / 1000` inputs.

Step-by-step comparison after the schedule fix:

```text
step  boundary       corr        cosine      mean_abs       max_abs
0     timestep       1.000000    1.000000    0              0
0     latent_start   0.99999998  0.99999998  0.000140507   0.00177097
0     model_output   0.99999046  0.99999048  0.00326725    0.0273438
0     post_update    0.99999994  0.99999994  0.000241364   0.00232959
1     timestep       1.000000    1.000000    0.0000775456  0.0000775456
1     latent_start   0.99999994  0.99999994  0.000241364   0.00232959
1     model_output   0.99996300  0.99996306  0.00561940    0.0947266
1     post_update    0.99999967  0.99999967  0.000531027   0.00727442
2     timestep       1.000000    1.000000    0.000192761   0.000192761
2     latent_start   0.99999967  0.99999967  0.000531027   0.00727442
2     model_output   0.99995189  0.99995196  0.00715932    0.220703
2     post_update    0.99999667  0.99999667  0.00142837    0.0391351
3     timestep       1.000000    1.000000    0.000211418   0.000211418
3     latent_start   0.99999667  0.99999667  0.00142837    0.0391350
3     model_output   0.99986389  0.99986402  0.01065468    1.02148
3     post_update    0.99988914  0.99988930  0.00838942    0.737918
final latent         0.99988914  0.99988930  0.00838942    0.737918
```

The final 4-step latent now clears the requested `corr > 0.999` threshold. Remaining drift is small accumulated numeric/model-output error, not a layout, schedule, or loop-carried latent bug.

Normal 512x512 image after the schedule fix:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt 'A bonsai tree in a quiet ceramic studio, soft morning light' --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 0 --runs 1 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_normal_512_after_schedule_fix
```

Result:

```text
output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-oracle\sdcpp_normal_512_after_schedule_fix\bonsai_bench_run_00.png
sha256=BD70BC5BC58723655ECD8957A97A4178AEBABDF6ACEAC1DEDEFEA244896F1E8A
min=0 max=255 mean=121.903511 std=66.143736 unique_rgb=134411
quality=visually plausible bonsai image
text_encode_ms=204
denoise_ms=15445
vae_decode_ms=245
full_image_ms=15973.354
elapsed_includes_model_load=false
```

## Clean quality and performance baseline after parity fixes

After the stream, final-layer, packed-latent, and Bonsai FlowMatch schedule fixes, the path was rebuilt and run without oracle conditioning/noise overrides, tensor dumps, line profiling, tiled-kernel experiments, or forced sync.

Build:

```powershell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench sd-cli --parallel 8
```

Debug env cleared before quality/timing:

```text
SDCPP_BONSAI_DUMP_TENSORS
SDCPP_BONSAI_DUMP_TENSOR_LIMIT
SDCPP_BONSAI_CONDITIONING_NPY
SDCPP_BONSAI_NOISE_NPY
SDCPP_PROFILE_BONSAI_INT1
SDCPP_BONSAI_INT1_TILED_KERNEL
SDCPP_TRACE_BONSAI_STREAMS
SDCPP_TRACE_BONSAI_INT1_STATS
SDCPP_BONSAI_FORCE_DEVICE_SYNC
```

Enabled env:

```text
SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1
SDCPP_MODEL_FAMILY_HINT=bonsai
```

Quality validation used:

```text
prompt=A bonsai tree in a quiet ceramic studio, soft morning light
seed=42
steps=4
sampler=euler
scheduler=discrete
cfg=1.0
guidance=1.0
flow_shift=3.0
prediction=flux2_flow
```

Quality outputs:

```text
512 output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\quality-512\bonsai_bench_run_00.png
512 sha256=BD70BC5BC58723655ECD8957A97A4178AEBABDF6ACEAC1DEDEFEA244896F1E8A
512 stats min=0 max=255 mean=121.903511 std=66.143736 unique_rgb=134411
512 quality=visually plausible

1024 output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\quality-1024\bonsai_bench_run_00.png
1024 sha256=8BC6430FF1B2ED576CE3DECA00E4A5A7C11CC1E4049FC580958086D678070525
1024 stats min=0 max=255 mean=118.832656 std=62.672439 unique_rgb=244154
1024 quality=visually plausible
```

Warm timing, model loaded once:

```text
resolution  runs  load_ms  text_mean_ms  denoise_mean_ms  denoise_median_ms  vae_mean_ms  png_mean_ms  full_mean_ms  full_median_ms
512         3     4519.46  138.000       13590.333        13563.000          187.000      68.930       13988.672     13968.096
1024        3     4736.26  115.000       41894.667        41576.000          803.000      206.463      43031.862     42684.117
```

The clean baseline shows quality is no longer the blocker. Denoise dominates both resolutions.

## GemLite kernel parity pass

The next performance pass focused only on the dominant Bonsai INT1 linear shape:

```text
linear=single_transformer_blocks.0.attn.to_qkv_mlp_proj
internal=model.diffusion_model.single_blocks.0.linear1.weight
M=1536
K=3072
N=27648
W_nbits=1
group_size=128
packing_bitwidth=8
W_group_mode=4
```

Python/GemLite oracle command:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear single_transformer_blocks.0.attn.to_qkv_mlp_proj --m 1536 --warmup 1 --runs 5 --out F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\gemlite_shape_dispatch.json
```

GemLite selected:

```text
matmul_type=GEMM
cache_key=(1536, 27648, 3072, 128, 8, 101)
BLOCK_SIZE_M=64
BLOCK_SIZE_N=128
BLOCK_SIZE_K=128
GROUP_SIZE_M=8
A_load_order=2
num_warps=4
num_stages=1
median_ms=3.212
mean_ms=3.337
```

The SDCPP single-shape benchmark target was added as:

```text
sd-bonsai-gemlite-int1-kernel-bench.exe
```

Current SDCPP kernel baseline:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --internal-name model.diffusion_model.single_blocks.0.linear1.weight --rows-m 1536 --warmup 1 --runs 5
```

Result:

```text
median_ms=94.638
mean_ms=94.929
finite=42467328 nan=0 inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

An experimental shape-specialized kernel was added behind:

```text
SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL=1
```

This path keeps the existing packed representation and uses GemLite's explicit group formula:

```text
w = bit * scale + zero
acc_group = scale * sum(bit-selected activations) + zero * sum(all group activations)
```

It also shares each row's 128-activation group across a 32-column tile. It is still not a Triton/Tensor Core equivalent, but it removes the worst repeated scale/zero and activation loads in the old one-output-thread loop.

Shape-kernel benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --internal-name model.diffusion_model.single_blocks.0.linear1.weight --rows-m 1536 --warmup 1 --runs 5 --compare-current
```

Result:

```text
compare_current corr=1.000000000 cos=1.000000000 mean_abs=0.000000249 max_abs=0.001953125
median_ms=37.411
mean_ms=37.439
speedup_vs_current=2.53x median
finite=42467328 nan=0 inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

512 image timing with `SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL=1`:

```text
output=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\gemlite-shape-512\bonsai_bench_run_00.png
sha256=23E37BFFBE87982667D1297E67EB04B0B8438B121782A1DDA6F40A209E0AAC17
stats min=0 max=255 mean=121.911393 std=66.142344 unique_rgb=134385
quality=visually plausible
```

Warm timing with the shape kernel:

```text
resolution  runs  load_ms  text_mean_ms  denoise_mean_ms  denoise_median_ms  vae_mean_ms  png_mean_ms  full_mean_ms  full_median_ms
512         3     4623.57  118.667       9935.333         9941.000           197.333      56.177       10311.397     10315.858
```

This is a material full-image improvement:

```text
512 denoise median: 13563 ms -> 9941 ms
512 full median:    13968 ms -> 10316 ms
```

Remaining gap:

```text
GemLite oracle dominant linear median: 3.212 ms
SDCPP shape kernel median:             37.411 ms
```

The next optimization target is not more model plumbing. Port the actual GemLite/Triton GEMM strategy more closely for this shape family: cooperative block-level accumulation across `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `BLOCK_SIZE_K=128`, and then expand to the other dominant single-block and projection shapes. The current C++ kernel is faster and correct, but it is still scalar-FMA CUDA, not GemLite-class low-bit GEMM.

## Dominant linear GemLite GEMM V2 pass

This pass stayed on the isolated dominant-linear benchmark. No scheduler, text encoder, VAE, 1024 generation, or full per-linear profiling was run.

Source/cache contract extraction was added as:

```text
tools\diagnostics\bonsai_gemlite_contract.py
```

Contract command:

```powershell
py -3.12 tools\diagnostics\bonsai_gemlite_contract.py --pack-dir F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1 --generated-cache F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\gemlite_autotune_cache_after_run.json --m 1536 --n 27648 --k 3072 --type-id 101 --out F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\dominant_shape_contract.json
```

Important finding:

```text
cache_key=(1536, 27648, 3072, 128, 8, 101)
family=GEMM
pack_cache_has_key=false
generated_cache_has_key=true
```

The model pack does include `gemlite_autotune.json`, but that bundled cache does not contain the fork/oracle dominant shape key. The authoritative config for this exact fork shape came from the generated GemLite runtime cache after the Python oracle ran the same dominant linear. That difference matters: the pack cache and the generated runtime cache must not be treated as interchangeable.

Source-backed contract:

```text
gemlite/core.py:
  batch_size > 64 selects GEMM, so this dominant M=1536 linear is not GEMV or split-K.

gemlite/bitpack.py:
  row packing gives W_q shape [K / elements_per_sample, N].

gemlite/triton_kernels/utils.py:
  W_group_mode=4 dequantizes as tl.fma(bit, scale, zero).

gemlite/triton_kernels/gemm_kernels.py:
  cached GEMM key is (get_closest_m(M), N, K, group_size, elements_per_sample, type_id).
  for packed non-block metadata, BLOCK_SIZE_K is clamped to group_size before pruning.
```

Refreshed GemLite oracle config:

```powershell
$out='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\dominant_shape_config.json'
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear single_transformer_blocks.0.attn.to_qkv_mlp_proj --m 1536 --warmup 1 --runs 3 --out $out
```

Oracle result:

```text
matmul_type=GEMM
cache_key=(1536, 27648, 3072, 128, 8, 101)
BLOCK_SIZE_M=64
BLOCK_SIZE_N=128
BLOCK_SIZE_K=128
GROUP_SIZE_M=8
A_load_order=2
num_warps=4
num_stages=1
median_ms=3.221
mean_ms=3.472
```

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

The benchmark now accepts aliases and explicit kernel selection:

```text
--linear <internal weight name>
--m <rows>
--kernel current|gemlite-shape-v1|gemlite-shape-v2|env
--compare-current
```

V2 is gated behind:

```text
SDCPP_BONSAI_INT1_GEMLITE_GEMM_V2=1
SDCPP_BONSAI_INT1_GEMM_V2_VARIANT=<optional>
```

The default V2 variant is 9. It uses the verified Bonsai relation `zero = -scale / 2` in an equivalent scale-only form:

```text
scale * (sum(bit-selected activations) - 0.5 * sum(all activations))
```

Correctness is checked against the known-good current SDCPP kernel. The best V2 candidate remained effectively exact:

```text
corr=1.000000000
cos=1.000000000
mean_abs=0.000000249
max_abs=0.001953125
finite=42467328 nan=0 inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Single-shape timings, `M=1536 K=3072 N=27648`, 10 measured runs:

```text
kernel/variant               median_ms  mean_ms   notes
current                      95.105     95.434   old scalar loop
gemlite-shape-v1             37.218     37.256   prior grouped shape kernel
v2 variant 1                 29.357     29.199   BM=4 BN=64  cols/thread=4
v2 variant 2                 25.149     25.068   BM=4 BN=128 cols/thread=8
v2 variant 3                 25.882     25.922   BM=8 BN=128 cols/thread=8
v2 variant 4                 27.090     27.109   BM=4 BN=128 cols/thread=16
v2 variant 5                 27.097     27.028   BM=8 BN=128 cols/thread=16
v2 variant 6                 25.287     25.362   BM=16 BN=128 cols/thread=8
v2 variant 7                 25.300     25.253   BM=8 BN=256 cols/thread=8
v2 variant 8                 27.033     26.863   BM=4 BN=256 cols/thread=16
v2 variant 9 default         24.908     24.956   BM=4 BN=128 cols/thread=8 scale-only
v2 variant 10                24.967     24.991   BM=8 BN=128 cols/thread=8 scale-only
v2 variant 11 split-K        30.497     30.640   4-way split-K plus reduction
```

Result:

```text
best_v2_median=24.908 ms
speedup_vs_current=3.82x
speedup_vs_v1=1.49x
target_below_20ms=not met
stretch_below_10ms=not met
reference_gemlite_median=3.221 ms
```

Split-K did not help in this scalar CUDA design. It created more parallelism across K, but the extra partial-output traffic and reduction outweighed the shorter per-thread K loop.

No 512 image timing was run for V2 because the pass requirement was a 2x single-shape win over V1 before full-image timing. V2 improved the dominant linear, but not enough.

Conclusion:

The remaining gap is not model plumbing. The scalar CUDA family is now near its useful ceiling for this shape. Closing the next gap requires a real low-bit GEMM design closer to GemLite/Triton: warp/block cooperative accumulation over the `BM=64 BN=128 BK=128` tile, with multiple threads contributing to each output tile instead of one thread owning one or several output columns serially. The next exact bottleneck is cooperative K reduction within a CTA, not another schedule/VAE/text fix.

## Tensor Core GemLite GEMM translation pass

This pass started with source-to-CUDA notes before code:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\gemm_translation_notes.md
```

The note records the exact GemLite `gemm_INT_kernel` contract:

```text
family=GEMM
cache_key=(1536, 27648, 3072, 128, 8, 101)
BLOCK_SIZE_M=64
BLOCK_SIZE_N=128
BLOCK_SIZE_K=128
GROUP_SIZE_M=8
A_load_order=2
num_warps=4
num_stages=1
W_q layout=[K / 8, N]
q_shift=(k % 8) * 1
scale/zero row=(k_tile / 128)
dequant=w=bit * scale + zero
```

The first Tensor Core CUDA path was added behind:

```text
SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM=1
```

Benchmark selection:

```text
sd-bonsai-gemlite-int1-kernel-bench.exe --kernel gemlite-tc-gemm
```

Implementation shape:

```text
CTA tile: 64x128 C
K tile: 128
warps: 4
grid for M=1536,N=27648: 24 x 216 = 5184 CTAs
shared A tile: 64x128 half
shared B tile: 128x128 half, dequantized from compact INT1 per CTA
WMMA: 16x16x16 fragments
```

The initial fp32 accumulator version needed an fp32 C staging tile and exceeded the current static shared-memory limit:

```text
ptxas: uses too much shared data (0x14000 bytes, 0xc000 max)
```

The built TC candidate therefore uses WMMA half accumulator fragments and stores fp16 output directly. This confirms Tensor Core-style execution, but it is not yet a faithful fp32-accumulation equivalent to the Triton `tl.dot` contract.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Isolated benchmark commands:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear1.weight --m 1536 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear1.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-shape-v2 --compare-current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear1.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-gemm --compare-current
```

Results:

```text
kernel             median_ms  mean_ms  correctness
current            92.074     91.818   reference
gemlite-shape-v2   24.417     24.253   corr=1.000000000 mean_abs=0.000000249 max_abs=0.001953125
gemlite-tc-gemm    15.634     15.395   corr=0.999995966 mean_abs=0.001177514 max_abs=0.023437500
Python GemLite      3.212      3.337   oracle
```

Speedups:

```text
TC vs old scalar:      5.89x median
TC vs best scalar/v2:  1.56x median
TC vs Python GemLite:  4.87x slower
```

Image timing was not run because the pass threshold required a 2x isolated win over the current best before full-image timing. The TC path is faster, but not enough.

Memory/occupancy reasoning:

```text
compact W_q bytes per dominant linear: 384 * 27648 ~= 10.1 MiB
scale bytes: 24 * 27648 * 4 ~= 2.5 MiB
zero bytes:  24 * 27648 * 4 ~= 2.5 MiB
output bytes per call: 1536 * 27648 * 2 ~= 81 MiB
TC shared memory used: A 16 KiB + B 32 KiB = 48 KiB
fp32 C staging would add 32 KiB and requires dynamic/shared-memory opt-in or a different store design
```

The likely remaining bottleneck is B dequant staging and shared-memory traffic. Each CTA dequantizes a 128x128 B tile into half shared memory, and that work is repeated for every M tile. GemLite/Triton fuses dequant and `tl.dot` in compiler-generated code with better register/shared-memory scheduling. The current WMMA port reaches Tensor Cores, but still materializes full B tiles in shared memory and uses half accumulation to fit the current shared-memory limit.

Next exact blocker:

```text
Implement a fp32-accumulation TC path without exceeding the static shared-memory limit, likely by using dynamic shared memory opt-in, inline mma.sync with register-resident C fragments, and a tighter B dequant staging layout. The next performance target is reducing B dequant/shared-memory staging overhead, not another scalar variant.
```

## Register-accum Tensor Core GEMM V2 pass

V1 Tensor Core verification:

```text
cuobjdump --dump-sass build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj
```

V1 contains Tensor Core instructions:

```text
HMMA.16816.F16
```

V2 was added behind:

```text
SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2=1
```

Benchmark selector:

```text
--kernel gemlite-tc-gemm-v2
```

V2 architecture:

```text
CTA C tile: 64x128
K group: 128
inner MMA step: 16
warps/CTA: 4
A shared tile: 64x16 half
B shared tile: 16x128 half
fp32 accumulator fragments stay in registers for all 3072 K
final store converts fp32 accumulator fragments to half accumulator fragments and stores fp16 output
```

V2 SASS confirms fp32 Tensor Core accumulation:

```text
HMMA.16816.F32
```

Resource usage from `cuobjdump --dump-resource-usage`:

```text
V1 TC: REG=174 SHARED=49152 LOCAL=0
V2 TC: REG=168 SHARED=6144  LOCAL=0
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Isolated benchmark, 10 measured runs:

```text
kernel                median_ms  mean_ms  correctness vs current
current scalar        89.985     90.379   reference
best scalar/shape     23.751     23.784   corr=1.000000000 mean_abs=0.000000249 max_abs=0.001953125
TC V1                 15.361     15.393   corr=0.999995966 mean_abs=0.001177514 max_abs=0.023437500
TC V2                 10.398     10.679   corr=1.000000000 mean_abs=0.000000290 max_abs=0.001953125
Python GemLite         3.212      3.337   oracle
```

V2 speedups:

```text
vs old scalar:       8.65x median
vs best scalar:      2.28x median
vs TC V1:            1.48x median
vs Python GemLite:   3.24x slower
```

V2 did not quite hit the sub-10 ms target, but it is the first fork kernel in the right architecture class: real Tensor Core fp32 accumulation with compact INT1 weights and no full fp16 expansion.

Because V2 was faster than TC V1 and more than 2x faster than the best scalar path, one 512 warmed timing was run:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-512
```

512 timing:

```text
full_mean_ms=7593.722
full_median_ms=7547.681
generate_mean_ms=7535.091
generate_median_ms=7490.552
denoise_mean_ms=7221.000
denoise_median_ms=7177.000
text_encode_mean_ms=115.333
vae_decode_mean_ms=194.667
png_save_mean_ms=58.631
elapsed_includes_model_load=false
```

512 image:

```text
path=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-512\bonsai_bench_run_00.png
sha256=D0BF977D011B6DE23F53B994B31322D3489CA93F1E6157A40F398806EE90D1C9
min=0 max=255 mean=121.933229 std=66.155181 unique_rgb=134488
quality=visually plausible
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Current 512 progression:

```text
baseline scalar denoise median:      13563 ms
shape kernel denoise median:          9941 ms
TC V2 denoise median:                 7177 ms
```

Remaining blocker:

```text
TC V2 still spends too much time staging/dequantizing B tiles. It now uses fp32 register accumulators and low shared memory, so the next likely parity step is a tile-major compact prepack for the dominant family or tighter fused dequant+mma scheduling, not more scalar work. The target remains Python GemLite's ~3.2 ms dominant-linear median.
```

## Tile-major B prepack and dequant-staging pass

This pass stayed on the isolated dominant linear benchmark first:

```text
linear: model.diffusion_model.single_blocks.0.linear1.weight
shape:  M=1536 K=3072 N=27648
pack:   W_q [K/8,N], group_size=128, elements_per_sample=8
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench sd-bonsai-gemlite-int1-bench --parallel 8
```

Benchmark log directory:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\tile-prepack-pass
```

TC V2 bottleneck estimate from diagnostic kernels:

```text
full TC V2:        median 10.507 ms
dequant/load only: median  4.818 ms
MMA-only:          median  7.459 ms
```

This shows raw B-side dequant/staging is a large cost, but not the only cost. The synthetic MMA-only path is already above GemLite's full 3.212 ms oracle timing, so closing the remaining gap probably requires lower-level scheduling than C++ WMMA with shared-memory B materialization.

Tile-major prepack layout:

```text
gate: SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK=1
variant: SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT={1,2,3}
tile: BK=128, BN=128
wq tile index: (k_group, n_tile, packed_byte_in_group, n_inner)
scale/zero tile index: (k_group, n_tile, n_inner)
storage: compact INT1 + f32 scale/zero only; no fp16 weight expansion
```

The runtime now allocates tile-major buffers only for the dominant `K=3072,N=27648` family used by `single_blocks.*.linear1`. Shape-only prepack increased compact device storage from 658.125 MiB to 961.875 MiB, so the overhead is 303.750 MiB. The earlier all-compatible-linear allocation was too broad and doubled storage to 1316.25 MiB; that was narrowed.

Isolated benchmark results, 10 measured runs:

```text
kernel                         median_ms  mean_ms  correctness vs current
current scalar                  89.882     90.387   reference
TC V2                           10.507     10.638   corr=1.000000000 mean_abs=0.000000290 max_abs=0.001953125
TC V2 dequant/load only          4.818      4.920   diagnostic checksum output only
TC V2 MMA-only                   7.459      7.665   diagnostic synthetic-B output only
prepack-v1 Wq tile only         13.398     13.471   corr=1.000000000 mean_abs=0.000000290 max_abs=0.001953125
prepack-v2 Wq+metadata tile     10.513     10.725   corr=1.000000000 mean_abs=0.000000290 max_abs=0.001953125
prepack-v3 padded B smem         7.008      7.213   corr=1.000000000 mean_abs=0.000000290 max_abs=0.001953125
Python GemLite oracle            3.212      3.337   reference
```

Best isolated result:

```text
TC V2 -> prepack-v3 median speedup: 1.50x
old scalar -> prepack-v3 median speedup: 12.82x
prepack-v3 remains 2.18x slower than Python GemLite
nan=0 inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Because prepack-v3 beat TC V2 by more than 25%, one 512 warmed image timing was run:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT='3'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-prepack-v3-shapeonly-512
```

512 timing:

```text
full_mean_ms=7923.317
full_median_ms=7928.742
generate_mean_ms=7867.179
generate_median_ms=7871.595
denoise_mean_ms=7549.667
denoise_median_ms=7560.000
text_encode_mean_ms=117.000
vae_decode_mean_ms=196.333
png_save_mean_ms=56.138
elapsed_includes_model_load=false
```

512 image:

```text
path=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-prepack-v3-shapeonly-512\bonsai_bench_run_00.png
sha256=D0BF977D011B6DE23F53B994B31322D3489CA93F1E6157A40F398806EE90D1C9
min=0 max=255 mean=121.933229 std=66.155181 unique_rgb=134488
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

The full-image timing did not improve versus the prior TC V2 run (`denoise_median_ms=7177`). The isolated dominant-linear win is real, but at 512 the total denoise path is now limited by more than this one shape, possible prepack cache/occupancy side effects, and remaining ggml graph/runtime overhead. The exact next bottleneck should be found with a filtered profiler that reports only the major Bonsai INT1 families after prepack-v3, not with another full unbounded profile.

Remaining technical blocker:

```text
prepack-v3 reduces B shared-memory bank/layout cost, but C++ WMMA still stages dequantized B as half shared-memory tiles. Reaching GemLite's ~3.2 ms likely requires a lower-level inline mma/ldmatrix path that fuses INT1 unpack/dequant into register fragments or an ldmatrix-compatible shared layout, plus work on the next dominant linear families shown by a filtered profile.
```

## Family-level INT1 bottleneck map after TC V2/prepack

This pass added a separate low-overhead family profiler instead of reusing the old expensive per-linear profiler.

Profiler controls:

```text
SDCPP_PROFILE_BONSAI_INT1_FAMILIES=1
SDCPP_PROFILE_BONSAI_INT1_FILTER=<substring>
SDCPP_PROFILE_BONSAI_INT1_MAX_CALLS=<N>
SDCPP_PROFILE_BONSAI_INT1_RECORD_EVENTS=0/1
SDCPP_PROFILE_BONSAI_INT1_SHAPE_CENSUS_JSON=<path>
```

When family profiling is off, the custom-op path does not create CUDA events. When profiling is on with `RECORD_EVENTS=0`, it only records a cheap shape/family census. Event timing is explicit and was used only for the bounded single-run 512 passes below.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Profile output:

```text
build-bonsai-int1\bonsai-profile\tc-v2-512.stdout.log
build-bonsai-int1\bonsai-profile\tc-v2-512.stderr.log
build-bonsai-int1\bonsai-profile\prepack-v3-512.stdout.log
build-bonsai-int1\bonsai-profile\prepack-v3-512.stderr.log
build-bonsai-int1\bonsai-profile\shape_census_512_tc_v2.json
build-bonsai-int1\bonsai-profile\shape_census_512_prepack_v3.json
```

512 profile command shape:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_PROFILE_BONSAI_INT1_FAMILIES='1'
$env:SDCPP_PROFILE_BONSAI_INT1_RECORD_EVENTS='1'
$env:SDCPP_PROFILE_BONSAI_INT1_MAX_CALLS='1000'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 0 --runs 1
```

Pass A used `SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2=1`.
Pass B used `SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK=1` and `SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT=3`.

Run-level timing:

```text
pass          device_mb  denoise_ms  measured_INT1_ms  full_image_ms
TC V2         658.125    8084        7417.657          8563.606
prepack-v3    961.875    7301        6657.340          7773.799
```

Top families, TC V2:

```text
family                       calls  total_ms   median_ms  pct     shape              kernel
single_blocks.linear2        80     3517.920   42.824     47.43   1536x12288x3072   native
double_blocks.img_mlp.0      20      922.674   45.446     12.44   1024x3072x18432   native
single_blocks.linear1        80      913.393   11.426     12.31   1536x3072x27648   tc_v2
double_blocks.img_mlp.2      20      473.153   24.070      6.38   1024x9216x3072    native
double_blocks.txt_mlp.0      20      456.684   22.984      6.16   512x3072x18432    native
double_blocks.img_attn.qkv   60      456.660    7.620      6.16   1024x3072x3072    native_strided
```

Top families, prepack-v3:

```text
family                       calls  total_ms   median_ms  pct     shape              kernel
single_blocks.linear2        80     3348.188   41.729     50.29   1536x12288x3072   native
double_blocks.img_mlp.0      20      832.805   41.346     12.51   1024x3072x18432   native
single_blocks.linear1        80      591.764    7.303      8.89   1536x3072x27648   tc_v2_prepack_v3
double_blocks.img_attn.qkv   60      420.123    6.798      6.31   1024x3072x3072    native_strided
double_blocks.txt_mlp.0      20      417.410   20.890      6.27   512x3072x18432    native
double_blocks.img_mlp.2      20      413.340   20.559      6.21   1024x9216x3072    native
```

Shape census:

```text
shape                  calls  families                                           kernels
M=1024,K=3072,N=3072   80     double_blocks.img_attn.qkv,img_attn.proj           native_strided,native
M=1536,K=12288,N=3072  80     single_blocks.linear2                              native
M=1536,K=3072,N=27648  80     single_blocks.linear1                              tc_v2 or tc_v2_prepack_v3
M=512,K=3072,N=3072    80     double_blocks.txt_attn.qkv,txt_attn.proj           native_strided,native
M=1024,K=3072,N=18432  20     double_blocks.img_mlp.0                            native
M=1024,K=9216,N=3072   20     double_blocks.img_mlp.2                            native
M=512,K=3072,N=18432   20     double_blocks.txt_mlp.0                            native
M=512,K=9216,N=3072    20     double_blocks.txt_mlp.2                            native
```

Isolated top-family benches:

```text
family                  shape              calls/4step  current_ms  best_ms  estimated_family_ms  correctness
single_blocks.linear2   1536x12288x3072    80           42.998      42.998   ~3439.8              finite, native only
double_blocks.img_mlp.0 1024x3072x18432    20           41.507      41.507   ~830.1               finite, native only
single_blocks.linear1   1536x3072x27648    80           91.952       7.008   ~560.6               corr=1.0 mean_abs=0.000000290
```

Why prepack-v3 did not clearly improve warmed full denoise:

```text
prepack-v3 does select in the full 512 path.
It reduces single_blocks.linear1 measured family time from 913.393 ms to 591.764 ms in the bounded profiled run.
But single_blocks.linear1 is no longer the dominant family: after prepack it is only 8.89% of measured INT1 time.
single_blocks.linear2 alone is now ~50% of measured INT1 time and still uses the native scalar INT1 kernel.
The earlier warmed median regression is therefore not a reason to keep optimizing linear1 blindly; the isolated win is hidden by larger native families and possible prepack memory/cache side effects.
```

Chosen next target:

```text
Stop optimizing single_blocks.linear1 for now.
Next kernel target should be single_blocks.linear2:
  shape M=1536 K=12288 N=3072
  calls=80 per 512x512 4-step run
  measured family time ~3348 ms with prepack-v3 run
  isolated native median ~42.998 ms
  about 50% of measured INT1 time
```

Recommended next implementation:

```text
Port the Tensor Core register-accum architecture to the single_blocks.linear2 shape family.
This is a different aspect ratio from linear1: K is 4x larger and N is 9x smaller.
Start with an isolated kernel bench for M=1536,K=12288,N=3072 using real Bonsai Wq/scales/zeros.
Use a GemLite-source/cache check for this shape before coding; do not assume the linear1 BM=64,BN=128,BK=128 mapping remains optimal.
Only after isolated linear2 improves materially should full 512 timing be rerun.
Keep prepack-v3 off by default until a later combined timing pass proves it helps end-to-end.
```

## single_blocks.linear2 Tensor Core pass

Target:

```text
family: single_blocks.linear2
internal: model.diffusion_model.single_blocks.0.linear2.weight
HF: single_transformer_blocks.0.attn.to_out
shape: M=1536 K=12288 N=3072
W_q layout: [K/8,N] = [1536,3072]
scales/zeros: [K/128,N] = [96,3072]
math: w = bit * scale + zero
```

GemLite oracle command:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear single_transformer_blocks.0.attn.to_out --m 1536 --warmup 1 --runs 5 --out F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\single_blocks_linear2_contract.json
```

GemLite contract:

```text
matmul_type=GEMM
cache_key=(1536, 3072, 12288, 128, 8, 101)
BLOCK_SIZE_M=64
BLOCK_SIZE_N=32
BLOCK_SIZE_K=128
GROUP_SIZE_M=8
A_load_order=0
num_warps=4
num_stages=3
split_k=false
Python GemLite median=1.564480 ms
```

Implementation:

```text
env gate: SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2=1
bench selector: --kernel gemlite-tc-linear2
CTA tile: 64x32
K group: 128
inner MMA step: 16
warps/CTA: 4
grid for 1536x3072 output: 24 x 96 = 2304 CTAs
accumulation: fp32 WMMA fragments in registers
shared memory: A[64,16] half + B[16,32] half
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench sd-bonsai-gemlite-int1-bench --parallel 8
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2 --compare-current
```

Results:

```text
kernel            median_ms  mean_ms  min_ms  max_ms
native current    42.005     42.733   41.861  46.054
TC linear2         5.230      5.364    5.151   5.849
Python GemLite     1.564      1.565    1.558   1.576
```

Correctness:

```text
corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

512 warmed timing with TC linear1 + TC linear2:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-512
```

512 timing:

```text
full_mean_ms=4790.680
full_median_ms=4791.565
generate_mean_ms=4735.615
generate_median_ms=4735.466
denoise_mean_ms=4427.333
denoise_median_ms=4424.000
text_encode_mean_ms=115.333
vae_decode_mean_ms=189.333
png_save_mean_ms=55.065
elapsed_includes_model_load=false
```

512 output:

```text
path=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-512\bonsai_bench_run_00.png
sha256=0DF7ED5250FFD30F4D664FD35A02BEB5950492BCB91CDA69AE843D29F19210D9
min=0 max=255 mean=121.891631 std=66.144950 unique_rgb=134544
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Updated family map after TC linear2:

```text
family                       calls  total_ms  median_ms  pct     shape              kernel
single_blocks.linear1        80     863.559   10.495     22.01   1536x3072x27648   tc_v2
double_blocks.img_mlp.0      20     810.639   40.300     20.66   1024x3072x18432   native
single_blocks.linear2        80     417.933    5.056     10.65   1536x12288x3072   tc_linear2
double_blocks.txt_mlp.0      20     406.951   20.531     10.37   512x3072x18432    native
double_blocks.img_attn.qkv   60     405.841    6.531     10.34   1024x3072x3072    native_strided
double_blocks.img_mlp.2      20     405.684   20.387     10.34   1024x9216x3072    native
```

Progression:

```text
baseline scalar denoise median:      13563 ms
shape kernel denoise median:          9941 ms
TC V2 linear1 denoise median:         7177 ms
TC V2 linear1 + TC linear2 median:    4424 ms
```

Split-K was not attempted because direct TC linear2 hit the primary target (`<10 ms`) and the full-denoise improvement was material.

Next target:

```text
double_blocks.img_mlp.0 is now the largest native scalar family with:
  shape M=1024 K=3072 N=18432
  calls=20 per 512x512 4-step run
  measured family time ~811 ms
  isolated native median ~41.5 ms

It has the same K=3072 and large-N style as linear1 but smaller M and N. Extract the GemLite contract for transformer_blocks.0.ff.linear_in and port a dedicated TC path for that shape next. Reconsider prepack-v3 for linear1 only after img_mlp.0 and related double-block MLP families are off the native scalar path.
```

## double_blocks.img_mlp.0 Tensor Core pass

Target:

```text
family: double_blocks.img_mlp.0
internal: model.diffusion_model.double_blocks.0.img_mlp.0.weight
HF: transformer_blocks.0.ff.linear_in
shape: M=1024 K=3072 N=18432
W_q layout: [K/8,N] = [384,18432]
scales/zeros: [K/128,N] = [24,18432]
math: w = bit * scale + zero
```

GemLite oracle command:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear transformer_blocks.0.ff.linear_in --m 1024 --warmup 1 --runs 5 --out F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\double_blocks_img_mlp0_contract.json
```

GemLite contract:

```text
matmul_type=GEMM
cache_key=(1024, 18432, 3072, 128, 8, 101)
BLOCK_SIZE_M=64
BLOCK_SIZE_N=128
BLOCK_SIZE_K=128
GROUP_SIZE_M=8
A_load_order=2
num_warps=4
num_stages=1
split_k=false
Python GemLite median=1.534784 ms
```

Implementation:

```text
env gate: SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0=1
bench selector: --kernel gemlite-tc-img-mlp0
kernel: reuses the register-accum TC V2 architecture for K=3072,N=18432
CTA tile: 64x128
K group: 128
inner MMA step: 16
warps/CTA: 4
accumulation: fp32 WMMA fragments in registers
shared memory: A[64,16] half + B[16,128] half
```

The shape gate also covers the matching text MLP input shape (`M=512,K=3072,N=18432`), so `double_blocks.txt_mlp.0` moves off the native scalar path in full denoise as well. This is intentional for the shared shape family, but the isolated correctness benchmark below was run on the requested image MLP representative.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench sd-bonsai-gemlite-int1-bench --parallel 8
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_mlp.0.weight --m 1024 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_mlp.0.weight --m 1024 --warmup 1 --runs 10 --kernel gemlite-tc-img-mlp0 --compare-current
```

Results:

```text
kernel            median_ms  mean_ms  min_ms  max_ms
native current    40.037     40.803   39.873  42.373
TC img_mlp0        5.005      5.183    4.969   5.680
Python GemLite     1.535      1.647    n/a     n/a
```

Correctness:

```text
corr=1.000000000
cos=1.000000000
mean_abs=0.000000292
max_abs=0.001953125
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

No prepack variant was added because direct TC hit the primary `<10 ms` target.

512 warmed timing with TC linear1 + TC linear2 + TC img_mlp0:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-imgmlp0-512
```

512 timing:

```text
full_mean_ms=3807.340
full_median_ms=3767.180
generate_mean_ms=3751.407
generate_median_ms=3709.337
denoise_mean_ms=3439.333
denoise_median_ms=3398.000
text_encode_mean_ms=116.333
vae_decode_mean_ms=192.000
png_save_mean_ms=55.933
elapsed_includes_model_load=false
```

512 output:

```text
path=F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-imgmlp0-512\bonsai_bench_run_00.png
sha256=3C4C500C84FF62A1B4FB0CB166A98BD9B9F38FC16092C5DEDF5B1097BA539ED1
min=0 max=255 mean=121.889810 std=66.118023 unique_rgb=134421
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Updated family map after TC img_mlp0:

```text
family                       calls  total_ms  median_ms  pct     shape              kernel
single_blocks.linear1        80     892.112   10.994     30.52   1536x3072x27648   tc_v2
single_blocks.linear2        80     430.408    5.295     14.72   1536x12288x3072   tc_linear2
double_blocks.img_attn.qkv   60     418.633    6.875     14.32   1024x3072x3072    native_strided
double_blocks.img_mlp.2      20     410.464   20.422     14.04   1024x9216x3072    native
double_blocks.txt_attn.qkv   60     208.919    3.278      7.15   512x3072x3072     native_strided
double_blocks.txt_mlp.2      20     204.633   10.164      7.00   512x9216x3072     native
double_blocks.img_attn.proj  20     140.353    6.832      4.80   1024x3072x3072    native
double_blocks.img_mlp.0      20      97.272    4.718      3.33   1024x3072x18432   tc_img_mlp0
double_blocks.txt_mlp.0      20      52.631    2.504      1.80   512x3072x18432    tc_img_mlp0
```

Progression:

```text
baseline scalar denoise median:              13563 ms
shape kernel denoise median:                  9941 ms
TC V2 linear1 denoise median:                 7177 ms
TC V2 linear1 + TC linear2 median:            4424 ms
TC linear1 + linear2 + img_mlp0 median:       3398 ms
```

Next target:

```text
The next measured native bottleneck is double_blocks.img_mlp.2:
  shape M=1024 K=9216 N=3072
  calls=20 per 512x512 4-step run
  measured family time ~410 ms
  median per call ~20.4 ms

double_blocks.img_attn.qkv is comparable at ~419 ms total but is a 3x strided Q/K/V path. The cleaner next implementation target is img_mlp.2 because it is a normal linear with one output buffer and should resemble the successful linear2 K-large/N-small Tensor Core path. Extract GemLite config for transformer_blocks.0.ff.linear_out before coding.
```

## double_blocks.img_mlp.2 Tensor Core pass

Phase 0 linear1 prepack sanity:

```text
settings: TC linear1 + TC linear2 + TC img_mlp0 + linear1 prepack-v3
run type: 512x512, 4 steps, warmup=0, runs=1, family profiler enabled
denoise_ms=3181
full_image_ms=3654.783
single_blocks.linear1 total_ms=573.000 median_ms=6.997 kernel=tc_v2_prepack_v3 uses_prepack=true
```

Decision: keep `SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK=1` with variant `3` enabled for combined timing. It selected in the full path and reduced measured linear1 family time versus the non-prepack TC V2 map.

GemLite contract for `double_blocks.img_mlp.2`:

```json
{
  "internal": "model.diffusion_model.double_blocks.0.img_mlp.2.weight",
  "hf": "transformer_blocks.0.ff.linear_out",
  "shape": [1024, 9216, 3072],
  "matmul_type": "GEMM",
  "cache_key": "(1024, 3072, 9216, 128, 8, 101)",
  "BLOCK_SIZE_M": 64,
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 128,
  "GROUP_SIZE_M": 8,
  "A_load_order": 2,
  "num_warps": 4,
  "num_stages": 1,
  "python_gemlite_median_ms": 0.922944
}
```

The config matches the existing register-accum TC V2 tile shape, so the implementation adds a narrow gate and selector instead of a new kernel body:

```text
env gate: SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2=1
bench selector: --kernel gemlite-tc-img-mlp2
kernel: bonsai_int1_linear_gemlite_tc_gemm_v2_kernel
shape gate: K=9216,N=3072,M%64==0,N%128==0,K%128==0
```

This also covers the matching text MLP output shape (`M=512,K=9216,N=3072`) in the full graph.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench sd-bonsai-gemlite-int1-bench --parallel 8
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_mlp.2.weight --m 1024 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_mlp.2.weight --m 1024 --warmup 1 --runs 10 --kernel gemlite-tc-img-mlp2 --compare-current
```

Results:

```text
kernel             median_ms  mean_ms  min_ms  max_ms
native current      21.491     30.208  20.019  59.181
TC img_mlp2          8.374      8.383   8.098   8.854
Python GemLite       0.923      0.912   n/a     n/a
```

Correctness:

```text
corr=0.999999999
cos=0.999999999
mean_abs=0.000001687
max_abs=0.003906250
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

No split-K or prepack was attempted for img_mlp.2 because direct TC hit the primary `<10 ms` target.

512 warmed timing with TC linear1 prepack-v3 + TC linear2 + TC img_mlp0 + TC img_mlp2:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT='3'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-imgmlp0-imgmlp2-prepack-512
```

512 timing:

```text
full_mean_ms=2997.383
full_median_ms=2997.656
generate_mean_ms=2940.900
generate_median_ms=2944.221
denoise_mean_ms=2617.000
denoise_median_ms=2617.000
text_encode_mean_ms=120.000
vae_decode_mean_ms=199.667
png_save_mean_ms=56.484
elapsed_includes_model_load=false
```

Output hashes:

```text
run_00 sha256=6FA56E6DFA3B9AB245A1A5AB2A4FBDEF75AD914560848D1BC0B9BA4044E3B283
run_01 sha256=FDAF495B42C51761F68E28826FD3A09A161256A5077FCC493225DA884B906695
run_02 sha256=F1C91388C2A5E2042F3A4F348310CD7CA3F58367FD9BB8900012DDEC86F8450A
```

Updated family map after TC img_mlp2:

```text
family                       calls  total_ms  median_ms  pct     shape              kernel
single_blocks.linear1        80     594.188    7.329     28.03   1536x3072x27648   tc_v2_prepack_v3
single_blocks.linear2        80     431.343    5.169     20.35   1536x12288x3072   tc_linear2
double_blocks.img_attn.qkv   60     426.177    6.878     20.11   1024x3072x3072    native_strided
double_blocks.txt_attn.qkv   60     213.699    3.278     10.08   512x3072x3072     native_strided
double_blocks.img_attn.proj  20     144.183    7.225      6.80   1024x3072x3072    native
double_blocks.img_mlp.0      20     100.523    4.728      4.74   1024x3072x18432   tc_img_mlp0
double_blocks.txt_attn.proj  20      71.258    3.251      3.36   512x3072x3072     native
double_blocks.img_mlp.2      20      53.415    2.623      2.52   1024x9216x3072    tc_img_mlp2
double_blocks.txt_mlp.0      20      52.373    2.501      2.47   512x3072x18432   tc_img_mlp0
double_blocks.txt_mlp.2      20      32.562    1.514      1.54   512x9216x3072    tc_img_mlp2
```

Progression:

```text
baseline scalar denoise median:                           13563 ms
shape kernel denoise median:                               9941 ms
TC V2 linear1 denoise median:                              7177 ms
TC V2 linear1 + TC linear2 median:                         4424 ms
TC linear1 + linear2 + img_mlp0 median:                    3398 ms
TC linear1 prepack + linear2 + img_mlp0 + img_mlp2 median: 2617 ms
```

Next target:

```text
The clean MLP output target is now solved for both image and text streams. The next measured wall is split between:
  single_blocks.linear1: still largest total at ~594 ms, but already TC prepack and harder to close further without lower-level dequant/mma fusion.
  single_blocks.linear2: ~431 ms, already TC linear2.
  double_blocks.img_attn.qkv: ~426 ms, native strided QKV.

The next prompt-worthy target is the QKV/proj group: either implement a fused Tensor Core QKV path for double_blocks.*_attn.qkv or a normal TC path for double_blocks.img_attn.proj / txt_attn.proj. QKV has the larger aggregate contribution, but it is a strided 3-output path and needs a dedicated design rather than reusing the normal-linear TC wrapper.
```

## double_blocks attention QKV Tensor Core pass

The virtual Bonsai QKV linears are stored as separate GemLite q/k/v tensors but consumed by the Flux path as one combined `[M, 3 * N_each]` tensor. The existing correct order is preserved:

```text
q slice: offset 0
k slice: offset N_each
v slice: offset 2 * N_each
```

Combined contract artifact:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\double_blocks_qkv_contract.json
```

Image QKV contracts:

```text
internal virtual: model.diffusion_model.double_blocks.0.img_attn.qkv.weight
hf q/k/v: transformer_blocks.0.attn.to_q / to_k / to_v
shape per linear: M=1024,K=3072,N=3072
matmul family: GEMM
cache key: (1024, 3072, 3072, 128, 8, 101)
config: BM=64 BN=128 BK=128 GROUP_SIZE_M=8 A_load_order=2 warps=4 stages=1
Python GemLite medians:
  q=0.413632 ms
  k=0.412864 ms
  v=0.411552 ms
```

Text QKV contracts:

```text
internal virtual: model.diffusion_model.double_blocks.0.txt_attn.qkv.weight
hf q/k/v: transformer_blocks.0.attn.add_q_proj / add_k_proj / add_v_proj
shape per linear: M=512,K=3072,N=3072
matmul family: GEMM
cache key: (512, 3072, 3072, 128, 8, 101)
config: BM=64 BN=64 BK=64 GROUP_SIZE_M=8 A_load_order=0 warps=4 stages=4
Python GemLite medians:
  q=0.290816 ms
  k=0.286752 ms
  v=0.300032 ms
```

Implementation:

```text
env gate: SDCPP_BONSAI_INT1_GEMLITE_TC_QKV=1
bench selectors:
  --kernel gemlite-tc-img-qkv
  --kernel gemlite-tc-txt-qkv
launch design: three TC launches inside the QKV custom op, q/k/v separately
kernel: bonsai_int1_linear_gemlite_tc_gemm_v2_strided_out_kernel
output layout: direct strided writes into combined [M,9216] q/k/v slices
```

This is option A from the design note: three backend-stream TC launches replace the old three native strided kernels. No full FP16 expansion or full B FP16 expansion is used. The current implementation uses the proven `BM=64,BN=128,BK=128` TC writer for both image and text QKV. The text GemLite cache prefers `BN=64,BK=64`, so there is still room for a text-specific variant later if QKV becomes hot again.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench sd-bonsai-gemlite-int1-bench --parallel 8
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_attn.qkv.weight --m 1024 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_attn.qkv.weight --m 1024 --warmup 1 --runs 10 --kernel gemlite-tc-img-qkv --compare-current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.txt_attn.qkv.weight --m 512 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.txt_attn.qkv.weight --m 512 --warmup 1 --runs 10 --kernel gemlite-tc-txt-qkv --compare-current
```

Results:

```text
target    kernel        median_ms  mean_ms  min_ms  max_ms
img_qkv   native         37.235     50.112  20.504  110.392
img_qkv   TC             15.694     15.749  12.759   18.982
txt_qkv   native         40.527     43.805  33.632   66.925
txt_qkv   TC             10.420     10.367   9.787   10.684
```

Correctness:

```text
img_qkv corr=1.000000000 cos=1.000000000 mean_abs=0.000000297 max_abs=0.001953125 nan=0 inf=0
txt_qkv corr=1.000000000 cos=1.000000000 mean_abs=0.000000280 max_abs=0.001953125 nan=0 inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

The isolated QKV bench uses three CUDA launches per virtual QKV custom op, one each for q/k/v.

512 warmed timing with TC linear1 prepack-v3 + TC linear2 + TC img_mlp0 + TC img_mlp2 + TC QKV:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_QKV='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT='3'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-mlps-qkv-prepack-512
```

512 timing:

```text
full_mean_ms=2390.921
full_median_ms=2397.044
generate_mean_ms=2336.298
generate_median_ms=2342.010
denoise_mean_ms=2023.333
denoise_median_ms=2020.000
text_encode_mean_ms=120.000
vae_decode_mean_ms=188.667
png_save_mean_ms=54.622
elapsed_includes_model_load=false
```

Output hashes:

```text
run_00 sha256=7C7E7D314AC6EA689A670E65C248E1CC05E7D846380AFA70CB60CED7A411B6E8
run_01 sha256=910975B42F2F027F38C65B93D055A5A3D1D4196DCD89BAD4C407C600FD4396CF
run_02 sha256=76064E1A0F5CB2B95881B0E5863FB5B65830A8B841C15273714FB3007BC720A6
```

Updated family map after TC QKV:

```text
family                       calls  total_ms  median_ms  pct     shape              kernel
single_blocks.linear1        80     578.741    7.227     37.66   1536x3072x27648   tc_v2_prepack_v3
single_blocks.linear2        80     428.116    5.306     27.86   1536x12288x3072   tc_linear2
double_blocks.img_attn.proj  20     136.549    6.813      8.89   1024x3072x3072    native
double_blocks.img_mlp.0      20      96.971    4.730      6.31   1024x3072x18432   tc_img_mlp0
double_blocks.txt_attn.proj  20      70.210    3.278      4.57   512x3072x3072     native
double_blocks.img_attn.qkv   60      56.849    0.887      3.70   1024x3072x3072    tc_qkv
double_blocks.img_mlp.2      20      53.861    2.626      3.50   1024x9216x3072    tc_img_mlp2
double_blocks.txt_mlp.0      20      51.284    2.495      3.34   512x3072x18432   tc_img_mlp0
double_blocks.txt_attn.qkv   60      32.843    0.517      2.14   512x3072x3072     tc_qkv
double_blocks.txt_mlp.2      20      31.281    1.519      2.04   512x9216x3072    tc_img_mlp2
```

Progression:

```text
baseline scalar denoise median:                           13563 ms
shape kernel denoise median:                               9941 ms
TC V2 linear1 denoise median:                              7177 ms
TC V2 linear1 + TC linear2 median:                         4424 ms
TC linear1 + linear2 + img_mlp0 median:                    3398 ms
TC linear1 prepack + linear2 + img_mlp0 + img_mlp2 median: 2617 ms
TC linear1 prepack + linear2 + MLPs + QKV median:          2020 ms
```

Next target:

```text
QKV is no longer a top wall. The remaining measured INT1 cost is led by:
  single_blocks.linear1: ~579 ms, already TC prepack-v3
  single_blocks.linear2: ~428 ms, already TC linear2
  double_blocks.img_attn.proj: ~137 ms, native
  double_blocks.txt_attn.proj: ~70 ms, native

The next clean pass is attention projection. It is a normal non-strided shape (K=3072,N=3072), so it can likely reuse the same strided-QKV TC primitive without the q/k/v concat concerns. After that, further gains require revisiting single_blocks.linear1/linear2 with lower-level GemLite-style dequant/mma fusion rather than more wrapper-level shape gates.
```

## Source-backed GemLite vs SDCPP kernel diff

After the QKV pass, the remaining Prism/GemLite parity gap is dominated by the two already-optimized single block families:

```text
single_blocks.linear1: M=1536 K=3072  N=27648, SDCPP best ~7 ms, Python GemLite 3.120 ms
single_blocks.linear2: M=1536 K=12288 N=3072,  SDCPP best ~5.2 ms, Python GemLite 1.584 ms
```

Artifact extraction:

```powershell
$env:TRITON_CACHE_DIR='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear1_gemlite_artifacts'
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear single_transformer_blocks.0.attn.to_qkv_mlp_proj --m 1536 --warmup 1 --runs 1 --out build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear1_gemlite_artifacts\contract.json

$env:TRITON_CACHE_DIR='F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_gemlite_artifacts'
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear single_transformer_blocks.0.attn.to_out --m 1536 --warmup 1 --runs 1 --out build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_gemlite_artifacts\contract.json

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cuobjdump.exe --dump-resource-usage build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cuobjdump.exe --dump-sass build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvdisasm.exe build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_gemlite_artifacts\3X4D3RWOSK3VTOHYHNJRW4PYLFBF7475N4PWQMGR5XDOGB3X3TYQ\gemm_INT_kernel.cubin
```

Generated comparison note:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\gemlite_vs_sdcpp_kernel_diff.md
```

GemLite selected configs:

```text
linear1: GEMM BM=64 BN=128 BK=128 GROUP_SIZE_M=8 A_load_order=0 num_warps=4 num_stages=1 shared=49152
linear2: GEMM BM=64 BN=32  BK=128 GROUP_SIZE_M=8 A_load_order=0 num_warps=4 num_stages=3 shared=49664
```

Source-backed findings:

```text
linear1:
  GemLite uses a large 49 KiB shared-memory schedule with vectorized global loads and HMMA.
  SDCPP prepack-v3 uses only 6.4 KiB shared, 168 registers, and repeatedly dequants/stages 16x128 B subtiles.
  The gap is B-side dequant/staging/register pressure, not missing Tensor Cores or wrong tile dimensions.

linear2:
  GemLite uses a 3-stage async-copy pipeline; PTX contains cp.async.cg.shared.global and cp.async.commit_group.
  SDCPP tc_linear2 uses synchronous load/dequant, __syncthreads, WMMA, __syncthreads, with 3 KiB shared.
  The gap is specifically missing cp.async / multi-stage shared-memory pipelining.
```

No kernel patch was applied in this pass. The evidence points to a non-trivial async/shared-memory pipeline rewrite, not a safe one-line correction. The next source-backed implementation target should be:

```text
1. single_blocks.linear2: implement GemLite-like BM=64 BN=32 BK=128, num_stages=3 async-copy pipeline.
2. Then single_blocks.linear1: replace compact streamed B staging with a larger Triton-like vectorized shared-memory schedule or equivalent inline-MMA/dequant fusion.
```

## Source-backed linear2 async pipeline pass

Pipeline note:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_async_pipeline_notes.md
```

GemLite artifact findings:

```text
target: single_blocks.linear2 / single_transformer_blocks.0.attn.to_out
shape: M=1536 K=12288 N=3072
GemLite config: GEMM BM=64 BN=32 BK=128 GROUP_SIZE_M=8 A_load_order=0 num_warps=4 num_stages=3
GemLite artifact shared memory: 49664 bytes
GemLite PTX: cp.async.cg.shared.global, cp.async.commit_group, cp.async.wait_group, mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
Python GemLite reference median: 1.584 ms
```

Implementation:

```text
gate: SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC=1
bench selector: --kernel gemlite-tc-linear2-async
scope: only K=12288,N=3072,M multiple of 64, group_size=128 path
design: separate opt-in kernel with 64x32 CTA tile, fp32 WMMA accumulators, exact bit*scale+zero B dequant, and a 3-slot cp.async pipeline for A subtiles
```

This is a bounded async translation step. It does not replace the full Triton/GemLite 49 KiB shared schedule for packed-B/meta staging, but it verifies the SDCPP path can launch a linear2 kernel with cp.async and HMMA on the ggml backend stream.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench --parallel 8
```

Executable hashes:

```text
sd-bonsai-gemlite-int1-kernel-bench.exe SHA256=07C8D9DEA993669B4A0B8E956DDA3D87692A08667D0548BB62EB708F39947C2D
sd-bonsai-gemlite-int1-bench.exe        SHA256=4B5FFDF8792C1ABE29A0D2167B82C66B9D1DA07B42010345EFB91D96A9CA2E8B
```

SDCPP artifact extraction:

```powershell
cuobjdump --dump-ptx build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_async_artifacts\bonsai_gemlite_int1_cuda.ptx
cuobjdump --dump-sass build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_async_artifacts\bonsai_gemlite_int1_cuda.sass
cuobjdump --dump-resource-usage build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_async_artifacts\resource_usage.txt
```

SASS/PTX verification:

```text
PTX cp.async count: 6
PTX wait count: 2
SASS LDGSTS count: 2
SASS HMMA count: 932
async kernel resource usage: REG=59 SHARED=7168 LOCAL=0
```

Isolated linear2 benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2

$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-async --compare-current
```

Results:

```text
current TC linear2 median_ms=16.167 min_ms=5.365 max_ms=16.925
async linear2 median_ms=3.656 min_ms=3.375 max_ms=4.398
Python GemLite reference median_ms=1.584

correctness vs native baseline:
corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

512 warmed timing was run because the isolated async kernel improved by more than 20%:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT='3'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_QKV='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\linear2-async-512
```

512 timing:

```text
model_load_ms=4740.1
warmup denoise_ms=2019
full_mean_ms=2298.421
full_median_ms=2295.887
generate_mean_ms=2243.038
generate_median_ms=2238.756
denoise_mean_ms=1925.000
denoise_median_ms=1927.000
text_encode_mean_ms=121.667
vae_decode_mean_ms=192.667
png_save_mean_ms=55.383
elapsed_includes_model_load=false
```

512 outputs:

```text
build-bonsai-int1\bonsai-baseline\linear2-async-512\bonsai_bench_run_00.png
sha256=7C7E7D314AC6EA689A670E65C248E1CC05E7D846380AFA70CB60CED7A411B6E8 min=0 max=255 mean=121.930691 std=66.170146 unique_rgb=134623

build-bonsai-int1\bonsai-baseline\linear2-async-512\bonsai_bench_run_01.png
sha256=910975B42F2F027F38C65B93D055A5A3D1D4196DCD89BAD4C407C600FD4396CF min=0 max=255 mean=117.919955 std=57.356189 unique_rgb=122126

build-bonsai-int1\bonsai-baseline\linear2-async-512\bonsai_bench_run_02.png
sha256=76064E1A0F5CB2B95881B0E5863FB5B65830A8B841C15273714FB3007BC720A6 min=0 max=255 mean=109.587111 std=59.490665 unique_rgb=132711
```

Progression:

```text
TC linear1 prepack + linear2 + MLPs + QKV median:       2020 ms
same stack + linear2 async median:                      1927 ms
```

Remaining blocker:

```text
The linear2 async path is correct and faster, but it is still slower than Python GemLite (3.656 ms vs 1.584 ms isolated).
The source-backed remaining gap is the full GemLite shared-memory schedule: packed-B/meta async staging plus dequant/repack overlap in the large ~49 KiB shared region. The implemented pass only pipelines A subtiles and keeps B dequant synchronous.
Next source-backed fix should clone more of GemLite's packed-B/meta shared layout for linear2, or move to the lower-level inline MMA/dequant fusion path if the B-side schedule remains the dominant cost.
```

## linear2 packed-B async pipeline pass

Artifact-backed pipeline note:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_async_b_pipeline_notes.md
```

GemLite artifact finding:

```text
GemLite linear2 does not async-copy A only.
TTGIR shows async global-to-shared copies for:
  A:      2 x 64 x 128 x f16
  B:      2 x 128 x 32 x i8
  scales: 2 x 1 x 32 x f32
  zeros:  2 x 1 x 32 x f32

After async_wait num=4, GemLite local-loads staged packed B/scales/zeros,
dequants to a 128 x 32 f16 shared tile, and feeds tt.dot/HMMA.
The implied shared partition is 49664 bytes:
  A staging:          32768 bytes
  packed B staging:    8192 bytes
  dequant B staging:   8192 bytes
  scale staging:        256 bytes
  zero staging:         256 bytes
```

Implementation:

```text
gate: SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B=1
bench selector: --kernel gemlite-tc-linear2-async-b
scope: only K=12288,N=3072,M multiple of 64, group_size=128 path
design: separate opt-in linear2 kernel with BM=64 BN=32 BK=128, fp32 WMMA accumulators, dynamic 49664-byte shared memory, cp.async for A + packed B + scales + zeros, exact bit*scale+zero dequant, and no C accumulator in shared memory
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Executable hash:

```text
sd-bonsai-gemlite-int1-kernel-bench.exe SHA256=F7F5BDE181DE1080996453FBC4DB056B5EA1D99A0EF09237C06186C18563B0AA
```

SDCPP artifact extraction:

```powershell
$artifactDir = 'build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_async_b_artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir
cuobjdump --dump-ptx build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\bonsai_gemlite_int1_cuda.ptx
cuobjdump --dump-sass build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\bonsai_gemlite_int1_cuda.sass
cuobjdump --dump-resource-usage build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\resource_usage.txt
```

SASS/PTX verification:

```text
PTX cp.async count: 32
SASS LDGSTS count: 14
SASS HMMA count: 964
async-B kernel resource usage: REG=54 STATIC_SHARED=0 DYNAMIC_SHARED=49664 LOCAL=0
```

Isolated linear2 benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-async --compare-current

$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-async-b --compare-current
```

Results:

```text
A-only async linear2 median_ms=3.596 min_ms=3.564 max_ms=4.114
async-B linear2 median_ms=6.041 min_ms=5.617 max_ms=6.428
Python GemLite reference median_ms=1.584

async-B correctness vs native baseline:
corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Conclusion:

```text
The packed-B async path is correct and emits the required cp.async/LDGSTS/HMMA instructions, but it is slower than the A-only async kernel.
The likely reason is that this direct CUDA translation stages the GemLite logical 128x32 packed-B tile as bytes, including repeated packed bytes for the 8 logical K lanes, then performs a CTA-wide dequant pass before eight WMMA inner steps.
That increases shared-memory traffic/barrier cost and does not reproduce Triton's lower-level blocked/swizzled dequant-to-dot schedule closely enough to win.

No 512 image timing was run because isolated async-B did not improve by the required 20%.
Keep async-B off by default. The next source-backed blocker is lower-level B-side dequant/MMA fusion or a closer ldmatrix/swizzled shared layout, not another wrapper-level async tile.
```

## linear2 inline ldmatrix/mma port

Schedule extraction note:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_ldmatrix_schedule.md
```

GemLite artifact inspected:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_gemlite_artifacts\3X4D3RWOSK3VTOHYHNJRW4PYLFBF7475N4PWQMGR5XDOGB3X3TYQ
```

Extracted generated forms:

```text
SASS LDMATRIX form: LDSM.16.MT88.4
LDSM count: 8
PTX MMA form: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
SASS HMMA form: HMMA.16816.F32
HMMA count: 32
LDGSTS count: 54
STS.128 count: 4
LDS.64 count: 36
BAR.SYNC count: 5
```

GemLite shared-memory layout from TTGIR:

```text
A staging:             memdesc<2x64x128xf16, swizzled_shared vec=16 perPhase=1 maxPhase=4> 32768 bytes
packed B staging:      memdesc<2x128x32xi8, swizzled_shared vec=1 perPhase=1 maxPhase=1>   8192 bytes
scales staging:        memdesc<2x1x32xf32, swizzled_shared vec=1 perPhase=1 maxPhase=1>     256 bytes
zeros staging:         memdesc<2x1x32xf32, swizzled_shared vec=1 perPhase=1 maxPhase=1>     256 bytes
dequantized B staging: memdesc<128x32xf16, swizzled_shared vec=8 perPhase=2 maxPhase=4>     8192 bytes
total shared: 49664 bytes
```

Generated pipeline summary:

```text
1. async-copy A, packed B, scales, and zeros into rolling shared slots.
2. wait on the current slot.
3. local-load packed B/scales/zeros from shared.
4. decode bit = (packed >> (k % 8)) & 1.
5. apply w = bit * scale + zero.
6. truncate to f16 and store B into the swizzled 128x32 half shared tile.
7. feed that dequantized-B shared tile through LDSM.16.MT88.4.
8. execute HMMA.16816.F32 / mma.sync.m16n8k16 row.col with fp32 accumulators.
```

Recovered `#shared2` B swizzle:

```c++
phase = (k >> 1) & 3;
phys_n = (((n >> 3) ^ phase) << 3) | (n & 7);
byte_offset = ((k * 32 + phys_n) * 2);
```

Implementation:

```text
gate: SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX=1
bench selector: --kernel gemlite-tc-linear2-ldmatrix
scope: single_blocks.linear2 only, K=12288,N=3072,M multiple of 64
design: inline PTX ldmatrix/mma path using the recovered #shared2 B swizzle, A ldmatrix feed, LDMATRIX x4/trans B feed, mma.sync.m16n8k16 row.col fp32 accumulation, and manual accumulator store using the ggml m16n8 lane mapping
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Artifact extraction:

```powershell
$artifactDir = 'build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_ldmatrix_artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir
cuobjdump --dump-ptx build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\bonsai_gemlite_int1_cuda.ptx
cuobjdump --dump-sass build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\bonsai_gemlite_int1_cuda.sass
cuobjdump --dump-resource-usage build-bonsai-int1\CMakeFiles\stable-diffusion.dir\src\bonsai_gemlite_int1_cuda.cu.obj > $artifactDir\resource_usage.txt
```

SDCPP ldmatrix SASS/resource verification:

```text
LDSM total: 24
LDSM.16.MT88.4: 16
LDSM.16.M88.4: 8
HMMA.16816.F32: 32
FFMA: 1
REG=39
SHARED=10240
LOCAL=0
```

Sanitizer check:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX='1'
compute-sanitizer --tool memcheck --print-limit 20 build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 64 --warmup 0 --runs 1 --kernel env
```

Result:

```text
ERROR SUMMARY: 0 errors
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-async --compare-current

$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-ldmatrix --compare-current
```

Results:

```text
Python GemLite reference median_ms: 1.584
A-only async median_ms: 4.056
ldmatrix median_ms: 9.810

ldmatrix correctness vs current:
corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

Conclusion:

```text
The inline ldmatrix/mma path is correct and emits the required LDSM/HMMA instructions, but it is slower than the A-only async kernel. No 512 image timing was run because the isolated kernel did not improve.

The source-backed remaining mismatch is now narrower: the recovered #shared2 swizzle and accumulator layout are correct, but this SDCPP port does not reproduce GemLite's 49 KiB async A+B+meta staging around the ldmatrix feed. It dequants B into a compact 10 KiB shared path and still pays a full CTA dequant/barrier path before the ldmatrix feed.

Keep SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX off by default. The next source-backed optimization would need to combine the known-good A+B+meta async staging with this now-correct ldmatrix feed, not replace the current default A-only async path.
```

## linear2 SASS-level instruction diff and one-fix pass

Diff artifact:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sass_instruction_diff.md
```

Static SASS count summary:

| Instruction family | GemLite | SDCPP A-only async | SDCPP ldmatrix before | SDCPP ldmatrix after full-A fix |
|---|---:|---:|---:|---:|
| HMMA / MMA | 32 | 4 | 32 | 32 |
| LDSM / LDMATRIX | 8 | 4 | 24 | 24 |
| LDGSTS / cp.async | 54 | 2 | 0 | 0 |
| LDG global loads | 0 | 21 | 11 | 4 |
| LDS shared loads | 52 | 6 | 0 | 0 |
| STS shared stores | 8 | 7 | 9 | 2 |
| STS.128 | 4 | 0 | 0 | 0 |
| F2F conversion | 24 | 16 | 9 | 9 |
| LOP3 | 102 | 30 | 28 | 13 |
| SHF | 40 | 54 | 36 | 17 |
| FFMA | 32 | 7 | 1 | 1 |
| BAR.SYNC | 5 | 4 | 17 | 2 |
| DEPBAR | 14 | 5 | 0 | 0 |
| Registers | unknown | 59 | 39 | 39 |
| Shared bytes | 49664 | 7168 | 10240 | 24576 |
| Local bytes | 0 | 0 | 0 | 0 |

Primary mismatch selected for the one-fix pass:

```text
A. Too many barriers / bad pipeline.

The ldmatrix kernel restaged A as 64x16 eight times per BK=128 group, creating 17 barriers in the static BK body versus GemLite's 5.
```

Fix applied:

```text
Stage A[64,128] once per BK=128 group in the ldmatrix kernel and load all K-inner fragments from that full staged tile.
```

SASS after fix:

```text
artifact_dir=build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear2_sdcpp_ldmatrix_after_full_a_artifacts
HMMA=32
LDSM=24
LDSM.16.MT88.4=16
LDSM.16.M88.4=8
LDGSTS=0
BAR.SYNC=2
REG=39
SHARED=24576
LOCAL=0
```

Isolated benchmark:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear2.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear2-ldmatrix --compare-current
```

Results:

```text
Python GemLite reference median_ms: 1.584
SDCPP A-only async median_ms: 4.056
SDCPP ldmatrix before full-A fix median_ms: 9.810
SDCPP ldmatrix after full-A fix median_ms: 9.887

corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0
inf=0
```

Conclusion:

```text
The one source-backed barrier fix worked at the instruction-count level but did not improve runtime, so no 512 image timing was run.

The current blocker is no longer barrier count alone. The remaining instruction-level mismatch is:

E. SDCPP still has extra ldmatrix grouping: 24 LDSM per BK body versus GemLite's 8 visible B-side LDSM.
G. SDCPP ldmatrix still has no GemLite-style LDGSTS/DEPBAR pipeline around the ldmatrix feed.

Keep SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX off by default. The current default should remain the faster A-only async linear2 path.
```

## double_blocks attention projection Tensor Core pass

Decision carried into this pass: stop chasing the hand-written `single_blocks.linear2` ldmatrix path. The recovered shared2 swizzle, LDSM/HMMA usage, and reduced barrier count proved correctness, but the ldmatrix path stayed slower than the A-only async linear2 kernel. The default/best linear2 path remains `SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC=1`; `SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B` and `SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX` stay off by default.

### GemLite projection contracts

Extracted with:

```powershell
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear transformer_blocks.0.attn.to_out.0 --m 1024 --warmup 1 --runs 5 --out build-bonsai-int1\bonsai-gemlite-kernel-oracle\double_blocks_attn_proj_img_contract.json
F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\.venv\Scripts\python.exe tools\diagnostics\bonsai_gemlite_shape_oracle.py --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear transformer_blocks.0.attn.to_add_out --m 512 --warmup 1 --runs 5 --out build-bonsai-int1\bonsai-gemlite-kernel-oracle\double_blocks_attn_proj_txt_contract.json
```

Combined sidecar:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\double_blocks_attn_proj_contract.json
```

| family | HF linear | shape MxKxN | GemLite family | config | Python median |
|---|---|---:|---|---|---:|
| double_blocks.img_attn.proj | transformer_blocks.0.attn.to_out.0 | 1024x3072x3072 | GEMM | BM=64 BN=128 BK=128 GROUP_SIZE_M=8 A_load_order=2 warps=4 stages=1 | 0.430 ms |
| double_blocks.txt_attn.proj | transformer_blocks.0.attn.to_add_out | 512x3072x3072 | GEMM | BM=64 BN=64 BK=64 GROUP_SIZE_M=8 A_load_order=0 warps=4 stages=4 | 0.297 ms |

### Implementation

Added gate:

```powershell
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ='1'
```

Bench selectors:

```text
--kernel gemlite-tc-img-attn-proj
--kernel gemlite-tc-txt-attn-proj
```

The image projection path reuses the existing proven BM64/BN128/BK128 register-accum Tensor Core normal-linear kernel. The text projection path adds a BM64/BN64/BK64 register-accum Tensor Core kernel to match its GemLite cache entry. Both launch on the active ggml CUDA backend stream and keep packed INT1 weights/scales/zeros resident; no full FP16 expansion or CPU dequant fallback is introduced.

### Isolated benchmark

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Commands:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_attn.proj.weight --m 1024 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.img_attn.proj.weight --m 1024 --warmup 1 --runs 10 --kernel gemlite-tc-img-attn-proj --compare-current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.txt_attn.proj.weight --m 512 --warmup 1 --runs 10 --kernel current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.double_blocks.0.txt_attn.proj.weight --m 512 --warmup 1 --runs 10 --kernel gemlite-tc-txt-attn-proj --compare-current
```

| target | native median | TC median | speedup | correctness |
|---|---:|---:|---:|---|
| img_attn.proj | 7.009 ms | 0.982 ms | 7.1x | corr=1.000000000 cos=1.000000000 mean_abs=0.000000279 max_abs=0.000976562 |
| txt_attn.proj | 3.732 ms | 0.540 ms | 6.9x | corr=1.000000000 cos=1.000000000 mean_abs=0.000000259 max_abs=0.000976562 |

Both outputs had `nan=0`, `inf=0`, `full_fp16_weight_expansion=false`, and `cpu_dequant_fallback_calls=0`.

### 512 warmed timing

Build:

```powershell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-bench sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Command:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
$env:SDCPP_MODEL_FAMILY_HINT='bonsai'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK='1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT='3'
$env:SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_QKV='1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 3 --output-dir F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-baseline\tc-v2-linear2-mlps-qkv-proj-prepack-512
```

Result:

```text
model_load_ms=4870.27
full_median_ms=2108.029
generate_median_ms=2049.918
denoise_median_ms=1738.000
text_encode_mean_ms=116.333
vae_decode_mean_ms=191.667
png_save_mean_ms=56.617
elapsed_includes_model_load=false
```

Outputs:

| image | SHA256 | min/max | mean/std | unique RGB |
|---|---|---|---|---:|
| build-bonsai-int1\bonsai-baseline\tc-v2-linear2-mlps-qkv-proj-prepack-512\bonsai_bench_run_00.png | FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC | 0/255 | 121.918/66.171 | 134697 |
| build-bonsai-int1\bonsai-baseline\tc-v2-linear2-mlps-qkv-proj-prepack-512\bonsai_bench_run_01.png | E48F0FBE9E5FA7884709CD210135154AD20315F6CA48972FA4E373DD25F310D0 | 0/255 | 117.917/57.350 | 122212 |
| build-bonsai-int1\bonsai-baseline\tc-v2-linear2-mlps-qkv-proj-prepack-512\bonsai_bench_run_02.png | 5DC84F0547E27A87223EDB0161C1913B88E34C08AEA5080CC1AF23A38610E1FF | 0/255 | 109.601/59.506 | 132728 |

No fail-fast abort occurred.

### Updated family map

One 512x512, 4-step, runs=1 family-profile pass with current best gates plus TC attention projection:

```text
single_blocks.linear1        572.075 ms  47.65%  median=6.977 ms  kernel=tc_v2_prepack_v3
single_blocks.linear2        278.807 ms  23.22%  median=3.333 ms  kernel=tc_linear2_async
double_blocks.img_mlp.0       96.104 ms   8.01%  median=4.691 ms  kernel=tc_img_mlp0
double_blocks.img_attn.qkv    55.579 ms   4.63%  median=0.883 ms  kernel=tc_qkv
double_blocks.img_mlp.2       54.003 ms   4.50%  median=2.617 ms  kernel=tc_img_mlp2
double_blocks.txt_mlp.0       51.323 ms   4.28%  median=2.496 ms  kernel=tc_img_mlp0
double_blocks.txt_attn.qkv    33.050 ms   2.75%  median=0.515 ms  kernel=tc_qkv
double_blocks.txt_mlp.2       31.501 ms   2.62%  median=1.511 ms  kernel=tc_img_mlp2
double_blocks.img_attn.proj   18.790 ms   1.57%  median=0.888 ms  kernel=tc_img_attn_proj
double_blocks.txt_attn.proj    9.288 ms   0.77%  median=0.452 ms  kernel=tc_txt_attn_proj
```

The projection bottleneck is no longer meaningful. The measured INT1 wall is back to `single_blocks.linear1` and `single_blocks.linear2`, with the remaining gap mainly inside already-optimized but still slower-than-GemLite Tensor Core paths.

## single_blocks.linear1 large-shared GemLite staging pass

### Schedule note

Required source-backed note:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear1_large_smem_schedule.md
```

GemLite artifact inspected:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear1_gemlite_artifacts\MTTZSHMXSTI3LRZLN6M6DSALCHFOKZRCPAAJB6EXHG6HWXB33ADQ
```

Key artifact findings:

- `single_blocks.linear1` uses GemLite `GEMM`, not split-K/GEMV.
- Config: `BM=64 BN=128 BK=128`, `GROUP_SIZE_M=8`, `A_load_order=0`, `num_warps=4`, `num_stages=1`.
- GemLite reported shared memory: `49152` bytes.
- TTGIR shared partition:
  - A: `64x128xf16` = `16384` bytes, `swizzled_shared<{vec=16, perPhase=1, maxPhase=4, order=[1,0]}>`
  - B: `128x128xf16` = `32768` bytes, `swizzled_shared<{vec=8, perPhase=1, maxPhase=8, order=[1,0]}>`
- GemLite stages a full `A[64,128]` tile and a full dequantized `B[128,128]` f16 tile before the dot.
- GemLite SASS/PTX contains `LDG.E.128` / `ld.global.v4.b32`, `STS.128` / `st.shared.v4.b32`, `LDSM.16.M88.4`, `LDSM.16.MT88.4`, and `HMMA.16816.F32`.
- GemLite linear1 artifact does not use `cp.async`; this pass intentionally did not add async staging.

### Implementation

Added an opt-in source-backed kernel:

```text
SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM=1
--kernel gemlite-tc-linear1-largesmem
```

Design:

- CTA tile: `64x128`
- K tile: `128`
- 4 warps
- dynamic shared memory requested at launch: `49152` bytes
- shared layout:
  - `A[64,128]`
  - `B[128,128]`
- fp32 accumulator fragments stay in registers
- final store converts to fp16
- no full FP16 weight expansion
- no CPU dequant fallback
- backend stream launch path reused

This is intentionally one kernel, not a tile sweep. It stages the full B tile once per `BK=128` group instead of re-staging `B[16,128]` per inner MMA slice.

### Build and SASS check

Build:

```powershell
cmake --build build-bonsai-int1 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

Executable:

```text
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe
SHA256=9BCC1DCFFB7E85ACFE791794D435E71F701FC8D6E944142244CEA4A0E091DAD9
```

SASS/resource artifact:

```text
build-bonsai-int1\bonsai-gemlite-kernel-oracle\linear1_sdcpp_largesmem_artifacts
```

New SDCPP largesmem kernel verification:

```text
HMMA=128
LDSM=128
LDG.E.128=7
STS.128=8
STS.U16=0
BAR.SYNC=2
CP.ASYNC=0
LDGSTS=0
REG=173
dynamic_shared_requested=49152 bytes
LOCAL=0
```

The kernel does use Tensor Core/SASS matrix-load machinery and vectorized load/store forms, and it does not use cp.async. It is still not a full Triton layout clone because the SDCPP implementation uses a straightforward row-major shared layout rather than GemLite's exact swizzled shared encodings.

### Isolated benchmark

Commands:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1='1'
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear1.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-gemm-v2-prepack-v3 --compare-current
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --linear model.diffusion_model.single_blocks.0.linear1.weight --m 1536 --warmup 1 --runs 10 --kernel gemlite-tc-linear1-largesmem --compare-current
```

Results:

| kernel | median | min | max | correctness vs native |
|---|---:|---:|---:|---|
| current best `tc_v2_prepack_v3` | 7.102 ms | 7.047 ms | 8.289 ms | corr=1.000000000 cos=1.000000000 mean_abs=0.000000290 max_abs=0.001953125 |
| new `tc_linear1_largesmem` | 12.594 ms | 12.463 ms | 13.367 ms | corr=1.000000000 cos=1.000000000 mean_abs=0.000000290 max_abs=0.001953125 |

Both runs had:

```text
nan=0
inf=0
full_fp16_weight_expansion=false
cpu_dequant_fallback_calls=0
```

### Result

The large-shared source-backed kernel is correct but slower than the current best prepack-v3 kernel. No 512 image timing was run because the isolated linear1 kernel did not improve by 20%.

Observed blocker:

- Matching GemLite's 48 KiB shared footprint and full `B[128,128]` staging is not enough.
- The new SDCPP kernel still lacks GemLite's exact swizzled shared-memory layout and compiler-generated instruction schedule.
- The large shared footprint plus `REG=173` likely reduces occupancy enough that avoiding repeated `B[16,128]` staging does not pay off.
- Current `tc_v2_prepack_v3` remains the best SDCPP linear1 path and should stay the default optimized choice.

Next concrete task:

- Do not continue linear1 with more guessed staging variants.
- Either port GemLite's exact swizzled shared layout and ldmatrix address pattern for linear1, or stop kernel parity work and decide whether the remaining ~572 ms family cost is acceptable relative to the engineering complexity.

## CUDA 13.3 side-by-side compiler evaluation

Full evaluation artifact:

```text
build-bonsai-int1\bonsai-cuda133-eval\compiler_compare.md
```

### Scope

This pass compared CUDA 13.2 against CUDA 13.3 using the same source and current best Bonsai kernel gates. No Bonsai kernels, model logic, scheduler, VAE, text encoder, or final-layer code was changed.

Separate CUDA 13.3 build directory:

```text
build-bonsai-int1-cuda133
```

The existing `build-bonsai-int1` directory was preserved.

### Toolchains

Current build:

```text
CMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe
nvcc/ptxas/cuobjdump/nvdisasm: CUDA 13.2 V13.2.51
```

CUDA 13.3 eval:

```text
CMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe
nvcc/ptxas: CUDA 13.3 V13.3.33
cuobjdump/nvdisasm: CUDA 13.3 V13.3.29
```

Runtime:

```text
GPU=NVIDIA GeForce RTX 4080 SUPER
driver=596.49
```

CUDA 13.3 GA notes list a newer driver floor than the installed local driver, but compilation and local benchmark execution both worked in this evaluation.

### Build commands

Configure:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake -S . -B build-bonsai-int1-cuda133 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_LENS_EXPERIMENTAL_RUNTIME=ON -DSD_BONSAI_GEMLITE_INT1_SPIKE=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
```

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1-cuda133 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
cmake --build build-bonsai-int1-cuda133 --config Release --target sd-bonsai-gemlite-int1-bench --parallel 8
```

CUDA 13.3 kernel bench:

```text
build-bonsai-int1-cuda133\bin\sd-bonsai-gemlite-int1-kernel-bench.exe
SHA256=9288600A8AC53C277040D38DC851774D1A35F3354FDB9B86945134C21921308C
```

### Isolated kernel comparison

All isolated runs used the real Bonsai pack, `--warmup 1 --runs 10 --compare-current`, and current best kernel selectors only.

| family | best kernel | CUDA 13.2 median | CUDA 13.3 median | delta |
|---|---|---:|---:|---:|
| `single_blocks.linear1` | `gemlite-tc-gemm-v2-prepack-v3` | 8.170 ms | 7.037 ms | -13.9% |
| `single_blocks.linear2` | `gemlite-tc-linear2-async` | 4.059 ms | 3.584 ms | -11.7% |
| `double_blocks.img_mlp.0` | `gemlite-tc-img-mlp0` | 5.012 ms | 4.999 ms | -0.3% |
| `double_blocks.img_mlp.2` | `gemlite-tc-img-mlp2` | 2.796 ms | 2.788 ms | -0.3% |
| `double_blocks.img_attn.qkv` | `gemlite-tc-img-qkv` | 2.763 ms | 2.757 ms | -0.2% |
| `double_blocks.txt_attn.qkv` | `gemlite-tc-txt-qkv` | 1.633 ms | 1.641 ms | +0.5% |
| `double_blocks.img_attn.proj` | `gemlite-tc-img-attn-proj` | 0.981 ms | 0.986 ms | +0.5% |
| `double_blocks.txt_attn.proj` | `gemlite-tc-txt-attn-proj` | 0.536 ms | 0.542 ms | +1.1% |

All isolated CUDA 13.3 correctness checks passed with `nan=0`, `inf=0`, and the same error ranges as the CUDA 13.2 build. The dominant families improved; the smaller families were effectively unchanged.

### SASS/resource comparison

High-level SASS/resource counts were unchanged between CUDA 13.2 and CUDA 13.3:

| kernel section | REG | shared | local | HMMA | LDSM | LDGSTS | BAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear1_prepack_v3` | 168 | 6400 | 0 | 128 | 128 | 0 | 16 |
| `linear2_async` | 59 | 7168 | 0 | 4 | 4 | 2 | 4 |
| `normal_tc_v2` | 168 | 6144 | 0 | 128 | 128 | 0 | 16 |
| `qkv_strided` | 168 | 6144 | 0 | 128 | 128 | 0 | 16 |
| `txt_proj_tc` | 80 | 4096 | 0 | 32 | 32 | 0 | 8 |

Artifacts:

```text
build-bonsai-int1\bonsai-cuda133-eval\current_cuda132_bonsai_cuda.sass
build-bonsai-int1\bonsai-cuda133-eval\cuda133_bonsai_cuda.sass
build-bonsai-int1\bonsai-cuda133-eval\sass_resource_summary.json
```

The CUDA 13.3 win appears to be lower-level scheduling/codegen inside the same visible instruction/resource envelope, not a structural kernel change.

### 512 warmed timing

Because CUDA 13.3 improved both dominant isolated kernels by more than 10%, one 512 warmed timing run was performed.

Settings:

```text
512x512
steps=4
sampler=euler
scheduler=discrete
prediction=flux2_flow
cfg=1
guidance=1
flow_shift=3
seed=42
warmup=1
runs=3
elapsed_includes_model_load=false
```

CUDA 13.3 result:

```text
denoise_median_ms=1652.000
full_median_ms=2006.305
text_encode_mean_ms=113.000
vae_decode_mean_ms=184.000
png_save_mean_ms=53.808
```

Prior comparable CUDA 13.2 result after projection TC path:

```text
denoise_median_ms=1738.000
full_median_ms=2108.029
```

Net full-run improvement:

```text
denoise: ~4.9%
full image: ~4.8%
```

Outputs:

```text
build-bonsai-int1\bonsai-cuda133-eval\cuda133-512-timing\bonsai_bench_run_00.png
SHA256=FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC
min=0 max=255 mean=121.918 std=66.171 unique_rgb=134697

build-bonsai-int1\bonsai-cuda133-eval\cuda133-512-timing\bonsai_bench_run_01.png
SHA256=E48F0FBE9E5FA7884709CD210135154AD20315F6CA48972FA4E373DD25F310D0
min=0 max=255 mean=117.917 std=57.350 unique_rgb=122212

build-bonsai-int1\bonsai-cuda133-eval\cuda133-512-timing\bonsai_bench_run_02.png
SHA256=5DC84F0547E27A87223EDB0161C1913B88E34C08AEA5080CC1AF23A38610E1FF
min=0 max=255 mean=109.601 std=59.506 unique_rgb=132728
```

No fail-fast abort occurred.

### CUDA Tile C++ smoke

Standalone diagnostic:

```text
tools\diagnostics\cuda_tile_cpp_smoke\tile_gemm_smoke.cu
```

Build command:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe' -std=c++20 -arch=sm_89 -enable-tile -c tools\diagnostics\cuda_tile_cpp_smoke\tile_gemm_smoke.cu -o build-bonsai-int1\bonsai-cuda133-eval\tile_cpp_smoke.obj
```

Result:

- Compile passed.
- CUDA Tile C++ required `-enable-tile` and `__tile_global__`.
- SASS contains `LDSM`, but no `HMMA`/`MMA` for this minimal Tile C++ half GEMM.
- The multiply path lowered to scalar `FMUL`/`FADD` in this smoke.

CUDA Tile C++ is not yet proven as a Bonsai rewrite path on this setup. It remains research-only until a minimal Tile C++ GEMM emits Tensor Core MMA instructions.

### Recommendation

CUDA 13.3 is useful as an optional Bonsai performance build for this branch: it gives a real 11-14% isolated improvement for `single_blocks.linear1` and `single_blocks.linear2`, and about 5% on a 512 warmed generation run.

It does not close GemLite parity. The SASS/resource counts are unchanged, and the remaining gap is still kernel schedule/layout quality versus Triton/GemLite.

Recommended path:

1. Use CUDA 13.3 for local Bonsai performance evaluation if the driver/runtime pairing remains stable.
2. Do not make CUDA Tile C++ part of the Bonsai runtime yet.
3. Treat CUDA 13.3 as a modest compiler win, not a replacement for GemLite-equivalent kernel design.

## CUDA 13.3 Windows diagnostics pass

This pass kept the CUDA 13.3 build and current best Bonsai gates, added focused profiler bracketing only, and did not change model math or kernel implementations.

### Tool inventory

Inventory:

```text
build-bonsai-int1\bonsai-cuda133-eval\windows_diagnostics_inventory.md
```

Key tools:

- Nsight Systems CLI: `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe`
- Nsight Compute CLI: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe`
- CUDA 13.3 `nvcc`/`ptxas`: V13.3.33
- CUDA 13.3 `cuobjdump`/`nvdisasm`: V13.3.29
- GPU: NVIDIA GeForce RTX 4080 SUPER
- Driver: 596.49

### Instrumentation

Added gated profiling ranges:

- `SDCPP_BONSAI_PROFILE_RANGE=1`
- `cudaProfilerStart()` immediately before Bonsai denoise
- `cudaProfilerStop()` immediately after Bonsai denoise
- NVTX labels for `bonsai.text_encode`, `bonsai.denoise`, `bonsai.vae_decode`, `bonsai.generate_image`, and `bonsai.png_save`

The profile gate is off by default. Normal Bonsai and non-Bonsai behavior is unchanged when the env var is unset.

Build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1-cuda133 --config Release --target sd-bonsai-gemlite-int1-bench --parallel 8
```

### Nsight Systems denoise capture

Command shape:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe' profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --force-overwrite=true -o build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise build-bonsai-int1-cuda133\bin\sd-bonsai-gemlite-int1-bench.exe ... --width 512 --height 512 --steps 4 --warmup 0 --runs 1
```

Outputs:

```text
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise.nsys-rep
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise.sqlite
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise_latest_cuda_gpu_kern_sum.csv
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise_latest_cuda_api_sum.csv
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise_latest_cuda_gpu_mem_time_sum.csv
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise_latest_cuda_gpu_trace.csv
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_denoise_latest_cuda_api_trace.csv
```

Summary:

| metric | value |
|---|---:|
| GPU kernel instances | 4,960 |
| summed GPU kernel time | 1,428.281 ms |
| GPU timeline span | 1,602.167 ms |
| positive GPU queue gaps | 167.032 ms |
| max observed GPU gap | 83.972 ms |
| `cudaLaunchKernel` calls | 4,952 |
| `cudaLaunchKernel` API total | 158.744 ms |
| `cudaLaunchKernel` API average | 32.057 us |
| `cudaStreamSynchronize` calls | 32 |
| `cudaStreamSynchronize` API total | 930.533 ms |
| GPU H2D memcpy time | 6.626 ms |
| GPU D2H memcpy time | 0.180 ms |
| GPU D2D memcpy time | 0.046 ms |

Top GPU kernels by total duration:

| kernel | total | calls | median |
|---|---:|---:|---:|
| `bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_padded_kernel` | 558.065 ms | 80 | 6.974 ms |
| `bonsai_int1_linear_gemlite_tc_linear2_async_kernel` | 266.718 ms | 80 | 3.332 ms |
| `bonsai_int1_linear_gemlite_tc_gemm_v2_kernel` | 247.060 ms | 100 | 2.475 ms |
| `bonsai_int1_linear_gemlite_tc_gemm_v2_strided_out_kernel` | 84.171 ms | 120 | 0.866 ms |
| `cpy_scalar<float,float>` family | 50.316 ms | 828 | 0.059 ms |
| `cpy_scalar_contiguous<half,float>` | 41.630 ms | 320 | 0.035 ms |
| `k_bin_bcast<op_mul>` | 35.852 ms | 844 | 0.015 ms |
| `flash_attn_ext_f16` | 28.206 ms | 100 | 0.278 ms |

Interpretation:

- Kernel execution still dominates the denoise range.
- Launch overhead is real but secondary: about 159 ms API time across 4,952 launches.
- Device memory copies during denoise are small, under 7 ms of GPU copy time.
- The large `cudaStreamSynchronize` entries are mostly waits for queued GPU work at graph/output boundaries, not standalone evidence of device-wide synchronization in the Bonsai custom kernels.
- There are GPU queue bubbles, but the queue is mostly occupied by the optimized Bonsai kernels.

### ETW attempt

Result:

```text
build-bonsai-int1\bonsai-cuda133-eval\etw_bonsai_attempt.md
```

`wpr` and `logman` are available, but `logman query providers` did not expose a CUDA/NVIDIA driver provider usable for a bounded Bonsai capture in this shell. The only GPU-ish match was `Microsoft.Windows.HyperV.GpupVDev`. Nsight Systems CUDA tracing remains the useful Windows driver/runtime view for this pass.

### Nsight Compute attempt

Result:

```text
build-bonsai-int1\bonsai-cuda133-eval\ncu_bonsai_blocker.md
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_stdout.txt
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_stderr.txt
```

Nsight Compute failed before metric collection:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

No `.ncu-rep` was produced. The isolated linear1 bench still ran and reported `median_ms=7.556`, `corr=1.0`, `nan=0`, `inf=0`. The same permissions blocker applies to linear2 microarchitecture metrics.

### Bottleneck classification

The remaining 512 denoise time is primarily kernel execution inside the optimized Bonsai INT1 kernels, especially `single_blocks.linear1` and `single_blocks.linear2`.

Secondary costs:

- Kernel launch overhead: measurable at roughly 159 ms, but not the primary wall.
- GPU queue gaps: roughly 167 ms positive gap total; worth tracking, but smaller than the Bonsai kernel execution total.
- Memory copies/allocation: not material during denoise in this capture.
- CPU-side orchestration: CPU sampling was disabled by Nsight Systems due to lack of admin privileges, but CUDA API data does not show CPU orchestration dominating over GPU kernels.

Recommended next action:

Use CUDA 13.3 as the current performance baseline and avoid more blind microkernel variants. The next implementation work should either:

1. reduce custom-op launch count with CUDA graph capture or graph reuse only if launch overhead becomes the chosen target, or
2. resume source-backed kernel work on `single_blocks.linear1` only with a lower-level Triton-like shared-layout/dequant feed design, because it remains the largest measured kernel execution wall.

## Nsight Compute counter pass

After enabling GPU performance counters for all users in NVIDIA Control Panel, Nsight Compute successfully collected targeted reports for the two dominant Bonsai kernels. No model logic, scheduler/VAE/text code, image generation, or kernel implementation was changed in this pass.

Counter access check:

```text
build-bonsai-int1\bonsai-cuda133-eval\ncu_counter_access_check.md
build-bonsai-int1\bonsai-cuda133-eval\ncu_counter_access_check.ncu-rep
```

### NCU report paths

```text
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current.txt
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current_details.csv
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_stalls.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_stalls_details.csv

build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_current.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_current.txt
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_current_details.csv
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_stalls.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_stalls_details.csv

build-bonsai-int1\bonsai-cuda133-eval\ncu_bonsai_kernel_diagnosis.md
```

### Commands

Linear1:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe' --target-processes application-only --set full --kernel-name 'regex:bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_padded_kernel' --launch-count 1 --force-overwrite --export build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current build-bonsai-int1-cuda133\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --internal-name model.diffusion_model.single_blocks.0.linear1.weight --rows-m 1536 --warmup 1 --runs 2 --kernel gemlite-tc-gemm-v2-prepack-v3 --compare-current
```

Linear2:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\target\windows-desktop-win7-x64\ncu.exe' --target-processes application-only --set full --kernel-name 'regex:bonsai_int1_linear_gemlite_tc_linear2_async_kernel' --launch-count 1 --force-overwrite --export build-bonsai-int1\bonsai-cuda133-eval\ncu_linear2_current build-bonsai-int1-cuda133\bin\sd-bonsai-gemlite-int1-kernel-bench.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --internal-name model.diffusion_model.single_blocks.0.linear2.weight --rows-m 1536 --warmup 1 --runs 2 --kernel gemlite-tc-linear2-async --compare-current
```

### Linear1 metrics

Kernel: `bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_padded_kernel`

Shape: `M=1536 K=3072 N=27648`

Correctness:

```text
corr=1.000000000
cos=1.000000000
mean_abs=0.000000290
max_abs=0.001953125
nan=0 inf=0
```

Key NCU metrics:

| metric | value |
|---|---:|
| `gpu__time_duration.sum` | 7.394 ms |
| SM throughput | 55.96% |
| DRAM throughput | 2.27% |
| L1/TEX throughput | 50.37% |
| issue active | 49.41% |
| active warps/scheduler | 2.97 |
| eligible warps/scheduler | 0.77 |
| achieved occupancy | 24.69% |
| theoretical occupancy | 25.00% |
| static shared memory/block | 6,400 bytes |
| local/register spilling | 0 |
| HMMA tensor-pipe active | 16.61% |
| HMMA instructions | 63,700,992 |

NCU classification:

- Primary limiter: shared-memory bank conflicts / shared-memory layout.
- Secondary limiter: occupancy/register pressure.
- Tensor Cores are not saturated.
- DRAM bandwidth is not saturated.

NCU rules called out:

- `SharedMemoryConflicts`: average 6.0-way shared-load conflict, estimated speedup 16.79%.
- `UncoalescedSharedAccess`: estimated speedup 26.45%.
- `TheoreticalOccupancy`: estimated speedup 44.04%.
- `IssueSlotUtilization`: estimated speedup 44.04%.

### Linear2 metrics

Kernel: `bonsai_int1_linear_gemlite_tc_linear2_async_kernel`

Shape: `M=1536 K=12288 N=3072`

Correctness:

```text
corr=0.999999999
cos=0.999999999
mean_abs=0.000002307
max_abs=0.003906250
nan=0 inf=0
```

Key NCU metrics:

| metric | value |
|---|---:|
| `gpu__time_duration.sum` | 3.520 ms |
| SM throughput | 54.30% |
| compute/memory throughput | 69.97% |
| DRAM throughput | 2.94% |
| L1/TEX throughput | 70.67% |
| L2 throughput | 49.34% |
| issue active | 54.82% |
| active warps/scheduler | 7.35 |
| eligible warps/scheduler | 1.32 |
| achieved occupancy | 61.26% |
| theoretical occupancy | 66.67% |
| static shared memory/block | 7,168 bytes |
| local/register spilling | 0 |
| HMMA tensor-pipe active | 15.50% |
| HMMA instructions | 28,311,552 |

NCU classification:

- Primary limiter: shared-memory bank conflicts / shared-memory layout.
- Secondary limiter: L1/shared pressure and issue-slot underutilization.
- Tensor Cores are not saturated.
- DRAM bandwidth is not saturated.
- Occupancy is not the main linear2 problem.

NCU rules called out:

- `SharedMemoryConflicts`: average 12.1-way shared-load conflict, estimated speedup 46.83%.
- `UncoalescedSharedAccess`: estimated speedup 64.06%.
- `MemoryCacheAccessPattern`: estimated speedup 34.98%.

### Next task

The NCU data says the next bottleneck is not launch overhead and not DRAM bandwidth. Both dominant kernels are leaving Tensor Core throughput on the table because the shared-memory feed is inefficient.

Recommended next implementation task:

```text
Implement a source-backed shared-memory layout/swizzle rewrite for the current best single_blocks.linear1 kernel first, targeting the NCU-reported 6-way shared-load bank conflicts and low eligible-warps issue pattern. Validate with NCU before running image timing.
```

If that direction works for linear1, apply the same shared-layout diagnosis to linear2 next.

## linear1 shared-bank-conflict pass

This pass tested exactly one shared-layout change for `single_blocks.linear1`: pad the A shared-memory tile stride in the current best prepack-v3 kernel. It did not change model math, scheduler/VAE/text/final-layer behavior, tile sizes, accumulator strategy, or global INT1/prepack layout.

### Conflict source

Source note:

```text
build-bonsai-int1\bonsai-cuda133-eval\linear1_bank_conflict_source.md
```

Baseline NCU showed:

```text
SharedMemoryConflicts:
  average 6.0-way shared-load conflict
  shared load requests: 63,700,992
  bank conflicts: 127,401,984
  conflict wavefront share: 33.33%

UncoalescedSharedAccess:
  excessive shared wavefronts: 127,401,984
```

The current prepack-v3 kernel already had padded B shared memory:

```cpp
__shared__ half b_smem[16][136];
```

but A used an unpadded 16-wide half stride:

```cpp
__shared__ half a_smem[64][16];
wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
```

The single fix was therefore:

```text
A_smem stride 16 -> 24
```

behind:

```text
SDCPP_BONSAI_INT1_LINEAR1_BANKFIX=1
--kernel gemlite-tc-linear1-bankfix
```

### Build

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build-bonsai-int1-cuda133 --config Release --target sd-bonsai-gemlite-int1-kernel-bench --parallel 8
```

### Isolated timing

Sequential isolated runs, CUDA 13.3, `--warmup 1 --runs 10 --compare-current`:

| kernel | median | min | max | correctness |
|---|---:|---:|---:|---|
| `gemlite-tc-gemm-v2-prepack-v3` | 7.184 ms | 7.032 ms | 7.583 ms | corr=1.0, mean_abs=0.000000290 |
| `gemlite-tc-linear1-bankfix` | 7.969 ms | 7.962 ms | 8.514 ms | corr=1.0, mean_abs=0.000000290 |

The first attempt accidentally overlapped the baseline and bankfix benches and was discarded. The table above uses sequential reruns only.

### NCU before/after

Reports:

```text
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_current_details.csv
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_stalls_details.csv
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_bankfix.ncu-rep
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_bankfix_details.csv
build-bonsai-int1\bonsai-cuda133-eval\ncu_linear1_bankfix_stalls_details.csv
```

| metric | baseline | bankfix |
|---|---:|---:|
| NCU kernel duration | 7.394 ms | 8.390 ms |
| SM throughput | 55.96% | 49.71% |
| DRAM throughput | 2.27% | 1.95% |
| L1/TEX throughput | 50.37% | 44.80% |
| issue active | 49.41% | 45.28% |
| active warps/scheduler | 2.97 | 2.98 |
| eligible warps/scheduler | 0.77 | 0.66 |
| achieved occupancy | 24.69% | 24.88% |
| static shared memory/block | 6,400 bytes | 7,424 bytes |
| HMMA tensor-pipe active | 16.61% | 14.21% |
| HMMA instructions | 63,700,992 | 63,700,992 |
| local/register spilling | 0 | 0 |
| `SharedMemoryConflicts` rule | present, 6.0-way | not emitted |
| `UncoalescedSharedAccess` rule | present | not emitted |

The A-padding fix materially reduced the shared-conflict rule output, but performance regressed. It lowered issue activity, eligible warps, Tensor Core active percentage, and L1/TEX throughput while increasing shared memory per block. No image timing was run because the isolated kernel did not improve.

### Result

The conflict source hypothesis was partially correct: padding A removed the NCU shared-conflict warnings. However, the fix is not useful as implemented because it worsens runtime.

Next bottleneck after this failed fix:

```text
The remaining useful fix is not simple A-stride padding. The next pass should inspect the generated SASS/source mapping for the bankfix and baseline side-by-side and target a layout that reduces shared conflicts without increasing register/shared pressure or lowering issue eligibility. A B/A ldmatrix-compatible swizzle or warp-fragment address remap is more likely than plain padding.
```

## CUDA Graph denoise replay prototype

This pass stopped shared-memory/kernel work and tested CUDA Graph replay as a measured launch/gap overhead target for the fixed Bonsai 512x512, 4-step benchmark lane.

### Feasibility

Feasibility note:

```text
build-bonsai-int1\bonsai-cuda133-eval\cuda_graph_feasibility.md
```

Key findings:

- ggml CUDA already has `cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` support, but the build had `GGML_CUDA_GRAPHS=OFF`.
- SDCPP rebuilds the ggml compute context every diffusion-model call, so the upstream graph cache key (`cgraph->nodes[0]`) is too transient for this runner.
- The Bonsai custom ops are graph-capture-safe in the normal timing lane: they launch on the active ggml backend stream and profiling/dump/sync paths are off.
- Capturing the entire CPU sampler loop is not the right first step because sampler math, host input copies, output materialization, and fail-fast checks still run outside the backend graph.

Implementation:

- Reconfigured the CUDA 13.3 build with `GGML_CUDA_GRAPHS=ON`.
- Added a Bonsai-gated stable ggml CUDA graph key based on graph topology/signature when `SDCPP_BONSAI_CUDA_GRAPH_DENOISE=1`.
- Added trace logging behind `SDCPP_BONSAI_CUDA_GRAPH_TRACE=1`.
- Graph use is off by default and applies only to env-gated graphs containing `GGML_OP_CUSTOM`, which is the Bonsai transformer graph in this lane.

### Build

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake -S . -B build-bonsai-int1-cuda133 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON -DSD_BUILD_SHARED_LIBS=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_LENS_EXPERIMENTAL_RUNTIME=ON -DSD_BONSAI_GEMLITE_INT1_SPIKE=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
cmake --build build-bonsai-int1-cuda133 --config Release --target sd-bonsai-gemlite-int1-bench --parallel 8
```

### Correctness

Command shape:

```powershell
build-bonsai-int1-cuda133\bin\sd-bonsai-gemlite-int1-bench.exe --diffusion-model F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --llm F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf --vae F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors --prompt "A bonsai tree in a quiet ceramic studio, soft morning light" --width 512 --height 512 --steps 4 --seed 42 --sampler euler --scheduler discrete --cfg-scale 1.0 --guidance 1.0 --flow-shift 3.0 --prediction flux2_flow --warmup 1 --runs 1 --output-dir <dir>
```

Outputs:

```text
non-graph: build-bonsai-int1\bonsai-cuda133-eval\graph_correctness\nongraph\bonsai_bench_run_00.png
graph:     build-bonsai-int1\bonsai-cuda133-eval\graph_correctness\graph\bonsai_bench_run_00.png
```

Both outputs matched exactly:

```text
SHA256 FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC
size 512x512
min/max 0/255
mean/std 121.918/66.171
unique RGB colors 134697
```

Trace showed:

```text
Bonsai CUDA graph warmup complete
Bonsai CUDA graph capture begin
Bonsai CUDA graph capture end
Bonsai CUDA graph instantiated
Bonsai CUDA graph launch update=true
Bonsai CUDA graph launch update=false
```

Text encode and VAE graphs were skipped because they do not contain Bonsai custom ops.

### Warm timing

512x512, 4 steps, 1 warmup, 5 measured runs, CUDA 13.3, current best Bonsai kernel gates, no tensor dumps, no line/family profiling.

| mode | denoise mean | denoise median | full mean | full median |
|---|---:|---:|---:|---:|
| non-graph | 1863.6 ms | 1847.0 ms | 2275.632 ms | 2246.673 ms |
| graph replay | 1845.6 ms | 1798.0 ms | 2230.647 ms | 2184.474 ms |

Normal timing gain:

```text
denoise median: -49 ms
full median:    -62 ms
```

This is a real but modest improvement, smaller than the raw launch/gap total suggested by the earlier Nsight Systems report.

### Nsight Systems

Reports:

```text
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_nongraph_denoise_range.nsys-rep
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_nongraph_denoise_range.sqlite
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_graph_denoise_range.nsys-rep
build-bonsai-int1\bonsai-cuda133-eval\nsys_bonsai_512_graph_denoise_range.sqlite
```

Measured denoise range comparison from the same CUDA 13.3 graph-enabled build:

| metric | non-graph | graph replay |
|---|---:|---:|
| denoise NVTX wall | 1834.590 ms | 1944.846 ms |
| CUDA kernel rows | 4960 | 0 reported as kernels; replay appears as graph trace |
| CUDA graph trace rows | 0 | 4 |
| graph trace GPU time | 0 | 1710.193 ms |
| CUDA launch API calls | 4960 | 4 |
| CUDA launch API time | 178.818 ms | 3.356 ms |
| memcpy API calls | 64 | 28 |
| memcpy API time | 9.887 ms | 8.161 ms |

Nsight graph tracing itself adds overhead and changes how replayed kernels are represented, so the Nsight graph run was slower in wall time than the non-graph Nsight run. The useful conclusion from Nsight is structural: graph replay collapses per-kernel launch APIs to four graph launches for the 4-step denoise, but total denoise remains dominated by GPU graph/kernel execution.

### Result

CUDA Graph replay is feasible and correct for the fixed Bonsai benchmark lane, but it is not the main remaining performance lever.

Recommendation:

```text
Keep SDCPP_BONSAI_CUDA_GRAPH_DENOISE as an opt-in benchmark prototype. Do not make it default yet. The next performance work should return to measured kernel execution time, especially single_blocks.linear1 and linear2 layout/scheduling, or consider deeper graph/sampler residency only if the goal is to remove host copies and sampler loop overhead rather than just launch API calls.
```

## Current best runtime profile and cleanup plan

This is the consolidation checkpoint after the quality/parity fixes, CUDA 13.3 side-by-side evaluation, Tensor Core kernel family work, Nsight diagnostics, and CUDA Graph replay prototype.

### Profile artifacts

```text
build-bonsai-int1\bonsai-cuda133-eval\best_known_bonsai_profile.md
scripts\bench_bonsai_int1_cuda133.ps1
build-bonsai-int1\bonsai-cuda133-eval\cleanup_commit_plan.md
```

The best known profile is CUDA 13.3 with Bonsai INT1 enabled and these kernel gates:

```powershell
$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1 = '1'
$env:SDCPP_MODEL_FAMILY_HINT = 'bonsai'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2 = '1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK = '1'
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT = '3'
$env:SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC = '1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0 = '1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2 = '1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_QKV = '1'
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ = '1'
```

CUDA Graph replay remains optional:

```powershell
$env:SDCPP_BONSAI_CUDA_GRAPH_DENOISE = '1'
```

The following gates remain disabled by default because they were correct but slower, or otherwise did not improve full denoise:

```text
SDCPP_BONSAI_INT1_LINEAR1_BANKFIX
SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM
SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX
SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B
SDCPP_BONSAI_INT1_TILED_KERNEL
```

### Final sanity run

Command:

```powershell
scripts\bench_bonsai_int1_cuda133.ps1 -Runs 3 -Warmup 1 -OutputDir build-bonsai-int1\bonsai-cuda133-eval\final_sanity_nongraph
scripts\bench_bonsai_int1_cuda133.ps1 -Graph -Runs 3 -Warmup 1 -OutputDir build-bonsai-int1\bonsai-cuda133-eval\final_sanity_graph
```

Results:

| mode | denoise median | full median | first image SHA256 | fail-fast/nonfinite |
|---|---:|---:|---|---|
| non-graph | 1806.000 ms | 2175.370 ms | `FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC` | false |
| graph replay | 1863.000 ms | 2247.203 ms | `FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC` | false |

Output images:

```text
build-bonsai-int1\bonsai-cuda133-eval\final_sanity_nongraph\bonsai_bench_run_00.png
build-bonsai-int1\bonsai-cuda133-eval\final_sanity_graph\bonsai_bench_run_00.png
```

Both first images match byte-for-byte and match the earlier known-good visually plausible 512 output. VRAM after the run was:

```text
0.9/16.0 GB (15.0 free)
```

### Recommendation

Keep CUDA Graph replay as opt-in. The graph prototype is correct and Nsight Systems proved it collapses launch API overhead, but the short final sanity sample did not beat non-graph timing. The remaining performance ceiling is primarily GPU kernel execution, with launch overhead now a secondary and mostly solved problem when graph replay is enabled.

The next pass should be cleanup, review, and commit planning rather than more optimization. Further performance research should be separated into a later branch or clearly scoped follow-up:

- source-backed Triton-like shared layout / generated-kernel strategy for the remaining dominant kernels
- CUDA Tile C++ only if a future smoke emits MMA/HMMA for the relevant shape
- broader CUDA Graph integration only after the benchmark prototype is cleaned up and the fixed-shape assumptions are made explicit

## Review and commit preparation

This pass did not add kernels or change model math. It produced review/commit ledgers, verified gate-off behavior, and ran one clean best-known 512 benchmark pair.

### Review artifacts

```text
build-bonsai-int1\bonsai-cuda133-eval\review_inventory.md
build-bonsai-int1\bonsai-cuda133-eval\gate_check.md
build-bonsai-int1\bonsai-cuda133-eval\review_sanity_512.md
build-bonsai-int1\bonsai-cuda133-eval\commit_split_plan.md
build-bonsai-int1\bonsai-cuda133-eval\precommit_risks.md
```

### Current performance checkpoint

CUDA 13.3 remains the preferred performance build, but should not become a hard runtime assumption. The best earlier sample was around 1.65-1.8 seconds denoise for 512x512, 4 steps. The review-prep sample was slower:

| mode | denoise median | full median | first image SHA256 | fail-fast/nonfinite |
|---|---:|---:|---|---|
| graph off | 2070.000 ms | 2498.678 ms | `FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC` | false |
| graph on | 1937.000 ms | 2329.285 ms | `FC60DE4D651DDD46D51A5587ACE01284AC72CCA071547C2F8FCE97F3F2D2DFCC` | false |

The images still match byte-for-byte and match the known-good visually plausible output. This was treated as timing variance; no optimization was attempted.

### Gate check

Gate-off smoke:

```powershell
build-bonsai-int1\bin\sd-bonsai-gemlite-int1-smoke.exe --pack F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt --summary-only
```

with Bonsai env vars cleared returned:

```text
exit_code=3
Bonsai GemLite INT1 spike is disabled. Set SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1=1.
```

Runtime attach, Bonsai schedule behavior, CUDA Graph replay, tensor dumps, oracle replay, and failed kernel branches are all controlled by env gates.

### Keep

- Bonsai state_dict indexer and runtime representation.
- Bonsai runtime mapping and BF16 support tensor loading.
- Stream-correct Bonsai custom-op launch path.
- Final-layer modulation order and Bonsai/Prism scheduler parity fixes.
- Current best Bonsai INT1 kernel gates:
  - linear1 TC V2 prepack-v3
  - linear2 A-only async
  - img_mlp0 TC
  - img_mlp2 TC
  - QKV TC
  - attention projection TC
- Bench/smoke tools and Python oracle diagnostics, subject to cleanup.
- CUDA Graph replay as opt-in only.

### Avoid committing

- `.nsys-rep`, `.sqlite`, `.ncu-rep`
- PNG outputs
- SASS dumps
- huge stdout/stderr logs
- `__pycache__/`
- tensor dumps and oracle outputs
- unrelated Lens/Z/Anima files unless split into their own commits

### Failed kernels off by default

```text
SDCPP_BONSAI_INT1_LINEAR1_BANKFIX
SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM
SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX
SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B
SDCPP_BONSAI_INT1_TILED_KERNEL
```

### Commit recommendation

Do not commit the full dirty tree as one change. The recommended first commit is only:

```text
Bonsai state_dict indexer and runtime representation
```

Use:

```text
build-bonsai-int1\bonsai-cuda133-eval\commit_split_plan.md
build-bonsai-int1\bonsai-cuda133-eval\precommit_risks.md
```

Current blockers before committing:

- separate unrelated Lens/Z/Anima changes from the Bonsai work
- decide whether `SD_BONSAI_GEMLITE_INT1_SPIKE` should default off in CMake
- decide whether `scripts\bench_bonsai_int1_cuda133.ps1` should keep machine-local defaults or remain uncommitted/build-local
- review noisy Bonsai `linear_call` logging before shared use
