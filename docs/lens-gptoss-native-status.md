# Lens GPT-OSS Native Text Encoder Status

Date: 2026-05-23

This note freezes the current native Lens GPT-OSS text encoder prototype state. The current recommendation is to stop long hidden-state archaeology and treat native-tolerant conditioning as the experimental non-Python conditioning path.

## What Works

- Token embedding parity passes with `max_diff=0`.
- One representative MXFP4 expert decode proof passes with `max_diff=0`.
- GPT-OSS RMSNorm parity is fixed.
- GPT-OSS q/k/v/o projection parity is fixed with cuBLASLt BF16 fused-bias linear.
- GPT-OSS split-half RoPE parity is fixed.
- GPT-OSS attention-with-sinks is source-faithful enough for the layer-0 gate.
- CUDA BF16 router logits and canonical top-k comparison work.
- Layer 0 and sequential 0->1 pass after the MoE cast fix.
- Oracle-router through feature_1/layer11 passes, proving structure through the first two captured features.
- Native-tolerant through all capture layers emits a complete `lens_cond_v1` bundle.
- CUDA MXFP4 dequant-to-device plus batched expert MoE reduced native text encoder runtime to production-usable smoke times.
- The emitted native-tolerant conditioning drives the existing native sd.cpp Lens denoiser/VAE to a coherent 256x256 image.
- A staged internal API wrapper now exists for the B1.26 native-tolerant path:
  - `src/lens_gptoss_text_encoder.hpp`
  - `src/lens_gptoss_text_encoder.cpp`
  - `sd-lens-text-encoder-native-smoke`

## Current Limits

- Native-router exact parity hard-fails early from BF16 router branch sensitivity.
- The known first native-router failure is layer 2 token 78, a near-boundary top-k change.
- Native-tolerant drift grows by layer11 and later captures; it is not a correctness pass.
- Tokenization is still bootstrapped from exported `input_ids` and masks.
- The native API is still an experimental internal staging path, not a public `stable-diffusion.h` surface.
- The current refactor keeps diagnostic code private to the native text encoder translation unit. The old kitchen-sink diagnostic smoke remains available for oracle/debug modes.

## B1.14 Checkpoint

Command log:

- `build/diagnostics/lens_b114_native_tolerant_cond.stdout.log`
- `build/diagnostics/lens_b114_native_tolerant_256_transformer.stdout.log`
- `build/diagnostics/lens_b114_native_tolerant_256_vae.stdout.log`

Artifacts:

- `build/diagnostics/lens_cond_v1_native_tolerant_robot_128.safetensors`
- `build/diagnostics/lens_b114_native_tolerant_256_latent.npy`
- `build/diagnostics/lens_b114_native_tolerant_256.png`

Native-tolerant feature diffs against available oracle data:

| Feature | Layer | Shape | Max diff | Mean diff |
| --- | ---: | --- | ---: | ---: |
| feature_0 | 5 | 1x31x2880 | 4 | 0.0352873 |
| feature_1 | 11 | 1x31x2880 | 20 | 0.3216 |
| feature_2 | 17 | 1x31x2880 | 96 | 1.83712 |
| feature_3 | 23 | 1x31x2880 | 464 | 8.57838 |

Image result:

- `build/diagnostics/lens_b114_native_tolerant_256.png`
- 256x256 RGB, nonblank
- SHA256 `38E6B0EAD935127F01D4AB872D29508A8C1DC9164C2B80BDC169D24F03059A45`
- Visually coherent for the robot workbench prompt.

## Recommendation

Do not continue exact native-router drift chasing right now. The native-tolerant path is fast enough for integration smokes, and tokenizer work can now be separated from math/runtime work.

## B1.26 Baseline

Command log:

- `build/diagnostics/lens_b126_emit.stdout.log`
- `build/diagnostics/lens_b126_emit_256_transformer.stdout.log`
- `build/diagnostics/lens_b126_emit_256_vae.stdout.log`

Artifacts:

- `build/diagnostics/lens_cond_v1_native_tolerant_robot_128_b126.safetensors`
- `build/diagnostics/lens_b126_emit_256.png`

Runtime:

- native-tolerant text encoder emit-cond: `9.47s`
- MXFP4 raw weight read: `8.33s -> 0.026s`
- layer other: `11.60s -> 2.64s`

Hashes:

- cond SHA256: `45ACAAC1B7F7F72268B24A24CA24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`
- image SHA256: `64A76437EDD391BD999C4426EECE7E8101C8D17D697622C335D83D62A4F4C838`

Supported mode:

- `router_mode=native-tolerant`
- `moe_backend=cuda-batched-expert-matmul`
- `cache_upload=cuda-mxfp4-dequant`
- `max_seq_len=128`
- `txt_offset=97`
- bootstrap `input_ids` and masks from the oracle bundle

## B2.0 Native API Staging

The B2.0 staging API exposes the current native-tolerant encode path without requiring the old diagnostic CLI:

- `sd_lens_text_encoder_create`
- `sd_lens_text_encoder_load`
- `sd_lens_text_encoder_encode`
- `sd_lens_text_encoder_free`

The minimal API smoke is:

```powershell
build\codex\bin\sd-lens-text-encoder-native-smoke.exe `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --bootstrap-oracle build\diagnostics\lens_text_encoder_oracle_robot_128_layer11_debug `
  --output build\diagnostics\lens_cond_v1_native_api_robot_128.safetensors
```

B2.0 native API result:

- cond SHA256: `45ACAAC1B7F7F72268B24A24CA24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`
- layer summary runtime: `9.23s`
- wrapper encode runtime: `11.02s`
- VRAM before text encoder create/load: `1.28 GiB used`
- VRAM after text encoder encode: `2.88 GiB used`
- VRAM after `sd_lens_text_encoder_free`: `1.29 GiB used`

The staged image smoke uses the emitted API conditioning, then exits/releases the text encoder runtime before starting the transformer and VAE stages. The 256 output remains hash-stable:

- image: `build/diagnostics/lens_b20_native_api_256.png`
- image SHA256: `64A76437EDD391BD999C4426EECE7E8101C8D17D697622C335D83D62A4F4C838`

Current next steps:

- integrate the native text encoder lifecycle into a real `sd_ctx_t` path after the staging API is stable
- implement native tokenizer later
- optionally optimize the remaining dense projection setup bucket
- validate 512 before any 1024 native-conditioning image test

## B2.1 Staged In-Memory E2E Smoke

B2.1 adds an experimental staged smoke for the full native Lens path:

- native GPT-OSS text encoder API
- in-memory `LensCondV1Native` handoff
- explicit text encoder release
- native Lens transformer/denoiser
- Lens VAE decode

New/updated targets:

- `sd-lens-native-e2e-smoke`
- `sd-lens-transformer-smoke-api`

The transformer smoke API is kept as a smoke-only DLL and loaded after
`sd_lens_text_encoder_free`. This avoids co-linking the two large experimental
smokes in the process image while preserving an in-memory condition handoff into
the transformer stage.

Command log:

- `build/diagnostics/lens_b21_native_e2e.stdout.log`
- `build/diagnostics/lens_b21_native_e2e.stderr.log`

Artifacts:

- `build/diagnostics/lens_cond_v1_native_e2e_robot_128.safetensors`
- `build/diagnostics/lens_b21_native_e2e_256.png`
- `build/diagnostics/lens_b21_native_e2e_256_latent.npy`
- `build/diagnostics/lens_b21_native_e2e_256_packed.npy`

Hashes:

- cond SHA256: `45ACAAC1B7F7F72268B24A24CA24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`
- image SHA256: `64A76437EDD391BD999C4426EECE7E8101C8D17D697622C335D83D62A4F4C838`

Runtime from the B2.1 smoke:

- text encoder encode: `11.34s`
- transformer/denoiser wall: `25.45s`
- VAE wall: `5.91s`
- VAE decode: `5.66s`

Lifecycle:

- `sd_lens_text_encoder_free` runs before the transformer smoke API DLL is
  loaded and before the Lens transformer context is created.
- Text encoder state and Lens transformer weights are not resident together.
- In-process numeric VRAM snapshots are disabled in this smoke because early
  attempts to query CUDA/`nvidia-smi` from the combined smoke path destabilized
  the process; lifecycle stage markers remain in the log and external VRAM
  checks should be used for numeric measurements.

## B2.2 Internal Staged Runtime Lane

B2.2 moves the B2.1 smoke-to-smoke handoff into an internal staged runtime
lane while keeping the path experimental and bootstrap-token based.

New internal files:

- `src/lens_staged_pipeline.hpp`
- `src/lens_staged_pipeline.cpp`
- `src/lens_transformer_runtime.hpp`
- `src/lens_transformer_runtime.cpp`

These implementation files are excluded from the stable `stable-diffusion.dll`
source sweep and are compiled into the experimental smoke target only.

New smoke target:

- `sd-lens-staged-native-smoke`

The staged objects make the lifecycle explicit:

- `sd_lens_pipeline_ctx`
- `sd_lens_text_encoder_stage`
- `sd_lens_transformer_stage`
- `sd_lens_vae_stage`
- `sd_lens_cond_v1_native`

The clean staged smoke runs:

1. native GPT-OSS text encoder API
2. in-memory `sd_lens_cond_v1_native` handoff
3. `sd_lens_text_encoder_free`
4. internal Lens transformer runtime
5. Lens VAE decode

The B2.1 DLL bridge is no longer used by the B2.2 smoke. The transformer
runtime is still backed by the existing experimental transformer-smoke core,
now linked as an internal static runtime core instead of late-loaded through
`sd-lens-transformer-smoke-api.dll`. Pulling the denoiser classes fully out of
the diagnostic smoke remains a later cleanup; the math path is unchanged.

Command log:

- `build/diagnostics/lens_b22_staged_native.stdout.log`
- `build/diagnostics/lens_b22_staged_native.stderr.log`

Artifacts:

- `build/diagnostics/lens_cond_v1_b22_staged_native.safetensors`
- `build/diagnostics/lens_b22_staged_native_256.png`
- `build/diagnostics/lens_b22_staged_native_256_latent.npy`
- `build/diagnostics/lens_b22_staged_native_256_packed.npy`

Hashes:

- cond SHA256: `45ACAAC1B7F7F72268B24A24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`
- image SHA256: `64A76437EDD391BD999C4426EECE7E8101C8D17D697622C335D83D62A4F4C838`

Runtime from the B2.2 smoke:

- text encoder encode: `11.59s`
- transformer/denoiser wall: `27.70s`
- VAE wall: `6.15s`
- VAE decode: `5.91s`

Lifecycle:

- Text encoder load and encode complete before the transformer runtime starts.
- `sd_lens_text_encoder_free` runs before the transformer stage.
- The B2.2 smoke reports `text_encoder_released_before_transformer=true`.
- The B2.2 smoke reports `transformer_smoke_api_dll=false`.
- External post-run VRAM check returned `2.8/16.0 GB (13.2 free)`.

Still experimental:

- `native-tolerant` routing is the supported mode for this path.
- Prompt handling still uses bootstrap `input_ids` and masks from the oracle
  bundle; there is no native tokenizer yet.
- The CLI is smoke-only and not a stable public API.
- The transformer runtime is still smoke-core derived internally.

Recommended next candidates:

- native tokenizer
- stable internal `sd_ctx_t` API surface after the smoke-only lane hardens
- 512 image validation
- later extraction of transformer denoiser classes out of the diagnostic smoke
- dense setup optimization only if integration needs it

## B2.4 Python Benchmark Correction

B2.4T3 creates a fresh Microsoft Lens-style Python MXFP4 environment for the
fair Python comparison instead of using the older Torch 2.6/cu118 venv.

Environment:

- Python: `3.12.11` (`3.12.12` was not installed locally)
- Torch: `2.11.0+cu126`
- CUDA: `12.6`
- TorchVision: `0.26.0+cu126`
- Transformers: `5.8.0`
- Diffusers: `0.38.0`
- Accelerate: `1.13.0`
- Kernels: `0.14.0`
- Triton: `3.6.0` via `triton-windows`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- `torch.cuda.get_device_properties(0).shared_memory_per_block_optin` exists

The old Torch 2.6/cu118 Triton probe is not a speed datapoint. It is an invalid
environment/API mismatch for the Microsoft Lens MXFP4 path.

Valid 256 benchmark rows:

| Path | Text encoder | Transformer / Denoise | VAE | Cold-ish total | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Python staged MXFP4/Triton-Windows/Torch2.11cu126 256 | `15.92s` | `0.89s` denoise / `21.44s` load-context | `0.17s` | `47.80s` | valid Python MXFP4 |
| Native staged 256 before B2.5 | `11.59s` | `27.70s` wall | `6.15s` | `~45.44s` | valid native staged |
| Python staged BF16/dequant fallback Torch2.6cu118 256 | `282.31s` | `3.43s` | `0.36s` | `324.41s` | valid fallback only |

Conclusion:

- Native GPT-OSS text encoding is competitive with and faster than the valid
  Python MXFP4 text encoder on this machine.
- The next native speed target is not GPT-OSS text encoding.
- The native bottlenecks are Lens transformer staging/runtime and the VAE
  backend.

## B2.5 Transformer / VAE Speed Split

B2.5 keeps the text encoder, routing, MoE, and transformer math unchanged and
classifies the post-conditioning bottleneck.

Native 256 staged transformer split from the B2.2 baseline:

- transformer context/load: `13.7582s`
- transformer total compute loop: `12.806s`
- scheduler flow: `0.0006382s`
- unpack: `0.0000507s`
- runner input copy/upload: `10.6483s`
- runner compute: `0.328739s`
- runner graph time: about `0.233s` from per-step graph timings
- streamed tensors: `4992`
- streamed bytes: `65.26GB`
- host cached weights: `true`
- GPU resident transformer weights: `false`

Native 512 staged transformer split from B2.3 is similar:

- transformer context/load: `13.7428s`
- transformer total compute loop: `13.6979s`
- runner input copy/upload: `9.19475s`
- runner compute: `0.876003s`
- streamed bytes: `65.26GB`

The transformer gap is therefore dominated by repeated block weight/input
movement and non-resident execution, not raw CUDA math.

VAE backend finding:

- The B2.2/B2.3 staged path hardcoded `keep_vae_on_cpu=true`.
- Logs showed `VAE Autoencoder: Using CPU backend`.
- Lens advertises `supports_gpu_latent_decode=false`, so the public GPU image
  output capability remains disabled, but the direct compatibility graph can
  execute on the CUDA backend when the VAE is not forced to CPU.

VAE-only 256 decode from the B2.2 latent:

| Backend | Decode graph | Output |
| --- | ---: | --- |
| CPU | `5571ms` | SHA exact vs B2.2 |
| CUDA | `212ms` | max RGB diff `1`, mean abs diff `0.100876` vs B2.2 |

B2.5 switches the staged smoke VAE stage to CUDA by setting
`keep_vae_on_cpu=false`.

B2.5 staged 256 with CUDA VAE:

- cond SHA256: `45ACAAC1B7F7F72268B24A24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`
- image SHA256: `E1A5612614E71F70FF499F48D1BB8C3C0E1472C9824AC758AE5FB3B49611E1A1`
- image diff vs B2.2 CPU VAE image: max RGB diff `1`, mean abs diff `0.100876`
- text encoder: `11.1055s`
- transformer wall: `23.9886s`
- transformer context/load: `13.0665s`
- transformer compute loop: `9.78798s`
- runner input copy/upload: `8.12142s`
- runner compute: `0.314602s`
- VAE wall: `0.336604s`
- VAE decode: `0.0886706s`

Recommendation:

- Keep CUDA VAE for the staged native smoke; it is a narrow backend switch with
  bounded pixel-level drift and a large speed win.
- Next optimization should target transformer residency/streaming:
  device-resident transformer weights or a block cache that avoids streaming
  `~65GB` per four-step 256 generation.
- Do not return to GPT-OSS text encoder optimization unless integration changes
  regress the frozen conditioning SHA.

## B2.6 Transformer Residency Probe

B2.6 tested whether native Lens transformer time could be reduced by keeping block weights resident on the GPU across denoise steps. The memory estimate rejected full GPU residency on the 16 GB RTX 4080 SUPER: the transformer payload is 16,416,900,608 bytes (15.29 GiB), with 48 blocks at about 339,887,104 bytes (324.14 MiB) each plus about 0.095 GiB of non-block weights. That leaves insufficient room for activations, compute buffers, VAE, and normal CUDA workspace.

A bounded `--transformer-residency gpu-window --window-blocks N` prototype was added. It pre-uploads the first N transformer blocks as CUDA backend resources and reuses them across the four denoise steps. This preserves math and does not change conditioning or VAE behavior.

Results:

- Baseline B2.5 transformer wall: about 23.99s.
- Baseline B2.5 transformer loop: about 9.79s.
- Baseline runner input copy/upload: about 8.12s.
- Baseline streamed bytes: about 65.26 GB.
- `gpu-window`, 4 blocks, full staged 256: cond hash unchanged, PNG hash unchanged. Transformer wall was 24.38s, so the added pre-upload did not improve end-to-end wall. Runner input copy dropped to 7.77s and streamed bytes dropped to 59.82 GB.
- `gpu-window`, 12 blocks, transformer-only: latent hash matched the 4-block run. Transformer loop dropped to 8.51s, runner input copy dropped to 6.84s, and streamed bytes dropped to 48.94 GB. Resident upload cost was 0.67s for 4.08 GB.

Conclusion: partial block weight residency works and preserves output, but it is not enough. The current runner still rebuilds per-block GGML graphs, copies resident weights into graph buffers, and materializes activations back to CPU between blocks. A meaningful transformer speedup likely requires a larger transformer runtime change that keeps activations and selected weights in backend resources across blocks, or a purpose-built Lens transformer runner. Do not keep expanding the window cache as the main optimization path.

## B2.8 One-Block Persistent-Weight Runner Proof

B2.8 tested whether one Lens transformer block can use persistent CUDA backend weight tensors directly instead of copying those weights into each per-block runner graph.

Copy boundary found:

- `LensBlockCudaRunner::input()` previously used `make_backend_input()` for resident weights.
- `GGMLRunner::make_backend_input()` duplicates the persistent tensor metadata into the current `compute_ctx` and records `backend_tensor_source_map[input] = resource.tensor`.
- `GGMLRunner::copy_data_to_backend_tensor()` later calls `ggml_backend_tensor_copy(src, dst)` for every entry in `backend_tensor_source_map`.
- Those bytes are counted in `pending_backend_input_bytes()` and show up as runner input copy/upload.
- Therefore the B2.6 `gpu-window` path had persistent source tensors, but still copied them into per-run graph input tensors.

Prototype:

- Added a direct alias path for persistent block weights.
- Mode: `--transformer-residency persistent-block --persistent-block 0`.
- Block 0 weights are uploaded once before the denoise loop into persistent backend resources.
- For block 0 only, graph inputs reference those backend tensors directly via `make_backend_input_alias()` instead of using the copy path.
- All other blocks remain on the existing host-cached streaming path.

Result, transformer-only 256:

- Initial block0 resident upload: `0.123916s` for `339,887,104` bytes / 26 tensors.
- Latent SHA matched baseline: `0C31526449C51BF40E953009A4CEF64E63F37BDB599CEA85371DBCF29E483CC5`.
- Block0 repeated input copy bytes after aliasing: `7,665,696` total across 4 steps, which is dynamic input/position data only.
- Total runner input copy bytes: `66,048,114,272 -> 64,688,565,856`, a drop of `1,359,548,416` bytes, exactly `4 * 339,887,104`.
- Streamed bytes: `65,258,323,968 -> 63,898,775,552`.
- Transformer loop time in this run: `10.2914s`; this did not improve wall time because only one block was changed and the remaining per-block graph/copy/materialization path dominates.

Conclusion: persistent weight tensors are viable and can be referenced without per-step weight copies. The next implementation should be a `persistent-blocks` path with a safe resident byte cap, not the old `gpu-window` path. Full transformer residency remains unsafe on 16 GB, and activation residency is still a later phase.

## B2.19 / B2.20 Transformer Speed Mode

B2.19 stops hidden-state parity archaeology for the Lens transformer and
validates `gpu-full-bf16` as an image-level speed mode. This mode is not
hidden-parity exact and is not the default. It is an explicit speed/visual mode
for the staged smoke.

Mode names:

- Lower-level residency flag: `--transformer-residency gpu-full-bf16`
- User-facing staged smoke alias: `--transformer-speed-mode bf16-resident`

The staged smoke reports:

- `speed_mode=bf16-resident`
- `hidden_parity_exact=false`
- `image_validated_256=true`
- `image_validated_512=true`

Quality status:

- 256 `gpu-full-bf16` is coherent and slightly closer to the Python MXFP4 image
  than the old native stable path by simple pixel metrics.
- 512 `gpu-full-bf16` is visually near-identical to the old native image:
  mean absolute pixel diff `0.708`, PSNR `42.38`.
- Keep the old native/stable path as the conservative reference/quality mode.

Speed status:

- Native GPT-OSS text encoder already beats the valid Python MXFP4 text encoder
  on this machine: native about `11.6s`, Python MXFP4 about `15.9s`.
- CUDA VAE is close enough for the staged lane.
- `gpu-full-bf16` reduces the warm transformer loop substantially:
  - old native stable 256 loop: about `7.02s`
  - `gpu-full-bf16` 256 loop: about `2.13s` in B2.11 and `1.88s/1.72s`
    in the B2.20 repeat smoke
  - `gpu-full-bf16` 512 loop: about `5.96s`
- Cold staged runs still pay about `10s` of BF16 resident weight upload.

Current modes:

1. Stable/quality mode:
   - old native transformer path, or `persistent-blocks` when selected
   - slower and more conservative
2. Speed mode:
   - `--transformer-speed-mode bf16-resident`
   - maps to `--transformer-residency gpu-full-bf16`
   - image-coherent at 256 and 512
   - faster warm transformer loop

B2.20 warm-context proof:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --bootstrap-oracle build\diagnostics\lens_text_encoder_oracle_robot_128_layer11_debug `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 256 --width 256 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --repeat-generations 2 `
  --output build\diagnostics\lens_b220_bf16_resident_repeat2_256.png `
  --cond-out build\diagnostics\lens_cond_v1_b220_bf16_resident_repeat2.safetensors
```

The repeat smoke reuses the same in-memory `lens_cond_v1` and keeps the
transformer context plus BF16 resident weights alive across both generations.

Warm proof result:

- BF16 resident upload: `11.1488s`, paid once.
- Generation 0 transformer loop: `1.84279s`.
- Generation 1 transformer loop: `1.70849s`.
- Generation 0 streamed bytes: `0`.
- Generation 1 streamed bytes: `0`.
- Generation 0 resident upload paid this generation: `11.1488s`.
- Generation 1 resident upload paid this generation: `0s`.
- Generation 0 runner input copy: `0.444056s`.
- Generation 1 runner input copy: `0.440889s`.
- Final image SHA256:
  `9A02E7CFD620043C9A7399FE129CEEAF03ABEE8E88019FB11FBF0FE35679738B`.
- Final image hash matches the prior B2.11 `gpu-full-bf16` 256 output.
- Conditioning SHA256:
  `45ACAAC1B7F7F72268B24A24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`.

Next likely step: integrate this warm transformer context into the internal
`sd_ctx_t` staged lane so prompt-time speed benefits from the resident BF16
transformer instead of paying the upload cost for every image.

## B2.21 Internal Warm Runtime Lane

B2.21 moves the warm reuse proof into the internal staged Lens runtime lane.
This is still experimental and internal; it does not add a stable public API and
does not make `bf16-resident` the default.

Internal runtime shape:

- `sd_lens_staged_runtime`
  - load text encoder
  - encode bootstrap condition into in-memory `lens_cond_v1`
  - unload text encoder before transformer preparation
  - prepare transformer stage with explicit speed/residency mode
  - run repeated transformer generations against the same condition
  - release transformer stage explicitly

The current warm mode is same-condition reuse. New prompt strings still require
the staged lifecycle because native tokenizer and multi-prompt context handling
are not integrated yet.

B2.21 command:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --warm-runtime `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --bootstrap-oracle build\diagnostics\lens_text_encoder_oracle_robot_128_layer11_debug `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 256 --width 256 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --repeat-generations 2 `
  --output build\diagnostics\lens_b221_warm_runtime_bf16_resident_repeat2_256.png `
  --cond-out build\diagnostics\lens_cond_v1_b221_warm_runtime_bf16_resident_repeat2.safetensors
```

B2.21 result:

- Conditioning SHA256:
  `45ACAAC1B7F7F72268B24A24BF554ECBDAD8A241DCF6C2D7E879A9EE2D5A`.
- Image SHA256:
  `9A02E7CFD620043C9A7399FE129CEEAF03ABEE8E88019FB11FBF0FE35679738B`.
- Image stats: RGB `256x256`, mean `95.777344`, std `77.995457`,
  nonblank `65516/65536`.
- Text encoder wrapper runtime: `11.7018s`.
- Transformer context load: `13.5478s`.
- BF16 resident upload: `11.1173s`, paid once.
- Generation 0 transformer loop: `1.88004s`.
- Generation 1 transformer loop: `1.71808s`.
- Generation 0 resident upload paid this generation: `11.1173s`.
- Generation 1 resident upload paid this generation: `0s`.
- Generation 1 reports `resident_upload_reused=true`.
- Streamed bytes: `0` for both generations.
- VAE wall: `0.369574s`, decode `0.12556s`.
- Final VRAM after cleanup returned to about `3.2/16.0 GB`.

Lifecycle proof:

- The text encoder is released before transformer preparation.
- The staged smoke reports `transformer_loaded_after_text_encoder_release=true`.
- The transformer stage is explicitly released before VAE load/decode.

Remaining limitations at B2.21:

- Bootstrap input IDs/mask only; no native tokenizer.
- `bf16-resident` is an explicit speed mode, not the default.
- Hidden-state parity is not exact in speed mode.
- Same-condition multi-seed reuse is proven; new-prompt warm strategy remains
  future work.

## B2.22 Native Tokenizer / Prompt Bootstrap Replacement

B2.22 adds a narrow native GPT-OSS tokenizer path for the Lens staged smoke.
This replaces the Python bootstrap token oracle for the validated Lens no-tools
prompt path, while keeping the text encoder, transformer, VAE, and speed-mode
math unchanged.

Source of truth:

- Lens prompt rendering: `F:\Paralol\local\Lens\lens\pipeline.py`
- Tokenizer files:
  `F:\Paralol\local\models\microsoft\Lens-Turbo\tokenizer\tokenizer.json`
  and `chat_template.jinja`
- Exported spec:
  `build\diagnostics\lens_b222_tokenizer_spec.json`

The Python Lens path renders a no-tools conversation with:

- system message: the Lens image-description instruction
- user message: the prompt string
- assistant analysis message:
  `Need to generate one image according to the description.`
- `tokenizer.apply_chat_template(..., tokenize=False,
  add_generation_prompt=False)`
- `text.split("<|return|>")[0]`
- tokenizer call with right padding, truncation, `max_length=128`, and
  `add_special_tokens=True`
- `txt_offset=97`, leaving `trimmed_seq_len=31` for the robot prompt

Implementation choice:

- Option A, generic Hugging Face tokenizers C++, was not used because the fork
  does not already depend on it.
- Option B was implemented as a contained native tokenizer for this GPT-OSS
  tokenizer JSON and the validated Lens no-tools prompt template.
- The implementation loads vocab, special tokens, and BPE merge ranks from
  `tokenizer.json`, uses the GPT-style byte encoder and byte-level BPE, and
  emits padded/truncated `input_ids` plus `attention_mask`.

Tokenizer-only command:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --tokenizer-only `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --bootstrap-oracle build\diagnostics\lens_text_encoder_oracle_robot_128_layer11_debug `
  --prompt "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" `
  --chat-current-date 2026-05-23
```

Tokenizer parity result:

- `tokenizer=native_gptoss_tokenizer_json`
- `rendered_chars=707`
- `raw_seq_len=128`
- `trimmed_seq_len=31`
- `txt_offset=97`
- `mask_ones=128`
- `input_ids_equal_oracle=true`
- `attention_mask_equal_oracle=true`
- `trimmed_attention_mask_equal_oracle=true`
- `chat_current_date=2026-05-23`

Prompt staged smoke command:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --warm-runtime `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --prompt "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" `
  --chat-current-date 2026-05-23 `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 256 --width 256 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --output build\diagnostics\lens_b222_prompt_no_oracle_bf16_resident_256.png `
  --cond-out build\diagnostics\lens_cond_v1_b222_prompt_no_oracle.safetensors
```

Prompt staged result:

- No `--bootstrap-oracle` was required.
- Text encoder runtime: `11.5575s`.
- Transformer context load: `13.4695s`.
- BF16 resident upload: `11.1238s`.
- Transformer loop: `1.79219s`.
- Streamed bytes: `0`.
- VAE wall: `0.370334s`, decode `0.126854s`.
- Image SHA256:
  `9A02E7CFD620043C9A7399FE129CEEAF03ABEE8E88019FB11FBF0FE35679738B`.
- Image stats: RGB `256x256`, mean `95.777344`, std `77.995457`,
  nonblank `65516/65536`.
- Conditioning artifact SHA256:
  `BA90F29E0FA4E498EA7090C2841F50DD9EC1AAD0324760D1061A5B2F14500D70`.

The B2.22 conditioning file SHA differs from B2.21 because the diagnostic
metadata now records the native tokenizer/prompt source instead of bootstrap
oracle source. The conditioning tensor payloads are bit-identical to B2.21:
`feature_0`, `feature_1`, `feature_2`, `feature_3`, and `attention_mask` all
compare equal with max and mean diff `0`.

Current tokenizer limitations:

- The tokenizer implementation is intentionally scoped to the Lens GPT-OSS
  tokenizer JSON and the no-tools Lens prompt-rendering path.
- The pre-tokenizer is an ASCII-focused implementation of the GPT-OSS
  byte-level pattern, validated on the robot prompt.
- Date-sensitive prompt rendering must be pinned with `--chat-current-date` for
  reproducible token and image hashes.
- Native tokenizer support is not yet exposed as a stable public API.

Recommended next checkpoint: either run 1024 validation now that prompt-string
support is present, or clean up the internal staged API around prompt strings
before exposing it beyond the smoke lane.

## B2.23 1024 Prompt-Driven Validation

B2.23 validates the prompt-driven native staged path at official 1024 scale.
The smoke-only size guard was widened from 256/512 to 256/512/1024; no tokenizer,
text encoder, transformer math, VAE, or default-mode behavior changed.

Command:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --warm-runtime `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --prompt "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" `
  --chat-current-date 2026-05-23 `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 1024 --width 1024 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --output build\diagnostics\lens_b223_prompt_bf16_resident_1024.png `
  --cond-out build\diagnostics\lens_cond_v1_b223_prompt_1024.safetensors
```

Result:

- Image:
  `build\diagnostics\lens_b223_prompt_bf16_resident_1024.png`
- Image SHA256:
  `00D171C3C71F988F6439790F74AFFB3BEBC213CB093976ACD9E190F892B3C975`
- Image stats: RGB `1024x1024`, mean `97.692817`, std `59.400884`,
  nonblank `1048511/1048576`.
- Visual verdict: coherent glass robot on a wooden workbench.
- Conditioning SHA256:
  `BA90F29E0FA4E498EA7090C2841F50DD9EC1AAD0324760D1061A5B2F14500D70`.
- Conditioning tensors match the 256 prompt-driven B2.22 condition exactly:
  `feature_0`, `feature_1`, `feature_2`, `feature_3`, and `attention_mask`
  all compare equal with max and mean diff `0`.

Runtime:

- Tokenizer-only smoke wall: `1.506764s`, including process startup and
  tokenizer JSON load. In-run tokenizer time is not separately instrumented.
- Text encoder: `12.1917s`.
- Transformer context load: `13.709s`.
- BF16 resident upload: `9.82064s`.
- Transformer loop: `27.632s`.
- Streamed bytes: `0`.
- VAE wall: `1.48152s`, decode `0.999139s`.
- Approximate staged total from reported stage timings: `66.10s`.
- Final VRAM returned to about `3.4/16.0 GB`.

Scaling against the B2.19 512 `gpu-full-bf16` result:

- Transformer loop: `5.96046s` at 512 -> `27.632s` at 1024.
- VAE wall: `0.598661s` at 512 -> `1.48152s` at 1024.
- VAE decode: `0.304449s` at 512 -> `0.999139s` at 1024.

Lifecycle:

- The prompt was tokenized natively.
- The text encoder emitted in-memory `lens_cond_v1` and was released before the
  transformer stage.
- The smoke reports `text_encoder_released_before_transformer=true` and
  `transformer_loaded_after_text_encoder_release=true`.
- `bf16-resident` remains an explicit speed mode, not the default.

Next likely checkpoint: internal staged prompt API cleanup or 1024
quality/reference comparison. Paralol host integration should remain deferred
until the internal API shape is cleaner.

## B2.24 Internal Staged Prompt API Cleanup

B2.24 turns the prompt-driven staged smoke path into a cleaner internal runtime
API. This is lifecycle/API cleanup only; tokenizer, text encoder math,
transformer math, VAE math, and `bf16-resident` default behavior are unchanged.

Internal API shape:

- `LensPromptRequest`
  - model paths: text encoder, transformer, VAE
  - prompt fields: prompt, tokenizer dir, chat current date
  - generation fields: width, height, steps, cfg, seed
  - mode fields: speed mode, quality mode, transformer residency
  - diagnostics: optional condition output and latent/packed token output paths
- `LensPromptResult`
  - timing fields for text encoder, transformer, and VAE
  - image dimensions and RGB image buffer
  - in-memory `lens_cond_v1` and latent
  - lifecycle flags and warnings
- `sd_lens_staged_runtime`
  - `configure`
  - `encode_prompt`
  - `prepare_transformer`
  - `generate`
  - `decode_image`
  - `unload_transformer`
  - `destroy`
- Internal helper:
  - `sd_lens_run_prompt_request`

The staged smoke is now a thin CLI wrapper around `LensPromptRequest` and
`sd_lens_run_prompt_request`. It still owns argument parsing and PNG writing,
but model lifecycle, prompt encoding, transformer generation, and VAE decode are
inside the internal staged runtime layer.

Regression build:

```powershell
. F:\Paralol\scripts\codex_windows.ps1
Enter-ParalolVsDevShell
cmake --build build\codex --config Release --target sd-lens-staged-native-smoke --parallel 8
```

256 regression:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --warm-runtime `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --prompt "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" `
  --chat-current-date 2026-05-23 `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 256 --width 256 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --output build\diagnostics\lens_b224_prompt_bf16_resident_256.png `
  --cond-out build\diagnostics\lens_cond_v1_b224_prompt_256.safetensors
```

1024 regression:

```powershell
build\codex\bin\sd-lens-staged-native-smoke.exe `
  --warm-runtime `
  --text-encoder F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder `
  --prompt "a small glass robot standing on a wooden workbench, studio lighting, sharp focus" `
  --chat-current-date 2026-05-23 `
  --transformer F:\Paralol\local\models\microsoft\Lens-Turbo\transformer `
  --vae F:\Paralol\local\models\microsoft\Lens-Turbo\vae\diffusion_pytorch_model.safetensors `
  --height 1024 --width 1024 --steps 4 --seed 42 `
  --transformer-speed-mode bf16-resident `
  --output build\diagnostics\lens_b224_prompt_bf16_resident_1024.png `
  --cond-out build\diagnostics\lens_cond_v1_b224_prompt_1024.safetensors
```

Regression results:

| Size | Image SHA256 | Cond SHA256 | Text encoder | Transformer wall | Transformer loop | VAE wall/decode |
|---|---|---|---:|---:|---:|---:|
| 256 | `9A02E7CFD620043C9A7399FE129CEEAF03ABEE8E88019FB11FBF0FE35679738B` | `BA90F29E0FA4E498EA7090C2841F50DD9EC1AAD0324760D1061A5B2F14500D70` | `12.0133s` | `26.9155s` | `1.97571s` | `0.347676s / 0.121879s` |
| 512 | B2.19 `gpu-full-bf16` validated | B2.19 `gpu-full-bf16` validated | `12.6562s` | `32.0251s` | `5.96046s` | `0.598661s / 0.304449s` |
| 1024 | `00D171C3C71F988F6439790F74AFFB3BEBC213CB093976ACD9E190F892B3C975` | `BA90F29E0FA4E498EA7090C2841F50DD9EC1AAD0324760D1061A5B2F14500D70` | `11.4233s` | `50.1358s` | `26.0731s` | `1.12059s / 0.902403s` |

Post-cleanup image stats:

- 256: RGB `256x256`, mean `95.777344`, std `77.995457`,
  nonblank `65516/65536`.
- 1024: RGB `1024x1024`, mean `97.692817`, std `59.400884`,
  nonblank `1048511/1048576`.

Conditioning:

- `feature_0`, `feature_1`, `feature_2`, `feature_3`, and `attention_mask`
  compare exactly between the 256 and 1024 B2.24 prompt runs.
- Max and mean diff are `0` for all compared condition tensors.

Lifecycle:

- Prompt string path is first-class through `LensPromptRequest`.
- Bootstrap-oracle remains diagnostic only.
- The text encoder is released before transformer preparation.
- Regression logs report `text_encoder_released_before_transformer=true`,
  `transformer_loaded_after_text_encoder_release=true`, and
  `transformer_released=true`.
- `bf16-resident` is still explicit speed mode, not default.

Remaining limitations:

- No stable public API yet.
- Tokenizer support remains scoped to the Lens no-tools path with
  ASCII-focused validation.
- `bf16-resident` is image-validated but not hidden-parity exact.
- New prompt lifecycle is still staged; warm multi-prompt strategy is not yet
  designed.

Recommended next checkpoint: internal API exposure/prep for Paralol integration,
with a separate 1024 quality/reference comparison if image-quality confidence is
needed before integration.
