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
- Qwen-Image and Anima expose a separate decode bridge capability:
  `supports_gpu_latent_decode_bridge=true` /
  `supports_gpu_image_output_bridge=true`. That bridge consumes and returns GPU
  handles, but internally downloads the latent for the legacy Wan/Qwen VAE
  decode and re-uploads the image. Strict GPU-resident mode refuses it.
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
| Flux2 / Klein | 128 ch | 16 | Supported | Supported for the strict GPU-init I2I lane | Yes, via `ref_images` | TAEF2 verified | Flux2 edit/reference conditioning uses `ref_images`; plain I2I uses the GPU VAE Encode -> GPU init-latent sampler path. |
| Z-Image / Z-Anime | 16 ch | 8 | Supported | Supported for Z-Image smoke path | Not claimed | Not claimed | Z-Turbo handoff and Z-Anime image smoke pass through the strict GPU path. Reference/edit capability is intentionally not advertised yet. |
| Qwen-Image | 16 ch | 8 | Supported for text-only `cfg=1` strict sampler; true GPU VAE decode is not claimed | Compatibility bridge | Qwen edit path, not strict-resident | TAEHV compatible, not claimed here | `SDCPP_EXPERIMENTAL_QWEN_IMAGE_BACKEND=1` enables the strict sampler lane. Non-strict VAE bridge is reported separately; the local Qwen VAE image remains unaccepted. |
| Anima | 16 ch | 8 | Supported for text-only strict sampler; true GPU VAE decode is not claimed | Compatibility bridge | No | Not claimed | `SDCPP_EXPERIMENTAL_ANIMA_BACKEND=1` enables the strict sampler lane. Validated sampler methods: `euler`, `euler_a`, `er_sde`, `dpmpp_2m_sde_gpu`. Non-strict Wan/Qwen VAE bridge produces coherent diagnostics at realistic settings. |
| Lens | 32 ch public, packed to 128 ch FLUX.2 VAE layout | 16 for VAE decode | Not claimed | Not claimed | Not claimed | Not claimed | Phase 2/3/4 plus tiny-model Phase 5 parity, real-shard metadata load, guarded Phase 6 scaffold, real-weight streamed tiny transformer smoke, precomputed-conditioning-to-real-transformer smoke, tiny denoise-artifact VAE decode, tiny first FlowMatch step decode, tiny 4-step FlowMatch loop decode, public external-flow loop API, real streamed outputs through that API, public Lens latent unpack API, source-backed first-image fixture replay, corrected Lens pipeline scheduler parity, and external transformer trace replay: family metadata, CPU-only `lens_cond_v1` precomputed-conditioning handle loading, Lens-Turbo schedule trace building against the local pipeline's empirical-mu/custom-sigmas path, CPU Lens VAE-only decode parity, native Lens transformer math parity through a tiny two-block full-model oracle, metadata-only validation of real Lens transformer shards, one-process validation of the precomputed-conditioning lane boundaries, bounded CPU execution of one real Lens transformer block, streamed execution of all 48 real Lens transformer blocks on tiny synthetic tokens, streamed execution using `lens_cond_v1` features plus additive-mask conversion, unpatch/decode of a tiny transformer-produced Lens latent, one native schedule update before decode, all four tiny FlowMatch updates with model recomputation between steps, an sd.cpp-owned CPU flow loop for caller-supplied model-output tensors, public-API integration of the real streamed tiny outputs, public BSC packed-token to `1x32xHxW` latent conversion, source-backed packed start tokens/timestep/RoPE/mask replay into a 64px decoded PNG, and a 1024px real prompt-conditioned external-transformer trace decoded by sd.cpp. Validated with `sd-lens-cond-smoke`, `sd-lens-vae-smoke`, `sd-lens-transformer-smoke`, `sd-lens-phase6-smoke`, and Lens transformer header inspection; native text encoder, production transformer forward inside `sd_ctx_t`, full-resolution native transformer execution, and GPU sampler are not implemented yet. |
| Marigold IID | 8 ch | model-specific | Not supported | Not supported | N/A | N/A | Uses the dedicated intrinsic-image decomposition API. |

## Verification snapshot

Known verified paths at the time this matrix was written:

- SDXL T2I GPU sampled latent -> GPU VAE Decode -> caller-owned image download.
- SDXL I2I VAE Encode GPU latent -> non-strict KSampler init bridge -> isolated
  GPU VAE Decode.
- Z-Image Turbo T2I GPU sampled latent -> GPU VAE Decode at 512x1024 with
  `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, `res_multistep` / `simple`, CFG 1.0,
  8 steps. The strict path produces a coherent doc-style image, with no sampler
  bridge flags and caller-owned final download.
- Z-Anime Base T2I strict GPU sampled latent -> GPU VAE Decode at 512 with
  `euler_a` / `beta`, CFG `4.0`, 28 steps. The output is nonblank and coherent
  enough for fork-side image-quality acceptance.
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
- Anima DMDX 4-step distill LoRA strict T2I lane: diffusion-only LoRA accepted
  with pre-encoded conditioning, `980 / 980` LoRA tensors applied,
  `er_sde` / `simple`, CFG `1.0`, 4 steps, `SDCPP_EXPERIMENTAL_WAN_QWEN_VAE_GPU=1`,
  no sampler bridge flags, no VAE host copies, and coherent 512px output.
- Anima/Qwen-image Wan21 latent mean/std transforms are applied for
  diffusion-latent <-> VAE-latent conversion. The CPU compatibility decode path
  can write coherent Anima diagnostic images at realistic settings
  (`er_sde`, CFG 4.5, 30 steps), but GPU VAE image output is still not
  advertised for Anima/Qwen-image.
- Anima non-strict GPU-handle VAE bridge: sampled CUDA latent ->
  `sd_decode_gpu_latent_normal_gpu()` -> legacy Wan/Qwen VAE decode ->
  uploaded CUDA image handle. The returned image handle carries CPU bridge
  provenance flags and strict mode refuses this path.
- Lens VAE-only CPU parity: `sd-lens-vae-smoke` decodes a tiny 1x32x8x8 Lens
  latent with only `vae/diffusion_pytorch_model.safetensors`; the Diffusers
  `AutoencoderKLFlux2` oracle compare reported `max_abs_u8=1` and
  `mean_abs_u8=0.012126`.
- Lens scheduler parity: `compare_lens_scheduler_oracle.py` compares
  `sd_lens_turbo_build_schedule` against Diffusers
  `FlowMatchEulerDiscreteScheduler` using the local Lens pipeline's empirical
  `mu` and custom Turbo sigmas for steps `1,2,4,8` and image sequence lengths
  `64,256,4096`; max sigma/timestep deltas are below `1e-4`.
- Lens transformer Phase 5 scaffold: the local Lens source
  `F:\Paralol\local\Lens\lens\transformer.py` is the Python oracle;
  `inspect_lens_transformer_headers.py` verifies the Lens transformer tensor
  shapes from safetensors headers without downloading the 16.4 GB shards; and
  `dump_lens_transformer_oracle.py` is ready to dump activation references once
  local transformer weights and precomputed Lens features exist.
- Lens transformer math parity: `make_lens_block_fixture.py` creates a tiny
  deterministic PyTorch fixture covering both `LensTransformerBlock` and a
  two-block `LensTransformer2DModel`; `sd-lens-transformer-smoke` matches the
  native C++ forward against that oracle with `max_encoder=4.47035e-08`,
  `max_hidden=2.98023e-08`, and `max_output=2.38419e-07`.
- Lens real transformer shard metadata: `sd_lens_transformer_load` validates
  the transformer-only Lens-Turbo payload without resident weight allocation.
  The current downloaded payload has 2 shards, 1264 tensors, and
  16,416,900,608 bytes; the descriptor reports 48 layers, 24 heads, 64 head
  dimension, and 1536 inner dimension.
- Lens real-weight allocation guard: the future non-metadata transformer load
  path now carries explicit memory guard fields and still fails closed until a
  resident LensRunner exists. The real block fixture generator also refuses by
  default after a local Windows memory-management crash during PyTorch tensor
  materialization.
- Lens Phase 6 guarded scaffold: `sd-lens-phase6-smoke` validates CPU
  `lens_cond_v1` loading, metadata-only transformer inspection, Lens-Turbo
  schedule generation, Lens latent packing, handle release, and fail-closed
  non-metadata transformer loading in one process. The latest run reported
  1264 transformer tensors, 16,416,900,608 metadata bytes, 4 schedule steps,
  `external_transformer_required=true`, and `native_real_forward=false`.
- Lens real-weight block-0 transformer slice: `sd-lens-transformer-smoke`
  can load only `transformer_blocks.0.*` from the real Lens-Turbo shards under
  a byte cap and run a native CPU block forward on tiny synthetic tokens. The
  latest run loaded 26 tensors / 339,887,104 bytes and reported finite outputs
  with `native_real_block_forward=true full_model_forward=false`.
- Lens streamed real-weight tiny transformer smoke: the same smoke can keep
  top-level Lens transformer weights resident, stream all 48 real transformer
  blocks one at a time under a 512 MiB per-block cap, and apply final projection
  on tiny synthetic tokens. The latest run streamed 1260 block tensors /
  16,405,878,272 block bytes and produced finite `1x1x128` output with
  `native_real_tiny_transformer=true text_encoder_native=false
  gpu_inference=false`.
- Lens precomputed-conditioning real transformer smoke: `sd-lens-transformer-smoke`
  can load a `lens_cond_v1` bundle, validate `feature_0..3` against the real
  2880-wide text feature path, convert the valid-token mask to additive
  attention form, stream all 48 real transformer blocks, and produce finite
  `1x1x128` output. The latest run reported `using_precomputed_cond=true`.
- Lens tiny denoise-artifact VAE decode: `sd-lens-transformer-smoke` can
  unpatch cond-fed transformer output from `[1,16,128]` to `1x32x8x8`, write
  it as f32 NPY, and `sd-lens-vae-smoke` can decode it through the Lens/FLUX.2
  VAE. The latest VAE smoke wrote `build\diagnostics\lens_tiny_denoise.png`
  as a `64x64x3` image.
- Lens tiny first FlowMatch step decode: `sd-lens-transformer-smoke` can apply
  the first native Lens schedule update to tiny packed latent tokens before
  unpatching. The latest run used `steps=4`, `sigma0=1`, `sigma1=0.760245`,
  `dt=-0.239755`, wrote `build\diagnostics\lens_tiny_flow_step_32x8x8.npy`,
  and `sd-lens-vae-smoke` decoded `build\diagnostics\lens_tiny_flow_step.png`.
- Lens tiny 4-step FlowMatch loop decode: `sd-lens-transformer-smoke` can keep
  the tiny packed sample as loop state, recompute the streamed real transformer
  output after each update, apply all four native schedule intervals, unpatch
  the final sample to `1x32x8x8`, and decode it through `sd-lens-vae-smoke`.
  The latest run wrote `build\diagnostics\lens_tiny_flow_loop4_32x8x8.npy`
  with `max_abs=0.275516` and `mean_abs=0.066624`, then decoded
  `build\diagnostics\lens_tiny_flow_loop4.png`.
- Lens public external-flow loop API: `sd_lens_run_external_flow_loop_f32`
  validates CPU `lens_cond_v1` and metadata-only Lens transformer handles,
  builds the native Lens Turbo schedule, applies caller-supplied CPU
  model-output tensors to packed Lens tokens, and returns a descriptor marking
  `used_external_model_output=true` and `native_transformer_forward=false`.
  `sd-lens-phase6-smoke` now verifies this path with deterministic tiny buffers
  and reports `native_flow_loop=true`.
- Lens real streamed outputs through public flow API: `sd-lens-transformer-smoke
  --verify-external-flow-api` collects all four tiny real transformer outputs,
  calls `sd_lens_run_external_flow_loop_f32`, verifies `max_diff=0` against the
  manual loop, writes `build\diagnostics\lens_tiny_flow_api_loop4_32x8x8.npy`,
  and decodes `build\diagnostics\lens_tiny_flow_api_loop4.png`.
- Lens public latent unpack API: `sd_lens_unpack_vae_latent_f32` converts packed
  BSC Lens tokens `[1,S,128]` to public Lens latents `1x32xHxW`.
  `sd-lens-phase6-smoke` covers pack/unpack roundtrip layout, and the real
  transformer smoke now writes `build\diagnostics\lens_tiny_unpack_api_loop4_32x8x8.npy`
  through the public unpack API before VAE decode.
- Lens source-backed first-image fixture replay: `export_lens_first_image_fixture.py`
  writes a `LENSBLK1` bundle containing seeded packed start tokens, the real
  Lens timestep embedding, real Lens RoPE tensors, and the additive attention
  mask derived from `lens_cond_v1`. `sd-lens-transformer-smoke
  --oracle-input-fixture` consumes that bundle instead of synthetic timestep
  and identity-RoPE placeholders, streams all 48 real blocks, writes
  `build\diagnostics\lens_source_backed_64_32x8x8.npy`, and `sd-lens-vae-smoke`
  decodes `build\diagnostics\lens_source_backed_64.png` as a nonblank
  `64x64x3` image.
- Lens external transformer trace replay: `export_lens_conditioning.py` writes
  real GPT-OSS prompt features as `lens_cond_v1`, `export_lens_external_flow_trace.py`
  writes real Python Lens transformer outputs for each Turbo step, and
  `sd-lens-transformer-smoke --external-flow-fixture` runs those outputs through
  the public sd.cpp external-flow API plus public unpack API. The latest
  official-scale run wrote
  `build\diagnostics\lens_external_trace_real_robot_1024_32x128x128.npy`,
  and `sd-lens-vae-smoke` decoded
  `build\diagnostics\lens_external_trace_real_robot_1024.png` as a coherent
  `1024x1024x3` image.
- Qwen-Image text-only `cfg=1` T2I strict sampler lane: resident Qwen
  conditioning handle -> backend flow sampler -> CUDA `1x16x64x64` latent, with
  no sampler bridge flags.

Pending or deliberately unclaimed:

- Direct Z sampler reference/edit image inputs.
- Qwen-Image and Anima VAE decode image-quality validation/fix for the
  Qwen-image VAE. The Qwen/X bridge is not accepted because it produced a
  blurry output in local testing.
- Qwen-Image CFG, reference/edit, and vision conditioning.
- Anima Comfy/reference-quality image acceptance for the Qwen-image VAE path.
  The current fork can decode diagnostics through the CPU compatibility path or
  non-strict GPU-handle bridge, but true device-resident VAE image output
  remains intentionally disabled for Anima/Qwen-image until the VAE output is
  verified.
- TAE previews for SD1, Z-Image/Z-Anime, and Anima.
- True all-GPU sampler internals. See
  `docs/paralol-true-gpu-sampler-plan.md`.
