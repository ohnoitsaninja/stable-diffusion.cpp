# Paralol GPU Latent Flow

This note covers the current stable-diffusion.cpp latent path used by Paralol.

## Current sampler path

`sd_sample_latent` is still a host-side sampler API:

- `StableDiffusionGGML::sample(...)` returns `sd::Tensor<float>`.
- `denoise_cb_t` is `std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)>`.
- The k-diffusion sampler implementations in `denoiser.hpp` update `x`, `denoised`, and noise as `sd::Tensor<float>` values.
- Each diffusion model callback returns `sd::Tensor<float>` from `DiffusionModel::compute(...)`.
- `GGMLRunner::compute<T>(...)` materializes the final ggml graph tensor with `sd::make_sd_tensor_from_ggml<T>(...)`.

That means the default sampler does not have a final device tensor that can be
wrapped directly as a true GPU-resident KSampler output. A narrow
`sd_sample_latent_gpu` wrapper can upload the final CPU latent into an
`sd_gpu_handle_t`, but strict GPU-resident mode rejects that bridge because it
is not an all-device sampler path. The exception today is the env-gated
SDXL/SD1 backend sampler path described below.

## Current materialization points

Sampler output materializes here:

- `src/stable-diffusion.cpp`: `sd_sample_latent(...)`
- `src/stable-diffusion.cpp`: `sd::Tensor<float> final_latent = sd_ctx->sd->sample(...)`
- `src/stable-diffusion.cpp`: `return make_sd_latent(std::move(final_latent))`

The pre-existing CPU VAE decode path uploaded this latent by converting the `sd_latent_t` back into `sd::Tensor<float>` and passing it into a VAE graph input.

## GPU latent handle support

The fork now supports `SD_GPU_RESOURCE_LATENT` handles and explicit latent download/upload helpers:

- `sd_sample_latent_gpu(...)`
- `sd_sample_latent_gpu_with_init_gpu(...)`
- `sd_gpu_latent_download(...)`
- `sd_cpu_latent_upload(...)`
- `sd_decode_gpu_latent_normal_gpu(...)`

`sd_decode_gpu_latent_normal_gpu(...)` consumes a CUDA latent handle directly.
For base SD1, SDXL, Flux, Flux2, and Z-Image scalar latent transforms it
keeps the diffusion-latent to VAE-latent conversion on the selected backend,
then runs COMFY_NORMAL VAE decode and returns a GPU image handle. SD1 support is
intentionally limited to the base model version until inpaint, pix2pix, tiny,
and other variants get their own smokes. SD2 is not a current Paralol target.

Anima is supported through a compatibility bridge, not true GPU-resident VAE
decode. The sampler/CPU-upload latent handle is downloaded into the existing
legacy Wan/Qwen VAE decode path, then the decoded RGB tensor is uploaded back
into an `SD_GPU_RESOURCE_IMAGE` handle. This is exposed so Paralol can use the
same latent/image handle contract for Anima while the fork still reports the
bridge honestly:

- `supports_gpu_latent_decode=true` for Anima
- `host_copies=1`, `device_copies=1` in the VAE report
- output image handle flags include `CPU_BRIDGE_DOWNLOAD | CPU_BRIDGE_UPLOAD`
- `SDCPP_STRICT_GPU_RESIDENT=1` refuses this path

This bridge does not change Anima image semantics; it uses the same
`decode_first_stage(...)` path as direct `sd-cli` generation. Anima public
normal VAE encode/decode keeps the Wan/Qwen VAE on the legacy convolution path.
The Anima public VAE encode/decode path reports large IM2COL workspaces instead
of failing the SDXL COMFY_NORMAL guard because that legacy graph is currently
the known-good Anima path.

Separated `sd_encode_image_normal(...)` and `sd_encode_image_normal_gpu(...)`
are enabled for Anima as compatibility bridges. The modular Paralol path
`VAE Encode -> KSampler -> Latent Decode` can therefore pass a 16-channel Anima
latent through the same `SD_GPU_RESOURCE_LATENT` contract. The implementation is
not true GPU-resident encode: it materializes the encoded latent as a CPU
`sd::Tensor<float>`, uploads an owned CUDA latent handle, and marks the report
with `host_copies=1`, `device_copies=1`, and
`fallback_reason="VAE Encode GPU output is bridge-uploaded after CPU latent conversion"`.
At 1024x1024 this Wan/Qwen VAE path currently plans about 7702 MB for encode and
7493 MB for decode, with IM2COL present. That is expected for Anima today and is
reported rather than hidden.

`sd_sample_latent_gpu(...)` has two modes. By default, or for unsupported
sampler/model requests, it is a compatibility bridge: it calls the existing
host-side sampler and uploads the final latent to CUDA. That bridge sets
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT | SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`
and strict GPU-resident mode refuses it.

With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` on supported SD1/SDXL CUDA
requests, `sd_sample_latent_gpu(...)` uses the backend tensor sampler path.
The returned latent is an owned CUDA handle with
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT` only and no CPU bridge flags.
`sd_get_gpu_capabilities(...)` reports `supports_sampler_gpu_latent_output=true`
only when that true path is available for the current context.

`SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` remains accepted as a compatibility
alias for older local workflows, but new tests and Paralol integration should
prefer `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`.

CPU sampled latents can also be uploaded explicitly with `sd_cpu_latent_upload(...)`.
That path is intended as a compatibility bridge for existing CPU latent values.
The uploaded handle is an owned CUDA buffer with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`;
if the source latent came from `sd_sample_latent(...)`, it also carries
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.

`sd_sample_latent_gpu_with_init_gpu(...)` is the I2I KSampler API for GPU init
latents. With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` on supported SD1/SDXL
CUDA requests, it accepts an `SD_GPU_RESOURCE_LATENT` handle, validates the
CUDA descriptor, copies it device-to-device into the sampler backend, applies
strength/noise on the backend, and returns a sampled CUDA latent handle without
CPU bridge flags. In that mode:

- `supports_sampler_gpu_init_latent_input=true`
- `supports_sampler_gpu_init_latent_bridge_input=true`
- `SDCPP_STRICT_GPU_RESIDENT=1` allows the supported true path
- logs say `init_bridge_download=false` and `output_bridge_upload=false`
- the sampled output handle is flagged `SAMPLER_OUTPUT`; if the init latent was
  produced by VAE Encode, the output also carries
  `REQUIRES_ISOLATED_VAE_DECODE`

For unsupported requests or when the experimental backend path is disabled, the
same API remains a non-strict compatibility bridge: it validates the CUDA init
latent, downloads that init latent through the CPU-compatible sampler path, and
uploads the sampled output as a CUDA latent handle. That bridge sets the CPU
bridge flags and strict mode refuses it.

The API validates dtype, channel count, and spatial latent dimensions against
the resolved request before downloading the GPU init latent. A mismatched
resolution now fails with a clear error before any bridge download occurs.

The sampler API emits structured timing logs:

- `[Timing] sd_sample_latent latent_prepare_ms=... prompt_encode_ms=... denoise_ms=...`
- `[Timing] sd_sample_latent_gpu cpu_sample_ms=... bridge_upload_ms=...`
- `[Timing] sd_sample_latent_gpu_with_init_gpu init_bridge_download_ms=... sample_bridge_ms=...`
- `[Timing] sd_sample_latent_gpu_with_init_gpu latent_prepare_ms=... init_backend_ms=... denoise_ms=... sampler_math_residency=gpu_backend_tensor init_bridge_download=false output_bridge_upload=false`
- `[Timing] sd_sample_latent_gpu latent_prepare_ms=... init_backend_ms=... denoise_ms=... sampler_math_residency=gpu_backend_tensor init_bridge_download=false output_bridge_upload=false`

Prompt/CLIP encode can now be moved out of KSampler with
`sd_conditioning_encode_text(...)`. For the supported CUDA SD1/SDXL backend
sampler lane, conditioning handles keep the host `SDCondition` for compatibility
and also upload the cross-attention tensor plus SDXL pooled/vector tensor into
owned backend resources once at encode time. The descriptor reports
`device_resident=true`, `backend=SD_BACKEND_CUDA`, and flags
`DEVICE_RESIDENT | UPLOADED_BACKEND_TENSOR` only when those backend resources
exist. KSampler handle APIs then pass those backend tensor references directly
into UNet graph inputs, so normal denoise steps do not call `make_input(...)`
and re-upload conditioning every step.

The conditioning path remains honest about fallback:

- `conditioning_storage=device_tensor`: handle tensors are backend-resident and
  the sampler consumes them by reference.
- `conditioning_storage=host_tensor`: handle tensors are CPU-backed and the
  sampler may upload them through the compatibility input path.
- `conditioning_per_step_upload=false`: no per-step conditioning upload is
  expected.
- `conditioning_per_step_upload=true`: host-backed fallback is in use.

`SDCPP_DISABLE_CONDITIONING_GPU_RESIDENT=1` forces the host-backed fallback for
diagnostics. `SDCPP_STRICT_GPU_RESIDENT=1` refuses that fallback and only allows
conditioning handles whose descriptor is device-resident. This means Paralol can
use strict mode as a real check that prompt encode has been separated from
KSampler and conditioning tensors are not silently re-uploaded during denoise.

The encode API logs:

- `[Timing] sd_conditioning_encode_text condition_encode_ms=... conditioning_backend_upload_ms=... storage=device_tensor|host_tensor device_resident=true|false`

The sampler handle APIs log:

- `condition_input=handle conditioning_storage=device_tensor|host_tensor conditioning_per_step_upload=false|true`

An experimental SDXL/SD1 backend sampler loop can be enabled with
`SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`. It keeps per-step latent state,
UNet input/output, CFG blend, denoised reconstruction, and sampler updates as
backend tensors. The path creates the initial Philox Gaussian noise on CUDA and
scales it on the backend, so the normal success case returns a CUDA latent
handle with only `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT` and no
`CPU_BRIDGE_UPLOAD`/`CPU_BRIDGE_DOWNLOAD` flags. With
`SDCPP_STRICT_GPU_RESIDENT=1`, strict mode allows this narrow path.

The lane remains experimental and should not be advertised as a global sampler
capability because it only supports SDXL/SD1, batch 1, no masks, ControlNet,
reference/edit/image-CFG, and non-flow denoisers. It supports both T2I and I2I
when I2I provides the init latent as a CUDA latent handle. Euler has the most
complete parity/performance validation; the other public SDXL/SD1 non-flow
samplers have strict GPU-resident smoke coverage and need broader workflow
promotion before they should be considered equally production-ready.

There is also a stricter proof API,
`sd_sample_latent_gpu_true_euler_spike(...)`, gated by
`SDCPP_EXPERIMENTAL_TRUE_GPU_SAMPLER=1`. It returns a latent handle without CPU
bridge flags and strict mode allows it, but it uses a deterministic procedural
device noise source rather than production Philox. Its purpose is to prove that
owned backend tensors can carry sampler state through UNet and Euler math.

When the init handle came from true GPU VAE Encode, the sampled output is tagged
with `SD_GPU_RESOURCE_FLAG_REQUIRES_ISOLATED_VAE_DECODE`. That keeps the later
Latent Decode on the safe isolated COMFY_NORMAL VAE context path, avoiding the
known same-context CUDA fault after VAE encode while still using device-to-device
handoff for the decode side.

## VAE-encoded latent handoff

For COMFY_NORMAL model families that use the AutoencoderKL path, VAE Encode GPU
latent handoff is now true GPU-resident:

- `supports_vae_encode_gpu_latent_output=true`
- `sd_encode_image_normal_gpu(...)` returns an `SD_GPU_RESOURCE_LATENT` handle
  marked with `SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT`.
- The handle is not marked `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`.
- The encode report must show `host_copies=0` and `device_resident=true`.
- `SDCPP_STRICT_GPU_RESIDENT=1` allows this path.

The VAE encoder output and latent sampling/scaling remain on CUDA. The only CPU
input is the source image bytes passed into the public CPU-image API. The
resulting diffusion latent is an owned backend buffer, not a transient graph
tensor.

The true GPU encode path must apply the same VAE input transform as the CPU
path: source RGB bytes are converted to planar float `[0, 1]`, then the encoder
graph receives `[-1, 1]` values. If that transform is skipped, encoded-latent
roundtrips become washed out even though BF16 decode itself is correct.

The earlier same-context decode path for VAE-encoded handles was unsafe: it
could trip a CUDA illegal memory access inside the implicit-GEMM VAE decode
convolution after a VAE encode on the same VAE object. The fixed path keeps the
latent resident while avoiding that unsafe route:

1. `sd_decode_gpu_latent_normal_gpu(...)` detects
   `SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT`.
2. It copies the CUDA latent device-to-device into a cached VAE decode-only
   context.
3. It decodes there using COMFY_NORMAL.
4. It copies the decoded RGB tensor device-to-device back into the caller's
   context and returns an `SD_GPU_RESOURCE_IMAGE` handle.

The VAE report marks this honestly:

- `device_resident_stages=true`
- `host_copies=0`
- `device_copies` includes the two extra D2D context handoff copies
- `decode_context_ms`, `decode_setup_ms`, `decode_latent_d2d_ms`,
  `decode_graph_ms`, `decode_image_d2d_ms`, and `decode_context_reuse` split
  pure graph time from isolated-context overhead
- `fallback_reason` explains the isolated VAE decode context

This is an all-GPU handoff across the VAE Encode -> Latent Decode boundary. It
does use an isolated decode context because the same-context VAE reuse path is
currently unsafe, but it does not download or upload the latent through CPU
memory.

The isolated decode context is cached on the owning `sd_ctx_t`. A cold
encoded-latent decode therefore includes one decode-only context creation/load
cost, while later decodes on the same resident context report
`decode_context_reuse=1` and should have near-zero `decode_context_ms`.
Paralol can explicitly prewarm this cache with:

- `sd_prewarm_vae_decode_bridge(sd_ctx_t*, const sd_vae_run_options_t*, sd_vae_memory_report_t*)`

That API does not decode pixels; it creates and configures the isolated
decode-only VAE context so the later Latent Decode report separates the real
VAE graph time from the prewarm cost. The same-context probe remains opt-in
through `SDCPP_EXPERIMENTAL_VAE_SAME_CONTEXT_DECODE=1`; it is not a production
path because the known same-context encoded-latent decode fault still
reproduces.

Paralol should surface the report fields directly:

- `vaeDecodeSetupMs` -> `decode_setup_ms`
- `vaeDecodeContextMs` -> `decode_context_ms`
- `vaeDecodeD2dMs` -> `decode_latent_d2d_ms`
- `vaeDecodeGraphMs` -> `decode_graph_ms`
- `vaeDecodeImageD2dMs` -> `decode_image_d2d_ms`
- `vaeDecodeDownloadMs` -> `decode_download_ms`
- `vaeDecodeContextReuse` -> `decode_context_reuse != 0`

For encoded-latent I2I, the first cold decode is expected to show non-zero
`decode_context_ms`. A prewarmed or already resident decode context should make
`decode_context_ms` near zero while leaving `decode_graph_ms` around the actual
BF16 COMFY_NORMAL graph time.

Anima uses the Wan/Qwen bridge instead of the SDXL cached decode-only bridge;
the decode report should say
`fallback_reason="Anima uses the Wan/Qwen VAE bridge: GPU latent is downloaded for legacy decode and decoded image is re-uploaded"`.

## SDXL 1024 latent descriptor

For a 1024x1024 SDXL image:

- Shape: `N=1, C=4, H=128, W=128`
- GGML descriptor order: `w=128, h=128, c=4, n=1`
- Layout: `SD_LAYOUT_WHCN_GGML`
- Dtype: `SD_DTYPE_F32`
- Byte size: `128 * 128 * 4 * 1 * sizeof(float) = 262144`
- Backend for GPU handles: `SD_BACKEND_CUDA`

## Ownership and lifetime

Exported GPU handles are opaque ids owned by the `sd_ctx_t` that created them.
Consumers must not persist borrowed CUDA pointers after releasing the handle.

Current exported latent/image handles use owned backend buffers, not transient
ggml graph tensors:

- Producer returns a handle with `refcount=1`.
- `sd_gpu_handle_retain(...)` increments ownership for stored graph values.
- `sd_gpu_handle_release(...)` releases ownership and destroys the resource at
  the final release.
- `sd_gpu_handle_get_desc(...)` reports kind, backend, dtype, layout, shape,
  strides, byte size, flags, and refcount.
- `sd_gpu_handle_borrow_cuda_ptr(...)` is same-process only and does not
  transfer ownership.
- `sd_gpu_image_download(...)` writes an `sd_image_t` view whose `data` pointer
  is allocated by the stable-diffusion.cpp DLL. Callers that copy those pixels
  into their own buffers must release `data` with
  `sd_free_downloaded_image(output.data)`.
- `free_sd_image(...)` remains for APIs that return an owned `sd_image_t*`
  wrapper struct, such as `sd_decode_latent_normal(...)`.
- `sd_gpu_image_download_to_buffer(...)` is the preferred Paralol path when the
  caller already owns an RGBA8 destination. It validates the GPU image handle,
  destination size, and stride, then copies directly into caller-owned RGBA8
  memory without returning DLL-owned CPU image memory.

Do not pass raw CUDA pointers across process boundaries. Future cross-process
or graphics interop should use a separate IPC/external-memory API.

## Paralol handoff

For the next Paralol worker integration:

- Treat latent values as `CPU latent OR GPU latent handle`.
- Treat image values as `CPU image OR GPU image handle`.
- If KSampler calls `sd_sample_latent_gpu`, inspect handle flags.
  `CPU_BRIDGE_UPLOAD` means the sampler itself still materialized on CPU. A
  supported SDXL/SD1 result produced with
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` should have only
  `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.
- Latent Decode should prefer `sd_decode_gpu_latent_normal_gpu` when it receives a CUDA latent handle.
- Save/export/debug consumers should explicitly call `sd_gpu_latent_download`,
  `sd_gpu_image_download`, or preferably `sd_gpu_image_download_to_buffer` when
  they already own the output buffer.
- VAE Encode can request `sd_encode_image_normal_gpu` for model families that
  report `supports_vae_encode_gpu_latent_output=true`. If the returned handle
  lacks `CPU_BRIDGE_UPLOAD`, it is a true CUDA latent handoff and can be used in
  strict mode. Treat `supports_vae_encode_gpu_latent_bridge_output=true` as a
  compatibility signal only for model families such as Anima, where the handle
  is safe but bridge-uploaded, not true resident.
- I2I KSampler can call `sd_sample_latent_gpu_with_init_gpu` when it receives a
  GPU VAE Encode latent and wants a GPU sampled latent result. With
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` and a supported SD1/SDXL sampler
  request, this is a strict true-GPU path. Otherwise it remains a non-strict
  compatibility bridge. `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` remains a
  compatibility alias for older Euler-only test workflows.

## Supported handoff paths

Supported now:

- Non-strict T2I bridge:
  `sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download`
- Experimental strict SDXL/SD1 backend sampler T2I:
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1 sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download_to_buffer`
  with sampler output flags `SAMPLER_OUTPUT` only
- Experimental strict SDXL/SD1 backend sampler I2I:
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1 sd_encode_image_normal_gpu -> sd_sample_latent_gpu_with_init_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download_to_buffer`
  with no CPU bridge flags on the sampled latent
- Non-strict I2I init-latent bridge for unsupported requests:
  `sd_encode_image_normal_gpu -> sd_sample_latent_gpu_with_init_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download_to_buffer`
- CPU sampled latent bridge:
  `sd_sample_latent -> sd_cpu_latent_upload -> sd_decode_gpu_latent_normal_gpu`
- Anima sampled-latent bridge:
  `sd_sample_latent_gpu -> legacy Wan/Qwen VAE decode bridge -> SD_GPU_RESOURCE_IMAGE`
- Anima VAE Encode bridge:
  `sd_encode_image_normal_gpu -> legacy Wan/Qwen VAE encode bridge -> SD_GPU_RESOURCE_LATENT -> legacy Wan/Qwen VAE decode bridge`
- GPU image output:
  `sd_decode_gpu_latent_normal_gpu -> SD_GPU_RESOURCE_IMAGE -> sd_gpu_image_download`
- Caller-owned image output:
  `sd_decode_gpu_latent_normal_gpu -> SD_GPU_RESOURCE_IMAGE -> sd_gpu_image_download_to_buffer`
- VAE Encode bridge:
  `sd_encode_image_normal_gpu -> SD_GPU_RESOURCE_LATENT -> sd_decode_gpu_latent_normal_gpu -> SD_GPU_RESOURCE_IMAGE`
  for Anima/Wan-Qwen compatibility only
- VAE Encode true GPU handoff:
  `sd_encode_image_normal_gpu -> CUDA latent handle -> D2D isolated COMFY_NORMAL decode -> CUDA image handle`

Model-family status is summarized in
`docs/paralol-model-capability-matrix.md`. Use that matrix and
`sd_get_model_pipeline_capabilities()` rather than inferring support from file
names or latent channel count alone.

Refused now:

- Strict default sampler GPU output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu` without
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`, or with unsupported sampler/model
  settings
- Strict sampler GPU init-latent bridge:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu_with_init_gpu` without
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`, or with unsupported sampler/model
  settings
- Strict VAE Encode compatibility bridge output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_encode_image_normal_gpu` for bridge-only
  model families such as Anima

The complete all-GPU KSampler project is still broader than this first lane.
The env-gated SDXL/SD1 backend sampler path now keeps T2I and GPU-init I2I
sampler state as backend tensors and uses implicit-GEMM direct diffusion conv by
default, with `SDCPP_DISABLE_GPU_EULER_DIFFUSION_IMPLICIT_GEMM_CONV=1` as the
current escape hatch for that diffusion-conv selection. Eligible SDXL/SD1 CFG
now defaults to a dual-branch UNet graph with compute-buffer reuse. It keeps
cond and uncond branches in one graph and returns a CFG-guided model output that
the sampler-specific update math consumes. Euler additionally uses a fused
variant that folds `c_in`, CFG blend, denoised reconstruction, and Euler update
into one graph because that update formula is Euler-specific.

Useful diagnostics and escape hatches:

- `cfg_eval_mode=dual_branch_reuse`: default eligible SDXL/SD1 CFG path
- `SDCPP_DISABLE_GPU_SAMPLER_DUAL_BRANCH_CFG=1`: use separate cond/uncond graph
  calls for the backend sampler path
- `SDCPP_DISABLE_GPU_SAMPLER_UNET_REUSE=1`: keep dual-branch CFG but disable
  UNet compute-buffer reuse
- `SDCPP_DISABLE_GPU_EULER_FUSED_UPDATE=1`: keep generic dual-branch CFG for
  Euler but use the normal sampler update graph instead of the Euler fused graph
- `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BATCHED_CFG=1`: force the older batch-2 CFG
  experiment when compatible

Strict backend sampler smokes now cover the SDXL/SD1 public non-flow sampler set
with `cfg_eval_mode=dual_branch_reuse`: Euler, Euler A, Heun, DPM2,
DPM++ 2S A, DPM++ 2M, modified DPM++ 2M, iPNDM, iPNDM_v, LCM, DDIM trailing,
TCD, Res Multistep, Res 2S, ER-SDE, DPM++ SDE, DPM++ SDE GPU,
DPM++ 2M SDE, DPM++ 2M SDE GPU, DPM++ 2M SDE Heun,
DPM++ 2M SDE Heun GPU, DPM++ 3M SDE, and DPM++ 3M SDE GPU. Euler remains the
deepest parity lane because it has the imported-noise tensor comparison work;
the other sampler families now share the same device-resident CFG evaluation
primitive. Flow denoisers, ControlNet, reference/edit/image-CFG, masks, and
model-family-specific paths still need backend tensor/resource variants instead
of `sd::Tensor<float>` host values. The concrete blocker map is in
`docs/paralol-true-gpu-sampler-plan.md`.
