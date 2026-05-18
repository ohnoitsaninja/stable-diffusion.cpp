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
SDXL/SD1 T2I Euler backend path described below.

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

With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` on supported SD1/SDXL CUDA Euler
T2I requests, `sd_sample_latent_gpu(...)` uses the backend tensor sampler path.
The returned latent is an owned CUDA handle with `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`
only and no CPU bridge flags. `sd_get_gpu_capabilities(...)` reports
`supports_sampler_gpu_latent_output=true` only when that true path is available
for the current context.

CPU sampled latents can also be uploaded explicitly with `sd_cpu_latent_upload(...)`.
That path is intended as a compatibility bridge for existing CPU latent values.
The uploaded handle is an owned CUDA buffer with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`;
if the source latent came from `sd_sample_latent(...)`, it also carries
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.

`sd_sample_latent_gpu_with_init_gpu(...)` is the I2I KSampler API for GPU init
latents. With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` on supported SD1/SDXL
CUDA Euler requests, it accepts an `SD_GPU_RESOURCE_LATENT` handle, validates
the CUDA descriptor, copies it device-to-device into the sampler backend,
applies strength/noise on the backend, and returns a sampled CUDA latent handle
without CPU bridge flags. In that mode:

- `supports_sampler_gpu_init_latent_input=true`
- `supports_sampler_gpu_init_latent_bridge_input=true`
- `SDCPP_STRICT_GPU_RESIDENT=1` allows the supported true path
- logs say `init_bridge_download=false` and `output_bridge_upload=false`
- the sampled output handle is flagged `SAMPLER_OUTPUT`; if the init latent was
  produced by VAE Encode, the output also carries
  `REQUIRES_ISOLATED_VAE_DECODE`

For unsupported requests or when the experimental Euler path is disabled, the
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

Prompt/CLIP encode is still folded into KSampler, but this gives Paralol a
named timing bucket to compare against ComfyUI `CLIPTextEncode` until a separate
public CLIP encode API exists.

An experimental SDXL/SD1 T2I Euler backend sampler loop can be enabled with
`SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1`. It keeps per-step latent state,
UNet input/output, CFG blend, denoised reconstruction, and Euler updates as
backend tensors. The path now creates the initial Philox Gaussian noise on
CUDA and scales it on the backend, so the normal success case returns a CUDA
latent handle with only `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT` and no
`CPU_BRIDGE_UPLOAD`/`CPU_BRIDGE_DOWNLOAD` flags. With
`SDCPP_STRICT_GPU_RESIDENT=1`, strict mode allows this narrow path.

The lane remains experimental and should not be advertised as a global sampler
capability because it only supports SDXL/SD1, Euler, batch 1, no masks,
ControlNet, reference/edit/image-CFG, and non-flow denoisers. It supports both
T2I and I2I when I2I provides the init latent as a CUDA latent handle.

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
  SDXL/SD1 T2I Euler result produced with
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` should have only
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
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` and a supported SD1/SDXL Euler
  request, this is a strict true-GPU path. Otherwise it remains a non-strict
  compatibility bridge.

## Supported handoff paths

Supported now:

- Non-strict T2I bridge:
  `sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download`
- Experimental strict SDXL/SD1 T2I Euler:
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1 sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download_to_buffer`
  with sampler output flags `SAMPLER_OUTPUT` only
- Experimental strict SDXL/SD1 I2I Euler:
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1 sd_encode_image_normal_gpu -> sd_sample_latent_gpu_with_init_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download_to_buffer`
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
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1`, or with unsupported sampler/model
  settings
- Strict sampler GPU init-latent bridge:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu_with_init_gpu` without
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1`, or with unsupported sampler/model
  settings
- Strict VAE Encode compatibility bridge output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_encode_image_normal_gpu` for bridge-only
  model families such as Anima

The complete all-GPU KSampler project is still broader than this first lane.
The env-gated SDXL/SD1 Euler backend path now keeps T2I and GPU-init I2I sampler
state as backend tensors and uses implicit-GEMM direct diffusion conv by
default, with `SDCPP_DISABLE_GPU_EULER_DIFFUSION_IMPLICIT_GEMM_CONV=1` as an
escape hatch. Explicit batch CFG is available only for investigation with
`SDCPP_EXPERIMENTAL_GPU_EULER_BATCHED_CFG=1`; it is not the default because the
current batch-2 UNet graph is not consistently faster. Non-Euler samplers, flow
denoisers, ControlNet, reference/edit/image-CFG, and model-family-specific
paths still need backend tensor/resource variants instead of `sd::Tensor<float>`
host values. The concrete blocker map is in
`docs/paralol-true-gpu-sampler-plan.md`.
