# Paralol GPU Latent Flow

This note covers the current stable-diffusion.cpp latent path used by Paralol.

## Current sampler path

`sd_sample_latent` is still a host-side sampler API:

- `StableDiffusionGGML::sample(...)` returns `sd::Tensor<float>`.
- `denoise_cb_t` is `std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)>`.
- The k-diffusion sampler implementations in `denoiser.hpp` update `x`, `denoised`, and noise as `sd::Tensor<float>` values.
- Each diffusion model callback returns `sd::Tensor<float>` from `DiffusionModel::compute(...)`.
- `GGMLRunner::compute<T>(...)` materializes the final ggml graph tensor with `sd::make_sd_tensor_from_ggml<T>(...)`.

That means the current sampler does not have a final device tensor that can be wrapped directly as a true GPU-resident KSampler output. A narrow `sd_sample_latent_gpu` wrapper can upload the final CPU latent into an `sd_gpu_handle_t`, but strict GPU-resident mode rejects that bridge because it is not an all-device sampler path.

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

`sd_sample_latent_gpu(...)` is intentionally marked as a compatibility bridge today. It calls the existing host-side sampler and uploads the final latent to CUDA. It sets `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT | SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD` on the returned handle. `sd_get_gpu_capabilities(...)` reports `supports_sampler_gpu_latent_output=false` and `supports_sampler_gpu_latent_bridge_output=true` so callers do not confuse this with a true GPU-resident sampler. With `SDCPP_STRICT_GPU_RESIDENT=1`, it fails instead of hiding the CPU materialization.

CPU sampled latents can also be uploaded explicitly with `sd_cpu_latent_upload(...)`.
That path is intended as a compatibility bridge for existing CPU latent values.
The uploaded handle is an owned CUDA buffer with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`;
if the source latent came from `sd_sample_latent(...)`, it also carries
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.

`sd_sample_latent_gpu_with_init_gpu(...)` is the I2I bridge API for KSampler
init latents. It accepts an `SD_GPU_RESOURCE_LATENT` handle, validates the CUDA
descriptor, downloads that init latent through the existing CPU-compatible
sampler path, then returns the sampled output as a CUDA latent handle. This is
not true all-GPU sampling yet, so:

- `supports_sampler_gpu_init_latent_input=false`
- `supports_sampler_gpu_init_latent_bridge_input=true`
- `SDCPP_STRICT_GPU_RESIDENT=1` refuses the API
- logs say `init_bridge_download=true` and `output_bridge_upload=true`
- the sampled output handle is flagged with both `CPU_BRIDGE_DOWNLOAD` and
  `CPU_BRIDGE_UPLOAD` so its provenance is visible through
  `sd_gpu_handle_get_desc(...)`

The value of the API is that Paralol can pass a VAE Encode GPU latent handle
into KSampler without inventing its own DLL-side lifetime or download/upload
sequence. The fork owns the validation and keeps the public seam stable for a
future true GPU-native sampler implementation.

The API validates dtype, channel count, and spatial latent dimensions against
the resolved request before downloading the GPU init latent. A mismatched
resolution now fails with a clear error before any bridge download occurs.

The sampler API emits structured timing logs:

- `[Timing] sd_sample_latent latent_prepare_ms=... prompt_encode_ms=... denoise_ms=...`
- `[Timing] sd_sample_latent_gpu cpu_sample_ms=... bridge_upload_ms=...`
- `[Timing] sd_sample_latent_gpu_with_init_gpu init_bridge_download_ms=... sample_bridge_ms=...`

Prompt/CLIP encode is still folded into KSampler, but this gives Paralol a
named timing bucket to compare against ComfyUI `CLIPTextEncode` until a separate
public CLIP encode API exists.

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
- `fallback_reason` explains the isolated VAE decode context

This is an all-GPU handoff across the VAE Encode -> Latent Decode boundary. It
does use an isolated decode context because the same-context VAE reuse path is
currently unsafe, but it does not download or upload the latent through CPU
memory.

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
- If KSampler calls `sd_sample_latent_gpu`, inspect handle flags. `CPU_BRIDGE_UPLOAD` means the sampler itself still materialized on CPU.
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
  GPU VAE Encode latent and wants a GPU sampled latent result. This is a
  non-strict bridge today, but it centralizes init-latent validation and keeps
  the worker off the raw latent bytes path.

## Supported handoff paths

Supported now:

- Non-strict T2I bridge:
  `sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download`
- Non-strict I2I init-latent bridge:
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

- Strict sampler GPU output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu`
- Strict sampler GPU init-latent bridge:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu_with_init_gpu`
- Strict VAE Encode compatibility bridge output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_encode_image_normal_gpu` for bridge-only
  model families such as Anima

The real all-GPU KSampler project is a separate sampler backend refactor: the denoiser callback and sampler math need backend tensor/resource variants instead of `sd::Tensor<float>` host values. The concrete blocker map is in `docs/paralol-true-gpu-sampler-plan.md`.
