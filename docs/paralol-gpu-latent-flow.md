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
- `sd_gpu_latent_download(...)`
- `sd_cpu_latent_upload(...)`
- `sd_decode_gpu_latent_normal_gpu(...)`

`sd_decode_gpu_latent_normal_gpu(...)` consumes a CUDA latent handle directly. For SDXL/Flux/Flux2/Z-Image scalar latent transforms it keeps the diffusion-latent to VAE-latent conversion on the selected backend, then runs COMFY_NORMAL VAE decode and returns a GPU image handle.

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

`sd_sample_latent_gpu(...)` is intentionally marked as a compatibility bridge today. It calls the existing host-side sampler and uploads the final latent to CUDA. It sets `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT | SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD` on the returned handle. With `SDCPP_STRICT_GPU_RESIDENT=1`, it fails instead of hiding the CPU materialization.

CPU sampled latents can also be uploaded explicitly with `sd_cpu_latent_upload(...)`.
That path is intended as a compatibility bridge for existing CPU latent values.
The uploaded handle is an owned CUDA buffer with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`;
if the source latent came from `sd_sample_latent(...)`, it also carries
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.

## VAE-encoded latent handoff

VAE Encode GPU latent handoff is supported as a safe compatibility bridge:

- `supports_vae_encode_gpu_latent_output=true`
- `supports_vae_encode_gpu_latent_bridge_output=true`
- `sd_encode_image_normal_gpu(...)` returns an `SD_GPU_RESOURCE_LATENT` handle
  marked with `SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT |
  SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`.

This is not yet true all-GPU VAE Encode. The current AutoencoderKL encode path
still materializes the encoded latent in host `sd::Tensor<float>` form for
Gaussian latent sampling/scaling, then uploads that latent into an owned CUDA
handle. `SDCPP_STRICT_GPU_RESIDENT=1` refuses this API so callers cannot mistake
the bridge for a zero-copy GPU encode path.

The earlier same-context decode path for VAE-encoded handles was unsafe: it
could trip a CUDA illegal memory access inside the implicit-GEMM VAE decode
convolution. The fixed path keeps the public GPU handle contract while avoiding
that unsafe route:

1. `sd_decode_gpu_latent_normal_gpu(...)` detects
   `SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT`.
2. It downloads the encoded latent into host memory.
3. It decodes through a cached VAE decode-only context, which is the split
   context path already proven stable by the SDXL smokes.
4. It uploads the decoded RGB tensor into the caller's original `sd_ctx_t` and
   returns an `SD_GPU_RESOURCE_IMAGE` handle.

The VAE report marks this honestly:

- `device_resident_stages=false`
- `host_copies=1`
- `device_copies=1`
- `fallback_reason` explains the encoded-latent bridge

The resulting handle is safe for Paralol's node contract, but the bridge is
still a compatibility lane. A future true-resident implementation would need
GPU-side VAE encode latent sampling/scaling plus a same-context decode fix.
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
  report `supports_vae_encode_gpu_latent_output=true`. Treat
  `supports_vae_encode_gpu_latent_bridge_output=true` as a sign that the handle
  is safe but bridge-uploaded, not true zero-copy. Anima reports this bridge as
  supported, but the Wan/Qwen VAE path uses large IM2COL workspaces at 1024.

## Supported handoff paths

Supported now:

- Non-strict T2I bridge:
  `sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download`
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

Refused now:

- Strict sampler GPU output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu`
- Strict VAE Encode GPU output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_encode_image_normal_gpu`

The real all-GPU KSampler project is a separate sampler backend refactor: the denoiser callback and sampler math need backend tensor/resource variants instead of `sd::Tensor<float>` host values.
