# Paralol GPU Resident Values

This fork exposes a first same-process GPU handle API for Paralol's native
Stable Diffusion lane. The goal is to let VAE decode return a CUDA-resident RGB
tensor and make CPU image materialization explicit.

## Scope

Implemented now:

- Opaque `sd_gpu_handle_t` values owned by `sd_ctx_t`
- Refcounted retain/release/get-desc/debug-name APIs
- CUDA pointer borrow API for same-process native consumers
- `sd_decode_latent_normal_gpu`
- `sd_encode_image_normal_gpu`
- `sd_decode_gpu_latent_normal_gpu`
- `sd_gpu_image_download`
- `sd_gpu_tensor_download`
- `sd_get_gpu_capabilities`
- Existing `sd_decode_latent_normal` preserved as a wrapper:
  `sd_decode_latent_normal_gpu -> sd_gpu_image_download`

Current invariants are unchanged:

- no tiled VAE
- no TAESD
- no legacy IM2COL
- COMFY_NORMAL full-frame staged decode
- CUDA implicit-GEMM conv remains the default VAE conv backend
- `SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` still forces direct-conv fallback

## Handle Shape

`sd_decode_latent_normal_gpu` returns an `SD_GPU_RESOURCE_IMAGE` handle.
For SDXL 1024 decode today, the descriptor is:

- backend: `SD_BACKEND_CUDA`
- dtype: `SD_DTYPE_F32`
- layout: `SD_LAYOUT_WHCN_GGML`
- shape: `N=1, C=3, H=1024, W=1024`
- bytes: `12582912`
- flags:
  - `SD_GPU_RESOURCE_FLAG_VAE_DECODE_OUTPUT`
  - `SD_GPU_RESOURCE_FLAG_REQUIRES_VAE_OUTPUT_SCALE`

The stored tensor is the VAE graph output before the old CPU-side
`(x + 1) / 2` clamp. `sd_gpu_image_download` applies that output transform
before packing the CPU `sd_image_t`, so legacy CPU decode output remains
compatible.

## Ownership

Producer returns a handle with `refcount=1`.

Paralol should:

1. retain a handle when storing it beyond immediate execution
2. release it when the node output is evicted
3. keep the handle retained for as long as any borrowed CUDA pointer is used

Borrowed CUDA pointers are same-process only and become invalid after the
underlying handle is released.

## Current Limits

The first useful zero-copy path is VAE decode output:

`CPU latent -> GPU RGB tensor handle -> explicit download only when requested`

`sd_encode_image_normal_gpu` preserves the API shape by running the existing
normal VAE encode path and uploading the resulting latent into a GPU handle.
That handle is safe to pass to `sd_decode_gpu_latent_normal_gpu`, but it is
marked as a bridge with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD` and
`SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT`. It is not yet a fully GPU-resident
image encode path.

For VAE-encoded latent handles, `sd_decode_gpu_latent_normal_gpu` uses a cached
VAE decode-only context internally. This avoids the unsafe same-context decode
path found during CUDA memcheck while preserving the Paralol GPU-handle
contract. The VAE report marks the bridge with `host_copies=1`,
`device_copies=1`, and a fallback reason. `SDCPP_STRICT_GPU_RESIDENT=1` refuses
this path.

`sd_encode_gpu_image_normal_gpu` is still a compatibility bridge that downloads
the GPU image before encode. It is useful for API symmetry, but not a final
all-GPU image encode path.

GPU RGBA packing, DLPack, CUDA IPC, and graphics interop are intentionally
deferred. `sd_get_gpu_capabilities` reports those as unsupported until they are
real.

## Strict Mode

Set:

```powershell
$env:SDCPP_STRICT_GPU_RESIDENT='1'
```

For `sd_decode_latent_normal_gpu`, strict mode fails if COMFY_NORMAL reports a
host stage boundary or non-device-resident VAE stage output. Strict mode also
refuses bridge paths that knowingly materialize CPU values:

- `sd_sample_latent_gpu`
- `sd_encode_image_normal_gpu`
- `sd_decode_gpu_latent_normal_gpu` when the latent handle was produced by VAE
  Encode
- `sd_encode_gpu_image_normal_gpu`

Set:

```powershell
$env:SDCPP_TRACE_GPU_HANDLES='1'
```

to log handle creation, descriptor shape, explicit downloads, and releases.

## Smoke

GPU handle without download:

```powershell
sd-latent-smoke.exe --model <sdxl.safetensors> --image <image.png> --image-channels 3 --type-f16 --split-decode-context --gpu-decode-output --strict-gpu-resident --dump-gpu-handle-desc --skip-estimate
```

GPU handle plus explicit CPU materialization:

```powershell
sd-latent-smoke.exe --model <sdxl.safetensors> --image <image.png> --image-channels 3 --type-f16 --split-decode-context --gpu-decode-output --download-gpu-output --strict-gpu-resident --dump-gpu-handle-desc --skip-estimate --output out.png
```

Paralol handoff:

- Latent Decode can move to `sd_decode_latent_normal_gpu` when the worker value
  protocol can carry an opaque same-process GPU handle.
- VAE Encode can move to `sd_encode_image_normal_gpu` for model families that
  report `supports_vae_encode_gpu_latent_output=true`, with the understanding
  that the current implementation is bridge-uploaded rather than true
  resident.
- Preview/Save nodes should explicitly call download or a future GPU RGBA path
  only when CPU/display bytes are actually required.
