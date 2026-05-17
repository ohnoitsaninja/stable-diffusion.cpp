# Experimental BF16 VAE ggml Patchset

This fork keeps the experimental BF16 COMFY_NORMAL VAE work in the
stable-diffusion.cpp parent repo as a reproducible patch series. The parent
commit records the `ggml` submodule at a public upstream commit, then applies
the local CUDA BF16 VAE changes with `scripts/apply_ggml_bf16_vae_patches.ps1`
when the BF16 VAE build path is requested.

No separate public `ggml` fork or unreachable nested `ggml` commit is required.

## Upstream Base

- `ggml` upstream base: `404fcb9d7c96989569e68c9e7881ee3465a05c50`
- Parent source commit before patchset packaging: `c15a29e7d4784f256336a018da0d15071255ee6a`
- Integration commit: this commit

The patch application script resets the submodule to the upstream base before
applying the series.

## Patch List

Patch directory: `patches/ggml/bf16-vae/`

1. `0001-Add-CUDA-implicit-GEMM-conv2d-backend.patch`
2. `0002-Add-CUDA-Philox-randn-helper.patch`
3. `0003-Allow-implicit-GEMM-conv-for-diffusion-graphs.patch`
4. `0004-Honor-fused-conv-scale-in-CUDA-implicit-conv.patch`
5. `0005-Normalize-implicit-conv-sources-to-UTF-8.patch`
6. `0006-Add-experimental-BF16-CUDA-VAE-op-support.patch`

Applying the series produces a local `ggml` HEAD equivalent to the previously
validated experimental tree, but the parent repo does not point at that local
commit.

## Rebuild

From a fresh clone of this parent fork:

```powershell
.\scripts\apply_ggml_bf16_vae_patches.ps1 -Force
```

Then build from a VS/CUDA-capable developer shell:

```powershell
cmake -S . -B build\codex -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON
cmake --build build\codex --config Release --target stable-diffusion sd-latent-smoke sd-vae-op-bench
```

Or use the wrapper:

```powershell
.\scripts\build_bf16_vae.ps1 `
  -Model F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors `
  -Image C:\tmp\stable-diffusion.cpp-paralol\build\bf16-patchset-validation\orc-rgba.png `
  -ComfyRoot F:\automatic1111\Stability\Packages\ComfyUI
```

The wrapper applies the patchset before building. It only runs the BF16 smoke
when `-Model` and `-Image` are provided.

## Enable And Disable

BF16 VAE is opt-in only:

```powershell
$env:SDCPP_EXPERIMENTAL_VAE_BF16 = "1"
```

Unset that variable to return to the default f32 COMFY_NORMAL path:

```powershell
Remove-Item Env:SDCPP_EXPERIMENTAL_VAE_BF16 -ErrorAction SilentlyContinue
```

`SDCPP_VAE_DTYPE=bf16` alone does not enable the experimental BF16 path.
`sd_get_vae_capabilities().supports_bf16_storage` remains `false` while this is
experimental so Paralol cannot accidentally auto-enable it from capability
discovery.

## Validation

Build artifact:

- DLL: `C:\tmp\stable-diffusion.cpp-paralol\build\codex\bin\stable-diffusion.dll`
- SHA256: `224ACAB6C90B01DC46F778EC7690DAEDF762584D20E6B5D877FDBEF12C5C1DE4`

Default COMFY_NORMAL f32 regression smoke:

- `SDCPP_EXPERIMENTAL_VAE_BF16` unset
- SDXL 1024 roundtrip smoke passed
- Encode workspace: `1536 MB`
- Decode workspace: `2816 MB`
- `used_im2col=false`
- `used_tiling=false`
- `used_taesd=false`
- `host_copies=0`
- Default dtype: `f32`

BF16 SDXL smoke:

- `SDCPP_EXPERIMENTAL_VAE_BF16=1`
- `SDCPP_VAE_STRICT_COMFY_NORMAL=1`
- SDXL 1024 no-init T2I decode smoke passed
- Encode estimate workspace: `1104 MB`
- Decode estimate workspace: `1408 MB`
- Decode graph time: about `0.28s`
- `used_im2col=false`
- `used_tiling=false`
- `used_taesd=false`
- `host_copies=0`
- `direct_conv=true`
- `compact=true`
- math policy: `storage=bf16 math=f32 reductions=f32`

`sd-vae-op-bench`:

- `sd-vae-op-bench --correctness --warmup 2 --iterations 5` passed
- Implicit conv correctness diffs were zero for tested SDXL VAE conv shapes
- `1024x1024x256 -> 256 k3`: direct `1327.597 ms`, implicit `60.028 ms`
- `1024x1024x128 -> 128 k3`: direct `334.001 ms`, implicit `15.137 ms`
- BF16 compact group norm/upscale/pad/pointwise tests reported no NaN/Inf

Comfy parity:

- Comfy root: `F:\automatic1111\Stability\Packages\ComfyUI`
- Comfy torch: `2.12.0+cu130`
- CUDA device: `NVIDIA GeForce RTX 4080 SUPER`
- Comfy VAE dtype: `torch.bfloat16`
- Comfy peak allocated: `2371.94 MiB`
- Comfy peak reserved: `3328 MiB`
- sdcpp BF16 encode workspace: `1104 MB`
- sdcpp BF16 decode workspace: `1408 MB`
- Mean abs diff: `0.002917021745815873`
- p95 abs diff: `0.00784313678741455`
- p99 abs diff: `0.019607841968536377`
- p999 abs diff: `0.047058820724487305`
- Max abs diff: `0.25882354378700256`
- PSNR: `45.28346360508625 dB`

Regression checker:

- `ok=true`
- thresholds used:
  - decode workspace <= `1600 MB`
  - encode workspace <= `1200 MB`
  - mean abs diff <= `0.01`
  - p99 abs diff <= `0.05`
  - PSNR >= `40 dB`

## Notes

The default path remains the existing f32 COMFY_NORMAL behavior. The BF16 path is
kept behind a single explicit environment flag until it is promoted out of
experimental status.

The VAE C API now trims RGBA CPU image input to RGB before VAE encode. This
preserves the VAE contract for callers that pass an RGBA `sd_image_t` while
leaving non-VAE image/control paths unchanged.
