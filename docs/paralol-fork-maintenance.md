# Paralol Fork Maintenance

This document is the working maintenance playbook for the Paralol
`stable-diffusion.cpp` fork. It is intentionally practical: keep the integration
branch useful for Paralol, keep local releases reproducible, and make future
upstream updates less risky.

The current integration branch is:

```text
paralol/comfy-normal-vae-cudnn
```

The upstream parent is:

```text
leejet/stable-diffusion.cpp
```

The fork remote used for publishing is:

```text
ohnoitsaninja/stable-diffusion.cpp
```

## Branch Policy

Keep `paralol/comfy-normal-vae-cudnn` as the integration branch that Paralol
can consume. Do not use it as the place to experiment broadly.

Use short-lived branches for new work:

- `paralol/<feature>` for fork runtime work that may land in the integration
  branch.
- `upstream-review/<feature>` for cleanup branches intended to become small
  upstream pull requests.
- `scratch/<feature>` for local experiments that should not be pushed as
  product history.

Before merging into the integration branch, the change should have:

- a concrete Paralol use case
- clear capability reporting or a clear refusal path
- bounded smoke coverage
- updated docs when the public API or workflow contract changes
- no hidden change to existing SDXL behavior unless that is the point of the
  change

## Upstream Sync Workflow

Do not rebase the integration branch casually. It carries a broad stack of API,
backend, VAE, ControlNet, Flux/Z, and Anima changes. Prefer a deliberate sync.

Recommended flow:

```powershell
git fetch origin master
git fetch ohnoitsaninja paralol/comfy-normal-vae-cudnn
git checkout -b paralol/upstream-sync-YYYYMMDD ohnoitsaninja/paralol/comfy-normal-vae-cudnn
git merge origin/master
```

Resolve conflicts feature by feature. The conflict-prone areas are usually:

- `include/stable-diffusion.h`
- `src/stable-diffusion.cpp`
- `src/denoiser.hpp`
- `src/auto_encoder_kl.hpp`
- `src/control.hpp`
- `src/ggml_extend.hpp`
- `examples/common/common.cpp`
- `examples/cli/main.cpp`
- `examples/server/routes_sdapi.cpp`
- the nested `ggml` submodule pointer

After the merge, run the minimal validation set before promoting it back to the
integration branch. If the upstream merge changes model loading, VAE, sampler,
ControlNet, or ggml CUDA behavior, run the extended validation set.

## Minimal Validation Set

This set is for documentation-only changes, packaging changes, or small API
cleanup that should not affect runtime behavior.

```powershell
git diff --check
cmake --build build\codex --config Release --target stable-diffusion sd-cli sd-latent-smoke --parallel 6
```

Then confirm the DLL exports that Paralol depends on:

- `sd_sample_latent`
- `sd_sample_latent_gpu`
- `sd_encode_image`
- `sd_decode_latent`
- `sd_decode_latent_normal`
- `sd_decode_latent_normal_gpu`
- `sd_decode_gpu_latent_normal_gpu`
- `sd_gpu_latent_download`
- `sd_cpu_latent_upload`
- `sd_gpu_image_download`
- `sd_gpu_image_download_to_buffer`
- `sd_free_downloaded_image`
- `sd_encode_image_normal_gpu`
- `sd_get_gpu_capabilities`
- `sd_get_model_pipeline_capabilities`

## Runtime Validation Set

Run these when touching VAE, samplers, GPU handles, model capabilities,
ControlNet, Flux/Z, Anima, or model loading.

### SDXL COMFY_NORMAL VAE

Validate:

- no tiled VAE
- no TAESD
- no IM2COL in the COMFY_NORMAL SDXL path
- `host_copies=0`
- decode workspace around `2816 MB`
- encode workspace around `1536 MB`
- implicit-GEMM enabled by default
- `SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV=1` still falls back to direct conv

Use the existing `sd-latent-smoke` paths and the Comfy parity harness when the
change could affect image semantics.

### GPU Handle Contract

Validate:

- `sd_sample_latent_gpu` works in non-strict mode and reports the bridge-upload
  honestly
- `SDCPP_STRICT_GPU_RESIDENT=1` refuses fake GPU-resident sampler output
- `sd_decode_gpu_latent_normal_gpu` consumes supported CUDA latent handles
- GPU image output can be downloaded with `sd_gpu_image_download_to_buffer`
- DLL-owned downloads are releasable with `sd_free_downloaded_image`
- VAE-encoded GPU latent handoff works in non-strict mode through the safe
  decode-only bridge and reports the bridge honestly
- `SDCPP_STRICT_GPU_RESIDENT=1` refuses VAE Encode GPU output because the encode
  path still materializes and uploads a CPU latent

### Flux2 and Z-Image

Validate capability reporting instead of relying on filenames:

- Flux2 Klein reports `latent_channels=128`, edit/reference support, and Qwen
  LLM requirements
- Z-Image reports `latent_channels=16` and Qwen LLM requirements
- GPU sampled latent -> GPU VAE decode -> explicit image download produces a
  coherent output
- Flux2 edit/reference conditioning still uses the reference image path

### Anima

Anima is currently supported through a compatibility bridge, not true
GPU-resident VAE decode.

Validate:

- model capabilities report `family=anima`
- `latent_channels=16`
- default sampler is `er_sde`
- default scheduler is `discrete`
- `supports_gpu_latent_decode=true`
- `supports_vae_encode=true`
- `supports_vae_encode_gpu_output=true`
- sampled latent descriptor is `1x16x128x128` for 1024 output
- decode reports the Wan/Qwen bridge with `host_copies=1` and `device_copies=1`
- VAE Encode reports the Wan/Qwen bridge with `host_copies=1` and `device_copies=1`
- `SDCPP_STRICT_GPU_RESIDENT=1` refuses the sampler, decode, and encode bridges
- 1024 Anima VAE Encode/Decode currently uses the legacy Wan/Qwen IM2COL graph
  and plans roughly 7702 MB encode / 7493 MB decode workspace

### ControlNet

Validate:

- fp16 ControlNet files allocate fp16-sized destination parameter buffers
- Diffusers-style ControlNet keys map correctly for the tested model
- ControlNet outputs are kept backend-resident and passed to UNet without
  per-step host download/re-upload
- timing logs can separate context/model load cost from denoise-step cost

## Release Packaging

The Paralol Windows CUDA release asset contains exactly these files at the ZIP
root:

```text
stable-diffusion.dll
sd-cli.exe
sd-server.exe
sd-latent-smoke.exe
```

Build:

```powershell
cmd /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cmake --build build\codex --config Release --target stable-diffusion sd-cli sd-server sd-latent-smoke --parallel 6'
```

Package naming:

```text
sd-master-<short_commit>-paralol-<release_name>-bin-win-cuda-x64.zip
```

Tag naming:

```text
paralol-<release_name>-<short_commit>
```

Release notes should include:

- target commit
- ZIP SHA256
- `stable-diffusion.dll` SHA256
- short feature summary
- explicit compatibility caveats, especially strict GPU-resident limitations

After publishing, stage the DLL for local Paralol testing:

```powershell
Copy-Item build\codex\bin\stable-diffusion.dll F:\Paralol\build\runtime\stable-diffusion-paralol-latent\stable-diffusion.dll -Force
```

## Documentation Rules

When a fork change adds or changes public behavior, update at least one of:

- `docs/sdcpp-fork-upstream-readiness.md` for broad fork delta and upstream
  splitting strategy
- `docs/paralol-gpu-latent-flow.md` for GPU latent/image ownership and path
  support
- `docs/paralol-gpu-resident-values.md` for opaque handle API usage
- `docs/paralol-comfy-normal-vae.md` for VAE execution behavior
- `docs/paralol-flux-z-pipeline-capabilities.md` for Flux/Z model-family
  routing
- this document for release, validation, and maintenance process changes

Avoid burying important behavior only in release notes. Release notes disappear
from the working tree; docs keep future agents and maintainers aligned.

## Upstream Contribution Strategy

Do not upstream this integration branch as one PR. Split it into small,
reviewable branches. The likely order is documented in
`docs/sdcpp-fork-upstream-readiness.md`.

The best early candidates are:

1. DPM++ SDE / ER-SDE sampler support.
2. Caller-owned image download and DLL-owned free function.
3. ControlNet dtype-aware loading.
4. Model-family capability reporting, narrowed to stable facts.
5. COMFY_NORMAL VAE memory reporting and staged execution.
6. CUDA implicit-GEMM VAE convolution as a backend optimization.

Keep Paralol-specific names, local paths, and workflow policy out of upstream
candidate branches.

## Known Long-Term Debt

- `src/stable-diffusion.cpp` carries too much of the public API implementation.
  GPU handle registry, VAE normal execution, and capability reporting should be
  split into focused files before upstream review.
- `sd_sample_latent_gpu` is a bridge upload, not a true GPU-native sampler.
- VAE Encode GPU latent handoff is a safe bridge, not true resident. It avoids
  the unsafe same-context decode path by downloading the encoded latent and
  decoding in a cached VAE decode-only context.
- Anima VAE decode uses a Wan/Qwen compatibility bridge with host/device copies.
- Capability structs are useful but should be reviewed for minimal stable
  upstream API shape.
- Some smokes are local-machine oriented and need upstream-style test fixtures.

## Maintenance Rule

Prefer honest refusal over a partial feature that appears to work. If a path is
not safe, capability-gate it, make the API return a clear error, and document
the limitation.
