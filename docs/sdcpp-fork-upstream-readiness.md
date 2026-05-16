# stable-diffusion.cpp Fork Upstream Readiness

This note documents the current Paralol `stable-diffusion.cpp` fork work,
why it exists, how it is implemented today, and what should be refactored before
we consider proposing any of it upstream.

Treat the current fork as a working integration spike, not as upstream-ready
code. It was built quickly to unblock Paralol's native SD node graph. It proves
several useful ideas, but the branch is too broad and too Paralol-shaped to send
as one pull request.

For the operational maintenance checklist, release packaging rules, and future
upstream-sync process, see
`docs/paralol-fork-maintenance.md`.

## Current Baseline

- Fork repo: `ohnoitsaninja/stable-diffusion.cpp`
- Upstream parent: `leejet/stable-diffusion.cpp`
- Current fork branch commit:
  `6f5fbee988bc239eabadac59b805711b09ee78b0`
- Latest tagged release observed on this branch:
  `paralol-z-anime-beta-ca9050db3`
- Latest tagged release commit:
  `ca9050db3`
- Latest tagged release asset:
  `sd-master-ca9050db3-paralol-z-anime-beta-bin-win-cuda-x64.zip`
- Latest local staged DLL:
  `F:\Paralol\build\runtime\stable-diffusion-paralol-latent\stable-diffusion.dll`
- Staged DLL SHA256:
  `490165FA25FB4293B6B7C112D4CA4D7EC566D0205BDEF31EC15293685A5FD54B`
- Local fork branch observed:
  `paralol/comfy-normal-vae-cudnn`

The current branch stack is roughly:

1. `7ba087140` - Add Paralol latent API
2. `87f1783fe` - Add DPM++ SDE samplers
3. `b0fc82f3c` - Make VAE decode-only contexts skip CLIP and UNet
4. `cc1d2b955` - Add COMFY_NORMAL full-frame VAE encode/decode path
5. `ec4d8b283` - Add implicit-GEMM COMFY_NORMAL VAE backend
6. `5344579a1` - Optimize VAE image tensor packing
7. `c32369795` - Add GPU-resident VAE decode handles
8. `2e351a349` - Add GPU latent handle sampler handoff
9. `4e26875d3` - Harden GPU latent handoff contract
10. `2fac608a6` - Add caller-owned GPU image download API
11. `2061b5a8f` - Fix ControlNet dtype-aware loading
12. `855068ba4` - Keep ControlNet outputs on backend
13. `a8e80ee0a` - Add ControlNet denoise timing trace
14. `4eb47e64c` - Support scaled FP8 safetensors loading
15. `e96f1ba6d` - Add Flux and Z pipeline handoff capabilities
16. `ca9050db3` - Add beta sigma scheduler
17. `6f5fbee98` - Add Anima latent handoff support

Measured against the upstream base commit we started from, the current fork
changes about 40 files with roughly 8,623 insertions and 320 deletions. That is
not a reviewable upstream PR shape.

## Goals

The fork was created to support a ComfyUI-style Paralol SD graph:

```text
Load Image -> VAE Encode -> KSampler -> Latent Decode -> Preview/Save
Text Encode -> KSampler
```

The main goals were:

- expose real latent boundaries instead of forcing every graph to be a single
  `generate_image` call
- let KSampler output a latent that a later node can consume
- let VAE Encode and Latent Decode be independent nodes
- avoid unnecessary CPU image or latent materialization between nodes where the
  runtime can keep data on CUDA
- keep model resources, latents, and image outputs under explicit ownership
- support SDXL sampler choices used by common ComfyUI workflows, especially
  DPM++ SDE variants
- make normal full-frame SDXL VAE encode/decode fast and memory-bounded without
  secretly substituting tiled VAE or TAESD
- load SDXL ControlNet weights with the intended dtype and avoid needless
  per-step backend-to-host roundtrips
- expose Flux2, Z-Image, and Anima model-family metadata so Paralol can route
  non-SDXL latent shapes and text encoder requirements from capabilities rather
  than filename guesses
- make memory behavior diagnosable through reports, capability flags, strict
  modes, and smoke tools
- keep unsupported GPU paths honest rather than hiding unsafe partial behavior

The important product goal is not "use this exact API forever." The important
goal is proving that a modular native workflow can keep expensive values resident
and move explicit materialization to real output boundaries.

## What We Implemented

### Latent C API

The first change added public APIs around latent values:

- `sd_sample_latent`
- `sd_encode_image`
- `sd_decode_latent`
- `free_sd_latent`
- `free_sd_image`
- latent view/export/import helpers used by Paralol and the NN latent upscale
  spike

This let Paralol model `KSampler`, `VAE Encode`, and `Latent Decode` as separate
worker nodes. It also gave the worker a place to preserve resident latent handles
across node execution without turning the host into a model runtime.

Current upstream-readiness issue: the API was designed around Paralol's immediate
graph needs. Before upstreaming, it needs a smaller general-purpose problem
statement, tighter ownership docs, and tests that are useful outside Paralol.

### DPM++ SDE Samplers

The fork added sampler enum values, CLI/server plumbing, and denoiser
implementations for DPM++ SDE variants:

- `dpmpp_sde`
- `dpmpp_sde_gpu`
- `dpmpp_2m_sde`
- `dpmpp_2m_sde_gpu`
- `dpmpp_2m_sde_heun`
- `dpmpp_2m_sde_heun_gpu`
- `dpmpp_3m_sde`
- `dpmpp_3m_sde_gpu`
- `er_sde`

This was done because the SDXL Lightning model we were testing expected a
ComfyUI sampler named `dpm_sde`, and the upstream sampler list did not match
that workflow well enough. Later Anima testing also needed `er_sde`, and
Flux/Z testing needed flow-aware SDE logSNR handling plus beta sigma scheduler
plumbing.

Current upstream-readiness issue: this is probably the easiest part to separate,
but it still needs review against upstream sampler naming, ComfyUI/k-diffusion
parity, RNG/noise behavior, and CPU/GPU suffix semantics.

### Model Pipeline Capabilities

The fork added `sd_get_model_pipeline_capabilities(...)` so Paralol can ask the
context what a loaded model family needs instead of hard-coding SDXL-shaped
assumptions.

The capability report includes:

- model family and `family_name`
- latent channel count and VAE scale factor
- default sampler, scheduler, CFG, step count, and flow shift
- required text encoder lanes such as CLIP-L, T5XXL, or Qwen/Qwen3 LLM
- reference/edit-mode support
- GPU sampled-latent bridge support
- GPU latent decode and GPU image output support
- whether VAE Encode is safe for the family

This was added for Flux2 Klein, Z-Image/Z-Anime, and Anima, whose latent shapes
and conditioning paths do not match SDXL. Flux2 uses 128-channel latents and
reference/edit conditioning. Z-Image uses 16-channel latents with a Qwen LLM and
Flux-style AE path. Anima uses 16-channel latents and a Wan/Qwen VAE path.

Current upstream-readiness issue: the capability idea is useful outside Paralol,
but the exact struct fields were selected to unblock the worker. Before
upstreaming, the API should be narrowed to stable model-family facts and any
Paralol-specific workflow policy should move out of the fork.

### Decode-Only and Model Resource Release

The fork added behavior and APIs to reduce needless residency:

- decode-only contexts can skip CLIP and UNet loading where appropriate
- targeted release functions allow freeing CLIP or diffusion model params before
  VAE decode
- Paralol can use this to avoid holding large model resources while doing a
  VAE-only node

Current upstream-readiness issue: upstream may prefer a more general context
lifetime/resource API rather than Paralol-specific release functions. The code
should be reframed around "partial context residency" and "explicit component
release" with documented post-release validity rules.

### ControlNet Loading and Per-Step Handoff

The fork fixed two ControlNet issues that blocked Paralol SDXL workflows:

- ControlNet destination tensor dtypes now honor the context weight type and
  tensor rules before allocating the parameter buffer. This prevents fp16
  ControlNet files from being allocated as fp32-sized destination tensors.
- Diffusers-style ControlNet key mapping was extended for models that use keys
  such as `controlnet_down_blocks.*`.
- ControlNet outputs are kept on the backend and fed into UNet without the old
  per-step host materialization and re-upload path.
- ControlNet denoise timing and handoff diagnostics were added so the worker can
  distinguish model load cost from per-step ControlNet compute cost.

Observed local result: the fp16 ControlNet parameter buffer loads around
`2387.61 MB` instead of the earlier fp32-sized allocation, and the per-step path
no longer bounces large ControlNet tensors through CPU memory.

Current upstream-readiness issue: the dtype fix is a plausible upstream bug fix,
but the performance diagnostics and backend-handoff shape should be split from
the loader fix.

### COMFY_NORMAL VAE Path

The fork added a normal full-frame VAE encode/decode mode:

- `SD_VAE_EXEC_COMFY_NORMAL`
- `SD_VAE_EXEC_AUTO`
- `sd_encode_image_normal`
- `sd_decode_latent_normal`
- `sd_vae_run_options_t`
- `sd_vae_memory_report_t`
- `sd_vae_capabilities_t`
- strict guards for no tiled VAE, no TAESD, no legacy IM2COL, and no host stage
  copies in the intended CUDA path

This work was motivated by a concrete problem: normal SDXL 1024 VAE decode was
planning a 7680.25 MB workspace in the legacy graph path. Tiled VAE and TAESD
were not acceptable defaults because they change output behavior. The fork
instead added a staged full-frame VAE path intended to match ComfyUI normal VAE
semantics.

Observed fork-side numbers:

- legacy IM2COL decode workspace: 7680.25 MB
- COMFY_NORMAL staged decode workspace: 2816 MB
- COMFY_NORMAL staged encode workspace: 1536 MB
- implicit-GEMM VAE encode: about 0.7s
- implicit-GEMM VAE decode: about 0.8s
- Comfy reference: about 2371.94 MiB allocated / 3328 MiB reserved
- parity against Comfy normal VAE: mean absolute diff about `0.00295`, p99 about
  `0.0196`, PSNR about `45.17 dB`

Current upstream-readiness issue: this is valuable, but it is deep runtime work.
It touches graph planning, VAE stages, CUDA behavior, memory reporting, and
public API all at once. It should be split into smaller pieces before review.

### CUDA Implicit-GEMM VAE Convolution

The fork added a CUDA implicit-GEMM path scoped around COMFY_NORMAL VAE work.
This fixed the unacceptable direct-conv speed path where decode could take
around 12 seconds even though the lower-memory workspace was correct.

Current upstream-readiness issue: this needs to be presented as a backend
optimization with independent benchmarks and correctness tests. It should not be
buried inside a large Paralol API branch.

### GPU Handle API

The fork added same-process opaque GPU handles:

- `sd_gpu_handle_t`
- retain/release/get-desc/debug-name helpers
- CUDA latent/image descriptors with dtype, layout, shape, byte size, flags, and
  backend
- `sd_get_gpu_capabilities`
- `sd_decode_latent_normal_gpu`
- `sd_decode_gpu_latent_normal_gpu`
- `sd_gpu_latent_download`
- `sd_cpu_latent_upload`
- `sd_gpu_image_download`
- `sd_gpu_image_download_to_buffer`
- `sd_free_downloaded_image`

The supported Paralol path is currently:

```text
sd_sample_latent_gpu
-> bridge-uploaded CUDA latent handle
-> sd_decode_gpu_latent_normal_gpu
-> CUDA image handle
-> sd_gpu_image_download_to_buffer at the output boundary
```

This is useful, but it is intentionally not described as a true all-GPU sampler.
The sampler still materializes the final latent through the existing
`sd::Tensor<float>` path. `sd_sample_latent_gpu` uploads that sampled latent to
CUDA and marks it with a bridge-upload flag. In strict mode it refuses to run so
callers cannot claim zero-copy or fully GPU-resident sampling.

For Anima, GPU latent decode is also a compatibility bridge rather than a true
GPU-resident VAE path. A sampled 16-channel CUDA latent handle is downloaded into
the known-good Wan/Qwen VAE decode path and the decoded RGB tensor is uploaded
back into a GPU image handle. The report marks this with `host_copies=1`,
`device_copies=1`, and bridge flags. `SDCPP_STRICT_GPU_RESIDENT=1` refuses this
path.

Current upstream-readiness issue: this API is the most likely to need redesign
before upstreaming. Upstream may prefer a different abstraction such as a
ggml-backed tensor handle, DLPack-style export, or backend buffer ownership
instead of a Paralol-shaped opaque handle registry.

### Safety Refusals

The current DLL refuses several paths on purpose:

- strict mode refuses `sd_sample_latent_gpu` because sampler internals are not
  true GPU-resident
- VAE Encode GPU latent output is disabled
- VAE-encoded CPU latent upload into GPU decode is disabled
- GPU decode refuses handles marked as VAE-encoded latents
- Anima modular VAE Encode is disabled through model capabilities because the
  separated `VAE Encode -> KSampler -> Decode` path is not safe with the current
  Wan/Qwen VAE implementation
- strict mode refuses Anima's compatibility bridge because it contains an
  explicit latent download and decoded-image upload

This came from testing that found an unsafe VAE-encoded latent GPU handoff path.
The current behavior is conservative: T2I sampled-latent handoff works, I2I VAE
Encode GPU handoff does not pretend to work. Direct one-shot `sd-cli`
image-to-image remains available for model families where upstream supports it,
but Paralol's modular VAE Encode node should follow the capability report.

Current upstream-readiness issue: these refusals are good engineering behavior,
but they are also evidence that the GPU API is not mature enough to upstream as a
general resident-value system yet.

### Smoke and Diagnostic Tools

The fork added diagnostics and smoke harnesses:

- `examples/latent-smoke`
- `examples/vae-op-bench`
- VAE parity/regression scripts
- capability smoke script
- fork docs for COMFY_NORMAL VAE, GPU latent flow, GPU resident values, and VAE
  kernel feasibility
- Flux/Z/Anima handoff capability documentation
- ControlNet dtype and backend-handoff diagnostics

Current upstream-readiness issue: the tools are useful, but they need cleanup
and upstream-style names. Anything called `paralol-*` should either stay in our
fork or be renamed/reframed before submission.

## What Is Not Upstream-Ready

The current fork should not be submitted as-is because:

- it mixes several independent features in one branch
- it adds a large public C API surface before the abstractions are fully settled
- much of the code is designed around Paralol's immediate workflow needs
- it includes names and docs that are explicitly Paralol-specific
- GPU handles are same-process only and not yet a complete interop story
- the sampler GPU path is a bridge upload, not true GPU-resident sampling
- VAE Encode GPU handoff is known unsupported
- strict modes and environment flags are useful for development but need a
  cleaner upstream configuration story
- the VAE path, GPU handle registry, and public API are concentrated in
  already-large files like `src/stable-diffusion.cpp`
- some tests are local workflow/harness style rather than upstream CI-ready
- the code has not had human line-by-line review for upstream maintainability

The right mental model is: this fork proves product direction and exposes useful
candidate patches. It is not itself a polished upstream contribution.

## Refactor Before Upstream

### 1. Rebase and Produce a Clean Diff Map

Create a new upstream-review branch from current upstream `master`, then
cherry-pick or reimplement one feature at a time. Keep a diff map from the
current working fork to the candidate upstream branches so we know what has been
preserved, rewritten, or dropped.

Do not start from the full `paralol/comfy-normal-vae-cudnn` branch and try to
clean it in place.

### 2. Remove Paralol Branding From Generic Changes

Anything intended upstream should be framed as a general
`stable-diffusion.cpp` capability:

- use generic doc names
- remove Paralol-specific comments from code paths
- keep Paralol integration notes in the Paralol repo
- avoid saying a feature exists only for Paralol unless it truly does

### 3. Split Public API From Backend Internals

Separate API additions from implementation changes. For example:

- one small PR for latent encode/sample/decode CPU APIs
- one small PR for memory reports/capability structs
- one PR for VAE execution mode selection
- one PR for GPU image download ownership
- one later RFC/PR for GPU handles

Every public struct needs:

- `struct_size`
- versioning
- clear ownership rules
- reserved fields
- documented valid/invalid states
- tests for older callers passing smaller structs if ABI compatibility matters

### 4. Move Large Helpers Out of Monolithic Files

Before upstream review, split implementation into focused files where upstream
style allows it:

- GPU handle registry/resource helpers
- VAE execution/memory report helpers
- COMFY_NORMAL stage orchestration
- sampler-specific DPM++ SDE math

Avoid adding another thousand lines to `src/stable-diffusion.cpp` unless the
upstream project clearly prefers that shape.

### 5. Replace Environment-Only Controls With API/CLI Controls

Development flags like these are useful:

- `SDCPP_STRICT_GPU_RESIDENT`
- `SDCPP_VAE_STRICT_COMFY_NORMAL`
- `SDCPP_TRACE_VAE_STAGES`
- `SDCPP_TRACE_GRAPH_ALLOC`
- `SDCPP_DISABLE_COMFY_NORMAL_VAE`
- `SDCPP_DISABLE_VAE_IMPLICIT_GEMM_CONV`

For upstream, the normal path should be explicit options, CLI flags, and
capability reports. Environment variables can remain debug escapes, but they
should not be the only documented control surface.

### 6. Keep Backend Support Honest

CUDA-specific work must be guarded cleanly. CPU, Vulkan, Metal, OpenCL, and SYCL
builds should either compile and report unsupported capabilities, or have clear
compile-time guards. A feature that only works on CUDA is acceptable only if the
unsupported behavior elsewhere is explicit and tested.

### 7. Make Tests Reviewable

Each upstream candidate PR should include the smallest useful test surface:

- API smoke for symbol availability and struct initialization
- correctness comparison against existing output or Comfy reference where
  appropriate
- memory report regression checks for VAE workspace size
- backend capability tests that prove unsupported paths refuse clearly
- sampler determinism/parity checks for DPM++ SDE variants

Local machine paths, Paralol workflow paths, and Windows-only assumptions should
not be required for upstream tests.

### 8. Document Ownership First

The upstream review should never have to infer who owns memory. Every API needs
explicit ownership language:

- caller-owned input
- DLL-owned output
- exact free function
- whether a handle belongs to an `sd_ctx_t`
- whether handles survive context release
- whether borrowed CUDA pointers are same-process and temporary
- whether downloads synchronize

The `sd_gpu_image_download_to_buffer` addition is a good example of the shape we
should prefer: caller-owned destination, explicit size and stride validation, no
ambiguous returned allocation.

## Suggested Upstream PR Sequence

Do not send these as one PR. The likely order is:

1. **DPM++ SDE sampler support**
   - smallest self-contained value
   - include sampler names, CLI/server plumbing, docs, and parity notes
   - verify naming against upstream conventions before opening

2. **Caller-owned image download/free ownership cleanup**
   - low-risk memory ownership improvement
   - could be useful even without the full GPU handle story
   - document exact allocation/free contract

3. **ControlNet dtype-aware loading**
   - focused loader fix for destination tensor dtype and allocation ordering
   - include Diffusers key mapping only if it is needed by the same failing
     model family
   - keep per-step performance changes and timing diagnostics out of this PR

4. **ControlNet backend output handoff**
   - keep ControlNet outputs backend-resident and feed them into UNet without
     host materialization
   - include timing counters for ControlNet compute and output upload/download
   - depends on reviewers accepting the tensor ownership shape

5. **CPU latent API**
   - `sd_sample_latent`, `sd_encode_image`, `sd_decode_latent`, latent
     view/export/import if still needed
   - keep GPU handles out of this PR
   - prove ComfyUI-style node split with CPU-resident values first

6. **Component release / decode-only context cleanup**
   - generalize CLIP/UNet/VAE residency controls
   - document which APIs remain valid after release
   - avoid naming this around Paralol's `unloadAfterRun`

7. **Model-family capability reporting**
   - narrow `sd_get_model_pipeline_capabilities(...)` to stable facts: family,
     latent channels, VAE scale, required text encoders, edit/reference support
   - keep Paralol workflow policy and local defaults out of the upstream API
   - use Flux2/Z-Image/Anima as validation cases

8. **VAE memory reporting and capabilities**
   - add reports and capability discovery without changing the default execution
     path yet
   - this gives reviewers visibility before changing behavior

9. **COMFY_NORMAL full-frame VAE execution**
   - introduce the staged normal VAE path behind explicit selection or safe auto
     selection
   - include Comfy parity and memory numbers
   - no tiled/TAESD substitution

10. **CUDA implicit-GEMM VAE convolution optimization**
   - backend-specific optimization with benchmark data
   - keep an escape hatch for direct-conv diagnostics
   - prove it does not alter output beyond expected float tolerance

11. **GPU image handle output**
   - start with VAE decode output only
   - keep final CPU download explicit
   - do not include sampler GPU bridge yet

12. **GPU latent handles and sampled-latent bridge**
   - only after upstream agrees on resource-handle shape
   - be honest that this is a CPU-materialized sampled latent upload
   - keep strict mode refusal so it cannot be marketed as zero-copy sampling

13. **True GPU-resident sampler RFC**
   - likely needs design discussion before code
   - requires changing denoiser callback/sampler math away from host
     `sd::Tensor<float>` materialization
   - should probably be an issue/RFC before a PR

## Review Checklist Before Any PR

Before opening an upstream PR, we should be able to answer:

- What exact user problem does this PR solve without mentioning Paralol?
- Is the PR small enough to review in one sitting?
- Does it compile without CUDA?
- Does it preserve existing CLI behavior by default?
- Are all public APIs versioned and documented?
- Are allocation and release rules unambiguous?
- Are unsupported backends or unsafe paths refused clearly?
- Is there a smoke/regression test that upstream maintainers can run?
- Are benchmark claims backed by commands and hardware notes?
- Did a human review every changed line for style, naming, and lifetime issues?
- Can the feature be reverted independently without breaking unrelated fork work?

## Immediate Next Step

Keep the released fork pinned for Paralol while we create a separate
`upstream-review/*` branch series. The first practical candidate should be
DPM++ SDE sampler support or the caller-owned download/free cleanup, because
those are easier to isolate than the COMFY_NORMAL VAE and GPU handle work.

For anything involving GPU handles or all-GPU sampler claims, start with a design
issue or RFC. The current bridge-upload path is useful for Paralol, but upstream
review should not be asked to accept it as a final GPU-resident sampler design.
