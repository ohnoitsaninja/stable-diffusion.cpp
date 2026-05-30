param(
    [switch] $Graph,
    [int] $Runs = 3,
    [int] $Warmup = 1,
    [string] $OutputDir = "",
    [string] $BuildDir = "build-bonsai-int1-cuda133",
    [string] $DiffusionModel = "F:\automatic1111\Stability\Models\DiffusionModels\bonsai-image-binary-4B-gemlite-1bit\transformer-gemlite-int1\state_dict.pt",
    [string] $Llm = "F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    [string] $Vae = "F:\Paralol\local\stable-diffusion.cpp-speed\build-bonsai-int1\bonsai-reference-components\vae\diffusion_pytorch_model.safetensors"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "CMakeLists.txt") -or -not (Test-Path "examples\bonsai-gemlite-int1-bench\main.cpp")) {
    throw "Run this script from the stable-diffusion.cpp-speed repo root."
}

$exe = Join-Path $BuildDir "bin\sd-bonsai-gemlite-int1-bench.exe"
if (-not (Test-Path $exe)) {
    throw "Benchmark executable not found: $exe"
}

foreach ($path in @($DiffusionModel, $Llm, $Vae)) {
    if (-not (Test-Path $path)) {
        throw "Required model/component path not found: $path"
    }
}

$clearVars = @(
    "SDCPP_BONSAI_DUMP_TENSORS",
    "SDCPP_BONSAI_DUMP_TENSOR_LIMIT",
    "SDCPP_BONSAI_DUMP_DIR",
    "SDCPP_BONSAI_CONDITIONING_NPY",
    "SDCPP_BONSAI_NOISE_NPY",
    "SDCPP_PROFILE_BONSAI_INT1",
    "SDCPP_PROFILE_BONSAI_INT1_FAMILIES",
    "SDCPP_PROFILE_BONSAI_INT1_RECORD_EVENTS",
    "SDCPP_PROFILE_BONSAI_INT1_FILTER",
    "SDCPP_PROFILE_BONSAI_INT1_MAX_CALLS",
    "SDCPP_PROFILE_BONSAI_INT1_SHAPE_CENSUS_JSON",
    "SDCPP_BONSAI_PROFILE_RANGE",
    "SDCPP_TRACE_BONSAI_STREAMS",
    "SDCPP_TRACE_BONSAI_INT1_STATS",
    "SDCPP_BONSAI_FORCE_DEVICE_SYNC",
    "SDCPP_BONSAI_CUDA_GRAPH_TRACE",
    "SDCPP_BONSAI_CUDA_GRAPH_DENOISE",
    "SDCPP_TRACE_EULER_PARITY",
    "SDCPP_TRACE_EULER_PARITY_TENSORS",
    "SDCPP_PROFILE_COPY_BREAKDOWN",
    "SDCPP_PROFILE_RUNNER_TIMINGS",
    "SDCPP_BONSAI_INT1_LINEAR1_BANKFIX",
    "SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM",
    "SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX",
    "SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B",
    "SDCPP_BONSAI_INT1_TILED_KERNEL"
)

foreach ($name in $clearVars) {
    Remove-Item "Env:\$name" -ErrorAction SilentlyContinue
}

$env:SDCPP_EXPERIMENTAL_BONSAI_GEMLITE_INT1 = "1"
$env:SDCPP_MODEL_FAMILY_HINT = "bonsai"
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2 = "1"
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK = "1"
$env:SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT = "3"
$env:SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC = "1"
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0 = "1"
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2 = "1"
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_QKV = "1"
$env:SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ = "1"

if ($Graph) {
    $env:SDCPP_BONSAI_CUDA_GRAPH_DENOISE = "1"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $mode = if ($Graph) { "graph" } else { "nongraph" }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = "build-bonsai-int1\bonsai-cuda133-eval\best-profile-$mode-$stamp"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stdout = Join-Path $OutputDir "stdout.log"
$stderr = Join-Path $OutputDir "stderr.log"

$args = @(
    "--diffusion-model", $DiffusionModel,
    "--llm", $Llm,
    "--vae", $Vae,
    "--prompt", "A bonsai tree in a quiet ceramic studio, soft morning light",
    "--width", "512",
    "--height", "512",
    "--steps", "4",
    "--seed", "42",
    "--sampler", "euler",
    "--scheduler", "discrete",
    "--cfg-scale", "1.0",
    "--guidance", "1.0",
    "--flow-shift", "3.0",
    "--prediction", "flux2_flow",
    "--warmup", "$Warmup",
    "--runs", "$Runs",
    "--output-dir", $OutputDir
)

$runMode = if ($Graph) { "graph" } else { "nongraph" }
Write-Host "mode=$runMode"
Write-Host "exe=$exe"
Write-Host "output_dir=$OutputDir"
Write-Host "runs=$Runs warmup=$Warmup"

& $exe @args > $stdout 2> $stderr
$exitCode = $LASTEXITCODE

$stdoutText = Get-Content -Path $stdout -Raw
$stderrText = Get-Content -Path $stderr -Raw

if ($exitCode -ne 0) {
    Write-Host "exit_code=$exitCode"
    Write-Host "stdout=$stdout"
    Write-Host "stderr=$stderr"
    throw "Bonsai benchmark failed."
}

$summary = [regex]::Match($stdoutText, "\[BonsaiBench\] summary.*")
$denoiseMedian = ""
$fullMedian = ""
if ($summary.Success) {
    $dm = [regex]::Match($summary.Value, "denoise_median_ms=([0-9.]+)")
    $fm = [regex]::Match($summary.Value, "full_median_ms=([0-9.]+)")
    if ($dm.Success) { $denoiseMedian = $dm.Groups[1].Value }
    if ($fm.Success) { $fullMedian = $fm.Groups[1].Value }
}

$runOutputs = [regex]::Matches($stdoutText, "output=([^\r\n]+\.png)") | ForEach-Object { $_.Groups[1].Value.Trim() }
$firstImage = $runOutputs | Select-Object -First 1
$lastImage = $runOutputs | Select-Object -Last 1
$hash = ""
if ($firstImage -and (Test-Path $firstImage)) {
    $hash = (Get-FileHash -Algorithm SHA256 -Path $firstImage).Hash
}

$failFastTriggered = ($stdoutText + "`n" + $stderrText) -match "(fail-fast|non-finite|nonfinite|NaN/Inf|has nan|has inf)"

Write-Host "denoise_median_ms=$denoiseMedian"
Write-Host "full_median_ms=$fullMedian"
Write-Host "first_image=$firstImage"
Write-Host "last_image=$lastImage"
Write-Host "sha256_first_image=$hash"
Write-Host "fail_fast_or_nonfinite_seen=$failFastTriggered"
Write-Host "stdout=$stdout"
Write-Host "stderr=$stderr"
