param(
    [switch]$Force,
    [string]$RepoRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$ExpectedBase = "404fcb9d7c96989569e68c9e7881ee3465a05c50"
$GgmlDir = Join-Path $RepoRoot "ggml"
$PatchDir = Join-Path $RepoRoot "patches\ggml\bf16-vae"

function Invoke-Git {
    param(
        [Parameter(Mandatory=$true)][string[]]$Args
    )
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $GgmlDir)) {
    Invoke-Git -Args @("-C", $RepoRoot, "submodule", "update", "--init", "ggml")
}

if (-not (Test-Path $PatchDir)) {
    throw "Patch directory not found: $PatchDir"
}

$patches = @(Get-ChildItem -Path $PatchDir -Filter "*.patch" | Sort-Object Name)
if ($patches.Count -eq 0) {
    throw "No patch files found in $PatchDir"
}

& git -C $GgmlDir cat-file -e "$ExpectedBase^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Expected ggml base $ExpectedBase is not local; fetching from origin..."
    Invoke-Git -Args @("-C", $GgmlDir, "fetch", "origin", $ExpectedBase)
}

$dirty = (& git -C $GgmlDir status --porcelain)
if ($dirty -and -not $Force) {
    throw "ggml has local changes. Re-run with -Force to reset ggml before applying BF16 VAE patches."
}

Write-Host "Resetting ggml to upstream base $ExpectedBase"
& git -C $GgmlDir am --abort 2>$null
Invoke-Git -Args @("-C", $GgmlDir, "reset", "--hard", $ExpectedBase)
Invoke-Git -Args @("-C", $GgmlDir, "clean", "-fd")

foreach ($patch in $patches) {
    Write-Host "Applying $($patch.Name)"
    Invoke-Git -Args @("-C", $GgmlDir, "am", "--3way", $patch.FullName)
}

$head = (& git -C $GgmlDir rev-parse HEAD).Trim()
Write-Host "ggml BF16 VAE patchset applied. ggml HEAD: $head"
