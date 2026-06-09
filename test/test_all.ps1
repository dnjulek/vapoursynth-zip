$ErrorActionPreference = "Stop"

$scripts = @(
    "adaptive_binarize", "bilateral", "boxblur", "checkmate",
    "clahe", "color_map", "comb_mask", "comb_mask_mt",
    "limit_filter", "limiter", "packrgb", "plane_props",
    "planeminmax", "rfs_mismatch", "ssimulacra2"
)

$pass = 0
$total = $scripts.Count

foreach ($name in $scripts) {
    Write-Host "[$($pass + 1)/$total] $name"
    vspipe (Join-Path $PSScriptRoot "$name.vpy") .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $name"
        exit $LASTEXITCODE
    }
    $pass++
}

Write-Host "All $total tests passed"
