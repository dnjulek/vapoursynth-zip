$jsonUrl = "https://raw.githubusercontent.com/ziglang/www.ziglang.org/master/data/releases.json"
$jsonContent = Invoke-RestMethod -Uri $jsonUrl -Method Get
$version = $jsonContent.master.version
$zipUrl = $jsonContent.master.'x86_64-windows'.tarball
$zipFilePath = Join-Path $PSScriptRoot "zig-windows-x86_64-$version.zip"
$zigPath = Join-Path $PSScriptRoot "\zig-windows-x86_64-$version\zig.exe"

if (-not (Test-Path $zipFilePath)){
    Write-Host "Downloading zig-windows-x86_64-$version.zip..." -ForegroundColor Green
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipFilePath
}

if (-not (Test-Path $zigPath)){
    Write-Host "Extracting zig-windows-x86_64-$version.zip..." -ForegroundColor Green
    Expand-Archive -Path $zipFilePath -DestinationPath $PSScriptRoot -Force
}

Write-Host "Building vszip.dll..." -ForegroundColor Green
& $zigPath build -Doptimize=ReleaseFast