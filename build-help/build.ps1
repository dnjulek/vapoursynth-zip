$version = '0.15.2'
$zipUrl = "https://ziglang.org/download/0.15.2/zig-x86_64-windows-0.15.2.zip"

$parentFolder = Split-Path -Path $PSScriptRoot -Parent
$zipFilePath = Join-Path $PSScriptRoot "zig-x86_64-windows-$version.zip"
$zigPath = Join-Path $PSScriptRoot "\zig-x86_64-windows-$version\zig.exe"
$dllFilePath = Join-Path $parentFolder "\zig-out\bin\vszip.dll"
$destinationFileFolder = Join-Path $env:APPDATA "\VapourSynth\plugins64"
New-Item -ItemType Directory -Force -Path $destinationFileFolder | Out-Null
$destinationFilePath = Join-Path $destinationFileFolder "\vszip.dll"

if (-not (Test-Path $zipFilePath)){
    Write-Host "Downloading zig-x86_64-windows-$version.zip..." -ForegroundColor Green
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipFilePath
}

if (-not (Test-Path $zigPath)){
    Write-Host "Extracting zig-x86_64-windows-$version.zip..." -ForegroundColor Green
    Expand-Archive -Path $zipFilePath -DestinationPath $PSScriptRoot -Force
}

Write-Host "Building vszip.dll..." -ForegroundColor Green
& $zigPath build -Doptimize=ReleaseFast

if (Test-Path $dllFilePath){
    Write-Host "Installing 'vszip.dll' to '$destinationFilePath'" -ForegroundColor Blue
    Copy-Item -Path $dllFilePath -Destination $destinationFilePath -Force
} else {
    Write-Host "No '\zig-out\bin\vszip.dll' file, build error." -ForegroundColor Red
}
