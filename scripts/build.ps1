# Stable Diffusion CPP Build Script
param (
    [switch]$Clean
)

function Import-VSEnv {
    Write-Host "Loading Visual Studio Environment..."
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return }
    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsroot) { return }
    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { return }
    $envDump = cmd /c "call `"$vcvars`" > nul && set PATH && set INCLUDE && set LIB"
    $envDump | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}
Import-VSEnv

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $ProjectRoot "build"
$OriginalLocation = Get-Location

try {

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Path $BuildDir -Recurse -Force
}

if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir
}
Set-Location $BuildDir

# Configure
Write-Host "Configuring CMake..."
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $Generator = "-G Ninja"
} else {
    $Generator = ""
}

cmake $ProjectRoot $Generator `
    -DSD_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release 

# Build
Write-Host "Building stable-diffusion..."
cmake --build . --config Release --parallel

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
}
finally {
    Set-Location $OriginalLocation
}
