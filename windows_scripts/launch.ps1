# This script launches the C++ backend server.
# It now starts without an initial model, allowing selection via the WebUI.

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Script is in stable-diffusion.cpp/windows_scripts, repo root is parent
$SdRepo = Split-Path -Parent $ScriptRoot
# Project root (where models/ are) is grandparent
$ProjectRoot = Split-Path -Parent $SdRepo

Write-Host "Launching sd-server.exe..." -ForegroundColor Green
Write-Host "Server logs will appear below. Press Ctrl+C in this window to stop the server." -ForegroundColor Gray

$serverBinDir = Join-Path $SdRepo "build\bin"
$serverExe = Join-Path $serverBinDir "sd-server.exe"

if (-not (Test-Path $serverExe)) {
    Write-Host "Server executable not found at '$serverExe'." -ForegroundColor Red
    Write-Host "Please build the project first." -ForegroundColor Red
    exit 1
}

$modelDir = Join-Path $ProjectRoot "models"

Push-Location $serverBinDir

try {
    # Execute the server with only the model directory.
    # No initial model is passed, so the user can select one in the UI.
    Write-Host "Starting server (Model directory: $modelDir)..." -ForegroundColor Green
    & $serverExe --model-dir "$modelDir" --listen-port 1234
}
finally {
    Pop-Location
    Write-Host "`nServer stopped. Returned to: $(Get-Location)" -ForegroundColor Gray
}
