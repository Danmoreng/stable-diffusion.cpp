<#
    install_sd_cpp.ps1
    --------------------
    Installs all prerequisites and builds leejet/stable-diffusion.cpp on Windows.

    • Works on Windows PowerShell 7
    • Uses the Ninja generator (fast, no VS-integration dependency)
    • Re-usable: just run the script; it installs only what is missing
    • Pass -RepoUrl "https://github.com/DEIN_USER/dein-fork.git" to build your own fork!
#>

[CmdletBinding()]
param(
    [string]$RepoUrl = "https://github.com/Danmoreng/stable-diffusion.cpp",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Make PS5 iwr happy and TLS modern
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# --------------------------------------------------------------------------- 
# Helper functions (Identisch zu llama.cpp Script)
# --------------------------------------------------------------------------- 

function Assert-Admin {
    $id  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $prn = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $prn.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run this script from an *elevated* PowerShell window."
    }
}

function Test-Command ([string]$Name) {
    (Get-Command $Name -ErrorAction SilentlyContinue) -ne $null
}

function Test-VSTools {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $false }

    $instRoot = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null

    if ([string]::IsNullOrWhiteSpace($instRoot)) { return $false }

    $vcvars = Join-Path $instRoot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { return $false }

    $cl = Get-ChildItem -Path (Join-Path $instRoot 'VC\Tools\MSVC') `
        -Recurse -Filter cl.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $cl) { return $false }

    # Windows SDK tools (needed by CMake generator/linker steps)
    $sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin'
    $rc = Get-ChildItem $sdkBin -Recurse -Filter rc.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    $mt = Get-ChildItem $sdkBin -Recurse -Filter mt.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not ($rc -and $mt)) { return $false }

    return $true
}

# --- CUDA: generic discovery (12.4+ including 13.x) -------------------------

function Get-CudaInstalls {
    $root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (-not (Test-Path $root)) { return @() }
    $out = @()
    foreach ($d in Get-ChildItem $root -Directory) {
        $nvcc = Join-Path $d.FullName 'bin\nvcc.exe'
        if (($d.Name -match '^v(\d+)\.(\d+)$') -and (Test-Path $nvcc)) {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            $ver = [version]::new($maj, $min)
            $out += [pscustomobject]@{ Version=$ver; Major=$maj; Minor=$min; Path=$d.FullName }
        }
    }
    $out
}

function Test-CUDA {
    $min = [version]'12.4'
    $installs = Get-CudaInstalls
    if (-not $installs) { return $false }
    return ($installs | Where-Object { $_.Version -ge $min } | Select-Object -First 1) -ne $null
}

function Test-CUDAExact {
    param([Parameter(Mandatory=$true)][string]$MajorMinor) # e.g. '12.4'
    $target = [version]"$MajorMinor"
    $hit = Get-CudaInstalls | Where-Object {
        $_.Version.Major -eq $target.Major -and $_.Version.Minor -eq $target.Minor
    } | Select-Object -First 1
    return $null -ne $hit
}

function Install-CUDA124-FromNVIDIA {
    # Installs CUDA 12.4.1 silently (toolkit only; no driver, no GFE)
    $url = 'https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_551.78_windows.exe'
    $exe = Join-Path $env:TEMP 'cuda_12.4.1_551.78_windows.exe'
    if (-not (Test-Path $exe)) {
        Write-Host "-> downloading CUDA 12.4.1 (local installer) ..."
        Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    }

    # Toolkit-only selection (no driver = no GeForce Experience)
    $toolkitPkgs = @(
        'nvcc_12.4',         # compiler
        'cudart_12.4',       # CUDA runtime
        'cublas_12.4',       # cuBLAS runtime
        'cublas_dev_12.4'    # cuBLAS headers/libs for build
    )

    $args = @('-s') + $toolkitPkgs + '-n'

    Write-Host "-> installing CUDA 12.4.1 (silent, toolkit only) ..."
    $p = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -Wait -PassThru

    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
        throw "CUDA 12.4.1 installer failed with exit code $($p.ExitCode)."
    }

    Refresh-Env

    $nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe'
    if (-not (Test-Path $nvcc)) {
        throw "CUDA 12.4.1 appears not to be installed correctly (missing $nvcc)."
    }
    Write-Host "[OK] CUDA 12.4.1 (nvcc present)"
}

function Wait-Until ($TestFn, [int]$TimeoutMin, [string]$What) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $maxLen = 0
    while ($sw.Elapsed.TotalMinutes -lt $TimeoutMin) {
        if (& $TestFn) {
            $msg = "  $($What): done."
            $maxLen = [Math]::Max($maxLen, $msg.Length)
            Write-Host ("`r{0}{1}" -f $msg, ' ' * ($maxLen - $msg.Length)) -NoNewline
            Write-Host ""
            return
        }
        $msg = "  waiting for $($What) ... $($sw.Elapsed.ToString('mm\:ss'))"
        $maxLen = [Math]::Max($maxLen, $msg.Length)
        Write-Host ("`r{0}{1}" -f $msg, ' ' * ($maxLen - $msg.Length)) -NoNewline
        Start-Sleep -Milliseconds 250
    }
    Write-Host ""
    throw "$($What) did not finish in $TimeoutMin minutes."
}

function Refresh-Env {
    $machine = [Environment]::GetEnvironmentVariables('Machine')
    $user    = [Environment]::GetEnvironmentVariables('User')
    foreach ($k in $machine.Keys) { Set-Item -Path "Env:$k" -Value $machine[$k] }
    foreach ($k in $user.Keys)    { Set-Item -Path "Env:$k" -Value $user[$k] }
    $env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"
}

function Ensure-CommandAvailable([string]$Cmd, [int]$TimeoutMin = 5) {
    Refresh-Env
    Wait-Until { Test-Command $Cmd } $TimeoutMin "command '$Cmd' to appear on PATH"
}

function Add-ToMachinePath([string]$Dir) {
    if (-not (Test-Path $Dir)) { return }
    $regPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
    $current = (Get-ItemProperty -Path $regPath -Name Path).Path
    $parts = $current -split ';' | Where-Object { $_ -ne '' }
    if ($parts -contains $Dir) { return }
    $new = ($parts + $Dir) -join ';'
    Set-ItemProperty -Path $regPath -Name Path -Value $new
}

function Install-Winget {
    param(
        [Parameter(Mandatory=$true)][string]$Id,
        [string]$InstallerArgs = '',
        [string]$Version = ''
    )
    if (-not (Test-Command winget)) {
        throw "The 'winget' command is not available. Install the Microsoft 'App Installer' from the Store and try again."
    }
    Write-Host "-> installing $Id $($Version) ..."
    $argList = @(
        'install','--id',$Id,
        '--source','winget',
        '--silent','--disable-interactivity',
        '--accept-source-agreements','--accept-package-agreements'
    )
    if ($Version) { $argList += @('--version', $Version) }
    if ($InstallerArgs) { $argList += @('--custom', $InstallerArgs) }

    $log = Join-Path $env:TEMP ("winget_install_{0}.log" -f ($Id -replace '[^A-Za-z0-9]+','_'))

    & winget @argList *> $log
    $exitCode = $LASTEXITCODE

    if ($Version -and $exitCode -eq -1978335212) { throw "winget could not find $Id version $Version. See log: $log" }
    if ($exitCode -and $exitCode -notin @(-1978335189, -1978335212)) { throw "winget failed (exit $exitCode) while installing $Id. See log: $log" }
    Refresh-Env
}

function Install-VSTools {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "The 'winget' command is not available."
    }
    Write-Host "-> installing VS 2022 Build Tools (silent, via winget) ..."
    $installPath = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
    $customCommon = @(
        '--add Microsoft.VisualStudio.Workload.VCTools',
        '--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '--add Microsoft.VisualStudio.Component.VC.CoreBuildTools',
        '--add Microsoft.VisualStudio.Component.VC.Redist.14.Latest',
        '--includeRecommended',
        '--installPath "' + $installPath + '"'
    ) -join ' '

    $customWin11 = "$customCommon --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
    $customWin10 = "$customCommon --add Microsoft.VisualStudio.Component.Windows10SDK.19041"
    $logDir = Join-Path $env:TEMP "vsbuildtools_logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $log = Join-Path $logDir "winget_vstools.log"

    Get-Process -Name "vs_installer","VisualStudioInstaller" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    function Invoke-VsInstall([string]$customArgs) {
        & winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --silent --disable-interactivity --accept-source-agreements --accept-package-agreements --custom $customArgs *> $log
        return $LASTEXITCODE
    }

    $code = Invoke-VsInstall $customWin11
    if ($code -ne 0 -and $code -ne 3010) {
        Write-Host "  Win11 SDK component not available; retrying with Win10 SDK ..."
        $code = Invoke-VsInstall $customWin10
    }
    if ($code -ne 0 -and $code -ne 3010) { throw "VS Build Tools install failed (exit $code). See log: $log" }
    Refresh-Env
}

function Wait-VSToolsReady { Wait-Until { Test-VSTools } 20 'Visual Studio Build Tools' }
function Wait-CUDAReady    { Wait-Until { Test-CUDA    } 30 'CUDA Toolkit' }

function Import-VSEnv {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsroot  = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsroot) { throw "VS Build Tools not found." }
    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "VS C++ Build Tools look registered but vcvars64.bat is missing." }
    Write-Host "  importing MSVC environment from $vcvars"
    $envDump = cmd /s /c ""$vcvars"" && set"
    foreach ($line in $envDump -split "`r?`n") {
        if ($line -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
    }
}

function Install-NinjaPortable {
    if (Test-Command ninja) { return }
    Write-Host "-> installing Ninja (portable) ..."
    $url  = 'https://github.com/ninja-build/ninja/releases/latest/download/ninja-win.zip'
    $zip  = Join-Path $env:TEMP 'ninja-win.zip'
    $dest = 'C:\Program Files\Ninja'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -Force
    Add-ToMachinePath $dest
    Refresh-Env
    Ensure-CommandAvailable -Cmd 'ninja' -TimeoutMin 2
    Write-Host "[OK] Ninja"
}

function Use-LatestCuda {
    param([version]$Min=[version]'12.4',[version]$Prefer=$null)
    $installs = Get-CudaInstalls | Sort-Object Version -Descending
    if ($Prefer) {
        $pick = $installs | Where-Object { $_.Version.Major -eq $Prefer.Major -and $_.Version.Minor -eq $Prefer.Minor } | Select-Object -First 1
        if (-not $pick) { throw "Requested CUDA not found." }
    } else {
        $pick = $installs | Where-Object { $_.Version -ge $Min } | Select-Object -First 1
        if (-not $pick) { throw "No CUDA installation >= $Min found." }
    }
    $env:CUDA_PATH = $pick.Path
    Set-Item -Path ("Env:CUDA_PATH_V{0}_{1}" -f $pick.Major, $pick.Minor) -Value $pick.Path
    $cudaBin = Join-Path $pick.Path 'bin'
    if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $cudaBin })) { $env:Path = "$cudaBin;$env:Path" }
    Write-Host "  Using CUDA toolkit $($pick.Version) at $($pick.Path)"
    "-DCUDAToolkit_ROOT=$($pick.Path)"
}

function Get-GpuCudaArch {
    # Try generic NVML logic via System32 if available, or WMI fallback
    # (Simplified from original script for brevity, assuming standard NV logic)
    try {
        $gpu = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | Where-Object { $_.AdapterCompatibility -like '*NVIDIA*' } | Select-Object -First 1
        if ($gpu) {
            $name = $gpu.Name
            # Simple heuristic mapping
            if ($name -match 'RTX\s*50|Blackwell') { return 100 }
            if ($name -match 'RTX\s*40|Ada')       { return 89 }
            if ($name -match 'RTX\s*30|A\d+000')    { return 86 }
            if ($name -match 'RTX\s*20|T4|GTX\s*16'){ return 75 }
            if ($name -match 'GTX\s*10|P40')        { return 61 }
        }
    } catch {}
    return $null
}

# --------------------------------------------------------------------------- 
# Main routine
# --------------------------------------------------------------------------- 

Assert-Admin

# --- Base prerequisites ---
$reqs = @(
    @{ Name='Git'; Test={Test-Command git}; Id='Git.Git'; Cmd='git'; InstallerArgs='/VERYSILENT /NORESTART /SP- /NOCANCEL' },
    @{ Name='CMake'; Test={Test-Command cmake}; Id='Kitware.CMake'; Cmd='cmake'; InstallerArgs='ADD_CMAKE_TO_PATH=System ALLUSERS=1' },
    @{ Name='Node.js (LTS)'; Test={Test-Command npm}; Id='OpenJS.NodeJS.LTS'; Cmd='npm' },
    @{ Name='VS Build Tools'; Test={Test-VSTools}; Id='Microsoft.VisualStudio.2022.BuildTools' },
    @{ Name='Ninja'; Test={Test-Command ninja}; Id='Ninja-build.Ninja'; Cmd='ninja' }
)

$DetectedSm = Get-GpuCudaArch
$cudaReq = @{ Name='CUDA Toolkit'; Test={Test-CUDA}; Id='Nvidia.CUDA'; Version='' }
$PreferCudaVersion = $null

if ($DetectedSm) {
    if ($DetectedSm -lt 70) {
        Write-Host "-> GPU detected: sm_$DetectedSm (pre-Turing) – selecting CUDA 12.4 for compatibility."
        $cudaReq.Name = 'CUDA Toolkit 12.4'; $cudaReq.Version = '12.4.1'
        $cudaReq.Test = { Test-CUDAExact -MajorMinor '12.4' }
        $PreferCudaVersion = [version]'12.4'
    } else {
        Write-Host "-> GPU detected: sm_$DetectedSm – selecting latest CUDA."
    }
}
$reqs += $cudaReq

# --- Install all prerequisites ---
foreach ($r in $reqs) {
    if (-not (& $r.Test)) {
        switch ($r.Name) {
            'VS Build Tools' { Install-VSTools; Wait-VSToolsReady }
            'CUDA Toolkit 12.4' { Install-CUDA124-FromNVIDIA }
            default {
                $args = $r.ContainsKey('InstallerArgs') ? $r['InstallerArgs'] : ''
                $ver = $r.ContainsKey('Version') ? $r['Version'] : ''
                Install-Winget -Id $r.Id -InstallerArgs $args -Version $ver
                if ($r.ContainsKey('Cmd')) { Ensure-CommandAvailable -Cmd $r['Cmd'] }
            }
        }
        if (-not (& $r.Test)) { throw "$($r.Name) could not be installed." }
    }
    Write-Host ("[OK] {0}" -f $r.Name)
}

if (-not (Test-Command ninja)) { Install-NinjaPortable } else { Write-Host "[OK] Ninja" }

Import-VSEnv
if ($SkipBuild) { Write-Host 'SkipBuild set – done.'; return }

$cudaRootArg = Use-LatestCuda -Prefer $PreferCudaVersion

# --------------------------------------------------------------------------- 
# Build stable-diffusion.cpp (Frontend and Backend)
# --------------------------------------------------------------------------- 

# Script is in stable-diffusion.cpp/windows_scripts, repo root is parent
$SdRepo  = Split-Path -Parent $ScriptRoot
$SdBuild = Join-Path $SdRepo 'build'
$WebUiDir = Join-Path $SdRepo 'examples\server\webui'

if (-not (Test-Path $SdRepo)) {
    throw "stable-diffusion.cpp directory not found at '$SdRepo'."
}

# --- Build Frontend ---
if (-not (Test-Path (Join-Path $WebUiDir 'package.json'))) {
    throw "Could not find 'package.json' in the web UI directory: $WebUiDir"
}

Push-Location $WebUiDir
Write-Host "-> installing frontend dependencies in '$WebUiDir'..."
npm install
Write-Host "-> building frontend..."
npm run build
Pop-Location


# --- Update submodules ---
Write-Host "-> updating git submodules..."
git -C $SdRepo submodule update --init --recursive

# --- Configure & Build Backend ---
$CudaArchArg = $DetectedSm ? "$DetectedSm" : 'native'
Write-Host ("-> Using CMake Cuda Arch: {0}" -f $CudaArchArg)

New-Item $SdBuild -ItemType Directory -Force | Out-Null
Push-Location $SdBuild

Write-Host '-> generating CMake project for stable-diffusion.cpp ...'
cmake .. -G Ninja `
    -DSD_CUDA=ON `
    -DSD_BUILD_EXAMPLES=ON `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchArg" `
    $cudaRootArg

Write-Host '-> building targets: sd-cli (CLI) and sd-server (HTTP Server) ...'
cmake --build . --config Release --target sd-cli --parallel
cmake --build . --config Release --target sd-server --parallel

Pop-Location

# Ausgabe Pfade prüfen
$BinDir = Join-Path $SdBuild 'bin'
Write-Host ""
Write-Host "Done! Prüfe Binaries in: \"$BinDir\""
if (Test-Path (Join-Path $BinDir "sd-server.exe")) {
    Write-Host "-> [SUCCESS] sd-server.exe wurde erfolgreich gebaut!" -ForegroundColor Green
} else {
    Write-Host "-> [WARNING] sd-server.exe nicht gefunden. Prüfe den 'bin' Ordner manuell." -ForegroundColor Yellow
}
