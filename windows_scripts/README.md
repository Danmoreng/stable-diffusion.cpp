# Windows Scripts

This folder contains PowerShell scripts to automate the installation, build, and execution of `stable-diffusion.cpp` on Windows.

## Scripts

### 1. `install_sd_cpp.ps1`
This script automates the entire setup process.
- **Prerequisites**: It checks for and installs (via `winget`):
    - Git
    - CMake
    - Node.js (LTS)
    - Ninja (Portable)
    - Visual Studio 2022 Build Tools (C++ Workload)
    - CUDA Toolkit (Targeting 12.4+ depending on GPU)
- **Build**: 
    - Installs frontend dependencies and builds the Vue.js Web UI.
    - Configures and builds the C++ backend (`sd-server.exe` and `sd-cli.exe`) using Ninja for fast compilation.
    - Enables CUDA support by default.

**Usage**:
Run from an **elevated** (Administrator) PowerShell window:
```powershell
.\install_sd_cpp.ps1
```

### 2. `launch.ps1`
Launches the Stable Diffusion server.
- Automatically points to the `models/` directory in the project root.
- Starts the server on port `1234`.
- The server starts without a pre-loaded model, allowing you to select your desired model directly from the Web UI.

**Usage**:
```powershell
.\launch.ps1
```

## Requirements
- Windows 10 or 11
- PowerShell 5.1 or 7+ (7 is recommended)
- An NVIDIA GPU (for CUDA acceleration)
- Internet connection (for downloading dependencies)

## Notes
- These scripts are designed to be re-run safely. If a dependency is already installed, it will be skipped.
- All build artifacts are stored in the `build/` directory.
- The compiled frontend is served from `build/bin/public`.
