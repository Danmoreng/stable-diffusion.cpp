# Agent Instructions (Gemini CLI)

This document contains general instructions for the Gemini agent in this project.

## Environment & Language
*   **Operating System:** Windows (win32)
*   **Shell:** PowerShell
*   **Language:** All communication, documentation (PLAN.md), and code comments should be in **English**.

## Project Architecture
*   **Backend:** C++ server located in `stable-diffusion.cpp/examples/server/main.cpp`. It uses `httplib` for the API and `stable-diffusion.h` for generation.
*   **Frontend:** Vue.js (Vite) application located in `stable-diffusion.cpp/examples/server/webui`.
*   **Static Files:** The backend serves the compiled frontend from `stable-diffusion.cpp/build/bin/public`.

## Model Management
*   **Directory:** All models are stored in `C:\StableDiffusion\models`.
*   **Structure:** Models are categorized into subdirectories: `stable-diffusion/`, `vae/`, `lora/`, `text-encoder/`, `esrgan/`.
*   **Configurations:** Every main model in `stable-diffusion/` should have a matching `.json` sidecar file (e.g., `model.gguf.json`) to define its associated VAE, encoders, and performance flags (`clip_on_cpu`, `flash_attn`, `vae_tiling`, `offload_to_cpu`).

## Build & Run Procedures
*   **Full Build:** Use `.\install_sd_cpp.ps1` to build both the frontend and the backend.
*   **Launch:** Use `.\launch.ps1` to start the server. It automatically points to the `models/` directory.
*   **Code Changes:** Any changes to the C++ backend or the library require a re-run of the build script.

## Constraints & Rules
*   **NPM Commands:** Due to PowerShell execution policies, `npm` commands (like `npm install` or `npm run build`) cannot be executed directly by the agent. 
    *   **Rule:** The human user must execute all `npm` commands manually. The agent should instruct the user on which commands to run and wait for confirmation/output.
*   **Paths:** Always use absolute paths when possible or resolve them relative to the project root `C:\StableDiffusion`.
