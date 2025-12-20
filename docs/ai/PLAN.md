# Project Plan: stable-diffusion.cpp WebUI

## Status: Saturday, December 20, 2025

### Completed Milestones

1.  **C++ Backend:** `sd-server.exe` successfully compiled and configured to serve static web files from a `public` directory.
2.  **Frontend Setup:** Vue.js project (`webui`) initialized.
3.  **Build System Fixed:** Critical build issue resolved by removing incompatible Tailwind CSS dependency.
4.  **Successful Frontend Build:** Vue.js project compiles successfully, generating static files in `webui/dist/`.
5.  **Project Structure & Build Optimization:**
    *   Integrated `webui` into the server structure and automated the build process via CMake.
    *   Cleaned up "vendor" structure for a clean repository.
6.  **UI Redesign (Admin Panel Layout):**
    *   Implemented a compact, collapsible sidebar layout.
    *   Full support for Dark/Light mode (persisted in browser).
    *   Introduced `vue-router` for clean navigation between "Text-to-Image", "Image-to-Image", "History", and "Settings".
7.  **Image Gallery & Local Storage:**
    *   Optional automatic saving of generated images to a local `outputs` folder.
    *   Implemented "History" view with a modal image gallery (lightbox) and navigation.
    *   Proxy configuration for Vite dev server for seamless gallery development.
8.  **User Experience:**
    *   Automatic persistence of ALL generation parameters in `localStorage` (Prompt, Steps, Seed, Resolution, etc.).
    *   Introduced "Negative Prompt" field.
    *   Optimized default values for fast models like "z-image-turbo".
9.  **API Refactoring & Parameter Management:**
    *   Switched `/v1/images/generations` interface to pure JSON.
    *   Backend: Automatically saves parameters as `.json` files for each image in the `outputs` folder.
    *   Backend: Extended History API to read these JSON metadata files.
    *   Frontend: "Reuse Parameters" button in the gallery (loads all values back into the form).
10. **Image-to-Image (Img2Img) Functionality:**
    *   Backend: Support for `init_image` via Base64-JSON.
    *   Backend: Automatic provision of empty masks/control images to prevent crashes.
    *   Frontend: Dedicated "Image-to-Image" view with upload field, preview, and denoising strength slider.
    *   Optimization: `init_image` is not saved in JSON metadata to save space (replaced by `is_img2img` flag).
11. **Batch Generation:**
    *   Support for generating multiple images at once (`batch_count`).
    *   Frontend adjustment (grid view for results).
12. **Dynamic Model Management & Configuration:**
    *   Backend: Implemented structured model directory scanning (`models/stable-diffusion`, `models/vae`, etc.).
    *   Backend: Added support for JSON sidecar configuration files per model to define associated VAE, text encoders, and performance flags.
    *   Backend: Implemented `/v1/models/load` endpoint for runtime model switching without server restart.
    *   Frontend: Integrated model selector into the sidebar with grouping/filtering.
13. **Performance Optimizations:**
    *   Backend: Added model-specific configuration for `clip_on_cpu`, `offload_to_cpu`, `vae_tiling`, and `flash_attn`.
    *   Build System: Enabled `GGML_CUDA_FA` (Flash Attention) by default for massive speed boosts on modern GPUs.
    *   Resolved VAE decoding crashes for high resolutions using automatic VAE Tiling.
14. **Advanced Workflow Enhancements:**
    *   Implemented "Send to Img2Img" workflow from both the history gallery and the immediate result display.
        *   Added "Scale 2x" shortcut in the Img2Img form for quick high-resolution refinement.
    *   Separated UI loading states (Generating vs. Model Switching) for a smoother user experience.
15. **Dynamic Exploration Grid (3x3 Variations):**
    *   Implemented a 3x3 exploration grid where the center image acts as a promotes-able "Anchor".
    *   Added a dynamic variation engine that varies Seed, Steps, and Guidance based on what is not locked.
    *   Created a condensed parameter sidebar specifically for the exploration view.
    *   Implemented a per-field locking mechanism next to each input.
    *   Integrated `AbortController` support to cancel ongoing requests when shifting focus or refreshing.
    *   Viewport-constrained grid layout with dynamic aspect ratio preservation.
16. **Real-time Progress Streaming (SSE):**
    *   Backend: Implemented Server-Sent Events (SSE) endpoint (`/v1/stream/progress`) for zero-polling progress updates.
    *   Backend: Added explicit phase tracking (Initializing, Sampling, VAE Decoding).
    *   Frontend: Integrated real-time progress bars with phase labels, elapsed time, and rolling-average ETA.
17. **Advanced UI Navigation & Refinements:**
    *   Implemented "Mini-Sidebar" mode: Navigation shrinks to icons with automated tooltip management.
    *   Custom manual dropdown for model selection in collapsed mode (bypasses clipping issues).
    *   Persistent UI state: Sidebar collapse state and gallery preferences saved in `localStorage`.
    *   Removed top navbar to maximize vertical space for full-screen workflows.
    *   Improved result display with non-stretching image containers and integrated metadata cards.
18. **Flexible History Gallery:**
    *   Switched from Bootstrap Grid to custom CSS Grid for precise layout control.
    *   Added header controls with a range slider to choose between 2 and 12 images per row.
19. **Forge / A1111 Compatibility:**
    *   Backend: Switched sidecar parameter saving from `.json` to A1111-compatible `.txt` format.
    *   Backend: Implemented a text metadata parser for the history API (with JSON fallback).
    *   Backend: Added support for Forge-compatible JSON keys (`vae_path`, `clip_l_path`, `ae`, `clip_skip`) in model configurations.
    *   Frontend: Added "Apply Forge Parameters" magic button to instantly populate settings from pasted metadata.
    *   Frontend: Integrated "Forge Format" display and copy-to-clipboard functionality in the gallery and result views.

### Roadmap

#### Phase 1: Architecture & Technical Debt (Current Focus)
1.  **Server Refactoring (Crucial):**
    *   [ ] Split `examples/server/main.cpp` (1600+ lines) into modular components (`server_state`, `model_loader`, `api_endpoints`).
    *   [ ] Create dedicated headers/sources for API utilities and helpers.
    *   See `docs/ai/REFACTORING_PLAN.md` for details.

#### Phase 2: Upscaling & High-Res Fix
1.  **Upscale / Hires. Fix (Native ESRGAN):**
    *   [ ] Backend: Full integration of ESRGAN upscaler models via the dynamic model loading system.
    *   [ ] Frontend: Dedicated "Upscale" button and workflow.
    *   See `docs/ai/UPSCALING_PLAN.md` and `docs/ai/UPSCALING_IMPLEMENTATION.md`.

#### Phase 3: Advanced Features
1.  **Prompt Engineering (In-Browser LLM):**
    *   [ ] Integration of `transformers.js` for local prompt rewriting and enhancement.
    *   [ ] Intelligent "Variations" using small models like Qwen2.5-Instruct or Gemma 3.

2.  **Advanced Model Management:**
    *   [ ] Support for LoRA management and triggering via the UI.
    *   [ ] Model downloader/manager interface.
    