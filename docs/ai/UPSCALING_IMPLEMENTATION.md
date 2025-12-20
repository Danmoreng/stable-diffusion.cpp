# ESRGAN Upscaling and Highres-fix Implementation

This document describes the implementation of ESRGAN-based upscaling and the Highres-fix feature in the `stable-diffusion.cpp` server and WebUI.

## Backend Changes (C++ Server)

### 1. Data Structures (`examples/common/common.hpp`)
-   Updated `SDGenerationParams` to include Highres-fix configuration:
    -   `hires_fix` (bool): Toggle for the feature.
    -   `hires_upscale_model` (string): Path to the ESRGAN model.
    -   `hires_upscale_factor` (float): Desired upscale multiplier.
    -   `hires_denoising_strength` (float): Strength for the second pass.
    -   `hires_steps` (int): Number of steps for the second pass.
-   Enhanced JSON parsing in `load_if_exists` to robustly handle numeric types.

### 2. API Endpoints (`examples/server/main.cpp`)
-   **`POST /v1/upscale/load`**: Loads or switches the active ESRGAN model.
-   **`POST /v1/images/upscale`**: Standalone endpoint to upscale an image (base64 or existing output).
-   **`GET /v1/models`**: Now detects `.pth` files in `models/esrgan/` and tags them as `esrgan` type.

### 3. Highres-fix Logic
Implemented a chained process in `/v1/images/generations`:
1.  **Base Generation**: Generates the image at the initial requested resolution.
2.  **Upscaling**: Uses the loaded ESRGAN model to upscale the image. If no model is loaded, it falls back to high-quality bicubic resizing.
3.  **Resizing**: Ensures the upscaled image matches the exact `hires_upscale_factor` even if the model's native scale differs.
4.  **Second Pass (Img2Img)**: Runs a denoising pass on the upscaled image to add detail and fix artifacts.
5.  **Robustness**: Automatically resizes `mask_image` and `control_image` to prevent assertion failures during the second pass.

## Frontend Changes (Vue.js WebUI)

### 1. Store (`src/stores/generation.ts`)
-   Added state management for all upscaling and Highres-fix parameters.
-   Implemented persistence using `localStorage`.
-   Added actions for loading upscaler models and triggering standalone upscaling.

### 2. Generation Form (`src/components/GenerationForm.vue`)
-   Added a collapsible Highres-fix section for `txt2img` mode.
-   Includes controls for Upscaler selection, Scale Factor, Denoising Strength, and Steps.
-   Styled with theme-aware colors for consistent appearance in dark and light modes.

### 3. Image Gallery (`src/components/ImageGallery.vue`)
-   Added an "Upscale" button to the image viewer modal.
-   Allows users to post-process any previously generated image using the currently active upscaler.

### 4. Sidebar (`src/components/Sidebar.vue`)
-   Simplified the sidebar by removing the redundant upscaler selector, moving it into the context of generation and gallery actions.

## Dependencies
-   Requires ESRGAN models (e.g., `RealESRGAN_x4plus_anime_6B.pth`) in the `models/esrgan/` directory for full functionality.
