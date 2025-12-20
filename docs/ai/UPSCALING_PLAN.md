# Plan: Implementing Upscaling Support

This document outlines the strategy for adding ESRGAN-based upscaling support to the `stable-diffusion.cpp` server and WebUI.

## 1. Backend Implementation (C++ Server)

### Architecture Changes
- **Global State:** Introduce `upscaler_ctx_t* upscaler_ctx` and `std::string current_upscale_model_path` in `examples/server/main.cpp`.
- **Concurrency:** Ensure upscaling operations are protected by the existing `sd_mutex` to prevent simultaneous use of GPU resources with generation tasks.

### New API Endpoints
1.  **POST `/v1/upscale/load`**
    - **Purpose:** Load or switch the ESRGAN model.
    - **Payload:** `{ "id": "esrgan/model_name.pth" }`
    - **Logic:** 
        - If an upscaler is already loaded, free it using `free_upscaler_ctx`.
        - Resolve the path relative to the `models/` directory.
        - Initialize a new context using `new_upscaler_ctx`.
2.  **POST `/v1/images/upscale`**
    - **Purpose:** Upscale an existing image.
    - **Payload:** 
        - `image`: Base64 encoded string or path to an existing result.
        - `upscale_factor`: (Optional) Default to the model's native factor (usually 4).
    - **Logic:**
        - Decode the image into `sd_image_t`.
        - Call `upscale(upscaler_ctx, input_image, factor)`.
        - Save the resulting image to the `outputs/` directory with a `_upscaled` suffix.
        - Return metadata and the URL of the upscaled image.

### Model Management
- Update **GET `/v1/models`** to explicitly include models found in the `models/esrgan/` directory with a specific `type: "esrgan"` tag for the frontend to filter.

## 2. Frontend Implementation (Vue.js WebUI)

### Component Updates
- **ImageGallery.vue:** 
    - Add an "Upscale" button to the actions overlay for generated images.
    - Show a loading spinner during the upscaling process.
    - Automatically add the upscaled version to the gallery or replace the current view.
- **Settings/Model Selector:**
    - Add a dropdown to select and load the active Upscale model.

### Store Integration
- **generation.ts:** Add an `upscaleImage(imageUrl, modelId)` action to handle the API communication.

## 3. Verification & Build
1.  **C++ Build:** Run `.\install_sd_cpp.ps1` to recompile the server with the new endpoints.
2.  **Frontend Build:** The build script will automatically bundle the updated WebUI and copy it to `build/bin/public`.
3.  **Functional Test:** 
    - Verify model listing shows ESRGAN models.
    - Verify model loading works.
    - Verify a generated image can be upscaled and the result is saved/displayed.

## 4. Dependencies
- Requires an ESRGAN model (e.g., `RealESRGAN_x4plus_anime_6B.pth`) in the `models/esrgan/` directory.
