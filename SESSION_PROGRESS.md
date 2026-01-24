# Session Progress: SeedVR2 C++ Integration

**Date:** January 24, 2026
**Status:** Pipeline End-to-End connected, VAE Decode OOM (PixelShuffle Issue).

## 1. Accomplishments

### 1.1 Infrastructure & CLI
*   **Graph Size:** Increased `SEEDVR2_GRAPH_SIZE` to 81920 in `seedvr2.hpp` to prevent graph overflow during DiT construction.
*   **Upscale Factor:** Added `--upscale-factor` (default 4) argument to `sd-cli` to allow flexible testing (e.g., 2x upscale to save VRAM).
*   **Backend Memory:** Replaced `ggml_backend_tensor_get` with `memcpy` in `upscaler.cpp` for host-resident tensors to prevent assertion failures.

### 1.2 DiT Integration
*   **Context Passing:** Fixed `SeedVR2DiTRunner::compute` to correctly pass the `output_ctx`, ensuring the output tensor is allocated in the correct context.
*   **Unpatchify:** Implemented the "unpatchify" logic in `upscaler.cpp` to transform the DiT output `[132, N_tokens]` back into the spatial latent format `[W, H, 1, C]` required by the VAE decoder.

### 1.3 VAE Decoder
*   **PixelShuffle:** Attempted to implement a manual 2D/3D PixelShuffle mechanism in `VAEUpsample3D` using `ggml_reshape` and `ggml_permute` (as `ggml_pixel_shuffle` is unavailable).
*   **Status:** The VAE decode graph builds, but execution fails with a massive OOM error (~60GB request), indicating incorrect tensor dimension handling in the custom PixelShuffle logic.

## 2. Current Challenges
*   **VAE Decode OOM:** The `VAEUpsample3D` implementation is producing tensors with exploded dimensions, leading to an Out-Of-Memory error on the GPU. The manual reshape/permute sequence for PixelShuffle is likely mathematically correct in concept but implementation details (dimensions order) need debugging.

## 3. Next Steps
1.  **Fix PixelShuffle:** detailed review of the `reshape` -> `permute` -> `reshape` chain in `VAEUpsample3D` to ensure it correctly reduces channels while increasing spatial dimensions without creating intermediate massive tensors or wrong shapes.
2.  **Verify Decode:** Once OOM is resolved, verify the decoded image for correctness (visual artifacts).
3.  **Optimize:** Remove excessive logging added during debugging.