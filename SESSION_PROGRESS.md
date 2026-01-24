# Session Progress: SeedVR2 C++ Integration

**Date:** January 24, 2026
**Status:** VAE Decode functional (no OOM), Output is NOISE (Incorrect).

## 1. Accomplishments

### 1.1 Infrastructure & CLI
*   **Graph Size:** Increased `SEEDVR2_GRAPH_SIZE` to 81920 in `seedvr2.hpp` to prevent graph overflow.
*   **Upscale Factor:** Added `--upscale-factor` (default 4) argument.
*   **Backend Memory:** Replaced `ggml_backend_tensor_get` with `memcpy` in `upscaler.cpp` for host-resident tensors.

### 1.2 DiT Integration
*   **Context Passing:** Fixed `SeedVR2DiTRunner::compute` context handling.
*   **Unpatchify:** Implemented `unpatchify` logic in `upscaler.cpp` to transform DiT output tokens back to latent format.

### 1.3 VAE Decoder & Upscaler
*   **Performance:** Optimized `CausalConv3d` to use a loop of 2D convolutions instead of `im2col_3d`. This resolved OOM issues and reduced tile decode time from ~47s to ~2.3s.
*   **Tiling Fix:** Fixed a bug in `upscaler.cpp` where the tiled VAE input view used the wrong channel stride (`nb3` instead of `nb2`).
*   **PixelShuffle (VAEUpsample3D):** Overhauled implementation in `seedvr2.hpp` to match SeedVR2's custom `(x y z c)` channel layout.
    *   Factors are now extracted in `[C_out, Z, Y, X]` order (fastest to slowest).
    *   Upscaling is performed step-wise: Temporal (`Z`), then Spatial Width (`Y`), then Spatial Height (`X`).
*   **Causal Padding:** Implemented "replication padding" in `CausalConv3d` using `std::max(0, ...)` logic on temporal indices.

## 2. Current Challenges
*   **Output is Noise:** Despite fixing the stride and layout logic, the output image contains no recognizable content (just noise). This suggests a fundamental mismatch in how data is flowing through the VAE or DiT, or how the weights are being applied.
*   **Weight Application:** The model weights are 3D. The current C++ implementation sums convolutions over all temporal weight slices. If the reference implementation (ComfyUI/PyTorch) handles these weights differently (e.g., slicing for static images vs. video), this would cause total corruption of the signal.

## 3. Next Steps
1.  **Debug Weights:** Verify if the provided model weights (`seedvr2_ema_3b_fp16.safetensors`) require specific slice selection for 2D-like inference.
2.  **Trace Values:** Dump intermediate tensors from C++ (post-DiT, pre-VAE, post-VAE) and compare them numerically with the Python reference dumps to pinpoint exactly where the signal becomes noise.
3.  **Check Normalization:** Verify if the noise is due to massive scaling issues (float range vs uint8) or data corruption.

## 4. Debugging Session: VAE Loopback (Current Focus)

**Strategy:** Isolate the VAE Decoder by bypassing the DiT entirely ("VAE Loopback").
*   **Action:** Modified `upscaler.cpp` to feed the encoded latents directly back into the VAE decoder.
*   **Result:** The output is still noise. This rules out the DiT and confirms the issue is in **Preprocessing -> VAE Encode** OR **VAE Decode -> Postprocessing**.
*   **Active Refactoring:** `seedvr2.hpp` has been heavily modified to attempt correct `PixelShuffle` logic for 4D/5D dimension mapping (mapping `W, H, T, Factors` to GGML's 4D limits). This logic is complex and currently the suspect for the scrambled output.
