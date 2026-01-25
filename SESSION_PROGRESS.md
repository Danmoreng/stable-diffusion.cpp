# Session Progress: SeedVR2 C++ Integration

**Date:** January 25, 2026
**Status:** VAE Loopback Successful (Encode -> Decode verified). DiT pending.

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

### 1.4 VAE Correction (Critical Fixes)
*   **GroupNorm:** Corrected `SeedVR2GroupNorm` to perform per-frame normalization (spatial only) by treating `T` as the batch dimension in `ggml_group_norm`.
*   **Upsample Logic:** Implemented correct `remove_head` slicing logic in `VAEUpsample3D` for all temporal upscaling cases, dropping the second frame (index 1) to match "Tail" inflation mode.
*   **Attention:** Fixed `ggml_permute` arguments in `VAEAttnBlock` to use correct scatter semantics (`1, 2, 3, 0` for input, `3, 0, 1, 2` for output).

## 2. Current Challenges
*   **DiT 3D RoPE:** The DiT currently lacks a functional 3D Rotary Positional Embedding (RoPE) implementation. The current `SeedVR2RoPE` is a placeholder. Enabling DiT without this will likely result in geometric incoherence or noise.
*   **Weight Application:** The model weights are 3D. The current C++ implementation sums convolutions over all temporal weight slices. If the reference implementation (ComfyUI/PyTorch) handles these weights differently (e.g., slicing for static images vs. video), this would cause total corruption of the signal.

## 3. Next Steps
1.  **Enable DiT:** Uncomment the DiT computation in `upscaler.cpp` and verify if the signal passes through (even if geometrically distorted).
2.  **Implement RoPE:** Port the 3D RoPE logic from Python to C++, handling spatial and temporal frequency components correctly.
3.  **Debug Weights:** Verify if the provided model weights (`seedvr2_ema_3b_fp16.safetensors`) require specific slice selection for 2D-like inference.
4.  **Trace Values:** Dump intermediate tensors from C++ (post-DiT, pre-VAE, post-VAE) and compare them numerically with the Python reference dumps to pinpoint exactly where the signal becomes noise.
