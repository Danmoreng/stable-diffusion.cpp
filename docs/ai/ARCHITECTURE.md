# System Architecture

## Overview
The project consists of a C++ inference server acting as the backend and a Vue.js Single Page Application (SPA) as the frontend.

## 1. Backend: C++ Server (`stable-diffusion.cpp/examples/server`)

*   **Core Library:** Uses `stable-diffusion.cpp` (which depends on `ggml`) for neural network inference.
*   **HTTP Framework:** Uses `cpp-httplib` for handling REST API requests and serving static files.
*   **Concurrency:**
    *   Uses a global mutex (`sd_mutex` roughly, needs verification during refactor) to serialize inference requests (Stable Diffusion is heavy on GPU/CPU).
    *   Uses a separate thread or non-blocking mechanism (SSE) for progress reporting.

### Key Components (Current & Planned Refactor)
*   **`main.cpp`:** Entry point. Parses CLI arguments, initializes the `sd_ctx` (Stable Diffusion Context), and starts the HTTP server.
*   **Model Management:** Scans `models/` directory. Supports hot-swapping models via API. Uses JSON sidecars for configuration (VAE, params).
*   **API:**
    *   Standard `POST /v1/images/generations` (JSON).
    *   `GET /v1/stream/progress` (Server-Sent Events) for real-time feedback.
    *   Model management endpoints.

## 2. Frontend: Vue.js WebUI (`examples/server/webui`)

*   **Framework:** Vue 3 + Vite.
*   **Styling:** Bootstrap 5 (dark/light mode supported).
*   **State Management:** Pinia (implied, or simple reactive state) + `localStorage` for persistence.
*   **Communication:**
    *   Fetch API for command/control.
    *   EventSource for progress streaming.

### Key Features
*   **History Gallery:** Stores generation metadata and images (via proxy/static serving).
*   **Dynamic Exploration:** 3x3 grid for parameter variation.
*   **Model Selector:** Sidebar integration for model switching.

## 3. Directory Structure

*   `C:\StableDiffusion\models`: Central repository for checkpoints (GGUF, Safetensors converted, etc.).
*   `examples/server/public`: The compiled frontend assets are served from here.
*   `output`: Generated images are saved here (configurable).
