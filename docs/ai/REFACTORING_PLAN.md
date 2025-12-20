# Refactoring Plan: C++ Server Modularization

**Objective:** Split the monolithic `examples/server/main.cpp` (1600+ lines) into manageable, logical modules to improve maintainability, testability, and readability.

## 1. Module Structure

We will create a new directory structure under `examples/server/` (or use existing conventions) to house these modules.

### Proposed Modules

1.  **`server_state.hpp` / `.cpp`**
    *   **Responsibility:** Manages global state that needs to be shared across threads/handlers (e.g., `ProgressState`, `SDContextParams`, current model path).
    *   **Contents:** `ProgressState` struct, `on_progress` callback, global mutexes (if any).

2.  **`model_loader.hpp` / `.cpp`**
    *   **Responsibility:** Handling model discovery, configuration loading, and initialization.
    *   **Contents:** `load_model_config` (JSON sidecar), scanning directories (`get_file_list` logic if applicable), initializing the `sd_ctx`.

3.  **`api_utils.hpp` / `.cpp`**
    *   **Responsibility:** Stateless helper functions for API operations.
    *   **Contents:** Base64 encoding/decoding, `get_image_params` (string builder), `parse_image_params` (A1111 format parser), timestamp generators.

4.  **`api_endpoints.hpp` / `.cpp`**
    *   **Responsibility:** HTTP Request Handlers.
    *   **Contents:** The lambda functions currently passed to `httplib::Server`. These can be refactored into a class `SDServer` or free functions like `handle_generate_image`, `handle_model_list`.

5.  **`main.cpp` (Cleaned)**
    *   **Responsibility:** Entry point, argument parsing, server startup.
    *   **Contents:** `main()` function, `parse_args` (using the existing `argparse` logic), instantiation of `httplib::Server`, binding endpoints from `api_endpoints`.

## 2. Execution Strategy

1.  **Phase 1: Extraction of Utils**
    *   Move `base64`, `iso_timestamp`, `parse_image_params` to `api_utils`.
    *   Verify build.

2.  **Phase 2: Extraction of State & Model Logic**
    *   Move `ProgressState` and `load_model_config` to respective files.
    *   Verify build.

3.  **Phase 3: Endpoint Separation**
    *   This is the hardest part due to closure captures (lambdas capturing local variables).
    *   Create a `ServerContext` struct that holds everything the endpoints need (ptr to `sd_ctx`, params, etc.).
    *   Refactor lambdas to take `ServerContext` or be methods of a `SDServer` class.
    *   Move handlers to `api_endpoints`.

4.  **Phase 4: Cleanup**
    *   Clean up `main.cpp` includes.
    *   Ensure all headers have `#pragma once`.

## 3. Benefits

*   **Faster Navigation:** No more scrolling through 1.6k lines.
*   **Separation of Concerns:** Model logic is distinct from HTTP handling.
*   **Easier Testing:** Utils can be unit-tested without bringing in the whole server.
