#include "server_state.hpp"

ProgressState progress_state;

void on_progress(int step, int steps, float time, void* data) {
    ProgressState* state = (ProgressState*)data;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->step = step;
        state->steps = steps;
        state->time = time;
        state->version++;
    }
    state->cv.notify_all();
}

void reset_progress() {
    {
        std::lock_guard<std::mutex> lock(progress_state.mutex);
        progress_state.step = 0;
        progress_state.steps = 0;
        progress_state.phase = "";
        progress_state.version++;
    }
    progress_state.cv.notify_all();
}

void set_progress_phase(const std::string& phase) {
    {
        std::lock_guard<std::mutex> lock(progress_state.mutex);
        progress_state.phase = phase;
        progress_state.step = 0;
        progress_state.steps = 0;
        progress_state.version++;
    }
    progress_state.cv.notify_all();
}
