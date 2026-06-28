#include "halo/window_guard.hpp"

#include <mpi.h>

#include <utility>

namespace halo {

// ─── Construction ────────────────────────────────────────────────────────────

Window_Guard::Window_Guard(MPI_Win win) noexcept : win_{win} {}

// ─── Destruction ─────────────────────────────────────────────────────────────

Window_Guard::~Window_Guard() noexcept {
    if (win_ == MPI_WIN_NULL) {
        return;
    }

    // Do not call MPI_Win_free if MPI has already been finalized.
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return;
    }

    // If an access epoch is active, close it with MPI_Win_fence(0) first.
    if (epoch_active_) {
        // Swallow any error from fence — destructor must not throw.
        MPI_Win_fence(0, win_);
    }

    // Free the window. Swallow any error — destructor must not throw.
    MPI_Win_free(&win_);
}

// ─── Move Semantics ──────────────────────────────────────────────────────────

Window_Guard::Window_Guard(Window_Guard &&other) noexcept
    : win_{std::exchange(other.win_, MPI_WIN_NULL)}, epoch_active_{std::exchange(other.epoch_active_, false)} {}

Window_Guard &Window_Guard::operator=(Window_Guard &&other) noexcept {
    if (this != &other) {
        // Clean up current handle (same logic as destructor, errors swallowed)
        if (win_ != MPI_WIN_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                if (epoch_active_) {
                    MPI_Win_fence(0, win_);
                }
                MPI_Win_free(&win_);
            }
        }
        win_ = std::exchange(other.win_, MPI_WIN_NULL);
        epoch_active_ = std::exchange(other.epoch_active_, false);
    }
    return *this;
}

// ─── Handle Accessor ─────────────────────────────────────────────────────────

MPI_Win Window_Guard::handle() const noexcept {
    return win_;
}

// ─── Epoch State ─────────────────────────────────────────────────────────────

void Window_Guard::set_epoch_active(bool active) noexcept {
    epoch_active_ = active;
}

}  // namespace halo
