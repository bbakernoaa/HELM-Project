#include "halo/environment.hpp"

#include <mpi.h>
#include <stdexcept>

namespace halo {

// ─── Initialization ──────────────────────────────────────────────────────────

void Environment::initialize() {
    std::call_once(init_flag_, []() {
        // Verify MPI has been initialized.
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error(
                "halo::Environment::initialize(): MPI must be initialized "
                "before HALO. Call MPI_Init_thread before halo::Environment::initialize().");
        }

        // Query the thread support level provided by the MPI implementation.
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        thread_level_ = provided;
    });
}

// ─── Thread-Level Accessors ──────────────────────────────────────────────────

int Environment::thread_support_level() noexcept {
    return thread_level_;
}

bool Environment::is_thread_multiple() noexcept {
    return thread_level_ == MPI_THREAD_MULTIPLE;
}

} // namespace halo
