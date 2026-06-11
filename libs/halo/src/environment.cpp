#include "halo/environment.hpp"

#include <mpi.h>
#include <stdexcept>

#include <halo/detail/gpu_aware_probe.hpp>

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

        // Probe for GPU-aware MPI support and cache the result.
        gpu_aware_mpi_ = detail::gpu_aware_probe();
    });
}

// ─── Thread-Level Accessors ──────────────────────────────────────────────────

int Environment::thread_support_level() noexcept {
    return thread_level_;
}

bool Environment::is_thread_multiple() noexcept {
    return thread_level_ == MPI_THREAD_MULTIPLE;
}

bool Environment::is_gpu_aware_mpi() noexcept {
    return gpu_aware_mpi_;
}

// ─── Error Policy ────────────────────────────────────────────────────────────

void Environment::set_error_policy(ErrorPolicy policy) noexcept {
    error_policy_.store(policy, std::memory_order_release);
}

ErrorPolicy Environment::error_policy() noexcept {
    return error_policy_.load(std::memory_order_acquire);
}

} // namespace halo
