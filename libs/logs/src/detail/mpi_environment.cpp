/// @file detail/mpi_environment.cpp
/// @brief Mpi_Environment — thread-level + rank detection implementation.
///
/// Requirements: 2.1, 2.7, 11.3, 11.9
///
/// detect() performs the following steps in order:
///   1. Store the communicator and mark it as configured.
///   2. Call MPI_Comm_rank exactly once; on failure set rank to -1.
///   3. Call MPI_Initialized to check if MPI is active.
///   4. If initialized, call MPI_Query_thread to obtain the thread level.
///   5. If not initialized, store MPI_THREAD_SINGLE as the conservative default.
///
/// All operations are noexcept — any exception from MPI wrappers is caught
/// and the failure is absorbed (rank set to -1, thread level stays conservative).

#include "logs/detail/mpi_environment.hpp"

#include <mpi.h>

namespace logs::detail {

// Default constructor: rank=-1, thread_level=MPI_THREAD_SINGLE, no communicator.
Mpi_Environment::Mpi_Environment() noexcept = default;

void Mpi_Environment::detect(MPI_Comm comm) noexcept {
    try {
        // Step 1: Store the communicator.
        comm_ = comm;
        has_comm_ = true;

        // Step 2: Call MPI_Comm_rank exactly once; on failure set rank to -1.
        int r = -1;
        int rc = MPI_Comm_rank(comm, &r);
        if (rc == MPI_SUCCESS) {
            rank_.store(r, std::memory_order_release);
        } else {
            rank_.store(-1, std::memory_order_release);
        }

        // Step 3: Check if MPI is initialized.
        int initialized = 0;
        MPI_Initialized(&initialized);

        // Step 4/5: Query thread level if initialized; conservative default otherwise.
        if (initialized) {
            int level = MPI_THREAD_SINGLE;
            if (MPI_Query_thread(&level) == MPI_SUCCESS) {
                thread_level_.store(level, std::memory_order_release);
            }
            // If MPI_Query_thread fails, keep the conservative default.
        }
        // If not initialized, thread_level_ remains MPI_THREAD_SINGLE (default).

    } catch (...) {
        // Absorb any exception; ensure rank is sentinel on failure.
        rank_.store(-1, std::memory_order_release);
    }
}

int Mpi_Environment::rank() const noexcept {
    return rank_.load(std::memory_order_acquire);
}

int Mpi_Environment::thread_level() const noexcept {
    return thread_level_.load(std::memory_order_acquire);
}

bool Mpi_Environment::is_thread_multiple() const noexcept {
    return thread_level_.load(std::memory_order_acquire) == MPI_THREAD_MULTIPLE;
}

bool Mpi_Environment::has_communicator() const noexcept {
    return has_comm_;
}

MPI_Comm Mpi_Environment::communicator() const noexcept {
    return comm_;
}

} // namespace logs::detail
