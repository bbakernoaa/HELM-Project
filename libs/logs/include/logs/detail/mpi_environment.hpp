#ifndef LOGS_DETAIL_MPI_ENVIRONMENT_HPP
#define LOGS_DETAIL_MPI_ENVIRONMENT_HPP

/// @file detail/mpi_environment.hpp
/// @brief Mpi_Environment — MPI threading level and rank detection.
///
/// Wraps MPI_Comm_rank, MPI_Query_thread, and MPI_Initialized behind a
/// thread-safe interface. The detect() method is the sole mutation point;
/// accessor methods are safe for concurrent reads.
///
/// Requirements: 2.1, 2.7, 11.3, 11.9

#include <atomic>
#include <mutex>

#include <mpi.h>

namespace logs::detail {

// Forward declare so we can friend it.
class Serialized_MPI_Guard;

/// Encapsulates MPI rank detection and thread-level query for the Logger.
///
/// Default state before detect():
///   rank         = -1 (sentinel: unidentified)
///   thread_level = MPI_THREAD_SINGLE (conservative default)
///   communicator = MPI_COMM_NULL
///
/// detect() stores the communicator, queries rank exactly once, and
/// queries the MPI thread level if MPI is initialized.
class Mpi_Environment {
public:
    Mpi_Environment() noexcept;

    /// Detect rank and thread level for the given communicator.
    ///
    /// Behaviour:
    ///   1. Store the communicator.
    ///   2. Call MPI_Comm_rank exactly once; on failure set rank to -1.
    ///   3. Call MPI_Initialized to check MPI state.
    ///   4. If initialized, call MPI_Query_thread to get the thread level.
    ///   5. If not initialized, store MPI_THREAD_SINGLE (conservative default).
    ///
    /// All operations noexcept — exceptions are caught internally.
    void detect(MPI_Comm comm) noexcept;

    /// Thread-safe accessor: returns stored rank (-1 if unconfigured or failed).
    [[nodiscard]] int rank() const noexcept;

    /// Thread-safe accessor: returns stored MPI thread level constant.
    [[nodiscard]] int thread_level() const noexcept;

    /// Returns true iff stored thread level == MPI_THREAD_MULTIPLE.
    [[nodiscard]] bool is_thread_multiple() const noexcept;

    /// Returns true iff detect() has been called with a communicator.
    [[nodiscard]] bool has_communicator() const noexcept;

    /// Returns the stored communicator (MPI_COMM_NULL if unconfigured).
    [[nodiscard]] MPI_Comm communicator() const noexcept;

private:
    std::atomic<int> rank_{-1};
    std::atomic<int> thread_level_{MPI_THREAD_SINGLE};
    MPI_Comm         comm_{MPI_COMM_NULL};
    bool             has_comm_{false};
    mutable std::mutex serialization_mutex_;

    friend class Serialized_MPI_Guard;
};

} // namespace logs::detail

#endif // LOGS_DETAIL_MPI_ENVIRONMENT_HPP
