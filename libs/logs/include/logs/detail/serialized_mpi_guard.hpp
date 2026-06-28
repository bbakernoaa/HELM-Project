#ifndef LOGS_DETAIL_SERIALIZED_MPI_GUARD_HPP
#define LOGS_DETAIL_SERIALIZED_MPI_GUARD_HPP

/// @file detail/serialized_mpi_guard.hpp
/// @brief Serialized_MPI_Guard — RAII conditional MPI serialization.
///
/// When the detected MPI thread level is below MPI_THREAD_MULTIPLE, all MPI
/// calls issued by LOGS are serialized through the environment's mutex to avoid
/// undefined behavior. When MPI_THREAD_MULTIPLE is available, no locking is
/// needed because MPI itself is fully thread-safe.

#include "logs/detail/mpi_environment.hpp"

namespace logs::detail {

/// RAII guard that locks the environment serialization mutex when the detected
/// MPI thread level is below MPI_THREAD_MULTIPLE (Requirements 11.4, 11.9).
/// Non-copyable and non-movable.
class Serialized_MPI_Guard {
   public:
    /// Construct the guard. If the environment's thread level is below
    /// MPI_THREAD_MULTIPLE, immediately locks the serialization mutex.
    explicit Serialized_MPI_Guard(Mpi_Environment &env) noexcept : env_(env) {
        if (!env_.is_thread_multiple()) {
            env_.serialization_mutex_.lock();
            locked_ = true;
        }
    }

    /// Destroy the guard. Unlocks the serialization mutex if it was locked.
    ~Serialized_MPI_Guard() {
        if (locked_) {
            env_.serialization_mutex_.unlock();
        }
    }

    Serialized_MPI_Guard(const Serialized_MPI_Guard &) = delete;
    Serialized_MPI_Guard &operator=(const Serialized_MPI_Guard &) = delete;
    Serialized_MPI_Guard(Serialized_MPI_Guard &&) = delete;
    Serialized_MPI_Guard &operator=(Serialized_MPI_Guard &&) = delete;

   private:
    Mpi_Environment &env_;
    bool locked_{false};
};

}  // namespace logs::detail

#endif  // LOGS_DETAIL_SERIALIZED_MPI_GUARD_HPP
