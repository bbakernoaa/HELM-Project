#ifndef HALO_ENVIRONMENT_HPP
#define HALO_ENVIRONMENT_HPP

/// @file environment.hpp
/// @brief Singleton initialization and MPI thread-level query for HALO.
///
/// The Environment class queries the MPI thread support level at initialization
/// and provides thread-safe accessors. It also exposes an internal serialization
/// mutex used by detail::Serialized_MPI_Guard to protect MPI calls when the
/// thread level is below MPI_THREAD_MULTIPLE.
///
/// @note Environment::initialize() must be called after MPI_Init_thread.
/// Subsequent calls are no-ops (idempotent via std::once_flag).

#include <mpi.h>
#include <atomic>
#include <mutex>

#include <halo/error_policy.hpp>

namespace halo {

// Forward declaration for friend access.
namespace detail { class Serialized_MPI_Guard; }

/// @brief Singleton class managing HALO initialization and MPI thread-level state.
///
/// Call Environment::initialize() once after MPI_Init_thread. The class queries
/// MPI_Query_thread and stores the result. All accessors are thread-safe.
class Environment {
public:
    /// @brief Initialize HALO. Must be called after MPI_Init_thread.
    ///
    /// Queries MPI_Query_thread and stores the detected thread support level.
    /// Second and subsequent calls are no-ops (idempotent).
    ///
    /// @throws std::runtime_error if MPI has not been initialized.
    static void initialize();

    /// @brief Returns the MPI thread support level detected at initialization.
    /// @return One of MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED,
    ///         MPI_THREAD_SERIALIZED, or MPI_THREAD_MULTIPLE.
    ///         Returns -1 if initialize() has not been called.
    [[nodiscard]] static int thread_support_level() noexcept;

    /// @brief Returns true if the detected thread level is MPI_THREAD_MULTIPLE.
    /// @return true if concurrent MPI calls are safe without serialization.
    [[nodiscard]] static bool is_thread_multiple() noexcept;

    /// @brief Returns true if the MPI implementation supports GPU-aware
    ///        (device pointer) communication at runtime.
    ///
    /// The value is determined during initialize() via detail::gpu_aware_probe()
    /// and cached for the lifetime of the process.
    ///
    /// @return true if GPU-aware MPI is available.
    [[nodiscard]] static bool is_gpu_aware_mpi() noexcept;

    /// @brief Set the active error policy for MPI error handling.
    ///
    /// The policy determines whether HALO throws std::runtime_error (default)
    /// or writes diagnostics to stderr and calls MPI_Abort on MPI failures.
    ///
    /// @param policy The error policy to activate.
    static void set_error_policy(ErrorPolicy policy) noexcept;

    /// @brief Returns the active error policy.
    /// @return The current ErrorPolicy (throw_on_error by default).
    [[nodiscard]] static ErrorPolicy error_policy() noexcept;

    // Non-copyable, non-movable singleton.
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

private:
    Environment() = default;

    static inline std::once_flag init_flag_;
    static inline int thread_level_{-1};
    static inline bool gpu_aware_mpi_{false};
    static inline std::atomic<ErrorPolicy> error_policy_{ErrorPolicy::throw_on_error};
    static inline std::mutex serialization_mutex_;

    friend class detail::Serialized_MPI_Guard;
};

namespace detail {

/// @brief RAII guard that serializes MPI calls when thread level < MPI_THREAD_MULTIPLE.
///
/// When the detected MPI thread support level is below MPI_THREAD_MULTIPLE,
/// this guard locks the internal serialization mutex on construction and
/// unlocks it on destruction. When MPI_THREAD_MULTIPLE is available, the
/// guard is a no-op (no lock acquired).
///
/// Usage:
/// @code
///   {
///       halo::detail::Serialized_MPI_Guard guard;
///       // MPI calls here are serialized if needed
///       MPI_Isend(...);
///   }
/// @endcode
class Serialized_MPI_Guard {
public:
    /// @brief Construct the guard. Locks the serialization mutex if thread
    ///        level < MPI_THREAD_MULTIPLE.
    Serialized_MPI_Guard() {
        if (!Environment::is_thread_multiple()) {
            Environment::serialization_mutex_.lock();
            locked_ = true;
        }
    }

    /// @brief Destructor. Unlocks the serialization mutex if it was locked.
    ~Serialized_MPI_Guard() {
        if (locked_) {
            Environment::serialization_mutex_.unlock();
        }
    }

    // Non-copyable, non-movable.
    Serialized_MPI_Guard(const Serialized_MPI_Guard&) = delete;
    Serialized_MPI_Guard& operator=(const Serialized_MPI_Guard&) = delete;

private:
    bool locked_{false};
};

} // namespace detail
} // namespace halo

#endif // HALO_ENVIRONMENT_HPP
