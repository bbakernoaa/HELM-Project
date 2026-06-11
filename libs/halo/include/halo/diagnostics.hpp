#ifndef HALO_DIAGNOSTICS_HPP
#define HALO_DIAGNOSTICS_HPP

/// @file diagnostics.hpp
/// @brief Optional instrumentation hook for halo exchange operations.
///
/// Provides a callback-based diagnostics interface that allows external tools
/// to observe exchange events (begin, send_posted, recv_complete, end) without
/// modifying HALO source code.
///
/// When no callback is registered, emit() is a no-op: zero overhead — no
/// timing calls, no allocation, no virtual dispatch.
///
/// Thread safety: set_callback / clear_callback are protected by a mutex.
/// emit() reads the callback under lock to ensure safe concurrent access.
///
/// Tier 1 isolation: This header has NO dependency on LOGS or any other
/// HELM component.

#include <chrono>
#include <cstddef>
#include <functional>
#include <mutex>

namespace halo {

/// @brief Describes a single instrumentation event during a halo exchange.
struct Exchange_Event {
    /// @brief The phase of the exchange operation.
    enum class Phase {
        begin,          ///< Exchange function entered.
        send_posted,    ///< All MPI_Isend calls have been posted.
        recv_complete,  ///< All receives have completed (after MPI_Waitall).
        end             ///< Exchange function returning.
    };

    Phase phase;                          ///< Current event phase.
    int local_rank;                       ///< MPI rank of the calling process.
    int neighbor_count;                   ///< Number of active neighbors in this exchange.
    std::size_t total_bytes;              ///< Total bytes involved in the exchange.
    std::chrono::nanoseconds elapsed;     ///< Time since exchange begin (0 for begin events).
    bool is_async;                        ///< True if this is an async exchange.
};

/// @brief Callback type for diagnostics hooks.
using Diagnostics_Callback = std::function<void(const Exchange_Event&)>;

/// @brief Static diagnostics hook manager.
///
/// Provides set_callback / clear_callback / emit as static thread-safe methods.
/// When no callback is set, emit() short-circuits with minimal overhead (a
/// single null check under lock-free read of a boolean flag).
class Diagnostics {
public:
    /// @brief Register a diagnostics callback.
    ///
    /// The callback will be invoked for every exchange event emitted by
    /// exchange_blocking, exchange_async, and structured exchange functions.
    ///
    /// @param cb The callback to register. Must not be null.
    static void set_callback(Diagnostics_Callback cb) {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = std::move(cb);
        active_ = true;
    }

    /// @brief Remove the currently registered callback.
    ///
    /// After this call, emit() becomes a no-op.
    static void clear_callback() {
        std::lock_guard<std::mutex> lock(mutex_);
        callback_ = nullptr;
        active_ = false;
    }

    /// @brief Emit a diagnostics event.
    ///
    /// If a callback is registered, invokes it with the given event.
    /// If no callback is registered, this is a no-op (checks a boolean flag
    /// without acquiring the mutex for the fast path).
    ///
    /// @param event The exchange event to emit.
    static void emit(const Exchange_Event& event) {
        // Fast path: no callback registered — zero overhead.
        if (!active_) {
            return;
        }

        // Slow path: callback is registered, acquire lock and invoke.
        Diagnostics_Callback cb;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cb = callback_;
        }
        if (cb) {
            cb(event);
        }
    }

    /// @brief Returns true if a callback is currently registered.
    [[nodiscard]] static bool is_active() noexcept {
        return active_;
    }

private:
    static inline std::mutex mutex_;
    static inline Diagnostics_Callback callback_{nullptr};
    static inline bool active_{false};

    Diagnostics() = delete;
};

} // namespace halo

#endif // HALO_DIAGNOSTICS_HPP
