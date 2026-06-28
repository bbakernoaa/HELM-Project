#ifndef HALO_HALO_HANDLE_HPP
#define HALO_HALO_HANDLE_HPP

/// @file halo_handle.hpp
/// @brief Async halo exchange completion handle.
///
/// Halo_Handle is an RAII object returned by exchange_async() that owns all
/// pending Request_Guard objects associated with a non-blocking halo exchange.
/// It manages completion via test() (non-blocking) or wait() (blocking), and
/// performs post-receive deep-copy from host staging buffers back to the device
/// view when GPU-aware MPI is unavailable.
///
/// Move-only semantics ensure unique ownership of pending MPI operations.
/// The destructor calls wait() to guarantee all operations complete before
/// resources are released.
///
/// @note The handle() accessor on Request_Guard does not extend the lifetime
/// of this Halo_Handle. Callers must ensure this object outlives any external
/// use of its internal requests.

#include <functional>
#include <halo/request_guard.hpp>
#include <memory>
#include <vector>

namespace halo {

// Forward declarations for friend access
class Halo_Plan;
template <int Rank>
class Structured_Halo_Plan;

/// @brief Async halo exchange completion handle.
///
/// Owns all Request_Guard objects for pending sends and receives from an
/// asynchronous halo exchange. Provides test() for non-blocking completion
/// checks and wait() for blocking completion. When host-staged receive buffers
/// exist (non-GPU-aware MPI path), completion triggers a deep-copy from host
/// buffers back to the device Kokkos::View.
///
/// Move-only: copy construction and copy assignment are deleted.
/// The destructor calls wait() to ensure all pending operations complete.
class Halo_Handle {
   public:
    /// @brief Default construct an empty handle (no pending operations).
    Halo_Handle() = default;

    /// @brief Destructor: calls wait() to ensure all pending operations complete
    /// and post-receive deep-copies execute before resources are released.
    ~Halo_Handle();

    /// @brief Move constructor. Transfers ownership of all pending operations.
    /// The source is left in a valid empty state.
    Halo_Handle(Halo_Handle &&other) noexcept;

    /// @brief Move assignment. Waits on current operations (if any), then
    /// transfers ownership from other. The source is left in a valid empty state.
    Halo_Handle &operator=(Halo_Handle &&other) noexcept;

    /// @brief Copy construction is deleted (unique ownership of MPI requests).
    Halo_Handle(const Halo_Handle &) = delete;

    /// @brief Copy assignment is deleted (unique ownership of MPI requests).
    Halo_Handle &operator=(const Halo_Handle &) = delete;

    /// @brief Non-blocking test for completion of all pending operations.
    ///
    /// If all operations have completed, performs post-receive deep-copy
    /// (when staged receive buffers exist) and returns true.
    /// If any operation is still pending, returns false without side effects.
    ///
    /// @return true if all operations are complete; false if any are pending.
    [[nodiscard]] bool test();

    /// @brief Blocking wait for all pending operations to complete.
    ///
    /// Blocks until all Request_Guard objects report completion, then performs
    /// post-receive deep-copy (when staged receive buffers exist).
    /// No-op if the handle is empty.
    void wait();

    /// @brief Check if this handle has no pending operations.
    /// @return true if no Request_Guard objects are held (default-constructed
    ///         or moved-from state).
    [[nodiscard]] bool empty() const noexcept;

   private:
    // exchange_async is a friend so it can populate the handle's internals.
    template <typename ViewType>
    friend Halo_Handle exchange_async(const Halo_Plan &, ViewType &);

    // exchange_structured_async is a friend so it can populate the handle's internals.
    template <typename ViewType>
    friend Halo_Handle exchange_structured_async(const Structured_Halo_Plan<ViewType::rank> &, ViewType &);

    // Test-only friend for Property 18: allows injecting staged_recv state.
    friend class Halo_Handle_Test_Access;

    /// @brief Pending MPI request guards for sends and receives.
    std::vector<Request_Guard> requests_;

    /// @brief Type-erased post-receive deep-copy state.
    ///
    /// When GPU-aware MPI is unavailable and the view resides in device memory,
    /// MPI_Irecv targets host staging buffers. Upon completion, the data must
    /// be deep-copied back to the device view. This struct holds the host
    /// buffers and a type-erased callback to perform the deep-copy.
    struct Staged_Recv {
        /// @brief Type-erased callback that performs the post-receive deep-copy.
        /// Called once when all operations complete (from test() or wait()).
        std::function<void()> post_recv_copy;

        /// @brief Whether the post-receive deep-copy has already been performed.
        bool completed{false};
    };

    /// @brief Staged receive state (null when GPU-aware MPI is available or
    /// when the view resides in host memory).
    std::unique_ptr<Staged_Recv> staged_recv_;

    /// @brief Perform post-receive deep-copy if staged state exists and hasn't
    /// been executed yet.
    void finalize_staged_recv_();
};

}  // namespace halo

#endif  // HALO_HALO_HANDLE_HPP
