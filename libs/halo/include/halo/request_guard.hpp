#ifndef HALO_REQUEST_GUARD_HPP
#define HALO_REQUEST_GUARD_HPP

/// @file request_guard.hpp
/// @brief RAII wrapper for MPI_Request handles.
///
/// The Request_Guard class takes exclusive ownership of an MPI_Request handle
/// and guarantees that the associated non-blocking operation is either completed
/// (normal destruction) or cancelled (destruction during stack unwinding).
///
/// @note The handle() accessor returns a non-const pointer to the internal
/// MPI_Request without transferring ownership. Callers must ensure the
/// Request_Guard outlives any use of the returned pointer.

#include <mpi.h>

namespace halo {

/// @brief RAII wrapper around an MPI_Request handle.
///
/// Manages the lifetime of a non-blocking MPI operation. On normal scope exit,
/// the destructor calls MPI_Wait to complete the operation. During stack
/// unwinding (exception propagation), the destructor calls MPI_Cancel followed
/// by MPI_Request_free to safely abort the operation.
///
/// Move-only semantics enforce unique ownership. The source of a move is set
/// to MPI_REQUEST_NULL.
class Request_Guard {
public:
    /// @brief Construct from a raw MPI_Request handle. Takes ownership.
    /// @param req Reference to the MPI_Request. The source is set to
    ///            MPI_REQUEST_NULL after ownership transfer.
    explicit Request_Guard(MPI_Request& req) noexcept;

    /// @brief Default construct (empty, holds MPI_REQUEST_NULL).
    Request_Guard() noexcept = default;

    /// @brief Destructor.
    ///   - Normal exit: calls MPI_Wait to complete the pending operation.
    ///   - Stack unwinding: calls MPI_Cancel + MPI_Request_free to abort.
    ///   - MPI_REQUEST_NULL: no-op.
    ~Request_Guard();

    /// @brief Move constructor. Transfers ownership; source becomes MPI_REQUEST_NULL.
    Request_Guard(Request_Guard&& other) noexcept;

    /// @brief Move assignment. Transfers ownership; source becomes MPI_REQUEST_NULL.
    Request_Guard& operator=(Request_Guard&& other) noexcept;

    /// @brief Copy construction is deleted (unique ownership).
    Request_Guard(const Request_Guard&) = delete;

    /// @brief Copy assignment is deleted (unique ownership).
    Request_Guard& operator=(const Request_Guard&) = delete;

    /// @brief Non-blocking test for operation completion.
    /// @return true if the operation has completed or if the handle is
    ///         MPI_REQUEST_NULL; false if still pending.
    [[nodiscard]] bool test();

    /// @brief Blocking wait for operation completion.
    /// No-op if the handle is MPI_REQUEST_NULL.
    void wait();

    /// @brief Access the raw MPI_Request pointer without transferring ownership.
    /// @return Pointer to the internal MPI_Request. Returns a pointer to a
    ///         MPI_REQUEST_NULL value if moved-from or default-constructed.
    /// @note Does not extend the lifetime of this Request_Guard. Callers must
    /// ensure this object outlives any use of the returned pointer.
    [[nodiscard]] MPI_Request* handle() noexcept;

private:
    MPI_Request req_{MPI_REQUEST_NULL};
    int uncaught_on_entry_{0};  ///< Snapshot of std::uncaught_exceptions() at construction.
};

} // namespace halo

#endif // HALO_REQUEST_GUARD_HPP
