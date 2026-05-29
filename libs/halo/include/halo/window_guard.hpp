#ifndef HALO_WINDOW_GUARD_HPP
#define HALO_WINDOW_GUARD_HPP

/// @file window_guard.hpp
/// @brief RAII wrapper for MPI_Win handles.
///
/// The Window_Guard class takes exclusive ownership of an MPI_Win handle and
/// guarantees cleanup via MPI_Win_free on all exit paths (including exceptions).
/// If an access epoch is active at destruction time, MPI_Win_fence(0) is called
/// before MPI_Win_free to close the epoch.
///
/// @note The handle() accessor returns the raw MPI_Win by value without
/// transferring ownership. Callers must ensure the Window_Guard outlives any
/// use of the returned handle.
///
/// @note The destructor never throws. MPI errors during destruction are
/// swallowed to maintain the noexcept guarantee.

#include <mpi.h>

namespace halo {

/// @brief RAII wrapper around an MPI_Win handle.
///
/// Manages the lifetime of an MPI one-sided communication window. On
/// destruction, closes any active epoch via MPI_Win_fence(0) and then frees
/// the window via MPI_Win_free. Move-only semantics enforce unique ownership.
class Window_Guard {
public:
    /// @brief Construct from a raw MPI_Win handle. Takes exclusive ownership.
    /// @param win The MPI window handle to own.
    explicit Window_Guard(MPI_Win win) noexcept;

    /// @brief Destructor. If epoch active, calls MPI_Win_fence(0) first.
    /// Then calls MPI_Win_free if handle is not MPI_WIN_NULL and MPI is not
    /// finalized. Never throws — MPI errors are swallowed.
    ~Window_Guard() noexcept;

    /// @brief Move constructor. Transfers ownership; source becomes MPI_WIN_NULL.
    Window_Guard(Window_Guard&& other) noexcept;

    /// @brief Move assignment. Transfers ownership; source becomes MPI_WIN_NULL.
    Window_Guard& operator=(Window_Guard&& other) noexcept;

    /// @brief Copy construction is deleted (unique ownership).
    Window_Guard(const Window_Guard&) = delete;

    /// @brief Copy assignment is deleted (unique ownership).
    Window_Guard& operator=(const Window_Guard&) = delete;

    /// @brief Access the raw MPI_Win handle without transferring ownership.
    /// @return The owned MPI_Win value.
    /// @note Does not extend the lifetime of this Window_Guard. Callers must
    /// ensure this object outlives any use of the returned handle.
    [[nodiscard]] MPI_Win handle() const noexcept;

    /// @brief Mark whether an access epoch is currently active on this window.
    /// @param active true if an epoch is active, false otherwise.
    ///
    /// When active is true and the Window_Guard is destroyed, MPI_Win_fence(0)
    /// will be called before MPI_Win_free to properly close the epoch.
    void set_epoch_active(bool active) noexcept;

private:
    MPI_Win win_{MPI_WIN_NULL};
    bool epoch_active_{false};
};

} // namespace halo

#endif // HALO_WINDOW_GUARD_HPP
