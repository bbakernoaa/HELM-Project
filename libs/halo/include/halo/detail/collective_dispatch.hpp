#ifndef HALO_DETAIL_COLLECTIVE_DISPATCH_HPP
#define HALO_DETAIL_COLLECTIVE_DISPATCH_HPP

/// @file halo/detail/collective_dispatch.hpp
/// @brief Compile-time host-direct / device-direct / host-staged path selection
///        for collective operations.
///
/// This detail utility centralizes the compile-time decision of HOW a collective
/// hands its send/receive memory to MPI. The decision is driven entirely by
/// `detail::requires_staging_v<ViewType>` (which already folds in the
/// `HALO_GPU_AWARE_MPI` compile-time flag — see memory_traits.hpp) via
/// `if constexpr`. There is NO per-element runtime branch on memory space.
///
///   requires_staging_v == false  (host view, OR device view + GPU-aware MPI)
///       -> pass `view.data()` directly to MPI (host-direct / device-direct).
///          (Requirements 6.1, 6.2)
///
///   requires_staging_v == true   (device view, no GPU-aware MPI)
///       -> `detail::pack` any required on-device strided source reorder into a
///          contiguous device buffer (Requirement 6.4), `detail::stage_send` it
///          to a host mirror, run the collective on host memory, then
///          `detail::stage_recv` the received data back to the device
///          destination in `finalize` (Requirement 6.3).
///
/// Consumer contract (used by gather_replicated, task 6.1):
///   1. Call `prepare(src, dest, send_count, recv_count)`. This returns a
///      `Dispatch_Buffers` value that owns any host-mirror(s) required to keep
///      the send/receive memory alive for the duration of the collective, and
///      exposes:
///        - `send_ptr()` : the pointer to hand to MPI as the send buffer
///                         (host mirror on the staged path, `src.data()`
///                         otherwise). `const T*`.
///        - `recv_ptr()` : the pointer to hand to MPI as the receive buffer
///                         (host mirror on the staged path, `dest.data()`
///                         otherwise). `T*`.
///   2. Run the MPI collective against `send_ptr()` / `recv_ptr()`.
///   3. Call `finalize(dispatch_buffers, dest, recv_count)`. On the host-staged
///      path this stages the host receive buffer back into the device `dest`
///      view; on the direct paths it is a no-op.
///
/// The returned `Dispatch_Buffers` MUST outlive the MPI collective (its owned
/// host mirrors back the pointers). Keep it in scope until after `finalize`.
///
/// @see halo/detail/memory_traits.hpp for requires_staging_v / host_mirror_t.
/// @see halo/detail/staging.hpp for stage_send / stage_recv.
/// @see halo/detail/pack_unpack.hpp for pack / unpack (on-device reorder).

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/staging.hpp>

namespace halo::detail {

/// @brief Holds the send/receive pointers handed to MPI plus any owned host
///        mirror(s) that must stay alive for the duration of the collective.
///
/// On the host-direct / device-direct path (`Staged == false`) the mirror
/// members are empty and the pointers alias the source/destination views. On
/// the host-staged path (`Staged == true`) the mirror members own the host
/// buffers that back the pointers.
///
/// @tparam T        The element scalar type exchanged over MPI.
/// @tparam SrcView  The (const) source Kokkos::View type.
/// @tparam DstView  The destination Kokkos::View type.
/// @tparam Staged   Whether host staging is in effect (== requires_staging_v).
template <typename T, typename SrcView, typename DstView, bool Staged>
struct Dispatch_Buffers;

// ── Direct path (host-direct / device-direct): no host mirrors owned ────────
template <typename T, typename SrcView, typename DstView>
struct Dispatch_Buffers<T, SrcView, DstView, /*Staged=*/false> {
    const T *send_ptr_{nullptr};
    T *recv_ptr_{nullptr};

    /// @return Pointer to hand to MPI as the send buffer.
    [[nodiscard]] const T *send_ptr() const noexcept { return send_ptr_; }
    /// @return Pointer to hand to MPI as the receive buffer.
    [[nodiscard]] T *recv_ptr() const noexcept { return recv_ptr_; }
};

// ── Staged path: owns host mirrors backing the send/receive pointers ────────
template <typename T, typename SrcView, typename DstView>
struct Dispatch_Buffers<T, SrcView, DstView, /*Staged=*/true> {
    host_mirror_t<SrcView> send_host_{};  ///< Host mirror backing the send buffer.
    host_mirror_t<DstView> recv_host_{};  ///< Host mirror backing the receive buffer.

    /// @return Pointer to the host send mirror handed to MPI.
    [[nodiscard]] const T *send_ptr() const noexcept { return send_host_.data(); }
    /// @return Pointer to the host receive mirror handed to MPI.
    [[nodiscard]] T *recv_ptr() const noexcept { return recv_host_.data(); }
};

/// @brief Compile-time selector for the collective transfer path.
///
/// The static member `staged` is true exactly when either the source or the
/// destination view requires host staging. When staging is required for either
/// view, both are staged so the collective runs entirely on host memory.
///
/// @tparam T        The element scalar type exchanged over MPI.
/// @tparam SrcView  The source Kokkos::View type.
/// @tparam DstView  The destination Kokkos::View type.
template <typename T, typename SrcView, typename DstView>
struct collective_dispatch {
    /// True when host staging is required for the source or destination view.
    static constexpr bool staged = requires_staging_v<SrcView> || requires_staging_v<DstView>;

    /// The buffer-holder type returned by prepare() for this path.
    using buffers_type = Dispatch_Buffers<T, SrcView, DstView, staged>;

    /// @brief Prepare the send/receive memory for the MPI collective.
    ///
    /// Selects the transfer path at COMPILE TIME (Requirement 6.5):
    ///   - direct path: returns pointers aliasing `src.data()` / `dest.data()`.
    ///     (Requirements 6.1, 6.2)
    ///   - staged path: deep-copies the first `send_count` elements of `src`
    ///     into a host mirror via `detail::stage_send`, and allocates a host
    ///     mirror of `recv_count` elements for the receive side; MPI then runs
    ///     on those host buffers. (Requirement 6.3)
    ///
    /// @note If the source band needs an on-device strided reorder into a
    ///       contiguous layout before staging, the caller performs that with
    ///       `detail::pack` into a contiguous device buffer and passes THAT
    ///       contiguous view here as `src` (Requirement 6.4). This keeps the
    ///       reorder on device and off the host critical path.
    ///
    /// @param src         Source view (contiguous send band for this rank).
    /// @param dest        Destination view receiving the collective result.
    /// @param send_count  Number of elements to send from `src`.
    /// @param recv_count  Number of elements the receive buffer must hold.
    /// @return A buffers holder owning any host mirrors; keep it alive until
    ///         after the collective and the matching finalize() call.
    [[nodiscard]] static buffers_type prepare(const SrcView &src, DstView &dest, std::size_t send_count, std::size_t recv_count) {
        if constexpr (staged) {
            buffers_type buffers{};
            // Deep-copy the send band [0, send_count) to a host mirror (Req 6.3).
            buffers.send_host_ = stage_send(src, std::size_t{0}, send_count);
            // Allocate an uninitialized host receive mirror sized to the result.
            buffers.recv_host_ = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{},
                                                            Kokkos::subview(dest, Kokkos::make_pair(std::size_t{0}, recv_count)));
            return buffers;
        } else {
            // Host-direct / device-direct: hand raw view pointers to MPI
            // (Req 6.1, 6.2). recv_count is unused on this path.
            (void)recv_count;
            (void)send_count;
            return buffers_type{src.data(), dest.data()};
        }
    }

    /// @brief Finalize after the MPI collective completes.
    ///
    /// On the host-staged path, stages the host receive buffer back into the
    /// device `dest` view for the first `recv_count` elements via
    /// `detail::stage_recv` (Requirement 6.3). On the direct paths this is a
    /// no-op because MPI wrote directly into `dest`.
    ///
    /// @param buffers     The holder returned by prepare().
    /// @param dest        The destination view to stage received data into.
    /// @param recv_count  Number of received elements to copy back to device.
    static void finalize([[maybe_unused]] const buffers_type &buffers, [[maybe_unused]] DstView &dest, [[maybe_unused]] std::size_t recv_count) {
        if constexpr (staged) {
            stage_recv<DstView>(buffers.recv_host_, dest, std::size_t{0}, recv_count);
        }
        // Direct path: nothing to do — MPI wrote straight into dest.
    }
};

}  // namespace halo::detail

#endif  // HALO_DETAIL_COLLECTIVE_DISPATCH_HPP
