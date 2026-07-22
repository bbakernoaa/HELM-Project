#ifndef HALO_EXCHANGE_INDEXED_HPP
#define HALO_EXCHANGE_INDEXED_HPP

/// @file exchange_indexed.hpp
/// @brief Blocking indexed (gather/scatter) halo exchange function template.
///
/// Provides exchange_indexed<ViewType> which executes a synchronous halo
/// exchange using a precomputed Indexed_Halo_Plan. Unlike exchange_blocking
/// (exchange.hpp), which assumes a single flat `[send | owned | recv]` buffer,
/// exchange_indexed operates on fields stored in an `[owned | halo]` layout with
/// arbitrary per-neighbor, per-layer index lists: it gathers owned values at
/// each send neighbor's local send indices into dense per-neighbor buffers,
/// communicates them via MPI, and scatters the received values into halo
/// elements at each recv neighbor's local recv indices.
///
/// The exchange operates over a caller-supplied subset of halo layers, so only
/// the indices belonging to those layers are gathered and scattered; halo
/// elements belonging solely to excluded layers are left unchanged.
///
/// Field views may be rank-1 (indexed by element) or rank-2
/// (nVertLevels x nElements, LayoutLeft); for rank-2 fields every vertical level
/// of each selected element is transferred as a contiguous per-element block.
///
/// Reuses the existing HALO infrastructure: detail::compute_tag for the
/// deterministic tag scheme, detail::Serialized_MPI_Guard for thread safety,
/// detail::staging (host mirrors) and Environment::is_gpu_aware_mpi() /
/// detail::requires_staging_v for path selection, and Diagnostics for begin/end
/// instrumentation. MPI failures are routed through detail::handle_mpi_error.

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <halo/communicator.hpp>
#include <halo/detail/compute_tag.hpp>
#include <halo/detail/gpu_aware_probe.hpp>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/mpi_datatype.hpp>
#include <halo/detail/staging.hpp>
#include <halo/diagnostics.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/indexed_halo_plan.hpp>

namespace halo {

namespace detail {

/// @brief Create a device-resident (view's memory space) index view from a host
///        index list.
///
/// Allocates a rank-1 View<std::size_t*, MemSpace>, fills a host mirror from the
/// supplied std::vector, and deep-copies it into the target memory space so the
/// index list is usable inside gather/scatter kernels launched in the view's
/// execution space.
///
/// @tparam MemSpace The target memory space (the field view's memory space).
/// @param indices The 0-based local indices to upload.
/// @param label   A Kokkos allocation label for diagnostics.
/// @return A view in MemSpace containing the indices (size == indices.size()).
template <typename MemSpace>
[[nodiscard]] Kokkos::View<std::size_t *, MemSpace> make_index_view(const std::vector<std::size_t> &indices, const char *label) {
    // Wrap the label in std::string so Kokkos treats it unambiguously as an
    // allocation label rather than a pointer-to-memory to wrap.
    Kokkos::View<std::size_t *, MemSpace> dev(Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing), indices.size());
    auto host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, dev);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        host(i) = indices[i];
    }
    Kokkos::deep_copy(dev, host);
    return dev;
}

/// @brief Gather field values at the given local indices into a dense buffer.
///
/// For a rank-1 view: `buf(k) = view(idx(k))`.
/// For a rank-2 (nVertLevels x nElements, LayoutLeft) view, each element's
/// vertical column is stored as a contiguous per-element block:
/// `buf(k*bs + lev) = view(lev, idx(k))`.
///
/// The kernel runs in the view's native execution space and fences on
/// completion so the buffer is safe to hand to MPI.
///
/// @tparam ViewType Field view type (rank 1 or 2).
/// @tparam BufType  Rank-1 destination buffer view type.
/// @tparam IdxType  Rank-1 index view type (indices in the view's memory space).
/// @param view The source field view.
/// @param buf  The destination dense buffer (size >= n_idx * bs).
/// @param idx  The local indices to gather.
/// @param bs   Per-element block size (1 for rank-1, nVertLevels for rank-2).
/// @param exec The execution space instance to launch the kernel on.
template <typename ViewType, typename BufType, typename IdxType>
void gather_indexed(const ViewType &view, const BufType &buf, const IdxType &idx, std::size_t bs, typename ViewType::execution_space exec) {
    using exec_space = typename ViewType::execution_space;
    using size_type = typename ViewType::size_type;

    const size_type n = idx.extent(0);
    if (n == 0) {
        return;
    }

    if constexpr (ViewType::rank == 1) {
        (void)bs;
        Kokkos::parallel_for(
            "halo_gather_indexed_r1", Kokkos::RangePolicy<exec_space>(exec, 0, n), KOKKOS_LAMBDA(const size_type k) { buf(k) = view(idx(k)); });
    } else {
        const size_type nlev = static_cast<size_type>(bs);
        Kokkos::parallel_for(
            "halo_gather_indexed_r2", Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(exec, {0, 0}, {n, nlev}),
            KOKKOS_LAMBDA(const size_type k, const size_type lev) { buf(k * nlev + lev) = view(lev, idx(k)); });
    }

    exec.fence("halo::detail::gather_indexed");
}

/// @brief Scatter a dense buffer into field values at the given local indices.
///
/// Inverse of gather_indexed(). For a rank-1 view: `view(idx(k)) = buf(k)`.
/// For a rank-2 view: `view(lev, idx(k)) = buf(k*bs + lev)`.
///
/// The kernel runs in the view's native execution space and fences on
/// completion so the scattered values are visible before computation resumes.
///
/// @tparam ViewType Field view type (rank 1 or 2).
/// @tparam BufType  Rank-1 source buffer view type.
/// @tparam IdxType  Rank-1 index view type (indices in the view's memory space).
/// @param view The destination field view.
/// @param buf  The source dense buffer (size >= n_idx * bs).
/// @param idx  The local indices to scatter into.
/// @param bs   Per-element block size (1 for rank-1, nVertLevels for rank-2).
/// @param exec The execution space instance to launch the kernel on.
template <typename ViewType, typename BufType, typename IdxType>
void scatter_indexed(const ViewType &view, const BufType &buf, const IdxType &idx, std::size_t bs, typename ViewType::execution_space exec) {
    using exec_space = typename ViewType::execution_space;
    using size_type = typename ViewType::size_type;

    const size_type n = idx.extent(0);
    if (n == 0) {
        return;
    }

    if constexpr (ViewType::rank == 1) {
        (void)bs;
        Kokkos::parallel_for(
            "halo_scatter_indexed_r1", Kokkos::RangePolicy<exec_space>(exec, 0, n), KOKKOS_LAMBDA(const size_type k) { view(idx(k)) = buf(k); });
    } else {
        const size_type nlev = static_cast<size_type>(bs);
        Kokkos::parallel_for(
            "halo_scatter_indexed_r2", Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(exec, {0, 0}, {n, nlev}),
            KOKKOS_LAMBDA(const size_type k, const size_type lev) { view(lev, idx(k)) = buf(k * nlev + lev); });
    }

    exec.fence("halo::detail::scatter_indexed");
}

}  // namespace detail

/// @brief Blocking indexed (gather/scatter) halo exchange over a layer subset.
///
/// Gathers owned values at each send neighbor's local send indices (for the
/// selected halo layers) into dense per-neighbor buffers, posts all MPI_Irecv
/// before any MPI_Isend, calls MPI_Waitall, then scatters each received buffer
/// into the field view at the corresponding recv neighbor's local recv indices.
/// Halo elements belonging only to layers outside @p layer_subset are left
/// unchanged.
///
/// Dispatches the GPU-direct or host-staged path using the same compile-time
/// trait (detail::requires_staging_v) and runtime probe
/// (Environment::is_gpu_aware_mpi()) as the contiguous exchange.
///
/// @tparam ViewType A Kokkos::View type of rank 1 (nElements) or rank 2
///         (nVertLevels x nElements, LayoutLeft).
/// @param plan         The precomputed indexed halo exchange plan.
/// @param field_view   The field view to exchange, in `[owned | halo]` order.
/// @param layer_subset 0-based halo layer indices to exchange. Passing every
///        layer exchanges the full halo.
///
/// @throws std::runtime_error (or aborts, per the active ErrorPolicy) if any MPI
///         operation fails, naming the operation and neighbor rank.
///
/// @note Uses detail::Serialized_MPI_Guard for thread safety when the MPI
///       thread level is below MPI_THREAD_MULTIPLE.
template <typename ViewType>
void exchange_indexed(const Indexed_Halo_Plan &plan, ViewType &field_view, std::span<const int> layer_subset) {
    static_assert(ViewType::rank == 1 || ViewType::rank == 2, "exchange_indexed supports rank-1 or rank-2 field views only");

    // ─── Early return for empty plans (no neighbors in either direction) ─────
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return;
    }

    using value_type = typename ViewType::value_type;
    using exec_space = typename ViewType::execution_space;
    using mem_space = typename ViewType::memory_space;
    using buffer_t = Kokkos::View<value_type *, mem_space>;
    // Derive the host staging buffer type from create_mirror_view rather than
    // buffer_t::HostMirror: for a host-resident buffer_t (e.g.
    // View<double*, HostSpace>) some Kokkos configurations do not expose a
    // HostMirror member typedef, whereas create_mirror_view is always valid and
    // yields the correct host-space mirror type in both host and device cases.
    using host_buffer_t = decltype(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, std::declval<buffer_t>()));

    // ─── Per-element block size ─────────────────────────────────────────────
    // rank-1 -> 1 value per index; rank-2 (nVertLevels x nElements, LayoutLeft)
    // -> extent(0) (nVertLevels) values per index, one contiguous column block.
    std::size_t bs = 1;
    if constexpr (ViewType::rank == 2) {
        bs = static_cast<std::size_t>(field_view.extent(0));
    }

    const auto send_info = plan.send_info();
    const auto recv_info = plan.recv_info();
    const std::size_t num_send = send_info.size();
    const std::size_t num_recv = recv_info.size();

    // ─── Resolve the per-neighbor index lists for the selected layers ────────
    std::vector<std::vector<std::size_t>> send_idx(num_send);
    std::vector<std::vector<std::size_t>> recv_idx(num_recv);
    std::size_t total_send_elems = 0;
    std::size_t total_recv_elems = 0;
    for (std::size_t i = 0; i < num_send; ++i) {
        send_idx[i] = send_info[i].indices_for(layer_subset);
        total_send_elems += send_idx[i].size() * bs;
    }
    for (std::size_t i = 0; i < num_recv; ++i) {
        recv_idx[i] = recv_info[i].indices_for(layer_subset);
        total_recv_elems += recv_idx[i].size() * bs;
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        const int num_neighbors = static_cast<int>(num_send + num_recv);
        const std::size_t total_bytes = (total_send_elems + total_recv_elems) * sizeof(value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::begin, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::nanoseconds{0},
                                         /*is_async=*/false});
    }

    // ─── Acquire serialization guard for thread safety ──────────────────────
    detail::Serialized_MPI_Guard guard;

    const auto &comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();
    const MPI_Datatype mpi_dtype = detail::mpi_datatype_for<value_type>();

    exec_space exec{};

    // ─── Gather owned values into per-neighbor send buffers ─────────────────
    std::vector<buffer_t> send_buf;
    send_buf.reserve(num_send);
    for (std::size_t i = 0; i < num_send; ++i) {
        const std::size_t n_elem = send_idx[i].size() * bs;
        buffer_t buf(Kokkos::view_alloc(std::string("halo_indexed_send_buf"), Kokkos::WithoutInitializing), n_elem);
        if (!send_idx[i].empty()) {
            auto didx = detail::make_index_view<mem_space>(send_idx[i], "halo_indexed_send_idx");
            detail::gather_indexed(field_view, buf, didx, bs, exec);
        }
        send_buf.push_back(buf);
    }

    // ─── Allocate per-neighbor recv buffers ─────────────────────────────────
    std::vector<buffer_t> recv_buf;
    recv_buf.reserve(num_recv);
    for (std::size_t i = 0; i < num_recv; ++i) {
        const std::size_t n_elem = recv_idx[i].size() * bs;
        recv_buf.emplace_back(Kokkos::view_alloc(std::string("halo_indexed_recv_buf"), Kokkos::WithoutInitializing), n_elem);
    }

    // ─── Select the communication path and set up MPI buffer pointers ───────
    // Compile-time trait says whether a device view would normally require host
    // staging; the runtime probe may still reveal GPU-aware MPI is available, in
    // which case device pointers are passed directly (mirrors exchange_blocking).
    bool staged = false;
    if constexpr (detail::requires_staging_v<ViewType>) {
        staged = !Environment::is_gpu_aware_mpi();
    }

    std::vector<value_type *> send_ptr(num_send, nullptr);
    std::vector<value_type *> recv_ptr(num_recv, nullptr);

    // Host staging buffers must outlive the MPI operations; keep them in scope.
    std::vector<host_buffer_t> host_send;
    std::vector<host_buffer_t> host_recv;

    if (staged) {
        host_send.reserve(num_send);
        host_recv.reserve(num_recv);
        for (std::size_t i = 0; i < num_send; ++i) {
            auto hb = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, send_buf[i]);
            Kokkos::deep_copy(hb, send_buf[i]);
            host_send.push_back(hb);
            send_ptr[i] = hb.data();
        }
        for (std::size_t i = 0; i < num_recv; ++i) {
            auto hb = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, recv_buf[i]);
            host_recv.push_back(hb);
            recv_ptr[i] = hb.data();
        }
    } else {
        for (std::size_t i = 0; i < num_send; ++i) {
            send_ptr[i] = send_buf[i].data();
        }
        for (std::size_t i = 0; i < num_recv; ++i) {
            recv_ptr[i] = recv_buf[i].data();
        }
    }

    // ─── Post all MPI_Irecv, then all MPI_Isend, then MPI_Waitall ───────────
    const std::size_t total_requests = num_send + num_recv;
    std::vector<MPI_Request> requests(total_requests, MPI_REQUEST_NULL);

    for (std::size_t i = 0; i < num_recv; ++i) {
        const int neighbor_rank = recv_info[i].rank;
        const int tag = detail::compute_tag(neighbor_rank, my_rank, comm_size);
        const int count = static_cast<int>(recv_idx[i].size() * bs);

        int rc = MPI_Irecv(recv_ptr[i], count, mpi_dtype, neighbor_rank, tag, mpi_comm, &requests[i]);
        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor_rank, "MPI_Irecv", mpi_comm);
        }
    }

    for (std::size_t i = 0; i < num_send; ++i) {
        const int neighbor_rank = send_info[i].rank;
        const int tag = detail::compute_tag(my_rank, neighbor_rank, comm_size);
        const int count = static_cast<int>(send_idx[i].size() * bs);

        int rc = MPI_Isend(send_ptr[i], count, mpi_dtype, neighbor_rank, tag, mpi_comm, &requests[num_recv + i]);
        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor_rank, "MPI_Isend", mpi_comm);
        }
    }

    int rc = MPI_Waitall(static_cast<int>(total_requests), requests.data(), MPI_STATUSES_IGNORE);
    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, my_rank, "MPI_Waitall", mpi_comm);
    }

    // ─── Copy received host data back to device buffers (staged path) ───────
    if (staged) {
        for (std::size_t i = 0; i < num_recv; ++i) {
            Kokkos::deep_copy(recv_buf[i], host_recv[i]);
        }
    }

    // ─── Scatter received buffers into the halo indices for selected layers ──
    for (std::size_t i = 0; i < num_recv; ++i) {
        if (!recv_idx[i].empty()) {
            auto didx = detail::make_index_view<mem_space>(recv_idx[i], "halo_indexed_recv_idx");
            detail::scatter_indexed(field_view, recv_buf[i], didx, bs, exec);
        }
    }

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        const int num_neighbors = static_cast<int>(num_send + num_recv);
        const std::size_t total_bytes = (total_send_elems + total_recv_elems) * sizeof(value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::end, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
                                         /*is_async=*/false});
    }
}

}  // namespace halo

#endif  // HALO_EXCHANGE_INDEXED_HPP
