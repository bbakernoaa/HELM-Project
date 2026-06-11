#ifndef HALO_EXCHANGE_STRUCTURED_HPP
#define HALO_EXCHANGE_STRUCTURED_HPP

/// @file exchange_structured.hpp
/// @brief Blocking and non-blocking structured halo exchange function templates.
///
/// Provides exchange_structured_blocking<ViewType> and
/// exchange_structured_async<ViewType> which execute structured halo exchanges
/// using a precomputed Structured_Halo_Plan. These operate on multi-dimensional
/// Kokkos views (rank 1-4) with GPU-resident pack/unpack kernels.
///
/// The structured exchange:
/// 1. Packs all send regions into contiguous buffers (GPU kernels)
/// 2. Posts all MPI_Irecv (into contiguous recv buffers) before any MPI_Isend
/// 3. Posts all MPI_Isend from contiguous send buffers
/// 4. Waits for completion (blocking) or defers (async)
/// 5. Unpacks all recv buffers into the view's halo regions (GPU kernels)
///
/// Thread safety is ensured via detail::Serialized_MPI_Guard when the MPI
/// thread level is below MPI_THREAD_MULTIPLE.

#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include <halo/communicator.hpp>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/pack_unpack.hpp>
#include <halo/diagnostics.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/halo_handle.hpp>
#include <halo/request_guard.hpp>
#include <halo/structured_halo_plan.hpp>

namespace halo {

namespace detail {

/// @brief Compute a deterministic MPI tag for structured exchange.
///
/// tag = (sender_rank * comm_size + receiver_rank) % MPI_TAG_UB_VALUE
///
/// @param sender   The rank of the sending process.
/// @param receiver The rank of the receiving process.
/// @param comm_size Total number of processes in the communicator.
/// @return A tag value in [0, MPI_TAG_UB).
inline int structured_compute_tag(int sender, int receiver, int comm_size) noexcept {
    int* tag_ub_ptr = nullptr;
    int flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);
    int tag_ub = (flag && tag_ub_ptr != nullptr) ? *tag_ub_ptr : 32767;
    long long product = static_cast<long long>(sender) * comm_size + receiver;
    return static_cast<int>(product % tag_ub);
}

/// @brief Map a Kokkos::View value_type to the corresponding MPI_Datatype.
template <typename T>
inline MPI_Datatype mpi_type_for() noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<T, int>) {
        return MPI_INT;
    } else if constexpr (std::is_same_v<T, long>) {
        return MPI_LONG;
    } else if constexpr (std::is_same_v<T, long long>) {
        return MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<T, char>) {
        return MPI_CHAR;
    } else {
        return MPI_BYTE;
    }
}

/// @brief Create a subview of a multi-dimensional view given a Region.
///
/// Dispatches to the appropriate Kokkos::subview call based on view rank.
/// Returns a subview matching the region's index ranges.
template <typename ViewType, int Rank>
auto make_subview(ViewType& view, const Region<Rank>& region) {
    if constexpr (Rank == 1) {
        return Kokkos::subview(view,
            Kokkos::make_pair(region.ranges[0].first, region.ranges[0].second));
    } else if constexpr (Rank == 2) {
        return Kokkos::subview(view,
            Kokkos::make_pair(region.ranges[0].first, region.ranges[0].second),
            Kokkos::make_pair(region.ranges[1].first, region.ranges[1].second));
    } else if constexpr (Rank == 3) {
        return Kokkos::subview(view,
            Kokkos::make_pair(region.ranges[0].first, region.ranges[0].second),
            Kokkos::make_pair(region.ranges[1].first, region.ranges[1].second),
            Kokkos::make_pair(region.ranges[2].first, region.ranges[2].second));
    } else if constexpr (Rank == 4) {
        return Kokkos::subview(view,
            Kokkos::make_pair(region.ranges[0].first, region.ranges[0].second),
            Kokkos::make_pair(region.ranges[1].first, region.ranges[1].second),
            Kokkos::make_pair(region.ranges[2].first, region.ranges[2].second),
            Kokkos::make_pair(region.ranges[3].first, region.ranges[3].second));
    }
}

} // namespace detail

/// @brief Blocking structured halo exchange.
///
/// For each face with a valid neighbor:
/// 1. Pack the send region into a contiguous buffer using GPU pack kernels
/// 2. Post MPI_Irecv into contiguous recv buffers (before any sends)
/// 3. Post MPI_Isend from contiguous send buffers
/// 4. MPI_Waitall for all requests to complete
/// 5. Unpack each recv buffer into the view's halo region using GPU unpack kernels
///
/// @tparam ViewType A Kokkos::View type (rank 1-4).
/// @param plan The precomputed structured halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
///
/// @throws std::runtime_error if any MPI operation fails.
template <typename ViewType>
void exchange_structured_blocking(
    const Structured_Halo_Plan<ViewType::rank>& plan, ViewType& view)
{
    constexpr int Rank = ViewType::rank;
    using value_type = typename ViewType::non_const_value_type;
    using buffer_view_t = Kokkos::View<value_type*, typename ViewType::memory_space>;

    // Early return if no active neighbors
    if (plan.active_neighbor_count() == 0) {
        return;
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    constexpr int num_faces = Structured_Halo_Plan<Rank>::num_faces_value;
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        // Compute total bytes across all active faces
        std::size_t total_bytes = 0;
        for (int f = 0; f < num_faces; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::begin,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::nanoseconds{0},
            /*is_async=*/false});
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto& comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();
    const MPI_Datatype mpi_dtype = detail::mpi_type_for<value_type>();

    // ─── Phase 1: Pack all send regions ─────────────────────────────────────
    std::vector<buffer_view_t> send_buffers;
    send_buffers.reserve(num_faces);

    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            send_buffers.emplace_back();  // placeholder
            continue;
        }
        const auto& send_region = plan.send_region(f);
        const std::size_t count = send_region.size();

        buffer_view_t buf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                          "structured_send_buf"), count);

        auto subview = detail::make_subview(view, send_region);
        detail::pack(subview, buf);

        send_buffers.push_back(std::move(buf));
    }

    // ─── Phase 2: Allocate recv buffers and post MPI_Irecv ──────────────────
    std::vector<buffer_view_t> recv_buffers;
    recv_buffers.reserve(num_faces);

    std::vector<MPI_Request> requests;
    requests.reserve(2 * num_faces);

    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            recv_buffers.emplace_back();  // placeholder
            continue;
        }
        const auto& recv_region = plan.recv_region(f);
        const std::size_t count = recv_region.size();
        const int neighbor = plan.neighbor_rank(f);
        const int tag = detail::structured_compute_tag(neighbor, my_rank, comm_size);

        buffer_view_t buf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                          "structured_recv_buf"), count);

        MPI_Request req = MPI_REQUEST_NULL;
        int rc = MPI_Irecv(
            buf.data(),
            static_cast<int>(count),
            mpi_dtype,
            neighbor,
            tag,
            mpi_comm,
            &req);

        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor, "MPI_Irecv (structured)");
        }

        recv_buffers.push_back(std::move(buf));
        requests.push_back(req);
    }

    // ─── Phase 3: Post MPI_Isend ────────────────────────────────────────────
    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            continue;
        }
        const auto& send_region = plan.send_region(f);
        const std::size_t count = send_region.size();
        const int neighbor = plan.neighbor_rank(f);
        const int tag = detail::structured_compute_tag(my_rank, neighbor, comm_size);

        MPI_Request req = MPI_REQUEST_NULL;
        int rc = MPI_Isend(
            send_buffers[f].data(),
            static_cast<int>(count),
            mpi_dtype,
            neighbor,
            tag,
            mpi_comm,
            &req);

        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor, "MPI_Isend (structured)");
        }

        requests.push_back(req);
    }

    // ─── Phase 4: Wait for all requests ─────────────────────────────────────
    int rc = MPI_Waitall(
        static_cast<int>(requests.size()),
        requests.data(),
        MPI_STATUSES_IGNORE);

    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, my_rank, "MPI_Waitall (structured)");
    }

    // ─── Phase 5: Unpack recv buffers into halo regions ─────────────────────
    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            continue;
        }
        const auto& recv_region = plan.recv_region(f);
        auto subview = detail::make_subview(view, recv_region);
        detail::unpack(recv_buffers[f], subview);
    }

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        std::size_t total_bytes = 0;
        for (int f = 0; f < num_faces; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::end,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
            /*is_async=*/false});
    }
}

/// @brief Non-blocking asynchronous structured halo exchange.
///
/// Same pack/Irecv-before-Isend/unpack-on-completion pattern as blocking
/// but with deferred completion. Returns a Halo_Handle that the caller
/// uses to wait for completion. The post-receive unpack is attached as
/// the staged_recv callback on the handle, executed when wait() is called.
///
/// Buffer lifetime: All send and recv buffers are captured by the completion
/// lambda and kept alive until wait() is called on the returned handle.
///
/// @tparam ViewType A Kokkos::View type (rank 1-4).
/// @param plan The precomputed structured halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
/// @return A Halo_Handle owning all pending operations. Caller must call
///         wait() or test() to complete the exchange and trigger unpack.
///
/// @throws std::runtime_error if any MPI operation fails.
template <typename ViewType>
[[nodiscard]] Halo_Handle exchange_structured_async(
    const Structured_Halo_Plan<ViewType::rank>& plan, ViewType& view)
{
    constexpr int Rank = ViewType::rank;
    using value_type = typename ViewType::non_const_value_type;
    using buffer_view_t = Kokkos::View<value_type*, typename ViewType::memory_space>;

    // Early return for no active neighbors
    if (plan.active_neighbor_count() == 0) {
        return Halo_Handle{};
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        constexpr int nf = Structured_Halo_Plan<Rank>::num_faces_value;
        std::size_t total_bytes = 0;
        for (int f = 0; f < nf; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::begin,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::nanoseconds{0},
            /*is_async=*/true});
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto& comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();
    const MPI_Datatype mpi_dtype = detail::mpi_type_for<value_type>();

    constexpr int num_faces = Structured_Halo_Plan<Rank>::num_faces_value;

    Halo_Handle handle;
    handle.requests_.reserve(2 * num_faces);

    // ─── Phase 1: Pack all send regions ─────────────────────────────────────
    auto send_buffers = std::make_shared<std::vector<buffer_view_t>>();
    send_buffers->reserve(num_faces);

    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            send_buffers->emplace_back();  // placeholder
            continue;
        }
        const auto& send_region = plan.send_region(f);
        const std::size_t count = send_region.size();

        buffer_view_t buf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                          "structured_send_buf"), count);

        auto subview = detail::make_subview(view, send_region);
        detail::pack(subview, buf);

        send_buffers->push_back(std::move(buf));
    }

    // ─── Phase 2: Allocate recv buffers and post MPI_Irecv ──────────────────
    auto recv_buffers = std::make_shared<std::vector<buffer_view_t>>();
    recv_buffers->reserve(num_faces);

    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            recv_buffers->emplace_back();  // placeholder
            continue;
        }
        const auto& recv_region = plan.recv_region(f);
        const std::size_t count = recv_region.size();
        const int neighbor = plan.neighbor_rank(f);
        const int tag = detail::structured_compute_tag(neighbor, my_rank, comm_size);

        buffer_view_t buf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                          "structured_recv_buf"), count);

        MPI_Request req = MPI_REQUEST_NULL;
        int rc = MPI_Irecv(
            buf.data(),
            static_cast<int>(count),
            mpi_dtype,
            neighbor,
            tag,
            mpi_comm,
            &req);

        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor, "MPI_Irecv (structured async)");
        }

        recv_buffers->push_back(std::move(buf));
        handle.requests_.emplace_back(req);
    }

    // ─── Phase 3: Post MPI_Isend ────────────────────────────────────────────
    for (int f = 0; f < num_faces; ++f) {
        if (!plan.has_neighbor(f)) {
            continue;
        }
        const auto& send_region = plan.send_region(f);
        const std::size_t count = send_region.size();
        const int neighbor = plan.neighbor_rank(f);
        const int tag = detail::structured_compute_tag(my_rank, neighbor, comm_size);

        MPI_Request req = MPI_REQUEST_NULL;
        int rc = MPI_Isend(
            (*send_buffers)[f].data(),
            static_cast<int>(count),
            mpi_dtype,
            neighbor,
            tag,
            mpi_comm,
            &req);

        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, neighbor, "MPI_Isend (structured async)");
        }

        handle.requests_.emplace_back(req);
    }

    // ─── Phase 4: Attach unpack callback as staged_recv ─────────────────────
    // The lambda captures recv_buffers and send_buffers (shared_ptr) to keep
    // them alive until wait() is called. It also captures a reference to the
    // view and a copy of the plan's recv regions for unpacking.
    //
    // We capture the plan's face info needed for unpack by value (the regions
    // and neighbor flags) so the plan doesn't need to outlive the handle.
    struct Face_Info {
        bool active;
        Region<Rank> recv_region;
    };
    auto face_info = std::make_shared<std::vector<Face_Info>>();
    face_info->reserve(num_faces);
    for (int f = 0; f < num_faces; ++f) {
        Face_Info fi;
        fi.active = plan.has_neighbor(f);
        if (fi.active) {
            fi.recv_region = plan.recv_region(f);
        }
        face_info->push_back(fi);
    }

    auto staged = std::make_unique<Halo_Handle::Staged_Recv>();
    staged->post_recv_copy = [recv_buffers, send_buffers, face_info,
                              &view]() {
        constexpr int NF = Structured_Halo_Plan<Rank>::num_faces_value;
        for (int f = 0; f < NF; ++f) {
            const auto& fi = (*face_info)[f];
            if (!fi.active) {
                continue;
            }
            auto subview = detail::make_subview(view, fi.recv_region);
            detail::unpack((*recv_buffers)[f], subview);
        }
    };
    handle.staged_recv_ = std::move(staged);

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        constexpr int nf = Structured_Halo_Plan<Rank>::num_faces_value;
        std::size_t total_bytes = 0;
        for (int f = 0; f < nf; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::end,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
            /*is_async=*/true});
    }

    return handle;
}

/// @brief Topology-aware neighbor collective structured halo exchange.
///
/// Uses MPI_Dist_graph_create_adjacent to build a topology communicator from
/// the plan's neighbor lists, then performs the exchange via a single
/// MPI_Neighbor_alltoallv collective call. This allows the MPI implementation
/// to optimize routing based on network topology (e.g., Slingshot on HPE Cray).
///
/// The topology communicator is cached in the plan on first call and reused
/// for subsequent exchanges. The plan is taken by non-const reference because
/// lazy creation of the topology communicator mutates the plan's cached state.
///
/// Falls back to exchange_structured_blocking (Isend/Irecv path) when:
/// - MPI version < 3.0 (neighbor collectives not supported)
/// - The neighbor topology is asymmetric (different send/recv neighbor sets)
///
/// @tparam ViewType A Kokkos::View type (rank 1-4).
/// @param plan The precomputed structured halo exchange plan (non-const for
///        lazy topology comm caching).
/// @param view The Kokkos view containing data to exchange.
///
/// @throws std::runtime_error if any MPI operation fails.
template <typename ViewType>
void exchange_neighbor_collective(
    Structured_Halo_Plan<ViewType::rank>& plan, ViewType& view)
{
    constexpr int Rank = ViewType::rank;
    using value_type = typename ViewType::non_const_value_type;
    using buffer_view_t = Kokkos::View<value_type*, typename ViewType::memory_space>;

    // Early return if no active neighbors
    if (plan.active_neighbor_count() == 0) {
        return;
    }

    // ─── Check MPI version for neighbor collective support ──────────────────
    // MPI_Neighbor_alltoallv requires MPI-3.0 or later.
#if MPI_VERSION < 3
    // Fall back to Isend/Irecv path when MPI < 3.0
    exchange_structured_blocking(plan, view);
    return;
#else

    // ─── Check topology symmetry ────────────────────────────────────────────
    // Neighbor collectives require symmetric send/recv neighbor lists.
    if (!plan.is_topology_symmetric()) {
        exchange_structured_blocking(plan, view);
        return;
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    constexpr int num_faces = Structured_Halo_Plan<Rank>::num_faces_value;
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        std::size_t total_bytes = 0;
        for (int f = 0; f < num_faces; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::begin,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::nanoseconds{0},
            /*is_async=*/false});
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const MPI_Datatype mpi_dtype = detail::mpi_type_for<value_type>();

    // ─── Build the neighbor index mapping ───────────────────────────────────
    // Collect active face indices (faces that have valid neighbors).
    // The topology comm's neighbor ordering matches the order we provided
    // to MPI_Dist_graph_create_adjacent, which is faces in order [0..num_faces).
    std::vector<int> active_faces;
    active_faces.reserve(num_faces);
    for (int f = 0; f < num_faces; ++f) {
        if (plan.has_neighbor(f)) {
            active_faces.push_back(f);
        }
    }

    const int num_neighbors = static_cast<int>(active_faces.size());

    // ─── Get or create the topology communicator ────────────────────────────
    MPI_Comm topo_comm = plan.topology_comm();

    // ─── Phase 1: Pack all send regions into a concatenated buffer ──────────
    // Compute per-neighbor send counts and displacements.
    std::vector<int> send_counts(num_neighbors, 0);
    std::vector<int> send_displs(num_neighbors, 0);
    std::size_t total_send_count = 0;

    for (int i = 0; i < num_neighbors; ++i) {
        int f = active_faces[i];
        std::size_t count = plan.send_region(f).size();
        send_counts[i] = static_cast<int>(count);
        send_displs[i] = static_cast<int>(total_send_count);
        total_send_count += count;
    }

    // Allocate concatenated send buffer and pack all regions
    buffer_view_t send_buffer(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                              "neighbor_coll_send_buf"), total_send_count);

    for (int i = 0; i < num_neighbors; ++i) {
        int f = active_faces[i];
        std::size_t count = static_cast<std::size_t>(send_counts[i]);
        std::size_t offset = static_cast<std::size_t>(send_displs[i]);

        // Create a subview of the send buffer for this neighbor
        auto buf_slice = Kokkos::subview(send_buffer,
            Kokkos::make_pair(offset, offset + count));

        auto subview = detail::make_subview(view, plan.send_region(f));
        detail::pack(subview, buf_slice);
    }

    // ─── Phase 2: Allocate receive buffer ───────────────────────────────────
    std::vector<int> recv_counts(num_neighbors, 0);
    std::vector<int> recv_displs(num_neighbors, 0);
    std::size_t total_recv_count = 0;

    for (int i = 0; i < num_neighbors; ++i) {
        int f = active_faces[i];
        std::size_t count = plan.recv_region(f).size();
        recv_counts[i] = static_cast<int>(count);
        recv_displs[i] = static_cast<int>(total_recv_count);
        total_recv_count += count;
    }

    buffer_view_t recv_buffer(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                              "neighbor_coll_recv_buf"), total_recv_count);

    // ─── Phase 3: MPI_Neighbor_alltoallv ────────────────────────────────────
    int rc = MPI_Neighbor_alltoallv(
        send_buffer.data(), send_counts.data(), send_displs.data(), mpi_dtype,
        recv_buffer.data(), recv_counts.data(), recv_displs.data(), mpi_dtype,
        topo_comm);

    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, plan.communicator().rank(),
                                 "MPI_Neighbor_alltoallv (structured)");
    }

    // ─── Phase 4: Unpack receive buffer into halo regions ───────────────────
    for (int i = 0; i < num_neighbors; ++i) {
        int f = active_faces[i];
        std::size_t count = static_cast<std::size_t>(recv_counts[i]);
        std::size_t offset = static_cast<std::size_t>(recv_displs[i]);

        auto buf_slice = Kokkos::subview(recv_buffer,
            Kokkos::make_pair(offset, offset + count));

        auto subview = detail::make_subview(view, plan.recv_region(f));
        detail::unpack(buf_slice, subview);
    }

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        std::size_t total_bytes = 0;
        for (int f = 0; f < num_faces; ++f) {
            if (plan.has_neighbor(f)) {
                total_bytes += plan.send_region(f).size() * sizeof(value_type);
                total_bytes += plan.recv_region(f).size() * sizeof(value_type);
            }
        }
        Diagnostics::emit(Exchange_Event{
            Exchange_Event::Phase::end,
            plan.communicator().rank(),
            static_cast<int>(plan.active_neighbor_count()),
            total_bytes,
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
            /*is_async=*/false});
    }

#endif // MPI_VERSION < 3
}

} // namespace halo

#endif // HALO_EXCHANGE_STRUCTURED_HPP
