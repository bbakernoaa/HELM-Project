#ifndef HALO_GATHER_REPLICATED_HPP
#define HALO_GATHER_REPLICATED_HPP

/// @file gather_replicated.hpp
/// @brief Tier 2 single-collective execution of a Replicated_Gather_Plan.
///
/// gather_replicated<T, SrcView, DstView>() assembles a replicated multi-level
/// field on every rank from rank-local contiguous bands using a SINGLE
/// MPI_Allgatherv driven by the plan's derived, resized strided receive
/// datatype. Every rank's band, for every level, lands directly in the
/// replicated [level][j][i] layout with ZERO host reorder (Req 5.1, 5.2), and
/// the result is identical on every rank (Req 5.3).
///
/// The send/receive memory is prepared through detail::collective_dispatch,
/// which selects the host-direct / device-direct / host-staged path at COMPILE
/// TIME from requires_staging_v (no per-element runtime branch) (Req 6.1, 6.2,
/// 6.3, 6.4, 6.5). Any readiness/status gate is issued as AT MOST one vector
/// MPI_Allreduce before and one after the gather, each with a count independent
/// of num_levels (Req 7.1, 7.2, 7.3, 7.4). Diagnostics begin/end events are
/// emitted only when a callback is registered (Req 5.6).
///
/// Tier 1 isolation: this header depends only on the C++ standard library, MPI,
/// Kokkos, and other HALO headers (Req 9.2).

#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <halo/collectives.hpp>
#include <halo/communicator.hpp>
#include <halo/detail/collective_dispatch.hpp>
#include <halo/detail/mpi_datatype.hpp>
#include <halo/diagnostics.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/replicated_gather_plan.hpp>

namespace halo {

/// @brief Execute a Replicated_Gather_Plan as a SINGLE MPI_Allgatherv.
///
/// Places every rank's band, for every level, into @p dest in [level][j][i]
/// layout using the plan's derived resized strided receive datatype, with NO
/// host reorder (Req 5.1, 5.2). After success, @p dest holds identical contents
/// on every rank (Req 5.3).
///
/// This rank contributes `num_levels * band_counts[rank]` contiguous elements in
/// [level][band] order (base datatype T); the derived receive datatype scatters
/// them across levels via the plan's per-rank band_counts() (recvcounts) and
/// displacements() (rdispls) (see the design Data Models "crux").
///
/// Reductions: issues AT MOST one readiness allreduce before and one status
/// allreduce after the gather, each a single-element vector reduce whose count
/// is independent of num_levels (Req 7.1, 7.2, 7.3, 7.4).
///
/// Dispatch: host-direct, device-direct (GPU-aware), or host-staged is selected
/// at COMPILE TIME via detail::collective_dispatch (Req 6.1, 6.2, 6.3, 6.5). The
/// source band for this rank is expected contiguous in [level][band] order, so
/// no extra host reorder is performed; the host-staged path stages device data
/// through host mirrors only (Req 6.3, 6.4).
///
/// Emits Diagnostics begin/end events when a callback is registered (Req 5.6).
///
/// @tparam T       Element type of the gathered field.
/// @tparam SrcView Kokkos::View holding this rank's band for all levels.
/// @tparam DstView Kokkos::View for the replicated destination field.
/// @param plan The precomputed replicated-gather plan.
/// @param src  This rank's contiguous [level][band] source band.
/// @param dest The replicated destination field ([level][j][i], all ranks).
///
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allgatherv
///         fails (under ErrorPolicy::throw_on_error) (Req 5.4).
template <typename T, typename SrcView, typename DstView>
void gather_replicated(const Replicated_Gather_Plan<T> &plan, const SrcView &src, DstView &dest) {
    const Communicator &comm = plan.communicator();
    const int rank = comm.rank();

    // Element counts (independent of the memory-space dispatch path).
    // Send side: this rank contributes num_levels * band_counts[rank] elements
    // in contiguous [level][band] order (Req 5.2, receive-placement math). The
    // dispatch buffers are sized by this element count; the MPI send is issued
    // as band_counts[rank] instances of the plan's [band][level] send datatype
    // (see the MPI_Allgatherv call below and replicated_gather_plan.hpp).
    const int local_band = plan.band_counts()[static_cast<std::size_t>(rank)];
    const std::size_t send_count = static_cast<std::size_t>(plan.num_levels()) * static_cast<std::size_t>(local_band);
    // Receive side: the full replicated field is num_levels * per_level_total.
    const std::size_t recv_total = static_cast<std::size_t>(plan.num_levels()) * static_cast<std::size_t>(plan.per_level_total());

    // ─── Diagnostics: begin event (only when a callback is registered) ──────
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::begin, rank, /*neighbor_count=*/comm.size(),
                                         /*total_bytes=*/recv_total * sizeof(T), std::chrono::nanoseconds{0},
                                         /*is_async=*/false});  // Req 5.6
    }

    // Serialize MPI when the thread level is below MPI_THREAD_MULTIPLE (Req 5.5).
    detail::Serialized_MPI_Guard guard;

    // ─── Readiness gate — AT MOST one vector Allreduce, count independent of
    //     num_levels (Req 7.1, 7.3, 7.4). A single-element readiness flag. ───
    {
        const std::vector<int> readiness_flag{1};
        (void)allreduce<int>(comm, readiness_flag, MPI_MIN);
    }

    // ─── Compile-time path selection (Req 6.5): obtain send/recv pointers. ──
    using dispatch = detail::collective_dispatch<T, SrcView, DstView>;
    auto bufs = dispatch::prepare(src, dest, send_count, recv_total);

    // ─── Exactly ONE MPI_Allgatherv (Req 5.1). The send side presents this
    //     rank's contiguous [level][band] source as band_counts[rank] instances
    //     of the plan's [band][level] send datatype (one j-column across all
    //     levels per instance); the receive side uses the plan's per-rank
    //     band_counts() (recvcounts) and displacements() (rdispls) against the
    //     derived resized strided recv_type() so all levels land in
    //     [level][j][i] with no host reorder (Req 5.2). The send datatype's
    //     [band][level] ordering matches the order the receive datatype scatters
    //     (band is the outer recvcounts axis); see replicated_gather_plan.hpp. ─
    const int rc = MPI_Allgatherv(bufs.send_ptr(), local_band, plan.send_type(), bufs.recv_ptr(), plan.band_counts().data(),
                                  plan.displacements().data(), plan.recv_type(),
                                  comm.handle());  // Req 5.1
    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, rank, "MPI_Allgatherv", comm.handle());  // Req 5.4
    }

    // ─── Finalize the dispatch: host-staged path stages received data back to
    //     device; direct paths are a no-op (Req 6.3). ────────────────────────
    dispatch::finalize(bufs, dest, recv_total);

    // ─── Status gate — AT MOST one vector Allreduce, count independent of
    //     num_levels (Req 7.2, 7.4). A single-element status flag. ───────────
    {
        const std::vector<int> status_flag{rc == MPI_SUCCESS ? 1 : 0};
        (void)allreduce<int>(comm, status_flag, MPI_MIN);
    }

    // ─── Diagnostics: end event (only when a callback is registered) ────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::end, rank, /*neighbor_count=*/comm.size(),
                                         /*total_bytes=*/recv_total * sizeof(T), std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
                                         /*is_async=*/false});  // Req 5.6
    }
}

}  // namespace halo

#endif  // HALO_GATHER_REPLICATED_HPP
