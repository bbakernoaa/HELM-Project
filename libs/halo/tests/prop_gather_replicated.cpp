// ─── Property-Based Tests: HALO Collective Primitives (real multi-rank MPI) ──
// Feature: halo-collective-primitives
//
// Uses RapidCheck with a REAL MPI runtime (run via `mpirun -np 2` and `-np 4`
// by CTest) to verify the collective primitives with randomized inputs. The
// headline property is Property 6 (the batched replicated gather equals the
// naive per-level/per-rank reference, bit-for-bit for integer elements, and is
// identical on every rank); the file additionally exercises Properties 2, 4, 5
// and the zero-count edge.
//
// Cross-rank generator determinism (essential under MPI): every rank runs its
// own RapidCheck generator, so the generated SHAPE (num_levels and each rank's
// band width) must agree across ranks or the collective is ill-formed. This
// file mirrors prop_structured_exchange.cpp precisely: the shape parameters are
// drawn on rank 0 and MPI_Bcast to every rank, so all ranks share the same
// geometry. Each rank then derives its OWN band width deterministically from
// that broadcast base and its rank index (width_r = base + r), so bands differ
// per rank while every rank still agrees on ALL ranks' widths.
//
// Property 6: Batched replicated gather equals the naive reference
//   Validates: Requirements 5.2, 5.3, 11.1, 11.2, 11.3, 11.4, 11.5
// Property 2: Prefix-sum displacements reconstruct counts
//   Validates: Requirements 2.1, 2.3
// Property 4: Vector Allreduce is elementwise across ranks
//   Validates: Requirements 3.1, 3.2, 3.3
// Property 5: Plan bands tile each level without gaps or overlaps
//   Validates: Requirements 4.2, 4.4
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <vector>

#include "halo/collectives.hpp"
#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/gather_replicated.hpp"
#include "halo/replicated_gather_plan.hpp"

namespace {

// ─── Globals for MPI rank / size ────────────────────────────────────────────
int g_rank = 0;
int g_size = 0;

// ─── Budget caps for the ~7 GB container (Req 11.5) ─────────────────────────
// per_level_total = sum_r width_r must stay small. With width_r = base + r and
// up to 4 ranks, capping `base` keeps per_level_total <= ~3000 and the total
// destination buffer (num_levels * per_level_total ints) well under a few MB.
constexpr int kMaxLevels = 16;   // num_levels in [1, kMaxLevels]
constexpr int kMaxBase = 200;    // base band width in [0, kMaxBase]

// Deterministic, collision-free element encoding for (rank, level, band index).
// rank*100000 + level*1000 + b fits comfortably in int for the capped ranges
// (rank < 32, level < 16, b < ~800) and is unique per (rank, level, b).
constexpr int encode(int rank, int level, int b) noexcept {
    return rank * 100000 + level * 1000 + b;
}

// This rank's band width for a broadcast `base`. width_r = base + r, so every
// rank derives the SAME set of widths (it knows base and every rank index),
// while individual bands genuinely differ per rank.
constexpr int band_width_for(int base, int rank) noexcept { return base + rank; }

}  // namespace

// ─── Property 6: Batched replicated gather equals the naive reference ───────
// Feature: halo-collective-primitives, Property 6: batched replicated gather
// equals the naive per-level/per-rank reference
//
// For randomly generated per-rank band sizes, level count, and element values,
// gather_replicated (single Allgatherv + strided datatype) produces a
// destination field equal element-for-element (bit-for-bit for int) to a naive
// per-level, per-rank gather reference, and identical on every rank.
//
// **Validates: Requirements 5.2, 5.3, 11.1, 11.2, 11.3, 11.4, 11.5**

RC_GTEST_PROP(GatherReplicatedProps, BatchedGatherEqualsNaiveReference, ()) {
    // Draw the SHAPE on rank 0 and broadcast so every rank agrees (Req 11.4).
    int params[2] = {0, 0};  // {num_levels, base_band_width}
    if (g_rank == 0) {
        params[0] = *rc::gen::inRange(1, kMaxLevels + 1);  // num_levels in [1,16]
        params[1] = *rc::gen::inRange(0, kMaxBase + 1);    // base band width in [0,200]
    }
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);

    const int num_levels = params[0];
    const int base = params[1];
    const int my_width = band_width_for(base, g_rank);

    halo::Communicator comm(MPI_COMM_WORLD);

    // Build the plan for this rank's band width and the shared level count.
    halo::Replicated_Gather_Plan<int> plan(comm, my_width, num_levels);

    const int per_level_total = plan.per_level_total();
    // Budget guard: keep the destination buffer modest for the container.
    RC_ASSERT(static_cast<long long>(num_levels) * per_level_total <= 200000LL);

    // Source: this rank's contiguous [level][band] band, size num_levels*my_width.
    // src[level * my_width + b] = encode(rank, level, b).
    Kokkos::View<int *, Kokkos::HostSpace> src(
        "src", static_cast<std::size_t>(num_levels) * static_cast<std::size_t>(my_width));
    for (int level = 0; level < num_levels; ++level) {
        for (int b = 0; b < my_width; ++b) {
            src(static_cast<std::size_t>(level) * static_cast<std::size_t>(my_width) +
                static_cast<std::size_t>(b)) = encode(g_rank, level, b);
        }
    }

    // Destination: the full replicated field, zero-initialized.
    const std::size_t dest_size =
        static_cast<std::size_t>(num_levels) * static_cast<std::size_t>(per_level_total);
    Kokkos::View<int *, Kokkos::HostSpace> dest("dest", dest_size);
    Kokkos::deep_copy(dest, 0);

    // ── The operation under test: ONE Allgatherv, no host reorder. ──
    halo::gather_replicated<int>(plan, src, dest);

    // ── Naive per-level / per-rank reference (runs identically on every rank
    //    since the plan's band_counts()/displacements() are replicated). ──
    const std::vector<int> &band_counts = plan.band_counts();
    const std::vector<int> &displs = plan.displacements();
    std::vector<int> expected(dest_size, 0);
    for (int level = 0; level < num_levels; ++level) {
        for (int r = 0; r < g_size; ++r) {
            const int width_r = band_counts[static_cast<std::size_t>(r)];
            const int base_off = level * per_level_total + displs[static_cast<std::size_t>(r)];
            for (int b = 0; b < width_r; ++b) {
                expected[static_cast<std::size_t>(base_off + b)] = encode(r, level, b);
            }
        }
    }

    // ── Element-for-element (bit-for-bit int) equality (Req 5.2, 11.1, 11.2).
    //    This assertion holds on EVERY rank => replicated result (Req 5.3). ──
    for (std::size_t i = 0; i < dest_size; ++i) {
        RC_ASSERT(dest(i) == expected[i]);
    }
}

// ─── Property 6 (zero-count edge): every rank contributes a zero-width band ──
// Feature: halo-collective-primitives, Property 6: batched replicated gather
// equals the naive per-level/per-rank reference (zero-count edge)
//
// When every rank's band width is 0, per_level_total is 0 and the replicated
// field is empty/degenerate. gather_replicated must not crash and must produce
// a result matching the (empty) reference, consistently on every rank.
//
// **Validates: Requirements 5.2, 5.3, 11.1, 11.5**

RC_GTEST_PROP(GatherReplicatedProps, ZeroWidthBandsProduceEmptyConsistentField, ()) {
    // Only the level count varies; every rank's band width is fixed to 0.
    int num_levels = 0;
    if (g_rank == 0) {
        num_levels = *rc::gen::inRange(1, kMaxLevels + 1);
    }
    MPI_Bcast(&num_levels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Replicated_Gather_Plan<int> plan(comm, /*local_band_count=*/0, num_levels);

    RC_ASSERT(plan.per_level_total() == 0);

    // Empty source and destination (num_levels * 0 == 0 elements).
    Kokkos::View<int *, Kokkos::HostSpace> src("src_zero", 0);
    Kokkos::View<int *, Kokkos::HostSpace> dest("dest_zero", 0);

    // Must not crash on the degenerate shape.
    halo::gather_replicated<int>(plan, src, dest);

    RC_ASSERT(dest.extent(0) == 0u);
}

// ─── Property 2: Prefix-sum displacements reconstruct counts ────────────────
// Feature: halo-collective-primitives, Property 2: prefix-sum displacements
// reconstruct counts
//
// For any non-negative per-rank count array of length comm.size(), the
// displacements from detail::prefix_sum satisfy displacements[0] == 0 and
// displacements[k] == displacements[k-1] + counts[k-1], so consecutive
// differences exactly recover the counts. Pure computation (no MPI needed), but
// exercised in this MPI binary. The counts vector is drawn on rank 0 and
// broadcast so every rank checks the same instance.
//
// **Validates: Requirements 2.1, 2.3**

RC_GTEST_PROP(GatherReplicatedProps, PrefixSumReconstructsCounts, ()) {
    const auto n = static_cast<std::size_t>(g_size);
    std::vector<int> counts(n, 0);
    if (g_rank == 0) {
        for (std::size_t k = 0; k < n; ++k) {
            // Non-negative, budget-capped counts.
            counts[k] = *rc::gen::inRange(0, kMaxBase + 1);
        }
    }
    MPI_Bcast(counts.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);

    const std::vector<int> displs = halo::detail::prefix_sum(counts);

    RC_ASSERT(displs.size() == counts.size());
    RC_ASSERT(displs[0] == 0);
    for (std::size_t k = 1; k < n; ++k) {
        RC_ASSERT(displs[k] == displs[k - 1] + counts[k - 1]);
        // Consecutive differences recover the counts exactly.
        RC_ASSERT(displs[k] - displs[k - 1] == counts[k - 1]);
    }
}

// ─── Property 4: Vector Allreduce is elementwise across ranks ───────────────
// Feature: halo-collective-primitives, Property 4: vector Allreduce is
// elementwise across ranks
//
// For an input vector whose length is agreed across ranks and whose values are
// derived deterministically from the rank (input[i] = rank + i), allreduce with
// MPI_SUM / MPI_MIN / MPI_MAX yields, for each element i, the reduction across
// ranks. Because input[i] = rank + i is known for every rank, the expected
// cross-rank reduction is computable locally:
//   SUM over ranks of (r + i) = size*i + sum(0..size-1)
//   MIN over ranks           = 0 + i
//   MAX over ranks           = (size-1) + i
// The length is drawn on rank 0 and broadcast.
//
// **Validates: Requirements 3.1, 3.2, 3.3**

RC_GTEST_PROP(GatherReplicatedProps, VectorAllreduceIsElementwise, ()) {
    int len = 0;
    if (g_rank == 0) {
        len = *rc::gen::inRange(1, 65);  // one or more elements, small
    }
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    halo::Communicator comm(MPI_COMM_WORLD);

    // input[i] = rank + i, so the reduction across ranks is known analytically.
    std::vector<int> input(static_cast<std::size_t>(len));
    for (int i = 0; i < len; ++i) {
        input[static_cast<std::size_t>(i)] = g_rank + i;
    }

    const int rank_sum = g_size * (g_size - 1) / 2;  // sum(0..size-1)

    const std::vector<int> reduced_sum = halo::allreduce<int>(comm, input, MPI_SUM);
    const std::vector<int> reduced_min = halo::allreduce<int>(comm, input, MPI_MIN);
    const std::vector<int> reduced_max = halo::allreduce<int>(comm, input, MPI_MAX);

    RC_ASSERT(reduced_sum.size() == static_cast<std::size_t>(len));
    RC_ASSERT(reduced_min.size() == static_cast<std::size_t>(len));
    RC_ASSERT(reduced_max.size() == static_cast<std::size_t>(len));

    for (int i = 0; i < len; ++i) {
        const auto ii = static_cast<std::size_t>(i);
        RC_ASSERT(reduced_sum[ii] == g_size * i + rank_sum);  // SUM_r (r + i)
        RC_ASSERT(reduced_min[ii] == 0 + i);                  // MIN_r (r + i)
        RC_ASSERT(reduced_max[ii] == (g_size - 1) + i);       // MAX_r (r + i)
    }
}

// ─── Property 5: Plan bands tile each level without gaps or overlaps ────────
// Feature: halo-collective-primitives, Property 5: plan bands tile each level
// without gaps or overlaps
//
// For randomly generated non-negative per-rank band counts, the plan's
// displacements place each rank's band contiguously and without overlap so the
// bands exactly tile [0, per_level_total): displacements[r] == sum of the
// preceding band counts, and displacements.back() + band_counts.back() ==
// per_level_total. Verified from a real plan built with the broadcast shape.
//
// **Validates: Requirements 4.2, 4.4**

RC_GTEST_PROP(GatherReplicatedProps, PlanBandsTileEachLevel, ()) {
    int params[2] = {0, 0};  // {num_levels, base_band_width}
    if (g_rank == 0) {
        params[0] = *rc::gen::inRange(1, kMaxLevels + 1);
        params[1] = *rc::gen::inRange(0, kMaxBase + 1);
    }
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);

    const int num_levels = params[0];
    const int base = params[1];
    const int my_width = band_width_for(base, g_rank);

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Replicated_Gather_Plan<int> plan(comm, my_width, num_levels);

    const std::vector<int> &band_counts = plan.band_counts();
    const std::vector<int> &displs = plan.displacements();
    const int per_level_total = plan.per_level_total();

    RC_ASSERT(band_counts.size() == static_cast<std::size_t>(g_size));
    RC_ASSERT(displs.size() == static_cast<std::size_t>(g_size));

    // Each displacement is the running sum of the preceding band counts
    // (contiguous, non-overlapping bands in ascending rank order).
    int running = 0;
    for (int r = 0; r < g_size; ++r) {
        RC_ASSERT(displs[static_cast<std::size_t>(r)] == running);
        running += band_counts[static_cast<std::size_t>(r)];
    }

    // The bands exactly tile [0, per_level_total): the last band ends at the
    // total, with no gaps or overlaps.
    RC_ASSERT(displs.back() + band_counts.back() == per_level_total);
    RC_ASSERT(running == per_level_total);
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────
// Mirrors prop_structured_exchange.cpp / test_collectives.cpp: initialize the
// real MPI runtime (MPI_THREAD_MULTIPLE requested), Kokkos, and the HALO
// Environment singleton once for the whole binary, tearing them down in reverse
// order. Registered before RUN_ALL_TESTS via gtest_main.

namespace {

class HaloMpiEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        Kokkos::initialize();
        halo::Environment::initialize();

        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    }

    void TearDown() override {
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
