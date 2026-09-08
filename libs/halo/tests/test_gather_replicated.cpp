// ─── HALO Tier 2 Replicated-Gather Tests (real multi-rank MPI) ──────────────
// Feature: halo-collective-primitives
//
// Example-based GoogleTest end-to-end tests for the Tier 2 replicated gather
// (halo::Replicated_Gather_Plan + halo::gather_replicated), exercised against a
// REAL MPI runtime (run via `mpirun -np 2` / `-np 4` by CTest). These validate
// the single-collective, no-host-reorder gather end-to-end rather than through
// the single-rank MPI spy:
//
//   - Host-path gather_replicated on a small [level][band] field, compared
//     element-for-element to a naive per-level / per-rank gather reference; the
//     assembled field is asserted identical on EVERY rank (Req 5.1, 5.2, 5.3).
//   - Compile-time dispatch coverage via static_assert: host views select the
//     direct (non-staged) path; a device view under requires_staging_v (no
//     GPU-aware MPI) selects the staged path (Req 6.1, 6.2, 6.3, 6.5).
//
// Layout contract (from gather_replicated.hpp / replicated_gather_plan.hpp):
//   This rank contributes `num_levels * band_counts[rank]` contiguous ints in
//   [level][band] order: src[level * w + b]. The plan's resized strided receive
//   datatype scatters them so each element lands at
//       dest[level * per_level_total + (displacements[r] + b)]
//   in the replicated [level][j] layout with ZERO host reorder. With every rank
//   contributing the same band width w, per_level_total == w * size,
//   displacements[r] == r * w, so the destination index is
//       level * per_level_total + r * w + b.
//
// Each of the np ranks runs this GTest binary; because the gathered field is
// replicated, every rank asserts the SAME full reference. An assertion failure
// on ANY rank makes that process exit non-zero, so mpirun (and thus CTest)
// reports the failure.
//
// Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.5
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/detail/collective_dispatch.hpp"
#include "halo/detail/memory_traits.hpp"
#include "halo/environment.hpp"
#include "halo/gather_replicated.hpp"
#include "halo/replicated_gather_plan.hpp"

namespace {

/// 1D flattened host view for a replicated [level][j] int field (row-major:
/// index = level * per_level_total + j).
using HostView = Kokkos::View<int *, Kokkos::HostSpace>;

/// Encode a value that uniquely identifies (rank, level, band) so the reference
/// comparison can verify every element landed in exactly the right place.
constexpr int encode(int rank, int level, int band) noexcept { return rank * 1000 + level * 10 + band; }

/// Test fixture exposing this rank's rank/size over MPI_COMM_WORLD.
class GatherReplicatedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
};

// ─── Host-path gather_replicated matches a naive per-level reference ─────────
// Every rank contributes the SAME band width w for L levels. This rank fills its
// source in [level][band] order with encode(rank, level, b). After the single
// gather, the replicated destination must equal, on EVERY rank, the naive
// reference built independently: for each level, each rank r's band occupies
// [r*w, r*w+w) of that level and holds encode(r, level, b) (Req 5.1, 5.2, 5.3).
TEST_F(GatherReplicatedTest, HostPathMatchesNaivePerLevelReferenceOnEveryRank) {
    constexpr int w = 3;  // per-rank band width (same on every rank)
    constexpr int L = 4;  // number of levels

    halo::Replicated_Gather_Plan<int> plan(*comm_, w, L);

    // Plan invariants for the uniform-width setup.
    ASSERT_EQ(plan.num_levels(), L);
    ASSERT_EQ(plan.per_level_total(), w * size_);
    ASSERT_EQ(plan.band_counts().size(), static_cast<std::size_t>(size_));
    ASSERT_EQ(plan.displacements().size(), static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
        EXPECT_EQ(plan.band_counts()[static_cast<std::size_t>(r)], w);
        EXPECT_EQ(plan.displacements()[static_cast<std::size_t>(r)], r * w);
    }

    const int per_level_total = plan.per_level_total();

    // Source: this rank's L*w ints in [level][band] order (contiguous).
    HostView src("src", static_cast<std::size_t>(L) * static_cast<std::size_t>(w));
    for (int level = 0; level < L; ++level) {
        for (int b = 0; b < w; ++b) {
            src(static_cast<std::size_t>(level * w + b)) = encode(rank_, level, b);
        }
    }

    // Destination: full replicated field, zero-initialized (Kokkos View default
    // initializes to zero).
    const std::size_t dest_size =
        static_cast<std::size_t>(L) * static_cast<std::size_t>(per_level_total);
    HostView dest("dest", dest_size);

    // Single-collective replicated gather (Req 5.1).
    halo::gather_replicated<int>(plan, src, dest);

    // Naive reference: independently compute what dest MUST contain. For each
    // level, rank r's band [r*w, r*w+w) holds encode(r, level, b) (Req 5.2).
    std::vector<int> reference(dest_size, 0);
    for (int level = 0; level < L; ++level) {
        for (int r = 0; r < size_; ++r) {
            for (int b = 0; b < w; ++b) {
                const std::size_t idx =
                    static_cast<std::size_t>(level) * static_cast<std::size_t>(per_level_total) +
                    static_cast<std::size_t>(r * w + b);
                reference[idx] = encode(r, level, b);
            }
        }
    }

    // Every rank asserts the SAME full reference element-for-element (Req 5.3).
    ASSERT_EQ(dest.extent(0), dest_size);
    for (std::size_t idx = 0; idx < dest_size; ++idx) {
        EXPECT_EQ(dest(idx), reference[idx])
            << "replicated dest element " << idx << " mismatch on rank " << rank_;
    }
}

// ─── Compile-time dispatch coverage (Req 6.1, 6.2, 6.3, 6.5) ─────────────────
// The transfer path is selected at COMPILE TIME from requires_staging_v. Host
// views never require staging, so collective_dispatch selects the direct path.
// A device view without GPU-aware MPI requires staging, so the dispatch selects
// the host-staged path. The host static_assert is the essential one; the device
// one is guarded so a CPU-only container still compiles.
static_assert(halo::detail::collective_dispatch<int, HostView, HostView>::staged == false,
              "host views select the direct path");

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// A device view requires staging when GPU-aware MPI is not enabled.
using DeviceView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace::memory_space>;
#ifndef HALO_GPU_AWARE_MPI
static_assert(halo::detail::requires_staging_v<DeviceView>,
              "device view without GPU-aware MPI requires staging");
static_assert(halo::detail::collective_dispatch<int, DeviceView, DeviceView>::staged == true,
              "device views under requires_staging_v select staging");
#endif  // !HALO_GPU_AWARE_MPI
#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP

// ─── Runtime touch-point for the compile-time dispatch coverage ─────────────
// static_asserts above already enforce the path selection at compile time; this
// test documents that coverage and gives a runtime anchor for it (Req 6.5).
TEST_F(GatherReplicatedTest, HostViewsSelectDirectDispatchPath) {
    EXPECT_FALSE((halo::detail::collective_dispatch<int, HostView, HostView>::staged));
    EXPECT_FALSE(halo::detail::requires_staging_v<HostView>);
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────
// Initializes the real MPI runtime (MPI_THREAD_MULTIPLE requested), Kokkos, and
// the HALO Environment singleton once for the whole binary, tearing them down in
// reverse order. Registered before RUN_ALL_TESTS via gtest_main. Mirrors the
// setup used by test_collectives.cpp / test_exchange.cpp.
class HaloMpiEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        Kokkos::initialize();
        halo::Environment::initialize();
    }

    void TearDown() override {
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
