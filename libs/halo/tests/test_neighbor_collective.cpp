// ─── Neighbor Collective Exchange Integration Test (real multi-rank MPI) ────
// Feature: halo-production-hardening
// Requirements: 13.1, 13.2, 13.3
//
// Exercises exchange_neighbor_collective on multi-dimensional Kokkos views with
// real MPI across 4 ranks. Verifies:
//   1. Symmetric 2D periodic ring topology uses the neighbor collective path
//      and produces correct halo data.
//   2. Asymmetric (non-periodic boundary) topology transparently falls back
//      to the Isend/Irecv path and still produces correct halo data.
//   3. exchange_neighbor_collective produces identical results to
//      exchange_structured_blocking on the same data.
//
// Topology:
//   Symmetric: 4 ranks in a periodic ring along d0 with no neighbors on d1.
//   Asymmetric: Linear arrangement along d0 (endpoints have no neighbor on
//               one face), non-periodic.
//
// Data pattern: each interior cell = rank*1000 + local_linear_index.
// After exchange, halo zones must contain the correct neighbor's interior data.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/exchange_structured.hpp"
#include "halo/structured_halo_plan.hpp"

namespace {

// ─── Constants ──────────────────────────────────────────────────────────────

/// Interior grid size per dimension (not including halos).
constexpr std::size_t kInterior = 6;

/// Halo width for all tests.
constexpr std::size_t kHalo = 1;

/// Total grid size per dimension (interior + 2*halo).
constexpr std::size_t kTotal = kInterior + 2 * kHalo;  // 8

/// Sentinel value for halo zones that should NOT be touched.
constexpr double kSentinel = -999.0;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Encode a value based on rank and linearized interior-local index.
inline double encode(int rank, std::size_t local_idx) {
    return static_cast<double>(rank) * 1000.0 + static_cast<double>(local_idx);
}

/// Initialize a 2D view: interior gets encoded values, halos get sentinel.
template <typename ViewType>
void init_view_2d(ViewType &view, int rank) {
    auto h_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h_view, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
            h_view(i, j) = encode(rank, local_idx);
        }
    }
    Kokkos::deep_copy(view, h_view);
}

// ─── Test 1: Symmetric 2D periodic ring via neighbor collective ─────────────

class NeighborCollectiveSymmetricTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";

        // Periodic ring along d0
        int west = (rank_ - 1 + size_) % size_;
        int east = (rank_ + 1) % size_;
        // faces: low-d0(west), high-d0(east), low-d1(south), high-d1(north)
        neighbors_ = {west, east, -1, -1};
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 4> neighbors_{};
};

TEST_F(NeighborCollectiveSymmetricTest, PeriodicRingExchangeCorrectness) {
    // Validates Requirement 13.1: neighbor collective exchange on symmetric
    // topology produces correct halo data.
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    // Confirm topology is symmetric (precondition for neighbor collective path)
    ASSERT_TRUE(plan.is_topology_symmetric()) << "Periodic ring should be detected as symmetric topology";

    Kokkos::View<double **, Kokkos::LayoutRight> view("nc_sym_2d", kTotal, kTotal);
    init_view_2d(view, rank_);

    halo::exchange_neighbor_collective(plan, view);

    // Verify halo correctness
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    int west = neighbors_[0];
    int east = neighbors_[1];

    // West halo: received from west neighbor's high-d0 send region (last
    // interior column). West neighbor's view(6, j) for interior j:
    //   encode(west, (kInterior-1)*kInterior + (j-kHalo))
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t west_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
        double expected = encode(west, west_local_idx);
        EXPECT_DOUBLE_EQ(h_view(0, j), expected) << "West halo mismatch at (0, " << j << ") on rank " << rank_;
    }

    // East halo: received from east neighbor's low-d0 send region (first
    // interior column). East neighbor's view(1, j) for interior j:
    //   encode(east, 0*kInterior + (j-kHalo))
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t east_local_idx = 0 * kInterior + (j - kHalo);
        double expected = encode(east, east_local_idx);
        EXPECT_DOUBLE_EQ(h_view(kTotal - 1, j), expected) << "East halo mismatch at (" << kTotal - 1 << ", " << j << ") on rank " << rank_;
    }

    // South/North halos should remain sentinel (no neighbors)
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        EXPECT_DOUBLE_EQ(h_view(i, 0), kSentinel) << "South halo should be untouched at (" << i << ", 0) rank " << rank_;
        EXPECT_DOUBLE_EQ(h_view(i, kTotal - 1), kSentinel) << "North halo should be untouched at (" << i << ", " << kTotal - 1 << ") rank " << rank_;
    }
}

// ─── Test 2: Asymmetric topology fallback to Isend/Irecv ────────────────────

class NeighborCollectiveAsymmetricTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
};

TEST_F(NeighborCollectiveAsymmetricTest, NonPeriodicFallbackCorrectness) {
    // Validates Requirement 13.2: when topology is asymmetric (non-periodic
    // boundary endpoints have no neighbor on one face),
    // exchange_neighbor_collective internally falls back to the Isend/Irecv
    // path and still produces correct data.
    //
    // Linear arrangement along d0: rank 0 has no left, rank 3 has no right.
    int left = (rank_ > 0) ? rank_ - 1 : -1;
    int right = (rank_ < size_ - 1) ? rank_ + 1 : -1;

    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<int, 4> neighbors = {left, right, -1, -1};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    Kokkos::View<double **, Kokkos::LayoutRight> view("nc_asym_2d", kTotal, kTotal);
    init_view_2d(view, rank_);

    // Call neighbor collective — should transparently handle this topology
    halo::exchange_neighbor_collective(plan, view);

    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    // Verify halos on faces without a neighbor remain sentinel
    if (left == -1) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            EXPECT_DOUBLE_EQ(h_view(0, j), kSentinel) << "West halo should be sentinel on rank " << rank_ << " at (0, " << j << ")";
        }
    }
    if (right == -1) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            EXPECT_DOUBLE_EQ(h_view(kTotal - 1, j), kSentinel)
                << "East halo should be sentinel on rank " << rank_ << " at (" << kTotal - 1 << ", " << j << ")";
        }
    }

    // Verify halos on faces WITH neighbors have correct data
    if (left >= 0) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t left_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
            double expected = encode(left, left_local_idx);
            EXPECT_DOUBLE_EQ(h_view(0, j), expected) << "West halo data mismatch on rank " << rank_ << " at (0, " << j << ")";
        }
    }
    if (right >= 0) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t right_local_idx = 0 * kInterior + (j - kHalo);
            double expected = encode(right, right_local_idx);
            EXPECT_DOUBLE_EQ(h_view(kTotal - 1, j), expected)
                << "East halo data mismatch on rank " << rank_ << " at (" << kTotal - 1 << ", " << j << ")";
        }
    }

    // South/North halos always untouched
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        EXPECT_DOUBLE_EQ(h_view(i, 0), kSentinel) << "South halo should be sentinel on rank " << rank_;
        EXPECT_DOUBLE_EQ(h_view(i, kTotal - 1), kSentinel) << "North halo should be sentinel on rank " << rank_;
    }
}

// ─── Test 3: Result equivalence with standard blocking exchange ─────────────

class NeighborCollectiveEquivalenceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";

        // Periodic ring along d0
        int west = (rank_ - 1 + size_) % size_;
        int east = (rank_ + 1) % size_;
        neighbors_ = {west, east, -1, -1};
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 4> neighbors_{};
};

TEST_F(NeighborCollectiveEquivalenceTest, MatchesBlockingExchangeResults) {
    // Validates Requirement 13.3: exchange_neighbor_collective produces
    // identical halo data as exchange_structured_blocking on the same input.
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    // Two separate plans (each caches its own topology comm)
    halo::Structured_Halo_Plan<2> plan_blocking(extents, neighbors_, halo_widths, *comm_);
    halo::Structured_Halo_Plan<2> plan_neighbor(extents, neighbors_, halo_widths, *comm_);

    // View for standard blocking exchange
    Kokkos::View<double **, Kokkos::LayoutRight> view_blocking("blocking_2d", kTotal, kTotal);
    init_view_2d(view_blocking, rank_);

    // View for neighbor collective exchange (identical initial data)
    Kokkos::View<double **, Kokkos::LayoutRight> view_neighbor("neighbor_2d", kTotal, kTotal);
    init_view_2d(view_neighbor, rank_);

    // Execute both exchange variants
    halo::exchange_structured_blocking(plan_blocking, view_blocking);
    halo::exchange_neighbor_collective(plan_neighbor, view_neighbor);

    // Compare all cells (interior + halos) element-by-element
    auto h_blocking = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_blocking);
    auto h_neighbor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_neighbor);

    for (std::size_t i = 0; i < kTotal; ++i) {
        for (std::size_t j = 0; j < kTotal; ++j) {
            EXPECT_DOUBLE_EQ(h_blocking(i, j), h_neighbor(i, j))
                << "Mismatch between blocking and neighbor_collective at (" << i << ", " << j << ") on rank " << rank_;
        }
    }
}

TEST_F(NeighborCollectiveEquivalenceTest, MatchesBlockingNonPeriodic) {
    // Also verify equivalence for the non-periodic (asymmetric) case.
    int left = (rank_ > 0) ? rank_ - 1 : -1;
    int right = (rank_ < size_ - 1) ? rank_ + 1 : -1;
    std::array<int, 4> asym_neighbors = {left, right, -1, -1};

    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan_blocking(extents, asym_neighbors, halo_widths, *comm_);
    halo::Structured_Halo_Plan<2> plan_neighbor(extents, asym_neighbors, halo_widths, *comm_);

    Kokkos::View<double **, Kokkos::LayoutRight> view_blocking("blk_np", kTotal, kTotal);
    init_view_2d(view_blocking, rank_);

    Kokkos::View<double **, Kokkos::LayoutRight> view_neighbor("nc_np", kTotal, kTotal);
    init_view_2d(view_neighbor, rank_);

    halo::exchange_structured_blocking(plan_blocking, view_blocking);
    halo::exchange_neighbor_collective(plan_neighbor, view_neighbor);

    auto h_blocking = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_blocking);
    auto h_neighbor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_neighbor);

    for (std::size_t i = 0; i < kTotal; ++i) {
        for (std::size_t j = 0; j < kTotal; ++j) {
            EXPECT_DOUBLE_EQ(h_blocking(i, j), h_neighbor(i, j))
                << "Non-periodic: mismatch between blocking and neighbor_collective" << " at (" << i << ", " << j << ") on rank " << rank_;
        }
    }
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────

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
