// ─── Structured Halo Exchange Integration Test (real multi-rank MPI) ────────
// Feature: halo-production-hardening
// Requirements: 1.1, 1.2, 1.3, 2.5, 3.1
//
// Exercises the structured halo exchange (exchange_structured_blocking and
// exchange_structured_async) on multi-dimensional Kokkos views with real MPI
// across 4 ranks. Verifies:
//   - 2D periodic grid exchange (4 neighbors in a ring along both dims)
//   - 3D periodic grid exchange (ring along d0, non-periodic on d1/d2)
//   - Non-periodic boundaries (some faces have no neighbor)
//   - LayoutLeft and LayoutRight produce correct results
//   - Strided subview (one tracer from a 4D field) exchanges correctly
//
// Topology:
//   2D: 4 ranks in a periodic ring along d0. Each rank's west neighbor is
//       rank-1 (mod 4) and east neighbor is rank+1 (mod 4). d1 is non-periodic.
//   3D: 4 ranks in a periodic ring along d0. d1/d2 non-periodic.
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

// ─── 2D Periodic Ring Test ──────────────────────────────────────────────────
// 4 ranks arranged as a periodic ring along d0 (west/east neighbors).
// d1 has no neighbors (-1).

class StructuredExchange2DTest : public ::testing::Test {
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

    /// Initialize a 2D view: interior gets encoded values, halos get sentinel.
    template <typename ViewType>
    void init_view_2d(ViewType& view) {
        auto h_view = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(h_view, kSentinel);
        // Fill interior cells with encoded pattern
        for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
            for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
                std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
                h_view(i, j) = encode(rank_, local_idx);
            }
        }
        Kokkos::deep_copy(view, h_view);
    }

    /// Verify west/east halo zones contain correct neighbor data.
    /// Only verifies interior-j positions to avoid halo-on-halo corners.
    template <typename ViewType>
    void verify_halos_2d(const ViewType& view) {
        auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

        int west = neighbors_[0];
        int east = neighbors_[1];

        // West halo: recv region dim0=[0, kHalo)=[0,1), dim1=[0, kTotal)
        // Received from west neighbor's face 1 (high-d0) send region:
        //   dim0=[kTotal-2*kHalo, kTotal-kHalo)=[6,7), dim1=[0, kTotal)
        // West neighbor's view(6, j) for interior j:
        //   encode(west, (6-kHalo)*kInterior + (j-kHalo))
        //   = encode(west, (kInterior-1)*kInterior + (j-kHalo))
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t west_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
            double expected = encode(west, west_local_idx);
            EXPECT_DOUBLE_EQ(h_view(0, j), expected)
                << "West halo mismatch at (0, " << j << ") on rank " << rank_;
        }

        // East halo: recv region dim0=[kTotal-kHalo, kTotal)=[7,8), dim1=[0, kTotal)
        // Received from east neighbor's face 0 (low-d0) send region:
        //   dim0=[kHalo, 2*kHalo)=[1,2), dim1=[0, kTotal)
        // East neighbor's view(1, j) for interior j:
        //   encode(east, (1-kHalo)*kInterior + (j-kHalo))
        //   = encode(east, 0*kInterior + (j-kHalo))
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t east_local_idx = 0 * kInterior + (j - kHalo);
            double expected = encode(east, east_local_idx);
            EXPECT_DOUBLE_EQ(h_view(kTotal - 1, j), expected)
                << "East halo mismatch at (" << kTotal - 1 << ", " << j
                << ") on rank " << rank_;
        }

        // South/North halos should remain sentinel (no neighbors)
        for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
            EXPECT_DOUBLE_EQ(h_view(i, 0), kSentinel)
                << "South halo should be untouched at (" << i << ", 0) rank " << rank_;
            EXPECT_DOUBLE_EQ(h_view(i, kTotal - 1), kSentinel)
                << "North halo should be untouched at (" << i << ", " << kTotal - 1
                << ") rank " << rank_;
        }
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 4> neighbors_{};
};

TEST_F(StructuredExchange2DTest, BlockingPeriodicRingExchange) {
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double**, Kokkos::LayoutRight> view("field_2d", kTotal, kTotal);
    init_view_2d(view);

    halo::exchange_structured_blocking(plan, view);

    verify_halos_2d(view);
}

TEST_F(StructuredExchange2DTest, AsyncPeriodicRingExchange) {
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double**, Kokkos::LayoutRight> view("field_2d", kTotal, kTotal);
    init_view_2d(view);

    auto handle = halo::exchange_structured_async(plan, view);
    handle.wait();

    verify_halos_2d(view);
}

// ─── 3D Periodic Ring Topology Test ─────────────────────────────────────────

class StructuredExchange3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";

        // Ring topology along dimension 0 (periodic)
        // dim1 and dim2 are NOT periodic (no neighbors on those faces)
        int left  = (rank_ - 1 + size_) % size_;
        int right = (rank_ + 1) % size_;
        // faces: low-d0, high-d0, low-d1, high-d1, low-d2, high-d2
        neighbors_ = {left, right, -1, -1, -1, -1};
    }

    /// Initialize a 3D view: interior encoded, halos sentinel.
    void init_view_3d(Kokkos::View<double***, Kokkos::LayoutRight>& view) {
        auto h_view = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(h_view, kSentinel);
        for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
            for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
                for (std::size_t k = kHalo; k < kHalo + kInterior; ++k) {
                    std::size_t local_idx = (i - kHalo) * kInterior * kInterior
                                          + (j - kHalo) * kInterior
                                          + (k - kHalo);
                    h_view(i, j, k) = encode(rank_, local_idx);
                }
            }
        }
        Kokkos::deep_copy(view, h_view);
    }

    /// Verify halos on the d0 faces (ring dimension) are correct, and
    /// d1/d2 halos remain untouched (sentinel).
    void verify_halos_3d(const Kokkos::View<double***, Kokkos::LayoutRight>& view) {
        auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

        int left  = neighbors_[0];
        int right = neighbors_[1];

        // Left halo (face 0, recv region: dim0 [0, kHalo)=[0,1))
        // Left neighbor sends from face 1 (high-d0):
        //   dim0 [kTotal-2*kHalo, kTotal-kHalo) = [6, 7), dim1/dim2 full
        // Left neighbor's view(6, j, k) for interior j,k:
        //   encode(left, (6-kHalo)*kInterior^2 + (j-kHalo)*kInterior + (k-kHalo))
        //   = encode(left, (kInterior-1)*kInterior^2 + (j-kHalo)*kInterior + (k-kHalo))
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            for (std::size_t k = kHalo; k < kHalo + kInterior; ++k) {
                std::size_t left_local_idx =
                    (kInterior - 1) * kInterior * kInterior
                    + (j - kHalo) * kInterior + (k - kHalo);
                double expected = encode(left, left_local_idx);
                EXPECT_DOUBLE_EQ(h_view(0, j, k), expected)
                    << "Left halo mismatch at (0, " << j << ", " << k
                    << ") on rank " << rank_;
            }
        }

        // Right halo (face 1, recv region: dim0 [kTotal-kHalo, kTotal)=[7,8))
        // Right neighbor sends from face 0 (low-d0):
        //   dim0 [kHalo, 2*kHalo) = [1, 2), dim1/dim2 full
        // Right neighbor's view(1, j, k) for interior j,k:
        //   encode(right, 0*kInterior^2 + (j-kHalo)*kInterior + (k-kHalo))
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            for (std::size_t k = kHalo; k < kHalo + kInterior; ++k) {
                std::size_t right_local_idx =
                    0 * kInterior * kInterior
                    + (j - kHalo) * kInterior + (k - kHalo);
                double expected = encode(right, right_local_idx);
                EXPECT_DOUBLE_EQ(h_view(kTotal - 1, j, k), expected)
                    << "Right halo mismatch at (" << kTotal - 1 << ", " << j
                    << ", " << k << ") on rank " << rank_;
            }
        }

        // d1 and d2 halos should remain sentinel (no neighbors on those faces)
        // Check south halo: j=0, interior i and k
        for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
            for (std::size_t k = kHalo; k < kHalo + kInterior; ++k) {
                EXPECT_DOUBLE_EQ(h_view(i, 0, k), kSentinel)
                    << "South halo should be untouched at (" << i << ", 0, " << k
                    << ") on rank " << rank_;
            }
        }

        // Check bottom halo: k=0, interior i and j
        for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
            for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
                EXPECT_DOUBLE_EQ(h_view(i, j, 0), kSentinel)
                    << "Bottom halo should be untouched at (" << i << ", " << j
                    << ", 0) on rank " << rank_;
            }
        }
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 6> neighbors_{};
};

TEST_F(StructuredExchange3DTest, BlockingRingExchange) {
    std::array<std::size_t, 3> extents = {kTotal, kTotal, kTotal};
    std::array<std::size_t, 3> halo_widths = {kHalo, kHalo, kHalo};

    halo::Structured_Halo_Plan<3> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double***, Kokkos::LayoutRight> view("field_3d", kTotal, kTotal, kTotal);
    init_view_3d(view);

    halo::exchange_structured_blocking(plan, view);

    verify_halos_3d(view);
}

TEST_F(StructuredExchange3DTest, AsyncRingExchange) {
    std::array<std::size_t, 3> extents = {kTotal, kTotal, kTotal};
    std::array<std::size_t, 3> halo_widths = {kHalo, kHalo, kHalo};

    halo::Structured_Halo_Plan<3> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double***, Kokkos::LayoutRight> view("field_3d", kTotal, kTotal, kTotal);
    init_view_3d(view);

    auto handle = halo::exchange_structured_async(plan, view);
    handle.wait();

    verify_halos_3d(view);
}

// ─── Non-Periodic Boundary Test ─────────────────────────────────────────────

class StructuredExchangeNonPeriodicTest : public ::testing::Test {
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

TEST_F(StructuredExchangeNonPeriodicTest, BoundaryHalosUntouched) {
    // Linear arrangement along d0: rank 0, 1, 2, 3
    // Non-periodic: rank 0 has no left neighbor, rank 3 has no right neighbor
    int left  = (rank_ > 0) ? rank_ - 1 : -1;
    int right = (rank_ < size_ - 1) ? rank_ + 1 : -1;

    // 2D grid, only exchange along d0
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<int, 4> neighbors = {left, right, -1, -1};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    Kokkos::View<double**, Kokkos::LayoutRight> view("field_np", kTotal, kTotal);

    // Initialize interior
    auto h_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h_view, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
            h_view(i, j) = encode(rank_, local_idx);
        }
    }
    Kokkos::deep_copy(view, h_view);

    halo::exchange_structured_blocking(plan, view);

    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    // Verify halos on faces with no neighbor remain sentinel
    if (left == -1) {
        // West halo (i=0) should be untouched
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            EXPECT_DOUBLE_EQ(h_result(0, j), kSentinel)
                << "West halo should be sentinel on rank " << rank_
                << " at (0, " << j << ")";
        }
    }

    if (right == -1) {
        // East halo (i=kTotal-1) should be untouched
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            EXPECT_DOUBLE_EQ(h_result(kTotal - 1, j), kSentinel)
                << "East halo should be sentinel on rank " << rank_
                << " at (" << kTotal - 1 << ", " << j << ")";
        }
    }

    // South/North halos (no neighbor) should always be sentinel
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        EXPECT_DOUBLE_EQ(h_result(i, 0), kSentinel)
            << "South halo should be sentinel on rank " << rank_;
        EXPECT_DOUBLE_EQ(h_result(i, kTotal - 1), kSentinel)
            << "North halo should be sentinel on rank " << rank_;
    }

    // Verify that halos on faces WITH neighbors have correct data
    if (left >= 0) {
        // West halo: receives from left neighbor's face 1 (high-d0) send region
        // Left neighbor sends dim0 [6, 7) which is its last interior column
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t left_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
            double expected = encode(left, left_local_idx);
            EXPECT_DOUBLE_EQ(h_result(0, j), expected)
                << "West halo data mismatch on rank " << rank_
                << " at (0, " << j << ")";
        }
    }

    if (right >= 0) {
        // East halo: receives from right neighbor's face 0 (low-d0) send region
        // Right neighbor sends dim0 [1, 2) which is its first interior column
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t right_local_idx = 0 * kInterior + (j - kHalo);
            double expected = encode(right, right_local_idx);
            EXPECT_DOUBLE_EQ(h_result(kTotal - 1, j), expected)
                << "East halo data mismatch on rank " << rank_
                << " at (" << kTotal - 1 << ", " << j << ")";
        }
    }
}

// ─── LayoutLeft vs LayoutRight Test ─────────────────────────────────────────

class StructuredExchangeLayoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";

        // Periodic ring along d0
        int left  = (rank_ - 1 + size_) % size_;
        int right = (rank_ + 1) % size_;
        neighbors_ = {left, right, -1, -1};
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 4> neighbors_{};
};

TEST_F(StructuredExchangeLayoutTest, LayoutRightExchange) {
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double**, Kokkos::LayoutRight> view("field_lr", kTotal, kTotal);
    auto h_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h_view, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
            h_view(i, j) = encode(rank_, local_idx);
        }
    }
    Kokkos::deep_copy(view, h_view);

    halo::exchange_structured_blocking(plan, view);

    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    int left = neighbors_[0];
    // Verify west halo (interior j only)
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t left_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
        double expected = encode(left, left_local_idx);
        EXPECT_DOUBLE_EQ(h_result(0, j), expected)
            << "LayoutRight: West halo mismatch at (0, " << j << ") rank " << rank_;
    }
}

TEST_F(StructuredExchangeLayoutTest, LayoutLeftExchange) {
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    Kokkos::View<double**, Kokkos::LayoutLeft> view("field_ll", kTotal, kTotal);
    auto h_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(h_view, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
            h_view(i, j) = encode(rank_, local_idx);
        }
    }
    Kokkos::deep_copy(view, h_view);

    halo::exchange_structured_blocking(plan, view);

    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    int left = neighbors_[0];
    // Verify west halo (interior j only)
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t left_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
        double expected = encode(left, left_local_idx);
        EXPECT_DOUBLE_EQ(h_result(0, j), expected)
            << "LayoutLeft: West halo mismatch at (0, " << j << ") rank " << rank_;
    }
}

TEST_F(StructuredExchangeLayoutTest, BothLayoutsProduceSameResults) {
    // After exchange, both layouts should have the same halo data values.
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    // LayoutRight
    Kokkos::View<double**, Kokkos::LayoutRight> view_r("field_r", kTotal, kTotal);
    auto h_r = Kokkos::create_mirror_view(view_r);
    Kokkos::deep_copy(h_r, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            h_r(i, j) = encode(rank_, (i - kHalo) * kInterior + (j - kHalo));
        }
    }
    Kokkos::deep_copy(view_r, h_r);
    halo::exchange_structured_blocking(plan, view_r);

    // LayoutLeft
    Kokkos::View<double**, Kokkos::LayoutLeft> view_l("field_l", kTotal, kTotal);
    auto h_l = Kokkos::create_mirror_view(view_l);
    Kokkos::deep_copy(h_l, kSentinel);
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            h_l(i, j) = encode(rank_, (i - kHalo) * kInterior + (j - kHalo));
        }
    }
    Kokkos::deep_copy(view_l, h_l);
    halo::exchange_structured_blocking(plan, view_l);

    // Compare results on host
    auto result_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_r);
    auto result_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_l);

    for (std::size_t i = 0; i < kTotal; ++i) {
        for (std::size_t j = 0; j < kTotal; ++j) {
            EXPECT_DOUBLE_EQ(result_r(i, j), result_l(i, j))
                << "Layout mismatch at (" << i << ", " << j << ") rank " << rank_;
        }
    }
}

// ─── Strided Subview (4D field, one tracer) Test ────────────────────────────

class StructuredExchangeStridedTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        ASSERT_EQ(size_, 4) << "This test requires exactly 4 MPI ranks";

        // Periodic ring along d0
        int left  = (rank_ - 1 + size_) % size_;
        int right = (rank_ + 1) % size_;
        neighbors_ = {left, right, -1, -1};
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::array<int, 4> neighbors_{};
};

TEST_F(StructuredExchangeStridedTest, SingleTracerSubviewExchange) {
    // 4D field: (x, y, z, tracers) but we exchange halos on a 2D subview
    // (x, y) for a single tracer at a fixed z and tracer index.
    // The pack/unpack kernels handle strided views via MDRangePolicy.

    constexpr std::size_t nz = 4;
    constexpr std::size_t ntracers = 3;
    constexpr std::size_t target_z = 2;
    constexpr std::size_t target_tracer = 1;

    // Full 4D field: (kTotal, kTotal, nz, ntracers) in LayoutRight
    Kokkos::View<double****, Kokkos::LayoutRight> field_4d(
        "field_4d", kTotal, kTotal, nz, ntracers);

    auto h_field = Kokkos::create_mirror_view(field_4d);
    Kokkos::deep_copy(h_field, kSentinel);

    // Fill interior of the target tracer slice
    for (std::size_t i = kHalo; i < kHalo + kInterior; ++i) {
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            std::size_t local_idx = (i - kHalo) * kInterior + (j - kHalo);
            h_field(i, j, target_z, target_tracer) = encode(rank_, local_idx);
        }
    }
    Kokkos::deep_copy(field_4d, h_field);

    // Extract a 2D subview for the target tracer at a fixed z-level
    auto tracer_slice = Kokkos::subview(field_4d, Kokkos::ALL, Kokkos::ALL,
                                         target_z, target_tracer);

    // tracer_slice is a 2D strided subview (non-contiguous in general)
    // Create a plan for the 2D exchange
    std::array<std::size_t, 2> extents = {kTotal, kTotal};
    std::array<std::size_t, 2> halo_widths = {kHalo, kHalo};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors_, halo_widths, *comm_);

    // Exchange the strided subview
    halo::exchange_structured_blocking(plan, tracer_slice);

    // Verify on host
    Kokkos::deep_copy(h_field, field_4d);

    int left = neighbors_[0];
    int right = neighbors_[1];

    // Verify west halo of tracer slice
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t left_local_idx = (kInterior - 1) * kInterior + (j - kHalo);
        double expected = encode(left, left_local_idx);
        EXPECT_DOUBLE_EQ(h_field(0, j, target_z, target_tracer), expected)
            << "Strided: West halo mismatch at (0, " << j << ") rank " << rank_;
    }

    // Verify east halo of tracer slice
    for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
        std::size_t right_local_idx = 0 * kInterior + (j - kHalo);
        double expected = encode(right, right_local_idx);
        EXPECT_DOUBLE_EQ(h_field(kTotal - 1, j, target_z, target_tracer), expected)
            << "Strided: East halo mismatch at (" << kTotal - 1 << ", " << j
            << ") rank " << rank_;
    }

    // Verify that OTHER tracer slices were NOT touched
    for (std::size_t t = 0; t < ntracers; ++t) {
        if (t == target_tracer) continue;
        for (std::size_t j = kHalo; j < kHalo + kInterior; ++j) {
            EXPECT_DOUBLE_EQ(h_field(0, j, target_z, t), kSentinel)
                << "Other tracer " << t << " should be untouched at (0, " << j
                << ") rank " << rank_;
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
static ::testing::Environment* const halo_mpi_env =
    ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
