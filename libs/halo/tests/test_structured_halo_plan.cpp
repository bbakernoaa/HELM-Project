#include <gtest/gtest.h>
#include <mpi.h>

#include <halo/structured_halo_plan.hpp>

// ─── 1D Plan Tests ──────────────────────────────────────────────────────────

class StructuredHaloPlan1DTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
    }
    std::unique_ptr<halo::Communicator> comm_;
};

TEST_F(StructuredHaloPlan1DTest, ConstructionAndAccessors) {
    // 1D grid: 10 cells with halo width 1 on each side
    std::array<std::size_t, 1> extents = {10};
    std::array<int, 2> neighbors = {0, 1};  // left neighbor rank 0, right rank 1
    std::array<std::size_t, 1> halo_widths = {1};

    halo::Structured_Halo_Plan<1> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_EQ(plan.num_faces(), 2);
    EXPECT_EQ(plan.extent(0), 10u);
    EXPECT_EQ(plan.halo_width(0), 1u);
    EXPECT_EQ(plan.neighbor_rank(0), 0);
    EXPECT_EQ(plan.neighbor_rank(1), 1);
    EXPECT_TRUE(plan.has_neighbor(0));
    EXPECT_TRUE(plan.has_neighbor(1));
    EXPECT_EQ(plan.active_neighbor_count(), 2);
}

TEST_F(StructuredHaloPlan1DTest, SendRecvRegions) {
    // 1D grid: 10 cells, halo width 2
    // Layout: [hh|iiiiii|hh]  (h=halo, i=interior)
    //          0  2      8  10
    std::array<std::size_t, 1> extents = {10};
    std::array<int, 2> neighbors = {0, 1};
    std::array<std::size_t, 1> halo_widths = {2};

    halo::Structured_Halo_Plan<1> plan(extents, neighbors, halo_widths, *comm_);

    // Low face (face 0): send interior cells [2, 4), recv halo [0, 2)
    auto send_lo = plan.send_region(0);
    EXPECT_EQ(send_lo.ranges[0].first, 2u);
    EXPECT_EQ(send_lo.ranges[0].second, 4u);
    EXPECT_EQ(send_lo.size(), 2u);

    auto recv_lo = plan.recv_region(0);
    EXPECT_EQ(recv_lo.ranges[0].first, 0u);
    EXPECT_EQ(recv_lo.ranges[0].second, 2u);
    EXPECT_EQ(recv_lo.size(), 2u);

    // High face (face 1): send interior cells [6, 8), recv halo [8, 10)
    auto send_hi = plan.send_region(1);
    EXPECT_EQ(send_hi.ranges[0].first, 6u);
    EXPECT_EQ(send_hi.ranges[0].second, 8u);
    EXPECT_EQ(send_hi.size(), 2u);

    auto recv_hi = plan.recv_region(1);
    EXPECT_EQ(recv_hi.ranges[0].first, 8u);
    EXPECT_EQ(recv_hi.ranges[0].second, 10u);
    EXPECT_EQ(recv_hi.size(), 2u);
}

TEST_F(StructuredHaloPlan1DTest, NonPeriodicBoundary) {
    // Non-periodic: left has no neighbor (-1)
    std::array<std::size_t, 1> extents = {10};
    std::array<int, 2> neighbors = {-1, 2};
    std::array<std::size_t, 1> halo_widths = {1};

    halo::Structured_Halo_Plan<1> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_FALSE(plan.has_neighbor(0));
    EXPECT_TRUE(plan.has_neighbor(1));
    EXPECT_EQ(plan.neighbor_rank(0), -1);
    EXPECT_EQ(plan.active_neighbor_count(), 1);
}

TEST_F(StructuredHaloPlan1DTest, ValidationThrowsOnSmallExtent) {
    // Extent 3 < 2*halo_width(2) = 4 → should throw
    std::array<std::size_t, 1> extents = {3};
    std::array<int, 2> neighbors = {0, 1};
    std::array<std::size_t, 1> halo_widths = {2};

    EXPECT_THROW(
        (halo::Structured_Halo_Plan<1>(extents, neighbors, halo_widths, *comm_)),
        std::invalid_argument);
}

TEST_F(StructuredHaloPlan1DTest, MinimalValidExtent) {
    // Extent == 2*halo_width is the minimum valid case
    std::array<std::size_t, 1> extents = {4};
    std::array<int, 2> neighbors = {0, 1};
    std::array<std::size_t, 1> halo_widths = {2};

    EXPECT_NO_THROW(
        (halo::Structured_Halo_Plan<1>(extents, neighbors, halo_widths, *comm_)));
}

// ─── 2D Plan Tests ──────────────────────────────────────────────────────────

class StructuredHaloPlan2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
    }
    std::unique_ptr<halo::Communicator> comm_;
};

TEST_F(StructuredHaloPlan2DTest, ConstructionAndFaceCount) {
    std::array<std::size_t, 2> extents = {10, 8};
    std::array<int, 4> neighbors = {0, 1, 2, 3};  // west, east, south, north
    std::array<std::size_t, 2> halo_widths = {1, 1};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_EQ(plan.num_faces(), 4);
    EXPECT_EQ(plan.extent(0), 10u);
    EXPECT_EQ(plan.extent(1), 8u);
    EXPECT_EQ(plan.active_neighbor_count(), 4);
}

TEST_F(StructuredHaloPlan2DTest, SendRecvRegionsWestEast) {
    // 2D grid: 10x8, halo width 1 per dimension
    std::array<std::size_t, 2> extents = {10, 8};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {1, 1};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    // Face 0 = low-d0 (west): dim0 slice [1,2), dim1 full [0,8)
    auto send_west = plan.send_region(0);
    EXPECT_EQ(send_west.ranges[0].first, 1u);
    EXPECT_EQ(send_west.ranges[0].second, 2u);
    EXPECT_EQ(send_west.ranges[1].first, 0u);
    EXPECT_EQ(send_west.ranges[1].second, 8u);
    EXPECT_EQ(send_west.size(), 1u * 8u);

    auto recv_west = plan.recv_region(0);
    EXPECT_EQ(recv_west.ranges[0].first, 0u);
    EXPECT_EQ(recv_west.ranges[0].second, 1u);
    EXPECT_EQ(recv_west.ranges[1].first, 0u);
    EXPECT_EQ(recv_west.ranges[1].second, 8u);
    EXPECT_EQ(recv_west.size(), 1u * 8u);

    // Face 1 = high-d0 (east): dim0 slice [8,9), dim1 full [0,8)
    auto send_east = plan.send_region(1);
    EXPECT_EQ(send_east.ranges[0].first, 8u);
    EXPECT_EQ(send_east.ranges[0].second, 9u);
    EXPECT_EQ(send_east.ranges[1].first, 0u);
    EXPECT_EQ(send_east.ranges[1].second, 8u);
    EXPECT_EQ(send_east.size(), 1u * 8u);

    auto recv_east = plan.recv_region(1);
    EXPECT_EQ(recv_east.ranges[0].first, 9u);
    EXPECT_EQ(recv_east.ranges[0].second, 10u);
}

TEST_F(StructuredHaloPlan2DTest, SendRecvRegionsSouthNorth) {
    // 2D grid: 10x8, halo width 2 per dimension
    std::array<std::size_t, 2> extents = {10, 8};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {2, 2};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    // Face 2 = low-d1 (south): dim0 full [0,10), dim1 slice [2,4)
    auto send_south = plan.send_region(2);
    EXPECT_EQ(send_south.ranges[0].first, 0u);
    EXPECT_EQ(send_south.ranges[0].second, 10u);
    EXPECT_EQ(send_south.ranges[1].first, 2u);
    EXPECT_EQ(send_south.ranges[1].second, 4u);
    EXPECT_EQ(send_south.size(), 10u * 2u);

    auto recv_south = plan.recv_region(2);
    EXPECT_EQ(recv_south.ranges[0].first, 0u);
    EXPECT_EQ(recv_south.ranges[0].second, 10u);
    EXPECT_EQ(recv_south.ranges[1].first, 0u);
    EXPECT_EQ(recv_south.ranges[1].second, 2u);

    // Face 3 = high-d1 (north): dim0 full [0,10), dim1 slice [4,6)
    auto send_north = plan.send_region(3);
    EXPECT_EQ(send_north.ranges[0].first, 0u);
    EXPECT_EQ(send_north.ranges[0].second, 10u);
    EXPECT_EQ(send_north.ranges[1].first, 4u);
    EXPECT_EQ(send_north.ranges[1].second, 6u);
    EXPECT_EQ(send_north.size(), 10u * 2u);

    auto recv_north = plan.recv_region(3);
    EXPECT_EQ(recv_north.ranges[1].first, 6u);
    EXPECT_EQ(recv_north.ranges[1].second, 8u);
}

TEST_F(StructuredHaloPlan2DTest, AsymmetricHaloWidths) {
    // Different halo widths per dimension
    std::array<std::size_t, 2> extents = {12, 8};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {3, 1};  // 3 in x, 1 in y

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    // Face 0 (low-x): send [3,6) in dim0, full [0,8) in dim1
    auto send_west = plan.send_region(0);
    EXPECT_EQ(send_west.ranges[0].first, 3u);
    EXPECT_EQ(send_west.ranges[0].second, 6u);
    EXPECT_EQ(send_west.size(), 3u * 8u);

    // Face 2 (low-y): send [0,12) in dim0, [1,2) in dim1
    auto send_south = plan.send_region(2);
    EXPECT_EQ(send_south.ranges[1].first, 1u);
    EXPECT_EQ(send_south.ranges[1].second, 2u);
    EXPECT_EQ(send_south.size(), 12u * 1u);
}

TEST_F(StructuredHaloPlan2DTest, ValidationThrowsPerDimension) {
    // dim0: 4 >= 2*2 = 4 → OK
    // dim1: 3 < 2*2 = 4 → should throw
    std::array<std::size_t, 2> extents = {4, 3};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {2, 2};

    EXPECT_THROW(
        (halo::Structured_Halo_Plan<2>(extents, neighbors, halo_widths, *comm_)),
        std::invalid_argument);
}

// ─── 3D Plan Tests ──────────────────────────────────────────────────────────

class StructuredHaloPlan3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
    }
    std::unique_ptr<halo::Communicator> comm_;
};

TEST_F(StructuredHaloPlan3DTest, SixFaceNeighbors) {
    std::array<std::size_t, 3> extents = {10, 10, 10};
    std::array<int, 6> neighbors = {0, 1, 2, 3, 4, 5};
    std::array<std::size_t, 3> halo_widths = {1, 1, 1};

    halo::Structured_Halo_Plan<3> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_EQ(plan.num_faces(), 6);
    EXPECT_EQ(plan.active_neighbor_count(), 6);

    // Face 4 = low-d2 (bottom): send dim2 [1,2), other dims full
    auto send_bot = plan.send_region(4);
    EXPECT_EQ(send_bot.ranges[0].first, 0u);
    EXPECT_EQ(send_bot.ranges[0].second, 10u);
    EXPECT_EQ(send_bot.ranges[1].first, 0u);
    EXPECT_EQ(send_bot.ranges[1].second, 10u);
    EXPECT_EQ(send_bot.ranges[2].first, 1u);
    EXPECT_EQ(send_bot.ranges[2].second, 2u);
    EXPECT_EQ(send_bot.size(), 10u * 10u * 1u);

    // Face 5 = high-d2 (top): recv dim2 [9,10)
    auto recv_top = plan.recv_region(5);
    EXPECT_EQ(recv_top.ranges[2].first, 9u);
    EXPECT_EQ(recv_top.ranges[2].second, 10u);
    EXPECT_EQ(recv_top.size(), 10u * 10u * 1u);
}

TEST_F(StructuredHaloPlan3DTest, NonPeriodicMixedBoundary) {
    // Some faces have no neighbor
    std::array<std::size_t, 3> extents = {8, 8, 8};
    std::array<int, 6> neighbors = {-1, 1, 2, -1, 4, 5};
    std::array<std::size_t, 3> halo_widths = {2, 2, 2};

    halo::Structured_Halo_Plan<3> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_FALSE(plan.has_neighbor(0));
    EXPECT_TRUE(plan.has_neighbor(1));
    EXPECT_TRUE(plan.has_neighbor(2));
    EXPECT_FALSE(plan.has_neighbor(3));
    EXPECT_TRUE(plan.has_neighbor(4));
    EXPECT_TRUE(plan.has_neighbor(5));
    EXPECT_EQ(plan.active_neighbor_count(), 4);
}

// ─── 4D Plan Tests ──────────────────────────────────────────────────────────

class StructuredHaloPlan4DTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
    }
    std::unique_ptr<halo::Communicator> comm_;
};

TEST_F(StructuredHaloPlan4DTest, EightFaceNeighbors) {
    std::array<std::size_t, 4> extents = {6, 6, 6, 6};
    std::array<int, 8> neighbors = {0, 1, 2, 3, 4, 5, 6, 7};
    std::array<std::size_t, 4> halo_widths = {1, 1, 1, 1};

    halo::Structured_Halo_Plan<4> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_EQ(plan.num_faces(), 8);
    EXPECT_EQ(plan.active_neighbor_count(), 8);

    // Face 6 = low-d3: send dim3 [1,2), others full [0,6)
    auto send_f6 = plan.send_region(6);
    EXPECT_EQ(send_f6.ranges[0].first, 0u);
    EXPECT_EQ(send_f6.ranges[0].second, 6u);
    EXPECT_EQ(send_f6.ranges[3].first, 1u);
    EXPECT_EQ(send_f6.ranges[3].second, 2u);
    EXPECT_EQ(send_f6.size(), 6u * 6u * 6u * 1u);

    // Face 7 = high-d3: recv dim3 [5,6)
    auto recv_f7 = plan.recv_region(7);
    EXPECT_EQ(recv_f7.ranges[3].first, 5u);
    EXPECT_EQ(recv_f7.ranges[3].second, 6u);
}

// ─── Bounds Checking Tests ──────────────────────────────────────────────────

class StructuredHaloPlanBoundsTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
    }
    std::unique_ptr<halo::Communicator> comm_;
};

TEST_F(StructuredHaloPlanBoundsTest, FaceIndexOutOfRange) {
    std::array<std::size_t, 2> extents = {10, 10};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {1, 1};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_THROW(plan.send_region(4), std::out_of_range);
    EXPECT_THROW(plan.send_region(-1), std::out_of_range);
    EXPECT_THROW(plan.recv_region(4), std::out_of_range);
    EXPECT_THROW(plan.neighbor_rank(4), std::out_of_range);
}

TEST_F(StructuredHaloPlanBoundsTest, DimIndexOutOfRange) {
    std::array<std::size_t, 2> extents = {10, 10};
    std::array<int, 4> neighbors = {0, 1, 2, 3};
    std::array<std::size_t, 2> halo_widths = {1, 1};

    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, *comm_);

    EXPECT_THROW(plan.extent(2), std::out_of_range);
    EXPECT_THROW(plan.halo_width(-1), std::out_of_range);
}

// ─── Layout Detection Tests ─────────────────────────────────────────────────

TEST(StructuredHaloPlanLayoutTest, DetectsLayoutLeft) {
    using view_left = Kokkos::View<double**, Kokkos::LayoutLeft>;
    EXPECT_TRUE((halo::Structured_Halo_Plan<2>::is_layout_left<view_left>()));
    EXPECT_FALSE((halo::Structured_Halo_Plan<2>::is_layout_right<view_left>()));
}

TEST(StructuredHaloPlanLayoutTest, DetectsLayoutRight) {
    using view_right = Kokkos::View<double**, Kokkos::LayoutRight>;
    EXPECT_FALSE((halo::Structured_Halo_Plan<2>::is_layout_left<view_right>()));
    EXPECT_TRUE((halo::Structured_Halo_Plan<2>::is_layout_right<view_right>()));
}

// ─── Communicator Reference Test ────────────────────────────────────────────

TEST(StructuredHaloPlanCommTest, StoresNonOwningReference) {
    halo::Communicator comm(MPI_COMM_WORLD);

    std::array<std::size_t, 1> extents = {10};
    std::array<int, 2> neighbors = {0, 1};
    std::array<std::size_t, 1> halo_widths = {1};

    halo::Structured_Halo_Plan<1> plan(extents, neighbors, halo_widths, comm);

    // The communicator reference should point to the same object
    EXPECT_EQ(&plan.communicator(), &comm);
}
