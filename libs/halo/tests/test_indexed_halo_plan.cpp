// ─── HALO Indexed_Halo_Plan Unit Tests (real MPI Communicator) ──────────────
// Feature: cpp-dycore-halo-exchange
//
// Example-based GoogleTest unit tests for halo::Indexed_Halo_Plan, exercised
// against a REAL MPI runtime (run via `mpirun` by CTest). These validate the
// plan's construction-time rank validation, its acceptance of empty index
// lists, and the element-kind accessor, using a live Communicator over
// MPI_COMM_WORLD.
//
// Requirements: 1.2, 1.4, 1.5
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

using halo::Element_Kind;
using halo::Indexed_Halo_Plan;
using halo::Indexed_Neighbor;

// Test fixture supplying a live Communicator and its rank/size.
class IndexedHaloPlanTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        size_ = comm_->size();
    }
    std::unique_ptr<halo::Communicator> comm_;
    int size_{0};
};

// ─── Req 1.2: invalid rank rejected, error names the offending rank ─────────

// A rank equal to the communicator size is out of the valid range [0, size).
TEST_F(IndexedHaloPlanTest, RejectsSendRankEqualToSizeAndNamesRank) {
    std::vector<Indexed_Neighbor> bad_send{Indexed_Neighbor{size_, {{0, 1, 2}}}};
    try {
        Indexed_Halo_Plan plan(*comm_, Element_Kind::cell, bad_send, {});
        FAIL() << "expected std::invalid_argument for out-of-range send rank";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        // Message must identify the invalid rank value.
        EXPECT_NE(msg.find(std::to_string(size_)), std::string::npos)
            << "error message should name the invalid rank " << size_ << ": " << msg;
    }
}

// A negative rank is out of the valid range and must be rejected by name.
TEST_F(IndexedHaloPlanTest, RejectsNegativeRecvRankAndNamesRank) {
    std::vector<Indexed_Neighbor> bad_recv{Indexed_Neighbor{-1, {{0}}}};
    try {
        Indexed_Halo_Plan plan(*comm_, Element_Kind::edge, {}, bad_recv);
        FAIL() << "expected std::invalid_argument for negative recv rank";
    } catch (const std::invalid_argument &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("-1"), std::string::npos)
            << "error message should name the invalid rank -1: " << msg;
    }
}

// The exception thrown for an out-of-range rank is std::invalid_argument.
TEST_F(IndexedHaloPlanTest, RejectsInvalidRankThrowsInvalidArgument) {
    std::vector<Indexed_Neighbor> bad_send{Indexed_Neighbor{size_ + 5, {{0}}}};
    EXPECT_THROW(Indexed_Halo_Plan(*comm_, Element_Kind::vertex, bad_send, {}),
                 std::invalid_argument);
}

// ─── Req 1.4: empty send/recv index lists accepted as valid ─────────────────

// A plan with no neighbors at all is valid with zero totals.
TEST_F(IndexedHaloPlanTest, EmptyPlanIsValidWithZeroTotals) {
    EXPECT_NO_THROW(Indexed_Halo_Plan(*comm_, Element_Kind::cell, {}, {}));

    Indexed_Halo_Plan plan(*comm_, Element_Kind::cell, {}, {});
    EXPECT_EQ(plan.num_send_neighbors(), 0u);
    EXPECT_EQ(plan.num_recv_neighbors(), 0u);
    EXPECT_EQ(plan.total_send_indices(), 0u);
    EXPECT_EQ(plan.total_recv_indices(), 0u);
}

// A neighbor whose layers are entirely empty is valid and contributes zero
// exchanged elements for that neighbor.
TEST_F(IndexedHaloPlanTest, NeighborWithEmptyIndexListsIsValidWithZeroElements) {
    // Own rank (0) is a valid neighbor rank in the [0, size) range and lets this
    // test run on a single rank. The layer lists are present but empty.
    std::vector<Indexed_Neighbor> send{Indexed_Neighbor{0, {{}, {}}}};
    std::vector<Indexed_Neighbor> recv{Indexed_Neighbor{0, {{}}}};

    EXPECT_NO_THROW(Indexed_Halo_Plan(*comm_, Element_Kind::cell, send, recv));

    Indexed_Halo_Plan plan(*comm_, Element_Kind::cell, send, recv);
    EXPECT_EQ(plan.num_send_neighbors(), 1u);
    EXPECT_EQ(plan.num_recv_neighbors(), 1u);
    EXPECT_EQ(plan.total_send_indices(), 0u);
    EXPECT_EQ(plan.total_recv_indices(), 0u);
    // The empty send neighbor exchanges zero elements.
    EXPECT_EQ(plan.send_info()[0].total_indices(), 0u);
    EXPECT_EQ(plan.recv_info()[0].total_indices(), 0u);
}

// A neighbor with no layers at all (empty layers vector) is also valid.
TEST_F(IndexedHaloPlanTest, NeighborWithNoLayersIsValid) {
    std::vector<Indexed_Neighbor> send{Indexed_Neighbor{0, {}}};
    EXPECT_NO_THROW(Indexed_Halo_Plan(*comm_, Element_Kind::generic, send, {}));

    Indexed_Halo_Plan plan(*comm_, Element_Kind::generic, send, {});
    EXPECT_EQ(plan.num_send_neighbors(), 1u);
    EXPECT_EQ(plan.total_send_indices(), 0u);
}

// ─── Req 1.5: element-kind accessor returns the supplied kind ───────────────

TEST_F(IndexedHaloPlanTest, ElementKindAccessorReturnsSuppliedKindCell) {
    Indexed_Halo_Plan plan(*comm_, Element_Kind::cell, {}, {});
    EXPECT_EQ(plan.element_kind(), Element_Kind::cell);
}

TEST_F(IndexedHaloPlanTest, ElementKindAccessorReturnsSuppliedKindEdge) {
    Indexed_Halo_Plan plan(*comm_, Element_Kind::edge, {}, {});
    EXPECT_EQ(plan.element_kind(), Element_Kind::edge);
}

TEST_F(IndexedHaloPlanTest, ElementKindAccessorReturnsSuppliedKindVertex) {
    Indexed_Halo_Plan plan(*comm_, Element_Kind::vertex, {}, {});
    EXPECT_EQ(plan.element_kind(), Element_Kind::vertex);
}

TEST_F(IndexedHaloPlanTest, ElementKindAccessorReturnsSuppliedKindGeneric) {
    Indexed_Halo_Plan plan(*comm_, Element_Kind::generic, {}, {});
    EXPECT_EQ(plan.element_kind(), Element_Kind::generic);
}

// ─── Global MPI environment ─────────────────────────────────────────────────
class MpiEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        MPI_Init(nullptr, nullptr);
    }
    void TearDown() override {
        MPI_Finalize();
    }
};

}  // namespace

static ::testing::Environment *const mpi_env = ::testing::AddGlobalTestEnvironment(new MpiEnvironment);
