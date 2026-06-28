// ─── HALO Halo_Plan Unit Tests (real multi-rank MPI) ────────────────────────
// Feature: helm-halo-microlibrary
//
// Example-based GoogleTest unit tests for halo::Halo_Plan, exercised against a
// REAL MPI runtime (run via `mpirun -np 4` by CTest). These validate the
// precomputed-topology accessors, rank validation, empty plans, and copy/move
// semantics using a live Communicator over MPI_COMM_WORLD.
//
// Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <stdexcept>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/halo_plan.hpp"

namespace {

using halo::Neighbor_Info;

// Test fixture supplying a live Communicator and its rank/size.
class HaloPlanTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        size_ = comm_->size();
    }
    std::unique_ptr<halo::Communicator> comm_;
    int size_{0};
};

// ─── Construction round-trip: accessors echo the inputs (Req 5.1, 5.2, 5.3) ─
TEST_F(HaloPlanTest, StoresNeighborsInOrderWithCounts) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::vector<Neighbor_Info> send{{0, 10}, {1, 20}};
    std::vector<Neighbor_Info> recv{{1, 5}};
    halo::Halo_Plan plan(*comm_, send, recv);

    ASSERT_EQ(plan.num_send_neighbors(), 2u);
    ASSERT_EQ(plan.num_recv_neighbors(), 1u);

    const auto s = plan.send_info();
    EXPECT_EQ(s[0].rank, 0);
    EXPECT_EQ(s[0].count, 10u);
    EXPECT_EQ(s[1].rank, 1);
    EXPECT_EQ(s[1].count, 20u);

    const auto r = plan.recv_info();
    EXPECT_EQ(r[0].rank, 1);
    EXPECT_EQ(r[0].count, 5u);
}

// ─── Totals are precomputed sums (Req 5.6, 5.7 via totals) ──────────────────
TEST_F(HaloPlanTest, TotalElementsAreSums) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::vector<Neighbor_Info> send{{0, 10}, {1, 20}};
    std::vector<Neighbor_Info> recv{{0, 3}, {1, 4}};
    halo::Halo_Plan plan(*comm_, send, recv);

    EXPECT_EQ(plan.total_send_elements(), 30u);
    EXPECT_EQ(plan.total_recv_elements(), 7u);
}

// ─── Communicator reference is the one provided ─────────────────────────────
TEST_F(HaloPlanTest, CommunicatorAccessorReturnsProvidedComm) {
    halo::Halo_Plan plan(*comm_, {}, {});
    EXPECT_EQ(plan.communicator().handle(), comm_->handle());
}

// ─── Empty plan is valid with zero neighbors (Req 5.8) ──────────────────────
TEST_F(HaloPlanTest, EmptyPlanHasZeroNeighbors) {
    halo::Halo_Plan plan(*comm_, {}, {});
    EXPECT_EQ(plan.num_send_neighbors(), 0u);
    EXPECT_EQ(plan.num_recv_neighbors(), 0u);
    EXPECT_EQ(plan.total_send_elements(), 0u);
    EXPECT_EQ(plan.total_recv_elements(), 0u);
}

// ─── Out-of-range ranks are rejected (Req 5.4) ──────────────────────────────
TEST_F(HaloPlanTest, RejectsRankEqualToSize) {
    std::vector<Neighbor_Info> bad{{size_, 1}};  // size is out of [0, size)
    EXPECT_THROW(halo::Halo_Plan(*comm_, bad, {}), std::invalid_argument);
}

TEST_F(HaloPlanTest, RejectsNegativeRank) {
    std::vector<Neighbor_Info> bad{{-1, 1}};
    EXPECT_THROW(halo::Halo_Plan(*comm_, {}, bad), std::invalid_argument);
}

// ─── Duplicate ranks within a direction are rejected (Req 5.9) ──────────────
TEST_F(HaloPlanTest, RejectsDuplicateRankInSameDirection) {
    std::vector<Neighbor_Info> dup{{0, 1}, {0, 2}};
    EXPECT_THROW(halo::Halo_Plan(*comm_, dup, {}), std::invalid_argument);
}

// ─── The same rank may appear once in send and once in recv (not a dup) ─────
TEST_F(HaloPlanTest, SameRankInBothDirectionsIsAllowed) {
    std::vector<Neighbor_Info> send{{0, 1}};
    std::vector<Neighbor_Info> recv{{0, 1}};
    EXPECT_NO_THROW(halo::Halo_Plan(*comm_, send, recv));
}

// ─── Copy produces an equivalent plan (Req 5.5) ─────────────────────────────
TEST_F(HaloPlanTest, CopyIsEquivalent) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::vector<Neighbor_Info> send{{0, 10}, {1, 20}};
    std::vector<Neighbor_Info> recv{{1, 5}};
    halo::Halo_Plan original(*comm_, send, recv);
    halo::Halo_Plan copy = original;  // copy construction

    EXPECT_EQ(copy.num_send_neighbors(), original.num_send_neighbors());
    EXPECT_EQ(copy.num_recv_neighbors(), original.num_recv_neighbors());
    EXPECT_EQ(copy.total_send_elements(), original.total_send_elements());
    EXPECT_EQ(copy.total_recv_elements(), original.total_recv_elements());

    const auto a = original.send_info();
    const auto b = copy.send_info();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].rank, b[i].rank);
        EXPECT_EQ(a[i].count, b[i].count);
    }
}

// ─── Move construction transfers contents (Req 5.5) ─────────────────────────
TEST_F(HaloPlanTest, MoveTransfersContents) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::vector<Neighbor_Info> send{{0, 10}, {1, 20}};
    halo::Halo_Plan original(*comm_, send, {});
    halo::Halo_Plan moved = std::move(original);

    EXPECT_EQ(moved.num_send_neighbors(), 2u);
    EXPECT_EQ(moved.total_send_elements(), 30u);
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
