// ─── HALO Communicator Unit Tests (real multi-rank MPI) ─────────────────────
// Feature: helm-halo-microlibrary
//
// Example-based GoogleTest unit tests for halo::Communicator, exercised against
// a REAL MPI runtime (run via `mpirun -np 4` by CTest). These complement the
// single-rank spy-based property tests by validating actual MPI behavior:
// rank/size queries, collective duplicate()/split(), MPI_UNDEFINED handling,
// and move semantics.
//
// Collective operations (duplicate, split) are called by every rank in the same
// deterministic order, so they line up correctly across the 4-rank job. An
// assertion failure on any rank makes that process exit non-zero, which mpirun
// (and therefore CTest) reports as a failure.
//
// Requirements: 1.1, 1.3, 1.5, 4.1, 4.2, 4.3, 4.4, 4.6
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <utility>

#include "halo/communicator.hpp"

namespace {

// ─── Construction + handle accessor (Req 1.1, 1.5) ──────────────────────────
TEST(CommunicatorTest, WrapsWorldAndReturnsHandle) {
    halo::Communicator comm(MPI_COMM_WORLD);
    EXPECT_EQ(comm.handle(), MPI_COMM_WORLD);
}

// ─── rank() and size() agree with raw MPI (Req 4.3, 4.4) ────────────────────
TEST(CommunicatorTest, RankAndSizeMatchRawMpi) {
    halo::Communicator comm(MPI_COMM_WORLD);

    int raw_rank = -1;
    int raw_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &raw_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &raw_size);

    EXPECT_EQ(comm.rank(), raw_rank);
    EXPECT_EQ(comm.size(), raw_size);
    EXPECT_GE(comm.rank(), 0);
    EXPECT_LT(comm.rank(), comm.size());
}

// ─── duplicate() yields an independent comm with identical membership (Req 4.2)
TEST(CommunicatorTest, DuplicatePreservesRankAndSize) {
    halo::Communicator world(MPI_COMM_WORLD);
    halo::Communicator dup = world.duplicate();  // collective

    EXPECT_NE(dup.handle(), MPI_COMM_NULL);
    EXPECT_NE(dup.handle(), MPI_COMM_WORLD);  // a genuinely new context
    EXPECT_EQ(dup.rank(), world.rank());
    EXPECT_EQ(dup.size(), world.size());
}

// ─── split() partitions into sub-communicators of the right size (Req 4.1) ──
TEST(CommunicatorTest, SplitByEvenOddColorHasCorrectSize) {
    halo::Communicator world(MPI_COMM_WORLD);
    const int rank = world.rank();
    const int size = world.size();

    const int color = rank % 2;
    halo::Communicator sub = world.split(color, /*key=*/rank);  // collective
    EXPECT_NE(sub.handle(), MPI_COMM_NULL);

    int expected = 0;
    for (int r = 0; r < size; ++r) {
        if (r % 2 == color) ++expected;
    }
    EXPECT_EQ(sub.size(), expected);

    // Within the sub-communicator the keys equal the world ranks, so ordering
    // is preserved: this rank's sub-rank counts the same-color ranks below it.
    int expected_sub_rank = 0;
    for (int r = 0; r < rank; ++r) {
        if (r % 2 == color) ++expected_sub_rank;
    }
    EXPECT_EQ(sub.rank(), expected_sub_rank);
}

// ─── split() with MPI_UNDEFINED yields a NULL communicator (Req 4.6) ────────
TEST(CommunicatorTest, SplitUndefinedYieldsNullComm) {
    halo::Communicator world(MPI_COMM_WORLD);
    halo::Communicator empty = world.split(MPI_UNDEFINED, 0);  // collective
    EXPECT_EQ(empty.handle(), MPI_COMM_NULL);
    // Destroying a NULL-holding Communicator must not call MPI_Comm_free
    // (verified by absence of crashes / MPI errors at scope exit).
}

// ─── Move construction transfers ownership and nullifies source (Req 1.3) ───
TEST(CommunicatorTest, MoveConstructionNullifiesSource) {
    halo::Communicator world(MPI_COMM_WORLD);
    halo::Communicator src = world.duplicate();  // collective; owns a real comm
    const MPI_Comm original = src.handle();
    ASSERT_NE(original, MPI_COMM_NULL);

    halo::Communicator dst(std::move(src));
    EXPECT_EQ(dst.handle(), original);          // destination assumed ownership
    EXPECT_EQ(src.handle(), MPI_COMM_NULL);     // source nullified
}

// ─── Move assignment transfers ownership and nullifies source (Req 1.3) ─────
TEST(CommunicatorTest, MoveAssignmentNullifiesSource) {
    halo::Communicator world(MPI_COMM_WORLD);
    halo::Communicator src = world.duplicate();  // collective
    const MPI_Comm original = src.handle();
    ASSERT_NE(original, MPI_COMM_NULL);

    halo::Communicator dst(MPI_COMM_NULL);
    dst = std::move(src);
    EXPECT_EQ(dst.handle(), original);
    EXPECT_EQ(src.handle(), MPI_COMM_NULL);
}

// ─── A split sub-communicator can itself be duplicated/queried ──────────────
TEST(CommunicatorTest, SubCommunicatorIsReusable) {
    halo::Communicator world(MPI_COMM_WORLD);
    halo::Communicator sub = world.split(0, world.rank());  // collective: all ranks, color 0

    // Every rank used color 0, so the sub-communicator spans the whole world.
    EXPECT_EQ(sub.size(), world.size());
    EXPECT_EQ(sub.rank(), world.rank());

    halo::Communicator sub_dup = sub.duplicate();  // collective over sub == world
    EXPECT_EQ(sub_dup.size(), world.size());
}

// ─── Global MPI environment ─────────────────────────────────────────────────
class MpiEnvironment : public ::testing::Environment {
public:
    void SetUp() override { MPI_Init(nullptr, nullptr); }
    void TearDown() override { MPI_Finalize(); }
};

}  // namespace

static ::testing::Environment* const mpi_env =
    ::testing::AddGlobalTestEnvironment(new MpiEnvironment);
