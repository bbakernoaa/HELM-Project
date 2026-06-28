// ─── HALO Window_Guard Unit Tests (real MPI) ────────────────────────────────
// Feature: helm-halo-microlibrary
//
// Example-based GoogleTest unit tests for halo::Window_Guard against a REAL MPI
// runtime (run via `mpirun -np 4` by CTest). Real MPI_Win objects are created
// with MPI_Win_create over MPI_COMM_SELF so each rank operates independently
// (no cross-rank coordination needed).
//
// These complement the spy-based property tests (which verify fence-before-free
// ordering and the noexcept guarantee) by validating real window lifecycle:
// construction ownership, the handle() accessor, RMA via an active epoch closed
// on destruction, move semantics, and the MPI_WIN_NULL no-op path.
//
// Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <utility>

#include "halo/window_guard.hpp"

namespace {

// Create a real MPI_Win exposing `buffer` (length `count` ints) over
// MPI_COMM_SELF. The returned window is owned by the caller (wrap it in a
// Window_Guard). MPI_Win_create is collective, but over MPI_COMM_SELF each rank
// is its own group of one, so there is no inter-rank dependency.
MPI_Win make_self_window(int *buffer, int count) {
    MPI_Win win = MPI_WIN_NULL;
    MPI_Win_create(buffer, static_cast<MPI_Aint>(count * static_cast<int>(sizeof(int))), static_cast<int>(sizeof(int)), MPI_INFO_NULL, MPI_COMM_SELF,
                   &win);
    return win;
}

// ─── Construction takes ownership; handle() returns it (Req 3.1, 3.5) ───────
TEST(WindowGuardTest, ConstructionStoresHandle) {
    int buffer[4] = {0, 0, 0, 0};
    MPI_Win win = make_self_window(buffer, 4);
    ASSERT_NE(win, MPI_WIN_NULL);

    {
        halo::Window_Guard guard(win);
        EXPECT_EQ(guard.handle(), win);  // accessor returns the owned handle
    }  // destructor frees the window (no epoch active)
}

// ─── MPI_WIN_NULL is a no-op on destruction (Req 3.2) ───────────────────────
TEST(WindowGuardTest, NullWindowDestructionIsNoOp) {
    {
        halo::Window_Guard guard(MPI_WIN_NULL);
        EXPECT_EQ(guard.handle(), MPI_WIN_NULL);
    }  // must not call MPI_Win_free on a null handle
    SUCCEED();
}

// ─── A real RMA epoch: put-to-self, fence closes epoch on destruction (3.7) ─
// Open an epoch with MPI_Win_fence, perform an MPI_Put into our own window,
// mark the epoch active, and let the Window_Guard destructor issue the closing
// MPI_Win_fence(0) before MPI_Win_free. The data must be visible afterward.
TEST(WindowGuardTest, ActiveEpochFenceClosesAndDataLands) {
    int buffer[4] = {0, 0, 0, 0};
    MPI_Win win = make_self_window(buffer, 4);
    ASSERT_NE(win, MPI_WIN_NULL);

    int source[4] = {10, 20, 30, 40};

    {
        halo::Window_Guard guard(win);

        MPI_Win_fence(0, win);  // open access epoch
        // Put source -> our own rank-0-of-SELF window.
        MPI_Put(source, 4, MPI_INT, /*target_rank=*/0, /*target_disp=*/0, 4, MPI_INT, win);
        guard.set_epoch_active(true);  // destructor will fence(0) before free
    }  // ~Window_Guard: MPI_Win_fence(0) completes the Put, then MPI_Win_free

    // The Put has been completed by the closing fence; the window buffer holds
    // the transferred values.
    EXPECT_EQ(buffer[0], 10);
    EXPECT_EQ(buffer[1], 20);
    EXPECT_EQ(buffer[2], 30);
    EXPECT_EQ(buffer[3], 40);
}

// ─── Explicitly closing the epoch before destruction also works ─────────────
TEST(WindowGuardTest, ManuallyClosedEpochThenPlainFree) {
    int buffer[2] = {0, 0};
    MPI_Win win = make_self_window(buffer, 2);
    ASSERT_NE(win, MPI_WIN_NULL);

    int source[2] = {7, 8};
    {
        halo::Window_Guard guard(win);
        MPI_Win_fence(0, win);
        MPI_Put(source, 2, MPI_INT, 0, 0, 2, MPI_INT, win);
        MPI_Win_fence(0, win);          // close epoch ourselves
        guard.set_epoch_active(false);  // destructor just frees, no extra fence
    }
    EXPECT_EQ(buffer[0], 7);
    EXPECT_EQ(buffer[1], 8);
}

// ─── Move construction transfers ownership; source becomes NULL (Req 3.3) ───
TEST(WindowGuardTest, MoveConstructionNullifiesSource) {
    int buffer[1] = {0};
    MPI_Win win = make_self_window(buffer, 1);
    ASSERT_NE(win, MPI_WIN_NULL);

    halo::Window_Guard src(win);
    halo::Window_Guard dst(std::move(src));

    EXPECT_EQ(dst.handle(), win);           // destination owns the window
    EXPECT_EQ(src.handle(), MPI_WIN_NULL);  // source nullified
}  // only dst frees; src destruction is a no-op

// ─── Move assignment transfers ownership; source becomes NULL (Req 3.3) ─────
TEST(WindowGuardTest, MoveAssignmentNullifiesSource) {
    int buffer[1] = {0};
    MPI_Win win = make_self_window(buffer, 1);
    ASSERT_NE(win, MPI_WIN_NULL);

    halo::Window_Guard src(win);
    halo::Window_Guard dst(MPI_WIN_NULL);
    dst = std::move(src);

    EXPECT_EQ(dst.handle(), win);
    EXPECT_EQ(src.handle(), MPI_WIN_NULL);
}

// ─── set_epoch_active toggling is safe and does not change the handle ───────
TEST(WindowGuardTest, SetEpochActiveDoesNotAffectHandle) {
    int buffer[1] = {0};
    MPI_Win win = make_self_window(buffer, 1);
    ASSERT_NE(win, MPI_WIN_NULL);

    halo::Window_Guard guard(win);
    guard.set_epoch_active(true);
    guard.set_epoch_active(false);
    EXPECT_EQ(guard.handle(), win);

    // No epoch is actually open (we toggled back to false), so plain free.
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
