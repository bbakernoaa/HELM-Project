// ─── HALO Request_Guard Unit Tests (real MPI) ───────────────────────────────
// Feature: helm-halo-microlibrary
//
// Example-based GoogleTest unit tests for halo::Request_Guard against a REAL MPI
// runtime (run via `mpirun -np 4` by CTest). To keep every test independent and
// free of cross-rank coordination, the pending operations are posted on
// MPI_COMM_SELF: each rank sends to and receives from itself, so the requests
// genuinely complete without relying on neighbor ranks.
//
// These complement the spy-based property tests (which verify destructor call
// sequences) by validating real completion semantics: construction ownership
// transfer, wait(), test(), the handle() accessor, the empty/no-op path, and
// move semantics.
//
// Requirements: 2.1, 2.2, 2.3, 2.5, 2.6, 2.8, 8.2
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <utility>

#include "halo/request_guard.hpp"

namespace {

// Post a self Isend/Irecv pair on MPI_COMM_SELF and return the raw requests.
// The caller owns completion of both. `send_val` is transferred into `recv_out`.
void post_self_pair(int send_val, int& recv_out,
                    MPI_Request& send_req, MPI_Request& recv_req) {
    static thread_local int send_buf;  // must outlive the non-blocking op
    send_buf = send_val;
    recv_out = -1;
    MPI_Irecv(&recv_out, 1, MPI_INT, 0, /*tag=*/42, MPI_COMM_SELF, &recv_req);
    MPI_Isend(&send_buf, 1, MPI_INT, 0, /*tag=*/42, MPI_COMM_SELF, &send_req);
}

// ─── Construction takes ownership and nullifies the source (Req 2.1) ────────
TEST(RequestGuardTest, ConstructionNullifiesSourceRequest) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(7, recv, send_req, recv_req);

    halo::Request_Guard recv_guard(recv_req);
    halo::Request_Guard send_guard(send_req);

    // Ownership transferred: the source handles are now MPI_REQUEST_NULL.
    EXPECT_EQ(recv_req, MPI_REQUEST_NULL);
    EXPECT_EQ(send_req, MPI_REQUEST_NULL);

    // Complete both so nothing dangles.
    recv_guard.wait();
    send_guard.wait();
}

// ─── wait() completes the operation and transfers data (Req 2.6) ────────────
TEST(RequestGuardTest, WaitCompletesOperation) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(123, recv, send_req, recv_req);

    halo::Request_Guard recv_guard(recv_req);
    halo::Request_Guard send_guard(send_req);

    recv_guard.wait();
    send_guard.wait();

    EXPECT_EQ(recv, 123);  // the self-send delivered the value

    // A second wait() on a now-completed (null) guard is a harmless no-op.
    EXPECT_NO_THROW(recv_guard.wait());
}

// ─── test() reports completion (Req 2.5) ────────────────────────────────────
TEST(RequestGuardTest, TestReportsCompletion) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(55, recv, send_req, recv_req);

    halo::Request_Guard recv_guard(recv_req);
    halo::Request_Guard send_guard(send_req);

    // Poll both to completion. test() returns true once the op finishes (and
    // returns true immediately thereafter since the handle becomes null).
    int spins = 0;
    while (!(recv_guard.test() && send_guard.test())) {
        ++spins;
        ASSERT_LT(spins, 10'000'000) << "self requests failed to complete";
    }
    EXPECT_EQ(recv, 55);
}

// ─── Default-constructed guard: test() true, wait() no-op (Req 2.8) ─────────
TEST(RequestGuardTest, DefaultConstructedIsEmptyNoOp) {
    halo::Request_Guard empty;
    EXPECT_TRUE(empty.test());        // null handle => already "complete"
    EXPECT_NO_THROW(empty.wait());    // null handle => no-op
}

// ─── handle() exposes the internal request pointer (Req 8.2) ────────────────
TEST(RequestGuardTest, HandleAccessorReturnsNonNullPointer) {
    halo::Request_Guard empty;
    MPI_Request* p = empty.handle();
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(*p, MPI_REQUEST_NULL);  // empty guard points at a null request

    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(9, recv, send_req, recv_req);
    halo::Request_Guard recv_guard(recv_req);
    halo::Request_Guard send_guard(send_req);

    MPI_Request* live = recv_guard.handle();
    ASSERT_NE(live, nullptr);
    EXPECT_NE(*live, MPI_REQUEST_NULL);  // owns a live, pending request

    recv_guard.wait();
    send_guard.wait();
}

// ─── Move construction transfers ownership; source becomes empty (Req 2.3) ──
TEST(RequestGuardTest, MoveConstructionTransfersOwnership) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(31, recv, send_req, recv_req);

    halo::Request_Guard src(recv_req);
    halo::Request_Guard send_guard(send_req);

    halo::Request_Guard dst(std::move(src));
    // Moved-from source is now empty: test() true, handle() points at null.
    EXPECT_TRUE(src.test());
    EXPECT_EQ(*src.handle(), MPI_REQUEST_NULL);

    dst.wait();          // destination owns and completes the recv
    send_guard.wait();
    EXPECT_EQ(recv, 31);
}

// ─── Move assignment transfers ownership (Req 2.3) ──────────────────────────
TEST(RequestGuardTest, MoveAssignmentTransfersOwnership) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(64, recv, send_req, recv_req);

    halo::Request_Guard src(recv_req);
    halo::Request_Guard send_guard(send_req);

    halo::Request_Guard dst;
    dst = std::move(src);
    EXPECT_TRUE(src.test());  // source emptied

    dst.wait();
    send_guard.wait();
    EXPECT_EQ(recv, 64);
}

// ─── Destructor completes a pending op on normal scope exit (Req 2.2) ───────
// The recv_guard goes out of scope normally; its destructor calls MPI_Wait,
// which must complete cleanly (the matching self-send is also waited on). A
// leak or hang here would fail the test via timeout.
TEST(RequestGuardTest, DestructorCompletesOnNormalExit) {
    int recv = 0;
    MPI_Request send_req, recv_req;
    post_self_pair(88, recv, send_req, recv_req);

    {
        halo::Request_Guard recv_guard(recv_req);
        halo::Request_Guard send_guard(send_req);
        // Both guards destruct here via normal scope exit -> MPI_Wait on each.
    }
    EXPECT_EQ(recv, 88);  // data delivered before the destructors returned
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
