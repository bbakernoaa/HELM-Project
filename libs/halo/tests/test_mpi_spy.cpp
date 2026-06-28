// ─── HALO MPI_Spy Unit Tests ─────────────────────────────────────────────────
// Verifies the MPI interposition spy layer correctly records calls, injects
// errors, and resets state. These tests validate the test infrastructure itself
// to ensure property tests built on top of MPI_Spy are reliable.
//
// Feature: helm-halo-microlibrary
// Requirements: 11.7
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi_interposition.hpp"

using halo::testing::MPI_Call_Record;
using halo::testing::MPI_Spy;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class MPI_Spy_Test : public ::testing::Test {
   protected:
    void SetUp() override {
        MPI_Spy::instance().reset();
    }

    void TearDown() override {
        MPI_Spy::instance().reset();
    }
};

// ─── Call Recording Tests ────────────────────────────────────────────────────

TEST_F(MPI_Spy_Test, RecordsMPICommFree) {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_free(&comm);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Comm_free);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&comm));
}

TEST_F(MPI_Spy_Test, RecordsMPIRequestFree) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Request_free(&req);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Request_free);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&req));
}

TEST_F(MPI_Spy_Test, RecordsMPICancel) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Cancel(&req);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Cancel);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&req));
}

TEST_F(MPI_Spy_Test, RecordsMPIWait) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Status status;
    MPI_Wait(&req, &status);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Wait);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&req));
}

TEST_F(MPI_Spy_Test, RecordsMPITest) {
    MPI_Request req = MPI_REQUEST_NULL;
    int flag = 0;
    MPI_Status status;
    MPI_Test(&req, &flag, &status);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Test);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&req));
    // Mock always completes immediately
    EXPECT_EQ(flag, 1);
}

TEST_F(MPI_Spy_Test, RecordsMPIWinFree) {
    MPI_Win win = MPI_WIN_NULL;
    MPI_Win_free(&win);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Win_free);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(&win));
}

TEST_F(MPI_Spy_Test, RecordsMPIWinFence) {
    MPI_Win win = MPI_WIN_NULL;
    MPI_Win_fence(0, win);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Win_fence);
    EXPECT_EQ(calls[0].arg, 0);
}

TEST_F(MPI_Spy_Test, RecordsMPIWinFenceWithAssertion) {
    MPI_Win win = MPI_WIN_NULL;
    MPI_Win_fence(MPI_MODE_NOSTORE, win);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Win_fence);
    EXPECT_EQ(calls[0].arg, MPI_MODE_NOSTORE);
}

TEST_F(MPI_Spy_Test, RecordsMPIIrecv) {
    char buf[64];
    MPI_Request req;
    MPI_Irecv(buf, 64, MPI_CHAR, 2, 42, MPI_COMM_WORLD, &req);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Irecv);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(buf));
    EXPECT_EQ(calls[0].arg, 2);  // source rank recorded as arg
    // Sentinel request should be non-null
    EXPECT_NE(req, MPI_REQUEST_NULL);
}

TEST_F(MPI_Spy_Test, RecordsMPIIsend) {
    char buf[64];
    MPI_Request req;
    MPI_Isend(buf, 64, MPI_CHAR, 3, 99, MPI_COMM_WORLD, &req);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Isend);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(buf));
    EXPECT_EQ(calls[0].arg, 3);  // dest rank recorded as arg
    // Sentinel request should be non-null
    EXPECT_NE(req, MPI_REQUEST_NULL);
}

TEST_F(MPI_Spy_Test, RecordsMPIWaitall) {
    MPI_Request reqs[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Waitall);
    EXPECT_EQ(calls[0].handle, static_cast<void *>(reqs));
    EXPECT_EQ(calls[0].arg, 3);  // count recorded as arg
}

// ─── Error Injection Tests ───────────────────────────────────────────────────

TEST_F(MPI_Spy_Test, ErrorInjectionReturnsConfiguredCode) {
    MPI_Spy::instance().set_next_error(MPI_ERR_COMM);

    MPI_Comm comm = MPI_COMM_NULL;
    int result = MPI_Comm_free(&comm);

    EXPECT_EQ(result, MPI_ERR_COMM);
    // Call is still recorded
    EXPECT_EQ(MPI_Spy::instance().call_count(), 1u);
}

TEST_F(MPI_Spy_Test, ErrorInjectionConsumedAfterOneCall) {
    MPI_Spy::instance().set_next_error(MPI_ERR_REQUEST);

    MPI_Request req = MPI_REQUEST_NULL;
    int result1 = MPI_Cancel(&req);
    int result2 = MPI_Cancel(&req);

    EXPECT_EQ(result1, MPI_ERR_REQUEST);
    EXPECT_EQ(result2, MPI_SUCCESS);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForWait) {
    MPI_Spy::instance().set_next_error(MPI_ERR_IN_STATUS);

    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Status status;
    int result = MPI_Wait(&req, &status);

    EXPECT_EQ(result, MPI_ERR_IN_STATUS);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForTest) {
    MPI_Spy::instance().set_next_error(MPI_ERR_REQUEST);

    MPI_Request req = MPI_REQUEST_NULL;
    int flag = 0;
    MPI_Status status;
    int result = MPI_Test(&req, &flag, &status);

    EXPECT_EQ(result, MPI_ERR_REQUEST);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForWinFree) {
    MPI_Spy::instance().set_next_error(MPI_ERR_WIN);

    MPI_Win win = MPI_WIN_NULL;
    int result = MPI_Win_free(&win);

    EXPECT_EQ(result, MPI_ERR_WIN);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForWinFence) {
    MPI_Spy::instance().set_next_error(MPI_ERR_WIN);

    MPI_Win win = MPI_WIN_NULL;
    int result = MPI_Win_fence(0, win);

    EXPECT_EQ(result, MPI_ERR_WIN);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForIrecv) {
    MPI_Spy::instance().set_next_error(MPI_ERR_BUFFER);

    char buf[64];
    MPI_Request req;
    int result = MPI_Irecv(buf, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);

    EXPECT_EQ(result, MPI_ERR_BUFFER);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForIsend) {
    MPI_Spy::instance().set_next_error(MPI_ERR_BUFFER);

    char buf[64];
    MPI_Request req;
    int result = MPI_Isend(buf, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);

    EXPECT_EQ(result, MPI_ERR_BUFFER);
}

TEST_F(MPI_Spy_Test, ErrorInjectionWorksForWaitall) {
    MPI_Spy::instance().set_next_error(MPI_ERR_IN_STATUS);

    MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int result = MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    EXPECT_EQ(result, MPI_ERR_IN_STATUS);
}

// ─── Reset Tests ─────────────────────────────────────────────────────────────

TEST_F(MPI_Spy_Test, ResetClearsAllRecordedCalls) {
    // Record several calls
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Win win = MPI_WIN_NULL;

    MPI_Comm_free(&comm);
    MPI_Cancel(&req);
    MPI_Win_free(&win);

    ASSERT_EQ(MPI_Spy::instance().call_count(), 3u);

    // Reset
    MPI_Spy::instance().reset();

    EXPECT_EQ(MPI_Spy::instance().call_count(), 0u);
    EXPECT_TRUE(MPI_Spy::instance().calls().empty());
    EXPECT_FALSE(MPI_Spy::instance().has_calls());
}

TEST_F(MPI_Spy_Test, ResetClearsErrorInjection) {
    MPI_Spy::instance().set_next_error(MPI_ERR_COMM);
    MPI_Spy::instance().reset();

    // After reset, next call should return MPI_SUCCESS
    MPI_Comm comm = MPI_COMM_NULL;
    int result = MPI_Comm_free(&comm);

    EXPECT_EQ(result, MPI_SUCCESS);
}

// ─── Multiple Call Sequence Tests ────────────────────────────────────────────

TEST_F(MPI_Spy_Test, RecordsMultipleCallsInOrder) {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Win win = MPI_WIN_NULL;

    MPI_Comm_free(&comm);
    MPI_Cancel(&req);
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    auto const &calls = MPI_Spy::instance().calls();
    ASSERT_EQ(calls.size(), 4u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Comm_free);
    EXPECT_EQ(calls[1].type, MPI_Call_Record::Type::Cancel);
    EXPECT_EQ(calls[2].type, MPI_Call_Record::Type::Win_fence);
    EXPECT_EQ(calls[3].type, MPI_Call_Record::Type::Win_free);
}

TEST_F(MPI_Spy_Test, CountOfFiltersCorrectly) {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Request req = MPI_REQUEST_NULL;

    MPI_Comm_free(&comm);
    MPI_Comm_free(&comm);
    MPI_Cancel(&req);
    MPI_Comm_free(&comm);

    EXPECT_EQ(MPI_Spy::instance().count_of(MPI_Call_Record::Type::Comm_free), 3u);
    EXPECT_EQ(MPI_Spy::instance().count_of(MPI_Call_Record::Type::Cancel), 1u);
    EXPECT_EQ(MPI_Spy::instance().count_of(MPI_Call_Record::Type::Wait), 0u);
}

// ─── Behavioral Correctness Tests ───────────────────────────────────────────

TEST_F(MPI_Spy_Test, CommFreeNullifiesHandle) {
    // Use a non-null value to verify it gets set to MPI_COMM_NULL
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_free(&comm);
    EXPECT_EQ(comm, MPI_COMM_NULL);
}

TEST_F(MPI_Spy_Test, WaitNullifiesRequest) {
    // Simulate a pending request (use a sentinel)
    char buf[64];
    MPI_Request req;
    MPI_Irecv(buf, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);
    ASSERT_NE(req, MPI_REQUEST_NULL);

    MPI_Status status;
    MPI_Wait(&req, &status);
    EXPECT_EQ(req, MPI_REQUEST_NULL);
}

TEST_F(MPI_Spy_Test, TestSetsCompletionFlag) {
    char buf[64];
    MPI_Request req;
    MPI_Isend(buf, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &req);

    int flag = 0;
    MPI_Status status;
    MPI_Test(&req, &flag, &status);

    EXPECT_EQ(flag, 1);
    EXPECT_EQ(req, MPI_REQUEST_NULL);
}

TEST_F(MPI_Spy_Test, WaitallNullifiesAllRequests) {
    char buf[64];
    MPI_Request reqs[3];
    MPI_Irecv(buf, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(buf, 64, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(buf, 64, MPI_CHAR, 2, 0, MPI_COMM_WORLD, &reqs[2]);

    MPI_Spy::instance().reset();  // Clear recording of Irecv/Isend

    MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(reqs[i], MPI_REQUEST_NULL);
    }
}

TEST_F(MPI_Spy_Test, WinFreeNullifiesHandle) {
    MPI_Win win = reinterpret_cast<MPI_Win>(0xDEAD);
    MPI_Win_free(&win);
    EXPECT_EQ(win, MPI_WIN_NULL);
}

TEST_F(MPI_Spy_Test, RequestFreeNullifiesHandle) {
    MPI_Request req = reinterpret_cast<MPI_Request>(0xBEEF);
    MPI_Request_free(&req);
    EXPECT_EQ(req, MPI_REQUEST_NULL);
}
