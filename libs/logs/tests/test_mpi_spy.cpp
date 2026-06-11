/// @file tests/test_mpi_spy.cpp
/// @brief Unit tests for the MPI_Spy interposition layer.
///
/// Verifies call recording, error injection, reset semantics, sequence
/// numbering, and type-based filtering.
///
/// Requirements: 13.5

#include "mpi_interposition.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <variant>

using logs::testing::Abort_Args;
using logs::testing::Allgather_Args;
using logs::testing::Comm_Rank_Args;
using logs::testing::Gather_Args;
using logs::testing::Initialized_Args;
using logs::testing::MPI_Call_Type;
using logs::testing::MPI_Spy;
using logs::testing::Query_Thread_Args;

class MpiSpyTest : public ::testing::Test {
protected:
    void SetUp() override { MPI_Spy::instance().reset(); }
    void TearDown() override { MPI_Spy::instance().reset(); }
};

// ─── 1. Default state ───────────────────────────────────────────────────────

TEST_F(MpiSpyTest, DefaultStateAfterReset) {
    auto& spy = MPI_Spy::instance();

    EXPECT_EQ(spy.rank(), 0);
    EXPECT_EQ(spy.thread_level(), MPI_THREAD_SINGLE);
    EXPECT_TRUE(spy.initialized());
    EXPECT_FALSE(spy.abort_called());
    EXPECT_EQ(spy.call_count(), 0u);
    EXPECT_TRUE(spy.calls().empty());
}

// ─── 2. record_comm_rank ────────────────────────────────────────────────────

TEST_F(MpiSpyTest, RecordCommRankSetsRankAndRecordsCall) {
    auto& spy = MPI_Spy::instance();
    spy.set_rank(42);

    int rank_out = -1;
    int err = spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);

    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(rank_out, 42);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::COMM_RANK);

    auto& args = std::get<Comm_Rank_Args>(calls[0].args);
    EXPECT_EQ(args.comm, MPI_COMM_WORLD);
    EXPECT_EQ(args.rank_out, 42);
}

// ─── 3. record_query_thread ─────────────────────────────────────────────────

TEST_F(MpiSpyTest, RecordQueryThreadSetsLevelAndRecordsCall) {
    auto& spy = MPI_Spy::instance();
    spy.set_thread_level(MPI_THREAD_MULTIPLE);

    int provided = -1;
    int err = spy.record_query_thread(&provided);

    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(provided, MPI_THREAD_MULTIPLE);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::QUERY_THREAD);

    auto& args = std::get<Query_Thread_Args>(calls[0].args);
    EXPECT_EQ(args.provided_out, MPI_THREAD_MULTIPLE);
}

// ─── 4. record_initialized ──────────────────────────────────────────────────

TEST_F(MpiSpyTest, RecordInitializedSetsFlagAndRecordsCall) {
    auto& spy = MPI_Spy::instance();

    // Default is initialized=true
    int flag = -1;
    int err = spy.record_initialized(&flag);

    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(flag, 1);

    // Switch to not initialized
    spy.set_initialized(false);
    flag = -1;
    err = spy.record_initialized(&flag);

    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(flag, 0);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::INITIALIZED);
    EXPECT_EQ(calls[1].type, MPI_Call_Type::INITIALIZED);

    auto& args0 = std::get<Initialized_Args>(calls[0].args);
    EXPECT_EQ(args0.flag_out, 1);

    auto& args1 = std::get<Initialized_Args>(calls[1].args);
    EXPECT_EQ(args1.flag_out, 0);
}

// ─── 5. record_abort ────────────────────────────────────────────────────────

TEST_F(MpiSpyTest, RecordAbortRecordsCommAndErrorCode) {
    auto& spy = MPI_Spy::instance();

    EXPECT_FALSE(spy.abort_called());

    int err = spy.record_abort(MPI_COMM_WORLD, 77);

    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_TRUE(spy.abort_called());

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::ABORT);

    auto& args = std::get<Abort_Args>(calls[0].args);
    EXPECT_EQ(args.comm, MPI_COMM_WORLD);
    EXPECT_EQ(args.errorcode, 77);
}

// ─── 6. record_gather and record_allgather ──────────────────────────────────

TEST_F(MpiSpyTest, RecordGatherRecordsCallWithCorrectArgs) {
    auto& spy = MPI_Spy::instance();

    int err = spy.record_gather(MPI_COMM_WORLD, 0, 4, 4);
    EXPECT_EQ(err, MPI_SUCCESS);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::GATHER);

    auto& args = std::get<Gather_Args>(calls[0].args);
    EXPECT_EQ(args.comm, MPI_COMM_WORLD);
    EXPECT_EQ(args.root, 0);
    EXPECT_EQ(args.sendcount, 4);
    EXPECT_EQ(args.recvcount, 4);
}

TEST_F(MpiSpyTest, RecordAllgatherRecordsCallWithCorrectArgs) {
    auto& spy = MPI_Spy::instance();

    int err = spy.record_allgather(MPI_COMM_WORLD, 8, 8);
    EXPECT_EQ(err, MPI_SUCCESS);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].type, MPI_Call_Type::ALLGATHER);

    auto& args = std::get<Allgather_Args>(calls[0].args);
    EXPECT_EQ(args.comm, MPI_COMM_WORLD);
    EXPECT_EQ(args.sendcount, 8);
    EXPECT_EQ(args.recvcount, 8);
}

// ─── 7. Error injection ─────────────────────────────────────────────────────

TEST_F(MpiSpyTest, ErrorInjectionReturnsConfiguredErrorCode) {
    auto& spy = MPI_Spy::instance();
    spy.set_comm_rank_error(MPI_ERR_COMM);

    int rank_out = 99;
    int err = spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);

    EXPECT_EQ(err, MPI_ERR_COMM);
    // rank_out should NOT be modified when error is injected
    EXPECT_EQ(rank_out, 99);

    // The call is still recorded
    EXPECT_EQ(spy.call_count(), 1u);
}

TEST_F(MpiSpyTest, ErrorInjectionQueryThread) {
    auto& spy = MPI_Spy::instance();
    spy.set_query_thread_error(MPI_ERR_OTHER);

    int provided = 99;
    int err = spy.record_query_thread(&provided);

    EXPECT_EQ(err, MPI_ERR_OTHER);
    // provided should NOT be modified when error is injected
    EXPECT_EQ(provided, 99);
}

TEST_F(MpiSpyTest, ErrorInjectionGather) {
    auto& spy = MPI_Spy::instance();
    spy.set_gather_error(MPI_ERR_COMM);

    int err = spy.record_gather(MPI_COMM_WORLD, 0, 1, 1);
    EXPECT_EQ(err, MPI_ERR_COMM);
    EXPECT_EQ(spy.call_count(), 1u);
}

TEST_F(MpiSpyTest, ErrorInjectionAllgather) {
    auto& spy = MPI_Spy::instance();
    spy.set_allgather_error(MPI_ERR_COMM);

    int err = spy.record_allgather(MPI_COMM_WORLD, 1, 1);
    EXPECT_EQ(err, MPI_ERR_COMM);
    EXPECT_EQ(spy.call_count(), 1u);
}

// ─── 8. Sequence numbers ────────────────────────────────────────────────────

TEST_F(MpiSpyTest, SequenceNumbersAreMonotonicallyIncreasing) {
    auto& spy = MPI_Spy::instance();

    int rank_out = 0;
    int provided = 0;
    int flag = 0;

    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_query_thread(&provided);
    spy.record_initialized(&flag);
    spy.record_abort(MPI_COMM_WORLD, 1);
    spy.record_gather(MPI_COMM_WORLD, 0, 1, 1);
    spy.record_allgather(MPI_COMM_WORLD, 1, 1);

    auto calls = spy.calls();
    ASSERT_EQ(calls.size(), 6u);

    for (std::size_t i = 0; i < calls.size(); ++i) {
        EXPECT_EQ(calls[i].sequence, i)
            << "Call at index " << i << " has unexpected sequence number";
    }
}

// ─── 9. reset() ─────────────────────────────────────────────────────────────

TEST_F(MpiSpyTest, ResetClearsAllCallsAndResetsConfiguration) {
    auto& spy = MPI_Spy::instance();

    // Configure non-default state
    spy.set_rank(99);
    spy.set_thread_level(MPI_THREAD_MULTIPLE);
    spy.set_initialized(false);
    spy.set_comm_rank_error(MPI_ERR_COMM);
    spy.set_query_thread_error(MPI_ERR_OTHER);
    spy.set_gather_error(MPI_ERR_COMM);
    spy.set_allgather_error(MPI_ERR_COMM);

    // Record some calls
    int rank_out = 0;
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_abort(MPI_COMM_WORLD, 1);

    ASSERT_GT(spy.call_count(), 0u);
    ASSERT_TRUE(spy.abort_called());

    // Reset
    spy.reset();

    // Verify all state returns to defaults
    EXPECT_EQ(spy.rank(), 0);
    EXPECT_EQ(spy.thread_level(), MPI_THREAD_SINGLE);
    EXPECT_TRUE(spy.initialized());
    EXPECT_FALSE(spy.abort_called());
    EXPECT_EQ(spy.call_count(), 0u);
    EXPECT_TRUE(spy.calls().empty());

    // Verify error injection is cleared (record_comm_rank should succeed)
    rank_out = -1;
    int err = spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    EXPECT_EQ(err, MPI_SUCCESS);
    EXPECT_EQ(rank_out, 0); // default rank
}

// ─── 10. calls_of_type() ────────────────────────────────────────────────────

TEST_F(MpiSpyTest, CallsOfTypeFiltersCorrectly) {
    auto& spy = MPI_Spy::instance();

    int rank_out = 0;
    int provided = 0;
    int flag = 0;

    // Record a mix of call types
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_query_thread(&provided);
    spy.record_initialized(&flag);
    spy.record_abort(MPI_COMM_WORLD, 1);
    spy.record_gather(MPI_COMM_WORLD, 0, 1, 1);
    spy.record_allgather(MPI_COMM_WORLD, 1, 1);
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);

    // Total calls
    EXPECT_EQ(spy.call_count(), 8u);

    // Filter by COMM_RANK
    auto comm_rank_calls = spy.calls_of_type(MPI_Call_Type::COMM_RANK);
    EXPECT_EQ(comm_rank_calls.size(), 3u);
    for (const auto& c : comm_rank_calls) {
        EXPECT_EQ(c.type, MPI_Call_Type::COMM_RANK);
    }

    // Filter by QUERY_THREAD
    auto query_calls = spy.calls_of_type(MPI_Call_Type::QUERY_THREAD);
    EXPECT_EQ(query_calls.size(), 1u);

    // Filter by INITIALIZED
    auto init_calls = spy.calls_of_type(MPI_Call_Type::INITIALIZED);
    EXPECT_EQ(init_calls.size(), 1u);

    // Filter by ABORT
    auto abort_calls = spy.calls_of_type(MPI_Call_Type::ABORT);
    EXPECT_EQ(abort_calls.size(), 1u);

    // Filter by GATHER
    auto gather_calls = spy.calls_of_type(MPI_Call_Type::GATHER);
    EXPECT_EQ(gather_calls.size(), 1u);

    // Filter by ALLGATHER
    auto allgather_calls = spy.calls_of_type(MPI_Call_Type::ALLGATHER);
    EXPECT_EQ(allgather_calls.size(), 1u);
}

TEST_F(MpiSpyTest, CallCountByTypeMatchesCallsOfType) {
    auto& spy = MPI_Spy::instance();

    int rank_out = 0;
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);
    spy.record_abort(MPI_COMM_WORLD, 1);

    EXPECT_EQ(spy.call_count(MPI_Call_Type::COMM_RANK), 2u);
    EXPECT_EQ(spy.call_count(MPI_Call_Type::ABORT), 1u);
    EXPECT_EQ(spy.call_count(MPI_Call_Type::GATHER), 0u);
    EXPECT_EQ(spy.call_count(MPI_Call_Type::ALLGATHER), 0u);
    EXPECT_EQ(spy.call_count(MPI_Call_Type::QUERY_THREAD), 0u);
    EXPECT_EQ(spy.call_count(MPI_Call_Type::INITIALIZED), 0u);
}
