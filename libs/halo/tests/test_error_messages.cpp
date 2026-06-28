/// @file test_error_messages.cpp
/// @brief Unit tests for enhanced error messages with rank/comm/plan context.
///
/// Validates Requirements 11.1, 11.2, 11.3:
/// - Error messages include local rank and communicator name
/// - Error messages include plan neighbor summary
/// - Fortran interop writes context to stderr before abort

#include <gtest/gtest.h>
#include <mpi.h>

#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <stdexcept>
#include <string>

class ErrorMessageTest : public ::testing::Test {
   protected:
    void SetUp() override {
        halo::Environment::initialize();
        // Ensure throw_on_error policy so tests can catch exceptions.
        halo::Environment::set_error_policy(halo::ErrorPolicy::throw_on_error);
    }

    void TearDown() override {
        halo::Environment::set_error_policy(halo::ErrorPolicy::throw_on_error);
    }
};

/// Requirement 11.1: Error messages include local rank.
TEST_F(ErrorMessageTest, MessageIncludesLocalRank) {
    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    try {
        halo::detail::handle_mpi_error(MPI_ERR_RANK, 42, "MPI_Isend");
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        // Verify the local rank is present in the message.
        std::string rank_prefix = "[rank " + std::to_string(my_rank) + "]";
        EXPECT_NE(msg.find(rank_prefix), std::string::npos) << "Expected local rank prefix in: " << msg;
        // Verify the operation name is present.
        EXPECT_NE(msg.find("MPI_Isend"), std::string::npos) << "Expected operation name in: " << msg;
        // Verify the neighbor rank is present.
        EXPECT_NE(msg.find("rank 42"), std::string::npos) << "Expected neighbor rank in: " << msg;
        return;
    }
    FAIL() << "Expected std::runtime_error to be thrown";
}

/// Requirement 11.1: Error messages include communicator name when set.
TEST_F(ErrorMessageTest, MessageIncludesCommName) {
    // Duplicate COMM_WORLD and set a name on it.
    MPI_Comm named_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &named_comm);
    MPI_Comm_set_name(named_comm, "test_comm_alpha");

    try {
        halo::detail::handle_mpi_error(MPI_ERR_RANK, 7, "MPI_Irecv", named_comm);
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        // Verify communicator name is in the message.
        EXPECT_NE(msg.find("test_comm_alpha"), std::string::npos) << "Expected comm name in: " << msg;
        MPI_Comm_free(&named_comm);
        return;
    }
    MPI_Comm_free(&named_comm);
    FAIL() << "Expected std::runtime_error to be thrown";
}

/// Requirement 11.1: No comm name appended when comm is MPI_COMM_NULL.
TEST_F(ErrorMessageTest, NullCommOmitsCommName) {
    try {
        halo::detail::handle_mpi_error(MPI_ERR_RANK, 5, "MPI_Isend", MPI_COMM_NULL);
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        // Should not have comm= field when comm is NULL.
        EXPECT_EQ(msg.find("comm="), std::string::npos) << "Did not expect comm name field for NULL comm in: " << msg;
        return;
    }
    FAIL() << "Expected std::runtime_error to be thrown";
}

/// Requirement 11.2: Error messages include plan neighbor summary.
TEST_F(ErrorMessageTest, MessageIncludesPlanNeighborSummary) {
    int send_ranks[] = {1, 2, 3};
    int recv_ranks[] = {0, 2};

    std::string summary = halo::detail::format_neighbor_summary(send_ranks, 3, recv_ranks, 2);

    MPI_Comm named_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &named_comm);
    MPI_Comm_set_name(named_comm, "grid_comm");

    try {
        halo::detail::handle_mpi_error(MPI_ERR_RANK, 1, "MPI_Isend", named_comm, summary);
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        // Verify send-to ranks are present.
        EXPECT_NE(msg.find("send-to:[1,2,3]"), std::string::npos) << "Expected send-to neighbor list in: " << msg;
        // Verify recv-from ranks are present.
        EXPECT_NE(msg.find("recv-from:[0,2]"), std::string::npos) << "Expected recv-from neighbor list in: " << msg;
        // Verify comm name is also present.
        EXPECT_NE(msg.find("grid_comm"), std::string::npos) << "Expected comm name in: " << msg;
        MPI_Comm_free(&named_comm);
        return;
    }
    MPI_Comm_free(&named_comm);
    FAIL() << "Expected std::runtime_error to be thrown";
}

/// Requirement 11.2: format_neighbor_summary handles empty lists.
TEST_F(ErrorMessageTest, FormatNeighborSummaryEmptyLists) {
    std::string summary = halo::detail::format_neighbor_summary(nullptr, 0, nullptr, 0);
    EXPECT_EQ(summary, "send-to:[] recv-from:[]");
}

/// Requirement 11.2: format_neighbor_summary with single neighbor.
TEST_F(ErrorMessageTest, FormatNeighborSummarySingleNeighbor) {
    int send_ranks[] = {5};
    int recv_ranks[] = {3};
    std::string summary = halo::detail::format_neighbor_summary(send_ranks, 1, recv_ranks, 1);
    EXPECT_EQ(summary, "send-to:[5] recv-from:[3]");
}

/// Backward compatibility: the 3-parameter overload still works.
TEST_F(ErrorMessageTest, ThreeParameterOverloadStillWorks) {
    try {
        halo::detail::handle_mpi_error(MPI_ERR_RANK, 10, "MPI_Waitall");
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("MPI_Waitall"), std::string::npos) << "Expected operation in: " << msg;
        EXPECT_NE(msg.find("rank 10"), std::string::npos) << "Expected neighbor rank in: " << msg;
        return;
    }
    FAIL() << "Expected std::runtime_error to be thrown";
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
