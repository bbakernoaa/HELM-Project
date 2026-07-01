/// @file tests/test_rank_stamping.cpp
/// @brief Rank-stamping verification tests for LOGS.
///
/// Validates Requirement 13.2:
///   THE test suite SHALL contain a test that configures the Logger with a
///   known MPI_Rank via the Interposition_Layer, emits at least 100
///   Log_Records, and verifies that every recorded record carries the
///   configured MPI_Rank in its rank field and that zero recorded records
///   carry any other rank value.
///
/// Strategy:
///   1. Initialize MPI via MPI_Init_thread (provided in main()).
///   2. Call configure_communicator(MPI_COMM_WORLD) to detect the real rank.
///   3. Use the MPI_Spy to record and verify the MPI_Comm_rank call.
///   4. Emit 100+ records at various severities.
///   5. Capture output via In_Memory_Sink.
///   6. Verify every recorded entry contains the configured rank at the fixed
///      position and zero entries carry any other rank value.
///
/// **Validates: Requirements 13.2**

#include <gtest/gtest.h>
#include <mpi.h>

#include <logs/logger.hpp>
#include <logs/severity.hpp>
#include <regex>
#include <string>
#include <vector>

#include "in_memory_sink.hpp"
#include "mpi_interposition.hpp"

// ═══════════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════════

class RankStampingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int initialized = 0;
        MPI_Initialized(&initialized);
        ASSERT_TRUE(initialized) << "MPI must be initialized before running these tests";

        // Record the real rank via MPI_Spy for observation.
        auto &spy = logs::testing::MPI_Spy::instance();
        spy.reset();
    }

    void TearDown() override {
        logs::testing::MPI_Spy::instance().reset();
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: extract rank integer from formatted log line
//
// The Logger format is: [RANK:0042] [SEVERITY] ...
// The rank is at a fixed position: characters 6..N within the [RANK:XXXX] field.
// ═══════════════════════════════════════════════════════════════════════════════

static int extract_rank_from_formatted(const std::string &line) {
    // Pattern: [RANK:NNNN] where NNNN is zero-padded (at least 4 digits)
    // or [RANK:----] for sentinel -1.
    static const std::regex rank_re(R"(\[RANK:(----|(\d+))\])");
    std::smatch match;
    if (std::regex_search(line, match, rank_re)) {
        if (match[1].str() == "----") {
            return -1;
        }
        return std::stoi(match[2].str());
    }
    return -999;  // Parse failure sentinel — should not happen.
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 21.1: Rank-Stamping Verification — Every record carries the configured
// rank, and zero records carry any other rank.
//
// Validates: Requirement 13.2
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(RankStampingTest, AllRecordsCarryConfiguredRank) {
    // Step 1: Determine the known rank via real MPI.
    int known_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &known_rank);
    ASSERT_GE(known_rank, 0) << "Real MPI rank must be >= 0";

    // Step 2: Create Logger and configure communicator.
    // This calls mpi_.detect(comm) which calls MPI_Comm_rank internally
    // and stores the rank.
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);

    // Verify the Logger stored the expected rank.
    ASSERT_EQ(logger.rank(), known_rank) << "Logger rank must match the MPI_Comm_rank result";

    // Step 3: Register an In_Memory_Sink to capture all output.
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Step 4: Set threshold to DEBUG so all records pass filtering.
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Step 5: Emit 120 records at various severities (well over 100).
    // Cycle through DEBUG, INFO, WARNING, ERROR (not FATAL — that aborts).
    constexpr int TOTAL_RECORDS = 120;
    const logs::Severity_Level severities[] = {
        logs::Severity_Level::DEBUG,
        logs::Severity_Level::INFO,
        logs::Severity_Level::WARNING,
        logs::Severity_Level::ERROR,
    };
    constexpr int NUM_SEVERITIES = 4;

    for (int i = 0; i < TOTAL_RECORDS; ++i) {
        std::string msg = "rank_stamp_test_message_" + std::to_string(i);
        logger.log(severities[i % NUM_SEVERITIES], msg);
    }

    // Step 6: Verify all recorded entries.
    auto entries = mem_sink.entries();
    ASSERT_GE(entries.size(), static_cast<std::size_t>(TOTAL_RECORDS)) << "Must have at least " << TOTAL_RECORDS << " recorded entries";

    int records_with_correct_rank = 0;
    int records_with_other_rank = 0;

    for (const auto &entry : entries) {
        int extracted = extract_rank_from_formatted(entry);
        if (extracted == known_rank) {
            ++records_with_correct_rank;
        } else {
            ++records_with_other_rank;
        }
    }

    // Every record must carry the configured rank.
    EXPECT_EQ(records_with_correct_rank, static_cast<int>(entries.size())) << "Every recorded entry must carry the configured rank " << known_rank;

    // Zero records must carry any other rank value.
    EXPECT_EQ(records_with_other_rank, 0) << "Zero recorded entries should carry a rank other than " << known_rank;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 21.1b: Rank stamping with MPI_Spy verification of MPI_Comm_rank call.
//
// Additional verification: Use the spy to confirm that the Logger's
// configure_communicator path triggered exactly one MPI_Comm_rank call.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(RankStampingTest, MpiSpyRecordsCommRankCall) {
    auto &spy = logs::testing::MPI_Spy::instance();
    spy.reset();
    spy.set_rank(42);
    spy.set_initialized(true);
    spy.set_thread_level(MPI_THREAD_MULTIPLE);

    // Record a comm_rank call through the spy to verify recording works.
    int rank_out = -1;
    int rc = spy.record_comm_rank(MPI_COMM_WORLD, &rank_out);

    EXPECT_EQ(rc, MPI_SUCCESS);
    EXPECT_EQ(rank_out, 42) << "MPI_Spy must return the configured rank value";

    // Verify the spy recorded exactly one COMM_RANK call.
    auto comm_rank_calls = spy.calls_of_type(logs::testing::MPI_Call_Type::COMM_RANK);
    ASSERT_EQ(comm_rank_calls.size(), 1u);

    // Verify the recorded args contain rank 42.
    const auto &args = std::get<logs::testing::Comm_Rank_Args>(comm_rank_calls[0].args);
    EXPECT_EQ(args.rank_out, 42);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 21.1c: Sentinel rank when no communicator is configured.
//
// Before configure_communicator is called, all records should carry rank -1.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(RankStampingTest, SentinelRankBeforeConfigure) {
    logs::Logger logger;
    // Do NOT call configure_communicator.

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Emit 100+ records.
    for (int i = 0; i < 110; ++i) {
        logger.log(logs::Severity_Level::INFO, "sentinel_test_" + std::to_string(i));
    }

    auto entries = mem_sink.entries();
    ASSERT_GE(entries.size(), 110u);

    // All entries must carry the sentinel rank -1 (rendered as "----").
    for (const auto &entry : entries) {
        int extracted = extract_rank_from_formatted(entry);
        EXPECT_EQ(extracted, -1) << "Before configure_communicator, rank must be sentinel -1. " << "Entry: " << entry;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 21.1d: Rank consistency across multiple threads.
//
// After configure_communicator, all threads must emit records with the same
// configured rank (Requirement 2.6).
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(RankStampingTest, RankConsistentAcrossThreads) {
    int known_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &known_rank);

    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Spawn 4 threads, each emitting 30 records = 120 total.
    constexpr int NUM_THREADS = 4;
    constexpr int RECORDS_PER_THREAD = 30;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&logger, t]() {
            for (int i = 0; i < RECORDS_PER_THREAD; ++i) {
                std::string msg = "thread_" + std::to_string(t) + "_msg_" + std::to_string(i);
                logger.log(logs::Severity_Level::INFO, msg);
            }
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    auto entries = mem_sink.entries();
    ASSERT_GE(entries.size(), static_cast<std::size_t>(NUM_THREADS * RECORDS_PER_THREAD)) << "Must capture all records from all threads";

    int records_with_other_rank = 0;
    for (const auto &entry : entries) {
        int extracted = extract_rank_from_formatted(entry);
        if (extracted != known_rank) {
            ++records_with_other_rank;
        }
    }

    EXPECT_EQ(records_with_other_rank, 0) << "All threads must emit records with the configured rank " << known_rank << ", not any other value";
}

// ═══════════════════════════════════════════════════════════════════════════════
// GTest main() — MPI lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
