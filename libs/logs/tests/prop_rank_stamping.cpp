/// @file prop_rank_stamping.cpp
/// @brief Property 6: Rank Stamping Invariance.
///
/// Uses RapidCheck + Google Test to verify that every record emitted by a
/// Logger configured with a real MPI communicator carries the correct rank
/// stamp. Runs with mpirun -np 4.
///
/// **Validates: Requirements 2.1, 2.2, 2.5, 2.6**

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <logs/logger.hpp>
#include <regex>
#include <string>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate a random valid Severity_Level (excluding FATAL which triggers abort).
rc::Gen<logs::Severity_Level> genNonFatalSeverity() {
    return rc::gen::map(rc::gen::inRange(0, 4), [](int v) { return static_cast<logs::Severity_Level>(v); });
}

/// Generate a random non-empty ASCII message (printable, no newlines).
rc::Gen<std::string> genMessage() {
    return rc::gen::map(rc::gen::container<std::string>(rc::gen::inRange(32, 127)), [](std::string s) {
        // Ensure non-empty
        if (s.empty()) s = "x";
        return s;
    });
}

// ─── Helper: Extract rank from formatted log line ────────────────────────────

/// Extracts the numeric rank from the "[RANK:NNNN]" prefix.
/// Returns -2 on parse failure (distinct from sentinel -1).
int extract_rank(const std::string &line) {
    // Format is: [RANK:0042] ...
    static const std::regex rank_re(R"(\[RANK:(\d{4,})\])");
    std::smatch match;
    if (std::regex_search(line, match, rank_re)) {
        return std::stoi(match[1].str());
    }
    // Check for unidentified rank sentinel: [RANK:----]
    if (line.find("[RANK:----]") != std::string::npos) {
        return -1;
    }
    return -2;  // parse failure
}

// ─── Test Fixture ────────────────────────────────────────────────────────────

class RankStampingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int initialized = 0;
        MPI_Initialized(&initialized);
        ASSERT_TRUE(initialized) << "MPI must be initialized before running these tests";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Property 6: Rank Stamping Invariance
//
// Configure Logger with known rank via MPI_COMM_WORLD; emit records at
// various severity levels with random messages; verify every record carries
// that rank.
//
// Validates: Requirements 2.1, 2.2, 2.5, 2.6
// ─────────────────────────────────────────────────────────────────────────────

/// Every emitted record must carry the same rank that the Logger reports.
RC_GTEST_FIXTURE_PROP(RankStampingTest, AllRecordsCarryConfiguredRank, ()) {
    // Create and configure the Logger.
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Query the stored rank — this is what every record must carry.
    const int expected_rank = logger.rank();
    RC_PRE(expected_rank >= 0);

    // Attach in-memory sink to capture formatted output.
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Generate a random severity and message, emit one record.
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logger.log(severity, message);

    // Verify the recorded entry carries the expected rank.
    const auto entries = mem_sink.entries();
    RC_ASSERT(entries.size() == 1u);

    const int actual_rank = extract_rank(entries[0]);
    RC_ASSERT(actual_rank == expected_rank);
}

/// Multiple records emitted at varying severities all carry the same rank.
RC_GTEST_FIXTURE_PROP(RankStampingTest, MultipleRecordsAllCarrySameRank, ()) {
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::DEBUG);

    const int expected_rank = logger.rank();
    RC_PRE(expected_rank >= 0);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Generate a random count of records to emit (2–20).
    const auto count = *rc::gen::inRange(2, 21);

    for (int i = 0; i < count; ++i) {
        const auto severity = *genNonFatalSeverity();
        const auto message = *genMessage();
        logger.log(severity, message);
    }

    const auto entries = mem_sink.entries();
    RC_ASSERT(static_cast<int>(entries.size()) == count);

    for (const auto &entry : entries) {
        const int actual_rank = extract_rank(entry);
        RC_ASSERT(actual_rank == expected_rank);
    }
}

/// The rank accessor always returns the same value that records are stamped with.
RC_GTEST_FIXTURE_PROP(RankStampingTest, RankAccessorConsistentWithStamp, ()) {
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Query rank multiple times — must always be the same.
    const int rank1 = logger.rank();
    const int rank2 = logger.rank();
    RC_ASSERT(rank1 == rank2);
    RC_PRE(rank1 >= 0);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    const auto message = *genMessage();
    logger.log(logs::Severity_Level::INFO, message);

    const auto entries = mem_sink.entries();
    RC_ASSERT(entries.size() == 1u);
    RC_ASSERT(extract_rank(entries[0]) == rank1);
}

/// Rank stamp is applied before severity filtering (i.e., the rank is
/// determined at initialization time, not per-record).
/// We verify this by emitting at the threshold boundary — if the record is
/// accepted, it must carry the correct rank.
RC_GTEST_FIXTURE_PROP(RankStampingTest, RankStampedBeforeFiltering, ()) {
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);

    const int expected_rank = logger.rank();
    RC_PRE(expected_rank >= 0);

    // Set threshold to a random non-FATAL level.
    const auto threshold = *genNonFatalSeverity();
    logger.set_threshold(threshold);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Emit at exactly the threshold level — should be accepted.
    const auto message = *genMessage();
    logger.log(threshold, message);

    const auto entries = mem_sink.entries();
    RC_ASSERT(entries.size() == 1u);
    RC_ASSERT(extract_rank(entries[0]) == expected_rank);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic complement: verify rank stamping across all severity levels
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(RankStampingTest, AllNonFatalSeveritiesCarryRank) {
    logs::Logger logger;
    logger.configure_communicator(MPI_COMM_WORLD);
    logger.set_threshold(logs::Severity_Level::DEBUG);

    const int expected_rank = logger.rank();
    ASSERT_GE(expected_rank, 0);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Emit one record at each non-FATAL severity.
    const logs::Severity_Level levels[] = {
        logs::Severity_Level::DEBUG,
        logs::Severity_Level::INFO,
        logs::Severity_Level::WARNING,
        logs::Severity_Level::ERROR,
    };

    for (auto level : levels) {
        logger.log(level, "test message");
    }

    const auto entries = mem_sink.entries();
    ASSERT_EQ(entries.size(), 4u);

    for (const auto &entry : entries) {
        EXPECT_EQ(extract_rank(entry), expected_rank) << "Entry: " << entry;
    }
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// GTest + MPI lifecycle
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
