/// @file prop_format_round_trip.cpp
/// @brief Property-based test: Record Formatting Round-Trip.
///
/// Format a record to text via Logger; extract fields at fixed positions;
/// verify rank, severity label, and message are recovered correctly.
///
/// **Validates: Requirements 2.3, 1.6**

#include <logs/logger.hpp>
#include "in_memory_sink.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <string>
#include <string_view>

namespace {

// ─── Generators ─────────────────────────────────────────────────────────────

/// Generate a non-FATAL severity (to avoid process termination).
rc::Gen<logs::Severity_Level> genNonFatalSeverity() {
    return rc::gen::map(rc::gen::inRange(0, 4), [](int v) {
        return static_cast<logs::Severity_Level>(v);
    });
}

/// Generate a printable message string without brackets.
/// Range 33-90 avoids '[' (91), ']' (93), and control chars.
rc::Gen<std::string> genMessage() {
    return rc::gen::map(
        rc::gen::container<std::string>(rc::gen::inRange(33, 91)),
        [](std::string s) {
            if (s.empty()) s = "x";
            return s;
        }
    );
}

// ─── Parsing Helpers ────────────────────────────────────────────────────────

/// Extract the rank field. Returns -1 for "----", -999 on parse failure.
int parse_rank(const std::string& formatted) {
    // Expected: starts with "[RANK:" and contains closing ']'
    if (formatted.size() < 11) return -999; // "[RANK:----]" is 11 chars minimum
    if (formatted.compare(0, 6, "[RANK:") != 0) return -999;

    auto close = formatted.find(']', 6);
    if (close == std::string::npos) return -999;

    std::string rank_str = formatted.substr(6, close - 6);
    if (rank_str == "----") return -1;

    try {
        return std::stoi(rank_str);
    } catch (...) {
        return -999;
    }
}

/// Extract the severity label.
std::string parse_severity(const std::string& formatted) {
    // First ']' ends the rank field; then look for the next '[...]'
    auto rank_close = formatted.find(']');
    if (rank_close == std::string::npos) return "";

    auto sev_open = formatted.find('[', rank_close + 1);
    if (sev_open == std::string::npos) return "";

    auto sev_close = formatted.find(']', sev_open + 1);
    if (sev_close == std::string::npos) return "";

    return formatted.substr(sev_open + 1, sev_close - sev_open - 1);
}

/// Extract the message: everything after last ']' + space, before '\n'.
std::string parse_message(const std::string& formatted) {
    auto last_bracket = formatted.rfind(']');
    if (last_bracket == std::string::npos) return "";

    std::size_t msg_start = last_bracket + 1;
    if (msg_start < formatted.size() && formatted[msg_start] == ' ') {
        ++msg_start;
    }

    std::size_t msg_end = formatted.find('\n', msg_start);
    if (msg_end == std::string::npos) {
        msg_end = formatted.size();
    }
    return formatted.substr(msg_start, msg_end - msg_start);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8: Record Formatting Round-Trip
// Validates: Requirements 2.3, 1.6
// ─────────────────────────────────────────────────────────────────────────────

/// Verify rank, severity label, and message can be recovered from formatted
/// output when no communicator is configured (rank == -1).
RC_GTEST_PROP(RecordFormattingRoundTrip,
              UnconfiguredRankRoundTrips,
              ()) {
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    logger.log(severity, message);

    RC_ASSERT(mem_sink.count() == 1);
    const auto entries = mem_sink.entries();
    const std::string& formatted = entries[0];

    // Verify rank: unconfigured -> [RANK:----] -> parsed as -1
    RC_ASSERT(parse_rank(formatted) == -1);

    // Verify severity label matches to_string() (Requirement 1.6)
    RC_ASSERT(parse_severity(formatted) == std::string(logs::to_string(severity)));

    // Verify message recovered verbatim
    RC_ASSERT(parse_message(formatted) == message);
}

/// Severity label in formatted output always matches to_string().
RC_GTEST_PROP(RecordFormattingRoundTrip,
              SeverityLabelMatchesToString,
              ()) {
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    logger.log(severity, message);

    RC_ASSERT(mem_sink.count() == 1);
    const auto entries = mem_sink.entries();
    RC_ASSERT(parse_severity(entries[0]) == std::string(logs::to_string(severity)));
}

/// The message field is preserved byte-for-byte through formatting.
RC_GTEST_PROP(RecordFormattingRoundTrip,
              MessagePreservedVerbatim,
              ()) {
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    logger.log(severity, message);

    RC_ASSERT(mem_sink.count() == 1);
    const auto entries = mem_sink.entries();
    RC_ASSERT(parse_message(entries[0]) == message);
}

/// The formatted output always ends with a newline character.
RC_GTEST_PROP(RecordFormattingRoundTrip,
              FormattedOutputEndsWithNewline,
              ()) {
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    logger.log(severity, message);

    RC_ASSERT(mem_sink.count() == 1);
    const auto entries = mem_sink.entries();
    RC_ASSERT(!entries[0].empty());
    RC_ASSERT(entries[0].back() == '\n');
}

/// The rank field always starts at position 0 (fixed position, Requirement 2.3).
RC_GTEST_PROP(RecordFormattingRoundTrip,
              RankFieldAtFixedPosition,
              ()) {
    const auto severity = *genNonFatalSeverity();
    const auto message = *genMessage();

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    logger.log(severity, message);

    RC_ASSERT(mem_sink.count() == 1);
    const auto entries = mem_sink.entries();
    RC_ASSERT(entries[0].size() >= 6);
    RC_ASSERT(entries[0].compare(0, 6, "[RANK:") == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic Round-Trip Test (Google Test)
// Validates: Requirement 13.8
// ─────────────────────────────────────────────────────────────────────────────

/// Format a Log_Record with known severity (INFO), rank (from Logger), and
/// message ("round_trip_test_msg"); extract from fixed positions; verify equality.
TEST(RecordFormattingRoundTrip, DeterministicRoundTrip) {
    // 1. Create a Logger with In_Memory_Sink
    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // 2. Set threshold to DEBUG so INFO passes filtering
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // 3. Known values
    const auto known_severity = logs::Severity_Level::INFO;
    const int known_rank = logger.rank(); // -1 (unconfigured)
    const std::string known_message = "round_trip_test_msg";

    // 4. Log the record
    logger.log(known_severity, known_message);

    // 5. Extract the formatted entry from the In_Memory_Sink
    ASSERT_EQ(mem_sink.count(), 1u);
    const auto entries = mem_sink.entries();
    const std::string& formatted = entries[0];

    // 6. Parse fields at fixed positions and verify equality
    // Verify: extracted rank == configured rank (-1)
    EXPECT_EQ(parse_rank(formatted), known_rank);

    // Verify: extracted severity label == "INFO"
    EXPECT_EQ(parse_severity(formatted), "INFO");

    // Verify: extracted message == "round_trip_test_msg"
    EXPECT_EQ(parse_message(formatted), known_message);
}

} // namespace
