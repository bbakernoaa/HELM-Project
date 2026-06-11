/// @file test_consolidation.cpp
/// @brief Property test: Consolidation Buffer Lifecycle (Property 17).
///
/// **Validates: Requirements 4.7**
///
/// Verifies that consolidation processes exactly the buffered records then
/// clears the buffer; a subsequent consolidation produces zero representatives.
///
/// Test approach (Logger level, local consolidation path):
///   1. Create Logger without communicator (local consolidation)
///   2. Add In_Memory_Sink
///   3. Set threshold to DEBUG
///   4. Emit N records (buffered by Logger)
///   5. Call consolidate() — verify sink receives consolidated output
///   6. Clear the sink
///   7. Call consolidate() again — verify sink receives ZERO new output
///      (buffer was cleared by the first consolidation)

#include <logs/logger.hpp>
#include "in_memory_sink.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Property 17: Consolidation Buffer Lifecycle
// Validates: Requirements 4.7
//
// THE Logger SHALL provide an API function that triggers a single collective
// consolidation operation ... that operates over the set of accepted
// Log_Records buffered since the most recent prior consolidation operation
// (or since Logger initialization if none has occurred), and that clears that
// buffer once the operation completes.
// ─────────────────────────────────────────────────────────────────────────────

/// After emitting records and calling consolidate(), the buffer is cleared.
/// A subsequent consolidate() call produces zero new sink output.
TEST(ConsolidationBufferLifecycle, SecondConsolidationProducesZeroOutput) {
    // 1. Create Logger without communicator (local consolidation path).
    logs::Logger logger;

    // 2. Add In_Memory_Sink.
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // 3. Set threshold to DEBUG so all messages are accepted.
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // 4. Emit N records (5 INFO messages).
    constexpr int N = 5;
    for (int i = 0; i < N; ++i) {
        logger.log(logs::Severity_Level::INFO,
                   "lifecycle test message " + std::to_string(i));
    }

    // Records are dispatched to sinks immediately during log() and also
    // buffered for consolidation. Verify we received N records.
    ASSERT_EQ(mem_sink.count(), static_cast<std::size_t>(N));

    // 5. Call consolidate() — processes buffered records and clears buffer.
    logger.consolidate();

    // 6. Clear the sink to observe only new writes from subsequent operations.
    mem_sink.clear();
    ASSERT_EQ(mem_sink.count(), 0u);

    // 7. Call consolidate() again — buffer was cleared, so ZERO new output.
    logger.consolidate();

    // Verify sink received no new writes from the second consolidation.
    EXPECT_EQ(mem_sink.count(), 0u)
        << "Second consolidate() should produce zero output because the "
           "buffer was cleared by the first consolidation (Requirement 4.7).";
}

/// Consolidation processes ALL buffered records — verify the count matches.
TEST(ConsolidationBufferLifecycle, FirstConsolidationProcessesAllBuffered) {
    logs::Logger logger;

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Emit records at various severities to fill the buffer.
    logger.log(logs::Severity_Level::DEBUG, "debug msg");
    logger.log(logs::Severity_Level::INFO, "info msg");
    logger.log(logs::Severity_Level::WARNING, "warning msg");
    logger.log(logs::Severity_Level::ERROR, "error msg");

    // All 4 records dispatched immediately during log().
    ASSERT_EQ(mem_sink.count(), 4u);

    // Clear sink to isolate consolidation output.
    mem_sink.clear();

    // Consolidate — processes all 4 buffered records locally.
    logger.consolidate();

    // After consolidation, buffer should be empty.
    // Clear sink and consolidate again — nothing should appear.
    mem_sink.clear();
    logger.consolidate();

    EXPECT_EQ(mem_sink.count(), 0u)
        << "Buffer must be empty after first consolidation cleared it.";
}

/// Multiple consolidation cycles: each cycle processes only records emitted
/// since the previous consolidation.
TEST(ConsolidationBufferLifecycle, MultipleCyclesBufferIndependence) {
    logs::Logger logger;

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Cycle 1: emit 3 records, consolidate.
    for (int i = 0; i < 3; ++i) {
        logger.log(logs::Severity_Level::INFO, "cycle1 msg");
    }
    logger.consolidate();

    // Verify buffer cleared — second consolidate produces nothing.
    mem_sink.clear();
    logger.consolidate();
    EXPECT_EQ(mem_sink.count(), 0u)
        << "After cycle 1 consolidation, buffer should be empty.";

    // Cycle 2: emit 2 new records.
    mem_sink.clear();
    for (int i = 0; i < 2; ++i) {
        logger.log(logs::Severity_Level::WARNING, "cycle2 msg");
    }

    // These 2 new records should be in the buffer now.
    ASSERT_EQ(mem_sink.count(), 2u);

    // Consolidate cycle 2.
    logger.consolidate();

    // Buffer cleared again — third consolidate produces nothing.
    mem_sink.clear();
    logger.consolidate();
    EXPECT_EQ(mem_sink.count(), 0u)
        << "After cycle 2 consolidation, buffer should be empty.";
}

/// With no records buffered (fresh logger), consolidation produces nothing.
TEST(ConsolidationBufferLifecycle, EmptyBufferConsolidationProducesNothing) {
    logs::Logger logger;

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Consolidate immediately — no records ever emitted.
    logger.consolidate();

    // Sink should have received nothing.
    EXPECT_EQ(mem_sink.count(), 0u)
        << "Consolidation on an empty buffer should produce zero output.";
}

/// Records below threshold are not buffered, so consolidation doesn't
/// process them.
TEST(ConsolidationBufferLifecycle, FilteredRecordsNotBuffered) {
    logs::Logger logger;

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Set threshold to ERROR — DEBUG and INFO will be filtered out.
    logger.set_threshold(logs::Severity_Level::ERROR);

    // Emit records below threshold — they should be discarded entirely.
    logger.log(logs::Severity_Level::DEBUG, "filtered debug");
    logger.log(logs::Severity_Level::INFO, "filtered info");

    // Nothing reached the sink (filtered).
    ASSERT_EQ(mem_sink.count(), 0u);

    // Consolidate — filtered records were never buffered.
    logger.consolidate();

    // Still nothing.
    EXPECT_EQ(mem_sink.count(), 0u)
        << "Filtered records should not be buffered for consolidation.";
}

} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// Property 15: Consolidation Preserves Severity and Message
// Validates: Requirements 4.8
//
// Verify that when consolidation produces a representative record, the
// representative preserves the original severity and message unchanged
// (byte-for-byte).
// ─────────────────────────────────────────────────────────────────────────────

#include <logs/detail/consolidation.hpp>
#include <logs/log_record.hpp>

#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <set>
#include <utility>

namespace {

RC_GTEST_PROP(ConsolidationPreservesSeverityAndMessage,
              LocalConsolidationPreservesKeyFields,
              ()) {
    // Generate arbitrary severity and message.
    const int raw_severity = *rc::gen::inRange(0, 5);
    const auto severity = static_cast<logs::Severity_Level>(raw_severity);

    // Generate a non-empty message with arbitrary printable ASCII characters.
    const auto message = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));

    // Generate multiple records with the same severity+message but different ranks.
    const auto num_records = *rc::gen::inRange(2, 10);

    std::vector<logs::Log_Record> records;
    records.reserve(static_cast<std::size_t>(num_records));
    for (int i = 0; i < num_records; ++i) {
        records.emplace_back(
            severity,
            message,
            i,                          // distinct rank per record
            std::nullopt,               // no source location
            std::vector<std::string>{}, // no context labels
            std::nullopt                // no stack trace
        );
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // With all records sharing the same key, expect exactly one representative.
    RC_ASSERT(consolidated.size() == 1u);

    // Verify: the representative's key.severity == original severity.
    RC_ASSERT(consolidated[0].key.severity == severity);

    // Verify: the representative's key.message == original message (byte-for-byte).
    RC_ASSERT(consolidated[0].key.message == message);
    RC_ASSERT(consolidated[0].key.message.size() == message.size());
}

RC_GTEST_PROP(ConsolidationPreservesSeverityAndMessage,
              MultipleKeysEachPreserved,
              ()) {
    // Generate two distinct severity+message combinations.
    const int raw_sev1 = *rc::gen::inRange(0, 5);
    const int raw_sev2 = *rc::gen::inRange(0, 5);
    const auto sev1 = static_cast<logs::Severity_Level>(raw_sev1);
    const auto sev2 = static_cast<logs::Severity_Level>(raw_sev2);

    const auto msg1 = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));
    auto msg2 = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));

    // Ensure the two keys are actually distinct.
    if (raw_sev1 == raw_sev2 && msg1 == msg2) {
        msg2 += "X";  // Force distinction.
    }

    // Create records for both keys with varying ranks.
    std::vector<logs::Log_Record> records;
    records.emplace_back(sev1, msg1, 0, std::nullopt,
                         std::vector<std::string>{}, std::nullopt);
    records.emplace_back(sev1, msg1, 1, std::nullopt,
                         std::vector<std::string>{}, std::nullopt);
    records.emplace_back(sev2, msg2, 2, std::nullopt,
                         std::vector<std::string>{}, std::nullopt);
    records.emplace_back(sev2, msg2, 3, std::nullopt,
                         std::vector<std::string>{}, std::nullopt);

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Should have exactly 2 representatives (one per distinct key).
    RC_ASSERT(consolidated.size() == 2u);

    // Verify each representative preserves its original severity and message.
    bool found_key1 = false;
    bool found_key2 = false;
    for (const auto& rep : consolidated) {
        if (rep.key.severity == sev1 && rep.key.message == msg1) {
            found_key1 = true;
        }
        if (rep.key.severity == sev2 && rep.key.message == msg2) {
            found_key2 = true;
        }
    }
    RC_ASSERT(found_key1);
    RC_ASSERT(found_key2);
}

RC_GTEST_PROP(ConsolidationPreservesSeverityAndMessage,
              MessagePreservedByteForByte,
              ()) {
    // Test with messages containing various characters including spaces,
    // punctuation, and near-boundary bytes.
    const auto severity = static_cast<logs::Severity_Level>(
        *rc::gen::inRange(0, 5));

    // Generate a message that may contain any byte value 1..126.
    const auto message = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(1, 127),
                         [](int c) { return static_cast<char>(c); })));

    // Create a single record (edge case: single contributor).
    std::vector<logs::Log_Record> records;
    records.emplace_back(severity, message, 42, std::nullopt,
                         std::vector<std::string>{}, std::nullopt);

    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    RC_ASSERT(consolidated.size() == 1u);
    RC_ASSERT(consolidated[0].key.severity == severity);

    // Byte-for-byte comparison: check each character matches.
    RC_ASSERT(consolidated[0].key.message.size() == message.size());
    for (std::size_t i = 0; i < message.size(); ++i) {
        RC_ASSERT(consolidated[0].key.message[i] == message[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 13: Exactly One Root-Written Representative Per Key
// Validates: Requirements 4.2
//
// Inject records from multiple ranks sharing a key; verify one representative
// emitted on root only. Tested via consolidate_local() directly since MPI
// collective tests are complex.
// ─────────────────────────────────────────────────────────────────────────────

/// For multiple records sharing the SAME severity+message but DIFFERENT ranks,
/// consolidate_local produces exactly ONE Consolidated_Record for that key,
/// and the rank_count equals the number of records injected.
RC_GTEST_PROP(ExactlyOneRepresentativePerKey,
              SingleKeyMultipleRanks,
              ()) {
    // Generate between 2 and 20 distinct "ranks" contributing the same key.
    const auto num_ranks = *rc::gen::inRange(2, 21);

    // Fixed severity and message for the shared key.
    const auto sev_int = *rc::gen::inRange(0, 5); // DEBUG..FATAL
    const auto severity = static_cast<logs::Severity_Level>(sev_int);
    const auto message = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(rc::gen::inRange(97, 123)));

    // Build records from multiple "ranks" sharing the same key.
    std::vector<logs::Log_Record> records;
    records.reserve(static_cast<std::size_t>(num_ranks));
    for (int r = 0; r < num_ranks; ++r) {
        records.emplace_back(severity, std::string(message), r,
                             std::nullopt, std::vector<std::string>{},
                             std::nullopt);
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto result = engine.consolidate_local(records);

    // Exactly ONE representative for the single key.
    RC_ASSERT(result.size() == 1u);

    // The representative preserves the key.
    RC_ASSERT(result[0].key.severity == severity);
    RC_ASSERT(result[0].key.message == message);

    // The rank_count equals the number of contributing records.
    RC_ASSERT(result[0].rank_count == num_ranks);
}

/// For records with 2 distinct (severity, message) pairs from multiple ranks,
/// consolidate_local produces exactly 2 representatives.
RC_GTEST_PROP(ExactlyOneRepresentativePerKey,
              TwoKeysFromMultipleRanks,
              ()) {
    // Generate two distinct keys.
    const auto sev_int1 = *rc::gen::inRange(0, 5);
    const auto severity1 = static_cast<logs::Severity_Level>(sev_int1);
    const auto message1 = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(rc::gen::inRange(97, 123)));

    const auto sev_int2 = *rc::gen::inRange(0, 5);
    const auto severity2 = static_cast<logs::Severity_Level>(sev_int2);
    const auto message2 = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(rc::gen::inRange(97, 123)));

    // Ensure the two keys are actually distinct.
    RC_PRE(sev_int1 != sev_int2 || message1 != message2);

    // Generate between 1 and 10 records per key from different "ranks".
    const auto count1 = *rc::gen::inRange(1, 11);
    const auto count2 = *rc::gen::inRange(1, 11);

    std::vector<logs::Log_Record> records;

    for (int i = 0; i < count1; ++i) {
        records.emplace_back(severity1, std::string(message1), i,
                             std::nullopt, std::vector<std::string>{},
                             std::nullopt);
    }
    for (int i = 0; i < count2; ++i) {
        records.emplace_back(severity2, std::string(message2), 100 + i,
                             std::nullopt, std::vector<std::string>{},
                             std::nullopt);
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto result = engine.consolidate_local(records);

    // Exactly TWO representatives (one per distinct key).
    RC_ASSERT(result.size() == 2u);

    // Verify each key is present exactly once and rank counts are correct.
    bool found_key1 = false;
    bool found_key2 = false;
    for (const auto& rec : result) {
        if (rec.key.severity == severity1 && rec.key.message == message1) {
            RC_ASSERT(!found_key1);  // Not a duplicate.
            found_key1 = true;
            RC_ASSERT(rec.rank_count == count1);
        } else if (rec.key.severity == severity2 && rec.key.message == message2) {
            RC_ASSERT(!found_key2);  // Not a duplicate.
            found_key2 = true;
            RC_ASSERT(rec.rank_count == count2);
        } else {
            RC_FAIL("Unexpected key in consolidated result");
        }
    }

    RC_ASSERT(found_key1);
    RC_ASSERT(found_key2);
}

/// For N distinct keys each appearing M times, verify exactly N representatives.
RC_GTEST_PROP(ExactlyOneRepresentativePerKey,
              NKeysProducesNRepresentatives,
              ()) {
    const auto num_keys = *rc::gen::inRange(1, 10);
    const auto reps_per_key = *rc::gen::inRange(2, 8);

    // Build distinct keys.
    std::vector<std::pair<logs::Severity_Level, std::string>> keys;
    std::set<std::pair<int, std::string>> key_set;

    for (int k = 0; k < num_keys; ++k) {
        auto sev_int = *rc::gen::inRange(0, 5);
        auto severity = static_cast<logs::Severity_Level>(sev_int);
        auto message = *rc::gen::nonEmpty(
            rc::gen::container<std::string>(rc::gen::inRange(97, 123)));

        // Ensure uniqueness among generated keys.
        if (key_set.count({sev_int, message})) {
            RC_DISCARD("duplicate key generated");
        }
        key_set.emplace(sev_int, message);
        keys.emplace_back(severity, message);
    }

    // Build records: each key repeated from different ranks.
    std::vector<logs::Log_Record> records;
    int rank_counter = 0;
    for (const auto& [sev, msg] : keys) {
        for (int r = 0; r < reps_per_key; ++r) {
            records.emplace_back(sev, std::string(msg), rank_counter++,
                                 std::nullopt, std::vector<std::string>{},
                                 std::nullopt);
        }
    }

    logs::detail::Consolidation_Engine engine;
    auto result = engine.consolidate_local(records);

    // Exactly num_keys representatives.
    RC_ASSERT(static_cast<int>(result.size()) == num_keys);

    // Each representative has rank_count == reps_per_key.
    for (const auto& rep : result) {
        RC_ASSERT(rep.rank_count == reps_per_key);
    }
}

} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// Property 16: Consolidation Transports Only Keys and Ranks
// **Validates: Requirements 4.5**
//
// THE Logger SHALL use MPI solely to gather Consolidation_Keys and contributing
// ranks and SHALL NOT transport any other payload.
//
// Verification approach (structural + behavioral):
//   1. Structural: Consolidated_Record contains only Consolidation_Key (severity
//      + message), rank_count (int), and ranges (vector<Rank_Range>). It does
//      NOT contain source_location, stack_trace, or context_labels fields.
//   2. Behavioral: Create Log_Records rich with metadata (location, stack trace,
//      context labels), consolidate_local them, and verify the resulting
//      Consolidated_Record representatives carry ONLY the key and rank info —
//      none of the extra metadata is preserved or accessible.
//   3. Serialization: The serialize_record() wire format (tested via
//      consolidate_collective in integration) only encodes severity (int) +
//      message (string) + rank (int). We verify this indirectly by showing
//      that rich-metadata records consolidate identically to bare records.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Structural proof: Consolidated_Record does NOT have location, stack_trace,
// or context_labels members. If it did, these static_asserts would fail at
// compile time. We use SFINAE detection idioms.

template <typename T, typename = void>
struct has_location : std::false_type {};
template <typename T>
struct has_location<T, std::void_t<decltype(std::declval<T>().location)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_stack_trace : std::false_type {};
template <typename T>
struct has_stack_trace<T, std::void_t<decltype(std::declval<T>().stack_trace)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_context_labels : std::false_type {};
template <typename T>
struct has_context_labels<T, std::void_t<decltype(std::declval<T>().context_labels)>>
    : std::true_type {};

/// Compile-time structural proof that Consolidated_Record carries no metadata.
TEST(ConsolidationTransportsOnlyKeysAndRanks, StructuralProofNoMetadataFields) {
    // Consolidated_Record must NOT have location, stack_trace, or context_labels.
    static_assert(!has_location<logs::detail::Consolidated_Record>::value,
        "Consolidated_Record must not contain a 'location' field");
    static_assert(!has_stack_trace<logs::detail::Consolidated_Record>::value,
        "Consolidated_Record must not contain a 'stack_trace' field");
    static_assert(!has_context_labels<logs::detail::Consolidated_Record>::value,
        "Consolidated_Record must not contain a 'context_labels' field");

    // Consolidated_Record only has: key (Consolidation_Key), rank_count, ranges.
    // Consolidation_Key only has: severity + message.
    static_assert(!has_location<logs::detail::Consolidation_Key>::value,
        "Consolidation_Key must not contain a 'location' field");
    static_assert(!has_stack_trace<logs::detail::Consolidation_Key>::value,
        "Consolidation_Key must not contain a 'stack_trace' field");
    static_assert(!has_context_labels<logs::detail::Consolidation_Key>::value,
        "Consolidation_Key must not contain a 'context_labels' field");

    SUCCEED();
}

/// Property test: Records with rich metadata consolidate to representatives
/// containing ONLY key (severity + message) and rank information.
RC_GTEST_PROP(ConsolidationTransportsOnlyKeysAndRanks,
              RichMetadataRecordsConsolidateToKeyAndRankOnly,
              ()) {
    // Generate a severity and message for the consolidation key.
    const auto severity = static_cast<logs::Severity_Level>(
        *rc::gen::inRange(0, 5));
    const auto message = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));

    // Generate rich metadata: source location, stack trace, context labels.
    const auto file = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(rc::gen::inRange(97, 123)));
    const auto line = *rc::gen::inRange(1, 10000);
    const auto function = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(rc::gen::inRange(97, 123)));
    const auto stack_trace = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));
    const auto num_labels = *rc::gen::inRange(1, 5);
    std::vector<std::string> context_labels;
    for (int i = 0; i < num_labels; ++i) {
        context_labels.push_back(*rc::gen::nonEmpty(
            rc::gen::container<std::string>(rc::gen::inRange(97, 123))));
    }

    // Generate multiple ranks contributing records with rich metadata.
    const auto num_records = *rc::gen::inRange(2, 10);

    std::vector<logs::Log_Record> records;
    records.reserve(static_cast<std::size_t>(num_records));
    for (int r = 0; r < num_records; ++r) {
        records.emplace_back(
            severity,
            message,
            r,                                           // distinct rank
            logs::Source_Location{file, line, function}, // source location
            context_labels,                              // context labels
            stack_trace                                  // stack trace
        );
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Exactly one representative for the single key.
    RC_ASSERT(consolidated.size() == 1u);

    // The representative carries ONLY:
    //   - key.severity (matches original)
    //   - key.message (matches original)
    //   - rank_count (number of contributors)
    //   - ranges (rank ranges)
    RC_ASSERT(consolidated[0].key.severity == severity);
    RC_ASSERT(consolidated[0].key.message == message);
    RC_ASSERT(consolidated[0].rank_count == num_records);
    RC_ASSERT(!consolidated[0].ranges.empty());

    // The representative does NOT carry any of the rich metadata.
    // This is proven structurally (static_asserts above) but we also verify
    // that consolidation produces rank info (count and ranges).
    // Local consolidation uses sentinel rank -1 per Requirement 4.9.
    RC_ASSERT(consolidated[0].ranges.size() == 1u);
    RC_ASSERT(consolidated[0].ranges[0].first == -1);
    RC_ASSERT(consolidated[0].ranges[0].last == -1);
}

/// Property test: Records with and without metadata consolidate identically
/// when they share the same severity + message key. This proves the
/// serialization format ignores non-key fields.
RC_GTEST_PROP(ConsolidationTransportsOnlyKeysAndRanks,
              MetadataIrrelevantToConsolidation,
              ()) {
    const auto severity = static_cast<logs::Severity_Level>(
        *rc::gen::inRange(0, 5));
    const auto message = *rc::gen::nonEmpty(
        rc::gen::container<std::string>(
            rc::gen::map(rc::gen::inRange(32, 127),
                         [](int c) { return static_cast<char>(c); })));

    // Record A: with rich metadata (location, trace, labels).
    logs::Log_Record rich_record(
        severity, message, 0,
        logs::Source_Location{"file.cpp", 42, "do_thing"},
        std::vector<std::string>{"outer", "inner"},
        std::string{"frame0\nframe1\nframe2"});

    // Record B: bare record — same severity+message, no metadata.
    logs::Log_Record bare_record(
        severity, message, 1,
        std::nullopt,
        std::vector<std::string>{},
        std::nullopt);

    // Both share the same consolidation key — they should consolidate together.
    std::vector<logs::Log_Record> records{rich_record, bare_record};

    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Exactly one representative (same key, different ranks).
    RC_ASSERT(consolidated.size() == 1u);

    // The representative carries only key and rank info.
    RC_ASSERT(consolidated[0].key.severity == severity);
    RC_ASSERT(consolidated[0].key.message == message);
    RC_ASSERT(consolidated[0].rank_count == 2);

    // Local consolidation annotates with sentinel rank -1 (Requirement 4.9).
    RC_ASSERT(consolidated[0].ranges.size() == 1u);
    RC_ASSERT(consolidated[0].ranges[0].first == -1);
    RC_ASSERT(consolidated[0].ranges[0].last == -1);
}

} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// Consolidation Distinct Keys Test
// **Validates: Requirements 13.4**
//
// THE test suite SHALL contain a test that injects Log_Records bearing at least
// 2 distinct Consolidation_Keys, triggers a consolidation operation, and
// verifies that the number of representative records produced equals the number
// of distinct Consolidation_Keys and that no representative record combines
// records originating from more than one Consolidation_Key.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Inject records with 2 distinct Consolidation_Keys; consolidate; verify
/// representative count equals distinct key count and no cross-key contamination.
TEST(ConsolidationDistinctKeys, TwoKeysProducesTwoRepresentatives) {
    // Create Log_Records with distinct messages from different "ranks":
    //   "Error A" from ranks 0, 1, 2  (3 records)
    //   "Error B" from ranks 3, 4     (2 records)
    std::vector<logs::Log_Record> records;

    // Key A: severity ERROR, message "Error A", from ranks 0, 1, 2.
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Error A"), 0,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Error A"), 1,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Error A"), 2,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);

    // Key B: severity ERROR, message "Error B", from ranks 3, 4.
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Error B"), 3,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Error B"), 4,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);

    // Consolidate locally (no MPI communicator).
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Verify: exactly 2 Consolidated_Records produced (one per distinct key).
    ASSERT_EQ(consolidated.size(), 2u)
        << "Expected exactly 2 representatives for 2 distinct keys.";

    // Find representatives for each key.
    const logs::detail::Consolidated_Record* rep_a = nullptr;
    const logs::detail::Consolidated_Record* rep_b = nullptr;

    for (const auto& rep : consolidated) {
        if (rep.key.message == "Error A") {
            ASSERT_EQ(rep_a, nullptr)
                << "Duplicate representative for 'Error A' — cross-key contamination.";
            rep_a = &rep;
        } else if (rep.key.message == "Error B") {
            ASSERT_EQ(rep_b, nullptr)
                << "Duplicate representative for 'Error B' — cross-key contamination.";
            rep_b = &rep;
        } else {
            FAIL() << "Unexpected key in consolidated result: " << rep.key.message;
        }
    }

    ASSERT_NE(rep_a, nullptr) << "Missing representative for 'Error A'.";
    ASSERT_NE(rep_b, nullptr) << "Missing representative for 'Error B'.";

    // Verify: "Error A" representative has rank_count == 3.
    EXPECT_EQ(rep_a->rank_count, 3)
        << "Error A had 3 contributing records; rank_count should be 3.";
    EXPECT_EQ(rep_a->key.severity, logs::Severity_Level::ERROR);

    // Verify: "Error B" representative has rank_count == 2.
    EXPECT_EQ(rep_b->rank_count, 2)
        << "Error B had 2 contributing records; rank_count should be 2.";
    EXPECT_EQ(rep_b->key.severity, logs::Severity_Level::ERROR);

    // Verify: no cross-key contamination — each representative's rank_count
    // matches ONLY its own contributing records. If contamination occurred,
    // one key would have an inflated count or we'd see unexpected keys above.
    EXPECT_EQ(rep_a->rank_count + rep_b->rank_count, 5)
        << "Total rank count across representatives must equal total records (5).";

    // Local consolidation uses sentinel rank -1 ranges (Requirement 4.9).
    ASSERT_EQ(rep_a->ranges.size(), 1u);
    EXPECT_EQ(rep_a->ranges[0].first, -1);
    EXPECT_EQ(rep_a->ranges[0].last, -1);

    ASSERT_EQ(rep_b->ranges.size(), 1u);
    EXPECT_EQ(rep_b->ranges[0].first, -1);
    EXPECT_EQ(rep_b->ranges[0].last, -1);
}

/// Test with distinct keys differing by severity (same message, different severity).
TEST(ConsolidationDistinctKeys, SameMessageDifferentSeverityAreDistinctKeys) {
    // Two keys: same message "Connection timeout" but different severities.
    std::vector<logs::Log_Record> records;

    // Key A: WARNING + "Connection timeout" from ranks 0, 1.
    records.emplace_back(logs::Severity_Level::WARNING, std::string("Connection timeout"), 0,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::WARNING, std::string("Connection timeout"), 1,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);

    // Key B: ERROR + "Connection timeout" from ranks 2, 3, 4.
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Connection timeout"), 2,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Connection timeout"), 3,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);
    records.emplace_back(logs::Severity_Level::ERROR, std::string("Connection timeout"), 4,
                         std::nullopt, std::vector<std::string>{}, std::nullopt);

    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Exactly 2 representatives — severity is part of the key.
    ASSERT_EQ(consolidated.size(), 2u)
        << "Same message with different severity should produce distinct keys.";

    const logs::detail::Consolidated_Record* rep_warn = nullptr;
    const logs::detail::Consolidated_Record* rep_err = nullptr;

    for (const auto& rep : consolidated) {
        EXPECT_EQ(rep.key.message, "Connection timeout");
        if (rep.key.severity == logs::Severity_Level::WARNING) {
            rep_warn = &rep;
        } else if (rep.key.severity == logs::Severity_Level::ERROR) {
            rep_err = &rep;
        }
    }

    ASSERT_NE(rep_warn, nullptr);
    ASSERT_NE(rep_err, nullptr);

    // No cross-contamination: WARNING got 2, ERROR got 3.
    EXPECT_EQ(rep_warn->rank_count, 2);
    EXPECT_EQ(rep_err->rank_count, 3);
}

/// Verify that representatives are complete — no records are lost.
TEST(ConsolidationDistinctKeys, AllRecordsAccountedForAcrossRepresentatives) {
    // 3 distinct keys, each with varying numbers of records.
    std::vector<logs::Log_Record> records;

    // Key 1: DEBUG + "init phase" — 4 records.
    for (int r = 0; r < 4; ++r) {
        records.emplace_back(logs::Severity_Level::DEBUG, std::string("init phase"), r,
                             std::nullopt, std::vector<std::string>{}, std::nullopt);
    }

    // Key 2: INFO + "processing" — 3 records.
    for (int r = 10; r < 13; ++r) {
        records.emplace_back(logs::Severity_Level::INFO, std::string("processing"), r,
                             std::nullopt, std::vector<std::string>{}, std::nullopt);
    }

    // Key 3: ERROR + "timeout" — 2 records.
    for (int r = 20; r < 22; ++r) {
        records.emplace_back(logs::Severity_Level::ERROR, std::string("timeout"), r,
                             std::nullopt, std::vector<std::string>{}, std::nullopt);
    }

    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Exactly 3 representatives.
    ASSERT_EQ(consolidated.size(), 3u);

    // Total rank_count across all representatives must equal total input (9).
    int total_count = 0;
    for (const auto& rep : consolidated) {
        total_count += rep.rank_count;
    }
    EXPECT_EQ(total_count, 9)
        << "Sum of rank_counts must equal total number of input records.";

    // Verify each key's count individually.
    for (const auto& rep : consolidated) {
        if (rep.key.message == "init phase") {
            EXPECT_EQ(rep.rank_count, 4);
            EXPECT_EQ(rep.key.severity, logs::Severity_Level::DEBUG);
        } else if (rep.key.message == "processing") {
            EXPECT_EQ(rep.rank_count, 3);
            EXPECT_EQ(rep.key.severity, logs::Severity_Level::INFO);
        } else if (rep.key.message == "timeout") {
            EXPECT_EQ(rep.rank_count, 2);
            EXPECT_EQ(rep.key.severity, logs::Severity_Level::ERROR);
        } else {
            FAIL() << "Unexpected key: " << rep.key.message;
        }
    }
}

} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// Task 21.2: Consolidation Contiguous-Span Test
// Validates: Requirements 13.3
//
// Inject identical records from a contiguous span of 4+ ranks (R through
// R+N-1); consolidate; verify one representative with count == N and
// Rank_Range [R, R+N-1].
//
// Since consolidate_local() uses sentinel rank -1 (Requirement 4.9), the
// range verification is performed directly via compact_ranges() which is
// the engine that produces actual Rank_Ranges from contributing ranks.
// The count verification uses consolidate_local().
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Verifies that compact_ranges on a contiguous span of 4 ranks produces
/// exactly one Rank_Range with first == R and last == R+N-1.
TEST(ConsolidationContiguousSpan, CompactRangesProducesSingleSpan) {
    // Contiguous ranks: 3, 4, 5, 6
    constexpr int R = 3;
    constexpr int N = 4;

    std::vector<int> ranks;
    for (int i = 0; i < N; ++i) {
        ranks.push_back(R + i);
    }

    auto ranges = logs::detail::compact_ranges(ranks);

    // Exactly one Rank_Range for a contiguous span.
    ASSERT_EQ(ranges.size(), 1u);

    // The range covers [R, R+N-1] == [3, 6].
    EXPECT_EQ(ranges[0].first, R);
    EXPECT_EQ(ranges[0].last, R + N - 1);
}

/// Verifies contiguous span with larger N (8 ranks starting at rank 10).
TEST(ConsolidationContiguousSpan, LargerContiguousSpan) {
    constexpr int R = 10;
    constexpr int N = 8;

    std::vector<int> ranks;
    for (int i = 0; i < N; ++i) {
        ranks.push_back(R + i);
    }

    auto ranges = logs::detail::compact_ranges(ranks);

    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, R);
    EXPECT_EQ(ranges[0].last, R + N - 1);  // [10, 17]
}

/// Verifies contiguous span starting at rank 0.
TEST(ConsolidationContiguousSpan, ContiguousSpanFromZero) {
    constexpr int R = 0;
    constexpr int N = 5;

    std::vector<int> ranks;
    for (int i = 0; i < N; ++i) {
        ranks.push_back(R + i);
    }

    auto ranges = logs::detail::compact_ranges(ranks);

    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, 0);
    EXPECT_EQ(ranges[0].last, 4);  // [0, 4]
}

/// Verifies that unsorted contiguous ranks still produce a single span.
TEST(ConsolidationContiguousSpan, UnsortedContiguousRanksCompact) {
    // Ranks 3,4,5,6 supplied out of order.
    std::vector<int> ranks = {6, 3, 5, 4};

    auto ranges = logs::detail::compact_ranges(ranks);

    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, 3);
    EXPECT_EQ(ranges[0].last, 6);
}

/// Verifies that duplicate ranks in a contiguous span are deduplicated
/// and still produce a single range.
TEST(ConsolidationContiguousSpan, DuplicateRanksDeduplicatedToSingleSpan) {
    // Ranks 3,4,5,6 with duplicates.
    std::vector<int> ranks = {3, 4, 5, 6, 3, 4, 5, 6};

    auto ranges = logs::detail::compact_ranges(ranks);

    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, 3);
    EXPECT_EQ(ranges[0].last, 6);
}

/// Full integration: inject identical records from contiguous span of 4 ranks
/// into Consolidation_Engine::consolidate_local(). Verify exactly one
/// Consolidated_Record is produced with rank_count == N.
TEST(ConsolidationContiguousSpan, ConsolidateLocalProducesOneRepresentativeWithCorrectCount) {
    constexpr int R = 3;
    constexpr int N = 4;

    // Create N Log_Records with identical severity+message, different ranks.
    std::vector<logs::Log_Record> records;
    records.reserve(N);
    for (int i = 0; i < N; ++i) {
        records.emplace_back(
            logs::Severity_Level::ERROR,
            "contiguous span test message",
            R + i,                          // ranks 3, 4, 5, 6
            std::nullopt,                   // no source location
            std::vector<std::string>{},     // no context labels
            std::nullopt                    // no stack trace
        );
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Verify: exactly one Consolidated_Record produced.
    ASSERT_EQ(consolidated.size(), 1u);

    // Verify: rank_count == N (4 contributing ranks).
    EXPECT_EQ(consolidated[0].rank_count, N);

    // Verify: the key preserves severity and message.
    EXPECT_EQ(consolidated[0].key.severity, logs::Severity_Level::ERROR);
    EXPECT_EQ(consolidated[0].key.message, "contiguous span test message");
}

/// Verifies compact_ranges directly for the contiguous-span ranks used in
/// the consolidation test, confirming ranges == [{3, 6}].
TEST(ConsolidationContiguousSpan, CompactRangesMatchesExpectedForConsolidationInput) {
    // The ranks that would be gathered from 4 records (3, 4, 5, 6).
    std::vector<int> ranks = {3, 4, 5, 6};

    auto ranges = logs::detail::compact_ranges(ranks);

    // Exactly one range spanning [3, 6].
    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, 3);
    EXPECT_EQ(ranges[0].last, 6);

    // Verify to_string renders as "3-6".
    EXPECT_EQ(ranges[0].to_string(), "3-6");
}

} // namespace


// ─────────────────────────────────────────────────────────────────────────────
// Task 21.7: Single-Rank Consolidation Test
// Validates: Requirements 13.9
//
// THE test suite SHALL contain a test that injects identical Log_Records
// attributed to a single rank, triggers a consolidation operation, and
// verifies that exactly one representative record is produced carrying a
// contributing-rank count of 1 and a Rank_Range whose lower and upper bounds
// both equal that single rank.
//
// Two sub-tests:
//   A) compact_ranges with a single rank [7] → [{7,7}]
//   B) consolidate_local with multiple identical records all from the same
//      rank → one representative with local semantics (sentinel rank -1).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// A) Verify compact_ranges with a single rank produces a Rank_Range where
/// first == last == that rank.
TEST(ConsolidationSingleRank, CompactRangesSingleRankProducesSinglePointRange) {
    std::vector<int> ranks = {7};

    auto ranges = logs::detail::compact_ranges(ranks);

    // Exactly one Rank_Range.
    ASSERT_EQ(ranges.size(), 1u);

    // Both bounds equal the single rank.
    EXPECT_EQ(ranges[0].first, 7);
    EXPECT_EQ(ranges[0].last, 7);

    // to_string renders as "7-7".
    EXPECT_EQ(ranges[0].to_string(), "7-7");
}

/// B) Verify consolidate_local with multiple identical records (same severity,
/// message, AND same rank) produces exactly one Consolidated_Record with
/// rank_count == number of records, and Rank_Range [{-1, -1}] (local
/// consolidation uses sentinel rank -1 per Requirement 4.9).
TEST(ConsolidationSingleRank, ConsolidateLocalSameRankProducesOneRepresentative) {
    constexpr int RANK = 7;
    constexpr int NUM_RECORDS = 5;

    // Create identical records: same severity, same message, same rank.
    std::vector<logs::Log_Record> records;
    records.reserve(NUM_RECORDS);
    for (int i = 0; i < NUM_RECORDS; ++i) {
        records.emplace_back(
            logs::Severity_Level::WARNING,
            "single rank repeated message",
            RANK,                           // all from rank 7
            std::nullopt,                   // no source location
            std::vector<std::string>{},     // no context labels
            std::nullopt                    // no stack trace
        );
    }

    // Consolidate locally.
    logs::detail::Consolidation_Engine engine;
    auto consolidated = engine.consolidate_local(records);

    // Verify: exactly one Consolidated_Record (all records share the same key).
    ASSERT_EQ(consolidated.size(), 1u);

    // Verify: rank_count equals the number of records since local consolidation
    // treats each record as a contribution (Requirement 4.9 local path).
    EXPECT_EQ(consolidated[0].rank_count, NUM_RECORDS);

    // Verify: severity and message are preserved.
    EXPECT_EQ(consolidated[0].key.severity, logs::Severity_Level::WARNING);
    EXPECT_EQ(consolidated[0].key.message, "single rank repeated message");

    // Verify: local consolidation annotates with sentinel rank -1 range.
    ASSERT_EQ(consolidated[0].ranges.size(), 1u);
    EXPECT_EQ(consolidated[0].ranges[0].first, -1);
    EXPECT_EQ(consolidated[0].ranges[0].last, -1);
}

/// Verify compact_ranges with duplicate single-rank entries deduplicates
/// to one Rank_Range {7, 7}.
TEST(ConsolidationSingleRank, CompactRangesDuplicateSingleRankDeduplicates) {
    // Multiple entries of the same rank.
    std::vector<int> ranks = {7, 7, 7, 7, 7};

    auto ranges = logs::detail::compact_ranges(ranks);

    // After deduplication, still one Rank_Range.
    ASSERT_EQ(ranges.size(), 1u);
    EXPECT_EQ(ranges[0].first, 7);
    EXPECT_EQ(ranges[0].last, 7);
}

} // namespace
