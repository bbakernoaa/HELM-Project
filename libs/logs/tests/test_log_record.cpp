/// @file test_log_record.cpp
/// @brief Property-based tests for Log_Record immutable value object.
///
/// Uses RapidCheck + Google Test to verify universal correctness properties
/// over arbitrary Log_Record constructions.

#include <logs/log_record.hpp>
#include <logs/source_location.hpp>

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace {

/// Generate a random valid Severity_Level (int 0–4 cast to enum).
rc::Gen<logs::Severity_Level> genSeverityLevel() {
    return rc::gen::map(rc::gen::inRange(0, 5), [](int v) {
        return static_cast<logs::Severity_Level>(v);
    });
}

/// Generate an arbitrary Source_Location with line >= 1.
rc::Gen<logs::Source_Location> genSourceLocation() {
    return rc::gen::exec([]() {
        logs::Source_Location loc;
        loc.file     = *rc::gen::arbitrary<std::string>();
        loc.line     = *rc::gen::inRange(1, 100000);
        loc.function = *rc::gen::arbitrary<std::string>();
        return loc;
    });
}

/// Generate an optional Source_Location (roughly 50% present).
rc::Gen<std::optional<logs::Source_Location>> genOptionalLocation() {
    return rc::gen::exec([]() -> std::optional<logs::Source_Location> {
        const bool present = *rc::gen::arbitrary<bool>();
        if (present) {
            return *genSourceLocation();
        }
        return std::nullopt;
    });
}

/// Generate an optional stack-trace string (roughly 50% present).
rc::Gen<std::optional<std::string>> genOptionalStackTrace() {
    return rc::gen::exec([]() -> std::optional<std::string> {
        const bool present = *rc::gen::arbitrary<bool>();
        if (present) {
            return *rc::gen::arbitrary<std::string>();
        }
        return std::nullopt;
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 3: Log_Record Construction Preserves Fields Immutably
// Validates: Requirements 1.2, 1.3
// ─────────────────────────────────────────────────────────────────────────────

/// For any severity, message, rank, optional location, context-label sequence,
/// and optional stack trace, verify all accessors return values equal to
/// constructor inputs with message preserved byte-for-byte.
RC_GTEST_PROP(LogRecordConstruction,
              PreservesAllFields,
              ()) {
    const auto severity       = *genSeverityLevel();
    const auto message        = *rc::gen::arbitrary<std::string>();
    const auto rank           = *rc::gen::arbitrary<int>();
    const auto location       = *genOptionalLocation();
    const auto context_labels = *rc::gen::arbitrary<std::vector<std::string>>();
    const auto stack_trace    = *genOptionalStackTrace();

    const logs::Log_Record record{
        severity, message, rank, location, context_labels, stack_trace
    };

    // Severity preserved
    RC_ASSERT(record.severity() == severity);

    // Message preserved byte-for-byte
    RC_ASSERT(record.message() == message);
    RC_ASSERT(record.message().size() == message.size());

    // Rank preserved
    RC_ASSERT(record.rank() == rank);

    // Context labels preserved (order and content)
    RC_ASSERT(record.context_labels() == context_labels);

    // Stack trace preserved
    RC_ASSERT(record.stack_trace() == stack_trace);

    // Location preserved
    if (location.has_value()) {
        RC_ASSERT(record.location().has_value());
        RC_ASSERT(record.location()->file == location->file);
        RC_ASSERT(record.location()->line == location->line);
        RC_ASSERT(record.location()->function == location->function);
    } else {
        RC_ASSERT(!record.location().has_value());
    }
}

/// Verify message is preserved byte-for-byte including embedded nulls and
/// arbitrary byte values.
RC_GTEST_PROP(LogRecordConstruction,
              MessageByteForBytePreservation,
              ()) {
    const auto message = *rc::gen::arbitrary<std::string>();
    const auto rank    = *rc::gen::arbitrary<int>();

    const logs::Log_Record record{
        logs::Severity_Level::INFO, message, rank,
        std::nullopt, {}, std::nullopt
    };

    // Byte-for-byte equality: same length and same content
    RC_ASSERT(record.message().size() == message.size());
    for (std::size_t i = 0; i < message.size(); ++i) {
        RC_ASSERT(record.message()[i] == message[i]);
    }
}

/// Verify that copying a Log_Record preserves all fields identically
/// (immutability guarantee).
RC_GTEST_PROP(LogRecordConstruction,
              CopyPreservesFields,
              ()) {
    const auto severity       = *genSeverityLevel();
    const auto message        = *rc::gen::arbitrary<std::string>();
    const auto rank           = *rc::gen::arbitrary<int>();
    const auto location       = *genOptionalLocation();
    const auto context_labels = *rc::gen::arbitrary<std::vector<std::string>>();
    const auto stack_trace    = *genOptionalStackTrace();

    const logs::Log_Record original{
        severity, message, rank, location, context_labels, stack_trace
    };

    const logs::Log_Record copy{original};  // NOLINT(performance-unnecessary-copy-initialization)

    RC_ASSERT(copy.severity() == original.severity());
    RC_ASSERT(copy.message() == original.message());
    RC_ASSERT(copy.rank() == original.rank());
    RC_ASSERT(copy.context_labels() == original.context_labels());
    RC_ASSERT(copy.stack_trace() == original.stack_trace());

    if (original.location().has_value()) {
        RC_ASSERT(copy.location().has_value());
        RC_ASSERT(copy.location()->file == original.location()->file);
        RC_ASSERT(copy.location()->line == original.location()->line);
        RC_ASSERT(copy.location()->function == original.location()->function);
    } else {
        RC_ASSERT(!copy.location().has_value());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 4: Absent Source Location Is Never Fabricated
// Validates: Requirements 1.7
// ─────────────────────────────────────────────────────────────────────────────

/// For any Log_Record constructed without source-location (std::nullopt),
/// verify the stored location is std::nullopt — never fabricated.
RC_GTEST_PROP(LogRecordSourceLocation,
              AbsentLocationIsNeverFabricated,
              ()) {
    const auto severity       = *genSeverityLevel();
    const auto message        = *rc::gen::arbitrary<std::string>();
    const auto rank           = *rc::gen::arbitrary<int>();
    const auto context_labels = *rc::gen::arbitrary<std::vector<std::string>>();
    const auto stack_trace    = *genOptionalStackTrace();

    // Construct with std::nullopt as the location argument.
    const logs::Log_Record record{
        severity, message, rank, std::nullopt, context_labels, stack_trace};

    // The stored location must be absent — never fabricated.
    RC_ASSERT(record.location() == std::nullopt);
    RC_ASSERT(record.location().has_value() == false);
}

} // namespace
