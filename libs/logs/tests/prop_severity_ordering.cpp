/// @file prop_severity_ordering.cpp
/// @brief Property-based tests for Severity_Level ordering and label mapping.
///
/// Uses RapidCheck + Google Test to verify universal correctness properties
/// over all Severity_Level values.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <logs/severity.hpp>
#include <set>
#include <string>
#include <string_view>

namespace {

/// Generate a random valid Severity_Level (int 0–4 cast to enum).
rc::Gen<logs::Severity_Level> genSeverityLevel() {
    return rc::gen::map(rc::gen::inRange(0, 5), [](int v) { return static_cast<logs::Severity_Level>(v); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 1: Severity Level Total Ordering
// Validates: Requirements 1.1
// ─────────────────────────────────────────────────────────────────────────────

/// For any two Severity_Level values, exactly one of a<b, a==b, a>b holds
/// (total ordering / trichotomy).
RC_GTEST_PROP(SeverityOrdering, Trichotomy, ()) {
    const auto a = *genSeverityLevel();
    const auto b = *genSeverityLevel();

    int count = 0;
    if (a < b) ++count;
    if (a == b) ++count;
    if (a > b) ++count;

    RC_ASSERT(count == 1);
}

/// Antisymmetric: if a <= b and b <= a then a == b.
RC_GTEST_PROP(SeverityOrdering, Antisymmetric, ()) {
    const auto a = *genSeverityLevel();
    const auto b = *genSeverityLevel();

    if (a <= b && b <= a) {
        RC_ASSERT(a == b);
    }
}

/// Transitive: if a <= b and b <= c then a <= c.
RC_GTEST_PROP(SeverityOrdering, Transitive, ()) {
    const auto a = *genSeverityLevel();
    const auto b = *genSeverityLevel();
    const auto c = *genSeverityLevel();

    if (a <= b && b <= c) {
        RC_ASSERT(a <= c);
    }
}

/// The ordering is consistent with the spec: DEBUG < INFO < WARNING < ERROR < FATAL.
RC_GTEST_PROP(SeverityOrdering, ConsistentWithSpecOrder, ()) {
    // Generate a random level and verify it obeys the fixed chain.
    const auto level = *genSeverityLevel();

    // Every level must satisfy its position in the chain.
    switch (level) {
        case logs::Severity_Level::DEBUG:
            RC_ASSERT(level < logs::Severity_Level::INFO);
            RC_ASSERT(level < logs::Severity_Level::WARNING);
            RC_ASSERT(level < logs::Severity_Level::ERROR);
            RC_ASSERT(level < logs::Severity_Level::FATAL);
            break;
        case logs::Severity_Level::INFO:
            RC_ASSERT(logs::Severity_Level::DEBUG < level);
            RC_ASSERT(level < logs::Severity_Level::WARNING);
            RC_ASSERT(level < logs::Severity_Level::ERROR);
            RC_ASSERT(level < logs::Severity_Level::FATAL);
            break;
        case logs::Severity_Level::WARNING:
            RC_ASSERT(logs::Severity_Level::DEBUG < level);
            RC_ASSERT(logs::Severity_Level::INFO < level);
            RC_ASSERT(level < logs::Severity_Level::ERROR);
            RC_ASSERT(level < logs::Severity_Level::FATAL);
            break;
        case logs::Severity_Level::ERROR:
            RC_ASSERT(logs::Severity_Level::DEBUG < level);
            RC_ASSERT(logs::Severity_Level::INFO < level);
            RC_ASSERT(logs::Severity_Level::WARNING < level);
            RC_ASSERT(level < logs::Severity_Level::FATAL);
            break;
        case logs::Severity_Level::FATAL:
            RC_ASSERT(logs::Severity_Level::DEBUG < level);
            RC_ASSERT(logs::Severity_Level::INFO < level);
            RC_ASSERT(logs::Severity_Level::WARNING < level);
            RC_ASSERT(logs::Severity_Level::ERROR < level);
            break;
    }
}

/// Exhaustive verification of the complete ordering chain.
TEST(SeverityOrdering, FullChainDeterministic) {
    using S = logs::Severity_Level;
    EXPECT_LT(S::DEBUG, S::INFO);
    EXPECT_LT(S::INFO, S::WARNING);
    EXPECT_LT(S::WARNING, S::ERROR);
    EXPECT_LT(S::ERROR, S::FATAL);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 2: Severity Label Mapping Is Fixed and Total
// Validates: Requirements 1.6
// ─────────────────────────────────────────────────────────────────────────────

/// For any valid Severity_Level, to_string returns the exact mandated label.
RC_GTEST_PROP(SeverityLabelMapping, FixedLabel, ()) {
    const auto level = *genSeverityLevel();

    const std::string_view label = logs::to_string(level);

    switch (level) {
        case logs::Severity_Level::DEBUG:
            RC_ASSERT(label == "DEBUG");
            break;
        case logs::Severity_Level::INFO:
            RC_ASSERT(label == "INFO");
            break;
        case logs::Severity_Level::WARNING:
            RC_ASSERT(label == "WARNING");
            break;
        case logs::Severity_Level::ERROR:
            RC_ASSERT(label == "ERROR");
            break;
        case logs::Severity_Level::FATAL:
            RC_ASSERT(label == "FATAL");
            break;
    }
}

/// The mapping is total: every valid enum value produces a non-empty result.
RC_GTEST_PROP(SeverityLabelMapping, TotalMapping, ()) {
    const auto level = *genSeverityLevel();

    const std::string_view label = logs::to_string(level);

    RC_ASSERT(!label.empty());
}

/// Distinct levels always map to distinct labels (injectivity).
RC_GTEST_PROP(SeverityLabelMapping, DistinctLevelsDistinctLabels, ()) {
    const auto a = *genSeverityLevel();
    const auto b = *genSeverityLevel();

    RC_PRE(a != b);

    RC_ASSERT(logs::to_string(a) != logs::to_string(b));
}

/// Exhaustive check: all five labels are unique (deterministic complement
/// to the randomized injectivity property above).
TEST(SeverityLabelMapping, AllFiveLabelsDistinct) {
    std::set<std::string_view> labels;
    labels.insert(logs::to_string(logs::Severity_Level::DEBUG));
    labels.insert(logs::to_string(logs::Severity_Level::INFO));
    labels.insert(logs::to_string(logs::Severity_Level::WARNING));
    labels.insert(logs::to_string(logs::Severity_Level::ERROR));
    labels.insert(logs::to_string(logs::Severity_Level::FATAL));

    EXPECT_EQ(labels.size(), 5u);
}

}  // namespace
