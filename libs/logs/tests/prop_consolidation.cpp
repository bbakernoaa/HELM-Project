/// @file prop_consolidation.cpp
/// @brief Property-based tests for consolidation types and compact_ranges.
///
/// Uses RapidCheck + Google Test to verify universal correctness properties
/// for Consolidation_Key equality and Rank_Range compaction.

#include <logs/detail/consolidation.hpp>

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Property 14: Consolidated Rank Partitioning Into Ordered Contiguous Spans
// Validates: Requirements 4.3, 4.4, 4.6
//
// For any set of contributing ranks, verify compact_ranges produces maximal
// contiguous spans covering exactly those ranks, non-overlapping, non-adjacent,
// ordered ascending.
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: collect the union of all integers covered by a set of Rank_Ranges.
static std::set<int> union_of_ranges(const std::vector<logs::detail::Rank_Range>& ranges) {
    std::set<int> result;
    for (const auto& r : ranges) {
        for (int v = r.first; v <= r.last; ++v) {
            result.insert(v);
        }
    }
    return result;
}

/// Helper: deduplicate and sort a vector of ints (to compute expected coverage).
static std::set<int> deduplicated_set(const std::vector<int>& ranks) {
    return std::set<int>(ranks.begin(), ranks.end());
}

/// Covering: the union of all ranges equals the deduplicated sorted input.
RC_GTEST_PROP(CompactRanges,
              Covering,
              ()) {
    const auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-100, 100));

    const auto ranges = logs::detail::compact_ranges(ranks);

    const auto expected = deduplicated_set(ranks);
    const auto actual   = union_of_ranges(ranges);

    RC_ASSERT(actual == expected);
}

/// Maximal: no two adjacent ranges could be merged (there's a gap > 1 between
/// consecutive ranges).
RC_GTEST_PROP(CompactRanges,
              Maximal,
              ()) {
    const auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-100, 100));

    RC_PRE(!ranks.empty());

    const auto ranges = logs::detail::compact_ranges(ranks);

    for (std::size_t i = 1; i < ranges.size(); ++i) {
        // If two consecutive ranges were adjacent (gap == 1), they should have
        // been merged into one. A gap of exactly 1 means they're mergeable.
        RC_ASSERT(ranges[i].first > ranges[i - 1].last + 1);
    }
}

/// Non-overlapping: ranges don't overlap.
RC_GTEST_PROP(CompactRanges,
              NonOverlapping,
              ()) {
    const auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-100, 100));

    RC_PRE(!ranks.empty());

    const auto ranges = logs::detail::compact_ranges(ranks);

    for (std::size_t i = 1; i < ranges.size(); ++i) {
        // The start of range[i] must be strictly after the end of range[i-1].
        RC_ASSERT(ranges[i].first > ranges[i - 1].last);
    }
}

/// Ordered ascending: ranges are sorted by first rank.
RC_GTEST_PROP(CompactRanges,
              OrderedAscending,
              ()) {
    const auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-100, 100));

    RC_PRE(!ranks.empty());

    const auto ranges = logs::detail::compact_ranges(ranks);

    for (std::size_t i = 1; i < ranges.size(); ++i) {
        RC_ASSERT(ranges[i].first > ranges[i - 1].first);
    }
}

/// Contiguous within spans: for each range, all ints from first to last are
/// present in the original input (after deduplication).
RC_GTEST_PROP(CompactRanges,
              ContiguousWithinSpans,
              ()) {
    const auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-100, 100));

    RC_PRE(!ranks.empty());

    const auto ranges     = logs::detail::compact_ranges(ranks);
    const auto rank_set   = deduplicated_set(ranks);

    for (const auto& r : ranges) {
        RC_ASSERT(r.first <= r.last);
        for (int v = r.first; v <= r.last; ++v) {
            RC_ASSERT(rank_set.count(v) == 1);
        }
    }
}

/// Empty input produces empty output.
TEST(CompactRanges, EmptyInputProducesEmptyOutput) {
    const auto ranges = logs::detail::compact_ranges({});
    EXPECT_TRUE(ranges.empty());
}

/// Single rank produces a single-element range [r, r].
RC_GTEST_PROP(CompactRanges,
              SingleRankSingleRange,
              ()) {
    const auto r = *rc::gen::inRange(-1000, 1000);

    const auto ranges = logs::detail::compact_ranges({r});

    RC_ASSERT(ranges.size() == 1u);
    RC_ASSERT(ranges[0].first == r);
    RC_ASSERT(ranges[0].last == r);
}

/// Duplicate ranks don't affect the result (idempotent with respect to
/// deduplication).
RC_GTEST_PROP(CompactRanges,
              DuplicatesIdempotent,
              ()) {
    auto ranks = *rc::gen::container<std::vector<int>>(
        rc::gen::inRange(-50, 50));

    RC_PRE(!ranks.empty());

    // Double the input (add duplicates).
    auto doubled = ranks;
    doubled.insert(doubled.end(), ranks.begin(), ranks.end());

    const auto ranges_original = logs::detail::compact_ranges(ranks);
    const auto ranges_doubled  = logs::detail::compact_ranges(doubled);

    RC_ASSERT(ranges_original.size() == ranges_doubled.size());
    for (std::size_t i = 0; i < ranges_original.size(); ++i) {
        RC_ASSERT(ranges_original[i].first == ranges_doubled[i].first);
        RC_ASSERT(ranges_original[i].last == ranges_doubled[i].last);
    }
}

} // namespace
