#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <tick/tick.hpp>
#include <tick/gregorian_calendar.hpp>
#include <tick/noleap_calendar.hpp>
#include <tick/cal360_calendar.hpp>

#include "generators.hpp"

#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using namespace tick;

// Convenience type aliases
using GregorianEngine = Aliasing_Engine<Gregorian_Calendar, Gregorian_Calendar>;
using CrossCalEngine  = Aliasing_Engine<Gregorian_Calendar, NoLeap_Calendar>;

// Helper: create a valid coverage window (10 years, divisible by common intervals)
// 3650 days from epoch — evenly divisible by many intervals (e.g., 5, 10, 25, 50, 73, 146, 365, 730, 1825)
inline Time_Window make_coverage() {
    return Time_Window{Time_Point{0}, Time_Point{days(3650).nanos()}};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 5.1: Constructor Validation Unit Tests
// Requirements: 7.5, 7.6, 7.7, 7.8
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AliasingEngine_Constructor, ZeroDurationInterval_Throws) {
    EXPECT_THROW(
        GregorianEngine(make_coverage(), Duration{0}, OutOfBoundsPolicy::clamp_to_edge),
        std::invalid_argument);
}

TEST(AliasingEngine_Constructor, NegativeInterval_Throws) {
    EXPECT_THROW(
        GregorianEngine(make_coverage(), Duration{-1}, OutOfBoundsPolicy::clamp_to_edge),
        std::invalid_argument);
}

TEST(AliasingEngine_Constructor, NonDivisibleInterval_Throws) {
    // 3650 days is not divisible by 7 days (3650 % 7 != 0)
    EXPECT_THROW(
        GregorianEngine(make_coverage(), days(7), OutOfBoundsPolicy::clamp_to_edge),
        std::invalid_argument);
}

TEST(AliasingEngine_Constructor, PureClimatology_OutOfRangeYear_Throws) {
    // Coverage spans epoch year 2026 to ~2035. Climatological year 1990 is out of range.
    auto coverage = make_coverage();
    auto interval = days(10);
    EXPECT_THROW(
        GregorianEngine(coverage, interval, OutOfBoundsPolicy::pure_climatology, 1990),
        std::invalid_argument);
}

TEST(AliasingEngine_Constructor, PureClimatology_YearAfterCoverage_Throws) {
    // Year 2050 is well beyond coverage end (~2035)
    auto coverage = make_coverage();
    auto interval = days(10);
    EXPECT_THROW(
        GregorianEngine(coverage, interval, OutOfBoundsPolicy::pure_climatology, 2050),
        std::invalid_argument);
}

TEST(AliasingEngine_Constructor, ValidConstruction_AccessorRoundTrip) {
    auto coverage = make_coverage();
    auto interval = days(10);
    auto policy = OutOfBoundsPolicy::cycle_last_year;
    std::int32_t clim_year = 0;

    GregorianEngine engine{coverage, interval, policy, clim_year};

    EXPECT_EQ(engine.coverage(), coverage);
    EXPECT_EQ(engine.snapshot_interval(), interval);
    EXPECT_EQ(engine.policy(), policy);
    EXPECT_EQ(engine.climatological_year(), clim_year);
}

TEST(AliasingEngine_Constructor, ValidConstruction_PureClimatology_WithinRange) {
    // Year 2028 is within coverage [2026, ~2035]
    auto coverage = make_coverage();
    auto interval = days(10);

    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::pure_climatology, 2028};

    EXPECT_EQ(engine.coverage(), coverage);
    EXPECT_EQ(engine.snapshot_interval(), interval);
    EXPECT_EQ(engine.policy(), OutOfBoundsPolicy::pure_climatology);
    EXPECT_EQ(engine.climatological_year(), 2028);
}

TEST(AliasingEngine_Constructor, ValidConstruction_CrossCalendar) {
    // Cross-calendar engine (Gregorian sim → NoLeap dataset) should also construct fine
    auto coverage = make_coverage();
    auto interval = days(50);  // 3650 / 50 = 73, evenly divisible

    CrossCalEngine engine{coverage, interval, OutOfBoundsPolicy::leap_hold};

    EXPECT_EQ(engine.coverage(), coverage);
    EXPECT_EQ(engine.snapshot_interval(), interval);
    EXPECT_EQ(engine.policy(), OutOfBoundsPolicy::leap_hold);
    EXPECT_EQ(engine.climatological_year(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.2: Property 1 — Weight Formula Correctness
// Requirements: 1.1, 1.2, 1.3
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property1, WeightFormulaCorrectness, ()) {
    // **Validates: Requirements 1.1, 1.2, 1.3**

    // Generate a random Time_Window with positive duration
    auto start_nanos = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000'000'000LL,
                                                        1'000'000'000'000'000'000LL);
    auto duration_nanos = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL * 365);

    auto start = Time_Point{start_nanos};
    auto end = Time_Point{start_nanos + duration_nanos};
    auto window = Time_Window{start, end};

    // Generate a random Time_Point within [start, end)
    auto offset = *rc::gen::inRange<std::int64_t>(0, duration_nanos);
    auto current = Time_Point{start_nanos + offset};

    // Compute expected alpha manually
    double expected = static_cast<double>(current.nanos() - window.start().nanos())
                    / static_cast<double>(window.end().nanos() - window.start().nanos());

    // Compute actual via calculate_weight
    double actual = GregorianEngine::calculate_weight(current, window);

    RC_ASSERT(actual == expected);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.3: Property 2 — Out-of-Range Rejection
// Requirements: 1.5
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property2, OutOfRangeRejection, ()) {
    // Generate a random Time_Window with positive duration
    auto start_nanos = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000'000'000LL,
                                                        1'000'000'000'000'000'000LL);
    auto duration_nanos = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL * 365);

    auto start = Time_Point{start_nanos};
    auto end = Time_Point{start_nanos + duration_nanos};
    auto window = Time_Window{start, end};

    // Generate a Time_Point outside [start, end)
    auto choice = *rc::gen::inRange<int>(0, 2);
    Time_Point current;
    if (choice == 0) {
        // Before start
        auto before_offset = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL * 30);
        current = Time_Point{start_nanos - before_offset};
    } else {
        // At or after end
        auto after_offset = *rc::gen::inRange<std::int64_t>(0, 86'400'000'000'000LL * 30);
        current = Time_Point{start_nanos + duration_nanos + after_offset};
    }

    RC_ASSERT_THROWS_AS(GregorianEngine::calculate_weight(current, window), std::out_of_range);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.5: Property 4 — Clamp-to-Edge Boundary Freeze
// Requirements: 3.1, 3.2
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property4, ClampToEdgeBoundaryFreeze, ()) {
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();
    auto interval = *snapshot_interval_for(coverage);

    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::clamp_to_edge};

    // Generate out-of-bounds Time_Point
    auto sim_time = *time_point_outside_coverage(coverage);

    auto result = engine.resolve(sim_time);

    // Alpha must be 0.0 (frozen)
    RC_ASSERT(result.alpha == 0.0);

    // Window must be first or last snapshot interval
    if (sim_time >= coverage.end()) {
        // Should clamp to last window
        auto expected_end = coverage.end();
        auto expected_start = coverage.end() - interval;
        RC_ASSERT(result.window.start() == expected_start);
        RC_ASSERT(result.window.end() == expected_end);
    } else {
        // Should clamp to first window
        auto expected_start = coverage.start();
        auto expected_end = coverage.start() + interval;
        RC_ASSERT(result.window.start() == expected_start);
        RC_ASSERT(result.window.end() == expected_end);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.4: Property 3 — Weight-Bounds Invariant
// Requirements: 7.3, 8.2, 11.1
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property3, WeightBoundsInvariant, ()) {
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();
    auto interval = *snapshot_interval_for(coverage);

    // Pick a random policy
    auto policy_idx = *rc::gen::inRange<int>(0, 4);
    auto policy = static_cast<OutOfBoundsPolicy>(policy_idx);

    // For pure_climatology, need a valid climatological year
    std::int32_t clim_year = 0;
    if (policy == OutOfBoundsPolicy::pure_climatology) {
        auto start_dt = Gregorian_Calendar::to_date_time(coverage.start());
        auto end_dt = Gregorian_Calendar::to_date_time(coverage.end() - Duration{1});
        clim_year = *rc::gen::inRange<std::int32_t>(start_dt.year, end_dt.year + 1);
    }

    GregorianEngine engine{coverage, interval, policy, clim_year};

    // Generate any Time_Point (in or out of coverage)
    auto sim_time_nanos = *rc::gen::inRange<std::int64_t>(
        coverage.start().nanos() - 86'400'000'000'000LL * 3650,
        coverage.end().nanos() + 86'400'000'000'000LL * 3650);
    auto sim_time = Time_Point{sim_time_nanos};

    auto result = engine.resolve(sim_time);

    RC_ASSERT(result.alpha >= 0.0);
    RC_ASSERT(result.alpha <= 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.6: Property 5 — Cycle-Last-Year Target Year Containment
// Requirements: 4.1, 4.3, 4.4, 11.3
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property5, CycleLastYearTargetYearContainment, ()) {
    // **Validates: Requirements 4.1, 4.3, 4.4, 11.3**
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();
    auto interval = *snapshot_interval_for(coverage);

    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::cycle_last_year};

    // Generate out-of-bounds Time_Point
    auto sim_time = *time_point_outside_coverage(coverage);

    auto result = engine.resolve(sim_time);

    // Determine the expected target year
    std::int32_t target_year;
    if (sim_time >= coverage.end()) {
        target_year = Gregorian_Calendar::to_date_time(coverage.end() - Duration{1}).year;
    } else {
        target_year = Gregorian_Calendar::to_date_time(coverage.start()).year;
    }

    // Both t_left and t_right should decompose to the target year
    auto left_dt = Gregorian_Calendar::to_date_time(result.window.start());
    auto right_dt = Gregorian_Calendar::to_date_time(result.window.end());

    RC_ASSERT(left_dt.year == target_year);
    // t_right can be in target_year or target_year + 1 (Dec→Jan boundary)
    RC_ASSERT(right_dt.year == target_year || right_dt.year == target_year + 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.7: Property 6 — Pure-Climatology Year-Lock
// Requirements: 5.1, 5.3, 5.5, 11.4
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property6, PureClimatologyYearLock, ()) {
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();
    auto interval = *snapshot_interval_for(coverage);

    // Pick a valid climatological year within coverage
    auto start_dt = Gregorian_Calendar::to_date_time(coverage.start());
    auto end_dt = Gregorian_Calendar::to_date_time(coverage.end() - Duration{1});
    auto clim_year = *rc::gen::inRange<std::int32_t>(start_dt.year, end_dt.year + 1);

    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::pure_climatology, clim_year};

    // Generate arbitrary Time_Point (in or out of coverage range)
    auto sim_time_nanos = *rc::gen::inRange<std::int64_t>(
        coverage.start().nanos() - 86'400'000'000'000LL * 3650,
        coverage.end().nanos() + 86'400'000'000'000LL * 3650);
    auto sim_time = Time_Point{sim_time_nanos};

    auto result = engine.resolve(sim_time);

    // Both endpoints should decompose to climatological_year or clim_year + 1
    auto left_dt = Gregorian_Calendar::to_date_time(result.window.start());
    auto right_dt = Gregorian_Calendar::to_date_time(result.window.end());

    RC_ASSERT(left_dt.year == clim_year || left_dt.year == clim_year + 1);
    RC_ASSERT(right_dt.year == clim_year || right_dt.year == clim_year + 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.8: Property 7 — Leap-Hold Freeze
// Requirements: 6.1, 11.5
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property7, LeapHoldFreeze, ()) {
    using namespace tick::gen;

    // Need coverage that encompasses the Feb 29 date we'll generate
    // Use a wide coverage from epoch (2026) spanning 10 years
    auto coverage = make_coverage();
    auto interval = days(50);  // 3650 / 50 = 73, evenly divisible

    CrossCalEngine engine{coverage, interval, OutOfBoundsPolicy::leap_hold};

    // Generate a Feb 29 Time_Point (in Gregorian calendar)
    auto sim_time = *feb29_time_point();

    auto result = engine.resolve(sim_time);

    // Alpha must be 0.0 (frozen on leap day)
    RC_ASSERT(result.alpha == 0.0);

    // t_left should be Feb 28 00:00:00.000000000 (in NoLeap calendar)
    auto left_dt = NoLeap_Calendar::to_date_time(result.window.start());
    RC_ASSERT(left_dt.month == 2);
    RC_ASSERT(left_dt.day == 28);
    RC_ASSERT(left_dt.hour == 0);
    RC_ASSERT(left_dt.minute == 0);
    RC_ASSERT(left_dt.second == 0);
    RC_ASSERT(left_dt.nanosecond == 0);

    // t_right should be Mar 1 00:00:00.000000000
    auto right_dt = NoLeap_Calendar::to_date_time(result.window.end());
    RC_ASSERT(right_dt.month == 3);
    RC_ASSERT(right_dt.day == 1);
    RC_ASSERT(right_dt.hour == 0);
    RC_ASSERT(right_dt.minute == 0);
    RC_ASSERT(right_dt.second == 0);
    RC_ASSERT(right_dt.nanosecond == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.9: Property 8 — Leap-Hold Fallback Equivalence
// Requirements: 6.2
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property8, LeapHoldFallbackEquivalence, ()) {
    // **Validates: Requirements 6.2**
    using namespace tick::gen;

    auto coverage = make_coverage();
    auto interval = days(50);  // 3650 / 50 = 73, evenly divisible

    CrossCalEngine leap_engine{coverage, interval, OutOfBoundsPolicy::leap_hold};
    CrossCalEngine clamp_engine{coverage, interval, OutOfBoundsPolicy::clamp_to_edge};

    // Generate a non-Feb-29 Time_Point
    auto sim_time = *non_feb29_time_point();

    auto leap_result = leap_engine.resolve(sim_time);
    auto clamp_result = clamp_engine.resolve(sim_time);

    // Results should be identical for non-Feb-29 dates
    RC_ASSERT(leap_result == clamp_result);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.10: Property 9 — Weight Monotonicity
// Requirements: 11.6
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property9, WeightMonotonicity, ()) {
    // **Validates: Requirements 11.6**

    // Generate a random Time_Window with positive duration
    auto start_nanos = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000'000'000LL,
                                                        1'000'000'000'000'000'000LL);
    auto duration_nanos = *rc::gen::inRange<std::int64_t>(2, 86'400'000'000'000LL * 365);

    auto start = Time_Point{start_nanos};
    auto end = Time_Point{start_nanos + duration_nanos};
    auto window = Time_Window{start, end};

    // Generate two ordered points A < B within [start, end)
    auto offset_a = *rc::gen::inRange<std::int64_t>(0, duration_nanos - 1);
    auto offset_b = *rc::gen::inRange<std::int64_t>(offset_a, duration_nanos);

    auto point_a = Time_Point{start_nanos + offset_a};
    auto point_b = Time_Point{start_nanos + offset_b};

    RC_PRE(point_a <= point_b);

    double alpha_a = GregorianEngine::calculate_weight(point_a, window);
    double alpha_b = GregorianEngine::calculate_weight(point_b, window);

    RC_ASSERT(alpha_a <= alpha_b);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.11: Property 10 — Resolve Determinism
// Requirements: 9.2, 11.2
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property10, ResolveDeterminism, ()) {
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();
    auto interval = *snapshot_interval_for(coverage);

    // Pick a random policy
    auto policy_idx = *rc::gen::inRange<int>(0, 4);
    auto policy = static_cast<OutOfBoundsPolicy>(policy_idx);

    // For pure_climatology, need a valid climatological year
    std::int32_t clim_year = 0;
    if (policy == OutOfBoundsPolicy::pure_climatology) {
        auto start_dt = Gregorian_Calendar::to_date_time(coverage.start());
        auto end_dt = Gregorian_Calendar::to_date_time(coverage.end() - Duration{1});
        clim_year = *rc::gen::inRange<std::int32_t>(start_dt.year, end_dt.year + 1);
    }

    GregorianEngine engine{coverage, interval, policy, clim_year};

    // Generate any Time_Point
    auto sim_time_nanos = *rc::gen::inRange<std::int64_t>(
        coverage.start().nanos() - 86'400'000'000'000LL * 3650,
        coverage.end().nanos() + 86'400'000'000'000LL * 3650);
    auto sim_time = Time_Point{sim_time_nanos};

    // Call resolve twice
    auto result1 = engine.resolve(sim_time);
    auto result2 = engine.resolve(sim_time);

    // Bitwise-identical via operator==
    RC_ASSERT(result1 == result2);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Section 4.12: Property 11 — Construction Rejects Non-Divisible Intervals
// Requirements: 7.5
// ═══════════════════════════════════════════════════════════════════════════════

RC_GTEST_PROP(AliasingEngine_Property11, ConstructionRejectsNonDivisibleIntervals, ()) {
    using namespace tick::gen;

    auto coverage = *aliasing_coverage();

    // Generate a positive interval that does NOT evenly divide coverage
    auto cov_nanos = coverage.duration().nanos();
    auto interval_nanos = *rc::gen::inRange<std::int64_t>(1, cov_nanos);

    RC_PRE(cov_nanos % interval_nanos != 0);

    auto interval = Duration{interval_nanos};

    RC_ASSERT_THROWS_AS(
        GregorianEngine(coverage, interval, OutOfBoundsPolicy::clamp_to_edge),
        std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 5.2: calculate_weight Edge Cases Unit Tests
// Requirements: 1.2, 1.3, 1.5, 1.6
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AliasingEngine_CalculateWeight, AtWindowStart_ReturnsZero) {
    auto window = Time_Window{Time_Point{1000}, Time_Point{2000}};
    auto alpha = GregorianEngine::calculate_weight(Time_Point{1000}, window);
    EXPECT_DOUBLE_EQ(alpha, 0.0);
}

TEST(AliasingEngine_CalculateWeight, OneNanosecondBeforeEnd_LessThanOne) {
    auto window = Time_Window{Time_Point{0}, Time_Point{1'000'000'000}};  // 1 second
    auto alpha = GregorianEngine::calculate_weight(Time_Point{999'999'999}, window);
    EXPECT_LT(alpha, 1.0);
    EXPECT_GT(alpha, 0.0);
}

TEST(AliasingEngine_CalculateWeight, OutsideWindow_ThrowsOutOfRange) {
    auto window = Time_Window{Time_Point{100}, Time_Point{200}};

    // Before start
    EXPECT_THROW(GregorianEngine::calculate_weight(Time_Point{50}, window), std::out_of_range);

    // At end (end is exclusive)
    EXPECT_THROW(GregorianEngine::calculate_weight(Time_Point{200}, window), std::out_of_range);

    // After end
    EXPECT_THROW(GregorianEngine::calculate_weight(Time_Point{300}, window), std::out_of_range);
}

TEST(AliasingEngine_CalculateWeight, ZeroDurationWindow_ThrowsInvalidArgument) {
    // Time_Window constructor enforces start < end, so a zero-duration window cannot be created.
    // This tests that the Time_Window constructor throws.
    EXPECT_THROW(Time_Window(Time_Point{100}, Time_Point{100}), std::invalid_argument);
}

TEST(AliasingEngine_CalculateWeight, MidpointReturnsHalf) {
    auto window = Time_Window{Time_Point{0}, Time_Point{1000}};
    auto alpha = GregorianEngine::calculate_weight(Time_Point{500}, window);
    EXPECT_DOUBLE_EQ(alpha, 0.5);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Section 5.3: Policy-Specific Edge Cases Unit Tests
// Requirements: 4.2, 5.3, 6.3, 8.3
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AliasingEngine_PolicyEdgeCases, CycleLastYear_Feb29_ClampedToFeb28) {
    // Coverage: 2026-01-01 to 2036-01-01 in NoLeap (3650 days)
    auto coverage = make_coverage();
    auto interval = days(10);

    // Cross-calendar: Gregorian sim → NoLeap dataset with cycle_last_year
    CrossCalEngine engine{coverage, interval, OutOfBoundsPolicy::cycle_last_year};

    // Simulate Feb 29, 2040 in Gregorian (a leap year, but beyond coverage)
    auto sim_time = Gregorian_Calendar::to_time_point(Date_Time{2040, 2, 29, 12, 0, 0, 0});
    auto result = engine.resolve(sim_time);

    // The remapped time should have day clamped to Feb 28 (NoLeap has no Feb 29)
    auto left_dt = NoLeap_Calendar::to_date_time(result.window.start());
    EXPECT_LE(left_dt.day, 28);
    EXPECT_EQ(left_dt.month, 2);
}

TEST(AliasingEngine_PolicyEdgeCases, PureClimatology_Dec31NearMidnight_YearBoundaryWrap) {
    // Create coverage spanning 2026-2036 with a 5-day interval
    auto start_tp = Gregorian_Calendar::to_time_point(Date_Time{2026, 1, 1, 0, 0, 0, 0});
    auto end_tp = Gregorian_Calendar::to_time_point(Date_Time{2036, 1, 1, 0, 0, 0, 0});
    auto coverage = Time_Window{start_tp, end_tp};
    auto interval = days(5);

    // Check divisibility first: this coverage might not divide by 5 evenly
    // Use a simpler coverage: epoch-based, 3650 days
    auto cov = make_coverage();
    GregorianEngine engine{cov, days(5), OutOfBoundsPolicy::pure_climatology, 2028};

    // Simulate Dec 31 23:00 in year 2050 — should map to climatological year 2028
    auto sim_time = Gregorian_Calendar::to_time_point(Date_Time{2050, 12, 31, 23, 0, 0, 0});
    auto result = engine.resolve(sim_time);

    // The window should be anchored in climatological year
    auto left_dt = Gregorian_Calendar::to_date_time(result.window.start());
    auto right_dt = Gregorian_Calendar::to_date_time(result.window.end());

    // t_left should be in year 2028 (December)
    EXPECT_EQ(left_dt.year, 2028);
    // t_right could be in 2028 or 2029 (year-boundary wrap)
    EXPECT_TRUE(right_dt.year == 2028 || right_dt.year == 2029);
}

TEST(AliasingEngine_PolicyEdgeCases, LeapHold_NonGregorianNoLeap_BehavesAsClamp) {
    // Gregorian→Gregorian with leap_hold: should behave as clamp_to_edge
    // since leap_hold special behavior only triggers for Gregorian→NoLeap
    auto coverage = make_coverage();
    auto interval = days(10);

    GregorianEngine leap_engine{coverage, interval, OutOfBoundsPolicy::leap_hold};
    GregorianEngine clamp_engine{coverage, interval, OutOfBoundsPolicy::clamp_to_edge};

    // Test with Feb 29 (Gregorian sim, Gregorian dataset — no special handling)
    auto sim_time = Gregorian_Calendar::to_time_point(Date_Time{2028, 2, 29, 12, 0, 0, 0});

    auto leap_result = leap_engine.resolve(sim_time);
    auto clamp_result = clamp_engine.resolve(sim_time);

    EXPECT_EQ(leap_result, clamp_result);
}

TEST(AliasingEngine_PolicyEdgeCases, AliasedWindow_EqualityIdentical) {
    auto window = Time_Window{Time_Point{100}, Time_Point{200}};
    AliasedWindow a{window, 0.5};
    AliasedWindow b{window, 0.5};
    EXPECT_EQ(a, b);
}

TEST(AliasingEngine_PolicyEdgeCases, AliasedWindow_DifferentAlpha_NotEqual) {
    auto window = Time_Window{Time_Point{100}, Time_Point{200}};
    AliasedWindow a{window, 0.5};
    AliasedWindow b{window, 0.500000000001};  // Different bits
    EXPECT_NE(a, b);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Section 5.4: Compile-Time and Thread-Safety Verification Tests
// Requirements: 2.3, 7.4, 9.3
// ═══════════════════════════════════════════════════════════════════════════════

// Compile-time: OutOfBoundsPolicy has uint8_t underlying type
static_assert(std::is_same_v<std::underlying_type_t<OutOfBoundsPolicy>, std::uint8_t>,
              "OutOfBoundsPolicy must have std::uint8_t underlying type");

// Compile-time: Aliasing_Engine cannot be instantiated with a non-Calendar type
// A struct that does NOT satisfy the Calendar concept:
struct NotACalendar {};
// Verify the Calendar concept rejects NotACalendar — since Aliasing_Engine is
// constrained by `Calendar Sim_Cal`, this proves the template cannot be instantiated.
static_assert(!Calendar<NotACalendar>,
              "NotACalendar must not satisfy the Calendar concept, "
              "proving Aliasing_Engine rejects non-Calendar types at compile time");

TEST(AliasingEngine_ThreadSafety, ConcurrentResolve_ProducesConsistentResults) {
    auto coverage = make_coverage();
    auto interval = days(10);
    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::clamp_to_edge};

    auto sim_time = Time_Point{days(500).nanos()};  // In-bounds

    constexpr int num_threads = 8;
    std::vector<AliasedWindow> results(num_threads, AliasedWindow::zero());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&engine, &results, i, sim_time]() {
            results[i] = engine.resolve(sim_time);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All results should be bitwise-identical
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(results[0], results[i])
            << "Thread " << i << " produced a different result";
    }
}

TEST(AliasingEngine_ThreadSafety, ConcurrentResolve_OutOfBounds_Consistent) {
    auto coverage = make_coverage();
    auto interval = days(10);
    GregorianEngine engine{coverage, interval, OutOfBoundsPolicy::cycle_last_year};

    // Out-of-bounds time point
    auto sim_time = Time_Point{days(5000).nanos()};

    constexpr int num_threads = 8;
    std::vector<AliasedWindow> results(num_threads, AliasedWindow::zero());
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&engine, &results, i, sim_time]() {
            results[i] = engine.resolve(sim_time);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(results[0], results[i])
            << "Thread " << i << " produced a different result";
    }
}

} // anonymous namespace
