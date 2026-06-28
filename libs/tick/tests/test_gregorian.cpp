#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "tick/date_time.hpp"
#include "tick/gregorian_calendar.hpp"
#include "tick/time_point.hpp"

// ═══════════════════════════════════════════════════════════════════════════════
// Property-Based Tests — Task 4.5
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Property 2: Gregorian calendar round-trip ───────────────────────────────
// For any valid Gregorian Date_Time dt:
//   Gregorian_Calendar::to_date_time(Gregorian_Calendar::to_time_point(dt)) == dt
// **Validates: Requirements 3.1, 3.2, 3.3**

RC_GTEST_PROP(GregorianCalendar, RoundTrip, ()) {
    // Generate valid Gregorian Date_Time within nanosecond representable range
    // ±292 years from epoch 2026 → approximately [1734, 2318]
    auto year = *rc::gen::inRange<std::int32_t>(1734, 2319);
    auto month = *rc::gen::inRange<std::int32_t>(1, 13);
    auto max_day = tick::Gregorian_Calendar::days_in_month(year, month);
    auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
    auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
    auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
    auto second = *rc::gen::inRange<std::int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);

    tick::Date_Time dt{year, month, day, hour, minute, second, nanosecond};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);
    auto rt = tick::Gregorian_Calendar::to_date_time(tp);
    RC_ASSERT(rt == dt);
}

// Also test the reverse direction: for any Time_Point tp,
// to_time_point(to_date_time(tp)) == tp
RC_GTEST_PROP(GregorianCalendar, RoundTripReverse, ()) {
    // Use a range that stays within representable dates (~±292 years from epoch)
    auto nanos = *rc::gen::inRange<std::int64_t>(-static_cast<std::int64_t>(200) * 365 * 86'400'000'000'000LL,
                                                 static_cast<std::int64_t>(200) * 365 * 86'400'000'000'000LL);
    auto tp = tick::Time_Point{nanos};
    auto dt = tick::Gregorian_Calendar::to_date_time(tp);
    auto rt = tick::Gregorian_Calendar::to_time_point(dt);
    RC_ASSERT(rt == tp);
}

// ─── Property 5: Gregorian leap year matches mathematical definition ─────────
// **Validates: Requirements 3.1, 3.4**

RC_GTEST_PROP(GregorianCalendar, LeapYearMatchesMath, ()) {
    auto year = *rc::gen::inRange<std::int32_t>(-10000, 10001);
    bool expected = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    RC_ASSERT(tick::Gregorian_Calendar::is_leap_year(year) == expected);
}

// ─── Property 6: Calendar-month addition preserves validity and advances correctly
// **Validates: Requirements 3.7, 11.1**

RC_GTEST_PROP(GregorianCalendar, MonthAdditionPreservesValidity, ()) {
    // Constrain year range to stay within representable nanosecond range after month addition
    auto year = *rc::gen::inRange<std::int32_t>(1740, 2312);
    auto month = *rc::gen::inRange<std::int32_t>(1, 13);
    auto max_day = tick::Gregorian_Calendar::days_in_month(year, month);
    auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
    tick::Date_Time dt{year, month, day, 12, 0, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);

    auto n = *rc::gen::inRange<std::int32_t>(-24, 25);
    auto result_tp = tick::Gregorian_Calendar::add_months(tp, n);

    // Result must be a valid date (round-trip succeeds)
    auto result_dt = tick::Gregorian_Calendar::to_date_time(result_tp);
    auto verify_tp = tick::Gregorian_Calendar::to_time_point(result_dt);
    RC_ASSERT(verify_tp == result_tp);

    // Verify month was advanced correctly using floor division (matching implementation)
    std::int32_t total_months = (dt.year * 12 + (dt.month - 1)) + n;
    // Floor division for negative values
    std::int32_t expected_year = total_months >= 0 ? total_months / 12 : (total_months - 11) / 12;
    std::int32_t expected_month = total_months - expected_year * 12 + 1;

    RC_ASSERT(result_dt.year == expected_year);
    RC_ASSERT(result_dt.month == expected_month);

    // Day must be clamped to at most days_in_month for target month
    auto target_max_day = tick::Gregorian_Calendar::days_in_month(expected_year, expected_month);
    RC_ASSERT(result_dt.day >= 1);
    RC_ASSERT(result_dt.day <= target_max_day);

    // If original day was within target month's range, it should be preserved
    if (dt.day <= target_max_day) {
        RC_ASSERT(result_dt.day == dt.day);
    }
}

// ─── Property 8: Calendar-year addition equals 12 calendar-months ────────────
// **Validates: Requirements 3.7, 11.1**

RC_GTEST_PROP(GregorianCalendar, YearAdditionEquals12Months, ()) {
    // Use a range that stays representable after ±5 years addition
    auto year = *rc::gen::inRange<std::int32_t>(1740, 2312);
    auto month = *rc::gen::inRange<std::int32_t>(1, 13);
    auto max_day = tick::Gregorian_Calendar::days_in_month(year, month);
    auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
    tick::Date_Time dt{year, month, day, 6, 30, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);

    auto n = *rc::gen::inRange<std::int32_t>(-5, 6);
    auto by_years = tick::Gregorian_Calendar::add_years(tp, n);
    auto by_months = tick::Gregorian_Calendar::add_months(tp, n * 12);
    RC_ASSERT(by_years == by_months);
}

// ─── Gregorian Edge Cases: Century Leap Year Boundaries ─────────────────────

TEST(GregorianEdgeCases, Year2000IsLeap) {
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2000));
    EXPECT_EQ(tick::Gregorian_Calendar::days_in_month(2000, 2), 29);
}

TEST(GregorianEdgeCases, Year1900IsNotLeap) {
    EXPECT_FALSE(tick::Gregorian_Calendar::is_leap_year(1900));
    EXPECT_EQ(tick::Gregorian_Calendar::days_in_month(1900, 2), 28);
}

TEST(GregorianEdgeCases, Year2100IsNotLeap) {
    EXPECT_FALSE(tick::Gregorian_Calendar::is_leap_year(2100));
}

TEST(GregorianEdgeCases, Year2400IsLeap) {
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2400));
    EXPECT_EQ(tick::Gregorian_Calendar::days_in_month(2400, 2), 29);
}

// ─── Gregorian Edge Cases: Round-Trip for Known Dates ────────────────────────

TEST(GregorianEdgeCases, RoundTrip20000229) {
    tick::Date_Time dt{2000, 2, 29, 12, 30, 45, 123456789};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);
    auto rt = tick::Gregorian_Calendar::to_date_time(tp);
    EXPECT_EQ(rt, dt);
}

TEST(GregorianEdgeCases, RoundTrip19000301) {
    tick::Date_Time dt{1900, 3, 1, 0, 0, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);
    auto rt = tick::Gregorian_Calendar::to_date_time(tp);
    EXPECT_EQ(rt, dt);
}

TEST(GregorianEdgeCases, RoundTrip21000301) {
    tick::Date_Time dt{2100, 3, 1, 0, 0, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);
    auto rt = tick::Gregorian_Calendar::to_date_time(tp);
    EXPECT_EQ(rt, dt);
}

TEST(GregorianEdgeCases, RoundTrip24000229) {
    // Year 2400 is outside the ±292-year nanosecond representable range from epoch 2026.
    // Verify is_leap_year(2400) instead; round-trip a year within range.
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2400));
    EXPECT_EQ(tick::Gregorian_Calendar::days_in_month(2400, 2), 29);

    // Use 2300-02-28 (within representable range, 2300 is NOT a leap year)
    tick::Date_Time dt{2300, 2, 28, 23, 59, 59, 999999999};
    auto tp = tick::Gregorian_Calendar::to_time_point(dt);
    auto rt = tick::Gregorian_Calendar::to_date_time(tp);
    EXPECT_EQ(rt, dt);
}

// ─── Gregorian Edge Cases: Invalid Components Throw ──────────────────────────

TEST(GregorianEdgeCases, InvalidMonth) {
    tick::Date_Time dt{2026, 13, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(GregorianEdgeCases, InvalidDay) {
    tick::Date_Time dt{2023, 2, 29, 0, 0, 0, 0};  // 2023 is not a leap year
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(GregorianEdgeCases, InvalidHour) {
    tick::Date_Time dt{2026, 1, 1, 24, 0, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(GregorianEdgeCases, InvalidMinute) {
    tick::Date_Time dt{2026, 6, 15, 12, 60, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(GregorianEdgeCases, InvalidSecond) {
    tick::Date_Time dt{2026, 6, 15, 12, 30, 60, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(dt), std::invalid_argument);
}

// ============================================================
// Example-based unit tests — Task 4.8
// ============================================================

// Known Gregorian dates
TEST(GregorianUnit, KnownDates) {
    // 2000-02-29 (leap year, divisible by 400)
    tick::Date_Time d1{2000, 2, 29, 0, 0, 0, 0};
    auto tp1 = tick::Gregorian_Calendar::to_time_point(d1);
    EXPECT_EQ(tick::Gregorian_Calendar::to_date_time(tp1), d1);

    // 1900-03-01 (1900 is NOT a leap year)
    tick::Date_Time d2{1900, 3, 1, 0, 0, 0, 0};
    auto tp2 = tick::Gregorian_Calendar::to_time_point(d2);
    EXPECT_EQ(tick::Gregorian_Calendar::to_date_time(tp2), d2);

    // 2100-03-01 (2100 is NOT a leap year, century not div by 400)
    tick::Date_Time d3{2100, 3, 1, 0, 0, 0, 0};
    auto tp3 = tick::Gregorian_Calendar::to_time_point(d3);
    EXPECT_EQ(tick::Gregorian_Calendar::to_date_time(tp3), d3);

    // Verify 2400 is leap even though it's outside representable nanosecond range
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2400));
    EXPECT_EQ(tick::Gregorian_Calendar::days_in_month(2400, 2), 29);
}

// Century leap year rules
TEST(GregorianUnit, CenturyLeapRules) {
    EXPECT_FALSE(tick::Gregorian_Calendar::is_leap_year(1900));
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2000));
    EXPECT_FALSE(tick::Gregorian_Calendar::is_leap_year(2100));
    EXPECT_TRUE(tick::Gregorian_Calendar::is_leap_year(2400));
}

// Invalid dates throw
TEST(GregorianUnit, InvalidDateThrows) {
    // Feb 29 in non-leap year
    tick::Date_Time bad1{1900, 2, 29, 0, 0, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(bad1), std::invalid_argument);

    // Month 13
    tick::Date_Time bad2{2026, 13, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(bad2), std::invalid_argument);

    // Hour 24
    tick::Date_Time bad3{2026, 1, 1, 24, 0, 0, 0};
    EXPECT_THROW(tick::Gregorian_Calendar::to_time_point(bad3), std::invalid_argument);
}
