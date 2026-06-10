#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>
#include <tick/noleap_calendar.hpp>
#include <cstdint>
#include <stdexcept>

// ============================================================
// Property-based tests (RapidCheck) — Task 4.6
// ============================================================

// **Validates: Requirements 4.4, 11.2**

// --------------------------------------------------------------------------
// Property 3: NoLeap calendar round-trip
// For any valid NoLeap Date_Time dt (Feb always 28 days):
//   NoLeap_Calendar::to_date_time(NoLeap_Calendar::to_time_point(dt)) == dt
// --------------------------------------------------------------------------

RC_GTEST_PROP(NoLeapCalendar, RoundTrip, ()) {
    // Epoch is 2026. With nanosecond resolution, int64_t supports ~±292 years.
    // Use ±290 years from epoch to stay safely within representable range.
    auto year = *rc::gen::inRange<std::int32_t>(1736, 2317);
    auto month = *rc::gen::inRange<std::int32_t>(1, 13);
    auto max_day = tick::NoLeap_Calendar::days_in_month(year, month);
    auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
    auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
    auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
    auto second = *rc::gen::inRange<std::int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);

    tick::Date_Time dt{year, month, day, hour, minute, second, nanosecond};
    auto tp = tick::NoLeap_Calendar::to_time_point(dt);
    auto rt = tick::NoLeap_Calendar::to_date_time(tp);
    RC_ASSERT(rt == dt);
}

// Reverse round-trip: Time_Point → Date_Time → Time_Point
RC_GTEST_PROP(NoLeapCalendar, RoundTripReverse, ()) {
    // ±290 years from epoch in nanoseconds (safely within int64_t range)
    constexpr std::int64_t max_nanos = static_cast<std::int64_t>(290) * 365 * 86'400'000'000'000LL;
    auto nanos = *rc::gen::inRange<std::int64_t>(-max_nanos, max_nanos);
    auto tp = tick::Time_Point{nanos};
    auto dt = tick::NoLeap_Calendar::to_date_time(tp);
    auto rt = tick::NoLeap_Calendar::to_time_point(dt);
    RC_ASSERT(rt == tp);
}

// Verify that each year is exactly 365 days (Requirement 4.7)
RC_GTEST_PROP(NoLeapCalendar, YearIsExactly365Days, ()) {
    // Use range that keeps both year and year+1 within representable nanosecond range
    auto year = *rc::gen::inRange<std::int32_t>(1736, 2316);
    tick::Date_Time jan1{year, 1, 1, 0, 0, 0, 0};
    tick::Date_Time jan1_next{year + 1, 1, 1, 0, 0, 0, 0};
    auto tp1 = tick::NoLeap_Calendar::to_time_point(jan1);
    auto tp2 = tick::NoLeap_Calendar::to_time_point(jan1_next);
    auto diff = tp2 - tp1;
    RC_ASSERT(diff.nanos() == 365LL * 86'400'000'000'000LL);
}

// ============================================================
// Example-based unit tests (Google Test) — existing edge cases
// ============================================================

// ─── NoLeap Edge Cases: Feb 29 Rejection ─────────────────────────────────────

TEST(NoLeapEdgeCases, RejectsFeb29) {
    tick::Date_Time dt{2024, 2, 29, 0, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(NoLeapEdgeCases, RejectsFeb29AnyYear) {
    // Even years that would be leap in Gregorian are rejected
    tick::Date_Time dt{2000, 2, 29, 12, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(dt), std::invalid_argument);
}

// ─── NoLeap Edge Cases: Invalid Components Throw ─────────────────────────────

TEST(NoLeapEdgeCases, InvalidMonthZero) {
    tick::Date_Time dt{2026, 0, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(NoLeapEdgeCases, InvalidMonthThirteen) {
    tick::Date_Time dt{2026, 13, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(NoLeapEdgeCases, InvalidHour) {
    tick::Date_Time dt{2026, 1, 1, 24, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(dt), std::invalid_argument);
}

// ─── NoLeap Edge Cases: Year Always 365 Days ─────────────────────────────────

TEST(NoLeapEdgeCases, Year365Days) {
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_year(2024), 365);
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_year(1900), 365);
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_year(2000), 365);
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_year(2100), 365);
}

// ─── NoLeap Edge Cases: Feb Always 28 Days ───────────────────────────────────

TEST(NoLeapEdgeCases, FebAlways28Days) {
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_month(2024, 2), 28);
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_month(2000, 2), 28);
    EXPECT_EQ(tick::NoLeap_Calendar::days_in_month(1900, 2), 28);
}

// ============================================================
// Example-based unit tests — Task 4.8
// ============================================================

TEST(NoLeapUnit, RejectsFeb29) {
    tick::Date_Time bad{2000, 2, 29, 0, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(bad), std::invalid_argument);
}

TEST(NoLeapUnit, InvalidMonth) {
    tick::Date_Time bad{2026, 0, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::NoLeap_Calendar::to_time_point(bad), std::invalid_argument);
}
