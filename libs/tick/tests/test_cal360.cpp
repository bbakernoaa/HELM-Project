// test_cal360.cpp — Cal360 calendar property-based tests (Task 4.7) + unit tests (Task 4.8)
// Property 4: Cal360 calendar round-trip
// Property 7: Cal360 month addition is a fixed nanosecond shift
// **Validates: Requirements 5.4, 5.8, 5.9, 11.3**

#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "tick/cal360_calendar.hpp"
#include "tick/date_time.hpp"
#include "tick/time_point.hpp"

// ============================================================
// Property-based tests (RapidCheck) — Task 4.7
// ============================================================

// --------------------------------------------------------------------------
// Property 4: Cal360 calendar round-trip
// For any valid Cal360 Date_Time dt, to_date_time(to_time_point(dt)) == dt
// --------------------------------------------------------------------------

RC_GTEST_PROP(Cal360Calendar, RoundTrip, ()) {
    // ±292 years from epoch 2026 is the representable range for int64_t nanoseconds.
    // INT64_MAX / nanos_per_day / 360 ≈ 296, so [1734, 2319) is safe.
    auto year = *rc::gen::inRange<std::int32_t>(1734, 2319);
    auto month = *rc::gen::inRange<std::int32_t>(1, 13);
    auto day = *rc::gen::inRange<std::int32_t>(1, 31);  // 1-30 (exclusive upper)
    auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
    auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
    auto second = *rc::gen::inRange<std::int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);

    tick::Date_Time dt{year, month, day, hour, minute, second, nanosecond};
    auto tp = tick::Cal360_Calendar::to_time_point(dt);
    auto roundtrip = tick::Cal360_Calendar::to_date_time(tp);
    RC_ASSERT(roundtrip == dt);
}

// Reverse direction: for any Time_Point tp, to_time_point(to_date_time(tp)) == tp
RC_GTEST_PROP(Cal360Calendar, RoundTripReverse, ()) {
    // ±292 years of 360 days in nanoseconds ≈ ±9.2e18
    // Use a safe subset: ±280 years → 280*360*86400e9 ≈ 8.7e18
    constexpr std::int64_t bound = 280LL * 360 * tick::nanos_per_day;
    auto nanos = *rc::gen::inRange<std::int64_t>(-bound, bound);
    auto tp = tick::Time_Point{nanos};
    auto dt = tick::Cal360_Calendar::to_date_time(tp);
    auto roundtrip = tick::Cal360_Calendar::to_time_point(dt);
    RC_ASSERT(roundtrip == tp);
}

// --------------------------------------------------------------------------
// Property 7: Cal360 month addition is a fixed nanosecond shift
// add_months(tp, n).nanos() == tp.nanos() + n * 30 * nanos_per_day
// --------------------------------------------------------------------------

RC_GTEST_PROP(Cal360Calendar, MonthAdditionIsFixedShift, ()) {
    auto nanos = *rc::gen::inRange<std::int64_t>(-10'000'000'000'000'000LL, 10'000'000'000'000'000LL);
    auto tp = tick::Time_Point{nanos};
    auto n = *rc::gen::inRange<std::int32_t>(-120, 121);
    auto result = tick::Cal360_Calendar::add_months(tp, n);
    std::int64_t expected = nanos + static_cast<std::int64_t>(n) * 30 * tick::nanos_per_day;
    RC_ASSERT(result.nanos() == expected);
}

// --------------------------------------------------------------------------
// Req 5.8: Year-to-year offset is exactly 360 days
// --------------------------------------------------------------------------

RC_GTEST_PROP(Cal360Calendar, YearIsExactly360Days, ()) {
    auto year = *rc::gen::inRange<std::int32_t>(1734, 2318);
    tick::Date_Time jan1{year, 1, 1, 0, 0, 0, 0};
    tick::Date_Time jan1_next{year + 1, 1, 1, 0, 0, 0, 0};
    auto tp1 = tick::Cal360_Calendar::to_time_point(jan1);
    auto tp2 = tick::Cal360_Calendar::to_time_point(jan1_next);
    auto diff = tp2 - tp1;
    RC_ASSERT(diff.nanos() == 360LL * tick::nanos_per_day);
}

// --------------------------------------------------------------------------
// Req 5.9: Month-to-month offset is exactly 30 days
// --------------------------------------------------------------------------

RC_GTEST_PROP(Cal360Calendar, MonthIsExactly30Days, ()) {
    auto year = *rc::gen::inRange<std::int32_t>(1734, 2319);
    auto month = *rc::gen::inRange<std::int32_t>(1, 12);  // [1,11] so month+1 is valid
    tick::Date_Time d1{year, month, 1, 0, 0, 0, 0};
    tick::Date_Time d2{year, static_cast<std::int32_t>(month + 1), 1, 0, 0, 0, 0};
    auto tp1 = tick::Cal360_Calendar::to_time_point(d1);
    auto tp2 = tick::Cal360_Calendar::to_time_point(d2);
    auto diff = tp2 - tp1;
    RC_ASSERT(diff.nanos() == 30LL * tick::nanos_per_day);
}

// ============================================================
// Example-based unit tests (Google Test) — Task 4.8
// ============================================================

// ─── Cal360 Edge Cases: Uniform 30-Day Months ────────────────────────────────

TEST(Cal360EdgeCases, UniformMonths) {
    for (int m = 1; m <= 12; ++m) {
        EXPECT_EQ(tick::Cal360_Calendar::days_in_month(2026, m), 30);
    }
}

TEST(Cal360EdgeCases, UniformMonthsAcrossYears) {
    for (int y = 1900; y <= 2400; y += 100) {
        for (int m = 1; m <= 12; ++m) {
            EXPECT_EQ(tick::Cal360_Calendar::days_in_month(y, m), 30);
        }
    }
}

TEST(Cal360EdgeCases, Year360Days) {
    EXPECT_EQ(tick::Cal360_Calendar::days_in_year(2026), 360);
    EXPECT_EQ(tick::Cal360_Calendar::days_in_year(2000), 360);
    EXPECT_EQ(tick::Cal360_Calendar::days_in_year(1900), 360);
}

// ─── Cal360 Edge Cases: Invalid Components Throw ─────────────────────────────

TEST(Cal360EdgeCases, InvalidDayOver30) {
    tick::Date_Time dt{2026, 1, 31, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(Cal360EdgeCases, InvalidDayZero) {
    tick::Date_Time dt{2026, 1, 0, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(Cal360EdgeCases, InvalidMonth) {
    tick::Date_Time dt{2026, 13, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(Cal360EdgeCases, InvalidMonthZero) {
    tick::Date_Time dt{2026, 0, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(Cal360EdgeCases, InvalidSecond) {
    tick::Date_Time dt{2026, 1, 1, 0, 0, 60, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

TEST(Cal360EdgeCases, InvalidHour) {
    tick::Date_Time dt{2026, 6, 15, 24, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(dt), std::invalid_argument);
}

// ============================================================
// Example-based unit tests — Task 4.8 (additional)
// ============================================================

// Unit tests
TEST(Cal360Unit, RejectsDay31) {
    tick::Date_Time bad{2026, 1, 31, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(bad), std::invalid_argument);
}

TEST(Cal360Unit, UniformMonths) {
    for (int m = 1; m <= 12; ++m) {
        EXPECT_EQ(tick::Cal360_Calendar::days_in_month(2026, m), 30);
    }
}

TEST(Cal360Unit, InvalidMonth) {
    tick::Date_Time bad{2026, 13, 1, 0, 0, 0, 0};
    EXPECT_THROW(tick::Cal360_Calendar::to_time_point(bad), std::invalid_argument);
}
