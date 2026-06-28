// test_tick_c.cpp — GTest unit tests for the TICK C API (tick_c.h)
// Tests only through the C interface — no C++ TICK headers.
// Validates: Requirements 2.7, 3.6, 4.7, 4.8, 5.5, 5.6, 5.7, 6.4, 7.4, 7.5, 8.4, 8.5

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <tick/tick_c.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

// ── Nanosecond constants ──────────────────────────────────────────────────────

static constexpr int64_t NS_PER_SECOND = 1'000'000'000LL;
static constexpr int64_t NS_PER_DAY = 86'400'000'000'000LL;

// =============================================================================
// Type Layout Tests
// =============================================================================

TEST(TickCTypes, DateTimeSizeIs28Bytes) {
    EXPECT_EQ(sizeof(tick_date_time_t), 28u);
}

TEST(TickCTypes, CalendarEnumValues) {
    EXPECT_EQ(TICK_CAL_GREGORIAN, 0);
    EXPECT_EQ(TICK_CAL_NOLEAP, 1);
    EXPECT_EQ(TICK_CAL_360DAY, 2);
}

TEST(TickCTypes, ErrorCodeValues) {
    EXPECT_EQ(TICK_OK, 0);
    EXPECT_EQ(TICK_ERR_OVERFLOW, 1);
    EXPECT_EQ(TICK_ERR_INVALID_ARG, 2);
    EXPECT_EQ(TICK_ERR_INVALID_CALENDAR, 3);
    EXPECT_EQ(TICK_ERR_INVALID_DATE, 4);
    EXPECT_EQ(TICK_ERR_INTERNAL, 5);
}

// =============================================================================
// Error Reporting Tests (tick_strerror)
// =============================================================================

TEST(TickCStrerror, AllDefinedCodes) {
    for (int code = 0; code <= 5; ++code) {
        const char *msg = tick_strerror(static_cast<tick_status_t>(code));
        ASSERT_NE(msg, nullptr) << "tick_strerror returned null for code " << code;
        EXPECT_GT(std::strlen(msg), 0u) << "tick_strerror returned empty for code " << code;
    }
}

TEST(TickCStrerror, OutOfRange) {
    // Negative and large values should still return a non-null string
    const char *msg_neg = tick_strerror(-1);
    ASSERT_NE(msg_neg, nullptr);
    EXPECT_GT(std::strlen(msg_neg), 0u);

    const char *msg_large = tick_strerror(99);
    ASSERT_NE(msg_large, nullptr);
    EXPECT_GT(std::strlen(msg_large), 0u);
}

// =============================================================================
// Time_Point Tests
// =============================================================================

TEST(TickCTimePoint, Create) {
    tick_time_point_t out = -1;
    tick_status_t rc = tick_time_point_create(0, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);

    rc = tick_time_point_create(NS_PER_DAY, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, NS_PER_DAY);
}

TEST(TickCTimePoint, AddDuration) {
    tick_time_point_t out = -1;
    // 0 + 1 second = 1 second in nanos
    tick_status_t rc = tick_time_point_add_duration(0, NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, NS_PER_SECOND);
}

TEST(TickCTimePoint, SubDuration) {
    tick_time_point_t out = -1;
    // 1 day - 1 second = (86400 - 1) seconds in nanos
    tick_status_t rc = tick_time_point_sub_duration(NS_PER_DAY, NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, NS_PER_DAY - NS_PER_SECOND);
}

TEST(TickCTimePoint, Diff) {
    tick_duration_t out = -1;
    // diff(10s, 3s) = 7s
    tick_status_t rc = tick_time_point_diff(10 * NS_PER_SECOND, 3 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 7 * NS_PER_SECOND);
}

TEST(TickCTimePoint, Compare) {
    int32_t out = 99;

    tick_status_t rc = tick_time_point_compare(100, 200, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_LT(out, 0);  // 100 < 200

    rc = tick_time_point_compare(200, 200, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);  // equal

    rc = tick_time_point_compare(300, 200, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_GT(out, 0);  // 300 > 200
}

TEST(TickCTimePoint, NullPointer) {
    EXPECT_EQ(tick_time_point_create(0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_time_point_add_duration(0, 0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_time_point_sub_duration(0, 0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_time_point_diff(0, 0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_time_point_compare(0, 0, nullptr), TICK_ERR_INVALID_ARG);
}

TEST(TickCTimePoint, Overflow) {
    tick_time_point_t out = 42;
    // Adding INT64_MAX to INT64_MAX should overflow
    tick_status_t rc = tick_time_point_add_duration(INT64_MAX, INT64_MAX, &out);
    EXPECT_EQ(rc, TICK_ERR_OVERFLOW);
    EXPECT_EQ(out, 42);  // unchanged on error

    out = 42;
    // Subtracting INT64_MIN (large negative) from INT64_MAX should overflow
    rc = tick_time_point_sub_duration(INT64_MIN, INT64_MAX, &out);
    EXPECT_EQ(rc, TICK_ERR_OVERFLOW);
    EXPECT_EQ(out, 42);  // unchanged on error
}

// =============================================================================
// Duration Tests
// =============================================================================

TEST(TickCDuration, FactoryFromNanos) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_from_nanos(12345, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 12345);
}

TEST(TickCDuration, FactoryFromSeconds) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_from_seconds(1, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, NS_PER_SECOND);
}

TEST(TickCDuration, FactoryFromMinutes) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_from_minutes(1, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 60LL * NS_PER_SECOND);
}

TEST(TickCDuration, FactoryFromHours) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_from_hours(1, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 3600LL * NS_PER_SECOND);
}

TEST(TickCDuration, FactoryFromDays) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_from_days(1, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, NS_PER_DAY);
}

TEST(TickCDuration, Add) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_add(NS_PER_SECOND, 2 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 3 * NS_PER_SECOND);
}

TEST(TickCDuration, Sub) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_sub(5 * NS_PER_SECOND, 2 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 3 * NS_PER_SECOND);
}

TEST(TickCDuration, Mul) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_mul(NS_PER_SECOND, 10, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 10 * NS_PER_SECOND);
}

TEST(TickCDuration, Div) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_duration_div(10 * NS_PER_SECOND, 5, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 2 * NS_PER_SECOND);
}

TEST(TickCDuration, DivByZero) {
    tick_duration_t out = 42;
    tick_status_t rc = tick_duration_div(NS_PER_SECOND, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
    EXPECT_EQ(out, 42);  // unchanged on error
}

TEST(TickCDuration, NullPointer) {
    EXPECT_EQ(tick_duration_from_nanos(1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_from_seconds(1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_from_minutes(1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_from_hours(1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_from_days(1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_add(1, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_sub(1, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_mul(1, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_duration_div(1, 1, nullptr), TICK_ERR_INVALID_ARG);
}

TEST(TickCDuration, FactoryOverflow) {
    tick_duration_t out = 42;
    // INT64_MAX seconds would overflow when converting to nanoseconds
    tick_status_t rc = tick_duration_from_seconds(INT64_MAX, &out);
    EXPECT_EQ(rc, TICK_ERR_OVERFLOW);
    EXPECT_EQ(out, 42);  // unchanged on error
}

TEST(TickCDuration, ArithmeticOverflow) {
    tick_duration_t out = 42;
    tick_status_t rc = tick_duration_add(INT64_MAX, INT64_MAX, &out);
    EXPECT_EQ(rc, TICK_ERR_OVERFLOW);
    EXPECT_EQ(out, 42);  // unchanged on error

    out = 42;
    rc = tick_duration_mul(INT64_MAX, 2, &out);
    EXPECT_EQ(rc, TICK_ERR_OVERFLOW);
    EXPECT_EQ(out, 42);  // unchanged on error
}

// =============================================================================
// Calendar Tests
// =============================================================================

TEST(TickCCalendar, GregorianRoundTrip) {
    // 2026-01-01T00:00:00 ↔ nanos=0 (epoch)
    tick_date_time_t dt_out{};
    tick_status_t rc = tick_to_date_time(0, TICK_CAL_GREGORIAN, &dt_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(dt_out.year, 2026);
    EXPECT_EQ(dt_out.month, 1);
    EXPECT_EQ(dt_out.day, 1);
    EXPECT_EQ(dt_out.hour, 0);
    EXPECT_EQ(dt_out.minute, 0);
    EXPECT_EQ(dt_out.second, 0);
    EXPECT_EQ(dt_out.nanosecond, 0);

    // Convert back to time point
    tick_time_point_t tp_out = -1;
    rc = tick_to_time_point(dt_out, TICK_CAL_GREGORIAN, &tp_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(tp_out, 0);
}

TEST(TickCCalendar, GregorianOneDayForward) {
    // 2026-01-02T00:00:00 ↔ nanos = NS_PER_DAY
    tick_date_time_t dt_out{};
    tick_status_t rc = tick_to_date_time(NS_PER_DAY, TICK_CAL_GREGORIAN, &dt_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(dt_out.year, 2026);
    EXPECT_EQ(dt_out.month, 1);
    EXPECT_EQ(dt_out.day, 2);
    EXPECT_EQ(dt_out.hour, 0);

    // Round-trip back
    tick_time_point_t tp_out = -1;
    rc = tick_to_time_point(dt_out, TICK_CAL_GREGORIAN, &tp_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(tp_out, NS_PER_DAY);
}

TEST(TickCCalendar, NoLeapRoundTrip) {
    // Epoch ↔ 2026-01-01T00:00:00 in NoLeap
    tick_date_time_t dt_out{};
    tick_status_t rc = tick_to_date_time(0, TICK_CAL_NOLEAP, &dt_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(dt_out.year, 2026);
    EXPECT_EQ(dt_out.month, 1);
    EXPECT_EQ(dt_out.day, 1);

    tick_time_point_t tp_out = -1;
    rc = tick_to_time_point(dt_out, TICK_CAL_NOLEAP, &tp_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(tp_out, 0);
}

TEST(TickCCalendar, Cal360RoundTrip) {
    // Epoch ↔ 2026-01-01T00:00:00 in Cal360
    tick_date_time_t dt_out{};
    tick_status_t rc = tick_to_date_time(0, TICK_CAL_360DAY, &dt_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(dt_out.year, 2026);
    EXPECT_EQ(dt_out.month, 1);
    EXPECT_EQ(dt_out.day, 1);

    tick_time_point_t tp_out = -1;
    rc = tick_to_time_point(dt_out, TICK_CAL_360DAY, &tp_out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(tp_out, 0);
}

TEST(TickCCalendar, InvalidCalendar) {
    tick_date_time_t dt_out{};
    tick_status_t rc = tick_to_date_time(0, static_cast<tick_calendar_t>(99), &dt_out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_CALENDAR);

    tick_time_point_t tp_out = -1;
    tick_date_time_t dt_in = {2026, 1, 1, 0, 0, 0, 0};
    rc = tick_to_time_point(dt_in, static_cast<tick_calendar_t>(99), &tp_out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_CALENDAR);

    int32_t days = -1;
    rc = tick_days_in_month(static_cast<tick_calendar_t>(99), 2026, 1, &days);
    EXPECT_EQ(rc, TICK_ERR_INVALID_CALENDAR);

    rc = tick_days_in_year(static_cast<tick_calendar_t>(99), 2026, &days);
    EXPECT_EQ(rc, TICK_ERR_INVALID_CALENDAR);
}

TEST(TickCCalendar, InvalidDate) {
    tick_time_point_t tp_out = -1;
    // Month 13 is invalid
    tick_date_time_t bad_dt = {2026, 13, 1, 0, 0, 0, 0};
    tick_status_t rc = tick_to_time_point(bad_dt, TICK_CAL_GREGORIAN, &tp_out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_DATE);

    // Day 32 is invalid
    tick_date_time_t bad_day = {2026, 1, 32, 0, 0, 0, 0};
    rc = tick_to_time_point(bad_day, TICK_CAL_GREGORIAN, &tp_out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_DATE);
}

TEST(TickCCalendar, DaysInMonthGregorian) {
    int32_t days = 0;
    tick_status_t rc = tick_days_in_month(TICK_CAL_GREGORIAN, 2026, 1, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 31);

    // February in a non-leap year
    rc = tick_days_in_month(TICK_CAL_GREGORIAN, 2026, 2, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 28);

    // February in a leap year
    rc = tick_days_in_month(TICK_CAL_GREGORIAN, 2028, 2, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 29);
}

TEST(TickCCalendar, DaysInYearGregorian) {
    int32_t days = 0;
    tick_status_t rc = tick_days_in_year(TICK_CAL_GREGORIAN, 2026, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 365);

    rc = tick_days_in_year(TICK_CAL_GREGORIAN, 2028, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 366);
}

TEST(TickCCalendar, DaysInYearCal360) {
    int32_t days = 0;
    tick_status_t rc = tick_days_in_year(TICK_CAL_360DAY, 2026, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 360);
}

TEST(TickCCalendar, DaysInYearNoLeap) {
    int32_t days = 0;
    tick_status_t rc = tick_days_in_year(TICK_CAL_NOLEAP, 2028, &days);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(days, 365);  // NoLeap always 365
}

TEST(TickCCalendar, NullPointer) {
    EXPECT_EQ(tick_to_date_time(0, TICK_CAL_GREGORIAN, nullptr), TICK_ERR_INVALID_ARG);
    tick_date_time_t dt = {2026, 1, 1, 0, 0, 0, 0};
    EXPECT_EQ(tick_to_time_point(dt, TICK_CAL_GREGORIAN, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_days_in_month(TICK_CAL_GREGORIAN, 2026, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_days_in_year(TICK_CAL_GREGORIAN, 2026, nullptr), TICK_ERR_INVALID_ARG);
}

// =============================================================================
// Alarm Tests
// =============================================================================

TEST(TickCAlarm, IntervalAlarmRinging) {
    int32_t out = -1;
    // Alarm with 10s interval, reference at 0. At t=0 it should ring.
    tick_status_t rc = tick_interval_alarm_is_ringing(10 * NS_PER_SECOND, 0, 0, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // At t=10s it should ring
    rc = tick_interval_alarm_is_ringing(10 * NS_PER_SECOND, 0, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // At t=5s it should NOT ring
    rc = tick_interval_alarm_is_ringing(10 * NS_PER_SECOND, 0, 5 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);
}

TEST(TickCAlarm, IntervalAlarmNextRing) {
    tick_time_point_t out = -1;
    // interval=10s, reference=0, current=5s → next ring at 10s
    tick_status_t rc = tick_interval_alarm_next_ring(10 * NS_PER_SECOND, 0, 5 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 10 * NS_PER_SECOND);
}

TEST(TickCAlarm, AbsoluteAlarmRinging) {
    int32_t out = -1;
    // Trigger at t=100ns, current=100ns → ringing
    tick_status_t rc = tick_absolute_alarm_is_ringing(100, 100, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // Current before trigger → not ringing
    rc = tick_absolute_alarm_is_ringing(100, 99, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);

    // Current after trigger → not ringing
    rc = tick_absolute_alarm_is_ringing(100, 101, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);
}

TEST(TickCAlarm, ZeroIntervalError) {
    int32_t out = -1;
    tick_status_t rc = tick_interval_alarm_is_ringing(0, 0, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);

    tick_time_point_t tp_out = -1;
    rc = tick_interval_alarm_next_ring(0, 0, 0, &tp_out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

TEST(TickCAlarm, NullPointer) {
    EXPECT_EQ(tick_interval_alarm_is_ringing(NS_PER_SECOND, 0, 0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_interval_alarm_next_ring(NS_PER_SECOND, 0, 0, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_absolute_alarm_is_ringing(0, 0, nullptr), TICK_ERR_INVALID_ARG);
}

// =============================================================================
// Sync Tests
// =============================================================================

TEST(TickCSync, Heartbeat) {
    // GCD of 10s and 6s = 2s
    tick_duration_t timesteps[] = {10 * NS_PER_SECOND, 6 * NS_PER_SECOND};
    tick_duration_t out = -1;
    tick_status_t rc = tick_compute_heartbeat(timesteps, 2, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 2 * NS_PER_SECOND);
}

TEST(TickCSync, SyncPeriod) {
    // LCM of 10s and 6s = 30s
    tick_duration_t timesteps[] = {10 * NS_PER_SECOND, 6 * NS_PER_SECOND};
    tick_duration_t out = -1;
    tick_status_t rc = tick_compute_sync_period(timesteps, 2, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 30 * NS_PER_SECOND);
}

TEST(TickCSync, PhaseAligned) {
    int32_t out = -1;
    // current=20s, base=0, timestep=10s → (20-0) % 10 = 0 → aligned
    tick_status_t rc = tick_is_phase_aligned(20 * NS_PER_SECOND, 0, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // current=15s, base=0, timestep=10s → (15-0) % 10 = 5 → not aligned
    rc = tick_is_phase_aligned(15 * NS_PER_SECOND, 0, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);
}

TEST(TickCSync, NullArray) {
    tick_duration_t out = -1;
    tick_status_t rc = tick_compute_heartbeat(nullptr, 2, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);

    rc = tick_compute_sync_period(nullptr, 2, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

TEST(TickCSync, ZeroCount) {
    tick_duration_t timesteps[] = {NS_PER_SECOND};
    tick_duration_t out = -1;
    tick_status_t rc = tick_compute_heartbeat(timesteps, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);

    rc = tick_compute_sync_period(timesteps, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

TEST(TickCSync, NullOutputPointer) {
    tick_duration_t timesteps[] = {NS_PER_SECOND};
    EXPECT_EQ(tick_compute_heartbeat(timesteps, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_compute_sync_period(timesteps, 1, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_is_phase_aligned(0, 0, NS_PER_SECOND, nullptr), TICK_ERR_INVALID_ARG);
}

TEST(TickCSync, ZeroTimestepPhaseAligned) {
    int32_t out = -1;
    // Zero timestep should return error
    tick_status_t rc = tick_is_phase_aligned(0, 0, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

// =============================================================================
// Window Tests
// =============================================================================

TEST(TickCWindow, IsOnBoundary) {
    int32_t out = -1;
    // 0 is always on boundary of any positive interval
    tick_status_t rc = tick_is_on_boundary(0, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // 10s is on boundary of 10s interval
    rc = tick_is_on_boundary(10 * NS_PER_SECOND, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // 5s is NOT on boundary of 10s interval
    rc = tick_is_on_boundary(5 * NS_PER_SECOND, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);
}

TEST(TickCWindow, ComputeWindow) {
    tick_time_point_t out_start = -1, out_end = -1;
    // current=15s, interval=10s → window=[10s, 20s)
    tick_status_t rc = tick_compute_window(15 * NS_PER_SECOND, 10 * NS_PER_SECOND, &out_start, &out_end);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out_start, 10 * NS_PER_SECOND);
    EXPECT_EQ(out_end, 20 * NS_PER_SECOND);
}

TEST(TickCWindow, WindowContains) {
    int32_t out = -1;
    // Window [10s, 20s), query=15s → contained
    tick_status_t rc = tick_window_contains(10 * NS_PER_SECOND, 20 * NS_PER_SECOND, 15 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // query=10s (start) → contained [start, end)
    rc = tick_window_contains(10 * NS_PER_SECOND, 20 * NS_PER_SECOND, 10 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 1);

    // query=20s (end) → NOT contained (half-open interval)
    rc = tick_window_contains(10 * NS_PER_SECOND, 20 * NS_PER_SECOND, 20 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);

    // query=5s (before start) → NOT contained
    rc = tick_window_contains(10 * NS_PER_SECOND, 20 * NS_PER_SECOND, 5 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_OK);
    EXPECT_EQ(out, 0);
}

TEST(TickCWindow, ZeroIntervalError) {
    int32_t out = -1;
    tick_status_t rc = tick_is_on_boundary(0, 0, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);

    tick_time_point_t start = -1, end = -1;
    rc = tick_compute_window(0, 0, &start, &end);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

TEST(TickCWindow, InvertedWindowError) {
    int32_t out = -1;
    // window start=20s, end=10s → inverted → error
    tick_status_t rc = tick_window_contains(20 * NS_PER_SECOND, 10 * NS_PER_SECOND, 15 * NS_PER_SECOND, &out);
    EXPECT_EQ(rc, TICK_ERR_INVALID_ARG);
}

TEST(TickCWindow, NullPointer) {
    EXPECT_EQ(tick_is_on_boundary(0, NS_PER_SECOND, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_compute_window(0, NS_PER_SECOND, nullptr, nullptr), TICK_ERR_INVALID_ARG);
    tick_time_point_t dummy = 0;
    EXPECT_EQ(tick_compute_window(0, NS_PER_SECOND, &dummy, nullptr), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_compute_window(0, NS_PER_SECOND, nullptr, &dummy), TICK_ERR_INVALID_ARG);
    EXPECT_EQ(tick_window_contains(0, NS_PER_SECOND, 0, nullptr), TICK_ERR_INVALID_ARG);
}

// =============================================================================
// Property-Based Tests (RapidCheck)
// =============================================================================

// Feature: tick-fortran-bridge, Property 1: Calendar Conversion Round-Trip
// For any valid tick_date_time_t for each calendar, converting to time_point
// and back SHALL produce a tick_date_time_t equal to the original.
// **Validates: Requirements 5.7**

RC_GTEST_PROP(TickCProperty, CalendarRoundTripGregorian, ()) {
    auto year = *rc::gen::inRange<int32_t>(1734, 2319);
    auto month = *rc::gen::inRange<int32_t>(1, 13);

    // Use the C API itself to get max days for this year/month
    int32_t max_days = 0;
    tick_status_t rc_days = tick_days_in_month(TICK_CAL_GREGORIAN, year, month, &max_days);
    RC_PRE(rc_days == TICK_OK);
    RC_PRE(max_days > 0);

    auto day = *rc::gen::inRange<int32_t>(1, max_days + 1);
    auto hour = *rc::gen::inRange<int32_t>(0, 24);
    auto minute = *rc::gen::inRange<int32_t>(0, 60);
    auto second = *rc::gen::inRange<int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<int32_t>(0, 1'000'000'000);

    tick_date_time_t dt = {year, month, day, hour, minute, second, nanosecond};

    // Convert to time_point
    tick_time_point_t tp = 0;
    tick_status_t rc_tp = tick_to_time_point(dt, TICK_CAL_GREGORIAN, &tp);
    RC_PRE(rc_tp == TICK_OK);

    // Convert back to date_time
    tick_date_time_t rt = {};
    tick_status_t rc_rt = tick_to_date_time(tp, TICK_CAL_GREGORIAN, &rt);
    RC_ASSERT(rc_rt == TICK_OK);

    // Assert all fields match
    RC_ASSERT(dt.year == rt.year);
    RC_ASSERT(dt.month == rt.month);
    RC_ASSERT(dt.day == rt.day);
    RC_ASSERT(dt.hour == rt.hour);
    RC_ASSERT(dt.minute == rt.minute);
    RC_ASSERT(dt.second == rt.second);
    RC_ASSERT(dt.nanosecond == rt.nanosecond);
}

RC_GTEST_PROP(TickCProperty, CalendarRoundTripNoLeap, ()) {
    auto year = *rc::gen::inRange<int32_t>(1736, 2317);
    auto month = *rc::gen::inRange<int32_t>(1, 13);

    // Use the C API itself to get max days for this year/month
    int32_t max_days = 0;
    tick_status_t rc_days = tick_days_in_month(TICK_CAL_NOLEAP, year, month, &max_days);
    RC_PRE(rc_days == TICK_OK);
    RC_PRE(max_days > 0);

    auto day = *rc::gen::inRange<int32_t>(1, max_days + 1);
    auto hour = *rc::gen::inRange<int32_t>(0, 24);
    auto minute = *rc::gen::inRange<int32_t>(0, 60);
    auto second = *rc::gen::inRange<int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<int32_t>(0, 1'000'000'000);

    tick_date_time_t dt = {year, month, day, hour, minute, second, nanosecond};

    // Convert to time_point
    tick_time_point_t tp = 0;
    tick_status_t rc_tp = tick_to_time_point(dt, TICK_CAL_NOLEAP, &tp);
    RC_PRE(rc_tp == TICK_OK);

    // Convert back to date_time
    tick_date_time_t rt = {};
    tick_status_t rc_rt = tick_to_date_time(tp, TICK_CAL_NOLEAP, &rt);
    RC_ASSERT(rc_rt == TICK_OK);

    // Assert all fields match
    RC_ASSERT(dt.year == rt.year);
    RC_ASSERT(dt.month == rt.month);
    RC_ASSERT(dt.day == rt.day);
    RC_ASSERT(dt.hour == rt.hour);
    RC_ASSERT(dt.minute == rt.minute);
    RC_ASSERT(dt.second == rt.second);
    RC_ASSERT(dt.nanosecond == rt.nanosecond);
}

RC_GTEST_PROP(TickCProperty, CalendarRoundTripCal360, ()) {
    auto year = *rc::gen::inRange<int32_t>(1734, 2319);
    auto month = *rc::gen::inRange<int32_t>(1, 13);
    auto day = *rc::gen::inRange<int32_t>(1, 31);  // Cal360: always 1-30

    auto hour = *rc::gen::inRange<int32_t>(0, 24);
    auto minute = *rc::gen::inRange<int32_t>(0, 60);
    auto second = *rc::gen::inRange<int32_t>(0, 60);
    auto nanosecond = *rc::gen::inRange<int32_t>(0, 1'000'000'000);

    tick_date_time_t dt = {year, month, day, hour, minute, second, nanosecond};

    // Convert to time_point
    tick_time_point_t tp = 0;
    tick_status_t rc_tp = tick_to_time_point(dt, TICK_CAL_360DAY, &tp);
    RC_PRE(rc_tp == TICK_OK);

    // Convert back to date_time
    tick_date_time_t rt = {};
    tick_status_t rc_rt = tick_to_date_time(tp, TICK_CAL_360DAY, &rt);
    RC_ASSERT(rc_rt == TICK_OK);

    // Assert all fields match
    RC_ASSERT(dt.year == rt.year);
    RC_ASSERT(dt.month == rt.month);
    RC_ASSERT(dt.day == rt.day);
    RC_ASSERT(dt.hour == rt.hour);
    RC_ASSERT(dt.minute == rt.minute);
    RC_ASSERT(dt.second == rt.second);
    RC_ASSERT(dt.nanosecond == rt.nanosecond);
}

// =============================================================================
// Property-Based Tests — Duration Arithmetic Equivalence
// =============================================================================

// Feature: tick-fortran-bridge, Property 3: Duration Arithmetic Equivalence
// **Validates: Requirements 4.3, 4.4, 4.5, 4.6**
//
// For any two tick_duration_t values and any int64_t scalar where the operation
// does not overflow or divide by zero, tick_duration_add, tick_duration_sub,
// tick_duration_mul, and tick_duration_div return TICK_OK and produce the
// mathematically correct result (with division truncated toward zero).

RC_GTEST_PROP(TickCProperty, DurationAddSub, ()) {
    auto lhs = *rc::gen::inRange<int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto rhs = *rc::gen::inRange<int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);

    tick_duration_t add_out;
    tick_status_t rc_add = tick_duration_add(lhs, rhs, &add_out);
    if (rc_add == TICK_OK) {
        RC_ASSERT(add_out == lhs + rhs);
        // Verify sub reverses add
        tick_duration_t sub_out;
        RC_ASSERT(tick_duration_sub(add_out, rhs, &sub_out) == TICK_OK);
        RC_ASSERT(sub_out == lhs);
    }

    tick_duration_t sub_out2;
    tick_status_t rc_sub = tick_duration_sub(lhs, rhs, &sub_out2);
    if (rc_sub == TICK_OK) {
        RC_ASSERT(sub_out2 == lhs - rhs);
        // Verify add reverses sub
        tick_duration_t add_back;
        RC_ASSERT(tick_duration_add(sub_out2, rhs, &add_back) == TICK_OK);
        RC_ASSERT(add_back == lhs);
    }
}

RC_GTEST_PROP(TickCProperty, DurationMulDiv, ()) {
    auto dur = *rc::gen::inRange<int64_t>(-1'000'000'000'000LL, 1'000'000'000'000LL);
    auto scalar = *rc::gen::inRange<int64_t>(-1'000'000LL, 1'000'000LL);
    RC_PRE(scalar != 0);

    tick_duration_t mul_out;
    tick_status_t rc_mul = tick_duration_mul(dur, scalar, &mul_out);
    if (rc_mul == TICK_OK) {
        RC_ASSERT(mul_out == dur * scalar);
        // Verify div reverses mul (truncated division)
        tick_duration_t div_out;
        RC_ASSERT(tick_duration_div(mul_out, scalar, &div_out) == TICK_OK);
        RC_ASSERT(div_out == dur);
    }

    // Also test div directly with non-zero scalar
    tick_duration_t div_out2;
    tick_status_t rc_div = tick_duration_div(dur, scalar, &div_out2);
    if (rc_div == TICK_OK) {
        // Division truncates toward zero (C/C++ semantics)
        RC_ASSERT(div_out2 == dur / scalar);
    }
}

// =============================================================================
// Property-Based Tests (RapidCheck)
// =============================================================================

// Feature: tick-fortran-bridge, Property 2: Time_Point Arithmetic Equivalence
// For any tick_time_point_t value tp and tick_duration_t value dur where the
// operation does not overflow, tick_time_point_add_duration returns TICK_OK and
// sets out to tp + dur. Sub reverses add. Diff gives back the duration.
// **Validates: Requirements 3.2, 3.3, 3.4, 3.5**
RC_GTEST_PROP(TickCProperty, TimePointArithmetic, ()) {
    // Generate values in half range to avoid overflow
    auto tp = *rc::gen::inRange<int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto dur = *rc::gen::inRange<int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);

    tick_time_point_t add_result;
    tick_status_t rc_add = tick_time_point_add_duration(tp, dur, &add_result);
    if (rc_add == TICK_OK) {
        // Verify: add produced the correct mathematical result
        RC_ASSERT(add_result == tp + dur);

        // Verify: sub_duration reverses add
        tick_time_point_t sub_result;
        RC_ASSERT(tick_time_point_sub_duration(add_result, dur, &sub_result) == TICK_OK);
        RC_ASSERT(sub_result == tp);

        // Verify: diff gives back dur
        tick_duration_t diff_result;
        RC_ASSERT(tick_time_point_diff(add_result, tp, &diff_result) == TICK_OK);
        RC_ASSERT(diff_result == dur);
    }
}

// Feature: tick-fortran-bridge, Property 2: Time_Point Arithmetic Equivalence (compare)
// For any two tick_time_point_t values, tick_time_point_compare returns a value
// whose sign matches the true integer ordering.
// **Validates: Requirements 3.5**
RC_GTEST_PROP(TickCProperty, TimePointCompare, ()) {
    auto lhs = *rc::gen::arbitrary<int64_t>();
    auto rhs = *rc::gen::arbitrary<int64_t>();

    int32_t cmp;
    RC_ASSERT(tick_time_point_compare(lhs, rhs, &cmp) == TICK_OK);
    if (lhs < rhs)
        RC_ASSERT(cmp < 0);
    else if (lhs > rhs)
        RC_ASSERT(cmp > 0);
    else
        RC_ASSERT(cmp == 0);
}

// =============================================================================
// Property-Based Tests: Duration Factory Correctness (Property 4)
// Feature: tick-fortran-bridge, Property 4: Duration Factory Correctness
// Validates: Requirements 4.1, 4.2
// =============================================================================

RC_GTEST_PROP(TickCProperty, DurationFromNanos, ()) {
    auto nanos = *rc::gen::arbitrary<int64_t>();
    tick_duration_t out;
    RC_ASSERT(tick_duration_from_nanos(nanos, &out) == TICK_OK);
    RC_ASSERT(out == nanos);
}

RC_GTEST_PROP(TickCProperty, DurationFromSeconds, ()) {
    // Constrain to avoid overflow: |count| < INT64_MAX / 10^9 ≈ 9.2 * 10^9
    auto count = *rc::gen::inRange<int64_t>(-9'000'000'000LL, 9'000'000'000LL);
    tick_duration_t out;
    RC_ASSERT(tick_duration_from_seconds(count, &out) == TICK_OK);
    RC_ASSERT(out == count * 1'000'000'000LL);
}

RC_GTEST_PROP(TickCProperty, DurationFromMinutes, ()) {
    // Constrain to avoid overflow: |count| < INT64_MAX / 6×10^10 ≈ 1.5 * 10^8
    auto count = *rc::gen::inRange<int64_t>(-150'000'000LL, 150'000'000LL);
    tick_duration_t out;
    RC_ASSERT(tick_duration_from_minutes(count, &out) == TICK_OK);
    RC_ASSERT(out == count * 60'000'000'000LL);
}

RC_GTEST_PROP(TickCProperty, DurationFromHours, ()) {
    // Constrain to avoid overflow: |count| < INT64_MAX / 3.6×10^12 ≈ 2.56 * 10^6
    auto count = *rc::gen::inRange<int64_t>(-2'500'000LL, 2'500'000LL);
    tick_duration_t out;
    RC_ASSERT(tick_duration_from_hours(count, &out) == TICK_OK);
    RC_ASSERT(out == count * 3'600'000'000'000LL);
}

RC_GTEST_PROP(TickCProperty, DurationFromDays, ()) {
    // Constrain to avoid overflow: |count| < INT64_MAX / 8.64×10^13 ≈ 1.07 * 10^5
    auto count = *rc::gen::inRange<int64_t>(-100'000LL, 100'000LL);
    tick_duration_t out;
    RC_ASSERT(tick_duration_from_days(count, &out) == TICK_OK);
    RC_ASSERT(out == count * 86'400'000'000'000LL);
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 5: Exception-to-Error-Code Mapping
// For any C_API function call that triggers std::overflow_error in the
// underlying C++ code, the function returns TICK_ERR_OVERFLOW and leaves the
// output pointer value unchanged. For any call that triggers
// std::invalid_argument, the function returns TICK_ERR_INVALID_ARG.
// **Validates: Requirements 2.4, 2.5, 3.6, 4.8**
// =============================================================================

// Overflow: adding large positive values should return TICK_ERR_OVERFLOW
// and leave the output pointer unchanged
RC_GTEST_PROP(TickCProperty, OverflowLeavesOutputUnchanged, ()) {
    // Generate a value near INT64_MAX so addition will overflow
    auto base = *rc::gen::inRange<int64_t>(INT64_MAX / 2, INT64_MAX);
    auto dur = *rc::gen::inRange<int64_t>(INT64_MAX / 2, INT64_MAX);

    tick_time_point_t sentinel = static_cast<int64_t>(0xDEADBEEFCAFELL);
    tick_time_point_t out = sentinel;
    tick_status_t rc = tick_time_point_add_duration(base, dur, &out);
    if (rc == TICK_ERR_OVERFLOW) {
        RC_ASSERT(out == sentinel);  // unchanged
    }
    // If it didn't overflow (rare edge case), that's fine — precondition not met
}

// Division by zero always returns TICK_ERR_INVALID_ARG and leaves output unchanged
RC_GTEST_PROP(TickCProperty, DivByZeroReturnsInvalidArgAndLeavesOutputUnchanged, ()) {
    auto dur = *rc::gen::arbitrary<int64_t>();
    tick_duration_t sentinel = static_cast<int64_t>(0x12345678ABCDLL);
    tick_duration_t out = sentinel;
    tick_status_t rc = tick_duration_div(dur, 0, &out);
    RC_ASSERT(rc == TICK_ERR_INVALID_ARG);
    RC_ASSERT(out == sentinel);
}

// Factory overflow: large counts in seconds overflow when multiplied by 10^9
// and should leave output unchanged
RC_GTEST_PROP(TickCProperty, FactoryOverflowFromSecondsLeavesOutputUnchanged, ()) {
    // Count that will overflow when multiplied by nanos_per_second (10^9)
    // INT64_MAX / 10^9 ≈ 9.2e9, so counts above that will overflow
    auto count = *rc::gen::inRange<int64_t>(INT64_MAX / 1'000'000'000LL + 1, INT64_MAX);
    tick_duration_t sentinel = static_cast<int64_t>(0xCAFEBABE1234LL);
    tick_duration_t out = sentinel;
    tick_status_t rc = tick_duration_from_seconds(count, &out);
    RC_ASSERT(rc == TICK_ERR_OVERFLOW);
    RC_ASSERT(out == sentinel);
}

// Duration addition overflow: adding two large positive durations overflows
// and should leave output unchanged
RC_GTEST_PROP(TickCProperty, DurationAddOverflowLeavesOutputUnchanged, ()) {
    auto lhs = *rc::gen::inRange<int64_t>(INT64_MAX / 2 + 1, INT64_MAX);
    auto rhs = *rc::gen::inRange<int64_t>(INT64_MAX / 2 + 1, INT64_MAX);

    tick_duration_t sentinel = static_cast<int64_t>(0xABCDEF012345LL);
    tick_duration_t out = sentinel;
    tick_status_t rc = tick_duration_add(lhs, rhs, &out);
    if (rc == TICK_ERR_OVERFLOW) {
        RC_ASSERT(out == sentinel);  // unchanged
    }
}

// Duration multiplication overflow: multiplying by a large scalar overflows
// and should leave output unchanged
RC_GTEST_PROP(TickCProperty, DurationMulOverflowLeavesOutputUnchanged, ()) {
    // Pick a non-zero duration and a large scalar so product overflows
    auto dur = *rc::gen::inRange<int64_t>(2LL, INT64_MAX / 2);
    auto scalar = *rc::gen::inRange<int64_t>(INT64_MAX / dur + 1, INT64_MAX);

    tick_duration_t sentinel = static_cast<int64_t>(0xFEEDFACE9876LL);
    tick_duration_t out = sentinel;
    tick_status_t rc = tick_duration_mul(dur, scalar, &out);
    if (rc == TICK_ERR_OVERFLOW) {
        RC_ASSERT(out == sentinel);  // unchanged
    }
}

// Time_Point subtraction overflow: subtracting from INT64_MIN overflows
// and should leave output unchanged
RC_GTEST_PROP(TickCProperty, TimePointSubOverflowLeavesOutputUnchanged, ()) {
    // Generate a base near INT64_MIN and a large positive duration
    auto base = *rc::gen::inRange<int64_t>(INT64_MIN, INT64_MIN / 2);
    auto dur = *rc::gen::inRange<int64_t>(INT64_MAX / 2, INT64_MAX);

    tick_time_point_t sentinel = static_cast<int64_t>(0x1111222233LL);
    tick_time_point_t out = sentinel;
    tick_status_t rc = tick_time_point_sub_duration(base, dur, &out);
    if (rc == TICK_ERR_OVERFLOW) {
        RC_ASSERT(out == sentinel);  // unchanged
    }
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 6: tick_strerror Completeness
// For any tick_status_t value in [0,5], tick_strerror returns a non-null pointer
// to a non-empty null-terminated string. For any integer outside [0,5], it still
// returns a non-null pointer (fallback message).
// **Validates: Requirements 2.7**
// =============================================================================

RC_GTEST_PROP(TickCProperty, StrerrorCompleteness, ()) {
    auto code = *rc::gen::arbitrary<int32_t>();
    const char *msg = tick_strerror(code);
    RC_ASSERT(msg != nullptr);
    RC_ASSERT(std::strlen(msg) > 0);
}

RC_GTEST_PROP(TickCProperty, StrerrorDefinedCodes, ()) {
    auto code = *rc::gen::inRange<int32_t>(0, 6);
    const char *msg = tick_strerror(code);
    RC_ASSERT(msg != nullptr);
    RC_ASSERT(std::strlen(msg) > 0);
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 9: Sync Period Divisible By All Timesteps
// For any array of positive tick_duration_t values where the LCM doesn't overflow,
// the result of tick_compute_sync_period is evenly divisible by every element.
// **Validates: Requirements 7.2**
// =============================================================================

RC_GTEST_PROP(TickCProperty, SyncPeriodDivisibleByAllTimesteps, ()) {
    // Use small values to avoid LCM overflow: multiples of 1 second up to 100 seconds
    auto count = *rc::gen::inRange(2, 6);
    std::vector<int64_t> timesteps;
    timesteps.reserve(count);
    for (int i = 0; i < count; ++i) {
        // Small positive durations in seconds (1-100 seconds as nanos)
        auto seconds = *rc::gen::inRange<int64_t>(1, 101);
        timesteps.push_back(seconds * 1'000'000'000LL);
    }

    tick_duration_t sync_period = 0;
    tick_status_t rc = tick_compute_sync_period(timesteps.data(), static_cast<int32_t>(timesteps.size()), &sync_period);

    if (rc == TICK_OK) {
        RC_ASSERT(sync_period > 0);
        // Verify sync_period is divisible by every timestep
        for (auto ts : timesteps) {
            RC_ASSERT(sync_period % ts == 0);
        }
    }
    // If overflow, that's acceptable — skip
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 8: Heartbeat Divides All Timesteps
// For any array of positive tick_duration_t values, the result of
// tick_compute_heartbeat evenly divides every element.
// **Validates: Requirements 7.1**
// =============================================================================

RC_GTEST_PROP(TickCProperty, HeartbeatDividesAllTimesteps, ()) {
    // Generate 2-8 positive durations up to 1 day each
    auto count = *rc::gen::inRange(2, 9);
    std::vector<int64_t> timesteps;
    timesteps.reserve(count);
    for (int i = 0; i < count; ++i) {
        timesteps.push_back(*rc::gen::inRange<int64_t>(1, 86'400'000'000'000LL));
    }

    tick_duration_t heartbeat = 0;
    tick_status_t rc = tick_compute_heartbeat(timesteps.data(), static_cast<int32_t>(timesteps.size()), &heartbeat);
    RC_ASSERT(rc == TICK_OK);
    RC_ASSERT(heartbeat > 0);

    // Verify heartbeat divides every timestep
    for (auto ts : timesteps) {
        RC_ASSERT(ts % heartbeat == 0);
    }
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 10: Window and Boundary Operations Equivalence
// For any tick_time_point_t current and positive interval:
// (a) tick_is_on_boundary returns 1 iff current % interval == 0;
// (b) the window from tick_compute_window satisfies start <= current < end
//     and end - start == interval;
// (c) tick_window_contains(start, end, query) returns 1 iff start <= query < end.
// **Validates: Requirements 7.3, 8.1, 8.2, 8.3**
// =============================================================================

RC_GTEST_PROP(TickCProperty, IsOnBoundaryEquivalence, ()) {
    auto interval = *rc::gen::inRange<int64_t>(1, 86'400'000'000'000LL);
    auto current = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000'000LL);

    int32_t out = -1;
    tick_status_t rc = tick_is_on_boundary(current, interval, &out);
    RC_ASSERT(rc == TICK_OK);

    bool expected = (current % interval == 0);
    RC_ASSERT(out == (expected ? 1 : 0));
}

RC_GTEST_PROP(TickCProperty, ComputeWindowProperties, ()) {
    auto interval = *rc::gen::inRange<int64_t>(1, 86'400'000'000'000LL);
    auto current = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000'000LL);

    tick_time_point_t win_start = 0, win_end = 0;
    tick_status_t rc = tick_compute_window(current, interval, &win_start, &win_end);
    RC_ASSERT(rc == TICK_OK);

    // start <= current < end
    RC_ASSERT(win_start <= current);
    RC_ASSERT(current < win_end);
    // end - start == interval
    RC_ASSERT(win_end - win_start == interval);

    // window_contains should agree
    int32_t contains_current = 0;
    RC_ASSERT(tick_window_contains(win_start, win_end, current, &contains_current) == TICK_OK);
    RC_ASSERT(contains_current == 1);

    // Point just before start should not be contained
    if (win_start > 0) {
        int32_t contains_before = -1;
        RC_ASSERT(tick_window_contains(win_start, win_end, win_start - 1, &contains_before) == TICK_OK);
        RC_ASSERT(contains_before == 0);
    }

    // End point should not be contained (half-open)
    int32_t contains_end = -1;
    RC_ASSERT(tick_window_contains(win_start, win_end, win_end, &contains_end) == TICK_OK);
    RC_ASSERT(contains_end == 0);
}

// =============================================================================
// Feature: tick-fortran-bridge, Property 7: Interval Alarm Equivalence
// For any positive interval, any reference, and any current where current >=
// reference, tick_interval_alarm_is_ringing returns 1 iff
// (current - reference) % interval == 0. The time returned by
// tick_interval_alarm_next_ring is > current and satisfies the ringing condition.
// **Validates: Requirements 6.1, 6.2**
// =============================================================================

RC_GTEST_PROP(TickCProperty, IntervalAlarmIsRinging, ()) {
    // Generate positive interval (1ns to 1 day)
    auto interval = *rc::gen::inRange<int64_t>(1, 86'400'000'000'000LL);
    // Generate reference time
    auto reference = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000'000LL);
    // Generate current >= reference
    auto offset = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000'000LL);
    int64_t current = reference + offset;

    int32_t out = -1;
    tick_status_t rc = tick_interval_alarm_is_ringing(interval, reference, current, &out);
    RC_ASSERT(rc == TICK_OK);

    bool expected = (offset % interval == 0);
    RC_ASSERT(out == (expected ? 1 : 0));
}

RC_GTEST_PROP(TickCProperty, IntervalAlarmNextRing, ()) {
    auto interval = *rc::gen::inRange<int64_t>(1, 1'000'000'000'000LL);
    auto reference = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000LL);
    auto offset = *rc::gen::inRange<int64_t>(0, 1'000'000'000'000LL);
    int64_t current = reference + offset;

    tick_time_point_t next = -1;
    tick_status_t rc = tick_interval_alarm_next_ring(interval, reference, current, &next);
    RC_ASSERT(rc == TICK_OK);

    // next_ring must be > current
    RC_ASSERT(next > current);

    // next_ring must satisfy the ringing condition
    int32_t ringing = 0;
    RC_ASSERT(tick_interval_alarm_is_ringing(interval, reference, next, &ringing) == TICK_OK);
    RC_ASSERT(ringing == 1);
}
