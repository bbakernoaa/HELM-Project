// test_alarm.cpp — Alarm system property-based tests (Task 6.3) and unit tests (Task 6.4)
// Property 9: Interval_Alarm is_ringing matches modulo condition
// Property 10: Interval_Alarm next_ring_at is the smallest future ring time
// Property 11: Absolute_Alarm is_ringing and has_passed match equality/inequality
// **Validates: Requirements 6.2, 6.6, 7.2, 7.4, 11.9**

#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <tick/absolute_alarm.hpp>
#include <tick/duration.hpp>
#include <tick/interval_alarm.hpp>
#include <tick/time_point.hpp>

// ============================================================
// Property-based tests (RapidCheck)
// ============================================================

// --------------------------------------------------------------------------
// Property 9: Interval_Alarm is_ringing matches modulo condition
//
// For any positive Duration interval, Time_Point reference, and Time_Point
// current_time: Interval_Alarm(interval, reference).is_ringing(current_time)
// returns true iff current_time >= reference AND
// (current_time.nanos() - reference.nanos()) % interval.nanos() == 0.
// **Validates: Requirements 6.2, 6.6**
// --------------------------------------------------------------------------
RC_GTEST_PROP(IntervalAlarm, IsRingingMatchesModulo, ()) {
    // Generate a positive interval (1ns to 1 day)
    auto interval_ns = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL);
    // Generate reference and current time in a manageable range
    auto ref_ns = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000LL, 1'000'000'000'000LL);
    auto current_ns = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000LL, 2'000'000'000'000LL);

    auto interval = tick::Duration{interval_ns};
    auto reference = tick::Time_Point{ref_ns};
    auto current = tick::Time_Point{current_ns};

    tick::Interval_Alarm alarm{interval, reference};

    bool expected = (current_ns >= ref_ns) && ((current_ns - ref_ns) % interval_ns == 0);
    RC_ASSERT(alarm.is_ringing(current) == expected);
}

// --------------------------------------------------------------------------
// Property 10: Interval_Alarm next_ring_at is the smallest future ring time
//
// For any valid Interval_Alarm and Time_Point current_time, let
// next = alarm.next_ring_at(current_time). Then:
//   1) next > current_time
//   2) alarm.is_ringing(next) == true
//   3) There is no Time_Point t with current_time < t < next where
//      alarm.is_ringing(t) is true (i.e., next - interval <= current_time)
// **Validates: Requirements 6.7**
// --------------------------------------------------------------------------
RC_GTEST_PROP(IntervalAlarm, NextRingAtIsSmallestFuture, ()) {
    // Generate a positive interval
    auto interval_ns = *rc::gen::inRange<std::int64_t>(1, 1'000'000'000'000LL);
    auto ref_ns = *rc::gen::inRange<std::int64_t>(0, 1'000'000'000'000LL);
    auto current_ns = *rc::gen::inRange<std::int64_t>(ref_ns, ref_ns + interval_ns * 100);

    auto interval = tick::Duration{interval_ns};
    auto reference = tick::Time_Point{ref_ns};
    auto current = tick::Time_Point{current_ns};

    tick::Interval_Alarm alarm{interval, reference};
    auto next = alarm.next_ring_at(current);

    // next must be strictly after current
    RC_ASSERT(next > current);
    // next must be a ringing point
    RC_ASSERT(alarm.is_ringing(next));
    // The previous ring time must be <= current (proving next is the smallest future ring)
    auto prev_ring = tick::Time_Point{next.nanos() - interval_ns};
    RC_ASSERT(prev_ring <= current);
}

// --------------------------------------------------------------------------
// Property 11: Absolute_Alarm is_ringing and has_passed match equality/inequality
//
// For any Time_Point trigger_time and Time_Point current_time:
//   is_ringing(current) == (current == trigger_time)
//   has_passed(current) == (current > trigger_time)
// **Validates: Requirements 7.2, 7.4**
// --------------------------------------------------------------------------
RC_GTEST_PROP(AbsoluteAlarm, IsRingingAndHasPassed, ()) {
    auto trigger_ns = *rc::gen::arbitrary<std::int64_t>();
    auto current_ns = *rc::gen::arbitrary<std::int64_t>();

    auto trigger = tick::Time_Point{trigger_ns};
    auto current = tick::Time_Point{current_ns};

    tick::Absolute_Alarm alarm{trigger};

    RC_ASSERT(alarm.is_ringing(current) == (current_ns == trigger_ns));
    RC_ASSERT(alarm.has_passed(current) == (current_ns > trigger_ns));
}

// ============================================================
// Example-based unit tests (Google Test)
// ============================================================

TEST(IntervalAlarmErrors, ZeroIntervalThrows) {
    EXPECT_THROW(tick::Interval_Alarm(tick::Duration{0}, tick::Time_Point{100}), std::invalid_argument);
}

TEST(IntervalAlarmErrors, NegativeIntervalThrows) {
    EXPECT_THROW(tick::Interval_Alarm(tick::Duration{-100}, tick::Time_Point{0}), std::invalid_argument);
}

TEST(IntervalAlarmBehavior, NotRingingBeforeBase) {
    tick::Interval_Alarm alarm{tick::seconds(10), tick::Time_Point{1000}};
    EXPECT_FALSE(alarm.is_ringing(tick::Time_Point{999}));
    EXPECT_FALSE(alarm.is_ringing(tick::Time_Point{0}));
}

TEST(AbsoluteAlarmBehavior, CopyPreservesTrigger) {
    tick::Absolute_Alarm original{tick::Time_Point{42'000'000'000LL}};
    tick::Absolute_Alarm copy = original;
    EXPECT_TRUE(copy.is_ringing(tick::Time_Point{42'000'000'000LL}));
    EXPECT_FALSE(copy.is_ringing(tick::Time_Point{0}));
    EXPECT_EQ(copy.trigger_time(), original.trigger_time());
}

TEST(AbsoluteAlarmBehavior, HasPassedAfterTrigger) {
    tick::Absolute_Alarm alarm{tick::Time_Point{100}};
    EXPECT_FALSE(alarm.has_passed(tick::Time_Point{99}));
    EXPECT_FALSE(alarm.has_passed(tick::Time_Point{100}));  // at trigger, not passed
    EXPECT_TRUE(alarm.has_passed(tick::Time_Point{101}));
}
