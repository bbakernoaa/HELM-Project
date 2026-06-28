// test_integration.cpp — End-to-end integration tests for TICK
// Validates cross-component workflows: calendar + arithmetic + alarms + sync + windows

#include <gtest/gtest.h>

#include <tick/tick.hpp>
#include <vector>

// ---------------------------------------------------------------------------
// Integration 1: Date_Time → Time_Point → advance by Duration → convert back
// Validates: Requirements 1.8, 3.3
// ---------------------------------------------------------------------------

TEST(Integration, GregorianAdvanceAndConvert) {
    // Start at 2026-06-15 12:00:00
    tick::Date_Time start_dt{2026, 6, 15, 12, 0, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(start_dt);

    // Advance by 30 days
    auto advanced = tp + tick::days(30);
    auto result_dt = tick::Gregorian_Calendar::to_date_time(advanced);

    // 2026-06-15 + 30 days = 2026-07-15
    EXPECT_EQ(result_dt.year, 2026);
    EXPECT_EQ(result_dt.month, 7);
    EXPECT_EQ(result_dt.day, 15);
    EXPECT_EQ(result_dt.hour, 12);
    EXPECT_EQ(result_dt.minute, 0);
    EXPECT_EQ(result_dt.second, 0);
    EXPECT_EQ(result_dt.nanosecond, 0);

    // Round-trip: converting back to Time_Point matches the advanced value
    auto round_trip_tp = tick::Gregorian_Calendar::to_time_point(result_dt);
    EXPECT_EQ(round_trip_tp, advanced);
}

TEST(Integration, GregorianAdvanceAcrossLeapYear) {
    // Start at 2024-02-28 00:00:00 (2024 is a leap year)
    tick::Date_Time start_dt{2024, 2, 28, 0, 0, 0, 0};
    auto tp = tick::Gregorian_Calendar::to_time_point(start_dt);

    // Advance by 1 day → should land on Feb 29 (leap day)
    auto advanced = tp + tick::days(1);
    auto result_dt = tick::Gregorian_Calendar::to_date_time(advanced);

    EXPECT_EQ(result_dt.year, 2024);
    EXPECT_EQ(result_dt.month, 2);
    EXPECT_EQ(result_dt.day, 29);

    // Advance by 2 days from Feb 28 → should land on March 1
    auto advanced2 = tp + tick::days(2);
    auto result_dt2 = tick::Gregorian_Calendar::to_date_time(advanced2);

    EXPECT_EQ(result_dt2.year, 2024);
    EXPECT_EQ(result_dt2.month, 3);
    EXPECT_EQ(result_dt2.day, 1);
}

// ---------------------------------------------------------------------------
// Integration 2: Interval_Alarm rings at correct multiples
// Validates: Requirements 6.2
// ---------------------------------------------------------------------------

TEST(Integration, AlarmRingsAtMultiples) {
    auto base = tick::Gregorian_Calendar::to_time_point({2026, 1, 1, 0, 0, 0, 0});
    auto interval = tick::hours(6);  // ring every 6 hours
    tick::Interval_Alarm alarm{interval, base};

    // Should ring at base + 0, 6h, 12h, 18h, 24h
    EXPECT_TRUE(alarm.is_ringing(base));
    EXPECT_TRUE(alarm.is_ringing(base + tick::hours(6)));
    EXPECT_TRUE(alarm.is_ringing(base + tick::hours(12)));
    EXPECT_TRUE(alarm.is_ringing(base + tick::hours(18)));
    EXPECT_TRUE(alarm.is_ringing(base + tick::hours(24)));

    // Should NOT ring at non-multiples
    EXPECT_FALSE(alarm.is_ringing(base + tick::hours(1)));
    EXPECT_FALSE(alarm.is_ringing(base + tick::hours(7)));
    EXPECT_FALSE(alarm.is_ringing(base + tick::hours(15)));
    EXPECT_FALSE(alarm.is_ringing(base + tick::minutes(30)));

    // Should NOT ring before base
    EXPECT_FALSE(alarm.is_ringing(base - tick::hours(6)));
}

TEST(Integration, AlarmNextRingProgression) {
    auto base = tick::Gregorian_Calendar::to_time_point({2026, 3, 1, 0, 0, 0, 0});
    auto interval = tick::minutes(30);
    tick::Interval_Alarm alarm{interval, base};

    // From the base, next ring should be at base + 30min
    auto next = alarm.next_ring_at(base);
    EXPECT_EQ(next, base + tick::minutes(30));

    // From base + 15min (between rings), next ring is still base + 30min
    auto mid = base + tick::minutes(15);
    auto next2 = alarm.next_ring_at(mid);
    EXPECT_EQ(next2, base + tick::minutes(30));

    // From base + 30min, next ring should be base + 60min
    auto next3 = alarm.next_ring_at(base + tick::minutes(30));
    EXPECT_EQ(next3, base + tick::minutes(60));
}

// ---------------------------------------------------------------------------
// Integration 3: Heartbeat + phase alignment with ESM timesteps
// Validates: Requirements 8.1, 9.1
// ---------------------------------------------------------------------------

TEST(Integration, HeartbeatAndPhaseAlignment) {
    // Real ESM timesteps: atmosphere 5min, ocean 15min, land 30min
    std::vector<tick::Duration> steps = {
        tick::seconds(300),  // atmosphere: 5 min
        tick::seconds(900),  // ocean: 15 min
        tick::seconds(1800)  // land: 30 min
    };

    auto heartbeat = tick::compute_heartbeat(steps);
    // GCD(300s, 900s, 1800s) = 300s
    EXPECT_EQ(heartbeat, tick::seconds(300));

    auto sync_period = tick::compute_sync_period(steps);
    // LCM(300s, 900s, 1800s) = 1800s
    EXPECT_EQ(sync_period, tick::seconds(1800));

    // All models should be phase-aligned at heartbeat multiples
    auto base = tick::Time_Point{0};  // epoch
    for (int k = 0; k <= 10; ++k) {
        auto t = base + heartbeat * k;
        EXPECT_TRUE(tick::is_phase_aligned(t, base, heartbeat));
    }

    // Ocean (900s) should be aligned at 0, 900s, 1800s...
    EXPECT_TRUE(tick::is_phase_aligned(base + tick::seconds(900), base, tick::seconds(900)));
    EXPECT_TRUE(tick::is_phase_aligned(base + tick::seconds(1800), base, tick::seconds(900)));

    // Ocean is NOT aligned at 300s (heartbeat tick, but not ocean's step)
    EXPECT_FALSE(tick::is_phase_aligned(base + tick::seconds(300), base, tick::seconds(900)));

    // At LCM (1800s), ALL models are simultaneously aligned
    auto lcm_time = base + sync_period;
    for (const auto &step : steps) {
        EXPECT_TRUE(tick::is_phase_aligned(lcm_time, base, step));
    }
}

TEST(Integration, HeartbeatDividesAllSteps) {
    // A more complex ESM configuration
    std::vector<tick::Duration> steps = {
        tick::seconds(120),  // fast physics: 2 min
        tick::seconds(600),  // radiation: 10 min
        tick::seconds(3600)  // chemistry: 1 hour
    };

    auto heartbeat = tick::compute_heartbeat(steps);
    // GCD(120, 600, 3600) = 120s
    EXPECT_EQ(heartbeat, tick::seconds(120));

    // Verify heartbeat divides every step exactly
    for (const auto &step : steps) {
        EXPECT_EQ(step.nanos() % heartbeat.nanos(), 0);
    }
}

// ---------------------------------------------------------------------------
// Integration 4: compute_window → verify contains returns true for points inside
// Validates: Requirements 10.4
// ---------------------------------------------------------------------------

TEST(Integration, WindowContainsPoints) {
    auto interval = tick::hours(1);
    // 3.5 hours from epoch
    auto current = tick::Time_Point{tick::nanos_per_hour * 3 + tick::nanos_per_minute * 30};

    auto w = tick::compute_window(current, interval);

    // Window should be [3h, 4h)
    EXPECT_EQ(w.start(), tick::Time_Point{tick::nanos_per_hour * 3});
    EXPECT_EQ(w.end(), tick::Time_Point{tick::nanos_per_hour * 4});
    EXPECT_EQ(w.duration(), interval);

    // current_time should be inside its own window
    EXPECT_TRUE(w.contains(current));

    // Start is inclusive
    EXPECT_TRUE(w.contains(w.start()));

    // End is exclusive
    EXPECT_FALSE(w.contains(w.end()));

    // Points just before and after the window
    EXPECT_FALSE(w.contains(w.start() - tick::nanoseconds(1)));
    EXPECT_TRUE(w.contains(w.end() - tick::nanoseconds(1)));
}

TEST(Integration, WindowBoundaryProgression) {
    // Verify that consecutive windows tile the timeline without gaps or overlaps
    auto interval = tick::minutes(15);
    auto base = tick::Time_Point{0};

    for (int i = 0; i < 10; ++i) {
        auto t = base + interval * i;
        auto w = tick::compute_window(t, interval);

        // The start of each window at a boundary is the boundary itself
        EXPECT_EQ(w.start(), t);

        // The end of one window is the start of the next
        if (i < 9) {
            auto next_w = tick::compute_window(t + interval, interval);
            EXPECT_EQ(w.end(), next_w.start());
        }
    }
}

TEST(Integration, WindowWithCalendarDerivedTime) {
    // Combine calendar conversion with window computation
    auto dt = tick::Gregorian_Calendar::to_time_point({2026, 6, 15, 14, 37, 22, 0});
    auto interval = tick::hours(1);
    auto w = tick::compute_window(dt, interval);

    // Should be in the [14:00, 15:00) window
    auto window_start_dt = tick::Gregorian_Calendar::to_date_time(w.start());
    auto window_end_dt = tick::Gregorian_Calendar::to_date_time(w.end());

    EXPECT_EQ(window_start_dt.hour, 14);
    EXPECT_EQ(window_start_dt.minute, 0);
    EXPECT_EQ(window_start_dt.second, 0);

    EXPECT_EQ(window_end_dt.hour, 15);
    EXPECT_EQ(window_end_dt.minute, 0);
    EXPECT_EQ(window_end_dt.second, 0);

    // Original time is contained
    EXPECT_TRUE(w.contains(dt));
}
