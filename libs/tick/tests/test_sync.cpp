#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <limits>
#include <tick/duration.hpp>
#include <tick/sync.hpp>
#include <tick/time_point.hpp>
#include <vector>

// ============================================================
// Property-based tests (RapidCheck)
// ============================================================

// Property 12: Heartbeat (GCD) divides all time steps
// **Validates: Requirements 8.1, 8.2**
RC_GTEST_PROP(Synchronization, HeartbeatDividesAllSteps, ()) {
    auto size = *rc::gen::inRange(2, 9);  // 2-8 elements
    std::vector<tick::Duration> steps;
    for (int i = 0; i < size; ++i) {
        auto ns = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL);
        steps.push_back(tick::Duration{ns});
    }

    auto heartbeat = tick::compute_heartbeat(steps);
    for (const auto &step : steps) {
        RC_ASSERT(step.nanos() % heartbeat.nanos() == 0);
    }
}

// Property 13: All time steps divide the sync period (LCM)
// **Validates: Requirements 8.2**
RC_GTEST_PROP(Synchronization, AllStepsDivideSyncPeriod, ()) {
    auto size = *rc::gen::inRange(2, 9);
    std::vector<tick::Duration> steps;
    for (int i = 0; i < size; ++i) {
        // Keep values small enough to avoid LCM overflow
        auto ns = *rc::gen::inRange<std::int64_t>(1, 1'000'000'000LL);  // up to 1 second
        steps.push_back(tick::Duration{ns});
    }

    try {
        auto sync_period = tick::compute_sync_period(steps);
        for (const auto &step : steps) {
            RC_ASSERT(sync_period.nanos() % step.nanos() == 0);
        }
    } catch (const std::overflow_error &) {
        // If LCM overflows, that's acceptable — discard this test case
        RC_DISCARD("LCM overflow");
    }
}

// Property 14: Phase alignment matches modulo condition
// **Validates: Requirements 9.1, 11.7, 11.8**
RC_GTEST_PROP(Synchronization, PhaseAlignmentMatchesModulo, ()) {
    auto current_ns = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000LL, 2'000'000'000'000LL);
    auto base_ns = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000LL, 1'000'000'000'000LL);
    auto step_ns = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL);

    auto current = tick::Time_Point{current_ns};
    auto base = tick::Time_Point{base_ns};
    auto step = tick::Duration{step_ns};

    bool expected = (current_ns >= base_ns) && ((current_ns - base_ns) % step_ns == 0);
    RC_ASSERT(tick::is_phase_aligned(current, base, step) == expected);
}

// ============================================================
// Example-based unit tests (Google Test)
// ============================================================

TEST(SyncUnit, ESMTimesteps) {
    // Atmosphere: 300s, Ocean: 900s, Land: 1800s
    std::vector<tick::Duration> steps = {tick::seconds(300), tick::seconds(900), tick::seconds(1800)};

    auto heartbeat = tick::compute_heartbeat(steps);
    EXPECT_EQ(heartbeat, tick::seconds(300));

    auto sync_period = tick::compute_sync_period(steps);
    EXPECT_EQ(sync_period, tick::seconds(1800));
}

TEST(SyncUnit, SingleElement) {
    std::vector<tick::Duration> steps = {tick::seconds(42)};
    EXPECT_EQ(tick::compute_heartbeat(steps), tick::seconds(42));
    EXPECT_EQ(tick::compute_sync_period(steps), tick::seconds(42));
}

TEST(SyncUnit, EmptyCollectionThrows) {
    std::vector<tick::Duration> empty;
    EXPECT_THROW(tick::compute_heartbeat(empty), std::invalid_argument);
    EXPECT_THROW(tick::compute_sync_period(empty), std::invalid_argument);
}

TEST(SyncUnit, ZeroStepThrows) {
    std::vector<tick::Duration> steps = {tick::seconds(10), tick::Duration{0}};
    EXPECT_THROW(tick::compute_heartbeat(steps), std::invalid_argument);
    EXPECT_THROW(tick::compute_sync_period(steps), std::invalid_argument);
}

TEST(SyncUnit, NegativeStepThrows) {
    std::vector<tick::Duration> steps = {tick::seconds(10), tick::Duration{-5}};
    EXPECT_THROW(tick::compute_heartbeat(steps), std::invalid_argument);
    EXPECT_THROW(tick::compute_sync_period(steps), std::invalid_argument);
}

TEST(SyncUnit, LCMOverflowThrows) {
    // Two large primes whose LCM would overflow int64_t
    std::vector<tick::Duration> steps = {tick::Duration{std::numeric_limits<std::int64_t>::max() / 2},
                                         tick::Duration{std::numeric_limits<std::int64_t>::max() / 3 + 1}};
    EXPECT_THROW(tick::compute_sync_period(steps), std::overflow_error);
}

TEST(SyncUnit, PhaseAlignedAtBase) {
    auto base = tick::Time_Point{1000};
    auto step = tick::seconds(10);
    EXPECT_TRUE(tick::is_phase_aligned(base, base, step));
}

TEST(SyncUnit, PhaseAlignedBeforeBaseReturnsFalse) {
    auto base = tick::Time_Point{1000};
    auto current = tick::Time_Point{999};
    auto step = tick::seconds(10);
    EXPECT_FALSE(tick::is_phase_aligned(current, base, step));
}

TEST(SyncUnit, PhaseAlignedZeroStepThrows) {
    EXPECT_THROW(tick::is_phase_aligned(tick::Time_Point{100}, tick::Time_Point{0}, tick::Duration{0}), std::invalid_argument);
}
