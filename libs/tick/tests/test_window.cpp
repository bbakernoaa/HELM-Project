// test_window.cpp — Time_Window property-based tests (Task 8.2) and unit tests (Task 8.3)
// Property 15: Time_Window contains matches half-open interval semantics
// Property 16: compute_window returns a window containing current_time on a boundary
// **Validates: Requirements 10.4, 10.5, 10.10**

#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <tick/duration.hpp>
#include <tick/time_point.hpp>
#include <tick/time_window.hpp>

// ============================================================
// Property-based tests (RapidCheck)
// ============================================================

// --------------------------------------------------------------------------
// Property 15: Time_Window contains matches half-open interval semantics
//
// For any valid Time_Window w (where w.start() < w.end()) and Time_Point t:
//   w.contains(t) == (t >= w.start() && t < w.end())
// **Validates: Requirements 10.4**
// --------------------------------------------------------------------------
RC_GTEST_PROP(TimeWindow, ContainsMatchesHalfOpen, ()) {
    auto start_ns = *rc::gen::inRange<std::int64_t>(-1'000'000'000'000LL, 1'000'000'000'000LL);
    auto duration_ns = *rc::gen::inRange<std::int64_t>(1, 1'000'000'000'000LL);
    auto t_ns = *rc::gen::inRange<std::int64_t>(start_ns - duration_ns, start_ns + 2 * duration_ns);

    auto start = tick::Time_Point{start_ns};
    auto end = tick::Time_Point{start_ns + duration_ns};
    auto t = tick::Time_Point{t_ns};

    tick::Time_Window w{start, end};
    bool expected = (t_ns >= start_ns) && (t_ns < start_ns + duration_ns);
    RC_ASSERT(w.contains(t) == expected);
}

// --------------------------------------------------------------------------
// Property 16: compute_window returns a window containing current_time on a boundary
//
// For any Time_Point current_time (>= 0) and positive Duration interval,
// let w = compute_window(current_time, interval). Then:
//   1) w.contains(current_time) == true
//   2) w.duration() == interval
//   3) is_on_boundary(w.start(), interval) == true
// **Validates: Requirements 10.5, 10.10**
// --------------------------------------------------------------------------
RC_GTEST_PROP(TimeWindow, ComputeWindowContainsCurrent, ()) {
    auto current_ns = *rc::gen::inRange<std::int64_t>(0, 1'000'000'000'000LL);
    auto interval_ns = *rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL);

    auto current = tick::Time_Point{current_ns};
    auto interval = tick::Duration{interval_ns};

    auto w = tick::compute_window(current, interval);

    // Window must contain current_time
    RC_ASSERT(w.contains(current));
    // Window duration must equal interval
    RC_ASSERT(w.duration() == interval);
    // Window start must be on a boundary
    RC_ASSERT(tick::is_on_boundary(w.start(), interval));
}

// ============================================================
// Example-based unit tests (Google Test)
// ============================================================

TEST(TimeWindowErrors, ZeroDurationThrows) {
    EXPECT_THROW(tick::Time_Window(tick::Time_Point{100}, tick::Duration{0}), std::invalid_argument);
}

TEST(TimeWindowErrors, NegativeDurationThrows) {
    EXPECT_THROW(tick::Time_Window(tick::Time_Point{100}, tick::Duration{-10}), std::invalid_argument);
}

TEST(TimeWindowErrors, StartEqualsEndThrows) {
    EXPECT_THROW(tick::Time_Window(tick::Time_Point{100}, tick::Time_Point{100}), std::invalid_argument);
}

TEST(TimeWindowErrors, StartGreaterThanEndThrows) {
    EXPECT_THROW(tick::Time_Window(tick::Time_Point{200}, tick::Time_Point{100}), std::invalid_argument);
}

TEST(TimeWindowContains, AtStartInclusive) {
    tick::Time_Window w{tick::Time_Point{100}, tick::Time_Point{200}};
    EXPECT_TRUE(w.contains(tick::Time_Point{100}));
}

TEST(TimeWindowContains, AtEndExclusive) {
    tick::Time_Window w{tick::Time_Point{100}, tick::Time_Point{200}};
    EXPECT_FALSE(w.contains(tick::Time_Point{200}));
}

TEST(TimeWindowContains, InsideWindow) {
    tick::Time_Window w{tick::Time_Point{100}, tick::Time_Point{200}};
    EXPECT_TRUE(w.contains(tick::Time_Point{150}));
}

TEST(TimeWindowContains, BeforeWindow) {
    tick::Time_Window w{tick::Time_Point{100}, tick::Time_Point{200}};
    EXPECT_FALSE(w.contains(tick::Time_Point{99}));
}

TEST(BoundaryErrors, ZeroDurationThrows) {
    EXPECT_THROW(tick::is_on_boundary(tick::Time_Point{0}, tick::Duration{0}), std::invalid_argument);
}

TEST(BoundaryErrors, NegativeDurationThrows) {
    EXPECT_THROW(tick::is_on_boundary(tick::Time_Point{0}, tick::Duration{-1}), std::invalid_argument);
}

TEST(ComputeWindow, KnownValues) {
    // current=150ns, interval=100ns -> window should be [100, 200)
    auto w = tick::compute_window(tick::Time_Point{150}, tick::Duration{100});
    EXPECT_EQ(w.start(), tick::Time_Point{100});
    EXPECT_EQ(w.end(), tick::Time_Point{200});
    EXPECT_TRUE(w.contains(tick::Time_Point{150}));
}

TEST(ComputeWindow, ExactBoundary) {
    // current=200ns, interval=100ns -> window should be [200, 300)
    auto w = tick::compute_window(tick::Time_Point{200}, tick::Duration{100});
    EXPECT_EQ(w.start(), tick::Time_Point{200});
    EXPECT_EQ(w.end(), tick::Time_Point{300});
}
