// test_duration.cpp — Duration & Time_Point tests
// Property-based tests (Task 2.5) and Example-based unit tests (Task 2.6)

#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <limits>
#include <tick/duration.hpp>
#include <tick/time_point.hpp>

// ============================================================
// Property-based tests (RapidCheck)
// ============================================================

// **Validates: Requirements 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.12**

// --------------------------------------------------------------------------
// Property 1: Duration arithmetic is exact integer arithmetic
// --------------------------------------------------------------------------

// Addition: (a + b).nanos() == a.nanos() + b.nanos()
RC_GTEST_PROP(DurationArithmetic, AdditionIsExactIntegerMath, ()) {
    auto a_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto b_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    // Precondition: no overflow
    bool no_overflow =
        (b_val > 0) ? (a_val <= std::numeric_limits<std::int64_t>::max() - b_val) : (a_val >= std::numeric_limits<std::int64_t>::min() - b_val);
    RC_PRE(no_overflow);
    auto a = tick::Duration{a_val};
    auto b = tick::Duration{b_val};
    RC_ASSERT((a + b).nanos() == a_val + b_val);
}

// Subtraction: (a - b).nanos() == a.nanos() - b.nanos()
RC_GTEST_PROP(DurationArithmetic, SubtractionIsExactIntegerMath, ()) {
    auto a_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto b_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    // Precondition: no overflow in subtraction (a - b)
    bool no_overflow =
        (b_val < 0) ? (a_val <= std::numeric_limits<std::int64_t>::max() + b_val) : (a_val >= std::numeric_limits<std::int64_t>::min() + b_val);
    RC_PRE(no_overflow);
    auto a = tick::Duration{a_val};
    auto b = tick::Duration{b_val};
    RC_ASSERT((a - b).nanos() == a_val - b_val);
}

// Multiplication: (a * s).nanos() == a.nanos() * s
RC_GTEST_PROP(DurationArithmetic, MultiplicationIsExactIntegerMath, ()) {
    auto a_val = *rc::gen::inRange<std::int64_t>(-1'000'000'000LL, 1'000'000'000LL);
    auto s_val = *rc::gen::inRange<std::int64_t>(-1'000'000'000LL, 1'000'000'000LL);
    // With these constrained ranges, a_val * s_val fits in int64_t
    auto a = tick::Duration{a_val};
    RC_ASSERT((a * s_val).nanos() == a_val * s_val);
}

// Division: (a / s).nanos() == a.nanos() / s (truncated toward zero)
RC_GTEST_PROP(DurationArithmetic, DivisionIsExactIntegerMath, ()) {
    auto a_val = *rc::gen::arbitrary<std::int64_t>();
    auto s_val = *rc::gen::arbitrary<std::int64_t>();
    // Precondition: divisor is non-zero
    RC_PRE(s_val != 0);
    // Precondition: avoid INT64_MIN / -1 overflow
    RC_PRE(!(a_val == std::numeric_limits<std::int64_t>::min() && s_val == -1));
    auto a = tick::Duration{a_val};
    RC_ASSERT((a / s_val).nanos() == a_val / s_val);
}

// Time_Point + Duration: (tp + d).nanos() == tp.nanos() + d.nanos()
RC_GTEST_PROP(DurationArithmetic, TimePointPlusDurationIsExact, ()) {
    auto tp_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto d_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    // Precondition: no overflow
    bool no_overflow =
        (d_val > 0) ? (tp_val <= std::numeric_limits<std::int64_t>::max() - d_val) : (tp_val >= std::numeric_limits<std::int64_t>::min() - d_val);
    RC_PRE(no_overflow);
    auto tp = tick::Time_Point{tp_val};
    auto d = tick::Duration{d_val};
    RC_ASSERT((tp + d).nanos() == tp_val + d_val);
}

// Time_Point - Time_Point: (tp1 - tp2).nanos() == tp1.nanos() - tp2.nanos()
RC_GTEST_PROP(DurationArithmetic, TimePointMinusTimePointIsExact, ()) {
    auto tp1_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    auto tp2_val = *rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL);
    // Precondition: no overflow in subtraction
    bool no_overflow = (tp2_val < 0) ? (tp1_val <= std::numeric_limits<std::int64_t>::max() + tp2_val)
                                     : (tp1_val >= std::numeric_limits<std::int64_t>::min() + tp2_val);
    RC_PRE(no_overflow);
    auto tp1 = tick::Time_Point{tp1_val};
    auto tp2 = tick::Time_Point{tp2_val};
    RC_ASSERT((tp1 - tp2).nanos() == tp1_val - tp2_val);
}

// --------------------------------------------------------------------------
// Property 17: Comparison operators reflect underlying integer ordering
// --------------------------------------------------------------------------

RC_GTEST_PROP(DurationComparison, ReflectsIntegerOrdering, ()) {
    auto a_val = *rc::gen::arbitrary<std::int64_t>();
    auto b_val = *rc::gen::arbitrary<std::int64_t>();
    auto a = tick::Duration{a_val};
    auto b = tick::Duration{b_val};
    RC_ASSERT((a < b) == (a_val < b_val));
    RC_ASSERT((a > b) == (a_val > b_val));
    RC_ASSERT((a == b) == (a_val == b_val));
    RC_ASSERT((a != b) == (a_val != b_val));
    RC_ASSERT((a <= b) == (a_val <= b_val));
    RC_ASSERT((a >= b) == (a_val >= b_val));
}

RC_GTEST_PROP(TimePointComparison, ReflectsIntegerOrdering, ()) {
    auto a_val = *rc::gen::arbitrary<std::int64_t>();
    auto b_val = *rc::gen::arbitrary<std::int64_t>();
    auto a = tick::Time_Point{a_val};
    auto b = tick::Time_Point{b_val};
    RC_ASSERT((a < b) == (a_val < b_val));
    RC_ASSERT((a > b) == (a_val > b_val));
    RC_ASSERT((a == b) == (a_val == b_val));
    RC_ASSERT((a != b) == (a_val != b_val));
    RC_ASSERT((a <= b) == (a_val <= b_val));
    RC_ASSERT((a >= b) == (a_val >= b_val));
}

// --------------------------------------------------------------------------
// Property 18: Factory functions are exact multiplications of conversion factors
// --------------------------------------------------------------------------

RC_GTEST_PROP(FactoryFunctionsProperty, AreExactMultiplications, ()) {
    // Test with values small enough to avoid overflow across all factory functions
    auto count = *rc::gen::inRange<std::int64_t>(-1'000'000'000LL, 1'000'000'000LL);
    RC_ASSERT(tick::nanoseconds(count).nanos() == count * 1);
    RC_ASSERT(tick::microseconds(count).nanos() == count * 1'000);
    RC_ASSERT(tick::milliseconds(count).nanos() == count * 1'000'000);

    // For seconds/minutes, use a smaller range to avoid overflow
    auto small = *rc::gen::inRange<std::int64_t>(-1'000'000LL, 1'000'000LL);
    RC_ASSERT(tick::seconds(small).nanos() == small * 1'000'000'000LL);
    RC_ASSERT(tick::minutes(small).nanos() == small * 60'000'000'000LL);

    // For hours/days, use an even smaller range
    auto tiny = *rc::gen::inRange<std::int64_t>(-100'000LL, 100'000LL);
    RC_ASSERT(tick::hours(tiny).nanos() == tiny * 3'600'000'000'000LL);
    RC_ASSERT(tick::days(tiny).nanos() == tiny * 86'400'000'000'000LL);
}

// ============================================================
// Example-based unit tests (Google Test) — Task 2.6
// ============================================================

// --- Overflow detection ---
TEST(DurationOverflow, AdditionOverflows) {
    auto big = tick::Duration{std::numeric_limits<std::int64_t>::max() - 1};
    auto two = tick::Duration{2};
    EXPECT_THROW(big + two, std::overflow_error);
}

TEST(DurationOverflow, SubtractionOverflows) {
    auto small = tick::Duration{std::numeric_limits<std::int64_t>::min() + 1};
    auto two = tick::Duration{2};
    EXPECT_THROW(small - two, std::overflow_error);
}

TEST(DurationOverflow, MultiplicationOverflows) {
    auto big = tick::Duration{std::numeric_limits<std::int64_t>::max() / 2 + 1};
    EXPECT_THROW(big * 2, std::overflow_error);
}

// --- Division by zero ---
TEST(DurationDivision, DivisionByZeroThrows) {
    auto d = tick::Duration{100};
    EXPECT_THROW(d / 0, std::invalid_argument);
}

// --- Factory functions (known values) ---
TEST(FactoryFunctions, KnownValues) {
    EXPECT_EQ(tick::nanoseconds(42).nanos(), 42);
    EXPECT_EQ(tick::microseconds(1).nanos(), 1'000);
    EXPECT_EQ(tick::milliseconds(1).nanos(), 1'000'000);
    EXPECT_EQ(tick::seconds(1).nanos(), 1'000'000'000LL);
    EXPECT_EQ(tick::minutes(1).nanos(), 60'000'000'000LL);
    EXPECT_EQ(tick::hours(1).nanos(), 3'600'000'000'000LL);
    EXPECT_EQ(tick::days(1).nanos(), 86'400'000'000'000LL);
}

// --- Constexpr correctness ---
static_assert(tick::Duration{5}.nanos() == 5);
static_assert((tick::Duration{3} + tick::Duration{4}).nanos() == 7);
static_assert((tick::Duration{10} - tick::Duration{3}).nanos() == 7);
static_assert((tick::Duration{6} * 3).nanos() == 18);
static_assert((tick::Duration{15} / 3).nanos() == 5);
static_assert(tick::Duration{3} < tick::Duration{5});
static_assert(tick::seconds(1) == tick::Duration{1'000'000'000LL});

// --- Time_Point arithmetic ---
static_assert(tick::Time_Point{100}.nanos() == 100);
static_assert((tick::Time_Point{10} + tick::Duration{5}).nanos() == 15);
static_assert((tick::Time_Point{10} - tick::Duration{3}).nanos() == 7);
static_assert((tick::Time_Point{10} - tick::Time_Point{3}).nanos() == 7);
static_assert(tick::epoch.nanos() == 0);

// --- Time_Point overflow ---
TEST(TimePointOverflow, AdditionOverflows) {
    auto tp = tick::Time_Point{std::numeric_limits<std::int64_t>::max() - 1};
    auto d = tick::Duration{2};
    EXPECT_THROW(tp + d, std::overflow_error);
}

TEST(TimePointOverflow, SubtractionOverflows) {
    auto tp = tick::Time_Point{std::numeric_limits<std::int64_t>::min() + 1};
    auto d = tick::Duration{2};
    EXPECT_THROW(tp - d, std::overflow_error);
}

TEST(TimePointOverflow, TimePointDifferenceOverflows) {
    auto tp1 = tick::Time_Point{std::numeric_limits<std::int64_t>::max()};
    auto tp2 = tick::Time_Point{std::numeric_limits<std::int64_t>::min()};
    EXPECT_THROW(tp1 - tp2, std::overflow_error);
}
