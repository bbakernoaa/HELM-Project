#pragma once

/// @file generators.hpp
/// @brief Custom RapidCheck generators for TICK property-based tests.
///
/// Provides constrained generators that produce valid inputs within
/// the representable nanosecond range of int64_t (~±292 years from epoch).
/// Requirements: 11.10, 11.11, 11.12

#include <rapidcheck.h>
#include <tick/date_time.hpp>
#include <tick/duration.hpp>
#include <tick/time_point.hpp>
#include <tick/gregorian_calendar.hpp>
#include <tick/noleap_calendar.hpp>
#include <tick/cal360_calendar.hpp>
#include <vector>
#include <cstdint>

namespace tick::gen {

// ─── Date_Time Generators ────────────────────────────────────────────────────

/// Generator for valid Gregorian Date_Time values.
/// Year range constrained to epoch ±~290 years to stay within int64_t nanosecond
/// representable range (epoch = 2026, so approximately [1734, 2318]).
inline auto gregorian_date_time() {
    return rc::gen::exec([] {
        auto year = *rc::gen::inRange<std::int32_t>(1734, 2319);
        auto month = *rc::gen::inRange<std::int32_t>(1, 13);
        auto max_day = Gregorian_Calendar::days_in_month(year, month);
        auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
        auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
        auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
        auto second = *rc::gen::inRange<std::int32_t>(0, 60);
        auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);
        return Date_Time{year, month, day, hour, minute, second, nanosecond};
    });
}

/// Generator for valid NoLeap Date_Time values.
/// Year range constrained to epoch ±~290 years. February always has 28 days.
inline auto noleap_date_time() {
    return rc::gen::exec([] {
        auto year = *rc::gen::inRange<std::int32_t>(1736, 2317);
        auto month = *rc::gen::inRange<std::int32_t>(1, 13);
        auto max_day = NoLeap_Calendar::days_in_month(year, month);
        auto day = *rc::gen::inRange<std::int32_t>(1, max_day + 1);
        auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
        auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
        auto second = *rc::gen::inRange<std::int32_t>(0, 60);
        auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);
        return Date_Time{year, month, day, hour, minute, second, nanosecond};
    });
}

/// Generator for valid Cal360 Date_Time values.
/// Every month has exactly 30 days, every year has 360 days.
/// Year range constrained to epoch ±~290 years.
inline auto cal360_date_time() {
    return rc::gen::exec([] {
        auto year = *rc::gen::inRange<std::int32_t>(1734, 2319);
        auto month = *rc::gen::inRange<std::int32_t>(1, 13);
        auto day = *rc::gen::inRange<std::int32_t>(1, 31); // always 1-30
        auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
        auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
        auto second = *rc::gen::inRange<std::int32_t>(0, 60);
        auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);
        return Date_Time{year, month, day, hour, minute, second, nanosecond};
    });
}

// ─── Duration Generators ─────────────────────────────────────────────────────

/// Generator for Duration values constrained to avoid overflow in arithmetic.
/// Range: ±4×10^18 ns (roughly ±half of int64_t max), leaving room for
/// addition/subtraction of two such values without exceeding int64_t bounds.
inline auto safe_duration() {
    return rc::gen::map(
        rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL,
                                        4'000'000'000'000'000'000LL),
        [](std::int64_t ns) { return Duration{ns}; });
}

/// Generator for positive Duration values up to one day (86,400,000,000,000 ns).
/// Useful for synchronization tests (GCD/LCM) where all timesteps must be positive.
inline auto positive_duration_up_to_one_day() {
    return rc::gen::map(
        rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL),
        [](std::int64_t ns) { return Duration{ns}; });
}

// ─── Collection Generators ───────────────────────────────────────────────────

/// Generator for collections of 2-8 positive Durations, each ≤ 1 day.
/// Suitable for compute_heartbeat / compute_sync_period property tests.
inline auto duration_collection() {
    return rc::gen::exec([] {
        auto size = *rc::gen::inRange(2, 9);
        std::vector<Duration> result;
        result.reserve(size);
        for (int i = 0; i < size; ++i) {
            result.push_back(*positive_duration_up_to_one_day());
        }
        return result;
    });
}

} // namespace tick::gen
