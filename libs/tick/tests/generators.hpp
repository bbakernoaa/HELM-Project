#pragma once

/// @file generators.hpp
/// @brief Custom RapidCheck generators for TICK property-based tests.
///
/// Provides constrained generators that produce valid inputs within
/// the representable nanosecond range of int64_t (~±292 years from epoch).
/// Requirements: 11.10, 11.11, 11.12

#include <rapidcheck.h>

#include <cstdint>
#include <tick/cal360_calendar.hpp>
#include <tick/date_time.hpp>
#include <tick/duration.hpp>
#include <tick/gregorian_calendar.hpp>
#include <tick/noleap_calendar.hpp>
#include <tick/out_of_bounds_policy.hpp>
#include <tick/time_point.hpp>
#include <tick/time_window.hpp>
#include <vector>

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
        auto day = *rc::gen::inRange<std::int32_t>(1, 31);  // always 1-30
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
    return rc::gen::map(rc::gen::inRange<std::int64_t>(-4'000'000'000'000'000'000LL, 4'000'000'000'000'000'000LL),
                        [](std::int64_t ns) { return Duration{ns}; });
}

/// Generator for positive Duration values up to one day (86,400,000,000,000 ns).
/// Useful for synchronization tests (GCD/LCM) where all timesteps must be positive.
inline auto positive_duration_up_to_one_day() {
    return rc::gen::map(rc::gen::inRange<std::int64_t>(1, 86'400'000'000'000LL), [](std::int64_t ns) { return Duration{ns}; });
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

// ─── Aliasing-Specific Generators ────────────────────────────────────────────

/// Generator for a valid dataset coverage Time_Window (multi-year range within
/// representable int64_t nanosecond bounds). Produces a half-open interval
/// from Jan 1 of a start year to Jan 1 of an end year (2–30 years apart).
inline auto aliasing_coverage() {
    return rc::gen::exec([] {
        auto start_year = *rc::gen::inRange<std::int32_t>(1800, 2201);
        auto offset_years = *rc::gen::inRange<std::int32_t>(2, 31);
        auto end_year = start_year + offset_years;

        auto start_tp = Gregorian_Calendar::to_time_point(Date_Time{start_year, 1, 1, 0, 0, 0, 0});
        auto end_tp = Gregorian_Calendar::to_time_point(Date_Time{end_year, 1, 1, 0, 0, 0, 0});

        return Time_Window{start_tp, end_tp};
    });
}

/// Generator for a Duration that evenly divides the given coverage duration.
/// Picks from a set of reasonable intervals (1, 5, 10, 30, 90, 365 days)
/// and selects one that divides evenly.
inline auto snapshot_interval_for(Time_Window cov) {
    return rc::gen::exec([cov] {
        auto cov_nanos = cov.duration().nanos();

        // Candidate intervals in nanoseconds
        constexpr std::int64_t candidates[] = {
            86'400'000'000'000LL * 1,   // 1 day
            86'400'000'000'000LL * 5,   // 5 days
            86'400'000'000'000LL * 10,  // 10 days
            86'400'000'000'000LL * 30,  // 30 days
            86'400'000'000'000LL * 90,  // 90 days
            86'400'000'000'000LL * 365  // 365 days
        };

        std::vector<std::int64_t> valid;
        for (auto c : candidates) {
            if (cov_nanos % c == 0) {
                valid.push_back(c);
            }
        }

        // At least 1-day should always divide a multi-year Jan-1 to Jan-1 coverage
        RC_ASSERT(!valid.empty());

        auto idx = *rc::gen::inRange<std::size_t>(0, valid.size());
        return Duration{valid[idx]};
    });
}

/// Generator for a Time_Point strictly within [cov.start(), cov.end()).
inline auto time_point_in_coverage(Time_Window cov) {
    return rc::gen::exec([cov] {
        auto range = cov.duration().nanos();
        // Generate offset in [0, range - 1] to stay strictly within [start, end)
        auto offset = *rc::gen::inRange<std::int64_t>(0, range);
        return Time_Point{cov.start().nanos() + offset};
    });
}

/// Generator for a Time_Point outside coverage: before start or at/after end.
inline auto time_point_outside_coverage(Time_Window cov) {
    return rc::gen::exec([cov] {
        constexpr std::int64_t ten_years_nanos = 86'400'000'000'000LL * 3650;
        auto before = *rc::gen::inRange<int>(0, 2);  // 0 = before, 1 = after

        if (before == 0) {
            // Generate in [start - 10 years, start - 1]
            auto lo = cov.start().nanos() - ten_years_nanos;
            auto hi = cov.start().nanos();
            auto offset = *rc::gen::inRange<std::int64_t>(lo, hi);
            return Time_Point{offset};
        } else {
            // Generate in [end, end + 10 years]
            auto lo = cov.end().nanos();
            auto hi = cov.end().nanos() + ten_years_nanos;
            auto offset = *rc::gen::inRange<std::int64_t>(lo, hi + 1);
            return Time_Point{offset};
        }
    });
}

/// Generator for Time_Point values on February 29 of random Gregorian leap years
/// (varying sub-day components). Year range [1800, 2200].
inline auto feb29_time_point() {
    return rc::gen::exec([] {
        // Generate a leap year in [1800, 2200]
        auto year =
            *rc::gen::suchThat(rc::gen::inRange<std::int32_t>(1800, 2201), [](std::int32_t y) { return Gregorian_Calendar::is_leap_year(y); });

        auto hour = *rc::gen::inRange<std::int32_t>(0, 24);
        auto minute = *rc::gen::inRange<std::int32_t>(0, 60);
        auto second = *rc::gen::inRange<std::int32_t>(0, 60);
        auto nanosecond = *rc::gen::inRange<std::int32_t>(0, 1'000'000'000);

        return Gregorian_Calendar::to_time_point(Date_Time{year, 2, 29, hour, minute, second, nanosecond});
    });
}

/// Generator for Time_Point values guaranteed not on February 29.
/// Uses the existing gregorian_date_time() generator with a filter.
inline auto non_feb29_time_point() {
    return rc::gen::map(rc::gen::suchThat(gregorian_date_time(), [](const Date_Time &dt) { return !(dt.month == 2 && dt.day == 29); }),
                        [](const Date_Time &dt) { return Gregorian_Calendar::to_time_point(dt); });
}

}  // namespace tick::gen
