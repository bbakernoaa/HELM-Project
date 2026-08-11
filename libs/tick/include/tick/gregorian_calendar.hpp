#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "tick/date_time.hpp"
#include "tick/time_point.hpp"

namespace tick {

struct Gregorian_Calendar {
    // ─── Leap year ───────────────────────────────────────────────────────

    [[nodiscard]] static constexpr bool is_leap_year(std::int32_t year) noexcept {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }

    // ─── Days in month / year ────────────────────────────────────────────

    [[nodiscard]] static constexpr std::int32_t days_in_month(std::int32_t year, std::int32_t month) {
        if (month < 1 || month > 12) {
            throw std::invalid_argument("Invalid month: " + std::to_string(month) + " not in range [1, 12]");
        }
        constexpr std::int32_t table[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (month == 2 && is_leap_year(year)) {
            return 29;
        }
        return table[month - 1];
    }

    [[nodiscard]] static constexpr std::int32_t days_in_year(std::int32_t year) noexcept {
        return is_leap_year(year) ? 366 : 365;
    }

    // ─── Day of week (ISO 8601: 1=Monday ... 7=Sunday) ─────────────────────

    [[nodiscard]] static constexpr std::int32_t day_of_week(Date_Time dt) {
        validate(dt);
        std::int64_t z = civil_to_days(dt.year, dt.month, dt.day);
        std::int64_t w = (z + 3) % 7;
        if (w < 0) w += 7;
        return static_cast<std::int32_t>(w + 1);
    }

    // ─── Date_Time → Time_Point ──────────────────────────────────────────

    [[nodiscard]] static constexpr Time_Point to_time_point(Date_Time dt) {
        validate(dt);

        // Convert civil date to days since the algorithm's internal epoch (0000-03-01)
        std::int64_t days = civil_to_days(dt.year, dt.month, dt.day);

        // Subtract the epoch offset so that 2026-01-01 maps to day 0
        days -= epoch_day_offset();

        // Accumulate sub-day nanoseconds
        std::int64_t nanos = days * nanos_per_day;
        nanos += static_cast<std::int64_t>(dt.hour) * nanos_per_hour;
        nanos += static_cast<std::int64_t>(dt.minute) * nanos_per_minute;
        nanos += static_cast<std::int64_t>(dt.second) * nanos_per_second;
        nanos += static_cast<std::int64_t>(dt.nanosecond);

        return Time_Point{nanos};
    }

    // ─── Time_Point → Date_Time ──────────────────────────────────────────

    [[nodiscard]] static constexpr Date_Time to_date_time(Time_Point tp) {
        std::int64_t nanos = tp.nanos();

        // Compute day offset and sub-day remainder
        // Use floored division so that negative nanos map to the previous day
        std::int64_t day_offset = floor_div(nanos, nanos_per_day);
        std::int64_t sub_day = nanos - day_offset * nanos_per_day;
        // sub_day is now in [0, nanos_per_day)

        // Shift day_offset to the algorithm's internal epoch
        day_offset += epoch_day_offset();

        // civil-from-days (Hinnant's algorithm)
        auto [year, month, day] = days_to_civil(day_offset);

        // Decompose sub-day nanoseconds
        std::int32_t hour = static_cast<std::int32_t>(sub_day / nanos_per_hour);
        sub_day %= nanos_per_hour;
        std::int32_t minute = static_cast<std::int32_t>(sub_day / nanos_per_minute);
        sub_day %= nanos_per_minute;
        std::int32_t second = static_cast<std::int32_t>(sub_day / nanos_per_second);
        std::int32_t nanosecond = static_cast<std::int32_t>(sub_day % nanos_per_second);

        return Date_Time{year, month, day, hour, minute, second, nanosecond};
    }

    // ─── Calendar-aware month addition ───────────────────────────────────

    [[nodiscard]] static constexpr Time_Point add_months(Time_Point tp, std::int32_t months) {
        Date_Time dt = to_date_time(tp);

        // Advance month (0-indexed arithmetic)
        std::int32_t total_months = (dt.year * 12 + (dt.month - 1)) + months;
        std::int32_t new_year = floor_div_i32(total_months, 12);
        std::int32_t new_month = total_months - new_year * 12 + 1;

        // Clamp day
        std::int32_t max_day = days_in_month(new_year, new_month);
        std::int32_t new_day = dt.day < max_day ? dt.day : max_day;

        return to_time_point(Date_Time{new_year, new_month, new_day, dt.hour, dt.minute, dt.second, dt.nanosecond});
    }

    // ─── Calendar-aware year addition ────────────────────────────────────

    [[nodiscard]] static constexpr Time_Point add_years(Time_Point tp, std::int32_t years) {
        return add_months(tp, years * 12);
    }

   private:
    // ─── Validation ──────────────────────────────────────────────────────

    static constexpr void validate(const Date_Time &dt) {
        if (dt.month < 1 || dt.month > 12) {
            throw std::invalid_argument("Invalid month: " + std::to_string(dt.month) + " not in range [1, 12]");
        }

        std::int32_t max_day = days_in_month(dt.year, dt.month);
        if (dt.day < 1 || dt.day > max_day) {
            throw std::invalid_argument("Invalid day: " + std::to_string(dt.day) + " not in range [1, " + std::to_string(max_day) + "]");
        }

        if (dt.hour < 0 || dt.hour > 23) {
            throw std::invalid_argument("Invalid hour: " + std::to_string(dt.hour) + " not in range [0, 23]");
        }

        if (dt.minute < 0 || dt.minute > 59) {
            throw std::invalid_argument("Invalid minute: " + std::to_string(dt.minute) + " not in range [0, 59]");
        }

        if (dt.second < 0 || dt.second > 59) {
            throw std::invalid_argument("Invalid second: " + std::to_string(dt.second) + " not in range [0, 59]");
        }

        if (dt.nanosecond < 0 || dt.nanosecond > 999'999'999) {
            throw std::invalid_argument("Invalid nanosecond: " + std::to_string(dt.nanosecond) + " not in range [0, 999999999]");
        }
    }

    // ─── Hinnant's civil_from_days algorithm ─────────────────────────────
    // Reference: https://howardhinnant.github.io/date_algorithms.html
    // Internal epoch: 0000-03-01 (day 0)

    struct Civil {
        std::int32_t year;
        std::int32_t month;
        std::int32_t day;
    };

    /// Convert a civil day number (days since 0000-03-01) to year/month/day
    [[nodiscard]] static constexpr Civil days_to_civil(std::int64_t z) noexcept {
        // Adjust so that day 0 = 0000-03-01
        z += 719468;  // days from 0000-03-01 to 1970-01-01 (Unix epoch offset)
        // Now z is days since 1970-01-01... but we want the algorithm that takes
        // days since the internal epoch. Let's use the standard formulation:
        // The algorithm from Hinnant uses days since 1970-01-01 (Unix epoch) as input.
        // We adjusted above so z is now relative to that.

        const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
        const std::int64_t doe = z - era * 146097;  // day of era [0, 146096]
        const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const std::int64_t y = yoe + era * 400;
        const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);  // day of year [0, 365]
        const std::int64_t mp = (5 * doy + 2) / 153;                       // month in March-based year [0, 11]
        const std::int64_t d = doy - (153 * mp + 2) / 5 + 1;
        const std::int64_t m = mp + (mp < 10 ? 3 : -9);
        const std::int64_t yr = y + (m <= 2 ? 1 : 0);

        return Civil{static_cast<std::int32_t>(yr), static_cast<std::int32_t>(m), static_cast<std::int32_t>(d)};
    }

    /// Convert year/month/day to civil day number (days since internal epoch)
    /// The internal epoch here means: we produce a value that when passed to
    /// days_to_civil produces the same y/m/d back.
    [[nodiscard]] static constexpr std::int64_t civil_to_days(std::int32_t y, std::int32_t m, std::int32_t d) noexcept {
        // Shift to March-based year
        const std::int64_t yr = static_cast<std::int64_t>(y) - (m <= 2 ? 1 : 0);
        const std::int64_t era = (yr >= 0 ? yr : yr - 399) / 400;
        const std::int64_t yoe = yr - era * 400;
        const std::int64_t mp = static_cast<std::int64_t>(m) + (m > 2 ? -3 : 9);  // March-based month [0, 11]
        const std::int64_t doy = (153 * mp + 2) / 5 + d - 1;
        const std::int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        const std::int64_t days = era * 146097 + doe;

        // days is relative to 0000-03-01; convert to match days_to_civil's input
        // (days_to_civil adds 719468, so we subtract 719468 here to get consistent values)
        return days - 719468;
    }

    /// The day number of 2026-01-01 in our civil day system
    /// This is the offset that makes Time_Point{0} correspond to 2026-01-01
    [[nodiscard]] static constexpr std::int64_t epoch_day_offset() noexcept {
        return civil_to_days(2026, 1, 1);
    }

    // ─── Utility ─────────────────────────────────────────────────────────

    /// Floored integer division (rounds towards negative infinity)
    [[nodiscard]] static constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept {
        return a / b - (a % b != 0 && (a ^ b) < 0);
    }

    /// Floored integer division for int32_t
    [[nodiscard]] static constexpr std::int32_t floor_div_i32(std::int32_t a, std::int32_t b) noexcept {
        return a / b - (a % b != 0 && (a ^ b) < 0);
    }
};

}  // namespace tick
