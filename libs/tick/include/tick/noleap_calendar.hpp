#pragma once

#include <cstdint>
#include <stdexcept>

#include "tick/date_time.hpp"
#include "tick/time_point.hpp"

namespace tick {

struct NoLeap_Calendar {
    /// Every year has exactly 365 days — no exceptions.
    [[nodiscard]] static constexpr std::int32_t days_in_year([[maybe_unused]] std::int32_t year) noexcept {
        return 365;
    }

    /// Month day counts: Jan=31, Feb=28, Mar=31, Apr=30, May=31, Jun=30,
    ///                   Jul=31, Aug=31, Sep=30, Oct=31, Nov=30, Dec=31
    /// Throws std::invalid_argument for month outside 1–12 or Feb 29.
    [[nodiscard]] static constexpr std::int32_t days_in_month([[maybe_unused]] std::int32_t year, std::int32_t month) {
        if (month < 1 || month > 12) {
            throw std::invalid_argument("Invalid month: value not in range [1, 12]");
        }
        constexpr std::int32_t table[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        return table[month - 1];
    }

    /// Convert a Time_Point (nanoseconds since epoch 2026-01-01) to a Date_Time.
    [[nodiscard]] static constexpr Date_Time to_date_time(Time_Point tp) {
        std::int64_t nanos = tp.nanos();

        // Separate into day_offset and sub-day remainder.
        // Use floor division so that day_in_day_nanos is always non-negative.
        std::int64_t day_offset = nanos / nanos_per_day;
        std::int64_t sub_day_nanos = nanos % nanos_per_day;
        if (sub_day_nanos < 0) {
            day_offset -= 1;
            sub_day_nanos += nanos_per_day;
        }

        // Floor division for year_offset so day_in_year is always [0, 365).
        std::int64_t year_offset;
        std::int64_t day_in_year;
        if (day_offset >= 0) {
            year_offset = day_offset / 365;
            day_in_year = day_offset % 365;
        } else {
            // For negative day_offset, use floor division:
            // floor_div(a, b) when a < 0 and b > 0:
            //   = (a + 1) / b - 1  (for C++ truncation toward zero)
            year_offset = (day_offset + 1) / 365 - 1;
            day_in_year = day_offset - year_offset * 365;
        }

        std::int32_t year = static_cast<std::int32_t>(year_offset) + 2026;  // epoch year

        // Cumulative days at start of each month (0-indexed months)
        constexpr std::int32_t cum_days[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

        // Find month by walking the cumulative table (reverse search)
        std::int32_t month = 12;
        for (std::int32_t m = 11; m >= 0; --m) {
            if (static_cast<std::int64_t>(cum_days[m]) <= day_in_year) {
                month = m + 1;
                break;
            }
        }

        std::int32_t day = static_cast<std::int32_t>(day_in_year - cum_days[month - 1]) + 1;

        // Decompose sub-day nanoseconds into h:m:s:ns
        std::int32_t hour = static_cast<std::int32_t>(sub_day_nanos / nanos_per_hour);
        std::int64_t rem = sub_day_nanos % nanos_per_hour;
        std::int32_t minute = static_cast<std::int32_t>(rem / nanos_per_minute);
        rem = rem % nanos_per_minute;
        std::int32_t second = static_cast<std::int32_t>(rem / nanos_per_second);
        std::int32_t nano = static_cast<std::int32_t>(rem % nanos_per_second);

        return Date_Time{year, month, day, hour, minute, second, nano};
    }

    /// Convert a Date_Time to a Time_Point (nanoseconds since epoch 2026-01-01).
    /// Throws std::invalid_argument for invalid components (including Feb 29).
    [[nodiscard]] static constexpr Time_Point to_time_point(Date_Time dt) {
        validate(dt);

        // Cumulative days at start of each month
        constexpr std::int32_t cum_days[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

        std::int64_t year_offset = static_cast<std::int64_t>(dt.year) - 2026;
        std::int64_t day_offset = year_offset * 365 + cum_days[dt.month - 1] + (dt.day - 1);

        std::int64_t nanos = day_offset * nanos_per_day + static_cast<std::int64_t>(dt.hour) * nanos_per_hour +
                             static_cast<std::int64_t>(dt.minute) * nanos_per_minute + static_cast<std::int64_t>(dt.second) * nanos_per_second +
                             static_cast<std::int64_t>(dt.nanosecond);

        return Time_Point{nanos};
    }

    /// Add calendar months to a Time_Point.
    /// The day is clamped to the last valid day of the target month.
    [[nodiscard]] static constexpr Time_Point add_months(Time_Point tp, std::int32_t months) {
        Date_Time dt = to_date_time(tp);

        // Compute target month/year using 0-based month arithmetic
        std::int32_t total_months = (dt.year - 1) * 12 + (dt.month - 1) + months;

        std::int32_t target_year;
        std::int32_t target_month;
        if (total_months >= 0) {
            target_year = total_months / 12 + 1;
            target_month = total_months % 12 + 1;
        } else {
            // Floor division for negative total_months
            target_year = (total_months - 11) / 12 + 1;
            target_month = total_months - (target_year - 1) * 12 + 1;
        }

        // Clamp day to valid range for target month
        std::int32_t max_day = days_in_month(target_year, target_month);
        std::int32_t target_day = dt.day > max_day ? max_day : dt.day;

        Date_Time result{target_year, target_month, target_day, dt.hour, dt.minute, dt.second, dt.nanosecond};
        return to_time_point(result);
    }

    /// Add calendar years to a Time_Point.
    /// Equivalent to add_months(tp, years * 12).
    [[nodiscard]] static constexpr Time_Point add_years(Time_Point tp, std::int32_t years) {
        return add_months(tp, years * 12);
    }

   private:
    /// Validate all Date_Time components for the NoLeap calendar.
    static constexpr void validate(const Date_Time &dt) {
        if (dt.month < 1 || dt.month > 12) {
            throw std::invalid_argument("Invalid month: value not in range [1, 12]");
        }

        std::int32_t max_day = days_in_month(dt.year, dt.month);

        // Specifically reject Feb 29
        if (dt.month == 2 && dt.day == 29) {
            throw std::invalid_argument("Invalid day: February 29 is invalid in the NoLeap calendar");
        }

        if (dt.day < 1 || dt.day > max_day) {
            throw std::invalid_argument("Invalid day: value out of range for the given month");
        }

        if (dt.hour < 0 || dt.hour > 23) {
            throw std::invalid_argument("Invalid hour: value not in range [0, 23]");
        }

        if (dt.minute < 0 || dt.minute > 59) {
            throw std::invalid_argument("Invalid minute: value not in range [0, 59]");
        }

        if (dt.second < 0 || dt.second > 59) {
            throw std::invalid_argument("Invalid second: value not in range [0, 59]");
        }

        if (dt.nanosecond < 0 || dt.nanosecond > 999'999'999) {
            throw std::invalid_argument("Invalid nanosecond: value not in range [0, 999999999]");
        }
    }
};

}  // namespace tick
