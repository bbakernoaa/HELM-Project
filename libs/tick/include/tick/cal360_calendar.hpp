#pragma once

#include <cstdint>
#include <stdexcept>

#include "tick/time_point.hpp"
#include "tick/date_time.hpp"

namespace tick {

/// Cal360_Calendar — a synthetic calendar where every month has exactly 30 days
/// and every year has exactly 360 days (12 months × 30 days).
/// This simplifies long-term climate statistics by providing uniform monthly periods.
struct Cal360_Calendar {

    /// Convert a Time_Point to a decomposed Date_Time.
    /// Epoch is 2026-01-01T00:00:00 (Time_Point{0}).
    [[nodiscard]] static constexpr Date_Time to_date_time(Time_Point tp)
    {
        std::int64_t total_nanos = tp.nanos();

        // Decompose into day_offset and sub-day nanoseconds.
        // Use floor division to handle negative time points correctly.
        std::int64_t day_offset = floor_div(total_nanos, nanos_per_day);
        std::int64_t sub_day_nanos = total_nanos - day_offset * nanos_per_day;

        // Decompose day_offset into year_offset and day_in_year using floor division.
        std::int32_t year_offset = static_cast<std::int32_t>(floor_div(day_offset, 360));
        std::int64_t day_in_year = day_offset - static_cast<std::int64_t>(year_offset) * 360;

        // Derive month (1-12) and day (1-30) from day_in_year (0-359).
        std::int32_t month = static_cast<std::int32_t>(day_in_year / 30) + 1;
        std::int32_t day   = static_cast<std::int32_t>(day_in_year % 30) + 1;

        // Decompose sub-day nanoseconds into hour, minute, second, nanosecond.
        std::int32_t hour       = static_cast<std::int32_t>(sub_day_nanos / nanos_per_hour);
        std::int64_t remainder  = sub_day_nanos % nanos_per_hour;
        std::int32_t minute     = static_cast<std::int32_t>(remainder / nanos_per_minute);
        remainder               = remainder % nanos_per_minute;
        std::int32_t second     = static_cast<std::int32_t>(remainder / nanos_per_second);
        std::int32_t nanosecond = static_cast<std::int32_t>(remainder % nanos_per_second);

        // Apply epoch year offset.
        std::int32_t year = epoch_year + year_offset;

        return Date_Time{year, month, day, hour, minute, second, nanosecond};
    }

    /// Convert a decomposed Date_Time to a Time_Point.
    /// Validates all components and throws std::invalid_argument on invalid input.
    [[nodiscard]] static constexpr Time_Point to_time_point(Date_Time dt)
    {
        validate(dt);

        // Compute year offset from epoch.
        std::int32_t year_offset = dt.year - epoch_year;

        // Total days from epoch.
        std::int64_t day_offset =
            static_cast<std::int64_t>(year_offset) * 360 +
            static_cast<std::int64_t>(dt.month - 1) * 30 +
            static_cast<std::int64_t>(dt.day - 1);

        // Sub-day nanoseconds.
        std::int64_t sub_day_nanos =
            static_cast<std::int64_t>(dt.hour)       * nanos_per_hour +
            static_cast<std::int64_t>(dt.minute)     * nanos_per_minute +
            static_cast<std::int64_t>(dt.second)     * nanos_per_second +
            static_cast<std::int64_t>(dt.nanosecond);

        std::int64_t total_nanos = day_offset * nanos_per_day + sub_day_nanos;

        return Time_Point{total_nanos};
    }

    /// Returns the number of days in a given month (always 30 for Cal360).
    /// Validates month range, throws std::invalid_argument if month is out of [1,12].
    [[nodiscard]] static constexpr std::int32_t days_in_month(
        [[maybe_unused]] std::int32_t year, std::int32_t month)
    {
        if (month < 1 || month > 12) {
            throw std::invalid_argument(
                "Invalid month: value not in range [1, 12]");
        }
        return 30;
    }

    /// Returns the number of days in a given year (always 360 for Cal360).
    [[nodiscard]] static constexpr std::int32_t days_in_year(
        [[maybe_unused]] std::int32_t year) noexcept
    {
        return 360;
    }

    /// Add a number of months to a Time_Point.
    /// In Cal360, every month is exactly 30 days, so this is a fixed nanosecond shift.
    [[nodiscard]] static constexpr Time_Point add_months(Time_Point tp, std::int32_t months)
    {
        std::int64_t shift = static_cast<std::int64_t>(months) * 30 * nanos_per_day;
        return Time_Point{tp.nanos() + shift};
    }

    /// Add a number of years to a Time_Point.
    /// In Cal360, every year is exactly 360 days, so this is a fixed nanosecond shift.
    [[nodiscard]] static constexpr Time_Point add_years(Time_Point tp, std::int32_t years)
    {
        std::int64_t shift = static_cast<std::int64_t>(years) * 360 * nanos_per_day;
        return Time_Point{tp.nanos() + shift};
    }

private:
    static constexpr std::int32_t epoch_year = 2026;

    /// Floor division: rounds toward negative infinity (unlike C++ truncation toward zero).
    [[nodiscard]] static constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept
    {
        std::int64_t q = a / b;
        std::int64_t r = a % b;
        // If remainder is nonzero and signs of a and b differ, adjust downward.
        if ((r != 0) && ((r ^ b) < 0)) {
            --q;
        }
        return q;
    }

    /// Validate a Date_Time for Cal360 rules.
    static constexpr void validate(const Date_Time& dt)
    {
        if (dt.month < 1 || dt.month > 12) {
            throw std::invalid_argument(
                "Invalid month: value not in range [1, 12]");
        }
        if (dt.day < 1 || dt.day > 30) {
            throw std::invalid_argument(
                "Invalid day: value not in range [1, 30]");
        }
        if (dt.hour < 0 || dt.hour > 23) {
            throw std::invalid_argument(
                "Invalid hour: value not in range [0, 23]");
        }
        if (dt.minute < 0 || dt.minute > 59) {
            throw std::invalid_argument(
                "Invalid minute: value not in range [0, 59]");
        }
        if (dt.second < 0 || dt.second > 59) {
            throw std::invalid_argument(
                "Invalid second: value not in range [0, 59]");
        }
        if (dt.nanosecond < 0 || dt.nanosecond > 999'999'999) {
            throw std::invalid_argument(
                "Invalid nanosecond: value not in range [0, 999999999]");
        }
    }
};

} // namespace tick
