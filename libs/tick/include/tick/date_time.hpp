#pragma once

#include <charconv>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tick {

/// A decomposed calendar representation — a plain aggregate with no invariants
/// of its own. Validation is performed by the calendar engines, not by Date_Time.
struct Date_Time {
    std::int32_t year;
    std::int32_t month;       // 1-12
    std::int32_t day;         // 1-31 (calendar-dependent max)
    std::int32_t hour;        // 0-23
    std::int32_t minute;      // 0-59
    std::int32_t second;      // 0-59
    std::int32_t nanosecond;  // 0-999999999

    constexpr auto operator<=>(const Date_Time &) const noexcept = default;
    constexpr bool operator==(const Date_Time &) const noexcept = default;
};

// ---------------------------------------------------------------------------
// ISO 8601 Parsing and Formatting
// Format: "YYYY-MM-DDThh:mm:ss" (exactly 19 characters)
// ---------------------------------------------------------------------------

/// Parse an ISO 8601 date-time string ("YYYY-MM-DDThh:mm:ss") into a Date_Time.
/// The nanosecond field of the returned Date_Time is always 0.
/// Throws std::invalid_argument on malformed input (with byte offset) or invalid
/// component ranges (month 1-12, day 1-31, hour 0-23, minute 0-59, second 0-59).
/// Note: calendar-specific day validation (e.g., Feb 30) is NOT performed here;
/// that is the responsibility of the Calendar engine.
[[nodiscard]] inline Date_Time parse_iso8601(std::string_view sv) {
    // Validate length
    if (sv.size() != 19) {
        if (sv.size() < 19) {
            throw std::invalid_argument(
                "ISO 8601 parse error: expected 19 characters \"YYYY-MM-DDThh:mm:ss\", "
                "got " +
                std::to_string(sv.size()) + " characters (unexpected end at byte " + std::to_string(sv.size()) + ")");
        } else {
            throw std::invalid_argument(
                "ISO 8601 parse error: expected 19 characters \"YYYY-MM-DDThh:mm:ss\", "
                "got " +
                std::to_string(sv.size()) + " characters (trailing data at byte 19)");
        }
    }

    // Validate separator positions: '-' at 4,7; 'T' at 10; ':' at 13,16
    auto check_sep = [&](std::size_t pos, char expected) {
        if (sv[pos] != expected) {
            throw std::invalid_argument("ISO 8601 parse error: expected '" + std::string(1, expected) + "' at byte " + std::to_string(pos) +
                                        ", got '" + std::string(1, sv[pos]) + "'");
        }
    };
    check_sep(4, '-');
    check_sep(7, '-');
    check_sep(10, 'T');
    check_sep(13, ':');
    check_sep(16, ':');

    // Parse numeric fields using std::from_chars
    auto parse_int = [&](std::size_t start, std::size_t len) -> std::int32_t {
        std::int32_t value{};
        const char *first = sv.data() + start;
        const char *last = first + len;
        auto [ptr, ec] = std::from_chars(first, last, value);
        if (ec != std::errc{} || ptr != last) {
            throw std::invalid_argument("ISO 8601 parse error: invalid numeric field at byte " + std::to_string(start));
        }
        return value;
    };

    std::int32_t year = parse_int(0, 4);
    std::int32_t month = parse_int(5, 2);
    std::int32_t day = parse_int(8, 2);
    std::int32_t hour = parse_int(11, 2);
    std::int32_t minute = parse_int(14, 2);
    std::int32_t second = parse_int(17, 2);

    // Validate component ranges (basic validation; calendar-specific checks are separate)
    if (month < 1 || month > 12) {
        throw std::invalid_argument("ISO 8601 parse error: month " + std::to_string(month) + " not in range [1, 12]");
    }
    if (day < 1 || day > 31) {
        throw std::invalid_argument("ISO 8601 parse error: day " + std::to_string(day) + " not in range [1, 31]");
    }
    if (hour < 0 || hour > 23) {
        throw std::invalid_argument("ISO 8601 parse error: hour " + std::to_string(hour) + " not in range [0, 23]");
    }
    if (minute < 0 || minute > 59) {
        throw std::invalid_argument("ISO 8601 parse error: minute " + std::to_string(minute) + " not in range [0, 59]");
    }
    if (second < 0 || second > 59) {
        throw std::invalid_argument("ISO 8601 parse error: second " + std::to_string(second) + " not in range [0, 59]");
    }

    return Date_Time{year, month, day, hour, minute, second, 0};
}

/// Format a Date_Time as an ISO 8601 string ("YYYY-MM-DDThh:mm:ss").
/// The nanosecond component is discarded. Uses a stack-allocated buffer
/// (no heap allocation for the resulting 19-character string on implementations
/// with small-string optimization for strings ≤ 20 characters).
[[nodiscard]] inline std::string format_iso8601(const Date_Time &dt) {
    // Fixed 19-character output + null terminator
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d", static_cast<int>(dt.year), static_cast<int>(dt.month), static_cast<int>(dt.day),
                  static_cast<int>(dt.hour), static_cast<int>(dt.minute), static_cast<int>(dt.second));
    return std::string(buf, 19);
}

}  // namespace tick
