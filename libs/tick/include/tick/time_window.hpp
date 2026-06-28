#pragma once

#include <compare>
#include <stdexcept>

#include "tick/duration.hpp"
#include "tick/time_point.hpp"

namespace tick {

class Time_Window {
   public:
    constexpr Time_Window(Time_Point start, Time_Point end) : start_{start}, end_{end} {
        if (!(start < end)) {
            throw std::invalid_argument("Time_Window requires start < end");
        }
    }

    constexpr Time_Window(Time_Point start, Duration duration) : Time_Window{start, start + duration} {}

    [[nodiscard]] constexpr bool contains(Time_Point t) const noexcept {
        return t >= start_ && t < end_;
    }

    [[nodiscard]] constexpr Duration duration() const noexcept {
        return Duration{end_.nanos() - start_.nanos()};
    }

    [[nodiscard]] constexpr Time_Point start() const noexcept {
        return start_;
    }
    [[nodiscard]] constexpr Time_Point end() const noexcept {
        return end_;
    }

    constexpr auto operator<=>(const Time_Window &) const noexcept = default;
    constexpr bool operator==(const Time_Window &) const noexcept = default;

   private:
    Time_Point start_;
    Time_Point end_;
};

// Boundary detection: is current_time on a boundary of the given interval from epoch?
[[nodiscard]] constexpr bool is_on_boundary(Time_Point current_time, Duration interval) {
    if (interval.nanos() <= 0) {
        throw std::invalid_argument("is_on_boundary interval must be positive");
    }
    return current_time.nanos() % interval.nanos() == 0;
}

// Window computation: find the enclosing window for current_time
[[nodiscard]] constexpr Time_Window compute_window(Time_Point current_time, Duration interval) {
    if (interval.nanos() <= 0) {
        throw std::invalid_argument("compute_window interval must be positive");
    }

    auto nanos = current_time.nanos();
    auto iv = interval.nanos();

    // Floor division: largest multiple of interval <= current_time
    // For negative dividends, C++ truncates toward zero, so we adjust
    auto floor_nanos = nanos >= 0 ? (nanos / iv) * iv : ((nanos - iv + 1) / iv) * iv;

    auto floor_time = Time_Point{floor_nanos};
    return Time_Window{floor_time, floor_time + interval};
}

}  // namespace tick
