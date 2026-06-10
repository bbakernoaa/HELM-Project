#pragma once

#include <cstdint>
#include <compare>

#include "tick/duration.hpp"

namespace tick {

class Time_Point {
public:
    constexpr Time_Point() noexcept = default;
    constexpr explicit Time_Point(std::int64_t nanos_since_epoch) noexcept
        : nanos_{nanos_since_epoch} {}

    [[nodiscard]] constexpr std::int64_t nanos() const noexcept { return nanos_; }

    // Arithmetic operators

    constexpr Time_Point operator+(Duration d) const
    {
        return Time_Point{detail::checked_add(nanos_, d.nanos())};
    }

    constexpr Time_Point operator-(Duration d) const
    {
        return Time_Point{detail::checked_sub(nanos_, d.nanos())};
    }

    constexpr Duration operator-(Time_Point other) const
    {
        return Duration{detail::checked_sub(nanos_, other.nanos_)};
    }

    // Compound assignment operators

    constexpr Time_Point& operator+=(Duration d)
    {
        nanos_ = detail::checked_add(nanos_, d.nanos());
        return *this;
    }

    constexpr Time_Point& operator-=(Duration d)
    {
        nanos_ = detail::checked_sub(nanos_, d.nanos());
        return *this;
    }

    // Comparison (C++20 spaceship)
    constexpr auto operator<=>(const Time_Point&) const noexcept = default;
    constexpr bool operator==(const Time_Point&)  const noexcept = default;

private:
    std::int64_t nanos_{0};
};

// Nanosecond conversion constants
inline constexpr std::int64_t nanos_per_second = 1'000'000'000LL;
inline constexpr std::int64_t nanos_per_minute = 60LL * nanos_per_second;
inline constexpr std::int64_t nanos_per_hour   = 60LL * nanos_per_minute;
inline constexpr std::int64_t nanos_per_day    = 24LL * nanos_per_hour;

// Epoch: 2026-01-01T00:00:00 — nanos() == 0 at this instant
inline constexpr Time_Point epoch{0};

} // namespace tick
