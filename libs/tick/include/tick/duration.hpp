#pragma once

#include <cstdint>
#include <compare>
#include <stdexcept>

#include "tick/detail/overflow.hpp"

namespace tick {

class Duration {
public:
    constexpr Duration() noexcept = default;
    constexpr explicit Duration(std::int64_t nanos) noexcept : nanos_{nanos} {}

    [[nodiscard]] constexpr std::int64_t nanos() const noexcept { return nanos_; }

    // Arithmetic operators

    constexpr Duration operator+(Duration other) const
    {
        return Duration{detail::checked_add(nanos_, other.nanos_)};
    }

    constexpr Duration operator-(Duration other) const
    {
        return Duration{detail::checked_sub(nanos_, other.nanos_)};
    }

    constexpr Duration operator*(std::int64_t scalar) const
    {
        return Duration{detail::checked_mul(nanos_, scalar)};
    }

    constexpr Duration operator/(std::int64_t scalar) const
    {
        if (scalar == 0) {
            throw std::invalid_argument("Duration division by zero");
        }
        return Duration{nanos_ / scalar};
    }

    constexpr Duration operator-() const noexcept
    {
        return Duration{-nanos_};
    }

    // Compound assignment operators

    constexpr Duration& operator+=(Duration other)
    {
        nanos_ = detail::checked_add(nanos_, other.nanos_);
        return *this;
    }

    constexpr Duration& operator-=(Duration other)
    {
        nanos_ = detail::checked_sub(nanos_, other.nanos_);
        return *this;
    }

    constexpr Duration& operator*=(std::int64_t scalar)
    {
        nanos_ = detail::checked_mul(nanos_, scalar);
        return *this;
    }

    constexpr Duration& operator/=(std::int64_t scalar)
    {
        if (scalar == 0) {
            throw std::invalid_argument("Duration division by zero");
        }
        nanos_ /= scalar;
        return *this;
    }

    // Comparison (C++20 spaceship)
    constexpr auto operator<=>(const Duration&) const noexcept = default;
    constexpr bool operator==(const Duration&)  const noexcept = default;

private:
    std::int64_t nanos_{0};
};

// Factory functions

[[nodiscard]] constexpr Duration nanoseconds(std::int64_t count) noexcept
{
    return Duration{count};
}

[[nodiscard]] constexpr Duration microseconds(std::int64_t count) noexcept
{
    return Duration{count * 1'000};
}

[[nodiscard]] constexpr Duration milliseconds(std::int64_t count) noexcept
{
    return Duration{count * 1'000'000};
}

[[nodiscard]] constexpr Duration seconds(std::int64_t count) noexcept
{
    return Duration{count * 1'000'000'000LL};
}

[[nodiscard]] constexpr Duration minutes(std::int64_t count) noexcept
{
    return Duration{count * 60'000'000'000LL};
}

[[nodiscard]] constexpr Duration hours(std::int64_t count) noexcept
{
    return Duration{count * 3'600'000'000'000LL};
}

[[nodiscard]] constexpr Duration days(std::int64_t count) noexcept
{
    return Duration{count * 86'400'000'000'000LL};
}

} // namespace tick
