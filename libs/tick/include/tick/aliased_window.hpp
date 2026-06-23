#pragma once

#include <cstring>

#include "tick/time_window.hpp"

namespace tick {

struct AliasedWindow {
    Time_Window window;
    double      alpha;   // interpolation weight in [0.0, 1.0]

    constexpr bool operator==(const AliasedWindow& other) const noexcept
    {
        // Bitwise comparison of alpha for determinism (not epsilon-based)
        return window == other.window &&
               std::memcmp(&alpha, &other.alpha, sizeof(double)) == 0;
    }

    /// Factory returning a sentinel AliasedWindow for default-construction scenarios.
    /// Uses Time_Window{Time_Point{0}, Time_Point{1}} since Time_Window requires start < end.
    static AliasedWindow zero() noexcept
    {
        return AliasedWindow{Time_Window{Time_Point{0}, Time_Point{1}}, 0.0};
    }
};

} // namespace tick
