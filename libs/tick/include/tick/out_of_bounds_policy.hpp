#pragma once

#include <cstdint>

namespace tick {

enum class OutOfBoundsPolicy : std::uint8_t { clamp_to_edge = 0, cycle_last_year = 1, pure_climatology = 2, leap_hold = 3 };

}  // namespace tick
