#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>

#include "tick/duration.hpp"
#include "tick/time_point.hpp"

namespace tick {

// GCD of all time steps — the fundamental heartbeat
[[nodiscard]] Duration compute_heartbeat(std::span<const Duration> time_steps);

// LCM of all time steps — the synchronization period
[[nodiscard]] Duration compute_sync_period(std::span<const Duration> time_steps);

// Phase alignment check
[[nodiscard]] constexpr bool is_phase_aligned(Time_Point current_time, Time_Point base_time, Duration time_step) {
    if (time_step.nanos() <= 0) {
        throw std::invalid_argument("time_step must be positive");
    }
    if (current_time < base_time) {
        return false;
    }
    return (current_time.nanos() - base_time.nanos()) % time_step.nanos() == 0;
}

}  // namespace tick
