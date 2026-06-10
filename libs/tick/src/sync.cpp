#include "tick/sync.hpp"

#include <cmath>
#include <limits>
#include <numeric>

namespace tick {

Duration compute_heartbeat(std::span<const Duration> time_steps)
{
    if (time_steps.empty()) {
        throw std::invalid_argument("At least one time step required");
    }

    std::int64_t result = 0;
    for (const auto& step : time_steps) {
        if (step.nanos() <= 0) {
            throw std::invalid_argument("All time steps must be positive");
        }
        result = std::gcd(result, step.nanos());
    }
    return Duration{result};
}

Duration compute_sync_period(std::span<const Duration> time_steps)
{
    if (time_steps.empty()) {
        throw std::invalid_argument("At least one time step required");
    }

    std::int64_t result = 1;
    for (const auto& step : time_steps) {
        if (step.nanos() <= 0) {
            throw std::invalid_argument("All time steps must be positive");
        }
        std::int64_t g = std::gcd(result, step.nanos());
        std::int64_t div = result / g;
        // Overflow check: div * step.nanos() must fit in int64_t
        if (step.nanos() != 0 &&
            div > std::numeric_limits<std::int64_t>::max() / step.nanos()) {
            throw std::overflow_error("Integer overflow in LCM computation");
        }
        result = div * step.nanos();
    }
    return Duration{result};
}

} // namespace tick
