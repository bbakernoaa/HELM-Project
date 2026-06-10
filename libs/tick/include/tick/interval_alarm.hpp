#pragma once

#include <stdexcept>

#include "tick/duration.hpp"
#include "tick/time_point.hpp"

namespace tick {

class Interval_Alarm {
public:
    constexpr Interval_Alarm(Duration interval, Time_Point reference)
        : interval_{interval}, reference_{reference}
    {
        if (interval.nanos() <= 0) {
            throw std::invalid_argument("Interval_Alarm interval must be positive");
        }
    }

    [[nodiscard]] constexpr bool is_ringing(Time_Point current_time) const noexcept
    {
        return current_time >= reference_ &&
               (current_time.nanos() - reference_.nanos()) % interval_.nanos() == 0;
    }

    [[nodiscard]] constexpr Time_Point next_ring_at(Time_Point current_time) const noexcept
    {
        if (current_time < reference_) {
            return reference_;
        }
        auto elapsed = current_time.nanos() - reference_.nanos();
        auto n = elapsed / interval_.nanos() + 1;
        return Time_Point{reference_.nanos() + n * interval_.nanos()};
    }

    [[nodiscard]] constexpr Duration interval() const noexcept
    {
        return interval_;
    }

    [[nodiscard]] constexpr Time_Point reference() const noexcept
    {
        return reference_;
    }

private:
    Duration   interval_;
    Time_Point reference_;
};

} // namespace tick
