#pragma once

#include "tick/time_point.hpp"

namespace tick {

class Absolute_Alarm {
   public:
    constexpr explicit Absolute_Alarm(Time_Point trigger_time) noexcept : trigger_time_{trigger_time} {}

    [[nodiscard]] constexpr bool is_ringing(Time_Point current_time) const noexcept {
        return current_time == trigger_time_;
    }

    [[nodiscard]] constexpr bool has_passed(Time_Point current_time) const noexcept {
        return current_time > trigger_time_;
    }

    [[nodiscard]] constexpr Time_Point trigger_time() const noexcept {
        return trigger_time_;
    }

   private:
    Time_Point trigger_time_;
};

}  // namespace tick
