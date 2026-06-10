#pragma once

#include <concepts>
#include <cstdint>

#include "tick/time_point.hpp"
#include "tick/date_time.hpp"

namespace tick {

template <typename Cal>
concept Calendar = requires(Time_Point tp, Date_Time dt) {
    { Cal::to_date_time(tp) } -> std::same_as<Date_Time>;
    { Cal::to_time_point(dt) } -> std::same_as<Time_Point>;
    { Cal::days_in_month(std::int32_t{}, std::int32_t{}) } -> std::same_as<std::int32_t>;
    { Cal::days_in_year(std::int32_t{}) } -> std::same_as<std::int32_t>;
};

} // namespace tick
