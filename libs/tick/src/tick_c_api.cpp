// tick_c_api.cpp — C ABI implementation for the TICK micro-library
// This file implements the extern "C" functions declared in tick_c.h.

#include "tick/tick.hpp"
#include "tick/tick_c.h"
#include "tick/detail/overflow.hpp"

#include <cstdint>
#include <stdexcept>

// ── Exception Firewall Macro ──────────────────────────────────────────────────
// Wraps a block of C++ code in try/catch, mapping exceptions to tick_status_t.

#define TICK_C_TRY(body)                        \
    try {                                       \
        body                                    \
        return TICK_OK;                         \
    } catch (const std::overflow_error&) {      \
        return TICK_ERR_OVERFLOW;               \
    } catch (const std::invalid_argument&) {    \
        return TICK_ERR_INVALID_ARG;            \
    } catch (...) {                             \
        return TICK_ERR_INTERNAL;               \
    }

// ── Nanosecond conversion constants ──────────────────────────────────────────

static constexpr std::int64_t nanos_per_second = 1'000'000'000LL;
static constexpr std::int64_t nanos_per_minute = 60LL * nanos_per_second;
static constexpr std::int64_t nanos_per_hour   = 3'600LL * nanos_per_second;
static constexpr std::int64_t nanos_per_day    = 86'400LL * nanos_per_second;

// ── Error Reporting ───────────────────────────────────────────────────────────

extern "C" const char* tick_strerror(tick_status_t status)
{
    switch (status) {
        case TICK_OK:                   return "success";
        case TICK_ERR_OVERFLOW:         return "integer overflow";
        case TICK_ERR_INVALID_ARG:      return "invalid argument";
        case TICK_ERR_INVALID_CALENDAR: return "invalid calendar";
        case TICK_ERR_INVALID_DATE:     return "invalid date";
        case TICK_ERR_INTERNAL:         return "internal error";
        default:                        return "unknown error";
    }
}

// ── Time_Point Functions ──────────────────────────────────────────────────────

extern "C" tick_status_t tick_time_point_create(int64_t nanos, tick_time_point_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = tick::Time_Point{nanos}.nanos();
    )
}

extern "C" tick_status_t tick_time_point_add_duration(tick_time_point_t tp,
                                                      tick_duration_t dur,
                                                      tick_time_point_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Time_Point{tp} + tick::Duration{dur};
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_time_point_sub_duration(tick_time_point_t tp,
                                                      tick_duration_t dur,
                                                      tick_time_point_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Time_Point{tp} - tick::Duration{dur};
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_time_point_diff(tick_time_point_t lhs,
                                              tick_time_point_t rhs,
                                              tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Time_Point{lhs} - tick::Time_Point{rhs};
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_time_point_compare(tick_time_point_t lhs,
                                                 tick_time_point_t rhs,
                                                 int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto ordering = tick::Time_Point{lhs} <=> tick::Time_Point{rhs};
        if (ordering < 0)      *out = -1;
        else if (ordering > 0) *out = 1;
        else                   *out = 0;
    )
}

// ── Duration Factory Functions ────────────────────────────────────────────────

extern "C" tick_status_t tick_duration_from_nanos(int64_t nanos, tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = nanos;
    )
}

extern "C" tick_status_t tick_duration_from_seconds(int64_t count, tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = tick::detail::checked_mul(count, nanos_per_second);
    )
}

extern "C" tick_status_t tick_duration_from_minutes(int64_t count, tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = tick::detail::checked_mul(count, nanos_per_minute);
    )
}

extern "C" tick_status_t tick_duration_from_hours(int64_t count, tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = tick::detail::checked_mul(count, nanos_per_hour);
    )
}

extern "C" tick_status_t tick_duration_from_days(int64_t count, tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        *out = tick::detail::checked_mul(count, nanos_per_day);
    )
}

// ── Duration Arithmetic Functions ─────────────────────────────────────────────

extern "C" tick_status_t tick_duration_add(tick_duration_t lhs, tick_duration_t rhs,
                                           tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Duration{lhs} + tick::Duration{rhs};
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_duration_sub(tick_duration_t lhs, tick_duration_t rhs,
                                           tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Duration{lhs} - tick::Duration{rhs};
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_duration_mul(tick_duration_t dur, int64_t scalar,
                                           tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Duration{dur} * scalar;
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_duration_div(tick_duration_t dur, int64_t scalar,
                                           tick_duration_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto result = tick::Duration{dur} / scalar;
        *out = result.nanos();
    )
}

// ── Calendar Conversion Functions ─────────────────────────────────────────────

extern "C" tick_status_t tick_to_date_time(tick_time_point_t tp, tick_calendar_t cal,
                                           tick_date_time_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;

    switch (cal) {
        case TICK_CAL_GREGORIAN:
        case TICK_CAL_NOLEAP:
        case TICK_CAL_360DAY:
            break;
        default:
            return TICK_ERR_INVALID_CALENDAR;
    }

    TICK_C_TRY(
        tick::Date_Time result;
        switch (cal) {
            case TICK_CAL_GREGORIAN: result = tick::Gregorian_Calendar::to_date_time(tick::Time_Point{tp}); break;
            case TICK_CAL_NOLEAP:    result = tick::NoLeap_Calendar::to_date_time(tick::Time_Point{tp}); break;
            case TICK_CAL_360DAY:    result = tick::Cal360_Calendar::to_date_time(tick::Time_Point{tp}); break;
            default: break; // unreachable — already validated above
        }
        out->year       = result.year;
        out->month      = result.month;
        out->day        = result.day;
        out->hour       = result.hour;
        out->minute     = result.minute;
        out->second     = result.second;
        out->nanosecond = result.nanosecond;
    )
}

extern "C" tick_status_t tick_to_time_point(tick_date_time_t dt, tick_calendar_t cal,
                                            tick_time_point_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;

    tick::Date_Time cpp_dt{dt.year, dt.month, dt.day,
                           dt.hour, dt.minute, dt.second, dt.nanosecond};

    try {
        tick::Time_Point result;
        switch (cal) {
            case TICK_CAL_GREGORIAN: result = tick::Gregorian_Calendar::to_time_point(cpp_dt); break;
            case TICK_CAL_NOLEAP:    result = tick::NoLeap_Calendar::to_time_point(cpp_dt); break;
            case TICK_CAL_360DAY:    result = tick::Cal360_Calendar::to_time_point(cpp_dt); break;
            default: return TICK_ERR_INVALID_CALENDAR;
        }
        *out = result.nanos();
        return TICK_OK;
    } catch (const std::invalid_argument&) {
        return TICK_ERR_INVALID_DATE;
    } catch (const std::overflow_error&) {
        return TICK_ERR_OVERFLOW;
    } catch (...) {
        return TICK_ERR_INTERNAL;
    }
}

extern "C" tick_status_t tick_days_in_month(tick_calendar_t cal, int32_t year,
                                            int32_t month, int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;

    try {
        int32_t result;
        switch (cal) {
            case TICK_CAL_GREGORIAN: result = tick::Gregorian_Calendar::days_in_month(year, month); break;
            case TICK_CAL_NOLEAP:    result = tick::NoLeap_Calendar::days_in_month(year, month); break;
            case TICK_CAL_360DAY:    result = tick::Cal360_Calendar::days_in_month(year, month); break;
            default: return TICK_ERR_INVALID_CALENDAR;
        }
        *out = result;
        return TICK_OK;
    } catch (const std::invalid_argument&) {
        return TICK_ERR_INVALID_DATE;
    } catch (const std::overflow_error&) {
        return TICK_ERR_OVERFLOW;
    } catch (...) {
        return TICK_ERR_INTERNAL;
    }
}

extern "C" tick_status_t tick_days_in_year(tick_calendar_t cal, int32_t year,
                                           int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;

    switch (cal) {
        case TICK_CAL_GREGORIAN:
        case TICK_CAL_NOLEAP:
        case TICK_CAL_360DAY:
            break;
        default:
            return TICK_ERR_INVALID_CALENDAR;
    }

    TICK_C_TRY(
        int32_t result;
        switch (cal) {
            case TICK_CAL_GREGORIAN: result = tick::Gregorian_Calendar::days_in_year(year); break;
            case TICK_CAL_NOLEAP:    result = tick::NoLeap_Calendar::days_in_year(year); break;
            case TICK_CAL_360DAY:    result = tick::Cal360_Calendar::days_in_year(year); break;
            default: result = 0; break; // unreachable
        }
        *out = result;
    )
}

// ── Alarm Query Functions ─────────────────────────────────────────────────────

extern "C" tick_status_t tick_interval_alarm_is_ringing(tick_duration_t interval,
                                                        tick_time_point_t reference,
                                                        tick_time_point_t current,
                                                        int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto dur = tick::Duration{interval};
        auto ref = tick::Time_Point{reference};
        tick::Interval_Alarm alarm(dur, ref);
        *out = alarm.is_ringing(tick::Time_Point{current}) ? 1 : 0;
    )
}

extern "C" tick_status_t tick_interval_alarm_next_ring(tick_duration_t interval,
                                                       tick_time_point_t reference,
                                                       tick_time_point_t current,
                                                       tick_time_point_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto dur = tick::Duration{interval};
        auto ref = tick::Time_Point{reference};
        tick::Interval_Alarm alarm(dur, ref);
        auto next = alarm.next_ring_at(tick::Time_Point{current});
        *out = next.nanos();
    )
}

extern "C" tick_status_t tick_absolute_alarm_is_ringing(tick_time_point_t trigger,
                                                        tick_time_point_t current,
                                                        int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        tick::Absolute_Alarm alarm{tick::Time_Point{trigger}};
        *out = alarm.is_ringing(tick::Time_Point{current}) ? 1 : 0;
    )
}

// ── Synchronization Functions ─────────────────────────────────────────────────

extern "C" tick_status_t tick_compute_heartbeat(const tick_duration_t* timesteps,
                                                int32_t count,
                                                tick_duration_t* out)
{
    if (!timesteps || count <= 0) return TICK_ERR_INVALID_ARG;
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto ptr = reinterpret_cast<const tick::Duration*>(timesteps);
        auto len = static_cast<std::size_t>(count);
        auto result = tick::compute_heartbeat({ptr, len});
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_compute_sync_period(const tick_duration_t* timesteps,
                                                  int32_t count,
                                                  tick_duration_t* out)
{
    if (!timesteps || count <= 0) return TICK_ERR_INVALID_ARG;
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto ptr = reinterpret_cast<const tick::Duration*>(timesteps);
        auto len = static_cast<std::size_t>(count);
        auto result = tick::compute_sync_period({ptr, len});
        *out = result.nanos();
    )
}

extern "C" tick_status_t tick_is_phase_aligned(tick_time_point_t current,
                                               tick_time_point_t base,
                                               tick_duration_t timestep,
                                               int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        bool aligned = tick::is_phase_aligned(
            tick::Time_Point{current},
            tick::Time_Point{base},
            tick::Duration{timestep});
        *out = aligned ? 1 : 0;
    )
}

// ── Accumulation Window Functions ─────────────────────────────────────────────

extern "C" tick_status_t tick_is_on_boundary(tick_time_point_t current,
                                             tick_duration_t interval,
                                             int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        bool on_boundary = tick::is_on_boundary(
            tick::Time_Point{current},
            tick::Duration{interval});
        *out = on_boundary ? 1 : 0;
    )
}

extern "C" tick_status_t tick_compute_window(tick_time_point_t current,
                                             tick_duration_t interval,
                                             tick_time_point_t* out_start,
                                             tick_time_point_t* out_end)
{
    if (!out_start || !out_end) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto window = tick::compute_window(
            tick::Time_Point{current},
            tick::Duration{interval});
        *out_start = window.start().nanos();
        *out_end = window.end().nanos();
    )
}

extern "C" tick_status_t tick_window_contains(tick_time_point_t win_start,
                                              tick_time_point_t win_end,
                                              tick_time_point_t query,
                                              int32_t* out)
{
    if (!out) return TICK_ERR_INVALID_ARG;
    TICK_C_TRY(
        auto start = tick::Time_Point{win_start};
        auto end   = tick::Time_Point{win_end};
        tick::Time_Window window(start, end);
        *out = window.contains(tick::Time_Point{query}) ? 1 : 0;
    )
}
