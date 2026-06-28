/* tick_c.h — C99/C++20 public API for TICK */
#ifndef TICK_C_H
#define TICK_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Types ─────────────────────────────────────────────────────── */

typedef int64_t tick_time_point_t; /* nanoseconds since TICK epoch */
typedef int64_t tick_duration_t;   /* nanoseconds */
typedef int32_t tick_status_t;     /* return code */

typedef struct tick_date_time_t {
    int32_t year;
    int32_t month;
    int32_t day;
    int32_t hour;
    int32_t minute;
    int32_t second;
    int32_t nanosecond;
} tick_date_time_t;

typedef enum tick_calendar_t { TICK_CAL_GREGORIAN = 0, TICK_CAL_NOLEAP = 1, TICK_CAL_360DAY = 2 } tick_calendar_t;

/* ── Error Codes ───────────────────────────────────────────────── */

#define TICK_OK 0
#define TICK_ERR_OVERFLOW 1
#define TICK_ERR_INVALID_ARG 2
#define TICK_ERR_INVALID_CALENDAR 3
#define TICK_ERR_INVALID_DATE 4
#define TICK_ERR_INTERNAL 5

/* ── Error Reporting ───────────────────────────────────────────── */

const char *tick_strerror(tick_status_t status);

/* ── Time_Point Functions ──────────────────────────────────────── */

tick_status_t tick_time_point_create(int64_t nanos, tick_time_point_t *out);
tick_status_t tick_time_point_add_duration(tick_time_point_t tp, tick_duration_t dur, tick_time_point_t *out);
tick_status_t tick_time_point_sub_duration(tick_time_point_t tp, tick_duration_t dur, tick_time_point_t *out);
tick_status_t tick_time_point_diff(tick_time_point_t lhs, tick_time_point_t rhs, tick_duration_t *out);
tick_status_t tick_time_point_compare(tick_time_point_t lhs, tick_time_point_t rhs, int32_t *out);

/* ── Duration Functions ────────────────────────────────────────── */

tick_status_t tick_duration_from_nanos(int64_t nanos, tick_duration_t *out);
tick_status_t tick_duration_from_seconds(int64_t count, tick_duration_t *out);
tick_status_t tick_duration_from_minutes(int64_t count, tick_duration_t *out);
tick_status_t tick_duration_from_hours(int64_t count, tick_duration_t *out);
tick_status_t tick_duration_from_days(int64_t count, tick_duration_t *out);
tick_status_t tick_duration_add(tick_duration_t lhs, tick_duration_t rhs, tick_duration_t *out);
tick_status_t tick_duration_sub(tick_duration_t lhs, tick_duration_t rhs, tick_duration_t *out);
tick_status_t tick_duration_mul(tick_duration_t dur, int64_t scalar, tick_duration_t *out);
tick_status_t tick_duration_div(tick_duration_t dur, int64_t scalar, tick_duration_t *out);

/* ── Calendar Conversions ──────────────────────────────────────── */

tick_status_t tick_to_date_time(tick_time_point_t tp, tick_calendar_t cal, tick_date_time_t *out);
tick_status_t tick_to_time_point(tick_date_time_t dt, tick_calendar_t cal, tick_time_point_t *out);
tick_status_t tick_days_in_month(tick_calendar_t cal, int32_t year, int32_t month, int32_t *out);
tick_status_t tick_days_in_year(tick_calendar_t cal, int32_t year, int32_t *out);

/* ── Alarm Queries ─────────────────────────────────────────────── */

tick_status_t tick_interval_alarm_is_ringing(tick_duration_t interval, tick_time_point_t reference, tick_time_point_t current, int32_t *out);
tick_status_t tick_interval_alarm_next_ring(tick_duration_t interval, tick_time_point_t reference, tick_time_point_t current, tick_time_point_t *out);
tick_status_t tick_absolute_alarm_is_ringing(tick_time_point_t trigger, tick_time_point_t current, int32_t *out);

/* ── Synchronization ───────────────────────────────────────────── */

tick_status_t tick_compute_heartbeat(const tick_duration_t *timesteps, int32_t count, tick_duration_t *out);
tick_status_t tick_compute_sync_period(const tick_duration_t *timesteps, int32_t count, tick_duration_t *out);
tick_status_t tick_is_phase_aligned(tick_time_point_t current, tick_time_point_t base, tick_duration_t timestep, int32_t *out);

/* ── Accumulation Windows ──────────────────────────────────────── */

tick_status_t tick_is_on_boundary(tick_time_point_t current, tick_duration_t interval, int32_t *out);
tick_status_t tick_compute_window(tick_time_point_t current, tick_duration_t interval, tick_time_point_t *out_start, tick_time_point_t *out_end);
tick_status_t tick_window_contains(tick_time_point_t win_start, tick_time_point_t win_end, tick_time_point_t query, int32_t *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TICK_C_H */
