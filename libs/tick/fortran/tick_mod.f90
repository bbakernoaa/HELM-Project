module tick_mod
  !! Fortran 2003+ iso_c_binding interfaces for the TICK C API.
  !! This module declares types, constants, and interface blocks only —
  !! no implementation logic.
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! ── Public derived type ──────────────────────────────────────────
  public :: tick_date_time

  ! ── Public error-code constants ──────────────────────────────────
  public :: TICK_OK, TICK_ERR_OVERFLOW, TICK_ERR_INVALID_ARG, &
            TICK_ERR_INVALID_CALENDAR, TICK_ERR_INVALID_DATE, &
            TICK_ERR_INTERNAL

  ! ── Public calendar constants ────────────────────────────────────
  public :: TICK_CAL_GREGORIAN, TICK_CAL_NOLEAP, TICK_CAL_360DAY

  ! ── Public procedure interfaces ──────────────────────────────────
  public :: tick_strerror
  public :: tick_time_point_create, tick_time_point_add_duration, &
            tick_time_point_sub_duration, tick_time_point_diff, &
            tick_time_point_compare
  public :: tick_duration_from_nanos, tick_duration_from_seconds, &
            tick_duration_from_minutes, tick_duration_from_hours, &
            tick_duration_from_days
  public :: tick_duration_add, tick_duration_sub, &
            tick_duration_mul, tick_duration_div
  public :: tick_to_date_time, tick_to_time_point, &
            tick_days_in_month, tick_days_in_year
  public :: tick_interval_alarm_is_ringing, tick_interval_alarm_next_ring, &
            tick_absolute_alarm_is_ringing
  public :: tick_compute_heartbeat, tick_compute_sync_period, &
            tick_is_phase_aligned
  public :: tick_is_on_boundary, tick_compute_window, tick_window_contains

  ! ── Error code constants ─────────────────────────────────────────
  integer(c_int32_t), parameter :: TICK_OK                  = 0
  integer(c_int32_t), parameter :: TICK_ERR_OVERFLOW        = 1
  integer(c_int32_t), parameter :: TICK_ERR_INVALID_ARG     = 2
  integer(c_int32_t), parameter :: TICK_ERR_INVALID_CALENDAR = 3
  integer(c_int32_t), parameter :: TICK_ERR_INVALID_DATE    = 4
  integer(c_int32_t), parameter :: TICK_ERR_INTERNAL        = 5

  ! ── Calendar enum constants ──────────────────────────────────────
  integer(c_int32_t), parameter :: TICK_CAL_GREGORIAN = 0
  integer(c_int32_t), parameter :: TICK_CAL_NOLEAP    = 1
  integer(c_int32_t), parameter :: TICK_CAL_360DAY    = 2

  ! ── Derived type: tick_date_time (matches C struct tick_date_time_t) ──
  type, bind(c) :: tick_date_time
    integer(c_int32_t) :: year
    integer(c_int32_t) :: month
    integer(c_int32_t) :: day
    integer(c_int32_t) :: hour
    integer(c_int32_t) :: minute
    integer(c_int32_t) :: second
    integer(c_int32_t) :: nanosecond
  end type tick_date_time


  ! ══════════════════════════════════════════════════════════════════
  ! Interface blocks for C API functions
  ! ══════════════════════════════════════════════════════════════════

  ! ── Error Reporting ──────────────────────────────────────────────
  interface
    function tick_strerror(status) result(ptr) bind(c, name="tick_strerror")
      import :: c_int32_t, c_ptr
      integer(c_int32_t), value, intent(in) :: status
      type(c_ptr) :: ptr
    end function tick_strerror
  end interface

  ! ── Time_Point Functions ─────────────────────────────────────────
  interface
    function tick_time_point_create(nanos, out) result(status) &
        bind(c, name="tick_time_point_create")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: nanos
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_time_point_create
  end interface

  interface
    function tick_time_point_add_duration(tp, dur, out) result(status) &
        bind(c, name="tick_time_point_add_duration")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: tp
      integer(c_int64_t), value, intent(in) :: dur
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_time_point_add_duration
  end interface

  interface
    function tick_time_point_sub_duration(tp, dur, out) result(status) &
        bind(c, name="tick_time_point_sub_duration")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: tp
      integer(c_int64_t), value, intent(in) :: dur
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_time_point_sub_duration
  end interface

  interface
    function tick_time_point_diff(lhs, rhs, out) result(status) &
        bind(c, name="tick_time_point_diff")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: lhs
      integer(c_int64_t), value, intent(in) :: rhs
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_time_point_diff
  end interface

  interface
    function tick_time_point_compare(lhs, rhs, out) result(status) &
        bind(c, name="tick_time_point_compare")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: lhs
      integer(c_int64_t), value, intent(in) :: rhs
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_time_point_compare
  end interface

  ! ── Duration Functions ───────────────────────────────────────────
  interface
    function tick_duration_from_nanos(nanos, out) result(status) &
        bind(c, name="tick_duration_from_nanos")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: nanos
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_from_nanos
  end interface

  interface
    function tick_duration_from_seconds(count, out) result(status) &
        bind(c, name="tick_duration_from_seconds")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_from_seconds
  end interface

  interface
    function tick_duration_from_minutes(count, out) result(status) &
        bind(c, name="tick_duration_from_minutes")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_from_minutes
  end interface

  interface
    function tick_duration_from_hours(count, out) result(status) &
        bind(c, name="tick_duration_from_hours")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_from_hours
  end interface

  interface
    function tick_duration_from_days(count, out) result(status) &
        bind(c, name="tick_duration_from_days")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_from_days
  end interface

  interface
    function tick_duration_add(lhs, rhs, out) result(status) &
        bind(c, name="tick_duration_add")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: lhs
      integer(c_int64_t), value, intent(in) :: rhs
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_add
  end interface

  interface
    function tick_duration_sub(lhs, rhs, out) result(status) &
        bind(c, name="tick_duration_sub")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: lhs
      integer(c_int64_t), value, intent(in) :: rhs
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_sub
  end interface

  interface
    function tick_duration_mul(dur, scalar, out) result(status) &
        bind(c, name="tick_duration_mul")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: dur
      integer(c_int64_t), value, intent(in) :: scalar
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_mul
  end interface

  interface
    function tick_duration_div(dur, scalar, out) result(status) &
        bind(c, name="tick_duration_div")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: dur
      integer(c_int64_t), value, intent(in) :: scalar
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_duration_div
  end interface


  ! ── Calendar Conversions ─────────────────────────────────────────
  interface
    function tick_to_date_time(tp, cal, out) result(status) &
        bind(c, name="tick_to_date_time")
      import :: c_int64_t, c_int32_t, tick_date_time
      integer(c_int64_t), value, intent(in) :: tp
      integer(c_int32_t), value, intent(in) :: cal
      type(tick_date_time), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_to_date_time
  end interface

  interface
    function tick_to_time_point(dt, cal, out) result(status) &
        bind(c, name="tick_to_time_point")
      import :: c_int64_t, c_int32_t, tick_date_time
      type(tick_date_time), intent(in) :: dt
      integer(c_int32_t), value, intent(in) :: cal
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_to_time_point
  end interface

  interface
    function tick_days_in_month(cal, year, month, out) result(status) &
        bind(c, name="tick_days_in_month")
      import :: c_int32_t
      integer(c_int32_t), value, intent(in) :: cal
      integer(c_int32_t), value, intent(in) :: year
      integer(c_int32_t), value, intent(in) :: month
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_days_in_month
  end interface

  interface
    function tick_days_in_year(cal, year, out) result(status) &
        bind(c, name="tick_days_in_year")
      import :: c_int32_t
      integer(c_int32_t), value, intent(in) :: cal
      integer(c_int32_t), value, intent(in) :: year
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_days_in_year
  end interface

  ! ── Alarm Queries ────────────────────────────────────────────────
  interface
    function tick_interval_alarm_is_ringing(interval, reference, current, out) &
        result(status) bind(c, name="tick_interval_alarm_is_ringing")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: interval
      integer(c_int64_t), value, intent(in) :: reference
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_interval_alarm_is_ringing
  end interface

  interface
    function tick_interval_alarm_next_ring(interval, reference, current, out) &
        result(status) bind(c, name="tick_interval_alarm_next_ring")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: interval
      integer(c_int64_t), value, intent(in) :: reference
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_interval_alarm_next_ring
  end interface

  interface
    function tick_absolute_alarm_is_ringing(trigger, current, out) &
        result(status) bind(c, name="tick_absolute_alarm_is_ringing")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: trigger
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_absolute_alarm_is_ringing
  end interface

  ! ── Synchronization ──────────────────────────────────────────────
  interface
    function tick_compute_heartbeat(timesteps, count, out) result(status) &
        bind(c, name="tick_compute_heartbeat")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), intent(in) :: timesteps(*)
      integer(c_int32_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_compute_heartbeat
  end interface

  interface
    function tick_compute_sync_period(timesteps, count, out) result(status) &
        bind(c, name="tick_compute_sync_period")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), intent(in) :: timesteps(*)
      integer(c_int32_t), value, intent(in) :: count
      integer(c_int64_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_compute_sync_period
  end interface

  interface
    function tick_is_phase_aligned(current, base, timestep, out) &
        result(status) bind(c, name="tick_is_phase_aligned")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int64_t), value, intent(in) :: base
      integer(c_int64_t), value, intent(in) :: timestep
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_is_phase_aligned
  end interface

  ! ── Accumulation Windows ─────────────────────────────────────────
  interface
    function tick_is_on_boundary(current, interval, out) result(status) &
        bind(c, name="tick_is_on_boundary")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int64_t), value, intent(in) :: interval
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_is_on_boundary
  end interface

  interface
    function tick_compute_window(current, interval, out_start, out_end) &
        result(status) bind(c, name="tick_compute_window")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: current
      integer(c_int64_t), value, intent(in) :: interval
      integer(c_int64_t), intent(out) :: out_start
      integer(c_int64_t), intent(out) :: out_end
      integer(c_int32_t) :: status
    end function tick_compute_window
  end interface

  interface
    function tick_window_contains(win_start, win_end, query, out) &
        result(status) bind(c, name="tick_window_contains")
      import :: c_int64_t, c_int32_t
      integer(c_int64_t), value, intent(in) :: win_start
      integer(c_int64_t), value, intent(in) :: win_end
      integer(c_int64_t), value, intent(in) :: query
      integer(c_int32_t), intent(out) :: out
      integer(c_int32_t) :: status
    end function tick_window_contains
  end interface

end module tick_mod
