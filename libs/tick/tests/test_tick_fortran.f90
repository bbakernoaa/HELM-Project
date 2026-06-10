program test_tick_fortran
  !! Integration test for tick_mod — exercises every public procedure.
  !! On any failure prints an error message and calls stop 1.
  !! On success prints "All Fortran tests passed." and exits normally.
  use tick_mod
  use, intrinsic :: iso_c_binding
  implicit none

  ! Variables
  integer(c_int32_t) :: rc, cmp, ringing, days, boundary, contains_val
  integer(c_int64_t) :: tp, tp2, dur, dur2, heartbeat, sync_period
  integer(c_int64_t) :: win_start, win_end, next_ring
  type(tick_date_time) :: dt, dt_out
  integer(c_int64_t) :: timesteps(2)

  integer(c_int64_t), parameter :: NS_PER_SECOND = 1000000000_c_int64_t
  integer(c_int64_t), parameter :: NS_PER_DAY = 86400000000000_c_int64_t

  ! ─── Time_Point tests ───────────────────────────────────────────
  rc = tick_time_point_create(0_c_int64_t, tp)
  call check(rc == TICK_OK, "tick_time_point_create(0)")
  call check(tp == 0, "tp == 0")

  rc = tick_time_point_add_duration(0_c_int64_t, NS_PER_SECOND, tp)
  call check(rc == TICK_OK, "tick_time_point_add_duration")
  call check(tp == NS_PER_SECOND, "tp == 1s")

  rc = tick_time_point_sub_duration(NS_PER_DAY, NS_PER_SECOND, tp)
  call check(rc == TICK_OK, "tick_time_point_sub_duration")
  call check(tp == NS_PER_DAY - NS_PER_SECOND, "tp == 1day - 1s")

  rc = tick_time_point_diff(10_c_int64_t * NS_PER_SECOND, 3_c_int64_t * NS_PER_SECOND, dur)
  call check(rc == TICK_OK, "tick_time_point_diff")
  call check(dur == 7_c_int64_t * NS_PER_SECOND, "diff == 7s")

  rc = tick_time_point_compare(100_c_int64_t, 200_c_int64_t, cmp)
  call check(rc == TICK_OK, "tick_time_point_compare")
  call check(cmp < 0, "100 < 200")

  ! ─── Duration tests ─────────────────────────────────────────────
  rc = tick_duration_from_nanos(12345_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == 12345, "duration_from_nanos")

  rc = tick_duration_from_seconds(1_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == NS_PER_SECOND, "duration_from_seconds")

  rc = tick_duration_from_minutes(1_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == 60_c_int64_t * NS_PER_SECOND, "duration_from_minutes")

  rc = tick_duration_from_hours(1_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == 3600_c_int64_t * NS_PER_SECOND, "duration_from_hours")

  rc = tick_duration_from_days(1_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == NS_PER_DAY, "duration_from_days")

  rc = tick_duration_add(NS_PER_SECOND, 2_c_int64_t * NS_PER_SECOND, dur)
  call check(rc == TICK_OK .and. dur == 3_c_int64_t * NS_PER_SECOND, "duration_add")

  rc = tick_duration_sub(5_c_int64_t * NS_PER_SECOND, 2_c_int64_t * NS_PER_SECOND, dur)
  call check(rc == TICK_OK .and. dur == 3_c_int64_t * NS_PER_SECOND, "duration_sub")

  rc = tick_duration_mul(NS_PER_SECOND, 10_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == 10_c_int64_t * NS_PER_SECOND, "duration_mul")

  rc = tick_duration_div(10_c_int64_t * NS_PER_SECOND, 5_c_int64_t, dur)
  call check(rc == TICK_OK .and. dur == 2_c_int64_t * NS_PER_SECOND, "duration_div")

  ! Division by zero
  rc = tick_duration_div(NS_PER_SECOND, 0_c_int64_t, dur)
  call check(rc == TICK_ERR_INVALID_ARG, "duration_div_zero")

  ! ─── Calendar round-trip (Gregorian) ────────────────────────────
  rc = tick_to_date_time(0_c_int64_t, TICK_CAL_GREGORIAN, dt_out)
  call check(rc == TICK_OK, "to_date_time gregorian epoch")
  call check(dt_out%year == 2026 .and. dt_out%month == 1 .and. dt_out%day == 1, &
             "gregorian epoch values")

  rc = tick_to_time_point(dt_out, TICK_CAL_GREGORIAN, tp)
  call check(rc == TICK_OK .and. tp == 0, "gregorian round-trip")

  ! Calendar round-trip (NoLeap)
  rc = tick_to_date_time(0_c_int64_t, TICK_CAL_NOLEAP, dt_out)
  call check(rc == TICK_OK, "to_date_time noleap epoch")
  rc = tick_to_time_point(dt_out, TICK_CAL_NOLEAP, tp)
  call check(rc == TICK_OK .and. tp == 0, "noleap round-trip")

  ! Calendar round-trip (Cal360)
  rc = tick_to_date_time(0_c_int64_t, TICK_CAL_360DAY, dt_out)
  call check(rc == TICK_OK, "to_date_time cal360 epoch")
  rc = tick_to_time_point(dt_out, TICK_CAL_360DAY, tp)
  call check(rc == TICK_OK .and. tp == 0, "cal360 round-trip")

  ! Invalid calendar
  rc = tick_to_date_time(0_c_int64_t, 99_c_int32_t, dt_out)
  call check(rc == TICK_ERR_INVALID_CALENDAR, "invalid calendar")

  ! days_in_month, days_in_year
  rc = tick_days_in_month(TICK_CAL_GREGORIAN, 2026_c_int32_t, 1_c_int32_t, days)
  call check(rc == TICK_OK .and. days == 31, "days_in_month Jan")

  rc = tick_days_in_year(TICK_CAL_360DAY, 2026_c_int32_t, days)
  call check(rc == TICK_OK .and. days == 360, "days_in_year cal360")

  ! ─── Alarm tests ────────────────────────────────────────────────
  rc = tick_interval_alarm_is_ringing(10_c_int64_t * NS_PER_SECOND, 0_c_int64_t, &
                                      0_c_int64_t, ringing)
  call check(rc == TICK_OK .and. ringing == 1, "alarm ringing at reference")

  rc = tick_interval_alarm_is_ringing(10_c_int64_t * NS_PER_SECOND, 0_c_int64_t, &
                                      5_c_int64_t * NS_PER_SECOND, ringing)
  call check(rc == TICK_OK .and. ringing == 0, "alarm not ringing at 5s")

  rc = tick_interval_alarm_next_ring(10_c_int64_t * NS_PER_SECOND, 0_c_int64_t, &
                                     5_c_int64_t * NS_PER_SECOND, next_ring)
  call check(rc == TICK_OK .and. next_ring == 10_c_int64_t * NS_PER_SECOND, &
             "alarm next ring")

  rc = tick_absolute_alarm_is_ringing(100_c_int64_t, 100_c_int64_t, ringing)
  call check(rc == TICK_OK .and. ringing == 1, "absolute alarm ringing")

  ! ─── Sync tests ─────────────────────────────────────────────────
  timesteps(1) = 10_c_int64_t * NS_PER_SECOND
  timesteps(2) = 6_c_int64_t * NS_PER_SECOND

  rc = tick_compute_heartbeat(timesteps, 2_c_int32_t, heartbeat)
  call check(rc == TICK_OK .and. heartbeat == 2_c_int64_t * NS_PER_SECOND, "heartbeat")

  rc = tick_compute_sync_period(timesteps, 2_c_int32_t, sync_period)
  call check(rc == TICK_OK .and. sync_period == 30_c_int64_t * NS_PER_SECOND, "sync_period")

  rc = tick_is_phase_aligned(20_c_int64_t * NS_PER_SECOND, 0_c_int64_t, &
                             10_c_int64_t * NS_PER_SECOND, ringing)
  call check(rc == TICK_OK .and. ringing == 1, "phase_aligned")

  ! ─── Window tests ───────────────────────────────────────────────
  rc = tick_is_on_boundary(0_c_int64_t, 10_c_int64_t * NS_PER_SECOND, boundary)
  call check(rc == TICK_OK .and. boundary == 1, "on_boundary")

  rc = tick_compute_window(15_c_int64_t * NS_PER_SECOND, 10_c_int64_t * NS_PER_SECOND, &
                           win_start, win_end)
  call check(rc == TICK_OK, "compute_window")
  call check(win_start == 10_c_int64_t * NS_PER_SECOND, "window start")
  call check(win_end == 20_c_int64_t * NS_PER_SECOND, "window end")

  rc = tick_window_contains(10_c_int64_t * NS_PER_SECOND, 20_c_int64_t * NS_PER_SECOND, &
                            15_c_int64_t * NS_PER_SECOND, contains_val)
  call check(rc == TICK_OK .and. contains_val == 1, "window_contains")

  write(*,*) "All Fortran tests passed."

contains
  subroutine check(condition, msg)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: msg
    if (.not. condition) then
      write(*,*) "FAIL: ", msg
      stop 1
    end if
  end subroutine check

end program test_tick_fortran
