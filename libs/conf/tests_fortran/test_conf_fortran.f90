!> @file test_conf_fortran.f90
!! @brief End-to-end Fortran integration test for the CONF public API.
!!
!! Exercises the idiomatic Fortran interface exposed by `conf_mod`
!! (fortran/conf_mod.f90) against a known fixture YAML. Drives the full
!! resource lifecycle: load, query typed scalars (integer, real, string via
!! the allocatable wrapper, logical), check key existence, close, and verify
!! that a closed handle returns CONF_BAD_HANDLE.
!!
!! Single-process test — no MPI required. Exits 0 on success, 1 on any
!! failure, detected by CTest.
!!
!! Covers requirements: 34.1, 34.2, 34.3
program test_conf_fortran
  use conf_mod
  use, intrinsic :: iso_c_binding
  implicit none

  integer(c_int)                  :: handle, status
  integer(c_int)                  :: int_val
  real(c_double)                  :: real_val
  character(len=:), allocatable   :: str_val
  logical                         :: exists
  integer                         :: local_fail

  local_fail = 0

  ! ── Load the fixture YAML ──────────────────────────────────────────────────
  call conf_load("fixture.yaml", handle, status)
  call expect_eq('conf_load', status, CONF_SUCCESS)

  ! ── Query integer: model.layers == 42 ──────────────────────────────────────
  call conf_get_int(handle, "model.layers", int_val, status)
  call expect_eq('conf_get_int status', status, CONF_SUCCESS)
  if (int_val /= 42) then
    local_fail = local_fail + 1
    write(*,'(A,I0)') '[test_conf_fortran] FAIL conf_get_int value: got ', int_val
  end if

  ! ── Query real: model.rate ≈ 3.14 ──────────────────────────────────────────
  call conf_get_real(handle, "model.rate", real_val, status)
  call expect_eq('conf_get_real status', status, CONF_SUCCESS)
  if (abs(real_val - 3.14_c_double) > 0.001_c_double) then
    local_fail = local_fail + 1
    write(*,'(A,F0.6)') '[test_conf_fortran] FAIL conf_get_real value: got ', real_val
  end if

  ! ── Query string: model.name == "test_model" (allocatable wrapper) ─────────
  call conf_get_string(handle, "model.name", str_val, status)
  call expect_eq('conf_get_string status', status, CONF_SUCCESS)
  if (str_val /= "test_model") then
    local_fail = local_fail + 1
    write(*,'(A,A,A)') '[test_conf_fortran] FAIL conf_get_string value: got "', str_val, '"'
  end if
  ! The allocatable wrapper should set len(str_val) to exactly the byte count.
  if (len(str_val) /= 10) then
    local_fail = local_fail + 1
    write(*,'(A,I0)') '[test_conf_fortran] FAIL string length: got ', len(str_val)
  end if

  ! ── Query has_key: "model.layers" exists ───────────────────────────────────
  call conf_has_key(handle, "model.layers", exists, status)
  call expect_eq('conf_has_key status', status, CONF_SUCCESS)
  if (.not. exists) then
    local_fail = local_fail + 1
    write(*,'(A)') '[test_conf_fortran] FAIL conf_has_key: expected .true.'
  end if

  ! ── Query has_key: "model.nonexistent" does NOT exist ──────────────────────
  call conf_has_key(handle, "model.nonexistent", exists, status)
  call expect_eq('conf_has_key (missing) status', status, CONF_SUCCESS)
  if (exists) then
    local_fail = local_fail + 1
    write(*,'(A)') '[test_conf_fortran] FAIL conf_has_key (missing): expected .false.'
  end if

  ! ── Close the handle ───────────────────────────────────────────────────────
  call conf_close(handle, status)
  call expect_eq('conf_close', status, CONF_SUCCESS)

  ! ── After close, a getter should return CONF_BAD_HANDLE ────────────────────
  call conf_get_int(handle, "model.layers", int_val, status)
  call expect_eq('conf_get_int after close', status, CONF_BAD_HANDLE)

  ! ── Report ─────────────────────────────────────────────────────────────────
  if (local_fail == 0) then
    write(*,'(A)') '[test_conf_fortran] ALL CONF FORTRAN TESTS PASSED'
  else
    write(*,'(A,I0,A)') '[test_conf_fortran] FAIL: ', local_fail, ' check(s) failed.'
    error stop 1
  end if

contains

  !> Assert that an actual integer return code equals the expected value,
  !! recording a failure (with a diagnostic) otherwise.
  subroutine expect_eq(step, actual, expected_code)
    character(len=*), intent(in) :: step
    integer(c_int), intent(in)   :: actual
    integer(c_int), intent(in)   :: expected_code
    if (actual /= expected_code) then
      local_fail = local_fail + 1
      write(*,'(A,A,A,I0,A,I0)') &
        '[test_conf_fortran] FAIL ', trim(step), &
        ': ierr=', actual, ' expected=', expected_code
    end if
  end subroutine expect_eq

end program test_conf_fortran
