!> @file test_conf_fortran.f90
!> @brief End-to-end Fortran integration test for the CONF micro-library.
!>
!> Drives CONF through the public conf_mod wrapper, exercising the full stack:
!>   Fortran -> iso_c_binding -> extern "C" bridge -> conf::Config -> yaml-cpp.
!>
!> To avoid any dependency on a Fortran preprocessor (a static fixture path
!> would require compiling this .f90 with -cpp to expand a CONF_FORTRAN_FIXTURE_DIR
!> macro), the test writes its own known YAML document to a scratch file at
!> runtime, loads it, runs the queries, and deletes the scratch file. The fixture
!> is removed as soon as conf_load returns, because conf::Config parses the file
!> into an owned in-memory tree at load time and never re-reads the path.
!>
!> Coverage (Requirement 34):
!>   * 34.1 — load a fixture, query an integer, a real, and a string; assert each
!>            status == CONF_SUCCESS and each value equals the fixture value.
!>   * 34.2 — the string query uses conf_get_string's allocatable wrapper so the
!>            returned length equals the value's byte count and the contents match,
!>            with no manual allocate/free by the test.
!>   * 34.3 — close the handle through the wrapper and assert CONF_SUCCESS.
!>
!> On any failed check the program prints which assertion failed and terminates
!> with a non-zero exit status (error stop) so CTest records a failure.
!>
!> Requirements: 34.1, 34.2, 34.3
program test_conf_fortran
    use conf_mod
    use, intrinsic :: iso_c_binding
    implicit none

    character(len=*), parameter :: fixture_path = 'conf_fortran_fixture.yaml'

    integer(c_int)                :: handle
    integer(c_int)                :: status
    integer(c_int)                :: ivalue
    real(c_double)                :: rvalue
    character(len=:), allocatable :: svalue

    ! ── Build a known YAML document on disk ──────────────────────────────────
    ! Nested maps so the dotted-path resolver (split on '.') reaches each leaf:
    !   model.layers -> 42        (integer)
    !   grid.spacing -> 0.25      (real / double)
    !   model.scheme -> spherical (string, 9 bytes)
    call write_fixture(fixture_path)

    ! ── 1. Load the configuration ────────────────────────────────────────────
    call conf_load(fixture_path, handle, status)
    ! The parsed tree is now owned in memory; the scratch file is no longer
    ! needed. Delete it immediately so it is cleaned up even if a later check
    ! fails and aborts the program.
    call delete_fixture(fixture_path)
    call check(status == CONF_SUCCESS, 'conf_load status == CONF_SUCCESS')

    ! ── 2. Query an integer ──────────────────────────────────────────────────
    call conf_get_int(handle, 'model.layers', ivalue, status)
    call check(status == CONF_SUCCESS, 'conf_get_int status == CONF_SUCCESS')
    call check(ivalue == 42_c_int, 'conf_get_int value == 42')

    ! ── 3. Query a real (double precision) ───────────────────────────────────
    call conf_get_real(handle, 'grid.spacing', rvalue, status)
    call check(status == CONF_SUCCESS, 'conf_get_real status == CONF_SUCCESS')
    call check(abs(rvalue - 0.25d0) < 1.0d-12, 'conf_get_real value == 0.25')

    ! ── 4. Query a string (allocatable length-first wrapper) ─────────────────
    ! conf_get_string performs the length-first protocol internally and returns
    ! a deferred-length allocatable string sized to exactly the value's byte
    ! count — the test does no manual allocate/free.
    call conf_get_string(handle, 'model.scheme', svalue, status)
    call check(status == CONF_SUCCESS, 'conf_get_string status == CONF_SUCCESS')
    call check(allocated(svalue), 'conf_get_string allocated result')
    call check(len(svalue) == 9, 'conf_get_string len == 9 (byte count)')
    call check(svalue == 'spherical', 'conf_get_string value == "spherical"')

    ! ── 5. Close the handle through the wrapper ──────────────────────────────
    call conf_close(handle, status)
    call check(status == CONF_SUCCESS, 'conf_close status == CONF_SUCCESS')

    print '(A)', 'PASS: all CONF Fortran integration checks succeeded'

contains

    !> Write the known fixture YAML document to `path`, overwriting any existing
    !> file. Aborts with a non-zero status if the file cannot be opened.
    subroutine write_fixture(path)
        character(len=*), intent(in) :: path
        integer :: unit
        integer :: ios

        open(newunit=unit, file=path, status='replace', action='write', &
             form='formatted', iostat=ios)
        call check(ios == 0, 'open scratch fixture for writing')

        write(unit, '(A)') 'model:'
        write(unit, '(A)') '  layers: 42'
        write(unit, '(A)') '  scheme: spherical'
        write(unit, '(A)') 'grid:'
        write(unit, '(A)') '  spacing: 0.25'

        close(unit)
    end subroutine write_fixture

    !> Delete the scratch fixture if present. Silent no-op when it is absent so
    !> cleanup is safe to call unconditionally.
    subroutine delete_fixture(path)
        character(len=*), intent(in) :: path
        integer :: unit
        integer :: ios

        open(newunit=unit, file=path, status='old', action='read', iostat=ios)
        if (ios == 0) close(unit, status='delete')
    end subroutine delete_fixture

    !> Assert `cond`; on failure print the labelled message and terminate the
    !> program with a non-zero exit status so CTest records the failure.
    subroutine check(cond, label)
        logical,          intent(in) :: cond
        character(len=*), intent(in) :: label

        if (.not. cond) then
            print '(A,A)', 'FAIL: ', label
            error stop 1
        end if
    end subroutine check

end program test_conf_fortran
