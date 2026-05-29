!> @file test_halo_mod.f90
!! @brief End-to-end Fortran integration test for the HALO public API.
!!
!! Exercises the idiomatic Fortran interface exposed by `halo_mod`
!! (fortran/halo_mod.f90) against a REAL 4-rank MPI run (mpirun -np 4), as
!! opposed to the single-rank mocked-MPI property tests on the C++ side. The
!! program drives the full resource lifecycle, the async completion path, error
!! code propagation, MPI integer-communicator compatibility, and contiguous
!! real(c_double) array passing through c_loc.
!!
!! Topology: a periodic ring. Each rank sends DCOUNT doubles to its right
!! neighbor (rank+1 mod size) and receives DCOUNT doubles from its left
!! neighbor (rank-1+size mod size). The exchange treats the field as a flat
!! byte buffer (the C interop builds a char-typed Kokkos view), so the per-
!! neighbor plan COUNT is expressed in BYTES = DCOUNT * 8. A single contiguous
!! field buffer holds the send region [1..DCOUNT] followed by the receive
!! region [DCOUNT+1..2*DCOUNT].
!!
!! Failure handling: every step checks its ierr against HALO_SUCCESS (or the
!! specific expected error code for the negative tests). A per-rank failure
!! counter is reduced across all ranks with MPI_Allreduce; the program performs
!! a normal `stop 0` only when EVERY rank passed, and `stop 1` (nonzero exit,
!! detected by ctest) if any check on any rank failed.
!!
!! Covers requirements: 14.2, 14.5, 14.6, 14.7, 14.8, 14.11, 14.12, 14.14
program test_halo_mod
  use, intrinsic :: iso_c_binding, only : c_double
  use mpi
  use halo_mod
  implicit none

  ! ── MPI state ──────────────────────────────────────────────────────────────
  integer :: mpi_ierr
  integer :: rank, nprocs

  ! ── HALO opaque handles ─────────────────────────────────────────────────────
  integer :: comm_handle
  integer :: plan_handle
  integer :: async_handle

  ! ── Return codes / scratch ──────────────────────────────────────────────────
  integer :: ierr
  integer :: ecode
  integer :: local_fail, global_fail
  logical :: is_complete

  ! ── Exchange geometry ───────────────────────────────────────────────────────
  ! DCOUNT doubles per neighbor. The plan COUNT is in BYTES because the C
  ! interop reinterprets the buffer as a flat char view (1 byte per element).
  integer, parameter :: DCOUNT     = 4            ! doubles exchanged per neighbor
  integer, parameter :: ELEM_BYTES = 8            ! sizeof(real(c_double))
  integer, parameter :: BYTE_COUNT = DCOUNT * ELEM_BYTES

  integer :: left, right
  integer :: send_ranks(1), send_counts(1)
  integer :: recv_ranks(1), recv_counts(1)

  ! Negative-test (invalid rank) neighbor arrays.
  integer :: bad_send_ranks(1)
  integer :: bad_plan_handle

  ! ── Field buffer: contiguous, target real(c_double) passed via c_loc ────────
  real(c_double), allocatable, target :: field(:)
  integer :: i
  real(c_double) :: expected
  real(c_double), parameter :: SENTINEL = -999.0_c_double
  real(c_double), parameter :: TOL      = 1.0e-9_c_double

  local_fail = 0

  ! ── MPI bring-up ────────────────────────────────────────────────────────────
  call MPI_Init(mpi_ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, mpi_ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, mpi_ierr)

  if (nprocs < 2) then
    if (rank == 0) then
      write(*,'(A)') '[test_halo_mod] ERROR: requires at least 2 ranks (run with -np 4).'
    end if
    call MPI_Finalize(mpi_ierr)
    stop 1
  end if

  ! Ring neighbors.
  right = mod(rank + 1, nprocs)
  left  = mod(rank - 1 + nprocs, nprocs)

  ! ── 14.11: MPI communicator integer compatibility ──────────────────────────
  ! Pass the plain integer MPI_COMM_WORLD (from the `use mpi` binding) straight
  ! into halo_init, mirroring the ESMF integer-handle convention.
  call halo_init(MPI_COMM_WORLD, comm_handle, ierr)
  call expect_eq('halo_init', ierr, HALO_SUCCESS)

  ! ── 14.12: plan creation from neighbor rank/count arrays ────────────────────
  send_ranks(1)  = right
  send_counts(1) = BYTE_COUNT
  recv_ranks(1)  = left
  recv_counts(1) = BYTE_COUNT

  call halo_plan_create(comm_handle, send_ranks, send_counts, &
                        recv_ranks, recv_counts, plan_handle, ierr)
  call expect_eq('halo_plan_create', ierr, HALO_SUCCESS)

  ! ── Allocate the contiguous field: send region || recv region ───────────────
  ! Bytes: [0, BYTE_COUNT) send, [BYTE_COUNT, 2*BYTE_COUNT) recv. Because
  ! BYTE_COUNT is a whole multiple of 8, the regions land on exact double
  ! boundaries: field(1:DCOUNT) = send, field(DCOUNT+1:2*DCOUNT) = recv.
  allocate(field(2 * DCOUNT))

  ! ── 14.2 / 14.5: blocking exchange over a contiguous real(c_double) array ───
  call fill_send_region()
  call clear_recv_region()

  call halo_exchange_blocking(plan_handle, field, ierr)
  call expect_eq('halo_exchange_blocking', ierr, HALO_SUCCESS)
  call verify_recv_region('blocking')

  ! ── 14.6: async exchange + wait over the same contiguous buffer ─────────────
  call fill_send_region()
  call clear_recv_region()

  call halo_exchange_async(plan_handle, field, async_handle, ierr)
  call expect_eq('halo_exchange_async', ierr, HALO_SUCCESS)

  ! halo_test must succeed regardless of whether the transfer has finished yet;
  ! is_complete is informational here (it may legitimately be .true. or .false.).
  call halo_test(async_handle, is_complete, ierr)
  call expect_eq('halo_test', ierr, HALO_SUCCESS)

  call halo_wait(async_handle, ierr)
  call expect_eq('halo_wait', ierr, HALO_SUCCESS)
  call verify_recv_region('async')

  ! ── 14.7 / 14.8: error code propagation ─────────────────────────────────────
  ! (a) Invalid neighbor rank (== comm size, out of range) => HALO_ERR_INVALID_ARG.
  bad_send_ranks(1) = nprocs          ! valid range is [0, nprocs)
  bad_plan_handle   = -1
  call halo_plan_create(comm_handle, bad_send_ranks, send_counts, &
                        recv_ranks, recv_counts, bad_plan_handle, ecode)
  call expect_eq('invalid-rank plan_create', ecode, HALO_ERR_INVALID_ARG)

  ! (b) Bogus handle on halo_wait => HALO_ERR_BAD_HANDLE.
  call halo_wait(999999, ecode)
  call expect_eq('bad-handle halo_wait', ecode, HALO_ERR_BAD_HANDLE)

  ! (c) Bogus handle on halo_exchange_blocking => HALO_ERR_BAD_HANDLE
  !     (plan lookup fails before the buffer is ever touched).
  call halo_exchange_blocking(999999, field, ecode)
  call expect_eq('bad-handle halo_exchange_blocking', ecode, HALO_ERR_BAD_HANDLE)

  ! (d) Bogus handle on halo_destroy_plan => HALO_ERR_BAD_HANDLE.
  call halo_destroy_plan(999999, ecode)
  call expect_eq('bad-handle halo_destroy_plan', ecode, HALO_ERR_BAD_HANDLE)

  ! ── Lifecycle teardown ──────────────────────────────────────────────────────
  call halo_destroy_plan(plan_handle, ierr)
  call expect_eq('halo_destroy_plan', ierr, HALO_SUCCESS)

  call halo_destroy_comm(comm_handle, ierr)
  call expect_eq('halo_destroy_comm', ierr, HALO_SUCCESS)

  if (allocated(field)) deallocate(field)

  ! ── Aggregate pass/fail across all ranks ────────────────────────────────────
  call MPI_Allreduce(local_fail, global_fail, 1, MPI_INTEGER, MPI_SUM, &
                     MPI_COMM_WORLD, mpi_ierr)

  if (rank == 0) then
    if (global_fail == 0) then
      write(*,'(A,I0,A)') '[test_halo_mod] PASS across ', nprocs, ' ranks.'
    else
      write(*,'(A,I0,A)') '[test_halo_mod] FAIL: ', global_fail, &
                          ' failed check(s) across all ranks.'
    end if
  end if

  call MPI_Finalize(mpi_ierr)

  ! Nonzero exit on any failure so ctest detects it; clean exit otherwise.
  if (global_fail /= 0) then
    stop 1
  end if
  stop 0

contains

  !> Populate the send region field(1:DCOUNT) with rank-encoded values so the
  !! receiver can verify exactly which neighbor's bytes arrived.
  subroutine fill_send_region()
    integer :: k
    do k = 1, DCOUNT
      field(k) = real(rank * 100 + k, c_double)
    end do
  end subroutine fill_send_region

  !> Reset the receive region field(DCOUNT+1:2*DCOUNT) to a sentinel so a
  !! missing/short transfer is detectable.
  subroutine clear_recv_region()
    integer :: k
    do k = 1, DCOUNT
      field(DCOUNT + k) = SENTINEL
    end do
  end subroutine clear_recv_region

  !> Verify the receive region holds the LEFT neighbor's send-region values,
  !! confirming a faithful byte-for-byte ring exchange.
  subroutine verify_recv_region(label)
    character(len=*), intent(in) :: label
    integer :: k
    do k = 1, DCOUNT
      expected = real(left * 100 + k, c_double)
      if (abs(field(DCOUNT + k) - expected) > TOL) then
        local_fail = local_fail + 1
        write(*,'(A,I0,A,A,A,I0,A,F0.1,A,F0.1)') &
          '[rank ', rank, '] FAIL ', trim(label), &
          ' recv mismatch at elem ', k, ': got ', field(DCOUNT + k), &
          ' expected ', expected
      end if
    end do
  end subroutine verify_recv_region

  !> Assert that an actual integer return code equals the expected value,
  !! recording a failure (with a diagnostic) otherwise.
  subroutine expect_eq(step, actual, expected_code)
    character(len=*), intent(in) :: step
    integer, intent(in)          :: actual
    integer, intent(in)          :: expected_code
    if (actual /= expected_code) then
      local_fail = local_fail + 1
      write(*,'(A,I0,A,A,A,I0,A,I0)') &
        '[rank ', rank, '] FAIL ', trim(step), &
        ': ierr=', actual, ' expected=', expected_code
    end if
  end subroutine expect_eq

end program test_halo_mod
