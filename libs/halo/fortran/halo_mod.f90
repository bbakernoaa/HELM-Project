!> @file halo_mod.f90
!! @brief Fortran public API for the HALO micro-library (HELM Tier 1).
!!
!! This module provides idiomatic Fortran subroutines for legacy NUOPC/ESMF
!! models to adopt HALO halo exchanges incrementally. It is a thin
!! `iso_c_binding` wrapper around the `extern "C"` interop layer implemented in
!! `src/fortran/halo_c_interop.cpp`.
!!
!! Design contract (mirrors the C interop layer):
!!   - Every C function returns an `integer(c_int)` error code where 0 means
!!     success and non-zero values indicate specific error conditions.
!!   - Opaque C++ resources (Communicator, Halo_Plan, Halo_Handle) are exposed
!!     to Fortran as plain `integer(c_int)` handle tokens, never as raw C
!!     pointers. The token is obtained from a create call and passed back to
!!     subsequent calls for resource identification.
!!   - Contiguous Fortran arrays are passed to C by base address via `c_loc`.
!!     The corresponding dummy arguments therefore carry the `target` and
!!     `contiguous` attributes.
!!   - MPI communicators are accepted as plain integers, compatible with both
!!     the Fortran 2008 MPI binding (`type(MPI_Comm)%mpi_val`) and the legacy
!!     ESMF integer-handle convention (`call ESMF_VMGet(vm, mpiCommunicator=...)`).
!!
!! ===========================================================================
!! CRITICAL MEMORY-LAYOUT CONTRACT (read before extending this interface)
!! ===========================================================================
!! Fortran stores multi-dimensional arrays in COLUMN-MAJOR order, whereas
!! C/C++ -- and the Kokkos default on HostSpace (Kokkos::LayoutRight) -- store
!! them in ROW-MAJOR order. To make the HALO interop layout-agnostic and to
!! guarantee that NO silent transposition ever occurs, the boundary
!! deliberately passes every field as a FLAT, CONTIGUOUS 1D BYTE BUFFER:
!!
!!   * The Fortran side passes the base address of a contiguous array
!!     (`c_loc(array(1))`) together with an element count and an element size
!!     in bytes.
!!   * The C++ side reconstructs a non-owning
!!     `Kokkos::View<char*, HostSpace, Unmanaged>` spanning exactly
!!     `num_elements * element_size` bytes -- a 1D view of raw bytes.
!!
!! Because the buffer is one-dimensional, column-major (LayoutLeft) and
!! row-major (LayoutRight) orderings are IDENTICAL: for rank-1 data
!! LayoutLeft == LayoutRight, so the bytes are interpreted the same on both
!! sides and the exchange is a faithful byte-for-byte transfer. Callers MUST
!! therefore pass contiguous arrays; the `contiguous` attribute on the dummy
!! arguments below enforces this, and `c_loc` is only ever applied to
!! contiguous, target arrays.
!!
!! WARNING -- FUTURE EXTENSION HAZARD: if HALO is ever extended to expose
!! MULTI-DIMENSIONAL Kokkos views across this boundary, the Fortran
!! column-major ordering MUST be mapped to `Kokkos::LayoutLeft` (NOT the
!! C-default `Kokkos::LayoutRight`). Mapping column-major Fortran storage onto
!! a LayoutRight view would silently transpose the data and corrupt every
!! halo exchange. Keep the flat-1D contract, or map explicitly to LayoutLeft.
!! ===========================================================================
!!
!! Requirements: 14.2, 14.4, 14.5, 14.6, 14.7, 14.8, 14.11, 14.12, 14.15
module halo_mod
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Public friendly API (idiomatic Fortran subroutines).
  ! ---------------------------------------------------------------------------
  public :: halo_init, halo_comm_create, halo_plan_create
  public :: halo_exchange_blocking, halo_exchange_async
  public :: halo_wait, halo_test
  public :: halo_destroy_plan, halo_destroy_comm

  ! ---------------------------------------------------------------------------
  ! Public error code constants. These values mirror the Halo_Error enum in
  ! src/fortran/halo_c_interop.cpp and MUST be kept in sync with it.
  ! ---------------------------------------------------------------------------
  integer(c_int), parameter, public :: HALO_SUCCESS         = 0   !< Operation succeeded.
  integer(c_int), parameter, public :: HALO_ERR_INVALID_ARG = 1   !< Invalid argument (bad rank, null pointer, ...).
  integer(c_int), parameter, public :: HALO_ERR_MPI         = 2   !< Underlying MPI operation failed.
  integer(c_int), parameter, public :: HALO_ERR_RUNTIME     = 3   !< Runtime error (e.g. MPI not initialized).
  integer(c_int), parameter, public :: HALO_ERR_BAD_HANDLE  = 4   !< Invalid or expired opaque handle token.
  integer(c_int), parameter, public :: HALO_ERR_UNKNOWN     = 99  !< Unknown/unexpected error.

  ! ---------------------------------------------------------------------------
  ! Element size (in bytes) of the data carried by the exchange routines.
  ! The Fortran friendly API exposes real(8) arrays, so the element size passed
  ! to the C interop layer is 8 bytes.
  ! ---------------------------------------------------------------------------
  integer(c_int), parameter :: HALO_REAL8_BYTES = 8

  ! ---------------------------------------------------------------------------
  ! Explicit interfaces to the extern "C" interop functions. Each function
  ! carries the bind(c, name=...) attribute matching the C symbol exactly.
  ! Scalar inputs are passed by value; output handles are written through
  ! intent(out) arguments; the array data pointer is a type(c_ptr) by value.
  ! Every function returns the integer error code as its result.
  ! ---------------------------------------------------------------------------
  interface

    !> Initialize HALO and create the root Communicator from an MPI comm int.
    function halo_init_c(mpi_comm_int, comm_handle_out) &
        bind(c, name='halo_init_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: mpi_comm_int
      integer(c_int), intent(out)       :: comm_handle_out
      integer(c_int)                    :: ierr
    end function halo_init_c

    !> Create a sub-communicator via MPI_Comm_split.
    function halo_comm_create_c(parent_handle, color, key, child_handle_out) &
        bind(c, name='halo_comm_create_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: parent_handle
      integer(c_int), value, intent(in) :: color
      integer(c_int), value, intent(in) :: key
      integer(c_int), intent(out)       :: child_handle_out
      integer(c_int)                    :: ierr
    end function halo_comm_create_c

    !> Create a Halo_Plan from neighbor rank/count arrays.
    function halo_plan_create_c(comm_handle, send_ranks, send_counts, &
        num_send, recv_ranks, recv_counts, num_recv, plan_handle_out) &
        bind(c, name='halo_plan_create_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: comm_handle
      integer(c_int), intent(in)        :: send_ranks(*)
      integer(c_int), intent(in)        :: send_counts(*)
      integer(c_int), value, intent(in) :: num_send
      integer(c_int), intent(in)        :: recv_ranks(*)
      integer(c_int), intent(in)        :: recv_counts(*)
      integer(c_int), value, intent(in) :: num_recv
      integer(c_int), intent(out)       :: plan_handle_out
      integer(c_int)                    :: ierr
    end function halo_plan_create_c

    !> Execute a blocking halo exchange over a contiguous byte buffer.
    function halo_exchange_blocking_c(plan_handle, data, &
        num_elements, element_size) &
        bind(c, name='halo_exchange_blocking_c') result(ierr)
      import :: c_int, c_ptr
      integer(c_int), value, intent(in) :: plan_handle
      type(c_ptr), value, intent(in)    :: data
      integer(c_int), value, intent(in) :: num_elements
      integer(c_int), value, intent(in) :: element_size
      integer(c_int)                    :: ierr
    end function halo_exchange_blocking_c

    !> Initiate a non-blocking halo exchange; returns a Halo_Handle token.
    function halo_exchange_async_c(plan_handle, data, &
        num_elements, element_size, handle_out) &
        bind(c, name='halo_exchange_async_c') result(ierr)
      import :: c_int, c_ptr
      integer(c_int), value, intent(in) :: plan_handle
      type(c_ptr), value, intent(in)    :: data
      integer(c_int), value, intent(in) :: num_elements
      integer(c_int), value, intent(in) :: element_size
      integer(c_int), intent(out)       :: handle_out
      integer(c_int)                    :: ierr
    end function halo_exchange_async_c

    !> Block until the async exchange identified by handle completes.
    function halo_wait_c(handle) &
        bind(c, name='halo_wait_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: handle
      integer(c_int)                    :: ierr
    end function halo_wait_c

    !> Test (non-blocking) whether the async exchange has completed.
    function halo_test_c(handle, complete_out) &
        bind(c, name='halo_test_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: handle
      integer(c_int), intent(out)       :: complete_out
      integer(c_int)                    :: ierr
    end function halo_test_c

    !> Destroy a Halo_Plan and invalidate its handle token.
    function halo_destroy_plan_c(plan_handle) &
        bind(c, name='halo_destroy_plan_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: plan_handle
      integer(c_int)                    :: ierr
    end function halo_destroy_plan_c

    !> Destroy a Communicator and invalidate its handle token.
    function halo_destroy_comm_c(comm_handle) &
        bind(c, name='halo_destroy_comm_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: comm_handle
      integer(c_int)                    :: ierr
    end function halo_destroy_comm_c

  end interface

contains

  !> @brief Initialize HALO from a Fortran MPI communicator handle.
  !!
  !! Calls Environment::initialize() on the C++ side, constructs a root
  !! Communicator from the supplied MPI communicator, and returns an opaque
  !! Communicator handle token.
  !!
  !! @param[in]  mpi_comm     Integer MPI communicator. Compatible with
  !!                          type(MPI_Comm)%mpi_val and the ESMF integer
  !!                          handle convention (Requirement 14.11).
  !! @param[out] comm_handle  Opaque Communicator handle token.
  !! @param[out] ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_init(mpi_comm, comm_handle, ierr)
    integer, intent(in)  :: mpi_comm
    integer, intent(out) :: comm_handle
    integer, intent(out) :: ierr
    integer(c_int) :: c_handle
    ierr = int(halo_init_c(int(mpi_comm, c_int), c_handle))
    comm_handle = int(c_handle)
  end subroutine halo_init

  !> @brief Create a sub-communicator via MPI_Comm_split.
  !!
  !! @param[in]  parent_handle  Opaque handle of the parent Communicator.
  !! @param[in]  color          Split color (MPI_UNDEFINED yields a null comm).
  !! @param[in]  key            Split key controlling rank ordering.
  !! @param[out] child_handle   Opaque handle of the created sub-communicator.
  !! @param[out] ierr           HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_comm_create(parent_handle, color, key, child_handle, ierr)
    integer, intent(in)  :: parent_handle, color, key
    integer, intent(out) :: child_handle
    integer, intent(out) :: ierr
    integer(c_int) :: c_handle
    ierr = int(halo_comm_create_c(int(parent_handle, c_int), &
                                  int(color, c_int), &
                                  int(key, c_int), c_handle))
    child_handle = int(c_handle)
  end subroutine halo_comm_create

  !> @brief Create a Halo_Plan from neighbor rank and count arrays.
  !!
  !! The number of send/receive neighbors is inferred from the size of the
  !! rank arrays. Callers must ensure send_ranks/send_counts have matching
  !! length, and likewise for recv_ranks/recv_counts (Requirement 14.12).
  !!
  !! @param[in]  comm_handle  Opaque handle of the owning Communicator.
  !! @param[in]  send_ranks   Send-neighbor ranks.
  !! @param[in]  send_counts  Per-neighbor send element counts.
  !! @param[in]  recv_ranks   Receive-neighbor ranks.
  !! @param[in]  recv_counts  Per-neighbor receive element counts.
  !! @param[out] plan_handle  Opaque handle of the created Halo_Plan.
  !! @param[out] ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_plan_create(comm_handle, send_ranks, send_counts, &
                              recv_ranks, recv_counts, plan_handle, ierr)
    integer, intent(in)  :: comm_handle
    integer, intent(in)  :: send_ranks(:), send_counts(:)
    integer, intent(in)  :: recv_ranks(:), recv_counts(:)
    integer, intent(out) :: plan_handle
    integer, intent(out) :: ierr
    integer(c_int) :: c_handle
    integer(c_int), allocatable :: c_send_ranks(:), c_send_counts(:)
    integer(c_int), allocatable :: c_recv_ranks(:), c_recv_counts(:)
    ! Copy into c_int arrays so the call is kind-correct regardless of the
    ! caller's default integer kind (e.g. when compiled with -fdefault-integer-8).
    c_send_ranks  = int(send_ranks,  c_int)
    c_send_counts = int(send_counts, c_int)
    c_recv_ranks  = int(recv_ranks,  c_int)
    c_recv_counts = int(recv_counts, c_int)
    ierr = int(halo_plan_create_c(int(comm_handle, c_int), &
                                  c_send_ranks, c_send_counts, &
                                  int(size(send_ranks), c_int), &
                                  c_recv_ranks, c_recv_counts, &
                                  int(size(recv_ranks), c_int), &
                                  c_handle))
    plan_handle = int(c_handle)
  end subroutine halo_plan_create

  !> @brief Execute a blocking halo exchange on a contiguous real(8) array.
  !!
  !! The array base address is passed to the C interop layer via c_loc; the
  !! dummy argument therefore has the target and contiguous attributes
  !! (Requirement 14.5).
  !!
  !! @param[in]     plan_handle  Opaque handle of the Halo_Plan.
  !! @param[in,out] array        Contiguous real(8) field buffer.
  !! @param[out]    ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_exchange_blocking(plan_handle, array, ierr)
    integer, intent(in)                         :: plan_handle
    real(c_double), intent(inout), target, contiguous :: array(:)
    integer, intent(out)                        :: ierr
    ierr = int(halo_exchange_blocking_c(int(plan_handle, c_int), &
                                        c_loc(array), &
                                        int(size(array), c_int), &
                                        HALO_REAL8_BYTES))
  end subroutine halo_exchange_blocking

  !> @brief Initiate a non-blocking halo exchange and return a handle token.
  !!
  !! The array base address is passed via c_loc (Requirement 14.6). The
  !! returned handle must later be passed to halo_wait or halo_test, and the
  !! array must remain valid and contiguous until the exchange completes.
  !!
  !! @param[in]     plan_handle  Opaque handle of the Halo_Plan.
  !! @param[in,out] array        Contiguous real(8) field buffer.
  !! @param[out]    handle       Opaque Halo_Handle token for the pending exchange.
  !! @param[out]    ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_exchange_async(plan_handle, array, handle, ierr)
    integer, intent(in)                         :: plan_handle
    real(c_double), intent(inout), target, contiguous :: array(:)
    integer, intent(out)                        :: handle
    integer, intent(out)                        :: ierr
    integer(c_int) :: c_handle
    ierr = int(halo_exchange_async_c(int(plan_handle, c_int), &
                                     c_loc(array), &
                                     int(size(array), c_int), &
                                     HALO_REAL8_BYTES, c_handle))
    handle = int(c_handle)
  end subroutine halo_exchange_async

  !> @brief Block until the async exchange identified by handle completes.
  !!
  !! @param[in]  handle  Opaque Halo_Handle token from halo_exchange_async.
  !! @param[out] ierr    HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_wait(handle, ierr)
    integer, intent(in)  :: handle
    integer, intent(out) :: ierr
    ierr = int(halo_wait_c(int(handle, c_int)))
  end subroutine halo_wait

  !> @brief Test (non-blocking) whether an async exchange has completed.
  !!
  !! @param[in]  handle       Opaque Halo_Handle token from halo_exchange_async.
  !! @param[out] is_complete  .true. if all operations are complete, else .false.
  !! @param[out] ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_test(handle, is_complete, ierr)
    integer, intent(in)  :: handle
    logical, intent(out) :: is_complete
    integer, intent(out) :: ierr
    integer(c_int) :: flag
    flag = 0_c_int
    ierr = int(halo_test_c(int(handle, c_int), flag))
    is_complete = (flag == 1_c_int)
  end subroutine halo_test

  !> @brief Destroy a Halo_Plan and invalidate its handle token.
  !!
  !! @param[in]  plan_handle  Opaque handle of the Halo_Plan to destroy.
  !! @param[out] ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_destroy_plan(plan_handle, ierr)
    integer, intent(in)  :: plan_handle
    integer, intent(out) :: ierr
    ierr = int(halo_destroy_plan_c(int(plan_handle, c_int)))
  end subroutine halo_destroy_plan

  !> @brief Destroy a Communicator and invalidate its handle token.
  !!
  !! @param[in]  comm_handle  Opaque handle of the Communicator to destroy.
  !! @param[out] ierr         HALO_SUCCESS on success, otherwise an error code.
  subroutine halo_destroy_comm(comm_handle, ierr)
    integer, intent(in)  :: comm_handle
    integer, intent(out) :: ierr
    ierr = int(halo_destroy_comm_c(int(comm_handle, c_int)))
  end subroutine halo_destroy_comm

end module halo_mod
