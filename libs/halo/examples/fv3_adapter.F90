!> @file fv3_adapter.F90
!! @brief FV3 Cubed-Sphere Adapter Example
!!
!! Demonstrates how to replace legacy ESMF_FieldHaloStore / ESMF_FieldHalo
!! calls with the HALO micro-library's Fortran interface (halo_mod).
!!
!! ┌─────────────────────────────────────────────────────────────────────────┐
!! │ ESMF Legacy Pattern           →   HALO Replacement                     │
!! ├─────────────────────────────────────────────────────────────────────────┤
!! │ ESMF_FieldHaloStore(field,    →   halo_plan_create(comm_handle,        │
!! │   routehandle=rh, rc=rc)           send_ranks, send_counts,            │
!! │                                    recv_ranks, recv_counts,            │
!! │                                    plan_handle, ierr)                  │
!! │                                                                         │
!! │ ESMF_FieldHalo(field,         →   halo_exchange_blocking(plan_handle,  │
!! │   routehandle=rh, rc=rc)           field_array, ierr)                  │
!! │                                                                         │
!! │ ESMF_FieldHaloRelease(rh,     →   halo_destroy_plan(plan_handle, ierr) │
!! │   rc=rc)                                                                │
!! └─────────────────────────────────────────────────────────────────────────┘
!!
!! Mapping from ESMF RouteHandle to Halo_Plan:
!! ─────────────────────────────────────────────
!!   An ESMF RouteHandle encapsulates the communication pattern discovered by
!!   ESMF_FieldHaloStore — which ranks to talk to, how many elements, and
!!   internal pack/unpack metadata. In HALO, the equivalent concept is a
!!   Halo_Plan, which stores the same logical information:
!!
!!   • send_ranks  — neighbor ranks this tile sends halo data TO
!!   • send_counts — number of real(8) elements sent to each neighbor
!!   • recv_ranks  — neighbor ranks this tile receives halo data FROM
!!   • recv_counts — number of real(8) elements received from each neighbor
!!
!!   The plan is created once during model initialization (analogous to
!!   ESMF_FieldHaloStore) and reused every timestep (analogous to repeated
!!   ESMF_FieldHalo calls with the same routehandle).
!!
!! This example simulates a single FV3 cubed-sphere tile exchanging ghost
!! zones with 4 face-adjacent neighbor tiles.
!!
!! Requirements: 14.1
!!
program fv3_adapter_example
  use halo_mod
  implicit none

  ! ---------------------------------------------------------------------------
  ! Tile geometry parameters (typical FV3 cubed-sphere tile)
  ! ---------------------------------------------------------------------------
  integer, parameter :: NX     = 48    ! Grid cells in x-direction (per tile)
  integer, parameter :: NY     = 48    ! Grid cells in y-direction (per tile)
  integer, parameter :: NHALO  = 3     ! Halo width (FV3 default is 3)
  integer, parameter :: NFIELD = (NX + 2*NHALO) * (NY + 2*NHALO)

  ! ---------------------------------------------------------------------------
  ! HALO handles (replace ESMF RouteHandle and VM)
  ! ---------------------------------------------------------------------------
  integer :: comm_handle      ! Replaces ESMF_VM / ESMF_DELayout communicator
  integer :: plan_handle      ! Replaces ESMF_RouteHandle from FieldHaloStore
  integer :: async_handle     ! For non-blocking exchange (overlap pattern)
  integer :: ierr

  ! ---------------------------------------------------------------------------
  ! Tile neighbor topology for a cubed-sphere face (4 neighbors: W, E, S, N)
  ! ---------------------------------------------------------------------------
  integer, parameter :: NUM_NEIGHBORS = 4

  ! On a 6-tile cubed sphere each tile has exactly 4 face-adjacent neighbors.
  ! In a real FV3 setup these come from the ESMF_DistGrid tile connectivity;
  ! here we hardcode a simple example for tile 0 on a 2x3 tile layout.
  integer :: send_ranks(NUM_NEIGHBORS)
  integer :: send_counts(NUM_NEIGHBORS)
  integer :: recv_ranks(NUM_NEIGHBORS)
  integer :: recv_counts(NUM_NEIGHBORS)

  ! ---------------------------------------------------------------------------
  ! Field data — contiguous 1D array representing a 2D tile with halo padding.
  ! In a real FV3 model this would be the field pointer from ESMF_FieldGet.
  ! The HALO Fortran API operates on contiguous real(8) 1D arrays.
  ! ---------------------------------------------------------------------------
  real(8), target, allocatable :: field(:)

  integer :: my_rank, i
  integer :: mpi_comm_world_int

  ! ===========================================================================
  ! Step 0: MPI Initialization (already done by FV3/ESMF in production)
  ! ===========================================================================
  ! In a real FV3 model, you would get the MPI communicator from ESMF:
  !   call ESMF_VMGet(vm, mpiCommunicator=mpi_comm_world_int, rc=rc)
  !
  ! For this standalone example, we initialize MPI directly.
  ! ---------------------------------------------------------------------------
  call mpi_init_stub(mpi_comm_world_int, my_rank)

  ! ===========================================================================
  ! Step 1: Initialize HALO (replaces implicit ESMF communication init)
  ! ===========================================================================
  ! ESMF handles communication initialization internally; with HALO you
  ! explicitly create a Communicator from the MPI comm integer.
  ! ---------------------------------------------------------------------------
  call halo_init(mpi_comm_world_int, comm_handle, ierr)
  if (ierr /= HALO_SUCCESS) then
    print *, '[ERROR] halo_init failed with code:', ierr
    stop 1
  end if
  print '(A,I2,A)', ' [Rank ', my_rank, '] HALO initialized successfully'

  ! ===========================================================================
  ! Step 2: Define the neighbor topology (replaces ESMF_FieldHaloStore)
  ! ===========================================================================
  ! In FV3 with ESMF, the topology is discovered from the DistGrid:
  !   call ESMF_FieldHaloStore(field, routehandle=rh, rc=rc)
  !
  ! With HALO, you provide the topology explicitly. For a cubed-sphere tile,
  ! each face has 4 neighbors. The counts represent the number of real(8)
  ! elements in each halo strip:
  !   - West/East strips:  NHALO * NY elements
  !   - South/North strips: NHALO * NX elements
  ! ---------------------------------------------------------------------------
  call setup_tile_neighbors(my_rank, send_ranks, send_counts, &
                            recv_ranks, recv_counts)

  ! ===========================================================================
  ! Step 3: Create the Halo Plan (replaces ESMF RouteHandle)
  ! ===========================================================================
  ! The plan is reusable across timesteps — create it once, use it many times.
  ! This is the direct replacement for ESMF_FieldHaloStore's routehandle.
  ! ---------------------------------------------------------------------------
  call halo_plan_create(comm_handle, send_ranks, send_counts, &
                        recv_ranks, recv_counts, plan_handle, ierr)
  if (ierr /= HALO_SUCCESS) then
    print *, '[ERROR] halo_plan_create failed with code:', ierr
    stop 1
  end if
  print '(A,I2,A)', ' [Rank ', my_rank, '] Halo plan created (4 neighbors)'

  ! ===========================================================================
  ! Step 4: Allocate field with halo padding
  ! ===========================================================================
  allocate(field(NFIELD))
  field = 0.0d0

  ! Fill the interior with rank-specific data (simulating FV3 dynamics output)
  do i = 1, NFIELD
    field(i) = dble(my_rank * 1000 + i)
  end do

  ! ===========================================================================
  ! Step 5a: BLOCKING exchange (replaces ESMF_FieldHalo)
  ! ===========================================================================
  ! ESMF Legacy:
  !   call ESMF_FieldHalo(field, routehandle=rh, rc=rc)
  !
  ! HALO Replacement:
  !   call halo_exchange_blocking(plan_handle, field_array, ierr)
  ! ---------------------------------------------------------------------------
  call halo_exchange_blocking(plan_handle, field, ierr)
  if (ierr /= HALO_SUCCESS) then
    print *, '[ERROR] halo_exchange_blocking failed with code:', ierr
    stop 1
  end if
  print '(A,I2,A)', ' [Rank ', my_rank, '] Blocking halo exchange complete'

  ! ===========================================================================
  ! Step 5b: ASYNC exchange with computation overlap (optional pattern)
  ! ===========================================================================
  ! For models that can overlap interior computation with halo communication,
  ! HALO provides an async exchange API. This has NO ESMF equivalent — it is
  ! a performance optimization unique to HALO.
  !
  ! Pattern:
  !   1. Initiate async exchange
  !   2. Compute the interior (does not depend on halo data)
  !   3. Wait for exchange completion
  !   4. Compute boundary (depends on halo data)
  ! ---------------------------------------------------------------------------
  call halo_exchange_async(plan_handle, field, async_handle, ierr)
  if (ierr /= HALO_SUCCESS) then
    print *, '[ERROR] halo_exchange_async failed with code:', ierr
    stop 1
  end if

  ! ... compute interior here (physics, dynamics on non-halo cells) ...
  call compute_interior_stub(field, NX, NY, NHALO)

  ! Wait for halo data to arrive
  call halo_wait(async_handle, ierr)
  if (ierr /= HALO_SUCCESS) then
    print *, '[ERROR] halo_wait failed with code:', ierr
    stop 1
  end if
  print '(A,I2,A)', ' [Rank ', my_rank, '] Async halo exchange + overlap complete'

  ! ... now safe to compute boundary stencils using halo data ...

  ! ===========================================================================
  ! Step 6: Cleanup (replaces ESMF_FieldHaloRelease + ESMF_Finalize)
  ! ===========================================================================
  ! ESMF Legacy:
  !   call ESMF_FieldHaloRelease(routehandle=rh, rc=rc)
  !
  ! HALO Replacement:
  !   call halo_destroy_plan(plan_handle, ierr)
  !   call halo_destroy_comm(comm_handle, ierr)
  ! ---------------------------------------------------------------------------
  call halo_destroy_plan(plan_handle, ierr)
  call halo_destroy_comm(comm_handle, ierr)

  deallocate(field)

  print '(A,I2,A)', ' [Rank ', my_rank, '] FV3 adapter example complete'

  call mpi_finalize_stub()

contains

  !> Set up cubed-sphere tile neighbor topology.
  !! In a real FV3 model, this information comes from the ESMF DistGrid
  !! tile connectivity table. Here we simulate a simple 6-tile cube.
  subroutine setup_tile_neighbors(rank, s_ranks, s_counts, r_ranks, r_counts)
    integer, intent(in)  :: rank
    integer, intent(out) :: s_ranks(NUM_NEIGHBORS)
    integer, intent(out) :: s_counts(NUM_NEIGHBORS)
    integer, intent(out) :: r_ranks(NUM_NEIGHBORS)
    integer, intent(out) :: r_counts(NUM_NEIGHBORS)

    ! Cubed-sphere connectivity for tile 0..5 (simplified).
    ! Each tile communicates with exactly 4 face-adjacent tiles.
    ! Neighbor indices: 1=West, 2=East, 3=South, 4=North
    select case (mod(rank, 6))
    case (0)
      s_ranks  = [4, 1, 5, 2]
      r_ranks  = [4, 1, 5, 2]
    case (1)
      s_ranks  = [0, 3, 5, 2]
      r_ranks  = [0, 3, 5, 2]
    case (2)
      s_ranks  = [0, 3, 1, 4]
      r_ranks  = [0, 3, 1, 4]
    case (3)
      s_ranks  = [1, 5, 2, 4]
      r_ranks  = [1, 5, 2, 4]
    case (4)
      s_ranks  = [0, 5, 2, 3]
      r_ranks  = [0, 5, 2, 3]
    case (5)
      s_ranks  = [1, 4, 0, 3]
      r_ranks  = [1, 4, 0, 3]
    end select

    ! Halo strip sizes:
    !   West/East  strips: NHALO * NY elements
    !   South/North strips: NHALO * NX elements
    s_counts = [NHALO * NY, NHALO * NY, NHALO * NX, NHALO * NX]
    r_counts = [NHALO * NY, NHALO * NY, NHALO * NX, NHALO * NX]
  end subroutine setup_tile_neighbors

  !> Stub: simulate MPI_Init and return comm + rank.
  !! In production, MPI is already initialized by the FV3 driver or ESMF.
  subroutine mpi_init_stub(comm, rank)
    integer, intent(out) :: comm, rank
    ! Placeholder: use MPI_COMM_WORLD = 0 (the standard Fortran integer handle)
    comm = 0
    rank = 0
  end subroutine mpi_init_stub

  !> Stub: simulate MPI_Finalize.
  subroutine mpi_finalize_stub()
    ! no-op in this example
  end subroutine mpi_finalize_stub

  !> Stub: simulate interior computation that does not touch halo cells.
  subroutine compute_interior_stub(fld, nx_dim, ny_dim, nhalo_dim)
    real(8), intent(inout) :: fld(:)
    integer, intent(in)    :: nx_dim, ny_dim, nhalo_dim
    ! In production, this would be the physics/dynamics kernel operating
    ! on cells [nhalo+1 : nx+nhalo, nhalo+1 : ny+nhalo].
    ! No-op here — just illustrates the overlap pattern.
  end subroutine compute_interior_stub

end program fv3_adapter_example
