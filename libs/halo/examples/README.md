# HALO Model Adapter Examples

This directory contains working examples showing how operational NWP models can
replace legacy ESMF halo exchange calls with the HALO micro-library.

## FV3 Cubed-Sphere Adapter (`fv3_adapter.F90`)

Demonstrates how a UFS/FV3 cubed-sphere tile replaces `ESMF_FieldHaloStore` /
`ESMF_FieldHalo` / `ESMF_FieldHaloRelease` with the HALO Fortran interface
(`halo_mod`).

### API Mapping: ESMF → HALO

| ESMF Call | HALO Equivalent | Notes |
|-----------|-----------------|-------|
| `ESMF_VMGet(vm, mpiCommunicator=comm)` | `halo_init(mpi_comm, comm_handle, ierr)` | One-time init; wraps the MPI communicator |
| `ESMF_FieldHaloStore(field, routehandle=rh)` | `halo_plan_create(comm, s_ranks, s_counts, r_ranks, r_counts, plan, ierr)` | Topology defined explicitly (not discovered) |
| `ESMF_FieldHalo(field, routehandle=rh)` | `halo_exchange_blocking(plan, array, ierr)` | Blocking exchange on a contiguous real(8) array |
| *(no ESMF equivalent)* | `halo_exchange_async(plan, array, handle, ierr)` | Non-blocking; enables computation overlap |
| *(no ESMF equivalent)* | `halo_wait(handle, ierr)` | Complete the async exchange |
| `ESMF_FieldHaloRelease(routehandle=rh)` | `halo_destroy_plan(plan, ierr)` | Free plan resources |
| `ESMF_Finalize()` | `halo_destroy_comm(comm, ierr)` | Free communicator resources |

### Conceptual Mapping: ESMF RouteHandle → Halo_Plan

An **ESMF RouteHandle** encapsulates:
- The communication pattern (who sends to whom)
- Pack/unpack metadata for field array regions
- Internal MPI request management

A **Halo_Plan** captures the same logical information:
- `send_ranks` / `recv_ranks` — neighbor topology
- `send_counts` / `recv_counts` — elements per neighbor
- Internal MPI buffer management

Key difference: ESMF discovers the topology from the DistGrid decomposition
automatically. With HALO, the model explicitly provides the neighbor lists.
For FV3, these come from the cubed-sphere tile connectivity table (which the
model already knows).

### Migration Strategy for FV3

1. **Extract the MPI communicator** from the ESMF VM:
   ```fortran
   call ESMF_VMGet(vm, mpiCommunicator=mpi_comm_int, rc=rc)
   call halo_init(mpi_comm_int, comm_handle, ierr)
   ```

2. **Build the neighbor topology** from the FV3 tile connectivity:
   ```fortran
   ! FV3 already knows its tile neighbors from the grid spec.
   ! West/East halo strip: nhalo * ny_tile elements
   ! South/North halo strip: nhalo * nx_tile elements
   send_ranks  = [tile_west, tile_east, tile_south, tile_north]
   send_counts = [nhalo*ny, nhalo*ny, nhalo*nx, nhalo*nx]
   recv_ranks  = send_ranks   ! Symmetric for cubed-sphere faces
   recv_counts = send_counts
   ```

3. **Replace the ESMF calls** in the timestep loop:
   ```fortran
   ! Before (ESMF):
   call ESMF_FieldHalo(field, routehandle=rh, rc=rc)

   ! After (HALO):
   call halo_exchange_blocking(plan_handle, field_1d, ierr)
   ```

4. **Optional — overlap communication with computation:**
   ```fortran
   call halo_exchange_async(plan_handle, field_1d, async_handle, ierr)
   call compute_interior(field_1d)   ! No halo dependency
   call halo_wait(async_handle, ierr)
   call compute_boundary(field_1d)   ! Needs halo data
   ```

### Data Layout Considerations

HALO's Fortran API operates on **contiguous 1D real(8) arrays**. FV3 fields are
typically 2D or 3D arrays stored in column-major (Fortran) order. To pass them:

```fortran
! Option A: Reshape to 1D (zero-copy if already contiguous)
real(8), target :: field_2d(nx_with_halo, ny_with_halo)
real(8), pointer, contiguous :: field_1d(:)
field_1d(1:size(field_2d)) => field_2d

! Option B: Pass a flat allocation directly
allocate(field_1d(nx_with_halo * ny_with_halo))
```

The HALO library transfers bytes faithfully without interpreting dimensionality,
so the exchange is layout-agnostic as long as both sender and receiver use the
same memory layout (which they do on a structured grid).

## Building the Examples

### Prerequisites

- HALO library installed with Fortran support
- MPI implementation with Fortran bindings (OpenMPI, MPICH, Intel MPI)
- CMake ≥ 3.21

### Build Steps

```bash
# 1. Set the path to your HALO installation
export HALO_PREFIX=/path/to/halo/install

# 2. Configure and build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$HALO_PREFIX
make

# 3. Run with MPI (6 ranks = one per cubed-sphere tile)
mpirun -np 6 ./fv3_adapter
```

### Integration with UFS Build System

To integrate HALO into the UFS/FV3 CMake build, add to the FV3 CMakeLists.txt:

```cmake
find_package(HALO REQUIRED)
target_link_libraries(fv3_atm PRIVATE HELM::HALO_Fortran)
```

Then replace the ESMF halo exchange calls incrementally, one field at a time.
