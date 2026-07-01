# Requirements: HALO Production Hardening for NWP

## Introduction

This spec upgrades HALO from a "correct Tier 1 foundation with flat-buffer exchanges" to a **production-grade micro-library trusted by operational NWP centers** (NOAA/NCEP, NCAR, ECMWF, BoM). The target consumers are structured-grid atmosphere/ocean models (UFS/FV3, MPAS, MOM6) that perform thousands of halo exchanges per timestep on multi-dimensional fields across GPU-accelerated HPC clusters.

## Requirements

### Requirement 1: Multi-Dimensional Structured Halo Descriptor

**User Story:** As a domain scientist with a 3D/4D structured grid, I want to describe my halo geometry (stencil width per dimension, grid extents) once and have HALO automatically pack/unpack the correct ghost-zone slices, so that I do not manually compute byte offsets for every field exchange.

#### Acceptance Criteria

1. HALO SHALL provide a `Structured_Halo_Plan` class that accepts a Kokkos::View layout descriptor (extents per dimension), a halo width per dimension, and a neighbor topology, and precomputes the subview regions for pack/unpack.
2. `Structured_Halo_Plan` SHALL support views of rank 1 through 4 (covering scalar fields, vector fields, and multi-tracer arrays).
3. The plan SHALL correctly handle `Kokkos::LayoutLeft` (Fortran column-major) and `Kokkos::LayoutRight` (C row-major) views, selecting the appropriate subview slicing order.
4. The plan SHALL be reusable across timesteps with different view instances of the same layout (bind-once, exchange-many pattern).
5. The plan SHALL validate that the view extents are large enough to hold the specified halo width on each face.

### Requirement 2: GPU Device-Side Pack/Unpack Kernels

**User Story:** As a performance engineer, I want halo pack and unpack to execute as Kokkos parallel_for kernels on the device (GPU), so that field data never round-trips through host memory just for MPI buffer preparation.

#### Acceptance Criteria

1. Packing from a multi-dimensional view into a contiguous send buffer SHALL execute as a `Kokkos::parallel_for` in the view's execution space.
2. Unpacking from a contiguous receive buffer into the halo region of a multi-dimensional view SHALL execute as a `Kokkos::parallel_for` in the view's execution space.
3. For host-space views, the pack/unpack kernels SHALL execute on the host execution space (Serial or OpenMP) without requiring GPU hardware.
4. The pack/unpack kernels SHALL be fenced (Kokkos::fence) before MPI send and after MPI receive to ensure data visibility.
5. Pack/unpack SHALL handle non-contiguous (strided) subviews correctly.

### Requirement 3: Strided / Non-Contiguous View Support

**User Story:** As a model developer, I want to exchange halos on a subview (e.g., a single tracer slice from a 4D array) without first copying it into a contiguous buffer myself.

#### Acceptance Criteria

1. The structured exchange functions SHALL accept Kokkos::View arguments that may be non-contiguous (strided layout from subview operations).
2. For non-contiguous views, the exchange SHALL use the pack/unpack kernels (Requirement 2) to stage data into contiguous MPI buffers transparently.
3. For contiguous views where the halo region is already contiguous in memory, the exchange MAY bypass packing and use the view's data pointer directly (zero-copy optimization).

### Requirement 4: MPI Error Handling with ERRORS_RETURN

**User Story:** As an operations engineer, I want HALO to detect MPI errors gracefully rather than having the MPI runtime abort the entire job, so that error context (rank, communicator, operation) is captured before termination.

#### Acceptance Criteria

1. HALO SHALL set `MPI_ERRORS_RETURN` on every Communicator it owns (not predefined ones) so that MPI errors are returned as codes rather than triggering process abort.
2. On detecting a non-success MPI return code, HALO SHALL include the local rank, remote rank, communicator name (if set), and operation name in the error message before throwing or aborting.
3. HALO SHALL provide a configurable error policy: throw (default), or abort with diagnostics (for Fortran callers where exceptions cannot propagate).
4. The Fortran C-interop layer SHALL map the abort policy to a call to `MPI_Abort` with a descriptive error message written to stderr before aborting.

### Requirement 5: Versioned API / ABI Stability

**User Story:** As an HPC center administrator, I want HALO to follow semantic versioning with public version macros, so that I can pin versions and detect incompatible upgrades at compile time.

#### Acceptance Criteria

1. HALO SHALL define `HALO_VERSION_MAJOR`, `HALO_VERSION_MINOR`, `HALO_VERSION_PATCH` as integer preprocessor macros in a public header (`halo/version.hpp`).
2. HALO SHALL provide a `HALO_VERSION` string macro in the form "MAJOR.MINOR.PATCH".
3. The CMake `project(VERSION ...)` SHALL be the single source of truth, and the version header SHALL be generated from it via `configure_file`.
4. HALO SHALL maintain a CHANGELOG.md documenting all breaking changes, new features, and bug fixes per release.

### Requirement 6: Spack Package

**User Story:** As an HPC sysadmin, I want to install HALO via Spack with automatic dependency resolution (MPI, Kokkos), so that it integrates into our module system without manual builds.

#### Acceptance Criteria

1. HALO SHALL provide a `spack/package.py` (or contribute to the Spack built-in repo) that builds HALO with configurable variants: `+fortran`, `+gpu_aware_mpi`, `+tests`.
2. The Spack package SHALL declare dependencies on `mpi`, `kokkos`, and optionally `googletest` and `rapidcheck`.
3. The Spack package SHALL support all Kokkos backends that HALO supports (OpenMP, CUDA, HIP, Serial).
4. `spack install halo` SHALL produce a working installation that passes `spack test run halo`.

### Requirement 7: Persistent Communication for Repeated Exchanges

**User Story:** As a performance engineer, I want to use MPI persistent communication (MPI_Send_init/Recv_init) for halo exchanges that repeat every timestep with the same buffers, so that per-call setup overhead is eliminated.

#### Acceptance Criteria

1. HALO SHALL provide a `Persistent_Halo_Handle` class that binds a `Halo_Plan` to a specific view, calling `MPI_Send_init` / `MPI_Recv_init` once.
2. `Persistent_Halo_Handle::start()` SHALL call `MPI_Startall` to initiate all persistent requests.
3. `Persistent_Halo_Handle::wait()` SHALL call `MPI_Waitall` to complete the exchange.
4. The persistent handle SHALL be reusable across timesteps: start() → wait() → start() → wait() ...
5. `Persistent_Halo_Handle` SHALL be move-only and RAII: destruction calls `MPI_Request_free` on all persistent requests.

### Requirement 8: Communication/Computation Overlap Documentation

**User Story:** As a model developer, I want clear documentation and examples showing how to overlap interior computation with halo communication using the async API.

#### Acceptance Criteria

1. HALO SHALL provide documentation (in `docs/overlap_pattern.md`) describing the interior/halo decomposition pattern with code examples.
2. The documentation SHALL show how to: (a) initiate async exchange, (b) compute the interior while communication proceeds, (c) wait for exchange completion, (d) compute the halo-dependent boundary.
3. HALO SHALL document buffer lifetime requirements: the view must not be written to (in the exchanged regions) between `exchange_async` and `wait()`.

### Requirement 9: Thread Safety Documentation and Communication Thread Option

**User Story:** As a model developer on a Cray system where only MPI_THREAD_SERIALIZED is available, I want clear guidance on how to call HALO safely from OpenMP parallel regions.

#### Acceptance Criteria

1. HALO SHALL document the thread-safety model in `docs/thread_safety.md`: what is safe to call concurrently, what requires serialization.
2. When `MPI_THREAD_SERIALIZED` is detected, HALO SHALL document that all exchange calls must be issued from a serial context (outside `#pragma omp parallel`).
3. HALO SHALL optionally support a dedicated communication thread mode (configurable at `Environment::initialize`) that runs all MPI calls on a background thread, enabling safe use from OpenMP parallel regions under `MPI_THREAD_SERIALIZED`.

### Requirement 10: Diagnostics and Instrumentation Hooks

**User Story:** As a performance analyst, I want to measure per-exchange timing, bytes transferred, and wait time without modifying HALO source code.

#### Acceptance Criteria

1. HALO SHALL provide a `halo::Diagnostics` callback interface (function pointer or `std::function`) that is invoked at exchange start, send-post, recv-complete, and exchange-end events.
2. The callback SHALL receive: local rank, exchange direction (blocking/async), neighbor count, total bytes, and elapsed time (nanoseconds).
3. The diagnostics hook SHALL be optional: when no callback is registered, zero overhead (no timing calls, no virtual dispatch).
4. The hook SHALL NOT create a compile-time dependency on LOGS or any external library (Tier 1 isolation preserved).

### Requirement 11: Comprehensive Error Messages at Scale

**User Story:** As a support engineer debugging a failure at 100K+ ranks, I want HALO error messages to include the local rank, remote rank, communicator identity, and plan descriptor.

#### Acceptance Criteria

1. All error messages from exchange operations SHALL include the local rank and the communicator name (if set via `MPI_Comm_set_name`) or handle integer.
2. All error messages SHALL include the plan's neighbor list summary (send-to ranks, recv-from ranks) so the failing topology is identifiable.
3. The Fortran C-interop layer SHALL propagate the error string (truncated to a fixed max length) to a Fortran-accessible character buffer when the abort error policy is active.

### Requirement 12: GPU-Aware MPI Runtime Detection

**User Story:** As a model developer, I want HALO to detect at runtime whether the MPI implementation supports device pointers, so I don't need a separate compile for GPU-aware vs non-GPU-aware clusters.

#### Acceptance Criteria

1. HALO SHALL provide an `Environment::is_gpu_aware_mpi()` query that returns true if the runtime MPI supports device pointers.
2. The runtime check SHALL use `MPIX_Query_cuda_support()` (CUDA), `MPIX_Query_rocm_support()` (HIP), or equivalent when available, falling back to the compile-time `HALO_GPU_AWARE_MPI` flag otherwise.
3. When runtime detection indicates GPU-aware MPI is available, the exchange functions SHALL pass device pointers directly even if `HALO_GPU_AWARE_MPI` was not defined at compile time.
4. The runtime-detected state SHALL be queryable so models can log which path is active.

### Requirement 13: Topology-Aware Neighbor Collectives

**User Story:** As a performance engineer on an HPE Cray with Slingshot, I want HALO to optionally use MPI neighbor collectives (MPI_Neighbor_alltoallv) to benefit from topology-aware routing.

#### Acceptance Criteria

1. HALO SHALL provide an exchange variant that uses `MPI_Dist_graph_create_adjacent` + `MPI_Neighbor_alltoallv` under the hood when the plan has symmetric send/recv neighbor lists.
2. This variant SHALL be selectable via a plan option or a separate function (`exchange_neighbor_collective`).
3. When the neighbor list is asymmetric or the MPI implementation does not support neighbor collectives, HALO SHALL fall back to the standard Isend/Irecv path transparently.

### Requirement 14: FV3/MPAS Adapter Examples

**User Story:** As a UFS/FV3 or MPAS developer, I want a working example showing how to replace legacy ESMF halo exchange calls with HALO, so I can evaluate adoption incrementally.

#### Acceptance Criteria

1. HALO SHALL provide `examples/fv3_adapter.F90` showing how an FV3 cubed-sphere tile replaces `ESMF_FieldHaloStore`/`ESMF_FieldHalo` with `halo_plan_create`/`halo_exchange_blocking` via the Fortran interface.
2. HALO SHALL provide `examples/mpas_adapter.cpp` showing how an MPAS unstructured mesh exchanges cell halos using the C++ API with a Kokkos::View.
3. Both examples SHALL compile against the installed HALO package (not require the HALO source tree) and include build instructions.

</content>
</invoke>
