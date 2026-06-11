# Implementation Plan: HALO Production Hardening for NWP

## Overview

Upgrades HALO from a spec-complete Tier 1 foundation to a production-grade micro-library for operational NWP. Adds multi-dimensional structured halo exchange with GPU pack/unpack kernels, persistent communication, runtime GPU-MPI detection, instrumentation hooks, Spack packaging, versioning, and model adapter examples.

## Tasks

- [x] 1. API versioning and stability infrastructure
  - [x] 1.1 Create version header and CHANGELOG
    - Create `include/halo/version.hpp.in` template with HALO_VERSION_MAJOR/MINOR/PATCH macros
    - Add `configure_file()` in CMakeLists.txt to generate version.hpp from project VERSION
    - Create `CHANGELOG.md` at libs/halo root with initial v0.1.0 entry documenting the current API
    - Add version.hpp to the umbrella header include
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 1.2 Implement error policy configuration
    - Create `include/halo/error_policy.hpp` with ErrorPolicy enum (throw_on_error, abort_with_diagnostics)
    - Add `Environment::set_error_policy()` and `Environment::error_policy()` static methods
    - Implement `detail::handle_mpi_error()` that branches on the active policy: throw std::runtime_error or write to stderr + MPI_Abort
    - Update existing exchange error paths to use the new handler
    - Update halo_c_interop to use abort policy when called from Fortran
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 1.3 Set MPI_ERRORS_RETURN on owned communicators
    - In Communicator constructor (for non-predefined, non-null handles), call MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN)
    - In Communicator::split() and duplicate(), set MPI_ERRORS_RETURN on newly created comms
    - Document that predefined communicators (WORLD, SELF) are not modified
    - _Requirements: 4.1_

- [x] 2. GPU pack/unpack kernels and runtime detection
  - [x] 2.1 Implement runtime GPU-aware MPI detection
    - Create `include/halo/detail/gpu_aware_probe.hpp` with `detail::gpu_aware_probe()` function
    - Implement MPIX_Query_cuda_support / MPIX_Query_rocm_support probes with compile-time fallback
    - Cache result in Environment::initialize(), expose via Environment::is_gpu_aware_mpi()
    - Update exchange dispatch to use runtime detection when compile-time flag is absent
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x] 2.2 Implement device-side pack/unpack kernels
    - Create `include/halo/detail/pack_unpack.hpp` with pack<ViewType> and unpack<ViewType> template functions
    - Implement Kokkos::parallel_for pack kernel (subview → contiguous buffer) with execution space fence
    - Implement Kokkos::parallel_for unpack kernel (contiguous buffer → subview) with execution space fence
    - Support both RangePolicy (contiguous subviews) and MDRangePolicy (strided subviews)
    - Detect contiguity at call time and select optimal kernel path
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.3 Write unit tests for pack/unpack kernels
    - Test pack from a contiguous 1D view into a buffer (host execution space)
    - Test unpack from a buffer into a contiguous 1D view
    - Test pack/unpack with a strided 2D subview (non-contiguous)
    - Test pack/unpack with LayoutLeft and LayoutRight views
    - Verify device-side execution via Kokkos execution space tags
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

- [x] 3. Multi-dimensional structured halo exchange
  - [x] 3.1 Implement Structured_Halo_Plan class template
    - Create `include/halo/structured_halo_plan.hpp` with Structured_Halo_Plan<Rank> class
    - Implement face-based neighbor topology for 1D/2D/3D/4D grids (periodic and non-periodic boundaries)
    - Precompute subview index ranges (Kokkos::pair) for each send/recv region per neighbor
    - Validate extents >= 2*halo_width per dimension
    - Store the owning Communicator reference (non-owning pointer)
    - Support both LayoutLeft and LayoutRight by detecting layout at compile time
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.2 Implement exchange_structured_blocking function template
    - Create `include/halo/exchange_structured.hpp` with exchange_structured_blocking<ViewType>
    - For each send-neighbor: pack the subview region into a contiguous send buffer using pack kernel
    - Post all MPI_Irecv (into contiguous recv buffers) before any MPI_Isend
    - Post all MPI_Isend from contiguous send buffers
    - MPI_Waitall, then unpack each recv buffer into the view's halo region
    - Use runtime GPU-aware detection: if GPU-aware and view is contiguous in the halo region, skip pack/unpack
    - _Requirements: 1.1, 2.1, 2.4, 3.1, 3.2, 3.3_

  - [x] 3.3 Implement exchange_structured_async function template
    - Add exchange_structured_async<ViewType> returning a Halo_Handle
    - Same pack/Irecv-before-Isend/unpack-on-completion pattern as blocking but deferred
    - Attach post-receive unpack as the staged_recv callback on the Halo_Handle
    - _Requirements: 1.1, 3.1, 3.2_

  - [x] 3.4 Write integration tests for structured exchange
    - Test 2D periodic grid exchange (4 neighbors) with data verification across 4 ranks
    - Test 3D periodic grid exchange (6 neighbors) with data verification
    - Test non-periodic boundary (some faces have no neighbor)
    - Test LayoutLeft and LayoutRight views produce correct results
    - Test strided subview (e.g., one tracer from a 4D field) exchanges correctly
    - Run with mpirun -np 4
    - _Requirements: 1.1, 1.2, 1.3, 2.5, 3.1_

- [x] 4. Persistent communication
  - [x] 4.1 Implement Persistent_Halo_Handle class
    - Create `include/halo/persistent_halo_handle.hpp` with Persistent_Halo_Handle class
    - Constructor binds a Halo_Plan + view: calls MPI_Send_init / MPI_Recv_init for each neighbor
    - Implement start() → MPI_Startall, wait() → MPI_Waitall, test() → MPI_Testall
    - Destructor calls MPI_Request_free on all persistent requests (RAII)
    - Move-only semantics (deleted copy, move nullifies source)
    - Handle pack/unpack for device views (pack before start, unpack after wait)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 4.2 Write tests for persistent communication
    - Test start/wait cycle completes and delivers correct data (ring topology, 4 ranks)
    - Test multiple start/wait cycles reuse the same persistent requests
    - Test destruction after start (before wait) safely frees requests
    - Test move semantics (source becomes empty)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 5. Diagnostics and instrumentation
  - [x] 5.1 Implement diagnostics hook interface
    - Create `include/halo/diagnostics.hpp` with Exchange_Event struct and Diagnostics class
    - Implement set_callback / clear_callback (static, thread-safe)
    - Implement emit() that invokes the callback when set, no-op when null
    - Instrument existing exchange_blocking and exchange_async with begin/end events
    - Instrument structured exchange functions
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 5.2 Enhance error messages with rank and plan context
    - Update detail::throw_mpi_error / handle_mpi_error to include local rank, comm name, and plan neighbor summary
    - Add MPI_Comm_get_name query when formatting errors
    - Update Fortran interop error path to write context to stderr before abort
    - _Requirements: 11.1, 11.2, 11.3_

  - [-] 5.3 Write tests for diagnostics
    - Register a callback, perform an exchange, verify events received with correct fields
    - Verify no callback registered → no overhead (timing-based sanity check)
    - Verify thread safety of set_callback / emit under concurrent exchanges
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 6. Neighbor collectives (optional exchange path)
  - [ ] 6.1 Implement topology-aware neighbor collective exchange
    - Create `exchange_neighbor_collective<ViewType>` in exchange_structured.hpp
    - Build MPI_Dist_graph_create_adjacent from the plan's neighbor topology
    - Cache the topology communicator in the Structured_Halo_Plan
    - Call MPI_Neighbor_alltoallv for the exchange
    - Fall back to Isend/Irecv when topology is asymmetric or MPI < 3.0
    - _Requirements: 13.1, 13.2, 13.3_

  - [ ] 6.2 Write tests for neighbor collective exchange
    - Test symmetric 2D periodic grid via neighbor collective path
    - Test asymmetric topology falls back to Isend/Irecv transparently
    - Verify data correctness matches the standard exchange path
    - _Requirements: 13.1, 13.2, 13.3_

- [ ] 7. Spack packaging
  - [ ] 7.1 Create Spack package.py
    - Create `spack/package.py` with HaloPackage class
    - Declare dependencies: mpi, kokkos, and optional googletest/rapidcheck
    - Implement variants: +fortran (default on), +gpu_aware_mpi (default off), +tests (default off)
    - Map Kokkos backend variants (openmp, cuda, hip, serial)
    - Implement cmake_args() passing the correct HALO CMake options
    - Test with `spack install halo` and `spack test run halo` documentation
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 8. Documentation
  - [ ] 8.1 Write communication/computation overlap guide
    - Create `docs/overlap_pattern.md` with interior/halo decomposition examples
    - Show async exchange + interior computation + wait pattern with code
    - Document buffer lifetime requirements for the async window
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 8.2 Write thread safety documentation
    - Create `docs/thread_safety.md` documenting the thread model
    - Document MPI_THREAD_MULTIPLE behavior (no serialization, concurrent exchanges safe)
    - Document MPI_THREAD_SERIALIZED behavior (exchanges must be from serial context)
    - Document the optional communication thread mode (if implemented)
    - _Requirements: 9.1, 9.2, 9.3_

  - [ ] 8.3 Update README with production features
    - Add structured exchange API section with code examples
    - Add persistent communication section
    - Add Spack installation instructions
    - Add diagnostics hook usage example
    - Update prerequisites table with new optional dependencies
    - _Requirements: 5.4, 8.1_

- [ ] 9. Model adapter examples
  - [ ] 9.1 Create FV3 cubed-sphere adapter example
    - Create `examples/fv3_adapter.F90` showing ESMF_FieldHalo replacement with halo_mod
    - Include build instructions (standalone CMakeLists using find_package(HALO))
    - Document the mapping from ESMF RouteHandle to Halo_Plan
    - _Requirements: 14.1_

  - [ ] 9.2 Create MPAS unstructured mesh adapter example
    - Create `examples/mpas_adapter.cpp` showing cell halo exchange with Kokkos views
    - Include build instructions (standalone CMakeLists using find_package(HALO))
    - Document the mapping from MPAS exchange_halo to HALO's flat-buffer API
    - _Requirements: 14.2_

- [ ] 10. Integration testing and CI updates
  - [ ] 10.1 Write structured exchange property tests
    - Property: structured exchange round-trip preserves halo data (random grid sizes, random halo widths)
    - Property: pack then unpack is identity (for any subview shape)
    - Property: persistent start/wait delivers same data as non-persistent exchange
    - Run with MPI spy for the property tests, mpirun for integration tests
    - _Requirements: 1.1, 2.1, 7.1_

  - [ ] 10.2 Update CI pipeline for new features
    - Add structured exchange tests to the CTest suite (mpirun -np 4)
    - Add Spack build smoke test to CI
    - Update docs/CI_PIPELINE.md with new stages
    - _Requirements: all_

  - [ ] 10.3 Final checkpoint — production hardening complete
    - Full test suite green (existing 22 + new structured/persistent/diagnostics tests)
    - Spack package builds
    - All documentation reviewed
    - README accurate

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.1"] },
    { "id": 4, "tasks": ["3.4", "4.2", "5.1", "5.2"] },
    { "id": 5, "tasks": ["5.3", "6.1"] },
    { "id": 6, "tasks": ["6.2", "7.1"] },
    { "id": 7, "tasks": ["8.1", "8.2", "8.3", "9.1", "9.2"] },
    { "id": 8, "tasks": ["10.1", "10.2"] },
    { "id": 9, "tasks": ["10.3"] }
  ]
}
```

## Notes

- All existing tests (22/22) must continue to pass throughout — no regressions allowed
- The existing flat-buffer exchange API is NOT deprecated; it remains the low-level primitive
- Structured exchange is built ON TOP of the flat-buffer API (it packs → exchanges → unpacks)
- Persistent communication is an optimization for the repeated-exchange-same-buffer pattern
- GPU pack/unpack kernels are the performance-critical path — benchmark them
- Spack package.py can be contributed upstream to spack/spack once validated
- Model adapter examples are documentation artifacts, not compiled library code
