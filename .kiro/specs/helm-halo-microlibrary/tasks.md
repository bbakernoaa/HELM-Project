# Implementation Plan: HALO (Hardware-Abstracted Link Operations)

## Overview

Incremental implementation of the HALO Tier 1 C++20 micro-library providing RAII-wrapped MPI communication primitives with GPU-aware halo exchange via Kokkos, plus a Fortran C-interop layer for legacy NUOPC/ESMF model adoption. Tasks are ordered to build foundational RAII wrappers first, then layer exchange logic, GPU dispatch, Fortran interop, and finally CI verification.

## Tasks

- [x] 1. Repository scaffolding and CMake build system
  - [x] 1.1 Create repository directory structure and CMakeLists.txt
    - Create `libs/halo/` directory with subdirectories: `include/halo/`, `include/halo/detail/`, `src/`, `src/detail/`, `src/fortran/`, `fortran/`, `tests/`, `tests_fortran/`, `cmake/`
    - Write root `CMakeLists.txt` with C++20 requirement, `find_package(MPI REQUIRED COMPONENTS CXX)`, `find_package(Kokkos REQUIRED)`, `HALO_GPU_AWARE_MPI` option, `BUILD_TESTING` option (default OFF), `BUILD_FORTRAN` option (default ON)
    - Create `HELM::HALO` alias target linking MPI::MPI_CXX and Kokkos::kokkos as PUBLIC dependencies
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8, 10.9_

  - [x] 1.2 Create CMake export and install configuration
    - Write `cmake/HALOConfig.cmake.in` and `cmake/HALOConfigVersion.cmake.in` templates
    - Configure `install(EXPORT HALOTargets ...)` with `NAMESPACE HELM::` and `SameMajorVersion` compatibility
    - Ensure downstream `find_package(HALO)` works with transitive MPI and Kokkos dependencies
    - _Requirements: 10.5_

  - [x] 1.3 Create test infrastructure CMakeLists.txt
    - Write `tests/CMakeLists.txt` that locates GTest when `BUILD_TESTING=ON`
    - Define test executable targets for unit tests and property tests (RapidCheck)
    - Configure `mpirun -np 4` test runner for MPI-based tests
    - _Requirements: 10.6, 10.7_

  - [x] 1.4 Create README and .gitignore
    - Write `libs/halo/README.md` documenting prerequisites, Docker container launch, CMake configure/build commands, and test execution
    - Write `libs/halo/.gitignore` for build artifacts
    - _Requirements: 12.4_

- [x] 2. RAII Communicator wrapper
  - [x] 2.1 Implement halo::Communicator class
    - Create `include/halo/communicator.hpp` with the full class declaration (constructor, destructor, move ops, deleted copy ops, handle accessor, rank, size, split, duplicate)
    - Create `src/communicator.cpp` implementing: ownership semantics, predefined communicator detection (MPI_COMM_WORLD, MPI_COMM_SELF), MPI_Comm_free in destructor with MPI-finalized check, move nullification
    - Ensure `handle()` is `noexcept` and returns by value
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 2.2 Write property test: RAII Construction Round-Trip (Communicator)
    - **Property 1: RAII Construction Round-Trip**
    - Generate arbitrary MPI_Comm handle values, construct Communicator, verify `handle()` returns original value
    - **Validates: Requirements 1.1, 1.5**

  - [x] 2.3 Write property test: Move Semantics Nullify Source (Communicator)
    - **Property 2: Move Semantics Nullify Source**
    - Generate non-null Communicator, move-construct/move-assign, verify source holds MPI_COMM_NULL and destination holds original
    - **Validates: Requirements 1.3**

  - [x] 2.4 Write property test: Communicator Destructor Cleanup
    - **Property 3: Communicator Destructor Cleanup**
    - Using MPI interposition spy, verify MPI_Comm_free called exactly once for non-predefined, non-null handles on destruction
    - **Validates: Requirements 1.2, 1.6**

- [x] 3. RAII Request_Guard wrapper
  - [x] 3.1 Implement halo::Request_Guard class
    - Create `include/halo/request_guard.hpp` with full class declaration (constructor from MPI_Request&, destructor with unwinding detection, move ops, deleted copy ops, test, wait, handle accessor)
    - Create `src/request_guard.cpp` implementing: normal destruction via MPI_Wait, unwinding destruction via MPI_Cancel + MPI_Request_free, `std::uncaught_exceptions()` snapshot at construction
    - Ensure `handle()` returns `MPI_Request*` and is `noexcept`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 8.2_

  - [x] 3.2 Write property test: Request_Guard Normal Destruction Completes Operation
    - **Property 4: Request_Guard Normal Destruction Completes Operation**
    - Using MPI spy, verify MPI_Wait called exactly once on non-null request during normal scope exit
    - **Validates: Requirements 2.2**

  - [x] 3.3 Write property test: Request_Guard Unwinding Destruction Cancels Operation
    - **Property 5: Request_Guard Unwinding Destruction Cancels Operation**
    - Using MPI spy, throw inside scope containing Request_Guard, verify MPI_Cancel then MPI_Request_free called in order
    - **Validates: Requirements 2.7**

- [x] 4. RAII Window_Guard wrapper
  - [x] 4.1 Implement halo::Window_Guard class
    - Create `include/halo/window_guard.hpp` with full class declaration (constructor, noexcept destructor, move ops, deleted copy ops, handle accessor, set_epoch_active)
    - Create `src/window_guard.cpp` implementing: MPI_Win_fence before MPI_Win_free when epoch active, MPI_Win_free on non-null/non-finalized, error swallowing in destructor
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 4.2 Write property test: Window_Guard Fence-Before-Free on Active Epoch
    - **Property 6: Window_Guard Fence-Before-Free on Active Epoch**
    - Using MPI spy, set epoch_active=true, destroy Window_Guard, verify MPI_Win_fence(0) called before MPI_Win_free
    - **Validates: Requirements 3.7**

  - [x] 4.3 Write property test: Window_Guard Destructor Never Throws
    - **Property 7: Window_Guard Destructor Never Throws**
    - Inject MPI errors via spy, verify destructor completes without throwing (static_assert noexcept or catch verification)
    - **Validates: Requirements 3.8**

- [x] 5. Checkpoint — RAII wrappers complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. MPI interposition test layer
  - [x] 6.1 Implement MPI_Spy interposition layer
    - Create `tests/mpi_interposition.hpp` with `MPI_Spy` singleton: call recording (Comm_free, Request_free, Cancel, Wait, Test, Win_free, Win_fence, Irecv, Isend, Waitall), error injection, reset
    - Implement weak-symbol overrides or link-time interposition for target MPI functions
    - Provide `MPI_Call_Record` struct with type enum, handle pointer, and argument fields
    - _Requirements: 11.7_

  - [x] 6.2 Write unit tests for MPI_Spy layer
    - Verify call recording works for each intercepted MPI function
    - Verify error injection returns configured error codes
    - Verify reset clears all recorded calls
    - _Requirements: 11.7_

- [x] 7. Environment singleton and thread safety
  - [x] 7.1 Implement halo::Environment class
    - Create `include/halo/environment.hpp` with static initialize(), thread_support_level(), is_thread_multiple()
    - Create `src/environment.cpp` implementing: MPI_Initialized check (throw if not), MPI_Query_thread, std::once_flag for idempotence, internal mutex for serialization
    - Implement `detail::Serialized_MPI_Guard` RAII lock that activates only when thread level < MPI_THREAD_MULTIPLE
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 7.2 Write property test: Environment Initialization Idempotence
    - **Property 20: Environment Initialization Idempotence**
    - Call initialize() N times, verify thread_support_level() returns same value after each call
    - **Validates: Requirements 9.6**

- [x] 8. Halo_Plan precomputation
  - [x] 8.1 Implement halo::Halo_Plan class
    - Create `include/halo/halo_plan.hpp` with Neighbor_Info struct and Halo_Plan class declaration
    - Create `src/halo_plan.cpp` implementing: construction with rank validation (range check, duplicate check), immutable storage of send/recv neighbor lists, num_send_neighbors, num_recv_neighbors, send_info, recv_info, total_send_elements, total_recv_elements
    - Ensure copyable and movable (defaulted special members)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 8.2 Write property test: Halo_Plan Construction Round-Trip
    - **Property 8: Halo_Plan Construction Round-Trip**
    - Generate valid neighbor lists (ranks in [0, comm_size), no duplicates), construct plan, verify send_info/recv_info match input
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [x] 8.3 Write property test: Halo_Plan Rejects Invalid Ranks
    - **Property 9: Halo_Plan Rejects Invalid Ranks**
    - Generate neighbor lists with at least one out-of-range rank, verify std::invalid_argument thrown
    - **Validates: Requirements 5.4**

  - [x] 8.4 Write property test: Halo_Plan Rejects Duplicate Ranks
    - **Property 10: Halo_Plan Rejects Duplicate Ranks**
    - Generate neighbor lists with at least one duplicate rank, verify std::invalid_argument thrown
    - **Validates: Requirements 5.9**

  - [x] 8.5 Write property test: Halo_Plan Copy Equivalence
    - **Property 11: Halo_Plan Copy Equivalence**
    - Copy-construct from valid plan, verify all accessors return equal values
    - **Validates: Requirements 5.5**

- [x] 9. Memory space traits and GPU-aware dispatch
  - [x] 9.1 Implement compile-time memory space traits
    - Create `include/halo/detail/memory_traits.hpp` with `is_device_space` trait specializations (CudaSpace, CudaUVMSpace, HIPSpace, OpenACCSpace), `is_device_space_v`, `view_memory_space_t`, `requires_staging_v`, `host_mirror_t`
    - Implement `HALO_GPU_AWARE_MPI` compile-time flag logic for `requires_staging_v`
    - _Requirements: 6.4, 6.5, 7.6, 7.7, 8.5_

  - [x] 9.2 Implement host-staging buffer utilities
    - Create `include/halo/detail/staging.hpp` and `src/detail/staging.cpp`
    - Implement `stage_send()`: deep_copy from device view to host mirror buffer
    - Implement `stage_recv()`: deep_copy from host mirror buffer back to device view
    - _Requirements: 6.5, 7.7_

- [x] 10. Blocking halo exchange
  - [x] 10.1 Implement halo::exchange_blocking function template
    - Create `include/halo/exchange.hpp` with `exchange_blocking<ViewType>` template
    - Implement: early return for empty plans, post all MPI_Irecv before any MPI_Isend, deterministic tag computation, MPI_Waitall, compile-time dispatch via `requires_staging_v`
    - Use `detail::Serialized_MPI_Guard` for thread safety
    - Throw std::runtime_error with MPI error string and failing rank on MPI errors
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x] 10.2 Write property test: MPI Tag Determinism and Boundedness
    - **Property 12: MPI Tag Determinism and Boundedness**
    - Generate random (sender, receiver, comm_size) triples, verify tag is deterministic and in [0, MPI_TAG_UB)
    - **Validates: Requirements 6.2**

  - [x] 10.3 Write property test: Exchange Posts All Receives Before Any Send
    - **Property 13: Exchange Posts All Receives Before Any Send**
    - Using MPI spy, execute exchange_blocking with non-empty plan, verify all Irecv records precede all Isend records
    - **Validates: Requirements 6.1**

- [x] 11. Non-blocking asynchronous halo exchange
  - [x] 11.1 Implement halo::Halo_Handle class
    - Create `include/halo/halo_handle.hpp` with Halo_Handle class (move-only, owns Request_Guards, staged_recv state, test/wait with post-receive deep-copy, destructor calls wait)
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.9_

  - [x] 11.2 Implement halo::exchange_async function template
    - Add `exchange_async<ViewType>` to `include/halo/exchange.hpp`
    - Implement: post all Irecv/Isend, construct Halo_Handle owning all Request_Guards, attach staged recv state when needed, return handle
    - _Requirements: 7.1, 7.6, 7.7, 7.8_

  - [x] 11.3 Write property test: Async Exchange Handle Owns Correct Request Count
    - **Property 17: Async Exchange Handle Owns Correct Request Count**
    - Using MPI spy, verify Halo_Handle owns exactly S + R Request_Guards for a plan with S sends and R receives
    - **Validates: Requirements 7.1, 7.2**

  - [x] 11.4 Write property test: Halo_Handle Destructor Ensures Completion
    - **Property 19: Halo_Handle Destructor Ensures Completion**
    - Destroy Halo_Handle without calling wait(), verify via MPI spy that all requests are completed
    - **Validates: Requirements 7.5**

- [x] 12. Umbrella header and public API finalization
  - [x] 12.1 Create umbrella header and verify Tier 1 isolation
    - Create `include/halo/halo.hpp` that includes all public headers
    - Verify no `#include` directives reference TICK, LOGS, AXIS, AMIO, SPAN, or DAGR headers
    - Verify only C++ standard library, MPI, and Kokkos headers are included
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 8.1, 8.4_

- [x] 13. Checkpoint — Core C++ library complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. RAII destructor exception-path test suite
  - [-] 14.1 Write RAII exception-path unit tests
    - Test: Communicator destroyed during exception unwinding verifies handle == MPI_COMM_NULL after catch
    - Test: Request_Guard with pending op destroyed during unwinding verifies cancel+free via spy
    - Test: Nested scopes (outer Communicator, inner Request_Guard) verify reverse destruction order
    - Test: Move-constructed Communicator source has NULL handle, source destruction is no-op
    - Test: Communicator wrapping MPI_COMM_WORLD does not call MPI_Comm_free
    - Test: Communicator via split/dup destroyed verifies handle == NULL (round-trip)
    - All tests use MPI interposition spy for deterministic verification
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [-] 14.2 Write property tests for GPU-aware dispatch paths
    - **Property 14: GPU-Aware MPI Uses Device Pointers Directly**
    - **Property 15: Non-GPU-Aware MPI Stages Through Host**
    - Verify compile-time dispatch selects correct path based on HALO_GPU_AWARE_MPI flag and view memory space
    - **Validates: Requirements 6.4, 6.5, 7.6, 7.7**

  - [-] 14.3 Write property test: MPI Errors During Exchange Throw With Context
    - **Property 16: MPI Errors During Exchange Throw With Context**
    - Inject MPI errors via spy, verify std::runtime_error thrown containing error string and failing rank
    - **Validates: Requirements 6.6, 7.8**

  - [-] 14.4 Write property test: Completion Triggers Post-Receive Deep-Copy When Staged
    - **Property 18: Completion Triggers Post-Receive Deep-Copy When Staged**
    - Verify that test()==true or wait() on staged Halo_Handle triggers deep_copy before returning
    - **Validates: Requirements 7.3, 7.4**

- [ ] 15. Fortran C-interop layer
  - [-] 15.1 Implement opaque handle registry
    - Create `src/fortran/handle_registry.hpp` with `halo::fortran::Handle_Registry` singleton
    - Implement thread-safe register_handle, lookup, release, valid methods
    - Use monotonically increasing integer tokens (0 reserved as invalid)
    - _Requirements: 14.3, 14.4, 14.13_

  - [-] 15.2 Implement extern "C" interop functions
    - Create `src/fortran/halo_c_interop.cpp` with all extern "C" functions: halo_init_c, halo_comm_create_c, halo_plan_create_c, halo_exchange_blocking_c, halo_exchange_async_c, halo_wait_c, halo_test_c, halo_destroy_plan_c, halo_destroy_comm_c
    - Implement HALO_C_TRY macro for exception-to-error-code translation
    - Use MPI_Comm_f2c for Fortran integer communicator conversion
    - Construct non-owning Kokkos::View over Fortran contiguous arrays
    - _Requirements: 14.1, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.13_

  - [ ] 15.3 Write property test: Handle Registry Round-Trip
    - **Property 21: Handle Registry Round-Trip**
    - Generate random create/destroy sequences, verify handle uniqueness and lookup correctness
    - **Validates: Requirements 14.3, 14.4, 14.16**

  - [ ] 15.4 Write property test: Exception Boundary Returns Error Code
    - **Property 22: Exception Boundary Returns Error Code**
    - Inject various exception types, verify all caught and mapped to non-zero error codes; verify success returns 0
    - **Validates: Requirements 14.9, 14.10**

  - [ ] 15.5 Write property test: Destroy Invalidates Handle
    - **Property 23: Destroy Invalidates Handle**
    - Create handle, destroy it, attempt reuse, verify HALO_ERR_BAD_HANDLE returned
    - **Validates: Requirements 14.13**

  - [ ] 15.6 Write property test: Plan Creation Validates Neighbor Arrays
    - **Property 24: Plan Creation Validates Neighbor Arrays**
    - Generate valid and invalid neighbor arrays via C interop, verify correct error codes
    - **Validates: Requirements 14.12**

  - [ ] 15.7 Write property test: Exchange Forwarding Preserves Pointer and Size
    - **Property 25: Exchange Forwarding Preserves Pointer and Size**
    - Verify non-owning view constructed over exactly num_elements * element_size bytes at provided pointer
    - **Validates: Requirements 14.5, 14.6**

- [ ] 16. Fortran halo_mod module
  - [ ] 16.1 Implement halo_mod.f90 Fortran module
    - Create `fortran/halo_mod.f90` with iso_c_binding interfaces for all C interop functions
    - Implement public subroutines: halo_init, halo_comm_create, halo_plan_create, halo_exchange_blocking, halo_exchange_async, halo_wait, halo_test, halo_destroy_plan, halo_destroy_comm
    - Define public error code constants (HALO_SUCCESS, HALO_ERR_INVALID_ARG, etc.)
    - Accept MPI communicator as integer(c_int) compatible with MPI_Comm%mpi_val and ESMF convention
    - Use c_loc for contiguous array passing, bind(c) on all interfaces
    - _Requirements: 14.2, 14.4, 14.5, 14.6, 14.7, 14.8, 14.11, 14.12, 14.15_

  - [ ] 16.2 Add Fortran build targets to CMakeLists.txt
    - Enable Fortran language, add halo_c_interop and halo_fortran library targets
    - Create HELM::HALO_Fortran alias, configure Fortran module directory
    - Add install rules for .mod files and Fortran libraries
    - Add tests_fortran subdirectory when BUILD_TESTING and BUILD_FORTRAN are ON
    - _Requirements: 14.14_

- [ ] 17. Checkpoint — Fortran interop layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. Fortran integration tests
  - [ ] 18.1 Write Fortran integration test suite
    - Create `tests_fortran/CMakeLists.txt` and `tests_fortran/test_halo_mod.f90`
    - Test full lifecycle: halo_init → halo_plan_create → halo_exchange_blocking → halo_destroy_plan → halo_destroy_comm
    - Test async path: halo_exchange_async → halo_wait
    - Test error code propagation (invalid plan handle, invalid ranks)
    - Test MPI communicator integer compatibility
    - Test contiguous array passing via c_loc
    - Run with `mpirun -np 4`
    - _Requirements: 14.2, 14.5, 14.6, 14.7, 14.8, 14.11, 14.12, 14.14_

- [ ] 19. CI pipeline and isolation compliance
  - [ ] 19.1 Create static isolation verification script
    - Write a script (shell or CMake custom target) that scans all HALO source and header files for `#include` directives matching other HELM component header paths (TICK, LOGS, AXIS, AMIO, SPAN, DAGR)
    - Fail the build if any forbidden includes are found
    - Integrate as a CMake custom target or CTest test
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [ ] 19.2 Configure CI pipeline steps
    - Document CI pipeline stages: static analysis (isolation scan), standalone CMake build inside Docker, unit tests (mpirun -np 4), property tests (single-rank mocked MPI), ASan + UBSan sanitizer builds
    - Verify standalone build produces HELM::HALO without other HELM source trees present
    - _Requirements: 12.2, 12.3, 12.5, 13.5_

- [ ] 20. Final checkpoint — All components integrated
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical boundaries
- Property tests validate universal correctness properties (Properties 1–25 from design)
- Unit tests validate specific examples, edge cases, and destruction order
- The MPI interposition spy layer (task 6.1) is a prerequisite for deterministic property testing of RAII destructors
- RapidCheck is used for C++ property-based tests with minimum 100 iterations per property
- Fortran tests require MPI with Fortran component and a Fortran 2008-compliant compiler

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "6.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "6.2", "7.1"] },
    { "id": 4, "tasks": ["3.1", "4.1", "7.2"] },
    { "id": 5, "tasks": ["3.2", "3.3", "4.2", "4.3"] },
    { "id": 6, "tasks": ["8.1", "9.1"] },
    { "id": 7, "tasks": ["8.2", "8.3", "8.4", "8.5", "9.2"] },
    { "id": 8, "tasks": ["10.1"] },
    { "id": 9, "tasks": ["10.2", "10.3", "11.1"] },
    { "id": 10, "tasks": ["11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4", "12.1"] },
    { "id": 12, "tasks": ["14.1", "14.2", "14.3", "14.4"] },
    { "id": 13, "tasks": ["15.1"] },
    { "id": 14, "tasks": ["15.2"] },
    { "id": 15, "tasks": ["15.3", "15.4", "15.5", "15.6", "15.7"] },
    { "id": 16, "tasks": ["16.1", "16.2"] },
    { "id": 17, "tasks": ["18.1"] },
    { "id": 18, "tasks": ["19.1", "19.2"] }
  ]
}
```
