# Implementation Plan: HALO Collective Primitives

## Overview

Incremental, test-driven implementation of HALO's two-tier collective primitives,
built bottom-up on the existing Tier 1 machinery (RAII `Communicator`,
`Environment`, `Serialized_MPI_Guard`, `handle_mpi_error`, `Diagnostics`,
`requires_staging_v`, `staging`, `pack_unpack`). Tier 1 supplies thin
`allgather` / `allgatherv` / `allreduce` wrappers (Pattern A); Tier 2 supplies a
precomputed `Replicated_Gather_Plan` + `gather_replicated` that assembles a
replicated multi-level field in ONE `MPI_Allgatherv` via a derived strided
datatype with zero host reorder (Pattern B). Each primitive is written, then
tested (GTest/MPI, MPI-spy, RapidCheck), then built and run once in the cece-dev
container. Structural checkpoints build the full existing HALO suite to prove no
regression as new pieces land.

**Build/test environment (applies to EVERY build/test/checkpoint task):** All
builds and tests run ONLY inside the cece-dev/HELM container via the repo-root
wrapper `./setup.sh -c "..."` with bounded `-j2` parallelism, from the CECE repo
root (which includes the HALO submodule), with `-DHALO_BUILD_TESTING=ON`. `ctest`
runs ONCE to completion (no watch mode). Known-benign container ctest noise
(unrelated `pbt.*` "(Not Run)" stale AMIO registrations and the python-venv
fail/Not-Run items) is NOT a regression and must be distinguished from real C++
build/assertion failures. Keep buffers modest for the ~7 GB container.

## Tasks

- [x] 1. Baseline structural checkpoint — existing HALO suite green before any change
  - Build the existing HALO tests in the container:
    `./setup.sh -c "cmake -S /work -B /work/build -DHALO_BUILD_TESTING=ON && cmake --build /work/build -j2"`
  - Run the existing HALO ctest suite ONCE:
    `./setup.sh -c "ctest --test-dir /work/build --output-on-failure"`
  - Record the existing HALO guards green as the pre-change baseline; note and set
    aside the known-benign `pbt.*`/python-venv noise so later checkpoints can
    distinguish real failures from it
  - _Requirements: 12.2, 12.3, 12.5_

- [x] 2. Tier 1 low-level collective wrappers
  - [x] 2.1 Implement non-template helpers in `src/collectives.cpp`
    - Implement `detail::prefix_sum(counts)` (displ[0]=0, displ[k]=sum(counts[0..k-1])) — the single shared prefix-sum used by `allgatherv` and the plan
    - Implement `detail::validate_counts(counts, comm_size)` throwing `std::invalid_argument` BEFORE any MPI call: wrong length names the expected length; any negative entry names the offending rank index
    - Add `src/collectives.cpp` to the existing `halo` library target (no new target)
    - _Requirements: 2.1, 2.4, 2.5, 12.1_

  - [x] 2.2 Implement `allgather<T>()` in `include/halo/collectives.hpp`
    - Header-only template: acquire `detail::Serialized_MPI_Guard`, resolve `detail::mpi_datatype_for<T>()`, call `MPI_Allgather` on `comm.handle()`, produce a receive buffer of length `count_per_rank * comm.size()` in ascending-rank order
    - Zero-count fast path: when `count_per_rank == 0`, return an empty buffer WITHOUT calling `MPI_Allgather`
    - Route non-success return codes through `detail::handle_mpi_error(rc, rank, "MPI_Allgather", comm.handle())`
    - Add the `std::vector<T>` convenience overload
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.3 Implement `allgatherv<T>()` and `Allgather_Result<T>` in `include/halo/collectives.hpp`
    - Define `struct Allgather_Result<T> { std::vector<T> buffer; std::vector<int> displacements; }`
    - `allgatherv<T>()`: call `detail::validate_counts` then `detail::prefix_sum` BEFORE any MPI; size the buffer to `displs.back() + counts.back()`; store displacements into the result; acquire `Serialized_MPI_Guard`; issue exactly one `MPI_Allgatherv` using counts + computed displs; route failures through `detail::handle_mpi_error(rc, rank, "MPI_Allgatherv", comm.handle())`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 2.4 Implement `allreduce<T>()` in `include/halo/collectives.hpp`
    - Header-only template: acquire `Serialized_MPI_Guard`, resolve `detail::mpi_datatype_for<T>()`, issue exactly one `MPI_Allreduce` over the whole input buffer with the caller-supplied `MPI_Op`, return an output vector of equal length that is the elementwise reduction across ranks
    - Support at minimum `MPI_MIN`, `MPI_MAX`, `MPI_SUM` for integer element types
    - Route failures through `detail::handle_mpi_error(rc, rank, "MPI_Allreduce", comm.handle())`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.5 Add the Tier 1 header to the umbrella and build in the container
    - Add `#include <halo/collectives.hpp>` to `include/halo/halo.hpp`
    - Build in the container: `./setup.sh -c "cmake -S /work -B /work/build -DHALO_BUILD_TESTING=ON && cmake --build /work/build -j2"` and confirm `collectives.cpp` compiles into `halo` with no errors
    - _Requirements: 9.2, 12.1_

- [x] 3. RAII strided-datatype wrapper
  - [x] 3.1 Implement `detail::Strided_Datatype` in `include/halo/detail/strided_datatype.hpp`
    - Move-only RAII wrapper over `MPI_Datatype`: default holds `MPI_DATATYPE_NULL`; explicit ctor takes ownership; deleted copy; move ctor/assign transfer ownership and leave the source holding `MPI_DATATYPE_NULL` (frees nothing on later destruction)
    - Destructor frees the datatype exactly once, guarded by `MPI_Finalized` (skip free when finalized); `get()` accessor is `noexcept`
    - Include only `<mpi.h>` (no other HELM headers)
    - _Requirements: 4.7, 8.2, 8.5_

- [x] 4. Tier 2 replicated-gather plan
  - [x] 4.1 Implement `Replicated_Gather_Plan<T>` in `include/halo/replicated_gather_plan.hpp`
    - Header-only template holding a non-owning `const Communicator*` (must outlive the plan), `num_levels_`, `per_level_total_`, `band_counts_`, `displacements_`, and a `detail::Strided_Datatype recv_type_`
    - Ctor: validate BEFORE any MPI/datatype creation — `num_levels == 0` throws `std::invalid_argument` naming the level count; `local_band_count < 0` throws `std::invalid_argument`
    - Ctor steps: `allgather` every rank's per-level band count into `band_counts_`; `detail::prefix_sum` into `displacements_`; compute `per_level_total_`; build + commit the resized strided receive datatype per the design Data Models "crux" (`MPI_Type_vector` over `num_levels_` blocks, stride `per_level_total_`, base `mpi_datatype_for<T>()`, then `MPI_Type_create_resized` to a 1-element extent, `MPI_Type_commit`, free the intermediate) and store it in `recv_type_`
    - Store level count and per-level total as immutable state; provide const `noexcept` accessors (`num_levels`, `per_level_total`, `band_counts`, `displacements`, `recv_type`, `communicator`)
    - Deleted copy; move ctor/assign transfer datatype ownership; destructor frees exactly once via the `Strided_Datatype` member (RAII, `MPI_Finalized`-guarded)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 8.6_

  - [x] 4.2 Add the plan header to the umbrella
    - Add `#include <halo/replicated_gather_plan.hpp>` to `include/halo/halo.hpp`
    - _Requirements: 9.2_

- [x] 5. Compile-time collective dispatch
  - [x] 5.1 Implement `detail::collective_dispatch` in `include/halo/detail/collective_dispatch.hpp`
    - Compile-time host-direct / device-direct / host-staged selection driven by `requires_staging_v<SrcView>` / `requires_staging_v<DstView>` (which fold in the GPU-aware-MPI flag), using `if constexpr` — NO per-element runtime memory-space branch
    - `requires_staging_v == false` (host view, or device view + GPU-aware MPI): pass `view.data()` directly to MPI
    - `requires_staging_v == true` (device view, no GPU-aware MPI): use `detail::pack` for any required on-device strided source reorder into a contiguous device buffer, `detail::stage_send` to a host mirror, run the collective on host memory, `detail::stage_recv` back to device (`prepare` / `finalize` hooks)
    - Include only HALO detail headers, Kokkos, MPI, and standard-library headers
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6. Tier 2 single-collective replicated gather
  - [x] 6.1 Implement `gather_replicated<T, SrcV, DstV>()` in `include/halo/gather_replicated.hpp`
    - Emit `Diagnostics` begin event when a callback is registered; acquire `Serialized_MPI_Guard`
    - Issue at most ONE readiness `allreduce` before and ONE status `allreduce` after the gather, each a single vector reduce whose count is independent of `num_levels`
    - Use `detail::collective_dispatch::prepare` to obtain src/dest pointers by the compile-time-selected path; issue exactly ONE `MPI_Allgatherv` using the plan's `band_counts_` (recvcounts), `displacements_` (rdispls), and `recv_type()` so all levels land in `[level][j][i]` with no host reorder
    - Route failures through `detail::handle_mpi_error(rc, rank, "MPI_Allgatherv", comm.handle())`; `collective_dispatch::finalize` (host-staged path only stages received data back to device); emit `Diagnostics` end event
    - Result field is identical on every rank
    - Add `#include <halo/gather_replicated.hpp>` to `include/halo/halo.hpp`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 9.2_

- [x] 7. Structural checkpoint — full existing HALO suite with new code compiled in
  - Rebuild everything in the container:
    `./setup.sh -c "cmake -S /work -B /work/build -DHALO_BUILD_TESTING=ON && cmake --build /work/build -j2"`
  - Run the full existing HALO ctest suite ONCE:
    `./setup.sh -c "ctest --test-dir /work/build --output-on-failure"`
  - Confirm NO regression now that the new (not-yet-covered-by-suite) collective/plan/dispatch code compiles into `halo`; distinguish real C++ failures from the known-benign `pbt.*`/python-venv noise
  - _Requirements: 12.1, 12.2, 12.3, 12.5_

- [x] 8. Tier 1 and Tier 2 tests
  - [x] 8.1 Write `tests/test_collectives.cpp` (GTest, MPI) — Tier 1 + Pattern A
    - Pattern A end-to-end: `allgather` of one int per rank, then `allgatherv` with per-rank counts; assert concatenation, length `count*size`, and returned displacements
    - Zero-count `allgather` returns empty; `allreduce` MIN/MAX/SUM elementwise correctness; host-path correctness
    - Invalid-argument guards: wrong-length counts and negative count both throw `std::invalid_argument` before any MPI call
    - _Requirements: 1.1, 1.2, 1.5, 2.1, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3_

  - [x] 8.2 Wire `test_collectives` into `tests/CMakeLists.txt` and build+run in the container
    - Register with the REAL helper: `halo_add_mpi_test(test_collectives SOURCES test_collectives.cpp LIBRARIES Kokkos::kokkos)` (gated by `if(EXISTS ...)`), launched at the default np4 (`HALO_MPI_TEST_NUMPROCS`)
    - Build (`-j2`) and run `ctest` ONCE in the container; confirm `test_collectives` passes and distinguish benign noise
    - _Requirements: 12.2, 12.3, 12.4, 12.5_

  - [x] 8.3 Extend the MPI-spy interposition (TEST-ONLY) for collectives
    - In `tests/mpi_interposition.hpp` / `mpi_interposition.cpp`, EXTEND `halo::testing::MPI_Call_Record::Type` with `Allgather`, `Allgatherv`, `Allreduce` variants and add interposition that intercepts and records those collectives
    - This change is confined to the test/interposition layer — production collective code is NOT modified to enable interception
    - _Requirements: 10.5_

  - [x] 8.4 Write `tests/test_collective_spy.cpp` (MPI-spy count tests)
    - **Property 7:** exactly one `MPI_Allgatherv` recorded per `gather_replicated`, swept over several level counts
    - **Property 8:** recorded `MPI_Allreduce` count identical across two different level counts
    - **Property 9:** `allgatherv` records exactly one `Allgatherv`; `allreduce` records exactly one `Allreduce`
    - **Property 10:** zero-count `allgather` records no `Allgather`
    - **Validates: Requirements 10.1, 10.2, 10.3, 1.5, 5.1, 7.1, 7.2, 7.4**

  - [x] 8.5 Wire `test_collective_spy` with np4 AND explicit np2, then build+run in the container
    - Register with the REAL helper `halo_add_mpi_test(test_collective_spy SOURCES test_collective_spy.cpp LIBRARIES halo_mpi_spy Kokkos::kokkos)` for the default np4; since `halo_add_mpi_test` has NO np-override arg, ALSO add an explicit second registration `add_test(NAME test_collective_spy_np2 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 2 $<TARGET_FILE:test_collective_spy>)` with matching test properties, so both np2 and up-to-np4 are covered
    - Build (`-j2`) and run `ctest` ONCE in the container; confirm both np2 and np4 spy tests pass
    - _Requirements: 10.4, 12.2, 12.3, 12.4, 12.5_

  - [x] 8.6 Write `tests/test_gather_replicated.cpp` (GTest, MPI) — Tier 2 end-to-end
    - Host-path `gather_replicated` on a small `[level][j]` field compared element-for-element to a naive per-level gather reference; assert identical result on every rank
    - Compile-time dispatch coverage via `static_assert`: host views select the direct path; device views under `requires_staging_v` select staging
    - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.5_

  - [x] 8.7 Wire `test_gather_replicated` into CMake and build+run in the container
    - Register with the REAL helper: `halo_add_mpi_test(test_gather_replicated SOURCES test_gather_replicated.cpp LIBRARIES Kokkos::kokkos)` (gated by `if(EXISTS ...)`), default np4
    - Build (`-j2`) and run `ctest` ONCE in the container; confirm the test passes
    - _Requirements: 12.2, 12.3, 12.4, 12.5_

  - [x] 8.8 Write `tests/prop_gather_replicated.cpp` (RapidCheck, MPI)
    - **Property 6:** batched `gather_replicated` equals naive per-level/per-rank reference, bit-for-bit for integer elements, identical on every rank
    - Also exercise Property 2 (prefix-sum reconstructs counts), Property 4 (elementwise vector reduce), Property 5 (bands tile each level with no gaps/overlaps), and the zero-count edge
    - ≥100 iterations; each test tagged `// Feature: halo-collective-primitives, Property N: <text>`; budget-capped generators (e.g. `per_level_total` ≤ a few thousand, `num_levels` ≤ 16) to stay within the ~7 GB container
    - **Validates: Requirements 5.2, 5.3, 11.1, 11.2, 11.3, 11.4, 11.5**

  - [x] 8.9 Wire `prop_gather_replicated` with np4 AND explicit np2, then build+run in the container
    - Register with the REAL helper `halo_add_mpi_test(prop_gather_replicated SOURCES prop_gather_replicated.cpp LIBRARIES Kokkos::kokkos rapidcheck)`, link `rapidcheck_gtest` when present, `TIMEOUT 120 LABELS "mpi;property"` — default np4; since there is NO np-override arg, ALSO add `add_test(NAME prop_gather_replicated_np2 COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe -np 2 $<TARGET_FILE:prop_gather_replicated>)` so both np2 and up-to-np4 are covered
    - Build (`-j2`) and run `ctest` ONCE in the container; confirm both np2 and np4 property runs pass
    - _Requirements: 11.6, 12.2, 12.3, 12.4, 12.5_

- [x] 9. Tier 1 isolation verification
  - [x] 9.1 Add the new collective files to the HALO include-hygiene scan and confirm link/API hygiene
    - Add `include/halo/collectives.hpp`, `replicated_gather_plan.hpp`, `gather_replicated.hpp`, `detail/strided_datatype.hpp`, `detail/collective_dispatch.hpp`, and `src/collectives.cpp` to HALO's existing isolation scan file list; the build FAILS if any references a `tick/`, `logs/`, `axis/`, `amio/`, `span/`, or `dagr/` header
    - Confirm the collective sources compile into the existing `halo` target only and that no `target_link_libraries`/`add_dependencies`/`find_package` for them names any `HELM::` target other than `HELM::HALO`
    - Extend the scan's deny-list with consumer/domain tokens to enforce the generic-API rule (no consumer, HELM-library, or domain-science names in the public API)
    - Run the isolation scan in the container: `./setup.sh -c "ctest --test-dir /work/build --output-on-failure -R isolation"` (or the equivalent scan target)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 12.1_

- [x] 10. Final checkpoint — full container build + single ctest run, np2 and np4
  - Full container build:
    `./setup.sh -c "cmake -S /work -B /work/build -DHALO_BUILD_TESTING=ON && cmake --build /work/build -j2"`
  - Single ctest run:
    `./setup.sh -c "ctest --test-dir /work/build --output-on-failure"`
  - Confirm all existing HALO guards PLUS all new collective/spy/property tests pass at both np2 and np4; confirm the isolation scan passes; distinguish real C++ failures from the known-benign `pbt.*`/python-venv noise
  - _Requirements: 12.2, 12.3, 12.4, 12.5, 5.1, 7.4, 10.1, 10.2, 10.3, 10.4, 11.6_

- [x] 11. Micro-benchmark — one-collective/no-reorder win (optional)
  - [x] 11.1 Time `gather_replicated` vs a naive per-level loop and record numbers
    - Write a small container-only harness that times `gather_replicated` against a naive per-level-gather reference at np2 and np4 across an `nlev` sweep, sizing buffers modestly for the ~7 GB container
    - Build and run ONCE in the container via `./setup.sh -c "..."` with `-j2`; record the measured ratio to confirm the design's "one collective, no host reorder" advantage
    - Remove the temporary harness files afterward so the working tree stays clean
    - _Requirements: 5.1, 7.4_

## Notes

- Every build/test/checkpoint task runs ONLY in the cece-dev container via
  `./setup.sh -c "..."` with bounded `-j2` parallelism and a single `ctest` run
  (no watch mode), from the CECE repo root with `-DHALO_BUILD_TESTING=ON`.
- Tasks marked with `*` are optional and can be skipped for faster MVP; the
  micro-benchmark (11.1) is a perf-validation artifact, not library code.
- Each task references specific requirements for traceability; property tasks
  additionally cite the design's numbered Correctness Property.
- The MPI-spy interposition extension (8.3) is TEST-ONLY: the `MPI_Call_Record::Type`
  enum gains `Allgather`/`Allgatherv`/`Allreduce` variants in the interposition
  layer; production collective code is never modified to enable interception.
- `halo_add_mpi_test` has NO np-override argument. Requirements 10.4 and 11.6
  require BOTH np2 AND up-to-np4 coverage, so the spy (8.5) and property (8.9)
  tests register the default np4 target AND an explicit second `add_test`
  `... -np 2 ...` variant.
- Known-benign container ctest noise (unrelated `pbt.*` "(Not Run)" stale AMIO
  registrations and python-venv fail/Not-Run items) is NOT a regression and must
  be distinguished from real C++ build/assertion failures.
- Property generators are budget-capped (`per_level_total` and `num_levels`) to
  stay within the ~7 GB container memory budget.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2"] },
    { "id": 2, "tasks": ["2.3"] },
    { "id": 3, "tasks": ["2.4", "5.1"] },
    { "id": 4, "tasks": ["2.5"] },
    { "id": 5, "tasks": ["4.1"] },
    { "id": 6, "tasks": ["4.2"] },
    { "id": 7, "tasks": ["6.1", "8.1", "8.3"] },
    { "id": 8, "tasks": ["8.2", "8.4", "8.6", "8.8"] },
    { "id": 9, "tasks": ["8.5", "8.7", "8.9", "9.1"] },
    { "id": 10, "tasks": ["11.1"] }
  ]
}
```
