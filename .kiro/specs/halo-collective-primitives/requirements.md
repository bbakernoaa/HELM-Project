# Requirements Document

## Introduction

HALO (Hardware-Abstracted Link Operations) is a Tier 1 C++20 micro-library within the HELM ecosystem that provides RAII-wrapped, GPU-aware MPI communication primitives built on Kokkos. Today HALO offers only NEIGHBOR halo-exchange primitives (point-to-point Isend/Irecv and MPI_Neighbor_alltoallv over precomputed structured plans). HALO has NO general COLLECTIVE primitives: there is no all-gather, no all-gatherv, and no all-reduce wrapper. This is the collective-primitive gap this feature closes.

Two external consumers of HALO independently need an all-gather pattern that HALO does not yet provide, so each currently hand-rolls raw MPI against the RAII Communicator. This feature introduces general-purpose collective primitives to HALO so those consumers (and any future caller) can express the pattern once, correctly, and portably.

As a Tier 1 component, HALO is completely blind to other HELM libraries (TICK, LOGS, AXIS, AMIO, SPAN, DAGR) and to all domain science code. Therefore this document frames the consumer needs generically. The primitives MUST NOT reference any consumer name, library name, or domain concept in their API. The two generic consumer patterns that MUST be served are:

- **Pattern A — Metadata all-gather with prefix-sum.** A caller gathers a small, fixed number of elements from every rank (for example, one integer per rank describing that rank's contribution size), then derives per-rank displacements from the gathered counts via a counts-to-displacements prefix sum. This is a small, latency-bound collective. The primitive MUST let the caller replace a hand-rolled `MPI_Allgather` plus a manual offset loop with a single wrapper call.

- **Pattern B — Replicated multi-level band gather with zero host reorder.** A caller assembles a REPLICATED multi-dimensional field on every rank from rank-local contiguous bands (each rank owns a contiguous slab of one axis of the field, for many levels of an outer axis). Today such a caller loops over the level axis issuing, PER LEVEL, one gather collective plus surrounding readiness/status reductions, producing a collective count that scales linearly with the level count. Profiling in the HELM development container (np=2 and np=4, a 1440x720 field, level counts swept from 1 to 72) established that: (1) fusing the small front-gate reductions is only a minor secondary win (about 1.0x, rising to only about 1.36x at np=4 with 72 levels); (2) naively batching the per-level gather into one large gather followed by a host-side reorder was 2 to 2.5 times SLOWER, and that regression was caused entirely by the host-side reorder copy and its buffer allocation, not by the collective; and (3) the isolated "one gather, no host reorder" variant was 1.07 to 1.43 times FASTER, with the advantage growing as rank count and level count increase. The measured conclusion is that the win requires collapsing many level-wise collectives into ONE collective AND eliminating the host reorder — that is, using a strided MPI datatype and/or on-device pack/unpack so that gathered data lands directly in the replicated `[level][j][i]` layout with zero host copy.

To serve both patterns with a coherent, reusable design, this feature provides a **two-tier API**:

- **Tier 1 (low-level collective wrapper).** RAII-Communicator-aware, error-policy-aware wrappers for `MPI_Allgather` and `MPI_Allgatherv` (with the counts-to-displacements prefix sum computed once inside the wrapper), plus a vector `MPI_Allreduce` wrapper. This tier directly serves Pattern A and provides the building blocks for Tier 2.

- **Tier 2 (high-level replicated-gather plan).** A precomputed, reusable `Replicated_Gather_Plan` and a `gather_replicated` operation, built on Tier 1, that performs a multi-level gather as a SINGLE all-gatherv using a strided MPI datatype (and/or on-device pack/unpack) so that all levels land in the replicated destination layout in ONE collective with NO host reorder. This tier serves Pattern B.

Both tiers reuse HALO's existing machinery: the RAII `Communicator`; `Environment` (GPU-aware-MPI and thread-level query); `Serialized_MPI_Guard` for sub-`MPI_THREAD_MULTIPLE` serialization; `ErrorPolicy` and the `handle_mpi_error` family; the `Diagnostics` hook; the `Structured_Halo_Plan` precompute-and-reuse pattern as the precedent for `Replicated_Gather_Plan`; and the `detail::memory_traits` (`requires_staging_v`), `detail::staging`, and `detail::pack_unpack` machinery for compile-time host/device dispatch. The feature MUST preserve HALO's Tier 1 independence and introduce no dependency on any other HELM library.

## Glossary

- **HALO**: Hardware-Abstracted Link Operations; the Tier 1 RAII-protected, GPU-aware MPI abstraction micro-library within the HELM ecosystem.
- **Communicator**: The existing HALO RAII wrapper around an MPI_Comm handle that manages communicator lifetime and exposes rank, size, and the raw handle.
- **Environment**: The existing HALO singleton that queries MPI thread support and GPU-aware-MPI capability at initialization and exposes `is_gpu_aware_mpi()`, `is_thread_multiple()`, `thread_support_level()`, and the active `ErrorPolicy`.
- **Serialized_MPI_Guard**: The existing HALO RAII guard that serializes MPI calls through an internal mutex when the detected thread level is below MPI_THREAD_MULTIPLE, and is a no-op otherwise.
- **ErrorPolicy**: The existing HALO enum (`throw_on_error`, `abort_with_diagnostics`) selecting how MPI failures are reported.
- **handle_mpi_error**: The existing HALO error-handling function family that reports an MPI failure according to the active ErrorPolicy (throwing `std::runtime_error` or writing diagnostics and calling MPI_Abort).
- **Diagnostics**: The existing HALO instrumentation hook that emits `Exchange_Event` records to an optional registered callback with zero overhead when no callback is registered.
- **Kokkos_View**: A Kokkos multi-dimensional array abstraction that manages memory across host and device (GPU) memory spaces.
- **Memory_Space**: A Kokkos concept representing where data physically resides (e.g., HostSpace, CudaSpace, HIPSpace).
- **Execution_Space**: A Kokkos concept representing where computation runs (e.g., Serial, OpenMP, Cuda, HIP).
- **GPU_Aware_MPI**: An MPI implementation capable of directly sending and receiving device (GPU) pointers without explicit host staging, as reported by `Environment::is_gpu_aware_mpi()`.
- **requires_staging_v**: The existing HALO compile-time trait (`detail::memory_traits`) that is true when a Kokkos_View resides in device memory and GPU-aware MPI is not compiled in, indicating host staging is required.
- **Staging**: The existing HALO `detail::staging` machinery that deep-copies device data to a host mirror before an MPI call and back to device afterward when `requires_staging_v` is true.
- **Pack_Unpack**: The existing HALO `detail::pack_unpack` machinery providing device-side Kokkos kernels that copy between strided views and contiguous buffers.
- **Structured_Halo_Plan**: The existing HALO precomputed, reusable plan for structured halo exchange, used here as the precedent pattern for precomputing and reusing collective metadata.
- **Collective_Wrapper**: The Tier 1 low-level component of this feature providing Allgather, Allgatherv, and vector Allreduce wrappers over a Communicator with ErrorPolicy-aware error handling.
- **Allgather_Result**: The output of an Allgatherv wrapper that provides both the gathered element buffer and the per-rank displacements computed from the per-rank counts.
- **Replicated_Gather_Plan**: The Tier 2 precomputed, reusable plan that describes a multi-level replicated gather: per-rank element counts, per-rank displacements, the strided MPI datatype for direct placement into the replicated layout, and the level count. Built once and reused across invocations.
- **gather_replicated**: The Tier 2 operation that executes a `Replicated_Gather_Plan` as a single Allgatherv, placing all levels into the replicated destination layout with no host reorder.
- **Replicated_Field**: A multi-dimensional destination buffer, present in identical form on every rank, holding the assembled result of a replicated gather in `[level][j][i]` (outer-level, then per-band) layout.
- **Rank_Local_Band**: A rank's contiguous contribution to a Replicated_Field: a contiguous slab of the gathered axis, for all levels the caller supplies.
- **Prefix_Sum**: The counts-to-displacements computation that converts a per-rank count array into a per-rank displacement array where displacement[0] is 0 and displacement[k] is the sum of counts[0..k-1].
- **Strided_Datatype**: A derived MPI datatype (for example, built with MPI_Type_create_hindexed or MPI_Type_vector) that describes non-contiguous receive placement so that a single Allgatherv writes each rank's band into the correct offset of every level in the Replicated_Field without a host reorder pass.
- **Readiness_Reduction**: A collective reduction (MPI_Allreduce) issued before a gather to confirm every rank is ready to participate.
- **Status_Reduction**: A collective reduction (MPI_Allreduce) issued after a gather to confirm every rank completed successfully.
- **MPI_Spy**: The existing HALO test interposition layer (`tests/mpi_interposition.hpp` / `mpi_interposition.cpp`) that records intercepted MPI calls for deterministic assertions.
- **RapidCheck**: The property-based testing library used by HALO for randomized correctness tests, configured to run at least 100 iterations per property.
- **RAII**: Resource Acquisition Is Initialization; a C++ idiom binding resource lifetime to object scope.

## Requirements

### Requirement 1: Low-Level Allgather Wrapper

**User Story:** As a library developer, I want a Communicator-aware wrapper around MPI_Allgather for a fixed number of elements per rank, so that I can gather uniform per-rank metadata without hand-rolling raw MPI against the RAII communicator.

#### Acceptance Criteria

1. WHEN the Collective_Wrapper Allgather operation is called with a Communicator, a send buffer, and a per-rank element count, THE Collective_Wrapper SHALL call MPI_Allgather on the Communicator's handle so that every rank receives the concatenation of all ranks' send buffers in ascending rank order.
2. WHEN the Allgather operation completes, THE Collective_Wrapper SHALL produce a receive buffer whose length equals the per-rank element count multiplied by the Communicator size.
3. IF the MPI_Allgather call returns a non-success error code, THEN THE Collective_Wrapper SHALL invoke handle_mpi_error with the MPI error code, the operation name, and the Communicator handle.
4. WHILE the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THE Collective_Wrapper SHALL acquire a Serialized_MPI_Guard for the duration of the MPI_Allgather call.
5. IF the per-rank element count is zero, THEN THE Collective_Wrapper SHALL produce an empty receive buffer without calling MPI_Allgather.

### Requirement 2: Low-Level Allgatherv Wrapper with Prefix-Sum Displacements

**User Story:** As a library developer, I want a wrapper around MPI_Allgatherv that computes per-rank displacements from per-rank counts once, so that I can gather variable-length per-rank contributions without maintaining a manual offset loop.

#### Acceptance Criteria

1. WHEN the Collective_Wrapper Allgatherv operation is called with a Communicator, a send buffer, and a per-rank count array of length equal to the Communicator size, THE Collective_Wrapper SHALL compute a per-rank displacement array from the per-rank counts using a Prefix_Sum where displacement[0] is 0 and displacement[k] equals the sum of counts[0] through counts[k-1].
2. WHEN the displacement array is computed, THE Collective_Wrapper SHALL call MPI_Allgatherv on the Communicator's handle using the per-rank counts and the computed displacements so that every rank receives all ranks' contributions placed at their computed displacements.
3. WHEN the Allgatherv operation completes, THE Collective_Wrapper SHALL expose an Allgather_Result containing both the gathered receive buffer and the computed per-rank displacement array.
4. IF any entry in the per-rank count array is negative, THEN THE Collective_Wrapper SHALL report an invalid-argument error identifying the offending rank index before any MPI call is issued.
5. IF the length of the per-rank count array is not equal to the Communicator size, THEN THE Collective_Wrapper SHALL report an invalid-argument error identifying the expected length before any MPI call is issued.
6. IF the MPI_Allgatherv call returns a non-success error code, THEN THE Collective_Wrapper SHALL invoke handle_mpi_error with the MPI error code, the operation name, and the Communicator handle.
7. WHILE the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THE Collective_Wrapper SHALL acquire a Serialized_MPI_Guard for the duration of the MPI_Allgatherv call.

### Requirement 3: Low-Level Vector Allreduce Wrapper

**User Story:** As a library developer, I want a wrapper around MPI_Allreduce that reduces a vector of values in a single call, so that multiple scalar reductions can be fused into one latency-bound collective.

#### Acceptance Criteria

1. WHEN the Collective_Wrapper Allreduce operation is called with a Communicator, an input buffer of one or more elements, and a reduction operation, THE Collective_Wrapper SHALL call MPI_Allreduce on the Communicator's handle once for the entire buffer.
2. WHEN the Allreduce operation completes, THE Collective_Wrapper SHALL produce an output buffer of the same length as the input buffer where each element is the reduction of the corresponding elements across all ranks.
3. THE Collective_Wrapper SHALL support at minimum the MPI_MIN, MPI_MAX, and MPI_SUM reduction operations for integer element types.
4. IF the MPI_Allreduce call returns a non-success error code, THEN THE Collective_Wrapper SHALL invoke handle_mpi_error with the MPI error code, the operation name, and the Communicator handle.
5. WHILE the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THE Collective_Wrapper SHALL acquire a Serialized_MPI_Guard for the duration of the MPI_Allreduce call.

### Requirement 4: Replicated Gather Plan Precomputation

**User Story:** As a library developer, I want a precomputed, reusable plan describing a multi-level replicated gather, so that per-rank counts, displacements, and the strided receive datatype are built once and reused across many gather invocations.

#### Acceptance Criteria

1. WHEN a Replicated_Gather_Plan is constructed with a Communicator reference, this rank's Rank_Local_Band element count for a single level, and the number of levels, THE Replicated_Gather_Plan SHALL gather every rank's per-level band count using the Allgather wrapper and store the resulting per-rank counts as immutable state.
2. WHEN the per-rank counts are known, THE Replicated_Gather_Plan SHALL compute and store the per-rank displacements using a Prefix_Sum so that each rank's band occupies a contiguous, non-overlapping region within a single level of the Replicated_Field.
3. WHEN the per-rank counts and displacements are known, THE Replicated_Gather_Plan SHALL construct and store a Strided_Datatype that describes placement of each rank's band into the correct offset of every level of the Replicated_Field so that a single Allgatherv writes all levels into the replicated layout with no host reorder.
4. THE Replicated_Gather_Plan SHALL store the number of levels and the total per-level element count (the sum of all per-rank band counts) as immutable state that cannot be modified after construction.
5. IF the supplied number of levels is zero, THEN THE Replicated_Gather_Plan SHALL report an invalid-argument error identifying the level count before constructing any MPI datatype.
6. IF this rank's Rank_Local_Band element count for a single level is negative, THEN THE Replicated_Gather_Plan SHALL report an invalid-argument error before constructing any MPI datatype.
7. WHEN a Replicated_Gather_Plan owns a Strided_Datatype, THE Replicated_Gather_Plan SHALL free that datatype exactly once on destruction, provided MPI has not been finalized, and SHALL follow RAII semantics equivalent to the Structured_Halo_Plan precedent (deleted copy operations, valid move that transfers datatype ownership and leaves the source in a state that frees nothing on destruction).
8. THE Replicated_Gather_Plan SHALL be reusable across multiple gather_replicated invocations without recomputing the per-rank counts, displacements, or Strided_Datatype.

### Requirement 5: Single-Collective Replicated Gather

**User Story:** As a domain-agnostic caller, I want to assemble a replicated multi-level field from rank-local contiguous bands in one collective with no host reorder, so that the gather cost does not scale with the number of levels.

#### Acceptance Criteria

1. WHEN gather_replicated is called with a Replicated_Gather_Plan, a source Kokkos_View holding this rank's Rank_Local_Band for all levels, and a destination Replicated_Field Kokkos_View, THE gather_replicated operation SHALL issue exactly one MPI_Allgatherv regardless of the number of levels.
2. WHEN the single MPI_Allgatherv completes, THE gather_replicated operation SHALL place every rank's band for every level into the destination Replicated_Field in `[level][j][i]` layout using the plan's Strided_Datatype, without performing any host-side reorder copy.
3. WHEN gather_replicated returns successfully, THE destination Replicated_Field SHALL contain identical contents on every rank.
4. IF the MPI_Allgatherv call returns a non-success error code, THEN THE gather_replicated operation SHALL invoke handle_mpi_error with the MPI error code, the operation name, and the Communicator handle.
5. WHILE the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THE gather_replicated operation SHALL acquire a Serialized_MPI_Guard for the duration of the MPI_Allgatherv call.
6. WHEN the Diagnostics hook has a registered callback, THE gather_replicated operation SHALL emit begin and end events describing the gather.

### Requirement 6: Compile-Time GPU-Aware Dispatch for Collectives

**User Story:** As a domain scientist, I want the collective primitives to pass device pointers directly under GPU-aware MPI and to fall back to host staging otherwise, so that the same code path runs correctly and efficiently on both host and GPU builds.

#### Acceptance Criteria

1. WHERE the source and destination Kokkos_Views reside in host-accessible memory, THE gather_replicated operation SHALL pass the views' data pointers directly to the MPI collective.
2. WHERE the Kokkos_Views reside in device memory and requires_staging_v is false for those view types, THE gather_replicated operation SHALL pass device data pointers directly to the MPI collective without staging through host memory.
3. WHERE the Kokkos_Views reside in device memory and requires_staging_v is true for those view types, THE gather_replicated operation SHALL stage the source data to a host mirror before the MPI collective and stage the received data back to the device destination after the MPI collective, reusing the Staging machinery.
4. WHERE a strided reorder on the source band is required before the collective, THE gather_replicated operation SHALL use the Pack_Unpack machinery to perform the reorder on device rather than on host.
5. THE gather_replicated operation SHALL select the host-direct, device-direct, and host-staged paths at compile time using requires_staging_v and the GPU-aware-MPI compile-time flag, without a runtime branch on memory space per element.

### Requirement 7: Non-Level-Scaling Readiness and Status Collectives

**User Story:** As a domain scientist, I want the readiness and status reductions that surround a multi-level gather to be issued once per gather rather than once per level, so that reduction overhead does not grow with the number of levels.

#### Acceptance Criteria

1. WHEN gather_replicated performs a Readiness_Reduction before the gather, THE gather_replicated operation SHALL issue at most one Readiness_Reduction per gather invocation regardless of the number of levels.
2. WHEN gather_replicated performs a Status_Reduction after the gather, THE gather_replicated operation SHALL issue at most one Status_Reduction per gather invocation regardless of the number of levels.
3. WHERE a caller must confirm multiple integer agreement checks before a gather, THE Collective_Wrapper SHALL allow those checks to be fused into a single vector Allreduce using the Requirement 3 wrapper.
4. THE total number of MPI_Allreduce calls issued by a single gather_replicated invocation SHALL be independent of the number of levels.

### Requirement 8: RAII, Thread-Safety, and Error-Policy Consistency

**User Story:** As a library developer, I want the collective primitives to follow the same RAII, thread-safety, and error-policy conventions as the rest of HALO, so that the new code is consistent and leak-free.

#### Acceptance Criteria

1. THE Collective_Wrapper and gather_replicated operations SHALL report all MPI failures through handle_mpi_error so that the active ErrorPolicy (throw_on_error or abort_with_diagnostics) governs the outcome.
2. THE Replicated_Gather_Plan SHALL own any derived MPI datatype it creates and SHALL release that resource exactly once on destruction following RAII semantics, provided MPI has not been finalized.
3. WHILE the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THE collective primitives SHALL serialize their MPI calls through a Serialized_MPI_Guard.
4. IF the detected MPI thread support level is MPI_THREAD_MULTIPLE, THEN THE collective primitives SHALL issue their MPI calls without acquiring the serialization mutex.
5. THE Replicated_Gather_Plan SHALL delete copy construction and copy assignment and SHALL provide move construction and move assignment that transfer datatype ownership and leave the source in a state that frees no MPI resource on destruction.
6. THE collective primitives SHALL reference a Communicator without taking ownership of it, requiring the Communicator to outlive the primitive that references it, consistent with the Structured_Halo_Plan precedent.

### Requirement 9: Tier 1 Isolation Compliance for Collective Primitives

**User Story:** As an architect, I want the collective primitives to preserve HALO's Tier 1 independence, so that adding collectives introduces no dependency on any other HELM library or domain concept.

#### Acceptance Criteria

1. THE collective-primitive source files, public headers, and internal headers SHALL NOT include any header from TICK, LOGS, AXIS, AMIO, SPAN, or DAGR.
2. THE collective-primitive public headers SHALL only contain `#include` directives referencing C++ standard library headers, MPI headers, Kokkos headers, and other HALO headers.
3. THE collective-primitive public API names, parameters, and documentation SHALL NOT reference any consumer name, other HELM library name, or domain science concept, and SHALL express the served patterns in generic terms (per-rank counts, rank-local bands, replicated fields, levels).
4. THE collective-primitive CMake configuration SHALL NOT reference any `HELM::` namespace target other than `HELM::HALO` in target_link_libraries, add_dependencies, or find_package directives.
5. WHEN the HALO Tier 1 isolation verification step runs, THE build SHALL fail if any collective-primitive source or header includes another HELM component header path.

### Requirement 10: MPI-Spy Collective-Count Verification

**User Story:** As a library developer, I want an MPI-spy test that counts collective calls, so that I can prove a multi-level gather issues exactly one Allgatherv and a reduction count that does not scale with the number of levels.

#### Acceptance Criteria

1. THE test suite SHALL contain an MPI_Spy test that records every MPI_Allgatherv call issued during a gather_replicated invocation and asserts that exactly one MPI_Allgatherv is recorded regardless of the number of levels.
2. THE test suite SHALL contain an MPI_Spy test that records every MPI_Allreduce call issued during a gather_replicated invocation for at least two different level counts and asserts that the recorded MPI_Allreduce count is identical across those level counts.
3. THE test suite SHALL contain an MPI_Spy test that records MPI_Allgatherv and MPI_Allreduce calls issued by the low-level Allgatherv and Allreduce wrappers and asserts that each wrapper issues exactly one corresponding MPI collective per call.
4. THE MPI_Spy collective-count tests SHALL run under mpirun with process counts of at least 2 and up to 4 within the HELM development container.
5. THE MPI_Spy collective-count tests SHALL use the existing HALO MPI interposition layer without requiring changes to production collective code to enable interception.

### Requirement 11: Numerical-Equivalence Property Test

**User Story:** As a library developer, I want a property test proving the batched replicated gather output equals a naive per-level, per-record reference, so that the strided single-collective optimization is guaranteed correct.

#### Acceptance Criteria

1. THE test suite SHALL contain a RapidCheck property test that, for randomly generated per-rank band sizes, level counts, and element values, computes a Replicated_Field via gather_replicated and a reference Replicated_Field via a naive per-level, per-rank gather.
2. WHEN the batched Replicated_Field and the reference Replicated_Field are compared, THE property test SHALL assert they are equal element-for-element (bit-for-bit for integer element types).
3. THE numerical-equivalence property test SHALL execute at least 100 randomized iterations.
4. THE numerical-equivalence property test SHALL be tagged with a comment referencing the design correctness property it validates.
5. THE numerical-equivalence property test SHALL constrain generated band sizes and level counts so that total buffer sizes remain modest enough to run within the approximately 7 GB memory budget of the HELM development container.
6. THE numerical-equivalence property test SHALL run under mpirun with process counts of at least 2 and up to 4 within the HELM development container.

### Requirement 12: Build and Test Integration

**User Story:** As a build engineer, I want the collective primitives and their tests to build and run within the existing HALO build and container workflow, so that they are exercised in CI without a separate toolchain.

#### Acceptance Criteria

1. THE collective-primitive sources SHALL be compiled into the existing `halo` library target under the `HELM::HALO` namespace alias without introducing a new library target.
2. WHERE the HALO BUILD_TESTING option is ON, THE build SHALL compile the collective-primitive Google Test and RapidCheck tests and register them with CTest.
3. THE collective-primitive tests SHALL build and run inside the HELM development container using the container's pinned GCC-13, OpenMPI, Kokkos, and RapidCheck toolchain.
4. THE collective-primitive MPI tests SHALL be registered so that CTest launches them under mpirun with the process counts required by Requirements 10 and 11.
5. THE collective-primitive tests SHALL complete a single CTest run to completion without requiring any watch mode or long-running process.
