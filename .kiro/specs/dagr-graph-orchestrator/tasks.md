# Implementation Plan: DAGR (Directed Acyclic Graph Router)

## Overview

Bottom-up implementation of the DAGR micro-library — the Tier 3 YAML-driven asynchronous event-loop DAG orchestrator for the HELM ecosystem. Tasks build from enums and value types upward through the YAML parser, rank pool, event loop, transform dispatch, and finally the GraphOrchestrator facade. DAGR performs zero mathematical calculations — it routes pointers between lower-tier engines (TICK, AMIO, BLEND, HALO) and manages execution dependencies. All code lives in `libs/dagr/` producing the `HELM::DAGR` static library target.

## Tasks

- [x] 1. Repository scaffolding and CMake build system
  - [x] 1.1 Create directory structure and top-level CMakeLists.txt
    - Create `libs/dagr/` with subdirectories: `include/dagr/`, `include/dagr/detail/`, `src/`, `tests/`
    - Write `CMakeLists.txt` with `project(DAGR VERSION 0.1.0)`, `cxx_std_20`, extensions OFF
    - Define `dagr` static library target with source files: `src/graph_orchestrator.cpp`, `src/parse_pipeline.cpp`, `src/event_loop.cpp`, `src/rank_pool.cpp`, `src/transform_node.cpp`
    - Define `HELM::DAGR` alias target
    - Set `target_include_directories` with BUILD_INTERFACE and INSTALL_INTERFACE
    - Link PRIVATE dependencies: `HELM::SPAN`, `HELM::TICK`, `HELM::HALO`, `HELM::LOGS`, `HELM::BLEND`, `HELM::AMIO`, `HELM::CONF`
    - Add `BUILD_TESTING` option defaulting to OFF
    - _Requirements: 11.1, 11.5_

  - [x] 1.2 Create tests/CMakeLists.txt with GTest and RapidCheck integration
    - Add `find_package(GTest REQUIRED)` and `find_package(rapidcheck REQUIRED)` under BUILD_TESTING
    - Define test executable targets for unit tests: `test_yaml_parser`, `test_rank_pool`, `test_transform_node`
    - Define test executable targets for property tests: `prop_topological_order`, `prop_no_orphan`, `prop_rank_conservation`, `prop_acyclicity_detection`, `prop_concurrency_limit`, `prop_idempotent_shutdown`, `prop_failure_cascade`, `prop_yaml_roundtrip`, `prop_dag_construction`, `prop_invalid_enum`, `prop_missing_field`, `prop_duplicate_stream`
    - Link all test targets against `dagr`, `GTest::gtest_main`, and `rapidcheck`
    - Register tests with `gtest_discover_tests()`
    - _Requirements: 13.7, 13.8_

- [x] 2. Enums, value types, and Pipeline_Config data structures
  - [x] 2.1 Implement pipeline_config.hpp with enums, Stream_Descriptor, and Pipeline_Config
    - Define `dagr::Temporal_Profile` scoped enum with enumerators `linear = 0` and `step = 1` (underlying `std::uint8_t`)
    - Define `dagr::OutOfBounds_Policy` scoped enum with enumerators `clamp = 0` and `cycle = 1` (underlying `std::uint8_t`)
    - Define `dagr::Stream_Descriptor` struct with: `name` (std::string, max 128 chars), `temporal_profile`, `oob_policy`, `dataset_path` (std::filesystem::path), `snapshot_interval` (tick::Duration)
    - Implement `operator==` via defaulted spaceship for Stream_Descriptor
    - Define `dagr::Dependency_Edge` struct with `producer_id` and `consumer_id` (std::uint32_t)
    - Define `dagr::Pipeline_Config` struct with: `streams` (vector), `task_names` (vector), `edges` (vector), `max_concurrency` (uint32_t, default 64), `deadlock_timeout_s` (uint32_t, default 30), `shutdown_timeout_s` (uint32_t, default 30)
    - Declare `parse_pipeline` free function signature: `[[nodiscard]] Pipeline_Config parse_pipeline(const std::filesystem::path&)`
    - _Requirements: 8.1, 8.2, 8.3, 8.8, 2.12_

  - [x] 2.2 Implement detail/task_node.hpp with TaskNode and Task_Status
    - Define `dagr::detail::Task_Status` scoped enum: `pending`, `ready`, `running`, `completed`, `failed`, `cancelled`
    - Define `dagr::detail::TaskNode` struct with: `id` (uint32_t), `name` (string), `pending_deps` (atomic<uint32_t>), `status` (Task_Status), `required_ranks` (uint32_t, default 1), `dependents` (vector<uint32_t>)
    - _Requirements: 3.1, 3.2_

  - [x] 2.3 Implement detail/completion_token.hpp
    - Define `dagr::detail::Completion_Token` struct with: `node_id` (uint32_t), `dispatch_timestamp_ns` (uint64_t)
    - _Requirements: 6.2_

- [x] 3. YAML Pipeline Parser
  - [x] 3.1 Implement parse_pipeline in src/parse_pipeline.cpp
    - Use `conf::Config::from_file` to load the YAML document; allow exceptions to propagate (Req 2.2)
    - Extract stream entries: map `temporal_profile` string to enum (`linear`→Temporal_Profile::linear, `step`→Temporal_Profile::step); throw `std::invalid_argument` on unrecognized value with stream name (Req 2.6)
    - Map `out_of_bounds_policy` string to enum (`clamp`→OutOfBounds_Policy::clamp, `cycle`→OutOfBounds_Policy::cycle); throw on unrecognized (Req 2.7)
    - Extract `dataset_path` and `snapshot_interval` fields; throw if missing (Req 2.8) or if dataset_path is empty / snapshot_interval invalid (Req 8.9)
    - Validate no duplicate stream names; throw on duplicates (Req 2.9)
    - Build DAG adjacency list from task dependencies section
    - Run Kahn's topological sort to validate acyclicity; throw `std::invalid_argument` with cycle path on failure (Req 2.11)
    - Emit WARNING via LOGS before each validation throw (Req 10.7)
    - Do NOT validate dataset_path existence at parse time (Req 2.13)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 8.4, 8.5, 8.6, 8.7, 8.9, 10.7_

  - [x] 3.2 Write property test: YAML round-trip (prop_yaml_roundtrip.cpp)
    - **Property 8: YAML Parsing Round-Trip (Stream_Descriptor)**
    - Generate 1–32 valid Stream_Descriptors with unique names, valid enums, non-empty paths, positive intervals
    - Serialize to YAML temp file, parse with `parse_pipeline`, verify equality and order preservation
    - Minimum 1000 iterations
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.10, 8.4, 8.5, 8.6, 8.7, 8.8**

  - [x] 3.3 Write property test: Invalid enum rejection (prop_invalid_enum.cpp)
    - **Property 10: Invalid Enum Rejection**
    - Generate stream entries with random strings ∉ {"linear", "step"} for temporal_profile OR ∉ {"clamp", "cycle"} for out_of_bounds_policy
    - Verify `parse_pipeline` throws `std::invalid_argument` reporting stream name and invalid value
    - Minimum 1000 iterations
    - **Validates: Requirements 2.6, 2.7**

  - [x] 3.4 Write property test: Missing required field rejection (prop_missing_field.cpp)
    - **Property 11: Missing Required Field Rejection**
    - Generate stream entries with random subsets of required fields removed
    - Verify `parse_pipeline` throws `std::invalid_argument` reporting stream name and missing field
    - Minimum 1000 iterations
    - **Validates: Requirements 2.8**

  - [x] 3.5 Write property test: Duplicate stream name rejection (prop_duplicate_stream.cpp)
    - **Property 12: Duplicate Stream Name Rejection**
    - Generate pipeline configs with injected duplicate stream names
    - Verify `parse_pipeline` throws `std::invalid_argument` reporting duplicate name
    - Minimum 1000 iterations
    - **Validates: Requirements 2.9**

  - [x] 3.6 Write unit tests for parser edge cases (test_yaml_parser.cpp)
    - Test missing file → exception propagation from CONF
    - Test malformed YAML → exception propagation from CONF
    - Test empty dataset_path throws with stream name
    - Test zero/negative snapshot_interval throws with stream name
    - Test valid minimal config parses successfully
    - Test stream ordering preservation
    - _Requirements: 2.1, 2.2, 2.8, 2.10, 8.9_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Rank_Pool implementation
  - [x] 5.1 Implement detail/rank_pool.hpp and src/rank_pool.cpp
    - Implement `Rank_Pool` constructor from `std::set<int>` of rank identifiers
    - Implement atomic bitset (`vector<atomic<bool>>`) with rank_ids mapping (index → MPI rank ID)
    - Implement `try_allocate(uint32_t count)` → set<int>: atomically flip bits, return allocated rank IDs or empty set if insufficient; throw `std::invalid_argument` on count == 0 (Req 5.6)
    - Implement `release(const set<int>&)`: return ranks to pool, flip bits back
    - Implement `add_ranks(const set<int>&)`: add newly yielded ranks, increment total and available
    - Implement `remove_available(const set<int>&)` → set<int>: remove only available ranks, return what was actually removed
    - Implement const accessors: `total_ranks()`, `available_ranks()`, `allocated_ranks()`
    - Maintain invariant: `available_ + allocated == total_` at every observable point (Req 5.9)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.9_

  - [x] 5.2 Write property test: Rank conservation invariant (prop_rank_conservation.cpp)
    - **Property 3: Rank-Conservation Invariant**
    - Generate Rank_Pool with 1–64 total ranks and random scheduling sequences (allocate/release/add/remove)
    - Verify `available_ranks() + allocated_ranks() == total_ranks()` before and after every event
    - Minimum 1000 iterations
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.9, 13.3**

  - [x] 5.3 Write unit tests for Rank_Pool (test_rank_pool.cpp)
    - Test zero-rank allocation request throws `std::invalid_argument`
    - Test allocate more than available returns empty set
    - Test allocate-then-release restores availability
    - Test add_ranks increases total and available
    - Test remove_available only removes non-allocated ranks
    - Test concurrent allocation/release safety (ThreadSanitizer-compatible)
    - _Requirements: 5.4, 5.6, 5.9_

- [x] 6. Event_Loop implementation
  - [x] 6.1 Implement detail/event_loop.hpp and src/event_loop.cpp
    - Implement `Event_Loop::Config` struct with `max_concurrency` and `deadlock_timeout_s`
    - Validate max_concurrency ∈ [1, 1024]; throw `std::invalid_argument` outside range (Req 6.8)
    - Validate deadlock_timeout_s ∈ [1, 3600]; throw `std::invalid_argument` outside range (Req 6.9)
    - Implement `cycle()` method with three sequential phases: poll → dispatch → complete (Req 6.1)
    - Phase 1 (poll): scan TaskNodes for `pending_deps == 0`, mark as ready (Req 3.4, 3.6)
    - Phase 2 (dispatch): dispatch ready nodes to available ranks, enforce max_concurrency cap (Req 3.5, 6.5)
    - Phase 3 (complete): process completion callbacks via non-blocking MPI probes (Req 6.3), decrement downstream pending_deps (Req 3.3)
    - Implement deadlock detection: emit WARNING via LOGS if no progress for deadlock_timeout_s (Req 6.7, 10.5)
    - On task failure: mark node as `failed`, cancel all transitively dependent nodes, emit ERROR via LOGS (Req 6.6, 10.3)
    - Implement `all_complete()`, `in_flight_count()` accessors
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 10.3, 10.5_

  - [x] 6.2 Write property test: Topological-order invariant (prop_topological_order.cpp)
    - **Property 1: Topological-Order Invariant**
    - Generate random acyclic DAGs (2–128 nodes, 1–512 edges) using layered construction
    - Simulate execution through Event_Loop, record completion sequence numbers via atomic counter
    - Verify every node's sequence number > all upstream dependency sequence numbers
    - Minimum 1000 iterations
    - **Validates: Requirements 3.1, 3.3, 3.4, 3.6, 13.1**

  - [x] 6.3 Write property test: No-orphan invariant (prop_no_orphan.cpp)
    - **Property 2: No-Orphan Invariant**
    - Generate random acyclic DAGs (2–128 nodes, 1–512 edges)
    - Run to completion, verify all nodes reach terminal state within (longest_path + 1) cycles
    - Minimum 1000 iterations
    - **Validates: Requirements 3.7, 13.2**

  - [x] 6.4 Write property test: Concurrency-limit enforcement (prop_concurrency_limit.cpp)
    - **Property 5: Concurrency-Limit Enforcement**
    - Generate random DAGs with max_concurrency N ∈ [1, 64]
    - Track peak in-flight count during Event_Loop execution
    - Verify peak in-flight never exceeds N
    - Minimum 1000 iterations
    - **Validates: Requirements 3.5, 6.5, 13.5**

  - [x] 6.5 Write property test: Transitive cancellation on failure (prop_failure_cascade.cpp)
    - **Property 7: Transitive Cancellation on Failure**
    - Generate random valid DAGs, inject failure at a randomly selected node
    - Verify all transitively reachable downstream nodes are cancelled
    - Verify no cancelled node was dispatched after the failure event
    - Minimum 1000 iterations
    - **Validates: Requirements 4.8, 4.9, 4.10, 6.6**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. DAG construction and validation in GraphOrchestrator
  - [x] 8.1 Implement DAG construction logic in src/graph_orchestrator.cpp (constructor)
    - Accept `Pipeline_Config` by value and `halo::Communicator&&` by rvalue reference (Req 1.2)
    - Create one TaskNode per declared task in `task_names`, initialize `pending_deps` to in-degree (Req 3.1)
    - Build downstream adjacency lists (`dependents` vectors) from `edges` (Req 3.1)
    - Validate acyclicity via Kahn's algorithm; throw `std::invalid_argument` with cycle path (Req 1.7)
    - Validate no dangling task references; throw `std::invalid_argument` with unresolved ID (Req 3.8)
    - Initialize Rank_Pool from world communicator size
    - Initialize Event_Loop with `max_concurrency` and `deadlock_timeout_s` from config
    - Use pImpl pattern (`struct Impl`) for ABI stability
    - _Requirements: 1.2, 1.7, 3.1, 3.2, 3.8_

  - [x] 8.2 Write property test: Acyclicity detection (prop_acyclicity_detection.cpp)
    - **Property 4: Acyclicity-Detection**
    - Generate random directed graphs (2–128 nodes) guaranteed to contain at least one cycle (add back-edge to acyclic DAG)
    - Verify `parse_pipeline` or GraphOrchestrator constructor throws `std::invalid_argument`
    - Minimum 1000 iterations
    - **Validates: Requirements 1.7, 2.11, 13.4**

  - [x] 8.3 Write property test: DAG construction correctness (prop_dag_construction.cpp)
    - **Property 9: DAG Construction Correctness**
    - Generate valid acyclic adjacency lists (2–128 nodes)
    - Verify exactly one TaskNode per declared task
    - Verify each TaskNode's initial `pending_deps` equals its in-degree
    - Verify dangling task references throw `std::invalid_argument`
    - Minimum 1000 iterations
    - **Validates: Requirements 3.1, 3.8**

- [x] 9. GraphOrchestrator public API and lifecycle
  - [x] 9.1 Implement run(), advance_step(), shutdown(), yield/reclaim methods
    - Implement `run()`: enter Event_Loop, process until all terminal; throw `std::runtime_error` on node failure after drain (Req 1.3)
    - Implement `advance_step()`: single cycle (resolve → dispatch → complete); return bool (Req 1.4)
    - Implement `shutdown()`: drain in-flight, release hijacked ranks, destroy state; idempotent (Req 1.5)
    - Implement `yield_ranks(set<int>)`: add ranks to Rank_Pool (Req 5.1)
    - Implement `reclaim_ranks(set<int>)`: remove available ranks, defer if allocated (Req 5.10)
    - Implement move constructor/assignment, delete copy (Req 1.6)
    - Implement const accessors: `total_ranks()`, `available_ranks()`, `in_flight_count()`, `is_shutdown()` (Req 5.9)
    - Emit INFO diagnostics on dispatch/completion, DEBUG on rank operations, ERROR on failure (Req 10.1, 10.2, 10.3, 10.4)
    - Use LOGS Scoped_Context for scheduling phase labeling (Req 10.8)
    - Enforce FIFO queuing when insufficient ranks available (Req 5.5)
    - Implement shutdown timeout with FATAL diagnostic for unreturned ranks (Req 5.7)
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 5.1, 5.5, 5.7, 5.10, 10.1, 10.2, 10.3, 10.4, 10.8, 10.9_

  - [x] 9.2 Write property test: Idempotent shutdown (prop_idempotent_shutdown.cpp)
    - **Property 6: Idempotent Shutdown**
    - Generate random GraphOrchestrator instances, complete execution
    - Call `shutdown()` 2–5 times, verify no throw after first call
    - Verify `available_ranks() == total_ranks()` after shutdown
    - Verify no LOGS emissions beyond first shutdown call
    - Minimum 1000 iterations
    - **Validates: Requirements 1.5, 13.6**

- [x] 10. Transform_Node dispatch implementation
  - [x] 10.1 Implement Transform_Node dispatch in src/transform_node.cpp
    - Implement `dispatch_temporal_bookend` method executing the four-step pointer routing sequence:
      - Step 1: Fetch left bookend FieldView from AMIO (dataset_path, t_left) (Req 4.1)
      - Step 2: Fetch right bookend FieldView from AMIO (dataset_path, t_right) (Req 4.1)
      - Step 3: Query TICK Aliasing_Engine for alpha weight (sim_time, oob_policy) (Req 4.2)
      - Step 4: Route to BLEND engine (left_view, right_view, alpha, output_view, profile) (Req 4.3)
    - Select BLEND dispatch variant via compile-time switch on Temporal_Profile (linear → linear-blend, step → step-select) (Req 4.4)
    - Pass all FieldViews as non-owning `std::mdspan` views via SPAN (Req 4.6)
    - Signal Completion_Token on success (Req 4.7)
    - Propagate errors from AMIO/TICK/BLEND to Event_Loop (Req 4.8, 4.9, 4.10)
    - Include full Doxygen comment block: @pre, @post, @throws, @note, @param, @see tags (Req 12.1–12.8)
    - Store alpha as const-qualified, pass by value without modification (Req 7.3, 7.5)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 7.3, 7.5, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [x] 10.2 Write unit tests for Transform_Node (test_transform_node.cpp)
    - Mock AMIO/TICK/BLEND interfaces to verify call sequence (left before right, query after fetch, route after query)
    - Verify Temporal_Profile::linear routes to linear-blend kernel
    - Verify Temporal_Profile::step routes to step-select kernel
    - Verify AMIO fetch failure propagates error and marks node failed
    - Verify TICK query failure propagates error and marks node failed
    - Verify BLEND execution failure propagates error and marks node failed
    - Verify no arithmetic on alpha or FieldView data (zero-computation guarantee)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9, 4.10_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Public header, RAII enforcement, and zero-computation compliance
  - [x] 12.1 Implement dagr.hpp public header with GraphOrchestrator declaration
    - Declare GraphOrchestrator class in `dagr` namespace with full public API (Req 1.1)
    - Include Doxygen `@brief` and `@details` block (Req 12.7)
    - Forward-declare `halo::Communicator` and `dagr::Pipeline_Config` only (Req 11.3)
    - Do NOT include any Tier 1/1b headers in the public header (Req 11.3)
    - Do NOT include prohibited math headers: `<cmath>`, `<numeric>`, `<complex>`, `<valarray>`, `<random>`, `<numbers>` (Req 1.8, 7.1)
    - _Requirements: 1.1, 1.6, 1.8, 7.1, 11.3, 12.7_

  - [x] 12.2 Implement RAII MPI communicator management
    - Ensure all sub-communicators stored as `halo::Communicator` RAII objects (Req 9.1)
    - No raw `MPI_Comm` handles anywhere in dagr source (Req 9.3)
    - Destructor destroys owned communicators in reverse creation order (Req 9.4)
    - On cancelled TaskNode, destroy pre-allocated communicator without MPI collective (Req 9.7)
    - Allow `halo::Communicator::split` exceptions to propagate with FATAL LOGS (Req 9.5)
    - No direct MPI function calls — all mediated through HALO (Req 9.6)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 12.3 Verify zero-computation and no-stdout constraints
    - Verify no prohibited includes in dagr source files (Req 7.1)
    - Verify no floating-point arithmetic on FieldView/alpha/timestamp data (Req 7.3)
    - Verify no element-wise loops over FieldView data (Req 7.4)
    - Verify all diagnostic output goes through LOGS exclusively (Req 10.6)
    - Verify no direct writes to stdout/stderr/file descriptors (Req 10.6)
    - Verify no raw MPI calls (Req 9.6)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 10.6_

- [x] 13. Test generators and remaining infrastructure
  - [x] 13.1 Implement tests/generators.hpp with RapidCheck DAG generators
    - Implement `dagr::gen::acyclic_dag(min_nodes, max_nodes, max_edges)`: generate random acyclic DAG via layered construction (nodes assigned to topological layers 0..L-1, edges only from lower to higher layers) (Req 13.8)
    - Implement `dagr::gen::cyclic_graph(min_nodes, max_nodes)`: create acyclic DAG then add a back-edge
    - Implement `dagr::gen::valid_pipeline_config(min_streams, max_streams)`: generate complete valid Pipeline_Config
    - Implement `dagr::gen::valid_stream_descriptor()`: generate random Stream_Descriptor with valid fields
    - Implement `dagr::gen::rank_pool_size()`: generate rank count in [1, 64]
    - Implement `dagr::gen::max_concurrency()`: generate value in [1, 64]
    - Implement atomic sequence counter utility for completion ordering verification (Req 13.9)
    - _Requirements: 13.7, 13.8, 13.9_

- [x] 14. Integration wiring and tier isolation verification
  - [x] 14.1 Wire all components and verify complete build
    - Verify full library compiles with `cmake --build` producing `libdagr.a`
    - Verify all test targets link and compile with BUILD_TESTING=ON
    - Verify no upward dependencies (grep for dagr/ in SPAN, TICK, HALO, LOGS, BLEND, AMIO, CONF headers)
    - Verify CMake PRIVATE linkage (no transitive leakage to consumers)
    - Verify `dagr.hpp` compiles standalone without lower-tier headers in include path
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x] 14.2 Create static analysis CI step for zero-computation enforcement
    - Scan dagr source for prohibited includes (`<cmath>`, `<numeric>`, `<complex>`, `<valarray>`, `<random>`, `<numbers>`)
    - Scan for raw MPI function calls (`MPI_Comm_free`, `MPI_Comm_split`, `MPI_Comm_dup`)
    - Scan for floating-point arithmetic tokens applied to field/alpha/timestamp variables
    - Validate CMake dependency graph via `cmake --graphviz` (no Tier 1 → Tier 3 edges)
    - Fail CI build on any violation
    - _Requirements: 7.6, 7.7, 11.6_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Design Properties 1–12)
- Unit tests validate specific examples, edge cases, and error conditions
- All code targets C++20 with strict tier isolation (PRIVATE CMake linkage)
- DAGR performs zero computation — it routes pointers between TICK, AMIO, BLEND, HALO
- RapidCheck uses `RC_GTEST_PROP` macro for unified GTest integration with 1000+ iterations per property
- The `generators.hpp` header provides layered DAG construction guaranteeing acyclicity by design
- All MPI handles are RAII-managed via `halo::Communicator` — no raw `MPI_Comm` in dagr source

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "13.1"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.4", "3.5", "3.6"] },
    { "id": 5, "tasks": ["5.1"] },
    { "id": 6, "tasks": ["5.2", "5.3"] },
    { "id": 7, "tasks": ["6.1"] },
    { "id": 8, "tasks": ["6.2", "6.3", "6.4", "6.5"] },
    { "id": 9, "tasks": ["8.1"] },
    { "id": 10, "tasks": ["8.2", "8.3"] },
    { "id": 11, "tasks": ["9.1"] },
    { "id": 12, "tasks": ["9.2", "10.1"] },
    { "id": 13, "tasks": ["10.2", "12.1"] },
    { "id": 14, "tasks": ["12.2", "12.3"] },
    { "id": 15, "tasks": ["14.1", "14.2"] }
  ]
}
```
