# Requirements Document

## Introduction

DAGR (Directed Acyclic Graph Router) is the Tier 3 YAML-driven orchestrator in the HELM ecosystem. It is an asynchronous event-driven task scheduler whose sole responsibility is routing pointers, managing execution dependencies, and triggering lower-tier computation engines. DAGR performs zero mathematical calculations and zero calendar manipulations — it queries TICK for temporal state, dispatches work to BLEND/AXIS execution engines, and routes non-owning memory views (std::mdspan via SPAN) between HELM components.

DAGR parses a YAML pipeline configuration that declares I/O streams with temporal profiles (linear, step) and out-of-bounds policies (clamp, cycle). At runtime, it maintains a DAG of task nodes whose dependencies resolve asynchronously. The Transform Node Dispatcher is DAGR's central execution mechanism: for Temporal Bookend tasks, it fetches left/right bookend FieldViews from AMIO, queries TICK for the aliased interpolation weight, and routes execution to the BLEND utility engine based on the stream's configured temporal profile.

DAGR also implements MPI rank hijacking: dynamically reallocating idle model processors to framework network tasks (halo exchanges, interpolation, I/O) to maximize HPC resource utilization. All MPI handles are RAII-wrapped via HALO. All parallel execution is delegated to Kokkos-based engines. DAGR itself contains no parallel loops, no mathematical kernels, and no calendar logic.

## Glossary

- **GraphOrchestrator**: The master class of DAGR; owns the DAG, the event loop, and the rank pool. It parses the pipeline YAML, constructs task nodes, resolves dependencies, and dispatches execution.
- **TaskNode**: A vertex in the DAG representing a unit of schedulable work (model step, transform, I/O flush, halo exchange). Each TaskNode declares its input dependencies and output products.
- **Transform_Node**: A specialized TaskNode that fetches data views, queries TICK for temporal state, and routes execution to a utility engine (BLEND, AXIS) without performing computation itself.
- **Temporal_Bookend**: A transform operation where two bounding dataset snapshots (left and right FieldViews) are fetched and blended using an interpolation weight from TICK's Aliasing Engine.
- **FieldView**: A non-owning multi-dimensional memory view (std::mdspan) provided by AMIO representing a single field snapshot at a specific time.
- **Temporal_Profile**: An enum (linear, step) configuring how the BLEND engine interpolates between bookend snapshots. DAGR routes to the correct BLEND kernel based on this value.
- **OutOfBounds_Policy**: An enum (clamp, cycle) specifying how TICK's Aliasing Engine remaps simulation time that exceeds dataset coverage for a given stream.
- **Pipeline_Config**: The YAML configuration document declaring all I/O streams, their temporal profiles, out-of-bounds policies, task dependencies, and rank allocation hints.
- **Stream_Descriptor**: A parsed representation of a single I/O stream from the Pipeline_Config, holding the stream name, temporal profile, out-of-bounds policy, dataset path, and snapshot interval.
- **BLEND**: A Tier 1 utility engine that performs temporal interpolation between two FieldViews using a supplied alpha weight and temporal profile. DAGR routes pointers to BLEND but never invokes interpolation math directly.
- **Rank_Pool**: The set of MPI ranks currently available for dynamic reallocation. DAGR tracks which ranks are idle (model yielded) and which are active (executing a task).
- **Rank_Hijacking**: The mechanism by which DAGR temporarily reassigns idle model ranks to framework tasks (halo exchanges, interpolation, I/O) to maximize processor utilization.
- **Event_Loop**: The main scheduling loop within GraphOrchestrator that polls for resolved dependencies, dispatches ready TaskNodes, and processes completion callbacks.
- **Dependency_Edge**: A directed edge in the DAG from a producer TaskNode to a consumer TaskNode, indicating that the consumer cannot begin until the producer signals completion.
- **Completion_Token**: An opaque handle returned when a TaskNode is dispatched, used by the Event_Loop to track in-flight tasks and signal downstream dependents upon completion.
- **YAML_Parser**: The Pipeline_Config reader that parses YAML documents into Stream_Descriptors and DAG topology. Uses CONF (HELM::CONF) for parsing.

## Requirements

### Requirement 1: GraphOrchestrator Class Interface

**User Story:** As a HELM integrator, I want a single master class that owns the full DAG lifecycle, so that pipeline construction, scheduling, and teardown are encapsulated behind one coherent API.

#### Acceptance Criteria

1. THE GraphOrchestrator SHALL be declared in the `dagr` namespace within the header `dagr.hpp`.
2. THE GraphOrchestrator SHALL be constructed from a Pipeline_Config object (parsed YAML) and a `halo::Communicator` representing the world communicator; the constructor SHALL take ownership of the Pipeline_Config by value and the halo::Communicator by rvalue reference (move semantics).
3. THE GraphOrchestrator SHALL expose a `run()` method that enters the Event_Loop and processes tasks until all terminal nodes in the DAG have completed; IF any TaskNode fails during `run()`, THE method SHALL throw `std::runtime_error` containing the failed node identifier and error message after all non-cancelled nodes have drained.
4. THE GraphOrchestrator SHALL expose an `advance_step()` method that executes exactly one scheduling cycle: resolving ready nodes, dispatching them, and processing completions for a single model timestep; the method SHALL return a `bool` where `true` indicates all nodes have completed and `false` indicates work remains.
5. THE GraphOrchestrator SHALL expose a `shutdown()` method that drains all in-flight tasks, releases hijacked ranks back to their owning models, and destroys internal state in dependency-safe order; calling `shutdown()` on an already-shut-down instance SHALL be a no-op.
6. THE GraphOrchestrator SHALL be move-constructible and move-assignable but SHALL NOT be copyable (copy constructor and copy assignment operator are deleted).
7. IF the GraphOrchestrator is constructed with a Pipeline_Config containing a cyclic dependency, THEN THE GraphOrchestrator SHALL throw std::invalid_argument indicating the cycle path (at least two node identifiers) at construction time.
8. THE `dagr.hpp` header SHALL NOT include any mathematical computation header (`<cmath>`, `<numeric>`, `<complex>`, `<valarray>`, `<random>`, `<numbers>`) or calendar manipulation header; all such operations are delegated to lower-tier engines via pointer routing.

### Requirement 2: Pipeline YAML Parser

**User Story:** As a model integrator, I want DAGR to read its pipeline topology and stream configurations from a YAML file, so that orchestration behavior is configurable without recompilation.

#### Acceptance Criteria

1. WHEN the YAML_Parser reads a pipeline configuration file, THE YAML_Parser SHALL use `conf::Config::from_file` from HELM::CONF to load the document.
2. IF `conf::Config::from_file` throws due to a missing file or malformed YAML, THEN THE YAML_Parser SHALL allow the exception to propagate to the caller without catching or wrapping it.
3. WHEN the YAML_Parser encounters a stream entry, THE YAML_Parser SHALL extract the `temporal_profile` field and map it to the Temporal_Profile enum (linear or step).
4. WHEN the YAML_Parser encounters a stream entry, THE YAML_Parser SHALL extract the `out_of_bounds_policy` field and map it to the OutOfBounds_Policy enum (clamp or cycle).
5. WHEN the YAML_Parser encounters a stream entry, THE YAML_Parser SHALL extract the `dataset_path` string and `snapshot_interval` string field, storing the snapshot_interval as an opaque duration token to be interpreted by TICK at runtime.
6. IF a stream entry contains an unrecognized `temporal_profile` value, THEN THE YAML_Parser SHALL throw std::invalid_argument reporting the stream name and the invalid value.
7. IF a stream entry contains an unrecognized `out_of_bounds_policy` value, THEN THE YAML_Parser SHALL throw std::invalid_argument reporting the stream name and the invalid value.
8. IF a stream entry is missing any of the four required fields (`temporal_profile`, `out_of_bounds_policy`, `dataset_path`, `snapshot_interval`), THEN THE YAML_Parser SHALL throw std::invalid_argument reporting the stream name and the missing field name.
9. IF two or more stream entries share the same stream name, THEN THE YAML_Parser SHALL throw std::invalid_argument reporting the duplicate stream name.
10. THE YAML_Parser SHALL produce a Pipeline_Config containing a vector of Stream_Descriptors (one per stream entry, preserving declaration order) and a DAG adjacency list representing task dependencies.
11. THE YAML_Parser SHALL validate that the DAG adjacency list is acyclic by performing a topological sort at parse time; IF a cycle is detected, THEN THE YAML_Parser SHALL throw std::invalid_argument reporting at least two node identifiers involved in the cycle.
12. THE YAML_Parser SHALL be a free function named `parse_pipeline` in the `dagr` namespace with signature accepting a `std::filesystem::path` and returning a `Pipeline_Config` by value.
13. THE YAML_Parser SHALL NOT validate that `dataset_path` refers to an existing file at parse time; path existence is deferred to runtime when AMIO opens the dataset.

### Requirement 3: DAG Construction and Dependency Resolution

**User Story:** As a scheduler, I want a well-formed DAG whose edges enforce execution ordering, so that no task executes before its upstream dependencies have completed.

#### Acceptance Criteria

1. WHEN the GraphOrchestrator constructs the DAG from a Pipeline_Config, THE GraphOrchestrator SHALL create one TaskNode per declared task and one Dependency_Edge per declared dependency in the adjacency list, and SHALL initialize each TaskNode's pending-dependency count to its in-degree (the number of incoming Dependency_Edges).
2. THE DAG SHALL store TaskNodes in a contiguous container (std::vector) indexed by a zero-based integer identifier of type `std::uint32_t`, assigned sequentially at construction time in the order tasks appear in the Pipeline_Config, and stable for the lifetime of the DAG instance.
3. WHEN a TaskNode completes execution, THE Event_Loop SHALL decrement the pending-dependency count of all downstream TaskNodes connected by Dependency_Edges from the completed node by exactly one per edge.
4. WHEN a downstream TaskNode's pending-dependency count reaches zero, THE Event_Loop SHALL mark that TaskNode as ready for dispatch within the same scheduling cycle in which the decrement occurred.
5. THE GraphOrchestrator SHALL track in-flight task count and SHALL NOT dispatch more concurrent tasks than the effective concurrency limit, defined as the minimum of the available Rank_Pool size (`available_ranks()`) and the configurable maximum concurrency specified in the Pipeline_Config.
6. IF the DAG contains TaskNodes with no incoming Dependency_Edges (root nodes), THEN THE Event_Loop SHALL mark those nodes as immediately ready for dispatch at the start of each scheduling cycle (before the first dispatch phase of that cycle).
7. WHEN all TaskNodes in the DAG have completed (in-flight count equals zero and no ready nodes remain), THE Event_Loop SHALL return from the current `advance_step()` invocation, indicating step completion to the caller via normal return.
8. IF the Pipeline_Config adjacency list references a task identifier that does not correspond to any declared task, THEN THE GraphOrchestrator SHALL throw `std::invalid_argument` at construction time, with an error message indicating the unresolved task identifier.

### Requirement 4: Transform Node Dispatcher for Temporal Bookend Tasks

**User Story:** As a forcing data consumer, I want DAGR to orchestrate the fetch-query-route sequence for temporal interpolation without performing any math itself, so that the orchestrator remains a pure pointer router.

#### Acceptance Criteria

1. WHEN the Event_Loop dispatches a Transform_Node configured as a Temporal_Bookend task, THE Transform_Node SHALL request the left bookend FieldView from AMIO using the stream's dataset path and the left bounding timestamp, then request the right bookend FieldView from AMIO using the stream's dataset path and the right bounding timestamp, issuing the two requests sequentially (left before right).
2. WHEN both bookend FieldViews have been obtained, THE Transform_Node SHALL query TICK's Aliasing Engine for the interpolation weight (alpha) using the current simulation time and the stream's configured OutOfBounds_Policy.
3. WHEN the alpha weight has been obtained, THE Transform_Node SHALL route execution to the BLEND utility engine by passing: the left FieldView pointer, the right FieldView pointer, the alpha weight, and the output FieldView pointer previously allocated by the Event_Loop as part of the node's output product slot.
4. THE Transform_Node SHALL select the BLEND dispatch variant using a compile-time switch on the stream's Temporal_Profile enum: the `linear` case routes to the linear-blend kernel, and the `step` case routes to the step-select kernel.
5. THE Transform_Node SHALL NOT perform any arithmetic on the FieldView data, the alpha weight, or the timestamp values; all computation is delegated to the called engines (AMIO, TICK, BLEND).
6. THE Transform_Node SHALL pass all FieldViews (left bookend, right bookend, and output) as non-owning `std::mdspan` views (via SPAN), adhering to the zero-copy memory law.
7. WHEN BLEND completes execution successfully, THE Transform_Node SHALL signal completion to the Event_Loop via the node's Completion_Token, enabling dispatch of dependent downstream TaskNodes.
8. IF AMIO returns an error when fetching either bookend FieldView, THEN THE Transform_Node SHALL propagate the error to the Event_Loop which marks the node as failed and cancels all transitively dependent TaskNodes.
9. IF TICK returns an error when computing the aliased alpha, THEN THE Transform_Node SHALL propagate the error to the Event_Loop which marks the node as failed and cancels all transitively dependent TaskNodes.
10. IF BLEND returns an error during kernel execution, THEN THE Transform_Node SHALL propagate the error to the Event_Loop which marks the node as failed and cancels all transitively dependent TaskNodes.

### Requirement 5: MPI Rank Hijacking

**User Story:** As an HPC operations engineer, I want DAGR to dynamically reallocate idle model processors to framework tasks, so that hardware utilization is maximized during periods when models have yielded their processors.

#### Acceptance Criteria

1. WHEN a model component signals that it has yielded its processors by calling the GraphOrchestrator's `yield_ranks(rank_set)` method with a set of MPI rank identifiers, THE GraphOrchestrator SHALL add those MPI ranks to the Rank_Pool as available for framework tasks within the same scheduling cycle.
2. WHEN a framework task (halo exchange, interpolation, I/O flush) requires MPI ranks, THE GraphOrchestrator SHALL allocate the requested number of ranks (minimum 1, maximum equal to Rank_Pool `available_ranks()`) from the Rank_Pool by splitting a sub-communicator from the available ranks using `halo::Communicator::split`.
3. WHEN a framework task completes and releases its allocated ranks, THE GraphOrchestrator SHALL return those ranks to the Rank_Pool and free the sub-communicator via RAII destruction of the `halo::Communicator` object.
4. THE Rank_Pool SHALL track rank availability using an atomic bitset or lock-free structure so that rank allocation and deallocation are safe under concurrent task completions without requiring an external mutex.
5. IF a framework task requests more ranks than are currently available in the Rank_Pool, THEN THE GraphOrchestrator SHALL enqueue the task in a FIFO queue and dispatch it when sufficient ranks become available, rather than blocking the Event_Loop.
6. IF a framework task requests zero ranks, THEN THE GraphOrchestrator SHALL reject the request by throwing std::invalid_argument indicating the task identifier and the invalid rank count.
7. WHEN the GraphOrchestrator's `shutdown()` method is called, THE GraphOrchestrator SHALL wait for all hijacked ranks to be returned to the Rank_Pool before releasing the world communicator, subject to a configurable timeout (read from Pipeline_Config, default 30 seconds); IF the timeout expires with ranks still outstanding, THEN THE GraphOrchestrator SHALL emit a FATAL diagnostic via LOGS identifying the unreturned ranks and proceed with forced shutdown.
8. THE GraphOrchestrator SHALL NOT perform any MPI collective operations directly; all MPI communication is delegated to HALO's Communicator interface.
9. THE Rank_Pool SHALL maintain a count of total managed ranks and currently available ranks, exposed via const accessors `total_ranks()` and `available_ranks()`, where the invariant `available_ranks() + allocated_ranks() == total_ranks()` holds at every observable point.
10. IF a model component calls `reclaim_ranks(rank_set)` to recover previously yielded ranks while those ranks are allocated to an in-flight framework task, THEN THE GraphOrchestrator SHALL defer the reclamation until the framework task completes and returns the ranks, and SHALL NOT pre-empt or abort the in-flight task.

### Requirement 6: Event Loop Execution Model

**User Story:** As a scheduler, I want an asynchronous event-driven loop that polls for task readiness and dispatches work without blocking on individual task completion, so that maximum concurrency is achieved.

#### Acceptance Criteria

1. THE Event_Loop SHALL operate in a poll-dispatch-complete cycle: first checking for newly ready nodes, then dispatching ready nodes to available ranks, then processing completion callbacks from finished tasks. Each cycle SHALL execute these three phases in strict sequential order before beginning the next cycle.
2. WHEN a TaskNode is dispatched, THE Event_Loop SHALL issue a Completion_Token and register a callback that decrements downstream dependency counts upon task completion. The Completion_Token SHALL remain valid until the associated task completes, fails, or is cancelled.
3. THE Event_Loop SHALL support asynchronous task completion via non-blocking MPI probes (delegated to HALO) so that multiple tasks can complete in arbitrary order within a single complete-phase of the cycle.
4. WHILE the Event_Loop has in-flight tasks but no ready nodes, THE Event_Loop SHALL poll for completions by invoking HALO's progress mechanism once per poll iteration, yielding control to the MPI progress engine for at least 1 microsecond per iteration to avoid busy-spinning.
5. THE Event_Loop SHALL enforce a configurable maximum concurrency limit (read from the `max_concurrency` field in Pipeline_Config) that caps the number of simultaneously dispatched tasks regardless of Rank_Pool availability. The value SHALL be an integer in the range 1 to 1024 inclusive.
6. IF a dispatched TaskNode fails (returns an error or throws), THEN THE Event_Loop SHALL mark the failed node's status as `failed`, mark all transitively dependent nodes' status as `cancelled`, and emit a diagnostic via LOGS at ERROR severity including the failed node identifier and the count of cancelled dependents.
7. WHEN the Event_Loop detects that a scheduling cycle produced no progress (no completions and no new ready nodes for a duration exceeding the `deadlock_timeout` value in Pipeline_Config, which defaults to 30 seconds and must be in the range 1 to 3600 seconds), THE Event_Loop SHALL emit a WARNING diagnostic via LOGS indicating potential deadlock including the in-flight task count and blocked node count, and continue polling.
8. IF the Pipeline_Config `max_concurrency` value is less than 1 or greater than 1024, THEN THE Event_Loop SHALL throw std::invalid_argument indicating the invalid value and the acceptable range.
9. IF the Pipeline_Config `deadlock_timeout` value is less than 1 second or greater than 3600 seconds, THEN THE Event_Loop SHALL throw std::invalid_argument indicating the invalid value and the acceptable range.

### Requirement 7: Zero-Computation Constraint

**User Story:** As a HELM architect, I want DAGR to contain zero mathematical calculations and zero calendar manipulations in its own source, so that the separation of concerns between orchestration and computation is absolute.

#### Acceptance Criteria

1. THE dagr namespace source files (all `.hpp` and `.cpp` files under the dagr library directory, excluding test files) SHALL NOT include `<cmath>`, `<numeric>`, `<complex>`, `<valarray>`, `<random>`, or `<numbers>`; the prohibited set is any C++ standard library header classified under the "Numerics library" category.
2. THE dagr namespace source files SHALL NOT include any TICK header other than the public dispatch API header (`tick/api.hpp` or equivalent installed interface header); DAGR SHALL NOT call any function that constructs, compares, or performs arithmetic on time-point or duration values — all temporal queries are delegated to TICK which returns opaque type-aliased results that DAGR forwards without calling member functions on them.
3. THE dagr namespace source files SHALL NOT contain floating-point arithmetic operators (`+`, `-`, `*`, `/`, `%`) applied to FieldView data, interpolation weights, or timestamp values; the alpha value received from TICK SHALL be stored in a const-qualified variable and passed by value to BLEND without reassignment or modification.
4. THE dagr namespace source files SHALL NOT contain loop constructs (`for`, `while`, `do-while`, range-based `for`, or standard algorithm calls such as `std::for_each`, `std::transform`, `std::accumulate`) that iterate over FieldView elements or field data arrays; all element-wise operations are the responsibility of the dispatched engine (BLEND, AXIS).
5. THE dagr namespace code SHALL treat interpolation weights, timestamps, and field data as opaque values: DAGR may store them in variables, pass them as function arguments, and check them for null or error states (null pointer check, `std::optional::has_value`), but SHALL NOT read their numeric content, compare their magnitudes, perform arithmetic on them, or log their internal values.
6. IF a code review or static analysis identifies any arithmetic operation in dagr source files beyond the following permitted integer operations — increment (`++`), decrement (`--`), addition/subtraction for index offset calculation, comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`), and bitwise operations for rank bitset manipulation — THEN that code SHALL be refactored into the appropriate Tier 1 engine before the review is approved.
7. THE DAGR CI pipeline SHALL enforce the zero-computation constraint via a static analysis step that scans all dagr namespace source files for prohibited includes and floating-point arithmetic tokens; IF the scan detects a violation, THEN the CI build SHALL fail and report the file, line number, and violation category.

### Requirement 8: Stream Descriptor and Temporal Profile Configuration

**User Story:** As a model integrator, I want each I/O stream to declare its interpolation behavior and out-of-bounds handling in the pipeline YAML, so that DAGR can route execution to the correct engine configuration without hardcoded logic.

#### Acceptance Criteria

1. THE Stream_Descriptor SHALL be a value type containing: a stream name (std::string, maximum 128 characters), a Temporal_Profile enum value, an OutOfBounds_Policy enum value, a dataset path (std::filesystem::path), and a snapshot interval stored as a `tick::Duration` value that is passed unmodified to TICK's Aliasing Engine.
2. THE Temporal_Profile SHALL be a scoped enum (enum class) in the `dagr` namespace with exactly two enumerators: `linear` and `step`.
3. THE OutOfBounds_Policy SHALL be a scoped enum (enum class) in the `dagr` namespace with exactly two enumerators: `clamp` and `cycle`.
4. WHEN the YAML_Parser encounters `temporal_profile: linear`, THE YAML_Parser SHALL set the Stream_Descriptor's temporal profile to `Temporal_Profile::linear`.
5. WHEN the YAML_Parser encounters `temporal_profile: step`, THE YAML_Parser SHALL set the Stream_Descriptor's temporal profile to `Temporal_Profile::step`.
6. WHEN the YAML_Parser encounters `out_of_bounds_policy: clamp`, THE YAML_Parser SHALL set the Stream_Descriptor's out-of-bounds policy to `OutOfBounds_Policy::clamp`.
7. WHEN the YAML_Parser encounters `out_of_bounds_policy: cycle`, THE YAML_Parser SHALL set the Stream_Descriptor's out-of-bounds policy to `OutOfBounds_Policy::cycle`.
8. THE Stream_Descriptor SHALL be copyable, movable, and SHALL support equality comparison via `operator==` where two Stream_Descriptors are equal if and only if all five members (stream name, temporal profile, out-of-bounds policy, dataset path, and snapshot interval) compare equal.
9. IF the YAML_Parser encounters a stream entry whose `dataset_path` field is empty or whose `snapshot_interval` value is zero or negative, THEN THE YAML_Parser SHALL throw std::invalid_argument reporting the stream name and the invalid field.

### Requirement 9: RAII MPI Handle Management

**User Story:** As a systems engineer, I want all MPI communicators created by DAGR to be RAII-managed via HALO, so that rank hijacking cannot leak communicator handles even under exceptional control flow.

#### Acceptance Criteria

1. WHEN the GraphOrchestrator creates a sub-communicator for a framework task, THE GraphOrchestrator SHALL store it as a `halo::Communicator` RAII object whose destructor calls `MPI_Comm_free` automatically.
2. WHEN a framework task completes (normally or via exception), THE associated `halo::Communicator` SHALL be destroyed, releasing the MPI handle regardless of the exit path.
3. THE GraphOrchestrator SHALL NOT store raw `MPI_Comm` handles in any data structure; all communicator state is held via `halo::Communicator` move-only objects.
4. WHEN the GraphOrchestrator's destructor runs, THE destructor SHALL destroy all owned `halo::Communicator` objects in reverse order of creation, ensuring no zombie communicators persist after GraphOrchestrator lifetime ends.
5. IF `halo::Communicator::split` throws during rank hijacking, THEN THE GraphOrchestrator SHALL allow the exception to propagate after HALO's RAII has cleaned up any partially-constructed communicator, and SHALL emit a FATAL diagnostic via LOGS.
6. THE GraphOrchestrator SHALL NOT call `MPI_Comm_free`, `MPI_Comm_split`, `MPI_Comm_dup`, or any raw MPI function directly; all MPI operations are mediated through HALO's Communicator interface.
7. WHEN a TaskNode is cancelled (due to upstream failure), THE Event_Loop SHALL destroy any pre-allocated `halo::Communicator` associated with that task's pending rank allocation, returning the ranks to the Rank_Pool without invoking any MPI collective on the doomed communicator.

### Requirement 10: Diagnostics and Error Reporting via LOGS

**User Story:** As an operations engineer, I want DAGR to emit structured diagnostics through HELM::LOGS, so that scheduling decisions, rank allocation events, and error conditions are traceable across MPI ranks.

#### Acceptance Criteria

1. WHEN the GraphOrchestrator dispatches a TaskNode, THE GraphOrchestrator SHALL emit an INFO-level diagnostic via LOGS indicating the node identifier (integer index), the set of assigned rank numbers, and the dispatch wall-clock timestamp as a monotonic time point.
2. WHEN a TaskNode completes successfully, THE GraphOrchestrator SHALL emit an INFO-level diagnostic via LOGS indicating the node identifier and the elapsed wall-clock time in milliseconds from dispatch to completion.
3. WHEN a TaskNode fails, THE GraphOrchestrator SHALL emit an ERROR-level diagnostic via LOGS including the node identifier, the error message text from the failure, and the identifiers of all transitively cancelled dependent nodes (may be an empty set if no dependents exist).
4. WHEN rank hijacking allocates or releases ranks, THE GraphOrchestrator SHALL emit a DEBUG-level diagnostic via LOGS indicating the allocated or released rank numbers, the requesting task's node identifier, and the Rank_Pool state after the operation (available rank count and total rank count).
5. IF the Event_Loop detects no progress (zero task completions and zero newly ready nodes) for a duration exceeding the deadlock timeout configured in the Pipeline_Config (with a default of 30 seconds if not specified), THEN THE Event_Loop SHALL emit a WARNING-level diagnostic via LOGS including the in-flight task count, the ready task count, and the blocked task count.
6. THE dagr namespace SHALL NOT write directly to stdout, stderr, or any file descriptor; all diagnostic and status output SHALL be emitted exclusively through the LOGS Logger API.
7. WHEN the YAML_Parser encounters a validation error, THE YAML_Parser SHALL emit a WARNING-level diagnostic via LOGS including the file path being parsed, the stream name or node name where the error was detected, and the field name or constraint that failed validation, before throwing the exception.
8. WHEN the GraphOrchestrator dispatches or completes a TaskNode, THE GraphOrchestrator SHALL emit the diagnostic within a LOGS Scoped_Context labelled with the active scheduling phase name (dispatch, completion, or failure) so that log records are attributable to the orchestration stage.
9. IF the GraphOrchestrator emits a FATAL-level diagnostic via LOGS (triggered by an unrecoverable internal error such as a failed communicator split), THEN THE GraphOrchestrator SHALL include the node identifier of the task being processed (or "none" if outside task context) and the originating error message before LOGS initiates Synchronized_Abort.

### Requirement 11: No Circular Dependencies

**User Story:** As a HELM architect, I want DAGR to depend only downward on Tier 1/1b libraries and never introduce upward or lateral dependencies from those libraries back to DAGR, so that the tiered architecture remains intact.

#### Acceptance Criteria

1. THE DAGR library (Tier 3) SHALL depend only on the following lower-tier libraries: Tier 2 (SPAN), Tier 1 (TICK, HALO, LOGS, BLEND) and Tier 1b (AMIO, CONF); DAGR SHALL NOT introduce any compile-time or link-time dependency from those libraries back to DAGR, verified by confirming that no installed header or CMake exported target of SPAN, TICK, HALO, LOGS, BLEND, AMIO, or CONF references any DAGR header or DAGR CMake target.
2. THE DAGR source files SHALL NOT include headers belonging to domain model code (atmosphere, ocean, land surface); DAGR interacts with models exclusively through the SPAN interface layer (Tier 2).
3. THE DAGR public header (`dagr.hpp`) SHALL NOT expose any Tier 1 or Tier 1b library types in its public interface; lower-tier types SHALL appear only in the implementation (compilation units or PRIVATE headers) or be accepted via forward-declared opaque handles that do not require including the lower-tier library's headers.
4. THE Tier 2 and Tier 1/1b libraries (SPAN, TICK, HALO, LOGS, BLEND, AMIO, CONF) SHALL compile and pass their own test suites without DAGR being present in the build tree; DAGR consumes only their public installed APIs without patches or forks.
5. THE DAGR CMakeLists.txt SHALL link HELM::SPAN, HELM::TICK, HELM::HALO, HELM::LOGS, HELM::BLEND, HELM::AMIO, and HELM::CONF as PRIVATE dependencies (not PUBLIC or INTERFACE), so that no downstream consumer of DAGR transitively inherits include paths or link flags from those libraries.
6. WHEN the CI pipeline builds the HELM workspace, THE CI pipeline SHALL execute a dependency-graph validation step (using `cmake --graphviz` or equivalent static analysis of the CMake target graph) that fails the build IF any Tier 1 or Tier 1b target depends on a Tier 2 or Tier 3 target.

### Requirement 12: Doxygen Documentation for Transform Node Dispatch Sequence

**User Story:** As a developer onboarding to HELM, I want the Transform Node's pointer routing sequence documented with Doxygen, so that the fetch-query-route pattern is immediately understandable without reading implementation code.

#### Acceptance Criteria

1. THE Transform_Node dispatch method SHALL include a Doxygen comment block documenting the four-step pointer routing sequence as a numbered list: (1) Fetch Left FieldView from AMIO, (2) Fetch Right FieldView from AMIO, (3) Query TICK for alpha weight, (4) Route to BLEND engine.
2. THE Doxygen comment SHALL use `@pre` tags to document preconditions: one `@pre` stating the stream's dataset must be open in AMIO, and one `@pre` stating the simulation time must be set in TICK.
3. THE Doxygen comment SHALL use `@post` tags to document postconditions: one `@post` stating the output FieldView contains the blended result written by BLEND, and one `@post` stating DAGR has not modified any FieldView data.
4. THE Doxygen comment SHALL use one `@throws` tag per failure mode, documenting at minimum: one for AMIO fetch failure specifying `std::runtime_error` propagated from AMIO, and one for TICK query failure specifying `std::runtime_error` propagated from TICK.
5. THE Doxygen comment SHALL include a `@note` tag explicitly stating that DAGR performs no arithmetic on the routed pointers or field values and acts as a pure dispatcher.
6. THE Doxygen comment SHALL include `@param` tags documenting each parameter of the dispatch method, including the Stream_Descriptor, the output FieldView destination, and the simulation time token.
7. THE GraphOrchestrator class SHALL include a Doxygen `@brief` tag summarizing GraphOrchestrator as the Tier 3 DAG-driven orchestrator, and a `@details` block that names the lower-tier engines it dispatches to (TICK, AMIO, BLEND, HALO) and states that GraphOrchestrator performs no computation itself — it routes pointers and manages execution dependencies.
8. THE Transform_Node Doxygen comment SHALL include `@see` cross-references to the AMIO FieldView fetch API, the TICK Aliasing Engine query API, and the BLEND dispatch API.

### Requirement 13: Property-Based Testing for DAG Scheduling Correctness

**User Story:** As a quality engineer, I want property-based tests proving DAGR's scheduling invariants hold for arbitrary DAG topologies, so that edge cases in dependency resolution and rank allocation are exhaustively verified.

#### Acceptance Criteria

1. THE test suite SHALL contain a property-based test proving the topological-order invariant: for all generated valid DAGs (2 to 128 nodes, 1 to 512 directed edges), every TaskNode's recorded completion sequence number is strictly greater than the completion sequence numbers of all its upstream dependencies.
2. THE test suite SHALL contain a property-based test proving the no-orphan invariant: for all generated valid DAGs (2 to 128 nodes, 1 to 512 directed edges), every TaskNode reaches a terminal state (completed or cancelled) within a number of scheduling cycles no greater than the DAG's longest path length plus one.
3. THE test suite SHALL contain a property-based test proving the rank-conservation invariant: for all scheduling sequences with a generated Rank_Pool of 1 to 64 total ranks, the sum returned by `available_ranks()` plus the count of ranks currently allocated to in-flight tasks equals `total_ranks()` immediately before and after every dispatch and completion event.
4. THE test suite SHALL contain a property-based test proving the acyclicity-detection property: for all generated directed graphs of 2 to 128 nodes containing at least one cycle, the YAML_Parser or GraphOrchestrator constructor SHALL throw std::invalid_argument before returning.
5. THE test suite SHALL contain a property-based test proving the concurrency-limit property: for all scheduling sequences with a generated max-concurrency value N in the range 1 to 64, the count of simultaneously dispatched-but-not-completed tasks never exceeds N at any point during the Event_Loop execution.
6. THE test suite SHALL contain a property-based test proving the idempotent-shutdown property: calling `shutdown()` two or more times (up to 5 calls) on the same GraphOrchestrator instance SHALL NOT throw an exception on any call after the first, SHALL leave `available_ranks()` equal to `total_ranks()`, and SHALL produce no LOGS emissions beyond those emitted by the first `shutdown()` call.
7. THE test suite SHALL use RapidCheck integrated with Google Test and execute a minimum of 1000 generated cases per property.
8. THE test suite SHALL provide a DAG generator in a dedicated header that produces random acyclic directed graphs by assigning each node a random topological layer (0 to L-1 where L is generated in range 2 to 32) and adding edges only from lower-layer nodes to higher-layer nodes, guaranteeing acyclicity by construction.
9. THE test suite SHALL record task completion ordering using a monotonically incrementing atomic sequence counter that is captured by each TaskNode's completion callback, providing the observable evidence for invariant verification in criteria 1, 2, and 5.
