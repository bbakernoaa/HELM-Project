# Implementation Plan: LOGS (Lightweight Operational Global Status)

## Overview

Incremental implementation of the LOGS Tier 1 C++20 micro-library providing thread-safe, MPI-rank-aware operational status, logging, and error-handling for distributed HPC jobs. LOGS consolidates redundant cross-rank messages into single representative entries to prevent filesystem I/O storms at scale. Tasks are ordered to build foundational value types first, then layer the thread-safe Logger, sinks, scoped context, consolidation engine, synchronized abort, and finally CI verification. LOGS has **no Kokkos dependency** and **no Fortran interface**.

## Tasks

- [ ] 1. Repository scaffolding and CMake build system
  - [ ] 1.1 Create repository directory structure and CMakeLists.txt
    - Create `libs/logs/` directory with subdirectories: `include/logs/`, `include/logs/detail/`, `src/`, `src/detail/`, `tests/`, `cmake/`
    - Write root `CMakeLists.txt` with CMake 3.21 minimum, C++20 requirement, `find_package(MPI REQUIRED COMPONENTS CXX)`, `BUILD_TESTING` option (default OFF)
    - Create `logs` shared library target with `HELM::LOGS` namespace alias, linking `MPI::MPI_CXX` as PUBLIC dependency
    - SHALL NOT introduce any Kokkos dependency or compiler-specific language extensions
    - Halt configuration with fatal error if MPI CXX component not found or C++20 not supported
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.8, 12.9, 14.2, 14.4_

  - [ ] 1.2 Create CMake export and install configuration
    - Write `cmake/LOGSConfig.cmake.in` and `cmake/LOGSConfigVersion.cmake.in` templates
    - Configure `install(EXPORT LOGSTargets ...)` with `NAMESPACE HELM::` and `SameMajorVersion` compatibility
    - Declare a LOGS project version; ensure downstream `find_package(LOGS)` works with transitive MPI dependency
    - _Requirements: 12.5_

  - [ ] 1.3 Create test infrastructure CMakeLists.txt
    - Write `tests/CMakeLists.txt` that locates GTest via `find_package(GTest REQUIRED)` when `BUILD_TESTING=ON`
    - Define test executable targets for unit tests and property tests (RapidCheck)
    - Configure `mpirun -np 4` test runner for MPI-based tests
    - Halt configuration with fatal error if BUILD_TESTING=ON and GTest not found
    - _Requirements: 12.6, 12.7, 12.10_

  - [ ] 1.4 Create README and .gitignore
    - Write `libs/logs/README.md` with copy-runnable commands for Docker container launch, CMake configure/build, and test execution
    - Write `libs/logs/.gitignore` for build artifacts
    - _Requirements: 15.4_

- [ ] 2. Severity_Level enum and to_string
  - [ ] 2.1 Implement logs::Severity_Level and to_string
    - Create `include/logs/severity.hpp` with scoped enum `Severity_Level : int` containing DEBUG=0, INFO=1, WARNING=2, ERROR=3, FATAL=4
    - Implement `to_string(Severity_Level)` returning fixed labels "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"
    - Ensure total ordering via defaulted `operator<=>` (C++20 three-way comparison)
    - _Requirements: 1.1, 1.6_

  - [ ]* 2.2 Write property test: Severity Level Total Ordering
    - **Property 1: Severity Level Total Ordering**
    - For any two Severity_Level values, verify comparison yields deterministic result consistent with DEBUG < INFO < WARNING < ERROR < FATAL; verify total, antisymmetric, transitive ordering
    - **Validates: Requirements 1.1**

  - [ ]* 2.3 Write property test: Severity Label Mapping Is Fixed and Total
    - **Property 2: Severity Label Mapping Is Fixed and Total**
    - For any Severity_Level value, verify to_string returns exactly the mandated label; verify distinct levels map to distinct labels
    - **Validates: Requirements 1.6**

- [ ] 3. Source_Location and Log_Record value types
  - [ ] 3.1 Implement logs::Source_Location and logs::Log_Record
    - Create `include/logs/source_location.hpp` with `Source_Location` struct (file, line, function)
    - Create `include/logs/log_record.hpp` with immutable `Log_Record` class: constructor taking severity, message, rank, optional location, context labels, optional stack trace text
    - Implement all const accessors; implement `triggers_abort()` returning true iff severity == FATAL
    - Ensure copy/move defaulted; no mutating operations exposed after construction
    - _Requirements: 1.2, 1.3, 1.5, 1.7_

  - [ ]* 3.2 Write property test: Log_Record Construction Preserves Fields Immutably
    - **Property 3: Log_Record Construction Preserves Fields Immutably**
    - For any severity, message, rank, optional location, and context-label sequence, verify all accessors return values equal to constructor inputs with message preserved byte-for-byte
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 3.3 Write property test: Absent Source Location Is Never Fabricated
    - **Property 4: Absent Source Location Is Never Fabricated**
    - For any Log_Record constructed without source-location, verify stored location is std::nullopt
    - **Validates: Requirements 1.7**

  - [ ]* 3.4 Write property test: FATAL Classification
    - **Property 5: FATAL Classification**
    - For any Log_Record, verify triggers_abort() returns true iff severity is FATAL
    - **Validates: Requirements 1.5**

- [ ] 4. Checkpoint — Foundation types complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Sink output-stream wrapper
  - [ ] 5.1 Implement logs::Sink
    - Create `include/logs/sink.hpp` with `Sink` class wrapping a non-owning `std::ostream*`
    - Implement `write(std::string_view)` returning bool (noexcept); implement `flush()` returning bool (noexcept)
    - Define `MAX_SINKS = 64` compile-time constant
    - Sink never opens, reads, closes, or destroys the stream
    - Create `src/sink.cpp` with implementation
    - _Requirements: 6.1, 6.4, 6.5_

  - [ ]* 5.2 Write property test: Borrowed Stream Lifecycle Is Untouched
    - **Property 25: Borrowed Stream Lifecycle Is Untouched**
    - Verify Sink only writes to and flushes the stream; never opens, closes, or destroys it
    - **Validates: Requirements 6.4, 6.5**

- [ ] 6. Scoped_Context RAII thread-local trace context
  - [ ] 6.1 Implement logs::Scoped_Context and detail::Context_Stack
    - Create `include/logs/detail/context_stack.hpp` with `thread_local std::vector<std::string>` and push/pop/snapshot accessors
    - Create `include/logs/scoped_context.hpp` with `Scoped_Context` class: constructor pushes label (empty → placeholder, >256 chars → truncated), destructor pops (noexcept)
    - Delete copy and move constructors/assignment operators
    - Create `src/scoped_context.cpp` with implementation
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 8.8, 8.9_

  - [ ]* 6.2 Write property test: Scoped_Context LIFO Push/Pop Balance
    - **Property 29: Scoped_Context LIFO Push/Pop Balance**
    - For any nesting of Scoped_Context objects, verify LIFO push/pop restores thread-local stack to pre-construction state at every level
    - **Validates: Requirements 8.1, 8.2, 8.7**

  - [ ]* 6.3 Write property test: Scoped_Context Pops On Exception Unwind
    - **Property 30: Scoped_Context Pops On Exception Unwind**
    - Throw through scope containing Scoped_Context; verify destructor pops label and does not propagate exception
    - **Validates: Requirements 8.3**

  - [ ]* 6.4 Write property test: Context Label Normalization
    - **Property 33: Context Label Normalization**
    - Verify empty label pushes fixed placeholder; verify label >256 chars is truncated to first 256
    - **Validates: Requirements 8.8, 8.9**

  - [ ]* 6.5 Write property test: Context Stack Per-Thread Independence
    - **Property 32: Context Stack Per-Thread Independence**
    - Spawn multiple threads each pushing distinct labels; verify records on one thread never include labels from another
    - **Validates: Requirements 8.6, 11.7**

- [ ] 7. Stack_Frame and format_stack_trace
  - [ ] 7.1 Implement logs::Stack_Frame and format_stack_trace
    - Create `include/logs/stack_trace.hpp` with `Stack_Frame` struct (optional function, file, line) and `format_stack_trace(std::span<const Stack_Frame>)` declaration
    - Define `MAX_TRACE_FRAMES = 256` constant
    - Create `src/stack_trace.cpp` implementing: one line per frame in supplied order, fixed field order, placeholder for missing fields, empty sequence → fixed "no frames" line, >256 frames → first 256 + omission indicator
    - Pure formatting only — reads/opens/parses NOTHING
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.8_

  - [ ]* 7.2 Write property test: Stack-Trace Formatting Determinism and Structure
    - **Property 26: Stack-Trace Formatting Determinism and Structure**
    - For any frame sequence, verify one line per frame, fixed field order, placeholder for missing fields, byte-identical output for byte-identical input
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 7.3 Write property test: Stack-Trace Frame Cap With Omission Indicator
    - **Property 27: Stack-Trace Frame Cap With Omission Indicator**
    - For any sequence >256 frames, verify first 256 rendered followed by exactly one omission line
    - **Validates: Requirements 7.8**

- [ ] 8. Checkpoint — Value types and pure functions complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. MPI interposition test layer
  - [ ] 9.1 Implement MPI interposition spy layer
    - Create `tests/mpi_interposition.hpp` with `MPI_Spy` singleton: call recording for MPI_Comm_rank, MPI_Query_thread, MPI_Abort, MPI_Gather/Allgather (consolidation), plus error injection and reset
    - Implement weak-symbol overrides or link-time interposition for target MPI functions
    - Provide `MPI_Call_Record` struct with type enum, handle, arguments, and relative order
    - _Requirements: 13.5_

  - [ ] 9.2 Create in-memory test Sink
    - Create `tests/in_memory_sink.hpp` with an in-memory Sink that records all written strings for assertion
    - Support thread-safe recording for concurrency tests
    - _Requirements: 13.1_

  - [ ] 9.3 Write unit tests for MPI_Spy layer
    - Verify call recording works for each intercepted MPI function
    - Verify error injection returns configured error codes
    - Verify reset clears all recorded calls
    - _Requirements: 13.5_

- [ ] 10. detail::Mpi_Environment and Serialized_MPI_Guard
  - [ ] 10.1 Implement logs::detail::Mpi_Environment
    - Create `include/logs/detail/mpi_environment.hpp` with `Mpi_Environment` class: atomic rank (default -1), atomic thread_level (default MPI_THREAD_SINGLE), MPI_Comm storage, detect() method
    - Create `src/detail/mpi_environment.cpp` implementing: MPI_Comm_rank (exactly once, failure → -1), MPI_Query_thread (if initialized), conservative fallback
    - _Requirements: 2.1, 2.7, 11.3, 11.9_

  - [ ] 10.2 Implement logs::detail::Serialized_MPI_Guard
    - Create `include/logs/detail/serialized_mpi_guard.hpp` with RAII guard that locks environment serialization mutex when thread level < MPI_THREAD_MULTIPLE
    - _Requirements: 11.4_

  - [ ]* 10.3 Write property test: MPI Thread-Level Detection and Serialization
    - **Property 37: MPI Thread-Level Detection and Serialization**
    - Verify stored thread level matches MPI_Query_thread result; verify MPI_THREAD_SINGLE stored when MPI not initialized; verify serialization mutex engaged when level < MULTIPLE
    - **Validates: Requirements 11.3, 11.4, 11.5, 11.9**

- [ ] 11. Logger central thread-safe entry point
  - [ ] 11.1 Implement logs::Logger core structure
    - Create `include/logs/logger.hpp` with `Logger` class declaration: constructor (noexcept), configure_communicator, set_threshold, add_sink, rank(), thread_support_level(), log(), fatal(), consolidate()
    - Create `src/logger.cpp` implementing:
      - Atomic severity threshold (default INFO, last-writer-wins)
      - Sink collection with mutex (bounded to MAX_SINKS=64, reject beyond)
      - Consolidation buffer with mutex
      - Rank stamping from Mpi_Environment
      - Context snapshot from thread-local stack
      - Severity filtering (accept >= threshold; FATAL never suppressed)
      - Format record with rank at fixed field position
      - Write to each sink exactly once (under sinks_mutex_)
      - Exception Boundary: all public entry points noexcept, absorb internal failures, at most one ERROR diagnostic (non-recursive)
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.7, 3.8, 6.1, 6.2, 6.3, 6.6, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 11.1, 11.2_

  - [ ]* 11.2 Write property test: Rank Stamping Invariance
    - **Property 6: Rank Stamping Invariance**
    - Configure Logger with known rank via interposition; emit records; verify every record carries that rank
    - **Validates: Requirements 2.1, 2.2, 2.5, 2.6**

  - [ ]* 11.3 Write property test: Unidentified-Rank Sentinel Before Configuration
    - **Property 7: Unidentified-Rank Sentinel Before Configuration**
    - Emit records before communicator configured; verify rank == -1
    - **Validates: Requirements 2.4, 2.5**

  - [ ]* 11.4 Write property test: Atomic Threshold Last-Writer-Wins
    - **Property 9: Atomic Threshold Last-Writer-Wins**
    - Concurrent threshold-setting calls; verify no torn values; verify last setter governs
    - **Validates: Requirements 3.1, 3.5, 3.8**

  - [ ]* 11.5 Write property test: Severity Filtering Monotonicity
    - **Property 10: Severity Filtering Monotonicity**
    - For any submission with severity s and threshold t, verify accepted when s >= t, discarded when s < t
    - **Validates: Requirements 3.2, 3.3**

  - [ ]* 11.6 Write property test: FATAL Is Never Suppressed
    - **Property 11: FATAL Is Never Suppressed**
    - For any threshold including FATAL, verify FATAL submissions are always accepted
    - **Validates: Requirements 3.7**

  - [ ]* 11.7 Write property test: Record Formatting Round-Trip
    - **Property 8: Record Formatting Round-Trip**
    - Format a record to text; extract fields at fixed positions; verify rank, severity label, and message recovered
    - **Validates: Requirements 2.3, 1.6**

  - [ ]* 11.8 Write property test: Public Entry Points Never Propagate Exceptions
    - **Property 34: Public Entry Points Never Propagate Exceptions**
    - Inject internal failures; verify no exception propagates to caller; verify state remains valid
    - **Validates: Requirements 9.1, 9.2, 9.7**

  - [ ]* 11.9 Write property test: Bounded Non-Recursive Absorbed-Failure Diagnostic
    - **Property 35: Bounded, Non-Recursive Absorbed-Failure Diagnostic**
    - Inject exception during logging; verify at most one ERROR diagnostic attempted; verify recursive failure silently discarded
    - **Validates: Requirements 9.3, 9.6**

- [ ] 12. Checkpoint — Logger core complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Consolidation_Key, Rank_Range, and compact_ranges
  - [ ] 13.1 Implement detail::Consolidation_Key, Rank_Range, and compact_ranges
    - Create `include/logs/detail/consolidation.hpp` with:
      - `Consolidation_Key` struct: severity + message, defaulted operator==, excludes rank
      - `Rank_Range` struct: first, last, to_string()
      - `compact_ranges(std::vector<int>)` function: sort, deduplicate, partition into maximal contiguous spans, return ordered by ascending first
      - `Consolidated_Record` struct: key, rank_count, ranges
    - Create `src/detail/consolidation.cpp` with compact_ranges implementation
    - _Requirements: 4.1, 4.3, 4.4, 4.6_

  - [ ]* 13.2 Write property test: Consolidation Key Excludes Rank
    - **Property 12: Consolidation Key Excludes Rank**
    - For any two records, verify keys equal iff severity and message equal regardless of rank
    - **Validates: Requirements 4.1**

  - [ ]* 13.3 Write property test: Consolidated Rank Partitioning Into Ordered Contiguous Spans
    - **Property 14: Consolidated Rank Partitioning Into Ordered Contiguous Spans**
    - For any set of contributing ranks, verify compact_ranges produces maximal contiguous spans covering exactly those ranks, non-overlapping, non-adjacent, ordered ascending
    - **Validates: Requirements 4.3, 4.4, 4.6**

- [ ] 14. Consolidation_Engine collective and local paths
  - [ ] 14.1 Implement detail::Consolidation_Engine
    - Add `Consolidation_Engine` class to `include/logs/detail/consolidation.hpp` and `src/detail/consolidation.cpp`
    - Implement `consolidate_collective`: use MPI gather for keys + ranks only (no other payload), group by key, compact_ranges, produce representatives on root rank 0 only
    - Implement `consolidate_local`: consolidate locally, annotate with sentinel rank -1 Rank_Range, issue no MPI
    - Use `Serialized_MPI_Guard` for all MPI calls
    - _Requirements: 4.2, 4.5, 4.7, 4.8, 4.9_

  - [ ] 14.2 Wire Consolidation_Engine into Logger::consolidate()
    - Implement `Logger::consolidate()`: collective when communicator configured, local fallback otherwise; emit representatives to sinks on root only; clear buffer after operation
    - _Requirements: 4.2, 4.7, 4.9_

  - [ ]* 14.3 Write property test: Exactly One Root-Written Representative Per Key
    - **Property 13: Exactly One Root-Written Representative Per Key**
    - Inject records from multiple ranks sharing a key; verify one representative emitted on root only
    - **Validates: Requirements 4.2**

  - [ ]* 14.4 Write property test: Consolidation Preserves Severity and Message
    - **Property 15: Consolidation Preserves Severity and Message**
    - Verify representative preserves original severity and message unchanged
    - **Validates: Requirements 4.8**

  - [ ]* 14.5 Write property test: Consolidation Transports Only Keys and Ranks
    - **Property 16: Consolidation Transports Only Keys and Ranks**
    - Via MPI spy, verify only keys and ranks transported; no location, stack trace, or context labels
    - **Validates: Requirements 4.5**

  - [ ]* 14.6 Write property test: Consolidation Buffer Lifecycle
    - **Property 17: Consolidation Buffer Lifecycle**
    - Verify consolidation processes exactly buffered records then clears; subsequent consolidation produces zero representatives
    - **Validates: Requirements 4.7**

  - [ ]* 14.7 Write property test: No-Communicator Local Consolidation Fallback
    - **Property 18: No-Communicator Local Consolidation Fallback**
    - Without communicator configured, verify local consolidation, sentinel -1 Rank_Range, no MPI issued
    - **Validates: Requirements 4.9**

- [ ] 15. Synchronized Abort (detail::Abort_Latch + Logger::fatal)
  - [ ] 15.1 Implement detail::Abort_Latch and Synchronized_Abort path
    - Add `Abort_Latch` to `include/logs/detail/` (or inline in logger): atomic bool with try_acquire() returning true exactly once
    - Implement `Logger::fatal()`: format FATAL record → dispatch to all sinks → flush all sinks (absorb failures) → acquire latch → MPI_Abort(comm, ABORT_EXIT_CODE) with ABORT_EXIT_CODE in [1,255]
    - No-communicator path: flush then `std::exit(ABORT_EXIT_CODE)`
    - Concurrent/subsequent FATALs suppressed by latch (MPI_Abort at most once)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 11.6_

  - [ ]* 15.2 Write property test: FATAL Format-and-Flush Precede Abort
    - **Property 19: FATAL Format-and-Flush Precede Abort**
    - Via MPI spy, verify all sinks written and flushed before MPI_Abort invoked
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 15.3 Write property test: Abort Exit Code Is a Fixed Value in 1..255
    - **Property 20: Abort Exit Code Is a Fixed Value in 1..255**
    - Via MPI spy, verify MPI_Abort called with deterministic exit code in [1,255]
    - **Validates: Requirements 5.3, 5.5**

  - [ ]* 15.4 Write property test: MPI_Abort Occurs At Most Once Per Process
    - **Property 21: MPI_Abort Occurs At Most Once Per Process**
    - Submit multiple concurrent FATALs; verify via spy that MPI_Abort invoked at most once
    - **Validates: Requirements 5.8, 11.6**

  - [ ]* 15.5 Write property test: Synchronized_Abort Is the Sole Termination Path
    - **Property 22: Synchronized_Abort Is the Sole Termination Path**
    - For any sequence of non-FATAL operations, verify MPI_Abort never invoked
    - **Validates: Requirements 5.6, 9.5**

- [ ] 16. Checkpoint — Core library feature-complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 17. Sink dispatch, write-exactly-once, and failure isolation
  - [ ] 17.1 Write unit tests for Sink dispatch semantics
    - Test: with k configured sinks (0 < k <= 64), verify formatted record written exactly once to each
    - Test: with zero configured sinks, verify record written to default stderr sink
    - Test: with one failing sink, verify record still written to remaining sinks; failing sink retained
    - Test: attempt to add sink beyond MAX_SINKS=64, verify rejection absorbed without exception
    - _Requirements: 6.1, 6.2, 6.3, 6.6, 6.7_

  - [ ]* 17.2 Write property test: Sink Write-Exactly-Once and Bounded Cardinality
    - **Property 23: Sink Write-Exactly-Once and Bounded Cardinality**
    - For any k sinks and accepted record, verify exactly one write per sink; zero sinks → stderr; count <= 64
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [ ]* 17.3 Write property test: Sink Write-Failure Isolation
    - **Property 24: Sink Write-Failure Isolation**
    - Inject write failure in one sink; verify exception absorbed, remaining sinks receive record, failed sink retained
    - **Validates: Requirements 6.6, 6.7**

- [ ] 18. Conditional Stack-Trace attachment and Context capture
  - [ ] 18.1 Write unit tests for stack-trace attachment and context capture
    - Test: with include_stack_trace=true and frames supplied, verify Log_Record carries stack_trace text
    - Test: with include_stack_trace=false, verify Log_Record carries no stack_trace text
    - Test: with active Scoped_Context labels, verify Log_Record carries labels outermost-to-innermost
    - Test: with no active labels, verify empty context-label sequence
    - _Requirements: 7.6, 7.7, 1.4, 8.4, 8.10_

  - [ ]* 18.2 Write property test: Conditional Stack-Trace Attachment
    - **Property 28: Conditional Stack-Trace Attachment**
    - For any submission, verify stack-trace present iff caller requested inclusion
    - **Validates: Requirements 7.6, 7.7**

  - [ ]* 18.3 Write property test: Context Capture Ordered Outermost-to-Innermost
    - **Property 31: Context Capture Ordered Outermost-to-Innermost**
    - Verify active labels carried in outermost-to-innermost order; empty when no context active
    - **Validates: Requirements 1.4, 8.4, 8.10**

- [ ] 19. Thread-safety concurrency test suite
  - [ ] 19.1 Write concurrency unit tests
    - Test: spawn 8+ threads each emitting 10,000+ records to shared in-memory sink; verify total count == sum across threads; verify no lost/duplicated/corrupted records
    - Test: concurrent threshold-setting from multiple threads; verify no torn reads
    - Test: concurrent Scoped_Context push/pop on separate threads; verify per-thread independence
    - _Requirements: 13.1, 11.1, 11.2, 11.8_

  - [ ]* 19.2 Write property test: Concurrent Record-Count Integrity and Non-Interleaving
    - **Property 36: Concurrent Record-Count Integrity and Non-Interleaving**
    - Multi-threaded submissions; verify emitted count == submitted count; verify no interleaving of one record's text with another
    - **Validates: Requirements 11.1, 11.2, 11.8**

- [ ] 20. Checkpoint — All functional tests complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 21. Google Test verification suite (requirement-mandated tests)
  - [ ] 21.1 Write rank-stamping verification test
    - Configure Logger with known rank via interposition; emit 100+ records; verify every record carries configured rank; verify zero records carry any other rank
    - _Requirements: 13.2_

  - [ ] 21.2 Write consolidation contiguous-span test
    - Inject identical records from contiguous span of 4+ ranks (R through R+N-1); consolidate; verify one representative with count == N and Rank_Range [R, R+N-1]
    - _Requirements: 13.3_

  - [ ] 21.3 Write consolidation distinct-keys test
    - Inject records with 2+ distinct Consolidation_Keys; consolidate; verify representative count == distinct key count; verify no cross-key contamination
    - _Requirements: 13.4_

  - [ ] 21.4 Write FATAL abort verification test
    - Emit single FATAL record; verify via spy: all sinks flushed before MPI_Abort; MPI_Abort called once; exit code in [1,255]
    - _Requirements: 13.6_

  - [ ] 21.5 Write Scoped_Context exception-unwind test
    - Enter Scoped_Context, throw std::exception, catch; verify stack depth and labels restored to pre-context state
    - _Requirements: 13.7_

  - [ ] 21.6 Write formatting round-trip test
    - Format Log_Record with known severity, rank, message; extract from fixed positions; verify equality
    - _Requirements: 13.8_

  - [ ] 21.7 Write single-rank consolidation test
    - Inject identical records from single rank; consolidate; verify one representative with count==1 and Rank_Range [r,r]
    - _Requirements: 13.9_

  - [ ] 21.8 Write non-FATAL no-abort test
    - Emit records at every severity below FATAL; verify via spy MPI_Abort never invoked
    - _Requirements: 13.10_

- [ ] 22. Umbrella header and Tier 1 isolation verification
  - [ ] 22.1 Create umbrella header
    - Create `include/logs/logs.hpp` that includes all public headers (severity, source_location, log_record, sink, scoped_context, stack_trace, logger)
    - Verify no `#include` directives reference TICK, HALO, AXIS, AMIO, SPAN, or DAGR headers
    - Verify only C++ standard library and MPI headers included in public headers
    - _Requirements: 14.1, 14.3_

- [ ] 23. CI pipeline scripts
  - [ ] 23.1 Create Tier 1 isolation scan script
    - Write `cmake/check_tier1_isolation.sh` that scans all LOGS source and header files for `#include` directives matching forbidden HELM component path segments (case-insensitive: tick/, halo/, axis/, amio/, span/, dagr/ — but NOT bare `<span>`)
    - Fail with non-zero exit status and report offending files if found; report passed otherwise
    - Integrate as CTest test or CMake custom target
    - _Requirements: 14.1, 14.5, 14.6, 14.7_

  - [ ] 23.2 Create no-input-parsing scan script
    - Write a static verification script that scans all LOGS source/header files for input-parsing indicators: YAML/markup-parsing library references, namelist/config-format readers, file-open-for-read or read-mode file APIs
    - Fail with non-zero exit and indicate offending file if found; report passed otherwise
    - Integrate as CTest test or CMake custom target
    - _Requirements: 10.5, 10.6, 10.7, 10.8_

  - [ ] 23.3 Document CI pipeline configuration
    - Document CI stages: static analysis (isolation scan + no-input-parsing scan), standalone CMake build inside Docker, unit tests (mpirun -np 4), property tests (single-rank mocked MPI), ASan + UBSan sanitizer builds
    - Verify standalone build produces HELM::LOGS without other HELM source trees present
    - _Requirements: 15.2, 15.3, 15.7_

- [ ] 24. Final checkpoint — All components integrated and verified
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical boundaries
- Property tests validate universal correctness properties (Properties 1–37 from design)
- Unit tests validate specific examples, edge cases, and concurrency guarantees
- The MPI interposition spy layer (task 9.1) is a prerequisite for deterministic property testing of rank stamping, consolidation, and abort
- RapidCheck is used for C++ property-based tests with minimum 100 iterations per property
- LOGS has NO Kokkos dependency and NO Fortran interface — these are explicit architectural decisions
- All run-time behavior is configured through programmatic API calls; LOGS never reads any file or parses any input

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "9.1", "9.2"] },
    { "id": 3, "tasks": ["2.2", "2.3", "3.1", "9.3"] },
    { "id": 4, "tasks": ["3.2", "3.3", "3.4", "5.1"] },
    { "id": 5, "tasks": ["5.2", "6.1"] },
    { "id": 6, "tasks": ["6.2", "6.3", "6.4", "6.5", "7.1"] },
    { "id": 7, "tasks": ["7.2", "7.3", "10.1", "10.2"] },
    { "id": 8, "tasks": ["10.3", "11.1"] },
    { "id": 9, "tasks": ["11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "11.9"] },
    { "id": 10, "tasks": ["13.1"] },
    { "id": 11, "tasks": ["13.2", "13.3", "14.1"] },
    { "id": 12, "tasks": ["14.2", "14.3", "14.4", "14.5", "14.6", "14.7"] },
    { "id": 13, "tasks": ["15.1"] },
    { "id": 14, "tasks": ["15.2", "15.3", "15.4", "15.5"] },
    { "id": 15, "tasks": ["17.1", "17.2", "17.3", "18.1", "18.2", "18.3"] },
    { "id": 16, "tasks": ["19.1", "19.2"] },
    { "id": 17, "tasks": ["21.1", "21.2", "21.3", "21.4", "21.5", "21.6", "21.7", "21.8"] },
    { "id": 18, "tasks": ["22.1", "23.1", "23.2", "23.3"] }
  ]
}
```
