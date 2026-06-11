# Requirements Document

## Introduction

LOGS (Lightweight Operational Global Status) is a Tier 1 C++20 micro-library within the HELM (Heterogeneous Execution & Load-balancing Manager) ecosystem. LOGS provides a highly optimized, thread-safe operational status, logging, and error-handling utility for distributed High-Performance Computing (HPC) clusters. It stamps every log record with the originating MPI rank, formats stack traces, consolidates redundant error messages emitted across thousands of MPI ranks into single representative entries to prevent filesystem I/O storms, and coordinates synchronized aborts on fatal conditions.

LOGS replaces the legacy ESMF logging and error-handling surface — the `ESMF_LogWrite` message-emission routine, the `ESMF_LogErr`/`ESMF_LogFoundError` error-reporting helpers, the `ESMF_LogSet` configuration routine, and the per-PET multi-rank log-file model — with a modern, stateless, target-centric design. As a Tier 1 component, LOGS is completely blind to every other HELM library (TICK, HALO, AXIS, AMIO, SPAN, DAGR) and to domain science code. LOGS is strictly an OUTPUT and STATUS mechanism: it contains no input-parsing logic whatsoever.

### Assumptions

This specification was authored without access to a live ESMF reference. The legacy-API references below are grounded in established general knowledge of the ESMF logging surface, and the following assumptions are stated explicitly:

- **ESMF behavioral model assumed**: ESMF is understood to expose `ESMF_LogWrite` (emit a message at a log-message type such as Info, Warning, or Error), `ESMF_LogSet`/`ESMF_LogGet` (configure logging behavior such as flush frequency and the active log object), and error helpers such as `ESMF_LogErr`/`ESMF_LogFoundError` (record and detect error conditions with return codes). LOGS reproduces the operational intent of these routines, not their exact signatures.
- **Per-PET log-file pathology assumed**: ESMF is understood to default to one log file per PET (Persistent Execution Thread / MPI process). At very large rank counts (e.g., 10,000 ranks) a simultaneous failure produces a filesystem denial-of-service through lock contention and an I/O storm. Eliminating this pathology via cross-rank consolidation is the central motivation for LOGS.
- **MPI usage scope assumed**: LOGS uses MPI ONLY for rank identification, cross-rank message consolidation coordination, and synchronized abort (MPI_Abort). LOGS performs no halo exchange, collective compute, or data transport; those concerns belong to HALO.
- **No GPU/compute role assumed**: LOGS is a status/output utility and performs no numerical computation or device offload. Consequently LOGS requires no Kokkos dependency; this is an explicit design choice recorded in the build-system requirements.
- **MPI initialization assumed external**: The host application (ultimately DAGR or a test harness) is assumed to call `MPI_Init` / `MPI_Init_thread` and `MPI_Finalize`. LOGS does not own the MPI lifecycle.

## Glossary

- **LOGS**: Lightweight Operational Global Status; the Tier 1 thread-safe logging, status, and error-handling micro-library within the HELM ecosystem.
- **Logger**: The central thread-safe entry point of LOGS that accepts log submissions, applies severity filtering, performs rank stamping, dispatches records to sinks, and coordinates consolidation and synchronized abort.
- **Log_Record**: An immutable value object representing a single log submission, carrying at minimum a severity level, a message payload, the originating MPI rank, and an optional source-location and scoped-context annotation.
- **Severity_Level**: An ordered enumeration of message importance. The defined levels in ascending order of severity are DEBUG, INFO, WARNING, ERROR, and FATAL.
- **FATAL**: The highest Severity_Level, which designates an unrecoverable condition and triggers a Synchronized_Abort.
- **Severity_Threshold**: A programmatically configured minimum Severity_Level; records whose severity is below the threshold are suppressed (not formatted and not dispatched to any Sink).
- **Sink**: An output destination abstraction to which formatted Log_Records are written. Permitted sinks are output streams only (e.g., stderr, stdout, or an application-supplied file stream). A Sink never reads or parses input.
- **MPI_Rank**: The integer identifier of the calling process within an MPI communicator, obtained via MPI_Comm_rank. Referred to as "rank".
- **PET**: Persistent Execution Thread; the legacy ESMF abstraction for an MPI process, equivalent to one MPI_Rank in the LOGS model.
- **Rank_Stamping**: The act of annotating every Log_Record with the originating MPI_Rank for traceability.
- **Consolidation**: The act of combining multiple Log_Records that share an identical Consolidation_Key but originate from different ranks into a single representative Log_Record annotated with the contributing rank count and rank ranges. Also referred to as throttling.
- **Consolidation_Key**: The tuple of fields used to determine whether two Log_Records are considered identical for consolidation purposes, comprising the Severity_Level and the message payload, and explicitly EXCLUDING the originating MPI_Rank.
- **Rank_Range**: A compact representation of a contiguous span of contributing ranks (e.g., "0-9999") used to summarize which ranks emitted a consolidated message.
- **Synchronized_Abort**: A coordinated termination of the distributed job initiated when a FATAL record is emitted, performed by flushing pending output and invoking MPI_Abort on the configured communicator with a non-zero exit code.
- **Scoped_Context**: An RAII object that pushes a named trace-context label onto a thread-local context stack at construction and pops it at destruction, so that emitted Log_Records carry the active trace context. Also referred to as a scoped trace context.
- **Stack_Trace**: A textual rendering of the active call frames at the point of capture. In LOGS, stack-trace handling is a pure formatting operation over already-captured frame data.
- **RAII**: Resource Acquisition Is Initialization; a C++ idiom where resource lifetime is bound to object scope, with acquisition in the constructor and release in the destructor.
- **Exception_Boundary**: The set of public LOGS entry points that are contractually guaranteed not to propagate exceptions to the caller; failures are absorbed or routed through the controlled FATAL abort path instead.
- **MPI_THREAD_MULTIPLE**: The MPI threading level at which multiple threads may call MPI concurrently without external serialization.
- **Interposition_Layer**: A mock, spy, or instrumented substitute for MPI symbols used by the test suite to observe and control MPI behavior (rank identity, MPI_Abort invocation) deterministically without a live MPI runtime.
- **Tier 1**: The HELM architectural tier of stateless foundational micro-libraries (TICK, LOGS, HALO, AXIS) that are mutually blind and free of domain-science dependencies.

## Requirements

### Requirement 1: Severity Levels and Log Record Model

**User Story:** As a library developer, I want a well-defined, ordered severity scheme and an immutable log record type, so that messages can be classified, filtered, and consolidated consistently across the distributed job.

#### Acceptance Criteria

1. THE LOGS library SHALL define a Severity_Level enumeration containing exactly the five levels DEBUG, INFO, WARNING, ERROR, and FATAL, and SHALL define a total ordering over those levels such that DEBUG < INFO < WARNING < ERROR < FATAL, so that any two Severity_Level values can be compared to yield a deterministic ordering result.
2. WHEN a Log_Record is constructed, THE Logger SHALL populate the record with exactly one Severity_Level value, the caller-supplied message payload as a text string preserved verbatim, and the originating MPI_Rank as an integer, and THE Log_Record SHALL expose no operation that modifies any of these fields after construction.
3. WHERE the caller provides a source-location annotation, THE Log_Record SHALL store that annotation comprising the file name as a text string, the line number as an integer greater than or equal to 1, and the function name as a text string.
4. WHEN a Log_Record is constructed for submission, THE Logger SHALL store within the record the ordered sequence of Scoped_Context labels active on the submitting thread at construction time, ordered from outermost to innermost scope, and SHALL store an empty sequence when no Scoped_Context label is active on that thread.
5. WHEN a Log_Record is submitted at Severity_Level FATAL, THE Logger SHALL classify the record as triggering a Synchronized_Abort.
6. THE LOGS library SHALL provide a function that maps each Severity_Level value to its fixed, human-readable text label, returning "DEBUG" for DEBUG, "INFO" for INFO, "WARNING" for WARNING, "ERROR" for ERROR, and "FATAL" for FATAL.
7. IF the caller does not provide a source-location annotation, THEN THE Log_Record SHALL record the source-location annotation as absent rather than populating the file name, line number, or function name with fabricated values.

### Requirement 2: MPI Rank Identification and Rank Stamping

**User Story:** As an operations engineer, I want every log record stamped with the MPI rank that produced it, so that I can trace any message back to the exact process on a 10,000-rank cluster.

#### Acceptance Criteria

1. WHEN the Logger is initialized with an MPI communicator, THE Logger SHALL call MPI_Comm_rank exactly once to determine and store the calling process MPI_Rank as an integer in the range 0 to the communicator process count minus 1, inclusive.
2. WHEN a Log_Record is created by the Logger, THE Logger SHALL stamp the record with the stored MPI_Rank before any filtering, formatting, or dispatch occurs.
3. WHEN a Log_Record is formatted for a Sink, THE Logger SHALL include the originating MPI_Rank in the formatted output at a fixed field position from which the integer rank is recoverable by position.
4. WHEN a Log_Record is created before an MPI communicator has been configured, THE Logger SHALL stamp the record with a sentinel MPI_Rank value of -1 indicating an unidentified rank.
5. THE Logger SHALL expose a thread-safe accessor that returns the stored MPI_Rank as an integer, and SHALL return the sentinel value -1 from that accessor while no MPI communicator has been configured.
6. WHILE the stored MPI_Rank is valid (greater than or equal to 0), THE Logger SHALL use that identical rank value for every Log_Record produced on every thread of the process.
7. IF MPI_Comm_rank fails during Logger initialization, THEN THE Logger SHALL set the stored MPI_Rank to the sentinel value -1, route the failure through the Exception_Boundary, and not propagate the failure to the caller.

### Requirement 3: Severity-Based Filtering

**User Story:** As a domain scientist, I want to set a minimum severity threshold programmatically through the API, so that I can suppress low-importance messages at run time without editing any configuration file.

#### Acceptance Criteria

1. WHEN the threshold-setting API function is invoked with a Severity_Level argument equal to one of the defined values DEBUG, INFO, WARNING, ERROR, or FATAL, THE Logger SHALL set the active Severity_Threshold to that value as a single atomic update that is safe for concurrent invocation from multiple threads and yields no torn or partially updated threshold.
2. WHEN a Log_Record is submitted with a Severity_Level greater than or equal to the active Severity_Threshold, THE Logger SHALL accept the record for formatting and dispatch.
3. WHEN a Log_Record is submitted with a Severity_Level strictly below the active Severity_Threshold, THE Logger SHALL discard the record without formatting it and without dispatching it to any Sink.
4. WHERE no Severity_Threshold has been set, THE Logger SHALL apply a default Severity_Threshold of INFO.
5. THE Logger SHALL provide a thread-safe accessor that returns the active Severity_Threshold.
6. THE Logger SHALL configure the Severity_Threshold exclusively through programmatic API calls and SHALL NOT read the threshold from any file, environment-parsed document, or external configuration source.
7. WHEN a Log_Record is submitted at Severity_Level FATAL, THE Logger SHALL accept the record for formatting and dispatch regardless of the active Severity_Threshold, including when the active Severity_Threshold equals FATAL, so that fatal conditions are never suppressed.
8. WHEN a threshold-setting call returns successfully, THE Logger SHALL apply the Severity_Threshold value established by that call to every Log_Record whose submission begins after the call returns, and SHALL retain that value until a subsequent threshold-setting call returns successfully, such that the most recent successful call governs the active Severity_Threshold (last-writer-wins).

### Requirement 4: Multi-Rank Message Consolidation

**User Story:** As an operations engineer, I want identical error messages emitted simultaneously across thousands of ranks consolidated into one representative entry, so that a mass failure does not trigger a filesystem denial-of-service through lock contention and an I/O storm.

#### Acceptance Criteria

1. THE Logger SHALL compute a Consolidation_Key for each accepted Log_Record from the Severity_Level and the message payload, excluding the originating MPI_Rank from the key.
2. WHEN a consolidation operation processes multiple Log_Records that share an identical Consolidation_Key and originate from two or more distinct ranks, THE Logger SHALL emit exactly one representative Log_Record for that Consolidation_Key and SHALL write that representative to the configured sinks on exactly one designated root rank (the process whose MPI_Rank is 0 in the configured communicator), such that no other rank writes that representative to any Sink.
3. WHEN a representative consolidated Log_Record is emitted, THE Logger SHALL annotate the record with the total count of contributing ranks and a set of Rank_Ranges identifying which ranks contributed.
4. WHEN the contributing ranks of a representative Log_Record are annotated, THE Logger SHALL partition the contributing ranks into maximal contiguous spans, represent each span as a single Rank_Range expressed as "first-last" (with first equal to last for a span of one rank), and order the resulting Rank_Ranges by ascending first rank, rather than enumerating each rank individually.
5. WHEN the Logger consolidates records across ranks, THE Logger SHALL use MPI solely to gather Consolidation_Keys and contributing ranks and SHALL NOT transport any other payload.
6. IF only a single rank contributes a Log_Record for a given Consolidation_Key, THEN THE Logger SHALL emit that record annotated with a contributing-rank count of 1 and a single-rank Rank_Range.
7. THE Logger SHALL provide an API function that triggers a single collective consolidation operation across the configured communicator at a caller-chosen point, that every rank of the communicator must invoke collectively, that operates over the set of accepted Log_Records buffered since the most recent prior consolidation operation (or since Logger initialization if none has occurred), and that clears that buffer once the operation completes.
8. WHEN consolidation produces a representative Log_Record, THE Logger SHALL preserve the original Severity_Level and message payload of the contributing records unchanged in the representative record.
9. IF the consolidation API function is invoked when no MPI communicator has been configured, THEN THE Logger SHALL consolidate only the Log_Records buffered on the calling process, emit the resulting representative records to the configured sinks on that process annotated with a single-rank Rank_Range bearing the sentinel rank value -1, clear the local buffer, and SHALL NOT issue any MPI communication.

### Requirement 5: Synchronized Abort on Fatal Severity

**User Story:** As a library developer, I want a FATAL message to deterministically terminate the entire distributed job through a coordinated abort, so that an unrecoverable error stops all ranks instead of leaving zombie processes.

#### Acceptance Criteria

1. WHEN a Log_Record at Severity_Level FATAL is submitted, THE Logger SHALL format the record and dispatch it to all configured sinks before initiating any termination action.
2. WHEN a FATAL record has been dispatched to all configured sinks, THE Logger SHALL flush every configured Sink so that the fatal message is written out before any termination action is initiated.
3. WHEN flushing of all configured sinks for a FATAL record completes, THE Logger SHALL initiate a Synchronized_Abort by invoking MPI_Abort on the configured communicator with a fixed, non-zero exit code in the inclusive range 1 to 255.
4. WHERE no MPI communicator has been configured at the time a FATAL record is submitted, THE Logger SHALL flush all configured sinks and then terminate the local process with the same fixed, non-zero exit code used for the Synchronized_Abort.
5. THE Logger SHALL provide an API function that emits a FATAL Log_Record and initiates the Synchronized_Abort in a single call.
6. THE Logger SHALL treat the Synchronized_Abort as the sole controlled termination path and SHALL route no other code path to MPI_Abort.
7. IF a Sink flush fails while handling a FATAL record, THEN THE Logger SHALL absorb the failure without propagating an exception to the caller, consistent with the Exception_Boundary, and SHALL still proceed to initiate the Synchronized_Abort.
8. WHILE a Synchronized_Abort triggered by a FATAL Log_Record is in progress, THE Logger SHALL suppress initiation of any further Synchronized_Abort from concurrently or subsequently submitted FATAL records so that MPI_Abort is invoked no more than once per process.

### Requirement 6: Output Sinks

**User Story:** As a library developer, I want log records written to configurable output streams, so that operational status can be directed to stderr, stdout, or an application-owned file stream without LOGS owning file lifecycle or parsing any input.

#### Acceptance Criteria

1. THE Logger SHALL accept up to a maximum of 64 configured Sinks, where each Sink is an output destination wrapping exactly one standard output stream such as stderr, stdout, or a caller-supplied std::ostream.
2. WHEN a Log_Record passes severity filtering, THE Logger SHALL write the formatted record exactly once to each configured Sink.
3. WHILE no Sink has been configured, WHEN a Log_Record passes severity filtering, THE Logger SHALL write the formatted record to stderr as the default Sink.
4. THE Sink SHALL perform write and flush operations only and SHALL NOT open, read, or parse any file or input source.
5. WHEN a caller supplies a file-backed output stream as a Sink, THE Logger SHALL write to that stream and SHALL NOT open, close, or destroy the stream, leaving the caller solely responsible for the stream lifecycle.
6. IF a write to one configured Sink fails, THEN THE Logger SHALL absorb the failure without propagating an exception to the submitting caller, SHALL continue writing the record to the remaining configured Sinks, and SHALL leave the failed Sink configured for subsequent records, consistent with the Exception_Boundary.
7. IF a caller attempts to configure a Sink beyond the maximum of 64, THEN THE Logger SHALL reject the additional Sink, retain the existing configured Sinks unchanged, and absorb the rejection without propagating an exception to the caller, consistent with the Exception_Boundary.

### Requirement 7: Stack-Trace Formatting

**User Story:** As a domain scientist, I want stack traces rendered into readable text when an error is logged, so that I can diagnose failures, while trusting that LOGS never parses external files to do so.

#### Acceptance Criteria

1. WHEN the Logger is given a sequence of captured stack frames, THE Logger SHALL format the frames into a multi-line text rendering that contains exactly one line per supplied frame, ordered to match the order in which the frames are supplied, such that any two identical frame sequences always produce byte-identical rendered text.
2. THE stack-trace formatting operation SHALL be a pure formatting transformation over the supplied frame data and SHALL NOT read, open, or parse any file, symbol table file, or external input source.
3. WHEN a stack frame includes a function name, a source file name, and a line number, THE Logger SHALL include all three fields in that frame's line using the same fixed field order applied to every frame in the rendering.
4. IF a stack frame is missing its function name, source file name, or line number, THEN THE Logger SHALL render a fixed placeholder token in place of each missing field and SHALL still render that frame's line rather than omitting the frame.
5. WHEN the supplied frame sequence is empty, THE Logger SHALL produce a fixed single-line indication that no frames are available.
6. WHEN the submitting caller requests stack-trace inclusion for a Log_Record, THE Logger SHALL attach the formatted stack-trace text to that Log_Record.
7. WHEN the submitting caller does not request stack-trace inclusion for a Log_Record, THE Logger SHALL NOT attach any stack-trace text to that Log_Record.
8. WHEN the supplied frame sequence contains more than 256 frames, THE Logger SHALL render the first 256 frames in supplied order and SHALL append a fixed single-line indication that the remaining frames were omitted.

### Requirement 8: RAII Scoped Logging Context

**User Story:** As a domain scientist, I want a scoped trace-context object that annotates all logs emitted within its scope, so that nested operations are traceable and context is removed automatically even when an exception unwinds the stack.

#### Acceptance Criteria

1. WHEN a Scoped_Context is constructed with a non-empty context label of 256 characters or fewer, THE Scoped_Context SHALL push that label onto the thread-local context stack as the innermost (most recently pushed) entry.
2. WHEN a Scoped_Context is destroyed, THE Scoped_Context SHALL pop the single label it pushed from the thread-local context stack, restoring the stack to the exact set and order of labels present immediately before that Scoped_Context was constructed.
3. IF an exception unwinds the stack through a scope containing a Scoped_Context, THEN THE Scoped_Context destructor SHALL pop the label it pushed from the thread-local context stack and SHALL NOT propagate any exception.
4. WHEN a Log_Record is submitted on a thread with one or more active Scoped_Context labels, THE Logger SHALL annotate the record with the active labels ordered from outermost (earliest constructed and still active) to innermost (most recently constructed and still active).
5. THE Scoped_Context SHALL delete its copy constructor, copy assignment operator, move constructor, and move assignment operator so that each pushed label is owned by exactly one Scoped_Context object bound to its constructing scope.
6. THE Scoped_Context thread-local context stack SHALL be independent per thread, such that labels pushed by Scoped_Context objects on one thread are never included in the annotations of Log_Records submitted on any other thread.
7. WHEN two or more Scoped_Context objects are nested within a single thread, THE Scoped_Context SHALL maintain last-in-first-out ordering of the thread-local context stack such that destroying them in reverse construction order restores each prior context state exactly.
8. IF a Scoped_Context is constructed with an empty context label, THEN THE Scoped_Context SHALL push a fixed placeholder label in place of the empty label so that the push and pop balance of the thread-local context stack is preserved.
9. IF a Scoped_Context is constructed with a context label exceeding 256 characters, THEN THE Scoped_Context SHALL truncate the label to its first 256 characters before pushing it onto the thread-local context stack.
10. WHEN a Log_Record is submitted on a thread with no active Scoped_Context labels, THE Logger SHALL annotate the record with an empty context-label sequence.

### Requirement 9: Exception Boundary at Public Entry Points

**User Story:** As a library developer, I want the public logging entry points to never let an exception escape and crash the logger itself, so that logging failures degrade gracefully and only the controlled FATAL path terminates the job.

#### Acceptance Criteria

1. THE public logging entry points of the Logger SHALL be declared and implemented so that no exception propagates to the caller across the Exception_Boundary.
2. IF an internal operation within a public logging entry point throws an exception, THEN THE Logger SHALL catch the exception, cease processing the offending submission without retrying it, and return control to the caller normally without propagating the exception across the Exception_Boundary.
3. WHEN a public logging entry point catches an internal exception, THE Logger SHALL attempt to emit at most one diagnostic record describing the absorbed failure at Severity_Level ERROR, so that diagnostic emission cannot recurse.
4. THE Scoped_Context destructor and every Sink flush operation SHALL be implemented as noexcept so that destructor-time and termination-time failures cannot propagate.
5. THE only public LOGS code path permitted to terminate the process SHALL be the Synchronized_Abort triggered by a FATAL Log_Record.
6. IF the attempt to emit the diagnostic record described in criterion 3 itself throws an exception, THEN THE Logger SHALL discard that diagnostic silently, make no further attempt to emit it, and SHALL NOT propagate the exception across the Exception_Boundary.
7. WHEN the Logger absorbs an exception at a public logging entry point, THE Logger SHALL leave its configured sinks, active Severity_Threshold, and consolidation buffers in a valid and usable state so that subsequent Log_Record submissions are processed normally.
8. IF an internal operation throws an exception while a Synchronized_Abort triggered by a FATAL Log_Record is in progress, THEN THE Logger SHALL absorb the exception without propagating it across the Exception_Boundary and SHALL still proceed to terminate the process with a non-zero exit code.

### Requirement 10: Strict Separation of Concerns — No Input Parsing

**User Story:** As an architect, I want LOGS to be strictly an output and status mechanism with zero input-parsing logic, so that configuration and file-format concerns remain entirely outside this Tier 1 utility.

#### Acceptance Criteria

1. THE LOGS library SHALL NOT contain any YAML parsing logic, namelist parsing logic, or configuration-file reading logic in any source file, public header, or internal header.
2. THE LOGS library SHALL configure all run-time behavior, including the Severity_Threshold, sinks, and the communicator, exclusively through programmatic API calls.
3. THE LOGS library SHALL treat every configured output stream strictly as a write-only destination.
4. THE LOGS library SHALL NOT open any path for reading.
5. WHEN the LOGS CI pipeline runs, THE build system SHALL execute a static verification step that scans every LOGS source file, public header, and internal header for input-parsing indicators, where input-parsing indicators are defined as exactly the following closed set: references to YAML or other markup-parsing libraries, namelist or other configuration-format readers, and file-open-for-read or read-mode file APIs.
6. IF the static verification step detects one or more input-parsing indicators, THEN THE build system SHALL fail the build with a non-zero exit status and SHALL indicate which file contains each detected indicator.
7. WHEN the static verification step completes with zero input-parsing indicators detected, THE build system SHALL report the verification as passed and SHALL allow the build to proceed.
8. THE LOGS library SHALL NOT depend on or link against any file-parsing library at build time.

### Requirement 11: Thread Safety and MPI Threading Support

**User Story:** As a library developer, I want LOGS to be safe under concurrent multi-threaded logging, so that multiple threads on a node can log simultaneously without data races or interleaved output.

#### Acceptance Criteria

1. WHEN two or more threads invoke Logger logging entry points concurrently, THE Logger SHALL serialize access to each shared mutable resource — the configured sink collection, the active severity threshold, and the consolidation buffers — so that no data race occurs on any of those resources for any number of concurrent threads.
2. WHEN multiple threads write Log_Records concurrently to a single Sink, THE Logger SHALL ensure that the formatted text of any one Log_Record is written without interleaving with the text of another Log_Record.
3. WHEN the Logger is initialized, THE Logger SHALL store the MPI thread support level returned by MPI_Query_thread if MPI has been initialized, and SHALL store the conservative default MPI_THREAD_SINGLE if MPI has not been initialized at the time of Logger initialization.
4. IF the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THEN THE Logger SHALL serialize all MPI calls it issues through an internal mutex to prevent undefined behavior.
5. THE Logger SHALL provide a thread-safe query function that returns the stored MPI thread support level as an integer matching the MPI threading constants, returning the conservative default MPI_THREAD_SINGLE when no level has been detected because MPI was not initialized at Logger initialization.
6. WHILE a Synchronized_Abort is in progress on one thread, THE Logger SHALL reject any concurrent attempt to initiate a second Synchronized_Abort and SHALL invoke MPI_Abort no more than once per process lifetime.
7. THE Scoped_Context context stack SHALL be stored in thread-local storage so that concurrent threads maintain independent context without locking.
8. WHEN two or more threads submit Log_Records concurrently, THE Logger SHALL emit each submitted Log_Record exactly once with no record lost and no record duplicated, so that the count of emitted records equals the count of submitted records.
9. IF the Logger is initialized before MPI has been initialized, THEN THE Logger SHALL store MPI_THREAD_SINGLE as the conservative thread support level and SHALL serialize all MPI calls it issues through an internal mutex.

### Requirement 12: CMake Build System

**User Story:** As a build engineer, I want LOGS to provide a standalone, target-centric CMake build system, so that the library can be built independently against only the C++20 standard library and MPI.

#### Acceptance Criteria

1. THE LOGS build system SHALL require CMake version 3.21 or later and SHALL build all LOGS targets against the ISO C++20 standard as a mandatory requirement, and SHALL NOT depend on compiler-specific language extensions.
2. THE LOGS build system SHALL locate MPI using find_package(MPI REQUIRED COMPONENTS CXX).
3. THE LOGS build system SHALL produce a shared library target named logs with a namespace alias HELM::LOGS, linking MPI as a PUBLIC dependency so that downstream consumers inherit MPI include paths and link flags transitively.
4. THE LOGS build system SHALL link the logs target only against the C++20 standard library and MPI, and SHALL NOT introduce a Kokkos dependency, consistent with LOGS having no compute or GPU role.
5. THE LOGS build system SHALL declare a LOGS project version and SHALL export CMake configuration files (LOGSConfig.cmake, LOGSConfigVersion.cmake, and LOGSTargets.cmake) so that downstream projects can consume LOGS via find_package(LOGS), using SameMajorVersion compatibility matched against the major component of the declared LOGS project version.
6. THE LOGS build system SHALL provide a BUILD_TESTING configuration option that defaults to OFF, such that a default configuration builds the logs target without requiring any test dependency.
7. WHERE BUILD_TESTING is set to ON, THE LOGS build system SHALL locate Google Test using find_package(GTest REQUIRED) and SHALL build the Google Test verification suite.
8. IF MPI is not found during configuration, THEN THE LOGS build system SHALL halt configuration with a fatal error indicating that MPI with the CXX component is required, and SHALL NOT generate build files or produce the logs target.
9. IF the configured C++ toolchain does not support the C++20 standard, THEN THE LOGS build system SHALL halt configuration with a fatal error indicating that a C++20-capable compiler is required, and SHALL NOT generate build files or produce the logs target.
10. IF BUILD_TESTING is set to ON and Google Test cannot be located, THEN THE LOGS build system SHALL halt configuration with a fatal error indicating that Google Test is required to build the verification suite, and SHALL NOT generate build files.

### Requirement 13: Google Test Verification Suite

**User Story:** As a library developer, I want a Google Test suite that proves thread safety, rank-stamping accuracy, and redundant-message consolidation, so that the critical LOGS guarantees are mathematically verified rather than assumed.

#### Acceptance Criteria

1. THE test suite SHALL contain a concurrency test that spawns at least 8 threads, each emitting at least 10,000 Log_Records to a single shared in-memory Sink, and SHALL verify that the total number of recorded records equals the exact sum of records emitted across all threads (confirming no lost and no duplicated records) and that every recorded record's fields match exactly one emitted record with no partially overwritten or corrupted fields.
2. THE test suite SHALL contain a test that configures the Logger with a known MPI_Rank via the Interposition_Layer, emits at least 100 Log_Records, and verifies that every recorded record carries the configured MPI_Rank in its rank field and that zero recorded records carry any other rank value.
3. THE test suite SHALL contain a test that injects identical Log_Records attributed to a contiguous span of at least 4 distinct ranks (ranks R through R+N-1, where N is at least 4), triggers a single consolidation operation, and verifies that exactly one representative record is produced carrying a contributing-rank count equal to N and a Rank_Range whose lower bound equals R and whose upper bound equals R+N-1.
4. THE test suite SHALL contain a test that injects Log_Records bearing at least 2 distinct Consolidation_Keys, triggers a consolidation operation, and verifies that the number of representative records produced equals the number of distinct Consolidation_Keys and that no representative record combines records originating from more than one Consolidation_Key.
5. THE test suite SHALL use an Interposition_Layer that intercepts and records the invocation count, argument values, and relative call order of MPI_Comm_rank, MPI_Query_thread, and MPI_Abort, enabling deterministic verification of rank stamping and Synchronized_Abort behavior without a live MPI runtime.
6. THE test suite SHALL contain a test that emits a single FATAL Log_Record and verifies via the Interposition_Layer that all registered Sinks were flushed before MPI_Abort was invoked, that MPI_Abort was invoked exactly once, and that MPI_Abort was called with a non-zero exit code in the range 1 to 255 inclusive.
7. THE test suite SHALL contain a test that enters a Scoped_Context, throws a std::exception while the context label is active, and after the exception is caught verifies that the active context stack depth and label set are identical to their values immediately before the Scoped_Context was entered, confirming the destructor popped the label during stack unwinding.
8. THE test suite SHALL contain a round-trip test that formats a Log_Record with a known severity label, MPI_Rank, and message payload to text and verifies that the severity label, MPI_Rank, and message payload extracted from the fixed field positions of the formatted output are each equal to the corresponding original input values.
9. THE test suite SHALL contain a test that injects identical Log_Records attributed to a single rank, triggers a consolidation operation, and verifies that exactly one representative record is produced carrying a contributing-rank count of 1 and a Rank_Range whose lower and upper bounds both equal that single rank.
10. THE test suite SHALL contain a test that emits at least one Log_Record at each severity level below FATAL and verifies via the Interposition_Layer that MPI_Abort was not invoked for any of those records.

### Requirement 14: Tier 1 Isolation Compliance

**User Story:** As an architect, I want LOGS to have zero compile-time dependencies on other HELM libraries, so that the no-circular-dependency invariant of the HELM architecture is preserved.

#### Acceptance Criteria

1. THE LOGS library SHALL NOT contain, in any source file, public header, or internal header, an `#include` directive whose path contains a directory segment that case-insensitively matches a forbidden HELM component name (TICK, HALO, AXIS, AMIO, SPAN, or DAGR) immediately followed by a path separator; accordingly an include such as `<span/...>` is forbidden while the C++20 standard header `<span>` is allowed.
2. THE LOGS library SHALL NOT link against any other HELM library target (HELM::TICK, HELM::HALO, HELM::AXIS, HELM::AMIO, HELM::SPAN, or HELM::DAGR) at build time.
3. THE LOGS library public headers SHALL only contain `#include` directives referencing C++ standard library headers or MPI headers, and SHALL NOT contain any other `#include` directive.
4. THE LOGS CMakeLists.txt SHALL NOT reference any HELM:: namespace target other than HELM::LOGS in its target_link_libraries, add_dependencies, or find_package directives.
5. WHEN the LOGS CI pipeline runs, THE build system SHALL execute a static verification step that scans all LOGS source and header files for `#include` directives matching the forbidden-path rule defined in criterion 1.
6. IF the static verification step detects one or more forbidden `#include` directives, THEN THE build system SHALL fail the build with a non-zero exit status and SHALL report each offending file path.
7. WHEN the static verification step completes with zero forbidden `#include` directives detected, THE build system SHALL report the verification as passed and SHALL allow the build to proceed.

### Requirement 15: Repository and Container Structure

**User Story:** As a build engineer, I want LOGS to live in its own dedicated repository added as a Git submodule within the HELM project, so that Tier 1 libraries maintain independent version histories and CI pipelines while remaining composable within the monorepo.

#### Acceptance Criteria

1. THE LOGS source code SHALL reside in a dedicated Git repository that is distinct from the HELM superproject, maintains its own independent commit history, and is added as a Git submodule at the relative path `libs/logs` within the HELM project workspace root.
2. THE LOGS build and test workflow SHALL use the Docker image built from the HELM project DockerFile as its development and CI container environment.
3. WHEN the standalone CMakeLists.txt at the LOGS repository root is configured and built inside the HELM project Docker container with no other HELM source trees present, THE LOGS build SHALL complete both the configure step and the build step with a zero process exit status, emit the HELM::LOGS library target artifact, and produce no error-level diagnostics.
4. THE LOGS repository SHALL include a README at its root containing copy-runnable commands for the Docker container launch, the CMake configure and build, and the test suite execution, such that executing those commands in their documented sequence without manual modification reproduces a successful build (zero exit status) and a passing test run.
5. WHEN the HELM project workspace is cloned with `--recurse-submodules`, THE LOGS submodule working-tree SHALL be checked out at the commit recorded by the HELM superproject for the `libs/logs` submodule, requiring no further fetch, checkout, or update operation.
6. IF the LOGS submodule is uninitialized such that the `libs/logs` directory is empty, THEN THE HELM project build SHALL fail with a non-zero exit status and emit an indication that the `libs/logs` submodule must be initialized.
7. THE LOGS CI pipeline SHALL build and test the LOGS repository inside the Docker image built from the HELM project DockerFile, independently of and without invoking the HELM superproject CI pipeline.
