# Requirements Document

## Introduction

This specification defines the requirements for removing the eckit dependency from AMIO (Asynchronous Multidimensional I/O) and replacing it with HELM-native micro-libraries. AMIO is a Tier 1b elective utility in the HELM ecosystem that provides asynchronous background I/O for non-blocking NetCDF-4, Zarr v3, and GRIB2 outputs. Currently, AMIO uses eckit for five distinct capabilities: YAML/JSON configuration parsing, MPI communicator management, exception hierarchy, factory/registry pattern, and diagnostic logging. Each of these will be replaced by the appropriate HELM micro-library or a self-contained internal implementation, eliminating the last legacy ECMWF dependency from the HELM ecosystem.

The replacement libraries are:

- **CONF** (HELM::CONF): Replaces eckit::YAMLConfiguration / eckit::JSONConfiguration for manifest parsing.
- **HALO** (HELM::HALO): Replaces eckit::mpi::Comm for RAII MPI communicator splitting and threading queries.
- **LOGS** (HELM::LOGS): Replaces eckit::Log for diagnostic emission and error reporting.
- **TICK** (HELM::TICK): Not directly used by AMIO currently but mentioned for completeness; AMIO does not depend on eckit time primitives.

This migration must be backward-compatible at the C99 public ABI level: downstream consumers of `amio.h` and `amio_mod.f90` observe zero behavioral change. The migration must also respect the HELM architectural constraints: zero-copy memory via std::mdspan, hardware portability via Kokkos, RAII MPI handles, and no circular dependencies between Tier 1 components.

## Glossary

- **AMIO**: Asynchronous Multidimensional I/O; the Tier 1b elective background I/O writer in the HELM ecosystem.
- **eckit**: The legacy ECMWF C++ toolkit providing configuration, MPI wrappers, exceptions, logging, and factory patterns that AMIO currently depends on.
- **CONF**: Configuration Object & Notation Framework; the HELM Tier 1b micro-library for YAML configuration parsing via dotted-path accessor API.
- **HALO**: Hardware-Abstracted Link Operations; the HELM Tier 1 micro-library providing RAII-wrapped MPI communication primitives.
- **LOGS**: Lightweight Operational Global Status; the HELM Tier 1 micro-library for thread-safe, rank-aware logging and error handling.
- **Config_Loader**: The AMIO-internal module that parses YAML/JSON manifests into the Config struct driving Staging_Pool, Worker_Pool, and Backend_Factory construction.
- **Backend_Driver**: The polymorphic interface that concrete I/O drivers (NetCDF, Zarr, GRIB2) implement, currently accepting `eckit::Configuration` for open_write/open_read.
- **Backend_Factory**: The string-keyed registry of Backend_Driver implementations, currently modeled after eckit::Factory semantics.
- **Exception_Bridge**: The worker pool exception cordon that catches exceptions thrown by Backend_Driver operations and translates them to AMIO_ERR_* codes.
- **Worker_Pool**: The background thread pool managing write and prefetch task queues with per-(dataset, variable) ordering.
- **IOCommunicator**: The AMIO-internal struct representing the result of an MPI communicator split between I/O ranks and compute ranks.
- **C_Boundary**: The translation layer between the public C99 API surface and the internal C++20 implementation.
- **Staging_Pool**: The buffer pool providing staging memory for asynchronous write/read operations.
- **RAII**: Resource Acquisition Is Initialization; the C++ idiom binding resource lifetime to object scope.
- **Communicator**: The HALO RAII wrapper around MPI_Comm that automatically calls MPI_Comm_free on destruction.
- **std_mdspan**: The C++20/23 non-owning multi-dimensional array view used throughout HELM for zero-copy interfaces.
- **Manifest**: A YAML or JSON configuration file specifying AMIO runtime parameters (buffer counts, thread counts, codec settings, backend selection).
- **Dotted_Path**: The CONF key syntax for addressing nested YAML values (e.g., `staging_pool.buffer_count`).

## Requirements

### Requirement 1: Replace eckit Configuration Parsing with HELM::CONF

**User Story:** As a build engineer, I want AMIO's manifest parsing to use HELM::CONF instead of eckit::YAMLConfiguration, so that AMIO's only configuration dependency is the lightweight HELM micro-library rather than the entire eckit toolkit.

#### Acceptance Criteria

1. WHEN AMIO's Config_Loader parses a YAML manifest file, THE Config_Loader SHALL use `conf::Config::from_file` from HELM::CONF to load the document, instead of eckit::YAMLConfiguration.
2. WHEN AMIO's Config_Loader parses a YAML manifest from an in-memory string, THE Config_Loader SHALL use `conf::Config::from_string` from HELM::CONF to load the document, instead of eckit::YAMLConfiguration.
3. WHEN AMIO's Config_Loader reads a scalar configuration value by dotted path, THE Config_Loader SHALL use CONF's typed accessors (`get_int`, `get_double`, `get_bool`, `get_string`) with dotted-path keys matching the existing manifest schema.
4. WHEN AMIO's Config_Loader reads a list configuration value by dotted path, THE Config_Loader SHALL use CONF's list accessors (`get_int_list`, `get_double_list`, `get_string_list`) to retrieve sequence values.
5. IF a manifest file cannot be opened, THEN THE Config_Loader SHALL translate `conf::Error_Code::File_Not_Found` to `AMIO_ERR_MANIFEST_NOT_FOUND` and report the file path in the ValidationError message.
6. IF a manifest contains invalid YAML syntax, THEN THE Config_Loader SHALL translate `conf::Error_Code::Parse_Error` to `AMIO_ERR_MANIFEST_INVALID` and include the CONF-provided diagnostic text in the ValidationError message.
7. IF a required manifest key is absent, THEN THE Config_Loader SHALL translate `conf::Error_Code::Key_Not_Found` to `AMIO_ERR_MANIFEST_INVALID` and report the missing dotted path in the ValidationError.
8. IF a manifest value exists at a dotted path but cannot be converted to the expected type, THEN THE Config_Loader SHALL translate `conf::Error_Code::Type_Mismatch` to `AMIO_ERR_MANIFEST_INVALID` and report the dotted path and the expected type name in the ValidationError message.
9. THE Config_Loader SHALL retain its existing round-trip guarantee: `parse_string(serialize(config), "yaml", ...)` produces a Config struct equal to the original for all valid Config values, where equality is defined as integers exactly equal, doubles bit-exact at 17 significant digits, booleans exactly equal, and strings byte-identical.
10. THE Config_Loader SHALL retain all existing schema validation rules (numeric range checks, codec allow-list enforcement, backpressure invariant) unchanged.
11. THE Config_Loader SHALL NOT include any eckit header after this replacement is complete.

### Requirement 2: Replace eckit::Configuration in Backend_Driver Interface

**User Story:** As a library developer, I want the Backend_Driver interface to accept a HELM-native configuration type instead of eckit::Configuration, so that backend drivers no longer depend on eckit headers or ABI.

#### Acceptance Criteria

1. THE Backend_Driver abstract class SHALL replace `const eckit::Configuration&` parameters in `open_write` and `open_read` with `const conf::Config&` from HELM::CONF.
2. WHEN a concrete Backend_Driver (NetCDF_Driver, Zarr_Driver, GRIB2_Driver) needs to read a scalar configuration value, THE driver SHALL use the appropriate CONF typed accessor (`get_string`, `get_int`, `get_double`, `get_bool`) or its non-throwing counterpart (`try_string`, `try_int`, `try_double`, `try_bool`) with a dotted-path key.
3. WHEN a concrete Backend_Driver needs to read a list configuration value, THE driver SHALL use the appropriate CONF list accessor (`get_string_list`, `get_int_list`, `get_double_list`) with a dotted-path key.
4. WHEN a concrete Backend_Driver queries whether an optional key is present, THE driver SHALL use `conf::Config::has` with a dotted-path key.
5. THE Backend_Driver header (`backend_driver.hpp`) SHALL NOT forward-declare or reference any eckit namespace, type, or header after this replacement.
6. IF a driver operation receives a `conf::Conf_Error` (due to a missing required key, type mismatch, or any other CONF error), THEN THE driver SHALL allow the `conf::Conf_Error` to propagate up to the Exception_Bridge, which translates it to the appropriate `AMIO_ERR_*` code as defined in Requirement 4.
7. THE C_Boundary layer SHALL construct a `conf::Config` by calling `conf::Config::from_file` with the dataset manifest path at dataset-open time and pass it by const reference to the Backend_Driver's `open_write` or `open_read`, rather than constructing an eckit::YAMLConfiguration.
8. THE mock Backend_Driver used in tests (`MockBackendDriver`) SHALL update its `open_write` and `open_read` signatures to accept `const conf::Config&` instead of `const eckit::Configuration&`, maintaining test compilation without eckit headers.

### Requirement 3: Replace eckit::mpi::Comm with HALO Communicator

**User Story:** As a library developer, I want AMIO's MPI communicator splitting to use HALO's RAII Communicator wrapper instead of eckit::mpi::Comm, so that communicator lifetime is guaranteed by RAII and AMIO aligns with the HELM MPI safety model.

#### Acceptance Criteria

1. WHEN AMIO performs an MPI communicator split to separate I/O ranks from compute ranks, THE comm_split module SHALL wrap the parent MPI_COMM_WORLD in a `halo::Communicator` and call its `split(color, key)` method, storing the returned `halo::Communicator` RAII object as the I/O communicator.
2. WHILE an AMIO IOCommunicator holds a split communicator, THE IOCommunicator SHALL store the `halo::Communicator` object as a member (not a raw MPI_Comm handle or integer identifier) so that the communicator is automatically freed via RAII when the IOCommunicator is destroyed or moved-from.
3. WHEN a Backend_Driver needs to issue MPI collective operations (e.g., nc_create_par, nc_open_par), THE Worker_Pool SHALL expose the raw MPI_Comm handle from the stored HALO Communicator via the `handle()` accessor for passing to third-party C APIs, and SHALL ensure the owning `halo::Communicator` object outlives all such usages.
4. WHEN AMIO queries the MPI threading level to validate MPI_THREAD_MULTIPLE support, THE mpi_threading module SHALL call `halo::Environment::thread_support_level()` (after `halo::Environment::initialize()` has been called) and compare the result against MPI_THREAD_MULTIPLE, instead of using eckit::mpi or direct MPI_Query_thread calls.
5. IF the `halo::Communicator::split` call throws `std::runtime_error` (e.g., MPI not initialized, invalid color/key), THEN THE comm_split module SHALL catch the exception and return `AMIO_ERR_COMM_SPLIT_FAILED` without leaking any MPI handle, relying on HALO's RAII destruction of any partially-constructed communicator.
6. THE comm_split module SHALL NOT include any eckit::mpi header (e.g., `eckit/mpi/Comm.h`) after this replacement.
7. WHEN MPI is not available at build time (the `AMIO_HAS_MPI` compile definition is absent), THE comm_split module SHALL return AMIO_OK with a default IOCommunicator indicating all ranks participate in I/O and SHALL NOT reference `halo::Communicator` or any MPI symbol, using compile-time conditional compilation to exclude HALO's MPI-dependent code paths.
8. WHEN the comm_split module performs a successful split, THE IOCommunicator SHALL expose the I/O sub-communicator rank and size via `halo::Communicator::rank()` and `halo::Communicator::size()` rather than issuing separate MPI_Comm_rank or MPI_Comm_size calls.

### Requirement 4: Replace eckit::Exception Hierarchy

**User Story:** As a library developer, I want AMIO's exception handling to use standard C++ exceptions instead of eckit::Exception, so that the exception bridge no longer depends on eckit headers and exception translation is self-contained.

#### Acceptance Criteria

1. THE Exception_Bridge SHALL remove the `catch (eckit::Exception&)` arm from the three-level exception cordon and SHALL implement a two-level catch hierarchy: first `catch (const std::exception&)`, then `catch(...)`.
2. WHEN a Backend_Driver operation throws a `conf::Conf_Error` with `Error_Code::File_Not_Found`, THE Exception_Bridge SHALL translate it to `AMIO_ERR_MANIFEST_NOT_FOUND` and preserve the `what()` message.
3. WHEN a Backend_Driver operation throws a `conf::Conf_Error` with `Error_Code::Parse_Error` or `Error_Code::Key_Not_Found`, THE Exception_Bridge SHALL translate it to `AMIO_ERR_MANIFEST_INVALID` and preserve the `what()` message.
4. WHEN a Backend_Driver operation throws a `conf::Conf_Error` with `Error_Code::Type_Mismatch` or `Error_Code::Invalid_Arg`, THE Exception_Bridge SHALL translate it to `AMIO_ERR_INVALID_INPUT` and preserve the `what()` message.
5. WHEN a Backend_Driver operation throws a `std::invalid_argument`, THE Exception_Bridge SHALL translate it to `AMIO_ERR_INVALID_INPUT`.
6. WHEN a Backend_Driver operation throws any other `std::exception` subclass not matched by criteria 2–5, THE Exception_Bridge SHALL translate it to `AMIO_ERR_BACKEND_FAILURE` and preserve the `what()` message.
7. THE Exception_Bridge SHALL NOT include any eckit header after this replacement.
8. WHEN concrete drivers (NetCDF_Driver, Zarr_Driver, GRIB2_Driver) encounter backend-specific errors, THE drivers SHALL throw `std::runtime_error` or a domain-specific subclass of `std::exception` rather than `eckit::Exception`.
9. THE `translate_exception_to_error` function SHALL order its catch clauses from most-specific to least-specific: `conf::Conf_Error` first, then `std::invalid_argument`, then `std::exception`, then `catch(...)`.

### Requirement 5: Replace eckit::Factory with Self-Contained Backend_Factory

**User Story:** As a library developer, I want AMIO's Backend_Factory to be a fully self-contained implementation rather than a wrapper around eckit::Factory, so that the factory pattern has no external dependency.

#### Acceptance Criteria

1. THE Backend_Factory SHALL be a string-keyed singleton registry with the public API: `register_driver`, `build`, `has`, `registered_keys`, and `clear`.
2. THE Backend_Factory SHALL retain the static-initialization registration pattern via `BackendRegistrar<ConcreteDriver>` at namespace scope.
3. THE Backend_Factory SHALL retain last-writer-wins semantics for duplicate key registration: if `register_driver` is called with a key that is already registered, the previous builder is silently replaced.
4. THE Backend_Factory SHALL use `std::shared_mutex` for thread-safe concurrent access to the registry map, acquiring a shared lock for read operations (`build`, `has`, `registered_keys`) and an exclusive lock for write operations (`register_driver`, `clear`).
5. THE Backend_Factory header and implementation SHALL NOT reference eckit::Factory, eckit::ConcreteBuilderT0, or any eckit header.
6. THE Backend_Factory comments and documentation SHALL describe the pattern as AMIO's own registry rather than referencing eckit::Factory semantics.
7. THE existing factory keys ("netcdf4", "zarr3", "grib2") SHALL continue to produce the same concrete driver instances after the replacement.
8. IF `build` is called with an unregistered key or an empty string, THEN THE Backend_Factory SHALL return nullptr and set the output error code to AMIO_ERR_UNKNOWN_BACKEND without mutating any internal state.
9. IF `register_driver` is called with an empty key, THEN THE Backend_Factory SHALL reject the registration by returning false and leaving the registry unchanged.
10. THE Backend_Factory SHALL perform case-sensitive exact-match lookup: keys differing only in letter case (e.g., "NetCDF4" vs "netcdf4") SHALL be treated as distinct entries.

### Requirement 6: Replace eckit::Log with HELM::LOGS for Diagnostics

**User Story:** As an operations engineer, I want AMIO's diagnostic output to route through HELM::LOGS instead of eckit::Log, so that error messages and stack traces are MPI-rank-aware and synchronized with the rest of the HELM ecosystem.

#### Acceptance Criteria

1. WHEN the Exception_Bridge emits a parallel stack trace diagnostic, THE Exception_Bridge SHALL route the diagnostic through LOGS's Logger at FATAL severity when the error is unrecoverable, or ERROR severity otherwise, instead of writing directly to stderr or eckit::Log.
2. WHEN AMIO emits any informational message (e.g., configuration summary, backend selection), THE message SHALL be emitted through LOGS's Logger at INFO severity.
3. WHEN AMIO encounters a non-fatal operational warning (e.g., thread pinning failure, codec fallback), THE warning SHALL be emitted through LOGS's Logger at WARNING severity.
4. WHEN AMIO encounters an unrecoverable error requiring process termination, THE Exception_Bridge SHALL use LOGS's FATAL path which performs synchronized abort across MPI ranks.
5. WHEN amio_init completes communicator setup, THE AMIO module SHALL configure its Logger instance with the I/O communicator so that MPI rank stamps reflect the I/O sub-communicator rank rather than the world rank.
6. THE LOGS integration SHALL NOT require AMIO to link any eckit logging or stream library.
7. WHILE LOGS is not yet initialized (before `amio_init` completes communicator setup), THE Exception_Bridge SHALL fall back to direct stderr output rather than invoking LOGS, using a module-level atomic boolean flag to track initialization state.
8. THE AMIO module SHALL own a single `logs::Logger` instance as a private member of the AMIO_Core state, ensuring the Logger lifetime is tied to the core lifecycle (created during amio_init, destroyed during amio_finalize).

### Requirement 7: Remove eckit from CMake Build System

**User Story:** As a build engineer, I want the AMIO CMakeLists.txt to have no `find_package(eckit)` call and no eckit link targets, so that AMIO can be built in environments where eckit is not installed.

#### Acceptance Criteria

1. THE AMIO root CMakeLists.txt SHALL NOT contain `find_package(eckit ...)` or any reference to the `eckit` imported target.
2. THE AMIO root CMakeLists.txt SHALL add `find_package(CONF REQUIRED)` and link `HELM::CONF` as PRIVATE to `amio_core`, causing configure to fail immediately if CONF is not found.
3. THE AMIO root CMakeLists.txt SHALL add `find_package(HALO REQUIRED)` and link `HELM::HALO` as PRIVATE to `amio_core`, causing configure to fail immediately if HALO is not found.
4. THE AMIO root CMakeLists.txt SHALL add `find_package(LOGS REQUIRED)` and link `HELM::LOGS` as PRIVATE to `amio_core`, causing configure to fail immediately if LOGS is not found.
5. THE AMIO public headers target (`amio_public_headers`) SHALL NOT transitively expose any HELM micro-library header to downstream consumers; all HELM dependencies remain PRIVATE.
6. THE `AMIO_HAS_ECKIT` compile definition SHALL be removed from all targets (amio_core, driver_netcdf, driver_zarr, driver_grib2).
7. THE AMIOConfig.cmake install package SHALL NOT list eckit, CONF, HALO, or LOGS as transitive dependencies via `find_dependency`, because they are PRIVATE implementation details.
8. WHEN AMIO is configured and built with HELM::CONF, HELM::HALO, and HELM::LOGS available, THE build SHALL produce `libamio.so` (or `libamio.dylib` on macOS) and all driver archives without errors.
9. THE driver_netcdf, driver_zarr, and driver_grib2 static archives SHALL link `HELM::CONF` PRIVATE (for `conf::Config` in open_write/open_read) but SHALL NOT link eckit.

### Requirement 8: Preserve Public ABI Stability

**User Story:** As a downstream consumer, I want the AMIO public C99 API and Fortran module to remain unchanged after the eckit removal, so that existing applications work without recompilation against a new AMIO release.

#### Acceptance Criteria

1. THE public C99 headers (`amio.h`, `amio_errors.h`, `amio_types.h`, `amio_export.h`, `amio_mdspan_fwd.h`) SHALL contain zero changes to function signatures, type definitions, enum values, or macro definitions.
2. THE Fortran module (`amio_mod.f90`) SHALL contain zero changes to the `bind(C)` interface blocks or public module procedures.
3. THE `amio_errors.h` error code enumeration SHALL retain all existing values at their current integer assignments; no existing error code is removed or renumbered.
4. THE C_Boundary handle-table API (`amio_init`, `amio_finalize`, `amio_open_dataset`, `amio_close_dataset`, `amio_write`, `amio_read`, `amio_flush`, `amio_close`, `amio_wait`, `amio_view_data`, `amio_release_view`, `amio_strerror`) SHALL exhibit identical observable behavior for all valid input sequences.
5. THE `amio_strerror` function SHALL continue to return the same human-readable strings for all existing error codes.
6. IF a caller passes an invalid manifest path to `amio_init`, THEN the returned error code SHALL remain `AMIO_ERR_MANIFEST_NOT_FOUND`, matching the pre-migration behavior.
7. THE shared library SHALL export the same set of `amio_*` C symbols (visible via `nm -D`) as the pre-migration build, ensuring binary compatibility for existing dynamically-linked consumers.

### Requirement 9: Eliminate eckit Compatibility Shims from Drivers

**User Story:** As a library developer, I want the backend drivers to compile without the eckit compatibility shim (the in-file `namespace eckit { class Configuration ... }` fallback), so that the code is clean and unambiguous.

#### Acceptance Criteria

1. THE netcdf_driver.cpp SHALL NOT contain any `namespace eckit` shim class, forward-declaration, or `#ifdef AMIO_HAS_ECKIT` conditional block after migration.
2. THE zarr_driver.cpp SHALL NOT contain any `namespace eckit` shim class, forward-declaration, or `#ifdef AMIO_HAS_ECKIT` conditional block after migration.
3. THE grib2_driver.cpp SHALL NOT contain any `namespace eckit` shim class, forward-declaration, or `#ifdef AMIO_HAS_ECKIT` conditional block after migration.
4. WHEN a driver needs to throw an exception on error, THE driver SHALL throw `std::runtime_error` (or a descriptive subclass) with a meaningful message string, rather than `eckit::Exception`.
5. THE `nc_check` helper function in netcdf_driver.cpp SHALL throw `std::runtime_error` with a message including the NetCDF error code and `nc_strerror` text, instead of `eckit::Exception`.
6. WHEN a driver validates configuration parameters (data model string, codec name, required path), THE driver SHALL use CONF's typed accessor error propagation (allow Conf_Error to propagate) or throw `std::invalid_argument` with a descriptive message.
7. THE driver source files SHALL NOT include any eckit header (`eckit/exception/Exceptions.h`, `eckit/config/Configuration.h`, `eckit/log/Log.h`, or any other `eckit/` path) after migration.
8. ALL `eckit::Log::info()`, `eckit::Log::error()`, and `eckit::Log::warning()` calls in driver source files SHALL be replaced with the appropriate LOGS severity-level emission (INFO, ERROR, WARNING respectively).

### Requirement 10: Thread-Safety and Concurrency Preservation

**User Story:** As a model developer, I want the eckit replacement to preserve AMIO's existing thread-safety guarantees, so that concurrent write submissions and background worker operations remain safe.

#### Acceptance Criteria

1. THE Worker_Pool SHALL maintain per-(dataset, variable) ordering mutex semantics unchanged after the migration, ensuring writes to the same (dataset, variable) pair execute in submission order with the ordering mutex held only across the backend serialize call.
2. THE Backend_Factory SHALL remain thread-safe for concurrent `create` calls from multiple Worker_Pool threads (1 to 256 concurrent threads) via its internal std::shared_mutex protecting the registry map.
3. WHEN multiple threads concurrently submit write tasks that trigger manifest loading, THE Config_Loader SHALL produce a Config struct identical to that produced by a single-threaded invocation given the same input, without requiring external synchronization by the caller.
4. WHILE the IOCommunicator holds a `halo::Communicator` after amio_init completes, THE IOCommunicator SHALL be treated as immutable (read-only shared state) by all Worker_Pool threads, requiring no runtime synchronization for read access.
5. THE LOGS Logger SHALL support concurrent diagnostic emissions from multiple Worker_Pool threads without message corruption or interleaving of individual log entries.
6. THE migration SHALL NOT introduce new lock-ordering dependencies between HELM library internal mutexes (CONF, HALO, LOGS) and the Worker_Pool's per-(dataset, variable) ordering mutex or queue mutex, preserving deadlock-freedom.
7. THE migration SHALL NOT introduce shared mutable state in Config_Loader, Backend_Factory, or IOCommunicator where none existed in the eckit-based implementation.

### Requirement 11: No Circular Dependencies

**User Story:** As an architect, I want the AMIO eckit replacement to respect the HELM no-circular-dependency law, so that Tier 1 utilities remain blind to each other and to higher-tier components.

#### Acceptance Criteria

1. THE AMIO library (Tier 1b) SHALL depend only on Tier 1 libraries (HALO, LOGS) and on the Tier 1b peer library CONF, and SHALL NOT introduce any compile-time or link-time dependency from CONF, HALO, or LOGS back to AMIO.
2. THE AMIO source tree SHALL NOT contain any `#include` directive referencing a header path belonging to TICK, AXIS, SPAN, or DAGR.
3. THE CONF, HALO, and LOGS libraries SHALL NOT require source-code modification, API additions, or build-configuration changes to support AMIO's usage; AMIO consumes only their public installed APIs as shipped.
4. THE AMIO public headers SHALL NOT expose any CONF, HALO, or LOGS type in their interface (no `#include` of CONF/HALO/LOGS headers and no forward-declarations of their types); these dependencies are strictly PRIVATE implementation details.
5. WHEN the CI pipeline builds AMIO, THE build system SHALL verify that the CMake link graph contains no path from any CONF, HALO, or LOGS target back to any AMIO target, confirming that no circular dependency exists at the build-system level.

### Requirement 12: Migration Verification and Testing

**User Story:** As a quality engineer, I want automated verification that eckit has been completely removed and the replacement libraries work correctly, so that no latent eckit dependency can regress into the codebase.

#### Acceptance Criteria

1. THE AMIO test suite SHALL include a static scan that verifies zero `#include` directives referencing any eckit header path pattern (`eckit/`, `<eckit`) exist in any AMIO source (`.cpp`, `.c`, `.f90`) or header (`.hpp`, `.h`) file under the `src/`, `include/`, and `fortran/` directories.
2. THE AMIO test suite SHALL include a build verification test that configures and compiles all AMIO targets (amio_core, driver_netcdf, driver_zarr, driver_grib2, amio_fortran) with HELM::CONF, HELM::HALO, and HELM::LOGS present but eckit absent from the CMake prefix path, and the build SHALL complete with zero compilation errors and zero linker errors.
3. THE existing Config_Loader round-trip property test SHALL continue to pass: for all valid Config values (as defined by the manifest schema validation rules), serializing to YAML and re-parsing produces a Config struct where every field compares equal to the original.
4. THE existing Config_Loader validation tests SHALL continue to pass: all numeric range violations, codec allow-list violations, and backpressure invariant violations produce the same `AMIO_ERR_*` error codes as before migration.
5. THE existing Backend_Factory tests SHALL continue to pass: registering a driver, looking it up by key, and creating an instance via `build` produces the same concrete driver type for keys "netcdf4", "zarr3", and "grib2".
6. THE existing exception bridge tests SHALL continue to pass: all exception types (`conf::Conf_Error`, `std::invalid_argument`, `std::runtime_error`, unknown exceptions) produce the expected `AMIO_ERR_*` codes as specified in Requirement 4.
7. THE existing comm_split tests SHALL continue to pass: valid communicator split configurations produce a usable IOCommunicator, and invalid configurations (MPI not initialized, invalid rank set) return `AMIO_ERR_COMM_SPLIT_FAILED`.
8. WHEN the CI pipeline runs `nm -D libamio.so` and demangles the output, THE output SHALL contain zero symbols whose demangled name includes the substring `eckit::`.
9. WHEN any AMIO source or header file is added or modified in a pull request, THE CI pipeline SHALL execute the static eckit-include scan and the `nm -D` symbol check as part of the automated gate, failing the pipeline if either check detects eckit references.

### Requirement 13: Configuration Object Lifetime in Driver Open Path

**User Story:** As a library developer, I want a clear ownership model for the conf::Config object passed to drivers, so that the configuration remains valid for the full duration of the driver's open lifetime.

#### Acceptance Criteria

1. WHEN the C_Boundary opens a dataset, THE C_Boundary SHALL construct a `conf::Config` from the manifest and transfer ownership to the dataset's internal state (DatasetRecord) so the Config outlives the open_write/open_read call.
2. THE Backend_Driver::open_write and Backend_Driver::open_read SHALL receive the `conf::Config` by const reference from the owning dataset state.
3. WHILE a dataset remains open (between open and close), THE owning dataset state SHALL keep the `conf::Config` alive and unmodified so that drivers may query it at any point without external synchronization.
4. WHEN a dataset is closed, THE owning dataset state SHALL destroy the Backend_Driver instance before destroying the `conf::Config`, so that the driver's destructor may still safely access configuration values during teardown.
5. IF a driver caches configuration values during open, THE driver SHALL copy the scalar values (int, double, bool, string) into its own member variables rather than storing references or pointers into the conf::Config node tree.
