# Implementation Plan: AMIO eckit Replacement

## Overview

Systematic removal of the eckit dependency from AMIO and replacement with three HELM-native micro-libraries (CONF, HALO, LOGS) plus confirmation that the Backend_Factory is already self-contained. The migration proceeds module-by-module: first the Backend_Driver interface (since all other modules depend on it), then Config_Loader, comm_split, Exception_Bridge, Backend_Factory cleanup, LOGS integration, driver eckit-shim removal, and finally CMake build system changes. Each step preserves the public C99 ABI unchanged. The implementation language is C++20.

## Tasks

- [x] 1. Replace eckit::Configuration in Backend_Driver interface
  - [x] 1.1 Update Backend_Driver abstract class to use conf::Config
    - In `src/factory/backend_driver.hpp`, remove the `namespace eckit { class Configuration; }` forward-declaration block
    - Add `#include <conf/config.hpp>` at the top of the file
    - Change `virtual void open_write(const eckit::Configuration& config) = 0` to `virtual void open_write(const conf::Config& config) = 0`
    - Change `virtual void open_read(const eckit::Configuration& config) = 0` to `virtual void open_read(const conf::Config& config) = 0`
    - Update all doc comments referencing eckit::Configuration or eckit::Exception to reference `conf::Config` and `std::exception`
    - Update the class-level comment block to remove eckit::Factory references and describe the Backend_Factory as AMIO's own registry
    - _Requirements: 2.1, 2.5, 11.4_

  - [x] 1.2 Update NetCDF_Driver to use conf::Config
    - In `src/drivers/netcdf/netcdf_driver.cpp`, remove the entire `#ifdef AMIO_HAS_ECKIT` / `#else` / `namespace eckit { ... }` shim block
    - Add `#include <conf/config.hpp>` and `#include <conf/error.hpp>`
    - Change `open_write(const eckit::Configuration& config)` signature to `open_write(const conf::Config& config)`
    - Change `open_read(const eckit::Configuration& config)` signature to `open_read(const conf::Config& config)`
    - Replace `config.getString("key")` calls with `config.get_string("key")`
    - Replace `config.getString("key", default)` calls with `config.get_or<std::string>("key", default)`
    - Replace `config.getStringVector(...)` calls with `config.get_string_list(...)`
    - Replace `config.has("key")` calls with `config.has("key")` (unchanged API)
    - Replace `throw eckit::Exception(...)` with `throw std::runtime_error(...)`
    - Update `nc_check` helper to throw `std::runtime_error` with NetCDF error code and `nc_strerror` text
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 9.1, 9.4, 9.5, 9.7_

  - [x] 1.3 Update Zarr_Driver to use conf::Config
    - In `src/drivers/zarr/zarr_driver.cpp`, remove any `#ifdef AMIO_HAS_ECKIT` / `namespace eckit` shim block
    - Add `#include <conf/config.hpp>` and `#include <conf/error.hpp>`
    - Change `open_write` and `open_read` signatures to accept `const conf::Config&`
    - Replace all eckit accessor calls with CONF typed accessors (`get_string`, `get_int`, `get_or`, `has`)
    - Replace `throw eckit::Exception(...)` with `throw std::runtime_error(...)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 9.2, 9.4, 9.7_

  - [x] 1.4 Update GRIB2_Driver to use conf::Config
    - In `src/drivers/grib2/grib2_driver.cpp`, remove any `#ifdef AMIO_HAS_ECKIT` / `namespace eckit` shim block
    - Add `#include <conf/config.hpp>` and `#include <conf/error.hpp>`
    - Change `open_write` and `open_read` signatures to accept `const conf::Config&`
    - Replace all eckit accessor calls with CONF typed accessors
    - Replace `throw eckit::Exception(...)` with `throw std::runtime_error(...)`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 9.3, 9.4, 9.7_

  - [x] 1.5 Update C_Boundary to construct conf::Config for drivers
    - In `src/c_boundary/amio_api.cpp`, replace construction of `eckit::YAMLConfiguration` with `conf::Config::from_file(manifest_path)`
    - Store the resulting `conf::Config` in the DatasetRecord so it outlives the driver
    - Pass `const conf::Config&` to `driver->open_write()` / `driver->open_read()`
    - Ensure the DatasetRecord destroys the driver before destroying the Config (destruction order)
    - Remove any `#include <eckit/config/...>` headers from the C_Boundary files
    - _Requirements: 2.7, 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 1.6 Update MockBackendDriver in tests
    - Update the mock driver's `open_write` and `open_read` to accept `const conf::Config&`
    - Remove any eckit includes from test files that reference the mock
    - Verify mock-based tests compile without eckit headers
    - _Requirements: 2.8_

- [x] 2. Checkpoint - Verify Backend_Driver interface migration
  - Ensure all driver compilation units build without eckit headers for the configuration interface.
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Replace eckit Configuration Parsing with HELM::CONF in Config_Loader
  - [x] 3.1 Rewrite Config_Loader::parse to use conf::Config::from_file
    - In `src/config/config_loader.cpp`, remove the `#ifdef AMIO_HAS_ECKIT` eckit parsing path
    - Remove the standalone YAML tokenizer (`tokenize_yaml`, `tokenize_json`, and all helper functions)
    - Add `#include <conf/config.hpp>` and `#include <conf/error.hpp>`
    - Implement `parse()` to call `conf::Config::from_file(path)` and then delegate to a new `populate_from_conf()` method
    - Catch `conf::Conf_Error` with `Error_Code::File_Not_Found` → return `AMIO_ERR_MANIFEST_NOT_FOUND` with file path in ValidationError
    - Catch `conf::Conf_Error` with `Error_Code::Parse_Error` → return `AMIO_ERR_MANIFEST_INVALID` with CONF diagnostic text
    - _Requirements: 1.1, 1.5, 1.6, 1.11_

  - [x] 3.2 Rewrite Config_Loader::parse_string to use conf::Config::from_string
    - Implement `parse_string()` to call `conf::Config::from_string(content, format)` then delegate to `populate_from_conf()`
    - Handle the same `conf::Conf_Error` translations as `parse()`
    - _Requirements: 1.2_

  - [x] 3.3 Implement populate_from_conf using CONF typed accessors
    - Create a private `populate_from_conf(const conf::Config& manifest, Config& config_out, ValidationError& error_out)` method
    - Read scalar values via `manifest.get_int("staging_pool.buffer_count")`, `manifest.get_string("backend")`, etc.
    - Read list values via `manifest.get_int_list("worker_pool.cpu_cores")`, `manifest.get_string_list("codec.lossless_allow_list")`
    - Guard optional keys with `manifest.has("key")` before reading
    - Catch `conf::Conf_Error` with `Key_Not_Found` → set `AMIO_ERR_MANIFEST_INVALID` with dotted path in ValidationError
    - Catch `conf::Conf_Error` with `Type_Mismatch` → set `AMIO_ERR_MANIFEST_INVALID` with dotted path and expected type
    - Retain all existing schema validation (numeric range, codec allow-list, backpressure invariant) via the unchanged `validate()` method
    - _Requirements: 1.3, 1.4, 1.7, 1.8, 1.9, 1.10_

  - [x] 3.4 Remove all eckit includes from Config_Loader
    - Remove `#include <eckit/config/YAMLConfiguration.h>`, `#include <eckit/config/JSONConfiguration.h>`, `#include <eckit/filesystem/PathName.h>`
    - Remove any remaining `#ifdef AMIO_HAS_ECKIT` conditional blocks
    - Verify `config_loader.hpp` and `config_loader.cpp` contain zero eckit references
    - _Requirements: 1.11_

  - [x] 3.5 Write unit tests for Config_Loader CONF integration
    - Test `parse()` with a valid manifest file produces correct Config struct
    - Test `parse()` with missing file returns `AMIO_ERR_MANIFEST_NOT_FOUND`
    - Test `parse_string()` with invalid YAML returns `AMIO_ERR_MANIFEST_INVALID`
    - Test `parse_string()` with missing required key returns `AMIO_ERR_MANIFEST_INVALID` with correct dotted path
    - Test `parse_string()` with type mismatch (string where int expected) returns `AMIO_ERR_MANIFEST_INVALID`
    - Test round-trip: `parse_string(serialize(config), "yaml", ...)` equals original config
    - _Requirements: 1.5, 1.6, 1.7, 1.8, 1.9, 12.3, 12.4_

- [x] 4. Replace eckit::mpi::Comm with HALO Communicator
  - [x] 4.1 Rewrite IOCommunicator struct to use halo::Communicator
    - In `src/workers/comm_split.hpp`, replace `int64_t io_comm_id` / `int64_t compute_comm_id` with `std::optional<halo::Communicator> io_comm`
    - Add `#include <halo/communicator.hpp>` inside an `#ifdef AMIO_HAS_MPI` guard
    - Implement inline `handle()`, `rank()`, `size()` accessors that delegate to `io_comm->handle()` / `io_comm->rank()` / `io_comm->size()`
    - When `io_comm` is nullopt (no split), `handle()` returns `MPI_COMM_WORLD`, `rank()` returns 0, `size()` returns 1
    - When `AMIO_HAS_MPI` is not defined, provide trivial stubs returning sentinel values (0, 0, 1)
    - _Requirements: 3.1, 3.2, 3.7, 3.8_

  - [x] 4.2 Rewrite split_communicator to use HALO split
    - In `src/workers/comm_split.cpp`, remove all `#include <eckit/mpi/Comm.h>` and the `#if defined(AMIO_HAS_ECKIT)` code path
    - Add `#include <halo/communicator.hpp>` inside the `#ifdef AMIO_HAS_MPI` guard
    - Implement split: construct `halo::Communicator world(MPI_COMM_WORLD)` (non-owning for predefined comms), then `result.io_comm = world.split(color, my_rank)`
    - Catch `std::runtime_error` from `halo::Communicator::split` and return `AMIO_ERR_COMM_SPLIT_FAILED`
    - Remove the raw `MPI_Comm_split` fallback path (HALO replaces it)
    - _Requirements: 3.1, 3.5, 3.6_

  - [x] 4.3 Replace MPI threading validation with HALO Environment
    - In `src/workers/mpi_threading.cpp`, replace `MPI_Query_thread` / eckit threading queries with `halo::Environment::thread_support_level()`
    - Compare result against `MPI_THREAD_MULTIPLE` and return appropriate error if insufficient
    - Remove any eckit::mpi includes
    - _Requirements: 3.4, 3.6_

  - [x] 4.4 Update Worker_Pool to expose MPI_Comm via IOCommunicator::handle()
    - Ensure the Worker_Pool passes `io_comm.handle()` to drivers that need raw `MPI_Comm` (e.g., nc_create_par)
    - Verify the owning `halo::Communicator` in IOCommunicator outlives all driver usages
    - _Requirements: 3.3, 10.4_

  - [ ] 4.5 Write unit tests for comm_split HALO integration
    - Test default config (no split) returns valid IOCommunicator with `is_io_rank = true`
    - Test invalid config (world_size ≤ 0, duplicate ranks) returns `AMIO_ERR_COMM_SPLIT_FAILED`
    - Test that IOCommunicator accessors (`handle()`, `rank()`, `size()`) return correct values after split
    - Test non-MPI build: split returns `AMIO_ERR_COMM_SPLIT_FAILED` for non-default configs
    - _Requirements: 3.5, 3.7, 12.7_

- [x] 5. Checkpoint - Verify CONF and HALO integration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Replace eckit::Exception hierarchy in Exception_Bridge
  - [x] 6.1 Rewrite translate_exception_to_error with new catch hierarchy
    - In `src/workers/exception_bridge.cpp`, remove the `#ifdef AMIO_HAS_ECKIT` block and `#include <eckit/exception/Exceptions.h>`
    - Add `#include <conf/error.hpp>` for `conf::Conf_Error`
    - Implement ordered catch clauses: `conf::Conf_Error` → `std::invalid_argument` → `std::exception` → `catch(...)`
    - Map `conf::Error_Code::File_Not_Found` → `AMIO_ERR_MANIFEST_NOT_FOUND`
    - Map `conf::Error_Code::Parse_Error` and `Key_Not_Found` → `AMIO_ERR_MANIFEST_INVALID`
    - Map `conf::Error_Code::Type_Mismatch` and `Invalid_Arg` → `AMIO_ERR_INVALID_INPUT`
    - Map `std::invalid_argument` → `AMIO_ERR_INVALID_INPUT`
    - Map `std::exception` → `AMIO_ERR_BACKEND_FAILURE`
    - Preserve `what()` message in all catch arms
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.9_

  - [x] 6.2 Rewrite execute_with_exception_cordon to remove eckit catch
    - Remove the `#ifdef AMIO_HAS_ECKIT` catch block for `eckit::Exception`
    - Add a `catch (const conf::Conf_Error& e)` arm before the `std::exception` arm
    - Use `translate_exception_to_error` for consistent mapping in both functions
    - _Requirements: 4.1, 4.7, 4.9_

  - [x] 6.3 Update exception_bridge.hpp to remove eckit references
    - Remove any eckit forward-declarations or includes from the header
    - Update doc comments to describe the two-level hierarchy (std::exception + catch(...)) plus conf::Conf_Error specialization
    - _Requirements: 4.7_

  - [x] 6.4 Write unit tests for Exception_Bridge CONF error mapping
    - Test `conf::Conf_Error{File_Not_Found, ...}` → `AMIO_ERR_MANIFEST_NOT_FOUND`
    - Test `conf::Conf_Error{Parse_Error, ...}` → `AMIO_ERR_MANIFEST_INVALID`
    - Test `conf::Conf_Error{Key_Not_Found, ...}` → `AMIO_ERR_MANIFEST_INVALID`
    - Test `conf::Conf_Error{Type_Mismatch, ...}` → `AMIO_ERR_INVALID_INPUT`
    - Test `conf::Conf_Error{Invalid_Arg, ...}` → `AMIO_ERR_INVALID_INPUT`
    - Test `std::invalid_argument` → `AMIO_ERR_INVALID_INPUT`
    - Test `std::runtime_error` → `AMIO_ERR_BACKEND_FAILURE`
    - Test unknown exception (`throw 42`) → `AMIO_ERR_BACKEND_FAILURE`
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 12.6_

- [x] 7. Clean up Backend_Factory (remove eckit references)
  - [x] 7.1 Remove eckit references from Backend_Factory header and implementation
    - In `src/factory/backend_factory.hpp`, remove all comments referencing `eckit::Factory`, `eckit::ConcreteBuilderT0`
    - Update the class-level doc comment to describe it as "AMIO's own string-keyed singleton registry" rather than "mimics eckit::Factory"
    - In `src/factory/backend_factory.cpp`, remove all comments referencing eckit::Factory semantics
    - Verify no eckit header is included in either file
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

- [x] 8. Integrate HELM::LOGS for diagnostics
  - [x] 8.1 Add LOGS Logger to AMIO_Core state
    - In the AMIO_Core state struct (or equivalent header), add `logs::Logger logger` member and `std::atomic<bool> logs_initialized{false}`
    - Add `#include <logs/logger.hpp>` to the relevant internal header
    - _Requirements: 6.8_

  - [x] 8.2 Initialize Logger during amio_init after comm_split
    - After successful communicator split in `amio_init`, call `state.logger.configure_communicator(state.io_comm.handle())`
    - Set threshold to `logs::Severity_Level::INFO`
    - Set `state.logs_initialized.store(true, std::memory_order_release)`
    - _Requirements: 6.5, 6.8_

  - [x] 8.3 Route Exception_Bridge diagnostics through LOGS
    - In `emit_parallel_stacktrace`, check `logs_initialized` atomic flag
    - If initialized: emit via `logger.log(logs::Severity_Level::FATAL, ...)` for unrecoverable errors, `ERROR` otherwise
    - If not initialized: fall back to `fprintf(stderr, ...)` (existing behavior)
    - Remove comment about "route through eckit::Log for structured logging"
    - _Requirements: 6.1, 6.4, 6.7_

  - [x] 8.4 Replace eckit::Log calls in driver source files with LOGS
    - In all three driver files, replace any `eckit::Log::info()` → `logs::Logger::log(Severity_Level::INFO, ...)`
    - Replace `eckit::Log::error()` → `logs::Logger::log(Severity_Level::ERROR, ...)`
    - Replace `eckit::Log::warning()` → `logs::Logger::log(Severity_Level::WARNING, ...)`
    - Remove `#include <eckit/log/Log.h>` from all driver files
    - _Requirements: 6.2, 6.3, 6.6, 9.8_

  - [x] 8.5 Write unit tests for LOGS integration
    - Test that Logger is initialized after amio_init completes
    - Test that pre-initialization diagnostics fall back to stderr
    - Test that post-initialization diagnostics route through Logger
    - _Requirements: 6.7, 6.8, 12.1_

- [x] 9. Checkpoint - Verify Exception_Bridge, Factory, and LOGS
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Update CMake build system to remove eckit and add HELM dependencies
  - [x] 10.1 Remove eckit from CMakeLists.txt and add HELM find_package calls
    - Remove `find_package(eckit CONFIG QUIET)` from the root CMakeLists.txt
    - Add `find_package(CONF REQUIRED)`, `find_package(HALO REQUIRED)`, `find_package(LOGS REQUIRED)`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 10.2 Update amio_core target link libraries
    - Replace `target_link_libraries(amio_core PRIVATE eckit)` with `target_link_libraries(amio_core PRIVATE HELM::CONF HELM::HALO HELM::LOGS)`
    - Remove all `target_compile_definitions(amio_core PRIVATE AMIO_HAS_ECKIT=1)` lines
    - Ensure HELM libraries are PRIVATE (not PUBLIC or INTERFACE) so they don't leak to consumers
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 10.3 Update driver static archive link libraries
    - For `driver_netcdf`, `driver_zarr`, `driver_grib2`: remove `target_link_libraries(... PRIVATE eckit)` and add `target_link_libraries(... PRIVATE HELM::CONF)`
    - Remove all `target_compile_definitions(driver_* PRIVATE AMIO_HAS_ECKIT=1)` lines
    - Remove the `if(eckit_FOUND)` guard blocks around eckit linking for each driver
    - _Requirements: 7.6, 7.9_

  - [x] 10.4 Verify AMIOConfig.cmake does not export HELM dependencies
    - In `cmake/AMIOConfig.cmake.in`, ensure no `find_dependency(CONF)`, `find_dependency(HALO)`, or `find_dependency(LOGS)` is added
    - Confirm no `find_dependency(eckit)` remains
    - _Requirements: 7.7_

  - [x] 10.5 Update configure-time summary message
    - Replace the `eckit` line in the status summary with lines for CONF, HALO, LOGS found status
    - Remove any `eckit_FOUND` variable references
    - _Requirements: 7.8_

- [x] 11. Final eckit removal verification
  - [x] 11.1 Scan all source files for residual eckit references
    - Grep all files under `src/`, `include/`, `fortran/`, and `tests/` for `eckit/`, `<eckit`, `namespace eckit`, `AMIO_HAS_ECKIT`
    - Remove any remaining references found (comments, dead code, conditional blocks)
    - _Requirements: 9.7, 12.1_

  - [x] 11.2 Verify public ABI is unchanged
    - Confirm `include/amio/amio.h`, `include/amio/amio_errors.h`, `include/amio/amio_types.h`, `include/amio/amio_export.h`, `include/amio/amio_mdspan_fwd.h` have zero modifications to signatures, enums, or macros
    - Confirm `fortran/amio_mod.f90` has zero changes to `bind(C)` interface blocks
    - Confirm no CONF/HALO/LOGS type appears in any public header
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 11.4_

  - [x] 11.3 Write static scan test for CI
    - Create a test script or CTest that scans all AMIO source/header files for `eckit` include patterns
    - The test SHALL fail if any `#include` referencing `eckit/` is found in `src/`, `include/`, or `fortran/`
    - _Requirements: 12.1, 12.9_

  - [x] 11.4 Write build verification test (no-eckit prefix)
    - Create a CTest that configures AMIO with HELM::CONF, HELM::HALO, HELM::LOGS available but eckit absent from CMAKE_PREFIX_PATH
    - Verify the build completes with zero compilation and linker errors
    - _Requirements: 12.2_

  - [x] 11.5 Write nm -D symbol check test
    - Create a test that runs `nm -D libamio.so`, demangles the output, and asserts zero symbols contain the substring `eckit::`
    - _Requirements: 12.8_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major module migration
- The public C99 ABI (`amio.h`, `amio_errors.h`, `amio_mod.f90`) must remain byte-for-byte unchanged throughout
- HELM dependencies (CONF, HALO, LOGS) are always PRIVATE — no type or header leaks to downstream consumers
- Thread-safety guarantees (Requirement 10) are preserved by construction: immutable IOCommunicator after init, shared_mutex in factory, per-(dataset,variable) ordering in Worker_Pool
- No circular dependencies (Requirement 11) are introduced: AMIO depends on CONF/HALO/LOGS, never the reverse

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4"] },
    { "id": 2, "tasks": ["1.5", "1.6"] },
    { "id": 3, "tasks": ["3.1", "3.2", "4.1"] },
    { "id": 4, "tasks": ["3.3", "3.4", "4.2", "4.3"] },
    { "id": 5, "tasks": ["3.5", "4.4", "4.5"] },
    { "id": 6, "tasks": ["6.1", "7.1"] },
    { "id": 7, "tasks": ["6.2", "6.3"] },
    { "id": 8, "tasks": ["6.4", "8.1"] },
    { "id": 9, "tasks": ["8.2", "8.3", "8.4"] },
    { "id": 10, "tasks": ["8.5", "10.1"] },
    { "id": 11, "tasks": ["10.2", "10.3", "10.4", "10.5"] },
    { "id": 12, "tasks": ["11.1", "11.2"] },
    { "id": 13, "tasks": ["11.3", "11.4", "11.5"] }
  ]
}
```
