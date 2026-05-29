# Implementation Plan: CONF — Configuration Parser Micro-Library

## Overview

This plan converts the approved CONF design into an incremental, test-driven build
sequence for a standalone micro-library under `libs/conf/` producing `HELM::CONF`.
The order mirrors the layering in the design: build system first, then the error
model, the private yaml-cpp backend, the public `conf::Config` / `conf::Value` API,
the `extern "C"` Fortran bridge with its handle registry, and finally the `conf_mod`
Fortran module — each layer wired into the previous so no orphaned code remains.

Implementation language is **C++20** (core + C bridge), **Fortran 2008**
(`conf_mod` + integration test), and **CMake 3.21+** (build), exactly as fixed by the
design. yaml-cpp (pinned 0.8.0) is a PRIVATE implementation detail and appears only
inside `src/detail/`.

Per the HELM testing convention (and mirroring HALO), test sub-tasks are placed
under the implementation task they validate, marked optional with `*`. The test
harness (`tests/CMakeLists.txt`) is created up front with per-file `EXISTS` guards so
each test source can be built as soon as it is authored. Property tests (RapidCheck)
realize the design's six correctness properties P1–P6; unit tests (Google Test)
cover the mandated example-based behaviors.

## Tasks

- [x] 1. Project scaffolding and standalone CMake build system
  - [x] 1.1 Author the root `libs/conf/CMakeLists.txt`
    - Declare `project(CONF VERSION 0.1.0 LANGUAGES CXX)`, require CMake 3.21+
    - Set C++20 as required with extensions OFF (`CMAKE_CXX_EXTENSIONS OFF`)
    - Define `add_library(conf)` and `add_library(HELM::CONF ALIAS conf)`; list core sources (`src/config.cpp`, `src/value.cpp`, `src/detail/yaml_tree.cpp`; `src/fortran/conf_c_interop.cpp` under `BUILD_FORTRAN`)
    - Expose `include/` as PUBLIC and `src/` (with `detail/`) as PRIVATE include path
    - Acquire yaml-cpp privately: `FetchContent` pinned to `0.8.0` (tests/tools/install OFF) with a `CONF_USE_SYSTEM_YAMLCPP` + `find_package(yaml-cpp 0.8)` Spack fallback; link it `PRIVATE` to `conf`
    - Add options `BUILD_TESTING` (default OFF), `BUILD_FORTRAN` (default ON), `CONF_USE_SYSTEM_YAMLCPP` (default OFF)
    - Configure install/export: install the `conf` target as `CONFTargets` under the `HELM::` namespace, generate `CONFConfig.cmake`/`CONFConfigVersion.cmake` (`SameMajorVersion`), install only `include/conf/**` public headers (never `detail/`)
    - Reference no `HELM::` target other than `HELM::CONF`; link neither Kokkos nor MPI
    - Wire `BUILD_TESTING`→`enable_testing()`+`add_subdirectory(tests)` and `BUILD_FORTRAN`→`enable_language(Fortran)`; add `tests_fortran` when both are ON
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 24.1, 24.2, 24.3, 24.4, 25.1, 25.2, 25.3, 25.4, 25.5, 26.2, 26.4_

  - [x] 1.2 Author the cmake package-config templates
    - Create `cmake/CONFConfig.cmake.in` that includes `CONFTargets.cmake` and **deliberately omits** `find_dependency(yaml-cpp)` (with an explanatory comment that yaml-cpp is a PRIVATE, statically-absorbed detail), then `check_required_components(CONF)`
    - Create `cmake/CONFConfigVersion.cmake.in` implementing `SameMajorVersion` compatibility
    - _Requirements: 23.5, 24.5, 26.3_

  - [x] 1.3 Author the test-harness CMake with per-file guards (mirrors HALO `tests/CMakeLists.txt`)
    - In `tests/CMakeLists.txt`: locate Google Test and RapidCheck (RapidCheck via cached build / FetchContent like HALO), add a helper that links each test against `conf` + GTest + RapidCheck, registers it with CTest and `LABELS "property"` / `"unit"`, and guards every planned source with `if(EXISTS ...)` (the unit + property + registry files referenced by later tasks)
    - Give `test_handle_registry` and the registry property test access to the private `src/` include path (registry header is in `src/fortran/`)
    - In `tests_fortran/CMakeLists.txt`: guard-add the Fortran integration test executable (built only when `BUILD_FORTRAN` and `BUILD_TESTING` are ON), linking `conf` and the `conf_mod` objects, and register it with CTest
    - Configure property tests to run single-process (no MPI/`mpirun`) with at least 100 generated iterations
    - _Requirements: 25.2, 25.4, 25.5, 33.2, 33.3, 34.1_

- [x] 2. Core error model
  - [x] 2.1 Implement `conf::Error_Code` and `conf::Conf_Error` in `include/conf/error.hpp`
    - Define `enum class Error_Code : int` with exactly the nine values: `Success=0`, `Invalid_Arg=1`, `File_Not_Found=2`, `Parse_Error=3`, `Key_Not_Found=4`, `Type_Mismatch=5`, `Bad_Handle=6`, `Buffer_Too_Small=7`, `Unknown=99` (append-only, never renumber)
    - Define `Conf_Error : std::runtime_error` carrying one `Error_Code` plus a non-empty message, exposing `code()`
    - Include only standard-library headers (no yaml-cpp, no HELM headers)
    - _Requirements: 11.1, 11.2, 11.3, 26.3_

  - [x] 2.2 Write unit tests for the error model
    - `tests/test_error_model.cpp`: assert each enumerator's exact integer value and that `Conf_Error::code()` returns the value supplied at construction with a non-empty `what()`
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 3. Private YAML backend (`conf::detail::Yaml_Tree`)
  - [x] 3.1 Declare the backend in `src/detail/yaml_tree.hpp`
    - The only header that `#include <yaml-cpp/yaml.h>`; declare `from_file`/`from_string`, `resolve(std::string_view)`, and `convert<T>(node)`; hold the `YAML::Node` root
    - Never placed under `include/` and never installed
    - _Requirements: 1.1, 1.2, 2.1, 3.1, 14.1_

  - [x] 3.2 Implement `src/detail/yaml_tree.cpp`
    - `from_file`/`from_string`: load via yaml-cpp, mapping a non-openable path to `File_Not_Found`, a YAML syntax error to `Parse_Error` (preserving the backend's diagnostic text), and empty/whitespace/comment-only input to a valid `Null`/`Undefined` root
    - Dotted-path resolver per the design algorithm: split on `.`; reject empty path or any empty segment with `Invalid_Arg` before walking; advance map nodes by literal byte-for-byte key (no trimming/case-folding), sequence nodes by ASCII-decimal in-range index; raise `Key_Not_Found` on absent key, non-digit/out-of-range/oversized-digit index, or descending past a scalar/null; guarantee totality (never crash/abort/leak on any input)
    - `convert<T>`: enforce scalar-or-string, delegate to yaml-cpp `as<T>()`, translate conversion failure to `Type_Mismatch`; preserve node `Node_Kind` mapping (`Map`/`Sequence`/`Scalar`/`Null`/`Undefined`) and child ordering/whitespace
    - _Requirements: 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 14.1, 14.2_

- [x] 4. `conf::Config` public RAII API
  - [x] 4.1 Declare `conf::Config` in `include/conf/config.hpp`
    - Move-only (deleted copy), pimpl over a `std::unique_ptr<Impl>`; forward-declare `conf::Value`
    - Declare named factories `from_file`/`from_string`; throwing scalar accessors `get_int/get_double/get_bool/get_string`; non-throwing `try_int/try_double/try_bool/try_string` (`noexcept`); `get_or<T>` template (`noexcept`); list accessors `get_int_list/get_double_list/get_string_list`; introspection `has/is_map/is_sequence/size` (`noexcept`); `at` returning `Value`
    - Include only standard-library + CONF public headers; expose no yaml-cpp type
    - _Requirements: 4.1, 4.3, 5.1, 6.1, 6.4, 7.1, 7.5, 8.1, 9.1, 9.7, 10.1, 26.3_

  - [x] 4.2 Implement `src/config.cpp`
    - `Impl` owns a `detail::Yaml_Tree`; factories construct via the backend; no raw `new`/`delete` in `conf::`
    - Move ctor/assign: leave moved-from source in a valid empty state (`has`/`is_map`/`is_sequence`→false, `size`→0, frees nothing), release a prior tree on move-assign, and be safe under self-move-assignment
    - Throwing accessors resolve then convert (`Invalid_Arg` checked first, then `Key_Not_Found`, then `Type_Mismatch`); `try_*`/`get_or`/introspection swallow `Conf_Error` and return `nullopt`/fallback/false/0 and are `noexcept`
    - List accessors return one converted element per sequence item in order (empty vector for empty sequence), raise `Key_Not_Found`/`Type_Mismatch`/`Invalid_Arg` as specified, and never return a partial vector on element mismatch
    - `at` resolves and returns a `Value` view; each instance owns an independent tree; remain valid after any failed call
    - _Requirements: 1.1, 1.2, 1.3, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 12.1, 12.2, 12.4, 13.1, 13.2, 15.1, 15.2, 15.4, 16.1, 16.2_

  - [x] 4.3 Write loading unit tests
    - `tests/test_config_load.cpp`: valid file loads and a known query returns the expected value; valid string loads; `from_file` on a missing path raises `Conf_Error` whose `code()==File_Not_Found`; query results are immune to later mutation of the source
    - _Requirements: 32.1, 1.1, 1.2, 1.3, 1.4_

  - [x] 4.4 Write malformed-YAML unit tests
    - `tests/test_malformed_yaml.cpp`: bad indentation / unclosed bracket / tab-indent raise `Conf_Error` whose `code()==Parse_Error` without crashing or aborting; message is non-empty
    - _Requirements: 32.2, 14.1, 14.2_

  - [x] 4.5 Write missing-key unit tests
    - `tests/test_missing_keys.cpp`: missing leaf, missing intermediate, out-of-range sequence index, and descend-past-scalar each raise `Key_Not_Found` for throwing accessors and yield `nullopt` for `try_*`
    - _Requirements: 32.3, 13.1, 13.2_

  - [x] 4.6 Write type-casting unit tests
    - `tests/test_type_casting.cpp`: int/double/bool/string happy paths return expected values; int-on-string and bool-on-arbitrary mismatches raise `Type_Mismatch` (throwing) and return `nullopt`/fallback (non-throwing/`get_or`)
    - _Requirements: 32.4, 5.1, 5.3, 6.1, 6.3, 7.3, 15.1, 15.2_

  - [x] 4.7 Write dotted-path resolver unit tests
    - `tests/test_dotted_path.cpp`: deep nesting `a.b.c.d`, sequence index `list.0.x`; empty path / leading-dot / trailing-dot / double-dot raise `Invalid_Arg`; whitespace-significant keys matched literally
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 16.1, 16.2_

  - [x] 4.8 Write property test for round-trip fidelity
    - `tests/prop_round_trip.cpp`
    - **Property 1: Round-Trip** — generated typed key/value maps serialized to YAML and read back with the matching accessor equal the originals (32-bit ints exact, doubles ≥17 sig-digits bit-exact, bools exact, strings 0–65535 bytes byte-identical)
    - **Validates: Requirements 27.1, 27.2, 27.3, 27.4, 27.5**

  - [x] 4.9 Write property test for type-safety
    - `tests/prop_type_safety.cpp`
    - **Property 2: Type-Safety** — for a node of type A, an incompatible request B never returns a value: `try_B`→`nullopt`, `get_B` raises `Type_Mismatch`, `get_or`→fallback; never returns garbage, never crashes
    - **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

  - [x] 4.10 Write property test for resolver totality
    - `tests/prop_path_resolution.cpp`
    - **Property 3: Resolver Totality** — for any arbitrary key string (0–1,048,576 bytes: empty, leading/trailing/consecutive dots, UTF-8, non-UTF-8 bytes, oversized digit runs) `try_*`/`get_or`/introspection return a defined result and throwing accessors return a value or a typed `Conf_Error`, with no crash/abort/hang/leak/UB
    - **Validates: Requirements 29.1, 29.2, 29.3, 3.7**

- [x] 5. `conf::Value` resolved-node view
  - [x] 5.1 Declare `conf::Value` and `Node_Kind` in `include/conf/value.hpp`
    - Define `enum class Node_Kind { Undefined, Null, Scalar, Sequence, Map }`
    - Declare `Value` as a non-owning view: `kind`, `is_defined`, `size`; throwing `as_int/as_double/as_bool/as_string`; non-throwing `try_int/try_double/try_bool/try_string` (`noexcept`); private type-erased node pointer and `friend class Config`
    - Expose no yaml-cpp type
    - _Requirements: 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 26.3_

  - [x] 5.2 Implement `src/value.cpp`
    - Interpret the type-erased node via `detail` helpers; `kind` reports exactly one `Node_Kind`; `size` returns child count for map/sequence and 0 for scalar/null/undefined; `as_*` convert scalars (raising `Type_Mismatch` on non-scalar or unparseable scalar); `try_*` mirror `as_*` returning `nullopt` on failure and are `noexcept`; the view neither owns nor copies the parent tree
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9_

  - [x] 5.3 Write unit tests for `Value`
    - `tests/test_value.cpp`: `Config::at` on a defined node returns a `Value`; `at` on a missing/malformed path raises `Key_Not_Found`/`Invalid_Arg`; `kind`/`size` correct across map/sequence/scalar/null; `as_*`/`try_*` agree on success and on failure
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

- [x] 6. Checkpoint — core C++ library
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Fortran C-API bridge
  - [x] 7.1 Implement the handle registry in `src/fortran/handle_registry.hpp`
    - Thread-safe singleton mirroring HALO: `register_handle` returns a unique token `>0`; `0` (`CONF_HANDLE_INVALID`) is reserved and never issued; `lookup` returns the pointer or null; `release` removes the mapping and returns the pointer or null; `valid` reports membership; tokens are monotonic and never reused; guard all state with `std::mutex`; non-copyable/non-movable
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 31.1, 31.2, 31.3_

  - [x] 7.2 Implement the `extern "C"` bridge in `src/fortran/conf_c_interop.cpp`
    - Define the `CONF_C_TRY` macro: return `Success` on no-throw; map `Conf_Error`→its `code()`, `std::invalid_argument`→`Invalid_Arg`, `std::bad_alloc`/other `std::exception`/`...`→`Unknown`; guarantee no exception escapes into Fortran
    - Lifecycle: `conf_load_c`, `conf_load_string_c` (register a handle only on success; never register a failed load), `conf_close_c` (release + delete, RAII-frees the tree)
    - Existence/structure: `conf_has_key_c` (sets 0/1, missing key is not an error), `conf_size_c` (0 for scalar/null/missing, `Success`)
    - Scalar getters: `conf_get_int_c`, `conf_get_double_c`, `conf_get_bool_c` (writes 1/0)
    - Length-first string getters: `conf_get_string_len_c` (reports exact byte count) and `conf_get_string_c` (copies into the caller buffer, returns `Buffer_Too_Small` writing nothing when capacity is short, supports the 0-length case, never null-terminates, retains no static buffer, performs no allocation outliving the call)
    - Validate every required pointer/length: null required pointer or `path_len`/`text_len`/`key_len < 1`→`Invalid_Arg` (a buffer capacity of 0 alone is not `Invalid_Arg`); unregistered/released/0 handle→`Bad_Handle` without dereferencing any `Config`; leave caller outputs unchanged on any non-zero return
    - _Requirements: 12.3, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 19.1, 19.2, 19.3, 19.4, 19.5, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 21.1, 21.2, 21.3, 21.4, 30.1, 30.2, 30.3_

  - [x] 7.3 Write unit tests for the handle registry
    - `tests/test_handle_registry.cpp`: register/lookup/release; released token invalid forever; `0` never issued; token never reused after a register/release/register cycle (links the private `src/` include path)
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.7_

  - [x] 7.4 Write unit tests for the C bridge
    - `tests/test_c_bridge.cpp`: each error path returns its exact `Error_Code` (`Invalid_Arg`, `Bad_Handle`, `Key_Not_Found`, `Type_Mismatch`, `Parse_Error`, `File_Not_Found`); length-first round trip asserts `conf_get_string_len_c` reports N and `conf_get_string_c` writes N bytes returning `Success`; a too-small buffer writes nothing and returns `Buffer_Too_Small`; a closed handle returns `Bad_Handle` on reuse
    - _Requirements: 17.4, 17.5, 17.6, 20.1, 20.2, 20.3, 21.4, 32.5, 32.6_

  - [x] 7.5 Write property test for handle uniqueness and non-reuse
    - `tests/prop_handle_registry.cpp`
    - **Property 5: Handle Uniqueness & Non-Reuse** — for any interleaving of register/release, all issued tokens are distinct positive integers, `0` is never issued, and a released token is never valid or re-issued again
    - **Validates: Requirements 31.1, 31.2, 31.3**

  - [x] 7.6 Write property test for the no-leak C-bridge invariant
    - `tests/prop_no_leak_c_bridge.cpp`
    - **Property 4: No-Leak on the C Bridge** — for any generated `load → N queries → close` sequence (including string queries and failed loads), resident CONF-owned heap bytes after the sequence equal those before, observed via leak sanitizer / allocation counter
    - **Validates: Requirements 30.1, 30.2, 30.3**

  - [x] 7.7 Write property test for error-code / exception lockstep
    - `tests/prop_error_lockstep.cpp`
    - **Property 6: Error-Code/Exception Lockstep** — for every failure mode, the `Error_Code` carried by the thrown `Conf_Error` equals the `int` the corresponding bridge function returns
    - **Validates: Requirements 11.4, 19.2**

- [x] 8. Fortran `conf_mod` module
  - [x] 8.1 Implement `fortran/conf_mod.f90`
    - `use, intrinsic :: iso_c_binding`; declare `bind(C)` interface blocks for each wrapped bridge function (passing `character(len=*)` arguments together with their length)
    - Declare exactly nine public integer parameters, one per `Error_Code` value, each equal to the enumerator's integer
    - Provide thin wrappers (`conf_load`, `conf_close`, `conf_get_int`, `conf_get_real`, `conf_get_logical`, `conf_has_key`) returning the `Error_Code` via an `intent(out)` status; store each `Opaque_Handle` as `integer(c_int)` and pass it back unchanged
    - Implement `conf_get_string` to perform the length-first protocol internally (query length → allocate `character(len=:)` of exactly that size → copy), returning a zero-length string and non-`Success` status on miss/mismatch without leaving a partial allocation; compatible with gfortran 9+, ifort/ifx, nvfortran
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8_

  - [x] 8.2 Write the Fortran integration test
    - `tests_fortran/test_conf_fortran.f90` plus a fixture YAML: load the fixture, query an integer, a real, and a string (exercising the allocatable wrapper so the returned length equals the value's byte count and contents match, with no manual allocate/free), assert each status `== CONF_SUCCESS`, then close the handle via the wrapper and assert `CONF_SUCCESS`
    - _Requirements: 34.1, 34.2, 34.3_

- [x] 9. Integration and final wiring
  - [x] 9.1 Author the umbrella header `include/conf/conf.hpp`
    - Include only CONF's own public headers (`error.hpp`, `value.hpp`, `config.hpp`); include no yaml-cpp type and no HELM header, upholding Tier-1 isolation
    - _Requirements: 26.1, 26.3, 23.4_

- [x] 10. Final checkpoint — full build (core + C bridge + Fortran + tests)
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific granular requirement clauses for traceability.
- Property tests (RapidCheck) realize the design's six correctness properties P1–P6 and are placed next to the code they validate; they run single-process with ≥100 iterations and require neither MPI nor `mpirun`.
- yaml-cpp stays a PRIVATE detail confined to `src/detail/`; `CONFConfig.cmake.in` deliberately omits `find_dependency(yaml-cpp)`.
- The test harness (1.3) lists every test file behind an `EXISTS` guard, so optional test sources build the moment they are authored.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "2.1", "7.1"] },
    { "id": 1, "tasks": ["1.3", "3.1", "4.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "7.3", "7.5", "9.1"] },
    { "id": 3, "tasks": ["4.2", "5.2"] },
    { "id": 4, "tasks": ["4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "5.3", "7.2"] },
    { "id": 5, "tasks": ["7.4", "7.6", "7.7", "8.1"] },
    { "id": 6, "tasks": ["8.2"] }
  ]
}
```
