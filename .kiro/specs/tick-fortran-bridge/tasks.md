# Implementation Plan: TICK Fortran Bridge

## Overview

Implement a two-layer C/Fortran bridge for the TICK micro-library. The C ABI layer (`tick_c.h` + `tick_c_api.cpp`) exposes TICK's C++20 value types as flat `extern "C"` functions with integer return codes. The Fortran module (`tick_mod.f90`) wraps those functions with idiomatic `iso_c_binding` interfaces. Both layers are integrated into TICK's existing CMake build system as additional library targets.

All code lives under `libs/tick/`. Builds run inside Docker via `docker compose run --rm helm-dev bash -c "..."` from the workspace root.

## Tasks

- [x] 1. Create C header and types
  - [x] 1.1 Create `include/tick/tick_c.h` with all type definitions, error codes, and function declarations
    - Define `tick_time_point_t` (`int64_t`), `tick_duration_t` (`int64_t`), `tick_status_t` (`int32_t`)
    - Define `tick_date_time_t` struct (7 × `int32_t` fields)
    - Define `tick_calendar_t` enum with three named values
    - Define `TICK_OK` through `TICK_ERR_INTERNAL` error code macros
    - Declare all function prototypes: Time_Point, Duration, Calendar, Alarm, Sync, and Window groups
    - Wrap in `extern "C"` guards and include-guard
    - Include only `<stdint.h>` — self-contained, C99/C++20 dual-compilable
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2_

- [x] 2. Implement C API functions
  - [x] 2.1 Create `src/tick_c_api.cpp` with exception firewall macro and Time_Point/Duration functions
    - Define `TICK_C_TRY` macro wrapping try/catch for `std::overflow_error`, `std::invalid_argument`, and catch-all
    - Implement `tick_strerror` returning static string descriptions
    - Implement `tick_time_point_create`, `tick_time_point_add_duration`, `tick_time_point_sub_duration`, `tick_time_point_diff`, `tick_time_point_compare`
    - Implement `tick_duration_from_nanos`, `tick_duration_from_seconds`, `tick_duration_from_minutes`, `tick_duration_from_hours`, `tick_duration_from_days`
    - Implement `tick_duration_add`, `tick_duration_sub`, `tick_duration_mul`, `tick_duration_div`
    - Null-pointer pre-checks on all output pointers returning `TICK_ERR_INVALID_ARG`
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x] 2.2 Add calendar conversion functions to `src/tick_c_api.cpp`
    - Implement calendar dispatch helper (switch on `tick_calendar_t` enum)
    - Implement `tick_to_date_time`, `tick_to_time_point` with inner try/catch for `TICK_ERR_INVALID_DATE` vs `TICK_ERR_INVALID_ARG`
    - Implement `tick_days_in_month`, `tick_days_in_year`
    - Invalid enum values return `TICK_ERR_INVALID_CALENDAR` before entering try block
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 2.3 Add alarm, sync, and window functions to `src/tick_c_api.cpp`
    - Implement `tick_interval_alarm_is_ringing`, `tick_interval_alarm_next_ring`, `tick_absolute_alarm_is_ringing`
    - Implement `tick_compute_heartbeat`, `tick_compute_sync_period`, `tick_is_phase_aligned`
    - Implement `tick_is_on_boundary`, `tick_compute_window`, `tick_window_contains`
    - Validate non-null pointers, positive intervals, non-zero counts
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 3. CMake integration for C API
  - [x] 3.1 Extend `libs/tick/CMakeLists.txt` to add the `tick_c` target
    - Add `tick_c` STATIC library from `src/tick_c_api.cpp`
    - Create `HELM::TICK_C` alias
    - Link `tick_c` PUBLIC against `tick`
    - Set public include directory so `tick_c.h` is exported
    - Add `tick_c` to the install/export rules alongside existing `tick` target
    - Update `TICKTargets` export set to include `tick_c`
    - _Requirements: 10.1, 10.2, 10.4, 10.6, 11.4_

- [x] 4. Checkpoint — Verify C API builds and links
  - Ensure all tests pass, ask the user if questions arise.
  - Build inside Docker: `docker compose run --rm helm-dev bash -c "cd /workspace/helm-project/libs/tick && cmake -B build -DBUILD_TESTING=ON && cmake --build build"`
  - Verify `tick_c.h` compiles under C99 and C++20 modes

- [x] 5. C API unit tests
  - [x] 5.1 Create `tests/test_tick_c.cpp` with GTest cases for all C API functions
    - Type layout assertions: `sizeof(tick_date_time_t)`, enum values
    - Happy-path tests for Time_Point create/add/sub/diff/compare
    - Happy-path tests for Duration factories and arithmetic
    - Happy-path tests for calendar round-trip, days_in_month, days_in_year
    - Happy-path tests for alarm, sync, and window functions
    - Error-path tests: null pointers → `TICK_ERR_INVALID_ARG`, invalid calendars → `TICK_ERR_INVALID_CALENDAR`, invalid dates → `TICK_ERR_INVALID_DATE`, overflow → `TICK_ERR_OVERFLOW`
    - `tick_strerror` returns non-null for all defined codes and out-of-range values
    - _Requirements: 2.7, 3.6, 4.7, 4.8, 5.5, 5.6, 5.7, 6.4, 7.4, 7.5, 8.4, 8.5_

  - [x] 5.2 Update `tests/CMakeLists.txt` to build `test_tick_c`
    - Add `test_tick_c` executable linked against `HELM::TICK_C`, `GTest::gtest_main`, and `rapidcheck`
    - Register with `gtest_discover_tests`
    - _Requirements: 10.1_

  - [x] 5.3 Write property test for calendar round-trip (Property 1)
    - **Property 1: Calendar Conversion Round-Trip**
    - Generate random valid `tick_date_time_t` for each calendar, convert to time_point and back, assert equality
    - Minimum 100 iterations
    - **Validates: Requirements 5.7**

  - [x] 5.4 Write property test for Time_Point arithmetic equivalence (Property 2)
    - **Property 2: Time_Point Arithmetic Equivalence**
    - Generate random `int64_t` pairs filtered for no overflow, verify add/sub/diff/compare consistency
    - Minimum 100 iterations
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5**

  - [x] 5.5 Write property test for Duration arithmetic equivalence (Property 3)
    - **Property 3: Duration Arithmetic Equivalence**
    - Generate random duration pairs and scalars filtered for no overflow/zero-div, verify add/sub/mul/div correctness
    - Minimum 100 iterations
    - **Validates: Requirements 4.3, 4.4, 4.5, 4.6**

  - [x] 5.6 Write property test for Duration factory correctness (Property 4)
    - **Property 4: Duration Factory Correctness**
    - Generate random `int64_t` counts filtered for no overflow, verify nanosecond conversion matches expected multiplier
    - Minimum 100 iterations
    - **Validates: Requirements 4.1, 4.2**

  - [x] 5.7 Write property test for exception-to-error-code mapping (Property 5)
    - **Property 5: Exception-to-Error-Code Mapping**
    - Generate inputs that trigger overflow and invalid_argument, verify correct return codes and output unchanged
    - Minimum 100 iterations
    - **Validates: Requirements 2.4, 2.5, 3.6, 4.8**

  - [x] 5.8 Write property test for tick_strerror completeness (Property 6)
    - **Property 6: tick_strerror Completeness**
    - For all `int32_t` values in [0,5] and random out-of-range values, verify non-null non-empty return
    - Minimum 100 iterations
    - **Validates: Requirements 2.7**

  - [x] 5.9 Write property test for interval alarm equivalence (Property 7)
    - **Property 7: Interval Alarm Equivalence**
    - Generate random positive intervals and time points, verify is_ringing iff `(current - reference) % interval == 0`, and next_ring is valid
    - Minimum 100 iterations
    - **Validates: Requirements 6.1, 6.2**

  - [x] 5.10 Write property test for heartbeat divides all timesteps (Property 8)
    - **Property 8: Heartbeat Divides All Timesteps**
    - Generate random arrays of positive durations, verify `timesteps[i] % heartbeat == 0` for all i
    - Minimum 100 iterations
    - **Validates: Requirements 7.1**

  - [x] 5.11 Write property test for sync period divisible by all timesteps (Property 9)
    - **Property 9: Sync Period Divisible By All Timesteps**
    - Generate random arrays of small positive durations (LCM won't overflow), verify `sync_period % timesteps[i] == 0` for all i
    - Minimum 100 iterations
    - **Validates: Requirements 7.2**

  - [x] 5.12 Write property test for window and boundary operations (Property 10)
    - **Property 10: Window and Boundary Operations Equivalence**
    - Generate random time points and positive intervals, verify boundary/window/contains consistency
    - Minimum 100 iterations
    - **Validates: Requirements 7.3, 8.1, 8.2, 8.3**

- [x] 6. Checkpoint — Verify C API tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run: `docker compose run --rm helm-dev bash -c "cd /workspace/helm-project/libs/tick && cmake -B build -DBUILD_TESTING=ON && cmake --build build && ctest --test-dir build --output-on-failure"`

- [x] 7. Fortran module
  - [x] 7.1 Create `fortran/tick_mod.f90` with iso_c_binding interfaces
    - Declare `tick_date_time` derived type with `bind(c)` and seven `c_int32_t` fields
    - Declare named integer constants for all error codes (`TICK_OK` through `TICK_ERR_INTERNAL`)
    - Declare named integer constants for calendar enum values
    - Declare `interface` blocks with `bind(c, name="...")` for every C API function
    - All arguments have explicit `intent(in)` or `intent(out)` attributes
    - Use `value` attribute for scalar input arguments (int64, int32)
    - Module is `implicit none` and `private` with explicit `public` statements
    - No implementation logic — interface declarations only
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 11.3_

- [x] 8. CMake Fortran support
  - [x] 8.1 Extend `libs/tick/CMakeLists.txt` with gated `tick_fortran` target
    - Add `option(TICK_BUILD_FORTRAN "Build the Fortran tick_mod module" OFF)`
    - Guard with `if(TICK_BUILD_FORTRAN)` block
    - Call `enable_language(Fortran)` inside the guard
    - Add `tick_fortran` STATIC library from `fortran/tick_mod.f90`
    - Create `HELM::TICK_Fortran` alias
    - Link `tick_fortran` PUBLIC against `tick_c`
    - Install `.mod` file to include directory
    - Add `tick_fortran` to install/export rules
    - _Requirements: 10.3, 10.5, 10.6_

- [x] 9. Fortran integration tests
  - [x] 9.1 Create `tests/test_tick_fortran.f90` — integration test program
    - Call every `tick_mod` procedure at least once
    - Verify return codes match expected values for known inputs
    - Confirm calendar round-trip from Fortran (Gregorian, NoLeap, Cal360)
    - Test error paths: pass invalid calendar, verify `TICK_ERR_INVALID_CALENDAR` returned
    - Use `stop 1` on failure for ctest integration
    - _Requirements: 5.7, 9.1, 9.5_

  - [x] 9.2 Update `tests/CMakeLists.txt` to build Fortran test (gated by `TICK_BUILD_FORTRAN`)
    - Add `test_tick_fortran` executable from Fortran source
    - Link against `HELM::TICK_Fortran`
    - Register with `add_test`
    - Guard inside `if(TICK_BUILD_FORTRAN)` block
    - _Requirements: 10.3_

- [x] 10. Final checkpoint — Build verification
  - Ensure all tests pass, ask the user if questions arise.
  - Full Docker build with Fortran enabled: `docker compose run --rm helm-dev bash -c "cd /workspace/helm-project/libs/tick && cmake -B build -DBUILD_TESTING=ON -DTICK_BUILD_FORTRAN=ON && cmake --build build && ctest --test-dir build --output-on-failure"`
  - Verify `tick_c.h` compiles as C99: `gcc -std=c99 -pedantic -Werror -fsyntax-only include/tick/tick_c.h`
  - Verify `tick_c.h` compiles as C++20: `g++ -std=c++20 -Werror -fsyntax-only include/tick/tick_c.h`
  - Verify `tick_mod.f90` compiles with gfortran: `gfortran -std=f2003 -Wall -c fortran/tick_mod.f90`
  - Confirm Tier 1 isolation: no includes from SPAN, DAGR, HALO, AXIS, LOGS, or AMIO in `tick_c_api.cpp`
  - _Requirements: 1.5, 11.1, 11.2, 11.3, 11.4_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation inside Docker
- Property tests validate universal correctness properties defined in the design document
- Unit tests validate specific examples and edge cases
- All builds execute inside Docker via `docker compose run --rm helm-dev bash -c "..."`
- The workspace is mounted at `/workspace/helm-project` inside the container
- The `tick_c` target is always built; `tick_fortran` is gated by `TICK_BUILD_FORTRAN=ON`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["5.1", "5.2"] },
    { "id": 5, "tasks": ["5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "5.9", "5.10", "5.11", "5.12"] },
    { "id": 6, "tasks": ["7.1"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["9.1", "9.2"] }
  ]
}
```
