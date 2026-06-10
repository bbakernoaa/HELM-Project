# Implementation Plan: TICK (Time Integration & Chronology Kernel)

## Overview

Bottom-up implementation of the TICK micro-library — a C++20 stateless time, calendar, alarm, and synchronization library for the HELM ecosystem. Tasks build from low-level value types upward through calendar engines, alarms, synchronization, and accumulation windows, ending with CI integration. All code lives in `libs/tick/` as a Git submodule producing the `HELM::TICK` static library target.

## Tasks

- [x] 1. Repository scaffolding and CMake build system
  - [x] 1.1 Create directory structure and top-level CMakeLists.txt
    - Create `libs/tick/` with subdirectories: `cmake/`, `include/tick/`, `include/tick/detail/`, `src/`, `tests/`
    - Write `CMakeLists.txt` with project(TICK VERSION 0.1.0), cxx_std_20, extensions OFF
    - Define `tick` static library target with `src/sync.cpp` (initially empty placeholder)
    - Define `HELM::TICK` alias target
    - Set up `target_include_directories` with BUILD_INTERFACE and INSTALL_INTERFACE
    - Add `BUILD_TESTING` option defaulting to OFF
    - Add `TICK_RESOLUTION` cache option (SECONDS/NANOSECONDS, default NANOSECONDS) with compile definition propagation
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.7_

  - [x] 1.2 Create CMake install and export configuration
    - Write `cmake/TICKConfig.cmake.in` template
    - Write `cmake/TICKConfigVersion.cmake.in` template (SameMajorVersion compatibility)
    - Add install rules for headers (include/tick), library artifact (lib), and export set
    - Add `configure_package_config_file` and `write_basic_package_version_file` calls
    - _Requirements: 12.6, 12.9_

  - [x] 1.3 Create tests/CMakeLists.txt with GTest and RapidCheck integration
    - Add `find_package(GTest REQUIRED)` and `find_package(rapidcheck REQUIRED)` under BUILD_TESTING
    - Define test executable targets for each test file (test_duration, test_gregorian, test_noleap, test_cal360, test_alarm, test_sync, test_window)
    - Link test targets against `HELM::TICK`, `GTest::gtest_main`, and `rapidcheck`
    - Register tests with `gtest_discover_tests()`
    - _Requirements: 12.5, 12.8_

- [x] 2. Core value types and overflow detection
  - [x] 2.1 Implement detail/overflow.hpp
    - Write overflow-checked addition, subtraction, and multiplication for int64_t
    - Use `__builtin_add_overflow` / `__builtin_sub_overflow` / `__builtin_mul_overflow` for runtime
    - Provide portable constexpr fallback using limit-based pre-checks
    - Throw `std::overflow_error` with descriptive message on overflow
    - _Requirements: 1.10_

  - [x] 2.2 Implement duration.hpp with Duration class and factory functions
    - Define `tick::Duration` class with int64_t nanos storage, explicit constexpr constructor, `nanos()` accessor
    - Implement operator+, operator-, operator* (scalar), operator/ (scalar) with overflow detection
    - Implement compound assignment operators (+=, -=, *=, /=)
    - Implement unary negation operator
    - Throw `std::invalid_argument` on division by zero
    - Default spaceship operator for all comparisons
    - Implement factory functions: `nanoseconds`, `microseconds`, `milliseconds`, `seconds`, `minutes`, `hours`, `days`
    - _Requirements: 1.2, 1.4, 1.5, 1.6, 1.7, 1.10, 1.11, 1.12, 1.13_

  - [x] 2.3 Implement time_point.hpp with Time_Point class
    - Define `tick::Time_Point` class with int64_t nanos storage, explicit constexpr constructor, `nanos()` accessor
    - Implement operator+(Duration), operator-(Duration), operator-(Time_Point) with overflow detection
    - Implement compound assignment operators (+=, -=)
    - Default spaceship operator for all comparisons
    - Define `inline constexpr Time_Point epoch{0}` and nanosecond conversion constants
    - _Requirements: 1.1, 1.8, 1.9, 1.10, 1.12, 1.13, 2.5, 2.7_

  - [x] 2.4 Implement date_time.hpp with Date_Time struct
    - Define `tick::Date_Time` aggregate struct with year, month, day, hour, minute, second, nanosecond fields (all int32_t)
    - Default spaceship operator for structural comparison
    - _Requirements: 3.1, 4.1, 5.1_

  - [x] 2.5 Write property tests for Duration arithmetic (test_duration.cpp - properties)
    - **Property 1: Duration arithmetic is exact integer arithmetic**
    - **Property 17: Comparison operators reflect underlying integer ordering**
    - **Property 18: Factory functions are exact multiplications of conversion factors**
    - **Validates: Requirements 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.12**

  - [x] 2.6 Write unit tests for Duration and Time_Point (test_duration.cpp - examples)
    - Test overflow boundaries (INT64_MAX - 1, INT64_MIN + 1)
    - Test division by zero throws std::invalid_argument
    - Test factory function conversions for known values
    - static_assert constexpr correctness for basic operations
    - _Requirements: 1.10, 1.11, 1.13_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Calendar engines
  - [x] 4.1 Implement calendar.hpp with Calendar concept
    - Define the `tick::Calendar` concept requiring `to_date_time`, `to_time_point`, `days_in_month`, `days_in_year` static members
    - _Requirements: 3.1, 4.1, 5.1_

  - [x] 4.2 Implement gregorian_calendar.hpp
    - Implement `Gregorian_Calendar` struct with static constexpr functions
    - Implement `is_leap_year(int32_t year)` using the standard rule (div 4, not div 100, unless div 400)
    - Implement `days_in_month(year, month)` and `days_in_year(year)`
    - Implement `to_time_point(Date_Time)` using era-based day accumulation + sub-day nanoseconds
    - Implement `to_date_time(Time_Point)` using Hinnant's civil-from-days algorithm
    - Validate all Date_Time fields, throw `std::invalid_argument` on invalid components
    - Implement `add_months` and `add_years` with day-clamping
    - Handle sub-second (nanosecond) preservation during round-trip
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 2.1, 2.8_

  - [x] 4.3 Implement noleap_calendar.hpp
    - Implement `NoLeap_Calendar` struct: every year 365 days, Feb always 28 days
    - Implement `to_time_point` / `to_date_time` using simplified year=365-day arithmetic
    - Implement `days_in_month`, `days_in_year` (constant 365)
    - Throw `std::invalid_argument` on Feb 29 or other invalid components
    - Implement `add_months` / `add_years`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 4.4 Implement cal360_calendar.hpp
    - Implement `Cal360_Calendar` struct: every month 30 days, every year 360 days
    - Implement `to_time_point` / `to_date_time` using trivial division (day_offset/360, remainder/30)
    - Implement `days_in_month` (always 30), `days_in_year` (always 360)
    - Throw `std::invalid_argument` on day > 30, month > 12, or invalid time components
    - Implement `add_months` / `add_years`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 4.5 Write property tests for Gregorian calendar (test_gregorian.cpp)
    - **Property 2: Gregorian calendar round-trip**
    - **Property 5: Gregorian leap year matches mathematical definition**
    - **Property 6: Calendar-month addition preserves validity and advances correctly**
    - **Property 8: Calendar-year addition equals 12 calendar-months**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.7, 11.1**

  - [x] 4.6 Write property tests for NoLeap calendar (test_noleap.cpp)
    - **Property 3: NoLeap calendar round-trip**
    - **Validates: Requirements 4.4, 11.2**

  - [x] 4.7 Write property tests for Cal360 calendar (test_cal360.cpp)
    - **Property 4: Cal360 calendar round-trip**
    - **Property 7: Cal360 month addition is a fixed nanosecond shift**
    - **Validates: Requirements 5.4, 5.8, 5.9, 11.3**

  - [x] 4.8 Write unit tests for calendar edge cases
    - Test known Gregorian dates: 2000-02-29, 1900-03-01, 2100-03-01, 2400-02-29
    - Test century leap year boundaries (1900 not leap, 2000 leap, 2100 not leap)
    - Test NoLeap rejection of Feb 29
    - Test Cal360 uniform 30-day months
    - Test invalid component throws for each calendar
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 4.5, 4.6, 5.5, 5.6, 5.7_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Alarm system
  - [x] 6.1 Implement interval_alarm.hpp
    - Define `Interval_Alarm` class with Duration interval and Time_Point reference as immutable members
    - Constructor validates interval > 0 (throw `std::invalid_argument` for zero or negative)
    - Implement `is_ringing(Time_Point)` using integer modulo: return `current >= reference && (current - reference) % interval == 0`
    - Implement `next_ring_at(Time_Point)` returning the smallest future ring time
    - Provide const accessors `interval()` and `reference()`
    - Ensure copyable and movable
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 6.2 Implement absolute_alarm.hpp
    - Define `Absolute_Alarm` class with Time_Point trigger_time as immutable member
    - Constructor takes Time_Point, marked noexcept
    - Implement `is_ringing(Time_Point)` as exact equality check
    - Implement `has_passed(Time_Point)` as `current > trigger_time`
    - Provide const accessor `trigger_time()`
    - Ensure copyable and movable
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 6.3 Write property tests for alarms (test_alarm.cpp)
    - **Property 9: Interval_Alarm is_ringing matches modulo condition**
    - **Property 10: Interval_Alarm next_ring_at is the smallest future ring time**
    - **Property 11: Absolute_Alarm is_ringing and has_passed match equality/inequality**
    - **Validates: Requirements 6.2, 6.6, 7.2, 7.4, 11.9**

  - [x] 6.4 Write unit tests for alarm error conditions (test_alarm.cpp)
    - Test zero interval throws std::invalid_argument
    - Test negative interval throws std::invalid_argument
    - Test is_ringing returns false before base time
    - Test Absolute_Alarm copy preserves trigger semantics
    - _Requirements: 6.4, 6.5, 6.6, 7.4, 7.5_

- [x] 7. Synchronization (GCD, LCM, phase alignment)
  - [x] 7.1 Implement sync.hpp (declarations) and src/sync.cpp (definitions)
    - Declare `compute_heartbeat(std::span<const Duration>)` returning Duration
    - Declare `compute_sync_period(std::span<const Duration>)` returning Duration
    - Declare `is_phase_aligned(Time_Point, Time_Point, Duration)` as constexpr
    - Implement GCD fold-left over span with `std::gcd` or manual Euclidean algorithm
    - Implement LCM fold-left with overflow-safe `abs(a / gcd(a,b)) * b` pattern
    - Throw `std::invalid_argument` on empty collection or zero/negative steps
    - Throw `std::overflow_error` if LCM overflows int64_t
    - Implement `is_phase_aligned` using modulo check with positive-step validation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 7.2 Write property tests for synchronization (test_sync.cpp)
    - **Property 12: Heartbeat (GCD) divides all time steps**
    - **Property 13: All time steps divide the sync period (LCM)**
    - **Property 14: Phase alignment matches modulo condition**
    - **Validates: Requirements 8.1, 8.2, 9.1, 11.7, 11.8**

  - [x] 7.3 Write unit tests for sync edge cases (test_sync.cpp)
    - Test representative ESM timesteps: 300s (atm), 900s (ocean), 1800s (land) → GCD=300s, LCM=1800s
    - Test single-element collection returns input unchanged
    - Test empty collection throws std::invalid_argument
    - Test zero/negative timestep throws std::invalid_argument
    - Test LCM overflow throws std::overflow_error
    - Test is_phase_aligned at base_time returns true
    - Test is_phase_aligned before base_time returns false
    - _Requirements: 8.3, 8.4, 8.5, 8.8, 8.9, 9.3, 9.4, 9.5_

- [x] 8. Accumulation windows
  - [x] 8.1 Implement time_window.hpp
    - Define `Time_Window` class with start and end Time_Points as immutable members
    - Constructor validates start < end (throw `std::invalid_argument` otherwise)
    - Provide convenience constructor from Time_Point start and Duration
    - Implement `contains(Time_Point)` as half-open interval check: `t >= start && t < end`
    - Implement `duration()` returning end - start
    - Implement `start()` and `end()` accessors
    - Default spaceship operator for comparisons
    - Implement free function `is_on_boundary(Time_Point, Duration)` with positive-interval validation
    - Implement free function `compute_window(Time_Point, Duration)` computing the enclosing window
    - Ensure copyable and movable
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

  - [x] 8.2 Write property tests for windows (test_window.cpp)
    - **Property 15: Time_Window contains matches half-open interval semantics**
    - **Property 16: compute_window returns a window containing current_time on a boundary**
    - **Validates: Requirements 10.4, 10.5, 10.10**

  - [x] 8.3 Write unit tests for window edge cases (test_window.cpp)
    - Test zero-duration window throws std::invalid_argument
    - Test negative-duration window throws std::invalid_argument
    - Test start >= end throws std::invalid_argument
    - Test contains at start (inclusive) returns true
    - Test contains at end (exclusive) returns false
    - Test is_on_boundary with zero duration throws
    - Test compute_window produces correct window for known values
    - _Requirements: 10.3, 10.6, 10.7_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Date-Time formatting/parsing and umbrella header
  - [x] 10.1 Add ISO 8601 parse/format functions to date_time.hpp
    - Implement `parse_iso8601(std::string_view)` → Date_Time, parsing "YYYY-MM-DDThh:mm:ss"
    - Implement `format_iso8601(Date_Time)` → std::string (no heap alloc for ≤20 chars, use small buffer)
    - Validate format structure, throw `std::invalid_argument` on malformed input with byte offset
    - Validate component ranges (month 1-12, hour 0-23, etc.), throw on invalid values
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

  - [x] 10.2 Create tick.hpp umbrella header
    - Include all public headers in dependency order
    - Ensure single `#include <tick/tick.hpp>` provides complete API
    - _Requirements: 14.1, 14.3_

  - [x] 10.3 Create generators.hpp for RapidCheck test infrastructure
    - Write custom RapidCheck `Arbitrary` specializations or generator functions for:
      - Valid `Date_Time` under each calendar (Gregorian, NoLeap, Cal360) with year range ±1000 from epoch
      - Constrained `Duration` values that avoid overflow
      - Collections of 2-8 positive Durations each ≤ 86400000000000 ns (one day)
    - Include overflow-safe precondition filters
    - _Requirements: 11.10, 11.11, 11.12_

- [x] 11. Integration wiring and Tier 1 isolation verification
  - [x] 11.1 Wire all components together and verify build
    - Ensure `src/sync.cpp` includes only necessary tick headers
    - Verify the full library compiles with `cmake --build` producing `libtick.a`
    - Verify all test targets link and compile with BUILD_TESTING=ON
    - Verify no HELM cross-includes exist (grep for halo/, logs/, axis/, amio/, span/, dagr/)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 12.1, 12.2, 12.3_

  - [x] 11.2 Write integration tests validating end-to-end workflows
    - Test: create Date_Time → convert to Time_Point via Gregorian → advance by Duration → convert back → verify
    - Test: set up Interval_Alarm → verify it rings at expected multiples
    - Test: compute_heartbeat with real ESM timesteps → verify phase alignment at heartbeat multiples
    - Test: compute_window → verify contains returns true for points inside
    - _Requirements: 1.8, 3.3, 6.2, 8.1, 10.4_

- [x] 12. CI pipeline and README
  - [x] 12.1 Create GitHub Actions CI workflow for TICK
    - Create `.github/workflows/ci.yml` in the tick repository
    - Configure to trigger on push and pull_request
    - Build inside the helm-project Docker container
    - Run CMake configure with BUILD_TESTING=ON, build, and ctest
    - Add Tier 1 isolation scan step (grep for forbidden includes, fail on match)
    - _Requirements: 13.6, 14.5_

  - [x] 12.2 Write README.md
    - Include prerequisites section (Docker container requirement)
    - Include exact `docker compose` command to launch development container
    - Include exact CMake configure and build commands
    - Include exact CTest command to run the test suite
    - Brief library overview and API summary
    - _Requirements: 13.4_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Design Properties 1–18)
- Unit tests validate specific examples, edge cases, and error conditions
- The library is header-only except for `src/sync.cpp` (GCD/LCM with `std::span`)
- All code targets C++20 with GCC-13 inside the HELM Docker container
- RapidCheck uses `RC_GTEST_PROP` macro for unified GTest integration

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1"] },
    { "id": 3, "tasks": ["2.2", "2.4"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.5", "2.6", "4.1"] },
    { "id": 6, "tasks": ["4.2", "4.3", "4.4"] },
    { "id": 7, "tasks": ["4.5", "4.6", "4.7", "4.8"] },
    { "id": 8, "tasks": ["6.1", "6.2"] },
    { "id": 9, "tasks": ["6.3", "6.4", "7.1"] },
    { "id": 10, "tasks": ["7.2", "7.3", "8.1"] },
    { "id": 11, "tasks": ["8.2", "8.3", "10.1"] },
    { "id": 12, "tasks": ["10.2", "10.3"] },
    { "id": 13, "tasks": ["11.1"] },
    { "id": 14, "tasks": ["11.2"] },
    { "id": 15, "tasks": ["12.1", "12.2"] }
  ]
}
```
