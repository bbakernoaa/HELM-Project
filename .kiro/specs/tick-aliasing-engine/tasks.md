# Implementation Plan: Time Aliasing Engine

## Overview

Implement the `Aliasing_Engine` class template extending the TICK micro-library with stateless, policy-driven temporal aliasing. All new library code is header-only (C++20 class template). A single test file exercises 11 correctness properties via RapidCheck + GTest, plus unit tests for edge cases and error conditions.

## Tasks

- [x] 1. Create foundational types and build integration
  - [x] 1.1 Create `include/tick/out_of_bounds_policy.hpp`
    - Define `OutOfBoundsPolicy` as `enum class : std::uint8_t` with four enumerators: `clamp_to_edge = 0`, `cycle_last_year = 1`, `pure_climatology = 2`, `leap_hold = 3`
    - Place in `tick` namespace with `#pragma once` and `<cstdint>` include
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 1.2 Create `include/tick/aliased_window.hpp`
    - Define `AliasedWindow` aggregate struct with `Time_Window window` and `double alpha` members
    - Implement `operator==` using `window == other.window` and `std::memcmp` for bitwise double comparison
    - Include `<cstring>` for `memcmp` and `"tick/time_window.hpp"`
    - Provide a static `zero()` factory that returns `AliasedWindow{Time_Window{Time_Point{0}, Time_Point{1}}, 0.0}` for default construction scenarios
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 1.3 Update umbrella header `include/tick/tick.hpp`
    - Add includes for `"tick/out_of_bounds_policy.hpp"`, `"tick/aliased_window.hpp"`, and `"tick/aliasing_engine.hpp"` in a new "Aliasing" section comment block after the "Accumulation windows" section
    - _Requirements: 7.4_

  - [x] 1.4 Add `test_aliasing_engine` to `TICK_TEST_SOURCES` in `tests/CMakeLists.txt`
    - Insert `test_aliasing_engine` into the `TICK_TEST_SOURCES` list, following the existing pattern
    - _Requirements: 11.7_

- [x] 2. Implement Aliasing_Engine class template
  - [x] 2.1 Create `include/tick/aliasing_engine.hpp` with constructor and accessors
    - Define class template `Aliasing_Engine<Calendar Sim_Cal, Calendar Ds_Cal>` in `tick` namespace
    - Implement constructor accepting `Time_Window coverage`, `Duration snapshot_interval`, `OutOfBoundsPolicy policy`, `std::int32_t climatological_year = 0`
    - Constructor validation: throw `std::invalid_argument` if `snapshot_interval <= 0`, if `coverage.duration().nanos() % snapshot_interval.nanos() != 0`, or if `pure_climatology` policy has `climatological_year` outside coverage year range
    - Store all parameters as `const` members; implement const accessors `coverage()`, `snapshot_interval()`, `policy()`, `climatological_year()`
    - Include all necessary TICK headers (`time_point.hpp`, `duration.hpp`, `time_window.hpp`, `date_time.hpp`, `calendar.hpp`, `out_of_bounds_policy.hpp`, `aliased_window.hpp`)
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6, 7.7, 7.8, 9.1, 9.4_

  - [x] 2.2 Implement `calculate_weight` static method
    - Implement as `static constexpr double calculate_weight(Time_Point current, Time_Window window)`
    - Throw `std::invalid_argument` if `window.duration().nanos() == 0` (degenerate window)
    - Throw `std::out_of_range` if `current < window.start()` or `current >= window.end()`
    - Compute `alpha = static_cast<double>(current.nanos() - window.start().nanos()) / static_cast<double>(window.end().nanos() - window.start().nanos())`
    - All differences computed as `int64_t` first, single `double` division at the end
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.1, 10.2_

  - [x] 2.3 Implement `resolve_in_bounds` and `enclosing_window` private helpers
    - `enclosing_window(Time_Point t, Time_Point coverage_start, Duration interval)`: use floor division on nanoseconds to find the window containing `t`, return `Time_Window{floor_start, floor_start + interval}`
    - `resolve_in_bounds(Time_Point sim_time)`: call `enclosing_window`, then `calculate_weight`, return `AliasedWindow{window, alpha}`
    - All arithmetic in `int64_t` nanoseconds; use existing `compute_window` logic from `time_window.hpp` as reference
    - _Requirements: 3.3, 10.1, 10.4_

  - [x] 2.4 Implement `resolve_clamp` private method
    - If `sim_time >= coverage_.end()`: return `AliasedWindow` with window `[coverage_end - interval, coverage_end)` and `alpha = 0.0`
    - If `sim_time < coverage_.start()`: return `AliasedWindow` with window `[coverage_start, coverage_start + interval)` and `alpha = 0.0`
    - _Requirements: 3.1, 3.2_

  - [x] 2.5 Implement `resolve_cycle` private method
    - Decompose `sim_time` via `Sim_Cal::to_date_time(sim_time)` to get Date_Time components
    - If `sim_time >= coverage_.end()`: substitute year with last complete year of coverage (derived from `Ds_Cal::to_date_time(coverage_.end() - Duration{1}).year`)
    - If `sim_time < coverage_.start()`: substitute year with first complete year of coverage (derived from `Ds_Cal::to_date_time(coverage_.start()).year`)
    - Clamp day using `Ds_Cal::days_in_month(target_year, month)` to handle Feb 29 → Feb 28 in NoLeap
    - Convert back via `Ds_Cal::to_time_point(remapped_dt)`, then call `resolve_in_bounds` on the result
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.6 Implement `resolve_climatology` private method
    - Decompose `sim_time` via `Sim_Cal::to_date_time(sim_time)` to get Date_Time components
    - Substitute year with `climatological_year_`, preserve month/day/hour/minute/second/nanosecond
    - Clamp day using `Ds_Cal::days_in_month(climatological_year_, month)`
    - Convert back via `Ds_Cal::to_time_point(remapped_dt)`
    - Compute `enclosing_window` on the remapped time point
    - Handle December-to-January year boundary: if window end crosses year boundary, construct `t_right` using `climatological_year_ + 1`
    - Call `calculate_weight` and return `AliasedWindow`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 2.7 Implement `resolve_leap_hold` private method and `is_feb29` helper
    - `is_feb29(const Date_Time& dt)`: return `dt.month == 2 && dt.day == 29`
    - In `resolve_leap_hold`: decompose via `Sim_Cal::to_date_time(sim_time)`
    - If `is_feb29` AND `Sim_Cal` is `Gregorian_Calendar` AND `Ds_Cal` is `NoLeap_Calendar`:
      - Construct `t_left` as Feb 28 00:00:00.000000000 of same year via `Ds_Cal::to_time_point`
      - Construct `t_right` as Mar 1 00:00:00.000000000 of same year via `Ds_Cal::to_time_point`
      - Return `AliasedWindow{Time_Window{t_left, t_right}, 0.0}`
    - Otherwise: delegate to `resolve_clamp` (for out-of-bounds) or `resolve_in_bounds` (for in-bounds)
    - Use `if constexpr` with `std::is_same_v` to check calendar types at compile time
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 2.8 Implement public `resolve` method with policy dispatch
    - For `pure_climatology`: always call `resolve_climatology` (bypasses bounds check per design)
    - For other policies: check if `sim_time` is within `coverage_`; if yes, call `resolve_in_bounds`
    - If out-of-bounds: dispatch to `resolve_clamp`, `resolve_cycle`, or `resolve_leap_hold` based on `policy_`
    - All arithmetic uses `int64_t` nanoseconds; no floating-point intermediates except final alpha division
    - _Requirements: 7.3, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 10.4_

- [x] 3. Checkpoint
  - Ensure all headers compile cleanly (include the umbrella header in a test TU). Verify constructor validation throws correctly. Ask the user if questions arise.

- [x] 4. Extend test generators and write property-based tests
  - [x] 4.1 Extend `tests/generators.hpp` with aliasing-specific generators
    - Add `aliasing_coverage()`: generates a valid `Time_Window` suitable as dataset coverage (multi-year range within representable bounds)
    - Add `snapshot_interval_for(Time_Window)`: generates a `Duration` that evenly divides the given coverage duration
    - Add `time_point_in_coverage(Time_Window)`: generates a `Time_Point` strictly within `[start, end)`
    - Add `time_point_outside_coverage(Time_Window)`: generates a `Time_Point` before start or at/after end
    - Add `feb29_time_point()`: generates `Time_Point` values on February 29 of random Gregorian leap years (varying sub-day components)
    - Add `non_feb29_time_point()`: generates `Time_Point` values guaranteed not on February 29
    - All generators respect `int64_t` nanosecond representable bounds
    - _Requirements: 11.7_

  - [x] 4.2 Write property test: Weight Formula Correctness (Property 1)
    - **Property 1: Weight Formula Correctness**
    - Generate random `Time_Window` and random `Time_Point` within it; verify `calculate_weight` result equals manual formula computation
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 4.3 Write property test: Out-of-Range Rejection (Property 2)
    - **Property 2: Out-of-Range Rejection**
    - Generate random `Time_Window` and random `Time_Point` outside `[start, end)`; verify `calculate_weight` throws `std::out_of_range`
    - **Validates: Requirements 1.5**

  - [x] 4.4 Write property test: Weight-Bounds Invariant (Property 3)
    - **Property 3: Weight-Bounds Invariant**
    - Generate valid engine configs (all four policies) and random `Time_Point`; verify `0.0 <= alpha <= 1.0` on resolved result
    - **Validates: Requirements 7.3, 8.2, 11.1**

  - [x] 4.5 Write property test: Clamp-to-Edge Boundary Freeze (Property 4)
    - **Property 4: Clamp-to-Edge Boundary Freeze**
    - Generate engine with `clamp_to_edge` policy and random out-of-bounds `Time_Point`; verify `alpha == 0.0` and window equals first/last snapshot interval
    - **Validates: Requirements 3.1, 3.2**

  - [x] 4.6 Write property test: Cycle-Last-Year Target Year Containment (Property 5)
    - **Property 5: Cycle-Last-Year Target Year Containment**
    - Generate engine with `cycle_last_year` policy and random out-of-bounds `Time_Point`; verify both `t_left` and `t_right` decompose to target year via Dataset_Calendar
    - **Validates: Requirements 4.1, 4.3, 4.4, 11.3**

  - [x] 4.7 Write property test: Pure-Climatology Year-Lock (Property 6)
    - **Property 6: Pure-Climatology Year-Lock**
    - Generate engine with `pure_climatology` policy and arbitrary `Time_Point`; verify both window endpoints decompose to `climatological_year` or `climatological_year + 1`
    - **Validates: Requirements 5.1, 5.3, 5.5, 11.4**

  - [x] 4.8 Write property test: Leap-Hold Freeze (Property 7)
    - **Property 7: Leap-Hold Freeze**
    - Generate `Aliasing_Engine<Gregorian_Calendar, NoLeap_Calendar>` with `leap_hold` and random Feb 29 `Time_Point`; verify `alpha == 0.0` and `t_left` is Feb 28
    - **Validates: Requirements 6.1, 11.5**

  - [x] 4.9 Write property test: Leap-Hold Fallback Equivalence (Property 8)
    - **Property 8: Leap-Hold Fallback Equivalence**
    - Generate engine with `leap_hold` and random non-Feb-29 `Time_Point`; verify result matches what `clamp_to_edge` would produce
    - **Validates: Requirements 6.2**

  - [x] 4.10 Write property test: Weight Monotonicity (Property 9)
    - **Property 9: Weight Monotonicity**
    - Generate random `Time_Window` and two ordered `Time_Point` values A < B within it; verify `calculate_weight(A) <= calculate_weight(B)`
    - **Validates: Requirements 11.6**

  - [x] 4.11 Write property test: Resolve Determinism (Property 10)
    - **Property 10: Resolve Determinism**
    - Generate random engine and random `Time_Point`; call `resolve` twice; verify bitwise-identical `AliasedWindow` results via `operator==`
    - **Validates: Requirements 9.2, 11.2**

  - [x] 4.12 Write property test: Construction Rejects Non-Divisible Intervals (Property 11)
    - **Property 11: Construction Rejects Non-Divisible Intervals**
    - Generate random coverage and positive interval where `coverage.duration().nanos() % interval.nanos() != 0`; verify constructor throws `std::invalid_argument`
    - **Validates: Requirements 7.5**

- [x] 5. Write unit tests for edge cases and error conditions
  - [x] 5.1 Write unit tests for constructor validation
    - Test zero-duration interval → `std::invalid_argument`
    - Test negative interval → `std::invalid_argument`
    - Test non-divisible interval → `std::invalid_argument`
    - Test `pure_climatology` with out-of-range climatological year → `std::invalid_argument`
    - Test valid construction + accessor round-trip (coverage, interval, policy, climatological_year all match)
    - _Requirements: 7.5, 7.6, 7.7, 7.8_

  - [x] 5.2 Write unit tests for `calculate_weight` edge cases
    - Test `current == window.start()` → alpha exactly 0.0
    - Test `current` one nanosecond before `window.end()` → alpha < 1.0
    - Test `current` outside window → `std::out_of_range`
    - Test zero-duration window → `std::invalid_argument` (from `Time_Window` constructor)
    - _Requirements: 1.2, 1.3, 1.5, 1.6_

  - [x] 5.3 Write unit tests for policy-specific edge cases
    - Cycle-last-year: simulation date Feb 29 remapped to NoLeap → day clamped to Feb 28
    - Pure-climatology: December 31 near midnight → year-boundary wrap (t_right in climatological_year + 1)
    - Leap-hold: non-Gregorian/NoLeap calendar pair → behaves as clamp_to_edge
    - `AliasedWindow` equality: identical windows compare equal; different alpha bits compare unequal
    - _Requirements: 4.2, 5.3, 6.3, 8.3_

  - [x] 5.4 Write compile-time and thread-safety verification tests
    - `static_assert(std::is_same_v<std::underlying_type_t<OutOfBoundsPolicy>, std::uint8_t>)`
    - `static_assert` that `Aliasing_Engine` cannot be instantiated with a non-Calendar type (use `static_assert(!std::is_constructible_v<...>)` or SFINAE check)
    - Concurrent `resolve()` from multiple threads produces consistent results (launch N threads, compare all results)
    - _Requirements: 2.3, 7.4, 9.3_

- [x] 6. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (Properties 1–11)
- Unit tests validate specific examples and edge cases
- All library code is header-only; only `test_aliasing_engine.cpp` is a new `.cpp` file
- The existing `RC_GTEST_PROP` macro integrates RapidCheck with GTest; use it for all property tests
- Custom generators extend the existing `tests/generators.hpp` in the `tick::gen` namespace

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.4"] },
    { "id": 1, "tasks": ["1.3", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7"] },
    { "id": 4, "tasks": ["2.8"] },
    { "id": 5, "tasks": ["4.1", "5.1"] },
    { "id": 6, "tasks": ["4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "4.11", "4.12", "5.2", "5.3", "5.4"] }
  ]
}
```
