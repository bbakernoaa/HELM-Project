# Requirements Document

## Introduction

The TICK Fortran Bridge provides a C ABI layer (`tick_c.h` / `tick_c_api.cpp`) and an idiomatic Fortran 2003+ module (`tick_mod`) that expose the TICK micro-library's core functionality to legacy Fortran models in the HELM ecosystem. The bridge passes scalar `int64_t` values and small C-interoperable structs across the ABI boundary — no `std::mdspan` or data-array copying is involved. All C++ exceptions are caught at the bridge boundary and translated to integer return codes. The bridge lives inside `libs/tick/` and integrates with TICK's existing CMake build system as additional library targets.

## Glossary

- **Bridge**: The combination of the C header (`tick_c.h`), the C++ implementation (`tick_c_api.cpp`), and the Fortran module (`tick_mod`) that together expose TICK to Fortran callers.
- **C_API**: The flat `extern "C"` function layer declared in `tick_c.h` and defined in `tick_c_api.cpp`, callable from both C and Fortran.
- **tick_mod**: The Fortran module that wraps C_API functions with idiomatic Fortran interfaces using `iso_c_binding`.
- **Return_Code**: A 32-bit signed integer value returned by every C_API function to signal success (0) or a specific error category (nonzero).
- **tick_time_point_t**: A C-interoperable scalar type (`int64_t`) representing a TICK Time_Point (nanoseconds since epoch).
- **tick_duration_t**: A C-interoperable scalar type (`int64_t`) representing a TICK Duration (nanoseconds).
- **tick_date_time_t**: A C-interoperable struct containing seven `int32_t` fields (year, month, day, hour, minute, second, nanosecond) representing a decomposed calendar date-time.
- **tick_calendar_t**: A C-interoperable enum (integer) identifying one of the three calendar engines: Gregorian (0), NoLeap (1), or Cal360 (2).
- **Fortran_Caller**: Any Fortran 2003+ program or subroutine that invokes tick_mod procedures via `iso_c_binding`.
- **ABI_Boundary**: The `extern "C"` function interface where C++ types are translated to C-compatible scalars and structs, and where all C++ exceptions are caught.

## Requirements

### Requirement 1: C-Interoperable Type Definitions

**User Story:** As a Fortran model developer, I want TICK's temporal types represented as simple C-interoperable scalars and structs, so that I can pass them across the ABI boundary without needing C++ types or memory management.

#### Acceptance Criteria

1. THE C_API SHALL represent Time_Point values as a typedef `tick_time_point_t` aliasing `int64_t`.
2. THE C_API SHALL represent Duration values as a typedef `tick_duration_t` aliasing `int64_t`.
3. THE C_API SHALL represent Date_Time values as a struct `tick_date_time_t` containing exactly seven fields: `year` (int32_t), `month` (int32_t), `day` (int32_t), `hour` (int32_t), `minute` (int32_t), `second` (int32_t), and `nanosecond` (int32_t), in that order.
4. THE C_API SHALL represent the calendar selection as an enum `tick_calendar_t` with three named values: `TICK_CAL_GREGORIAN` (0), `TICK_CAL_NOLEAP` (1), and `TICK_CAL_360DAY` (2).
5. THE C_API header SHALL be compilable as both C99 (`gcc -std=c99`) and C++20 (`g++ -std=c++20`) without modification.
6. THE C_API header SHALL wrap all declarations in `extern "C"` guards (`#ifdef __cplusplus`) so that C++ compilers use C linkage.

### Requirement 2: Error Handling via Return Codes

**User Story:** As a Fortran model developer, I want all bridge functions to return integer error codes, so that I can detect failures without relying on C++ exception handling that Fortran cannot intercept.

#### Acceptance Criteria

1. THE C_API SHALL define a Return_Code type as a typedef `tick_status_t` aliasing `int32_t`.
2. THE C_API SHALL define the following named Return_Code constants: `TICK_OK` (0), `TICK_ERR_OVERFLOW` (1), `TICK_ERR_INVALID_ARG` (2), `TICK_ERR_INVALID_CALENDAR` (3), `TICK_ERR_INVALID_DATE` (4), and `TICK_ERR_INTERNAL` (5).
3. WHEN a C_API function completes without error, THE C_API SHALL return `TICK_OK` (0).
4. WHEN a C++ `std::overflow_error` is caught at the ABI_Boundary, THE C_API SHALL return `TICK_ERR_OVERFLOW`.
5. WHEN a C++ `std::invalid_argument` is caught at the ABI_Boundary, THE C_API SHALL return `TICK_ERR_INVALID_ARG`.
6. WHEN any other C++ exception is caught at the ABI_Boundary, THE C_API SHALL return `TICK_ERR_INTERNAL`.
7. THE C_API SHALL provide a function `tick_strerror` that accepts a `tick_status_t` and returns a pointer to a null-terminated constant string describing the error.
8. THE C_API SHALL NOT allow any C++ exception to propagate across the ABI_Boundary.

### Requirement 3: Time_Point Operations

**User Story:** As a Fortran model developer, I want to create Time_Point values and perform arithmetic on them, so that I can track simulation time from Fortran code.

#### Acceptance Criteria

1. THE C_API SHALL provide a function `tick_time_point_create` that accepts an `int64_t` nanosecond value and writes the corresponding `tick_time_point_t` to a caller-provided output pointer, returning `tick_status_t`.
2. THE C_API SHALL provide a function `tick_time_point_add_duration` that accepts a `tick_time_point_t` and a `tick_duration_t`, writes the resulting `tick_time_point_t` to a caller-provided output pointer, and returns `tick_status_t`.
3. THE C_API SHALL provide a function `tick_time_point_sub_duration` that accepts a `tick_time_point_t` and a `tick_duration_t`, writes the resulting `tick_time_point_t` to a caller-provided output pointer, and returns `tick_status_t`.
4. THE C_API SHALL provide a function `tick_time_point_diff` that accepts two `tick_time_point_t` values (lhs, rhs), writes the resulting `tick_duration_t` (lhs minus rhs) to a caller-provided output pointer, and returns `tick_status_t`.
5. THE C_API SHALL provide a function `tick_time_point_compare` that accepts two `tick_time_point_t` values and writes a comparison result to a caller-provided `int32_t` output pointer (negative if lhs < rhs, zero if equal, positive if lhs > rhs), returning `tick_status_t`.
6. IF an overflow occurs during Time_Point arithmetic, THEN THE C_API SHALL return `TICK_ERR_OVERFLOW` and leave the output pointer value unchanged.

### Requirement 4: Duration Operations

**User Story:** As a Fortran model developer, I want to create Duration values in various time units and perform arithmetic on them, so that I can express timesteps and intervals naturally in Fortran.

#### Acceptance Criteria

1. THE C_API SHALL provide factory functions for creating Durations from common units: `tick_duration_from_seconds`, `tick_duration_from_minutes`, `tick_duration_from_hours`, and `tick_duration_from_days`, each accepting an `int64_t` count and writing the `tick_duration_t` (in nanoseconds) to a caller-provided output pointer, returning `tick_status_t`.
2. THE C_API SHALL provide a function `tick_duration_from_nanos` that accepts an `int64_t` nanosecond count and writes the `tick_duration_t` to a caller-provided output pointer, returning `tick_status_t`.
3. THE C_API SHALL provide a function `tick_duration_add` that accepts two `tick_duration_t` values, writes their sum to a caller-provided output pointer, and returns `tick_status_t`.
4. THE C_API SHALL provide a function `tick_duration_sub` that accepts two `tick_duration_t` values, writes their difference (lhs minus rhs) to a caller-provided output pointer, and returns `tick_status_t`.
5. THE C_API SHALL provide a function `tick_duration_mul` that accepts a `tick_duration_t` and an `int64_t` scalar, writes their product to a caller-provided output pointer, and returns `tick_status_t`.
6. THE C_API SHALL provide a function `tick_duration_div` that accepts a `tick_duration_t` and an `int64_t` scalar, writes the truncated quotient to a caller-provided output pointer, and returns `tick_status_t`.
7. IF a Duration division scalar is zero, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG` and leave the output pointer value unchanged.
8. IF a Duration arithmetic operation would overflow int64_t, THEN THE C_API SHALL return `TICK_ERR_OVERFLOW` and leave the output pointer value unchanged.

### Requirement 5: Calendar Conversions

**User Story:** As a Fortran model developer, I want to convert between Time_Point and Date_Time representations using any of the three HELM calendar engines, so that I can produce human-readable dates and parse date inputs within Fortran.

#### Acceptance Criteria

1. THE C_API SHALL provide a function `tick_to_date_time` that accepts a `tick_time_point_t` and a `tick_calendar_t`, writes the resulting `tick_date_time_t` to a caller-provided output pointer, and returns `tick_status_t`.
2. THE C_API SHALL provide a function `tick_to_time_point` that accepts a `tick_date_time_t` and a `tick_calendar_t`, writes the resulting `tick_time_point_t` to a caller-provided output pointer, and returns `tick_status_t`.
3. THE C_API SHALL provide a function `tick_days_in_month` that accepts a `tick_calendar_t`, a year (`int32_t`), and a month (`int32_t`), writes the day count to a caller-provided `int32_t` output pointer, and returns `tick_status_t`.
4. THE C_API SHALL provide a function `tick_days_in_year` that accepts a `tick_calendar_t` and a year (`int32_t`), writes the day count to a caller-provided `int32_t` output pointer, and returns `tick_status_t`.
5. IF the calendar argument is not one of the three valid `tick_calendar_t` values, THEN THE C_API SHALL return `TICK_ERR_INVALID_CALENDAR`.
6. IF the `tick_date_time_t` contains field values that violate the specified calendar's rules (invalid month, day, hour, minute, second, or nanosecond), THEN THE C_API SHALL return `TICK_ERR_INVALID_DATE`.
7. FOR ALL valid `tick_date_time_t` values under a given calendar, converting to `tick_time_point_t` via `tick_to_time_point` and back via `tick_to_date_time` SHALL produce a `tick_date_time_t` equal to the original (round-trip property).

### Requirement 6: Alarm Queries

**User Story:** As a Fortran model developer, I want to query interval-based and absolute alarms from Fortran, so that I can trigger model events (output writes, coupling exchanges) at the correct simulation times.

#### Acceptance Criteria

1. THE C_API SHALL provide a function `tick_interval_alarm_is_ringing` that accepts an interval (`tick_duration_t`), a reference time (`tick_time_point_t`), and a current time (`tick_time_point_t`), writes a boolean result (int32_t: 1 for ringing, 0 for not ringing) to a caller-provided output pointer, and returns `tick_status_t`.
2. THE C_API SHALL provide a function `tick_interval_alarm_next_ring` that accepts an interval (`tick_duration_t`), a reference time (`tick_time_point_t`), and a current time (`tick_time_point_t`), writes the next ring `tick_time_point_t` to a caller-provided output pointer, and returns `tick_status_t`.
3. THE C_API SHALL provide a function `tick_absolute_alarm_is_ringing` that accepts a trigger time (`tick_time_point_t`) and a current time (`tick_time_point_t`), writes a boolean result (int32_t: 1 for ringing, 0 for not ringing) to a caller-provided output pointer, and returns `tick_status_t`.
4. IF the interval Duration is zero or negative, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG`.

### Requirement 7: Synchronization Functions

**User Story:** As a Fortran model developer, I want to compute heartbeat (GCD), sync period (LCM), and phase alignment from Fortran, so that I can coordinate multi-rate model coupling without implementing these algorithms in Fortran.

#### Acceptance Criteria

1. THE C_API SHALL provide a function `tick_compute_heartbeat` that accepts a pointer to an array of `tick_duration_t` values and a count (`int32_t`), writes the GCD Duration to a caller-provided output pointer, and returns `tick_status_t`.
2. THE C_API SHALL provide a function `tick_compute_sync_period` that accepts a pointer to an array of `tick_duration_t` values and a count (`int32_t`), writes the LCM Duration to a caller-provided output pointer, and returns `tick_status_t`.
3. THE C_API SHALL provide a function `tick_is_phase_aligned` that accepts a current time (`tick_time_point_t`), a base time (`tick_time_point_t`), and a timestep (`tick_duration_t`), writes a boolean result (int32_t: 1 for aligned, 0 for not aligned) to a caller-provided output pointer, and returns `tick_status_t`.
4. IF the timestep count is zero or the array pointer is null, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG`.
5. IF the timestep Duration for phase alignment is zero or negative, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG`.

### Requirement 8: Accumulation Window Operations

**User Story:** As a Fortran model developer, I want to query temporal window boundaries and containment from Fortran, so that I can manage accumulation buffers (precipitation, radiation fluxes) at correct intervals.

#### Acceptance Criteria

1. THE C_API SHALL provide a function `tick_is_on_boundary` that accepts a current time (`tick_time_point_t`) and an interval (`tick_duration_t`), writes a boolean result (int32_t: 1 for on boundary, 0 for not) to a caller-provided output pointer, and returns `tick_status_t`.
2. THE C_API SHALL provide a function `tick_compute_window` that accepts a current time (`tick_time_point_t`) and an interval (`tick_duration_t`), writes the window start and window end (`tick_time_point_t` each) to two caller-provided output pointers, and returns `tick_status_t`.
3. THE C_API SHALL provide a function `tick_window_contains` that accepts a window start (`tick_time_point_t`), a window end (`tick_time_point_t`), and a query time (`tick_time_point_t`), writes a boolean result (int32_t: 1 if contained in [start, end), 0 otherwise) to a caller-provided output pointer, and returns `tick_status_t`.
4. IF the interval Duration is zero or negative, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG`.
5. IF the window start is not less than the window end, THEN THE C_API SHALL return `TICK_ERR_INVALID_ARG`.

### Requirement 9: Fortran Module (tick_mod)

**User Story:** As a Fortran model developer, I want an idiomatic Fortran module that wraps the C_API functions, so that I can call TICK operations using Fortran conventions (intent attributes, named parameters, derived types) without writing `iso_c_binding` interface blocks myself.

#### Acceptance Criteria

1. THE tick_mod SHALL declare Fortran `interface` blocks for every C_API function using `iso_c_binding` with `bind(c)` attributes.
2. THE tick_mod SHALL expose public derived types `tick_time_point`, `tick_duration`, and `tick_date_time` that map to the corresponding C types using `iso_c_binding` kind parameters (`c_int64_t`, `c_int32_t`).
3. THE tick_mod SHALL expose named integer constants for calendar selection: `TICK_CAL_GREGORIAN`, `TICK_CAL_NOLEAP`, and `TICK_CAL_360DAY` matching the C enum values.
4. THE tick_mod SHALL expose named integer constants for all error codes: `TICK_OK`, `TICK_ERR_OVERFLOW`, `TICK_ERR_INVALID_ARG`, `TICK_ERR_INVALID_CALENDAR`, `TICK_ERR_INVALID_DATE`, and `TICK_ERR_INTERNAL`.
5. THE tick_mod SHALL declare all procedure arguments with explicit `intent(in)` or `intent(out)` attributes.
6. THE tick_mod SHALL be compilable with any Fortran 2003+ compiler that supports `iso_c_binding` (gfortran, ifort, ifx, nvfortran).
7. THE tick_mod SHALL NOT contain any implementation logic beyond the `iso_c_binding` interface declarations and type definitions.

### Requirement 10: CMake Build Integration

**User Story:** As a build engineer, I want the Fortran bridge integrated into TICK's existing CMake build system as optional targets, so that downstream consumers can link against the C API library and optionally the Fortran module without disrupting the existing C++ build.

#### Acceptance Criteria

1. THE CMake build SHALL produce a library target `tick_c` (static and/or shared) containing the C_API implementation, linked against the existing `tick` library.
2. THE CMake build SHALL produce the `tick_c` target whenever the TICK project is built, without requiring additional CMake options.
3. WHERE the CMake option `TICK_BUILD_FORTRAN` is enabled, THE CMake build SHALL compile the `tick_mod` Fortran module and produce a `tick_fortran` library target linked against `tick_c`.
4. THE `tick_c` target SHALL export the C_API header (`tick_c.h`) in its public include path.
5. THE `tick_fortran` target SHALL install the compiled `.mod` file to the standard Fortran module installation directory.
6. THE CMake build SHALL integrate with the existing install and export commands so that `find_package(TICK)` provides imported targets `HELM::TICK_C` and (when built) `HELM::TICK_Fortran`.

### Requirement 11: Tier 1 Isolation

**User Story:** As an architect, I want the Fortran bridge to maintain TICK's Tier 1 isolation, so that the bridge introduces no dependencies on other HELM libraries or domain science code.

#### Acceptance Criteria

1. THE C_API implementation SHALL include only TICK headers and C/C++ standard library headers.
2. THE C_API implementation SHALL NOT include headers from SPAN, DAGR, HALO, AXIS, LOGS, or AMIO.
3. THE tick_mod Fortran module SHALL depend only on `iso_c_binding` and the `tick_c` library.
4. THE `tick_c` library target SHALL link only against the `tick` C++ library and the C/C++ standard library.
