# Requirements Document

## Introduction

TICK (Time Integration & Chronology Kernel) is a Tier 1 C++20 micro-library within the HELM ecosystem. It provides a stateless, zero-dependency time and calendar management system that replaces the legacy ESMF Time Manager. TICK uses fixed-point 64-bit integer arithmetic to guarantee zero temporal drift over arbitrarily long simulations. It natively supports ESM-specific synthetic calendars (No-Leap 365-day and 360-day) alongside the standard Gregorian calendar. TICK provides alarm triggers, multi-rate heartbeat synchronization, and temporal accumulation window boundaries required by DAGR and SPAN. As a Tier 1 component, TICK is completely blind to other HELM libraries (LOGS, HALO, AXIS) and domain science code.

## Glossary

- **TICK**: Time Integration & Chronology Kernel; the Tier 1 stateless time and calendar manager within the HELM ecosystem.
- **Epoch**: A fixed reference point in time from which all time values are measured as integer offsets (e.g., 2026-01-01T00:00:00 UTC).
- **Time_Point**: An immutable value representing a specific instant in time, stored as a 64-bit signed integer counting discrete units (seconds or nanoseconds) from the Epoch.
- **Duration**: An immutable value representing a time interval, stored as a 64-bit signed integer counting discrete units (seconds or nanoseconds). All arithmetic on Durations is exact integer math.
- **Calendar**: A strategy object that defines the rules for converting between Time_Point values and human-readable date-time components (year, month, day, hour, minute, second).
- **Gregorian_Calendar**: A Calendar implementation following the proleptic Gregorian rules including leap year logic (divisible by 4, except centuries not divisible by 400).
- **NoLeap_Calendar**: A Calendar implementation where every year has exactly 365 days and February always has 28 days, with no leap year exceptions.
- **Day360_Calendar**: A Calendar implementation where every year has exactly 12 months of 30 days each (360 days per year).
- **Date_Time**: A decomposed representation of a point in time as year, month, day, hour, minute, second, and sub-second components, used for human-readable input and output.
- **Alarm**: A stateless trigger object that can be queried with a Time_Point to determine whether it is ringing (active) at that instant.
- **Interval_Alarm**: An Alarm that rings at regular periodic intervals defined by a Duration relative to a base Time_Point.
- **Absolute_Alarm**: An Alarm that rings at a single specific Time_Point.
- **TimeWindow**: A value representing a half-open temporal interval [start, end) defined by two Time_Points, used by SPAN to determine accumulation buffer boundaries.
- **Heartbeat**: The greatest common divisor (GCD) of all registered model timesteps, representing the fundamental clock tick of the coupled system.
- **Phase_Alignment**: The mathematical property that a given Time_Point is an exact integer multiple of a model's timestep Duration from its start time, indicating the model is synchronized with the coupled system.
- **LCM_Period**: The least common multiple of all registered model timesteps, representing the coupling cycle after which all models are simultaneously aligned.
- **Resolution**: The discrete unit size of a Time_Point or Duration, either seconds or nanoseconds, determining the precision of temporal arithmetic.
- **Round_Trip**: The property that converting a Date_Time to a Time_Point and back to a Date_Time produces the original value, proving calendar correctness.

## Requirements

### Requirement 1: Fixed-Point Time Representation

**User Story:** As a climate model developer, I want time represented as 64-bit integers with exact arithmetic, so that temporal drift is mathematically impossible over simulations spanning thousands of years.

#### Acceptance Criteria

1. THE Time_Point SHALL store its value as a single signed 64-bit integer (int64_t) representing a count of discrete units from the Epoch.
2. THE Duration SHALL store its value as a single signed 64-bit integer (int64_t) representing a count of discrete units.
3. THE TICK library SHALL NOT use floating-point types (float, double, long double) in any internal computation or storage of time values.
4. WHEN two Durations are added, THE Duration SHALL return a new Duration whose integer value equals the exact arithmetic sum of the two operands, leaving both original operands unmodified.
5. WHEN two Durations are subtracted, THE Duration SHALL return a new Duration whose integer value equals the exact arithmetic difference of the two operands, leaving both original operands unmodified.
6. WHEN a Duration is multiplied by an integer scalar, THE Duration SHALL return a new Duration whose integer value equals the exact arithmetic product of the Duration value and the scalar, leaving the original Duration unmodified.
7. WHEN a Duration is divided by an integer scalar, THE Duration SHALL return a new Duration whose integer value equals the truncated-toward-zero integer quotient of the Duration value and the scalar, leaving the original Duration unmodified.
8. WHEN a Duration is added to a Time_Point, THE Time_Point SHALL return a new Time_Point whose integer value equals the exact arithmetic sum of the original Time_Point value and the Duration value, leaving both original operands unmodified.
9. WHEN a Time_Point is subtracted from another Time_Point, THE operation SHALL return a Duration whose integer value equals the exact arithmetic difference of the two Time_Point values (left operand minus right operand), leaving both original operands unmodified.
10. IF an arithmetic operation on Time_Point or Duration would produce a result outside the int64_t range (less than -9,223,372,036,854,775,808 or greater than 9,223,372,036,854,775,807), THEN THE operation SHALL throw std::overflow_error indicating the operation and operand values.
11. IF a Duration is divided by an integer scalar equal to zero, THEN THE Duration SHALL throw std::invalid_argument indicating division by zero.
12. THE Time_Point and Duration SHALL each support equality (==, !=) and relational (<, <=, >, >=) comparison operators, where comparison is performed on the underlying int64_t values.
13. THE Time_Point and Duration SHALL be immutable value types: all arithmetic operations return new instances and no public member function modifies the internal int64_t value after construction.

### Requirement 2: Epoch and Resolution Configuration

**User Story:** As a model integrator, I want to specify the base epoch and temporal resolution at construction time, so that different simulations can use context-appropriate reference dates and precision levels.

#### Acceptance Criteria

1. WHEN a Time_Point is constructed from a Date_Time, a Calendar, a Resolution, and an Epoch Date_Time, THE Time_Point SHALL compute the integer offset from the supplied Epoch in the specified Resolution units using the supplied Calendar rules.
2. THE Resolution SHALL support exactly two values: seconds and nanoseconds.
3. IF Resolution is seconds, THEN THE Time_Point SHALL represent whole seconds from the Epoch, providing a representable range of at least plus or minus 292 billion years using a signed 64-bit integer.
4. IF Resolution is nanoseconds, THEN THE Time_Point SHALL represent nanoseconds from the Epoch, providing a representable range of at least plus or minus 292 years using a signed 64-bit integer.
5. THE TICK library SHALL define a default Epoch of 2026-01-01T00:00:00 that is used when no explicit Epoch is supplied at Time_Point construction.
6. IF two Time_Points with different Resolutions are used in any comparison (equality or relational) or arithmetic operation (addition of Duration, subtraction yielding Duration), THEN THE operation SHALL be ill-formed at compile time, prevented via the type system or static_assert.
7. THE Time_Point SHALL provide a constexpr accessor that returns the raw int64_t value for direct numerical comparison and serialization.
8. IF a Time_Point is constructed with an Epoch Date_Time that is invalid according to the supplied Calendar (e.g., February 30 under Gregorian_Calendar), THEN THE construction SHALL throw std::invalid_argument indicating the invalid Epoch date components.

### Requirement 3: Gregorian Calendar

**User Story:** As a weather model developer, I want a Gregorian calendar engine with correct leap year handling, so that my simulations use real-world date arithmetic for short-range forecasts.

#### Acceptance Criteria

1. WHEN the Gregorian_Calendar converts a Date_Time to a time offset, THE Gregorian_Calendar SHALL apply leap year rules where a year is a leap year if divisible by 4, except years divisible by 100 are not leap years, unless also divisible by 400, and SHALL return the offset in the configured Resolution units (seconds or nanoseconds) from the Epoch.
2. WHEN the Gregorian_Calendar converts a time offset back to a Date_Time, THE Gregorian_Calendar SHALL produce the correct year, month, day, hour, minute, second, and sub-second components consistent with the proleptic Gregorian calendar, including year 0 and negative years.
3. THE Gregorian_Calendar SHALL guarantee the round-trip property: converting any valid Date_Time to a Time_Point and converting that Time_Point back to a Date_Time SHALL produce a Date_Time with identical year, month, day, hour, minute, second, and sub-second component values.
4. THE Gregorian_Calendar SHALL assign February exactly 29 days in leap years and exactly 28 days in non-leap years.
5. THE Gregorian_Calendar SHALL assign the correct day counts to each month: January 31, March 31, April 30, May 31, June 30, July 31, August 31, September 30, October 31, November 30, December 31.
6. IF a Date_Time contains any invalid component value (month outside 1–12, day outside 1–N where N is the number of days in the given month and year, hour outside 0–23, minute outside 0–59, or second outside 0–59), THEN THE Gregorian_Calendar SHALL throw std::invalid_argument indicating which component is out of range and the valid bounds for that component.
7. WHEN the Gregorian_Calendar computes the number of days between two Date_Time values spanning century boundaries, THE Gregorian_Calendar SHALL correctly account for the century leap year exception (e.g., 1900 is not a leap year, 2000 is a leap year).
8. WHEN the Resolution is nanoseconds and a Date_Time contains a sub-second component, THE Gregorian_Calendar SHALL preserve the sub-second value during conversion to and from a Time_Point with nanosecond precision.

### Requirement 4: No-Leap 365-Day Calendar

**User Story:** As a climate scientist, I want a 365-day calendar with no leap years, so that every simulated year has identical length for consistent solar radiation cycle calculations.

#### Acceptance Criteria

1. THE NoLeap_Calendar SHALL define every year as exactly 365 days with no exceptions.
2. THE NoLeap_Calendar SHALL assign February exactly 28 days in every year regardless of divisibility rules.
3. THE NoLeap_Calendar SHALL assign the same day counts to all other months as the Gregorian_Calendar (January 31, March 31, April 30, May 31, June 30, July 31, August 31, September 30, October 31, November 30, December 31).
4. FOR ALL valid Date_Time values within the representable range, converting a Date_Time to a Time_Point using NoLeap_Calendar and converting back to a Date_Time SHALL produce the original Date_Time value (round-trip property), where a valid Date_Time is one with month in 1–12, day in 1–N where N is the day count for that month per criteria 2 and 3, hour in 0–23, minute in 0–59, and second in 0–59.
5. IF a Date_Time contains February 29 under the NoLeap_Calendar, THEN THE NoLeap_Calendar SHALL throw std::invalid_argument indicating that February 29 is invalid in the NoLeap calendar.
6. IF a Date_Time contains an invalid component value under the NoLeap_Calendar (month outside 1–12, day outside 1–N where N is the defined day count for that month, hour outside 0–23, minute outside 0–59, or second outside 0–59), THEN THE NoLeap_Calendar SHALL throw std::invalid_argument indicating the invalid component and its allowed range.
7. WHEN the NoLeap_Calendar computes the time offset between January 1 of year Y and January 1 of year Y+1, THE NoLeap_Calendar SHALL return exactly 365 days in the configured Resolution units (31,536,000 for seconds Resolution or 31,536,000,000,000,000 for nanoseconds Resolution) for any year Y within the representable range.

### Requirement 5: 360-Day Calendar

**User Story:** As a climate scientist, I want a 360-day calendar where every month has exactly 30 days, so that long-term statistical climate balancing is simplified by uniform monthly periods.

#### Acceptance Criteria

1. THE Day360_Calendar SHALL define every year as exactly 360 days.
2. THE Day360_Calendar SHALL define every month as exactly 30 days regardless of month number.
3. THE Day360_Calendar SHALL define every year as having exactly 12 months.
4. FOR ALL valid Date_Time values within the representable range, converting a Date_Time to a Time_Point using Day360_Calendar and converting back to a Date_Time SHALL produce the original Date_Time value (round-trip property).
5. IF a Date_Time contains a day value less than 1 or greater than 30 under the Day360_Calendar, THEN THE Day360_Calendar SHALL throw std::invalid_argument indicating the invalid day value and the valid range of 1 to 30.
6. IF a Date_Time contains a month value less than 1 or greater than 12 under the Day360_Calendar, THEN THE Day360_Calendar SHALL throw std::invalid_argument indicating the invalid month value and the valid range of 1 to 12.
7. IF a Date_Time contains an hour value greater than 23, a minute value greater than 59, or a second value greater than 59 under the Day360_Calendar, THEN THE Day360_Calendar SHALL throw std::invalid_argument indicating which time component is out of range.
8. WHEN the Day360_Calendar computes the time offset between January 1 of year Y and January 1 of year Y+1, THE Day360_Calendar SHALL return exactly 360 days in the configured Resolution units for any year Y.
9. WHEN the Day360_Calendar computes the time offset between the first day of month M and the first day of month M+1, THE Day360_Calendar SHALL return exactly 30 days in the configured Resolution units for any month M from 1 through 11.

### Requirement 6: Interval Alarm

**User Story:** As the DAGR orchestrator, I want periodic alarms that fire at exact fixed intervals, so that I can schedule recurring model operations (regridding, output writes) with zero drift.

#### Acceptance Criteria

1. WHEN an Interval_Alarm is constructed with a base Time_Point and an interval Duration, THE Interval_Alarm SHALL store both values as immutable state.
2. WHEN Interval_Alarm::is_ringing is called with a current Time_Point, THE Interval_Alarm SHALL return true if and only if the difference between the current Time_Point and the base Time_Point is an exact non-negative integer multiple of the interval Duration, without modifying any internal state.
3. THE Interval_Alarm::is_ringing operation SHALL execute in O(1) time complexity using only integer modulo arithmetic.
4. IF an Interval_Alarm is constructed with an interval Duration of zero, THEN THE Interval_Alarm SHALL throw std::invalid_argument indicating that the interval must be positive.
5. IF an Interval_Alarm is constructed with a negative interval Duration, THEN THE Interval_Alarm SHALL throw std::invalid_argument indicating that the interval must be positive.
6. WHEN Interval_Alarm::is_ringing is called with a Time_Point that precedes the base Time_Point, THE Interval_Alarm SHALL return false.
7. THE Interval_Alarm SHALL be copyable and movable to allow storage in containers.
8. THE Interval_Alarm SHALL provide const accessors that return the base Time_Point and the interval Duration supplied at construction.

### Requirement 7: Absolute Alarm

**User Story:** As the DAGR orchestrator, I want single-fire alarms that trigger at a precise timestamp, so that I can schedule one-time events like simulation checkpoints and termination.

#### Acceptance Criteria

1. WHEN an Absolute_Alarm is constructed with a target Time_Point, THE Absolute_Alarm SHALL store that value as immutable state.
2. WHEN Absolute_Alarm::is_ringing is called with a current Time_Point, THE Absolute_Alarm SHALL return true if and only if the current Time_Point is exactly equal to the target Time_Point.
3. THE Absolute_Alarm::is_ringing operation SHALL execute in O(1) time complexity using only integer comparison.
4. THE Absolute_Alarm SHALL be copyable and movable, and a copy or move of an Absolute_Alarm SHALL preserve the target Time_Point such that the copy produces identical is_ringing results for any queried Time_Point.
5. WHEN Absolute_Alarm::is_ringing is called with any Time_Point not equal to the target Time_Point, THE Absolute_Alarm SHALL return false.
6. WHEN Absolute_Alarm::target is called, THE Absolute_Alarm SHALL return the target Time_Point provided at construction time.

### Requirement 8: Multi-Rate Heartbeat Synchronization

**User Story:** As the DAGR orchestrator, I want to compute the GCD heartbeat and LCM coupling period from multiple model timesteps, so that I can determine the fundamental scheduling tick and full coupling cycle for the coupled system.

#### Acceptance Criteria

1. WHEN compute_heartbeat is called with a collection of Duration values representing model timesteps, THE TICK library SHALL return a Duration equal to the Greatest Common Divisor (GCD) of all provided Duration nanosecond values using exact integer GCD arithmetic.
2. WHEN compute_sync_period is called with a collection of Duration values representing model timesteps, THE TICK library SHALL return a Duration equal to the Least Common Multiple (LCM) of all provided Duration nanosecond values using exact integer arithmetic.
3. IF compute_sync_period detects that the LCM computation would overflow the signed 64-bit integer range, THEN THE TICK library SHALL throw std::overflow_error indicating that the combination of timesteps produces an unrepresentable coupling period.
4. IF compute_heartbeat or compute_sync_period is called with an empty collection, THEN THE TICK library SHALL throw std::invalid_argument indicating that at least one timestep is required.
5. IF any provided timestep Duration is zero or negative, THEN THE TICK library SHALL throw std::invalid_argument indicating that all timesteps must be positive.
6. FOR ALL valid collections of positive Duration values, the Heartbeat returned by compute_heartbeat SHALL evenly divide each Duration in the collection with zero remainder.
7. FOR ALL valid collections of positive Duration values, each Duration in the collection SHALL evenly divide the LCM_Period returned by compute_sync_period with zero remainder.
8. WHEN compute_heartbeat is called with a collection containing a single Duration, THE TICK library SHALL return that Duration unchanged.
9. WHEN compute_sync_period is called with a collection containing a single Duration, THE TICK library SHALL return that Duration unchanged.

### Requirement 9: Phase Alignment Verification

**User Story:** As the DAGR orchestrator, I want to verify that a model is phase-aligned at a given time, so that I can deadlock-protect the coupling pipeline by preventing out-of-phase execution.

#### Acceptance Criteria

1. WHEN is_phase_aligned is called with a Time_Point current_time, a Time_Point base_time, and a Duration time_step, THE TICK library SHALL return true if and only if the difference (current_time minus base_time) is non-negative and evenly divisible by the time_step nanosecond value (remainder equals zero).
2. THE is_phase_aligned operation SHALL execute in O(1) time complexity using only integer modulo arithmetic.
3. IF is_phase_aligned is called with a time_step of zero nanoseconds or a negative time_step, THEN THE TICK library SHALL throw std::invalid_argument indicating that the time step must be positive.
4. WHEN is_phase_aligned is called with current_time before base_time, THE TICK library SHALL return false.
5. WHEN is_phase_aligned is called with current_time equal to base_time, THE TICK library SHALL return true for any positive time_step.

### Requirement 10: Temporal Accumulation Windows

**User Story:** As the SPAN coupler, I want to query TICK for explicit window boundaries, so that I know precisely when to seal accumulation buffers and present data views to the interpolation engine.

#### Acceptance Criteria

1. THE Time_Window SHALL store a start Time_Point (inclusive) and an end Time_Point (exclusive) as immutable state, representing the half-open interval [start, end).
2. WHEN a Time_Window is constructed with a start Time_Point and a Duration, THE Time_Window SHALL compute end as start plus Duration.
3. IF a Time_Window is constructed with a Duration of zero or negative, or with a start Time_Point that is greater than or equal to the end Time_Point, THEN THE Time_Window SHALL throw std::invalid_argument indicating that the window must have a positive duration with start strictly less than end.
4. WHEN Time_Window::contains is called with a Time_Point, THE Time_Window SHALL return true if and only if the Time_Point is greater than or equal to start and strictly less than end.
5. WHEN is_on_boundary is called with a current Time_Point, a window base Time_Point, and a window Duration, THE TICK library SHALL return true if and only if the difference between the current time and the window base is non-negative and evenly divisible by the window Duration nanosecond value (remainder equals zero).
6. IF is_on_boundary is called with a window Duration of zero or negative nanoseconds, THEN THE TICK library SHALL throw std::invalid_argument indicating that the window Duration must be positive.
7. THE is_on_boundary operation SHALL execute in O(1) time complexity using only integer modulo arithmetic.
8. THE Time_Window SHALL be copyable and movable to allow storage in containers.
9. WHEN Time_Window::duration is called, THE Time_Window SHALL return a Duration equal to end minus start.
10. WHEN compute_window is called with a Time_Point current_time, a Time_Point base, and a Duration interval, THE TICK library SHALL return the Time_Window [floor_time, floor_time + interval) where floor_time is the largest Time_Point less than or equal to current_time such that the difference (floor_time minus base) is a non-negative integer multiple of the interval.

### Requirement 11: Calendar Correctness Property-Based Testing

**User Story:** As a quality engineer, I want property-based tests that mathematically prove calendar correctness, so that edge cases in leap year logic, century boundaries, and synthetic calendar arithmetic are exhaustively verified.

#### Acceptance Criteria

1. THE test suite SHALL contain a property-based test proving the round-trip property for Gregorian_Calendar: for all valid Date_Time values generated with year in the range Epoch minus 1000 years to Epoch plus 1000 years, converting to Time_Point and back produces the original Date_Time.
2. THE test suite SHALL contain a property-based test proving the round-trip property for NoLeap_Calendar: for all valid Date_Time values generated with year in the range Epoch minus 1000 years to Epoch plus 1000 years, converting to Time_Point and back produces the original Date_Time.
3. THE test suite SHALL contain a property-based test proving the round-trip property for Day360_Calendar: for all valid Date_Time values generated with year in the range Epoch minus 1000 years to Epoch plus 1000 years, converting to Time_Point and back produces the original Date_Time.
4. THE test suite SHALL contain a property-based test proving that Duration addition is commutative: for all Duration pairs (A, B) where neither operand nor their sum overflows the signed 64-bit integer range, A + B equals B + A.
5. THE test suite SHALL contain a property-based test proving that Duration addition is associative: for all Duration triples (A, B, C) where no intermediate or final sum overflows the signed 64-bit integer range, (A + B) + C equals A + (B + C).
6. THE test suite SHALL contain a property-based test proving zero-drift accumulation: for all positive Duration D and positive integer N where N is in the range 1 to 10000 and the product D multiplied by N does not overflow a signed 64-bit integer, adding D to a Time_Point exactly N times produces the same result as adding (D * N) once.
7. THE test suite SHALL contain a property-based test proving that the Heartbeat (GCD) divides every input timestep: for all sets of 2 to 8 positive Durations each no greater than 86400000000000 nanoseconds (one day), each Duration modulo the computed GCD equals zero.
8. THE test suite SHALL contain a property-based test proving that the LCM_Period is divisible by every input timestep: for all sets of 2 to 8 positive Durations each no greater than 86400000000000 nanoseconds (one day) where the computed LCM does not overflow a signed 64-bit integer, the computed LCM modulo each Duration equals zero.
9. THE test suite SHALL contain a property-based test proving interval alarm consistency: for all base Time_Points, positive interval Durations, and non-negative integer multipliers K where K is in the range 0 to 10000 and (interval * K) does not overflow a signed 64-bit integer, is_ringing returns true at base + (interval * K).
10. THE test suite SHALL use a C++ property-based testing library (such as RapidCheck or a custom generator framework) integrated with Google Test.
11. THE test suite SHALL execute a minimum of 1000 generated input cases per property-based test to provide sufficient coverage of the input space.
12. THE test suite property-based generators SHALL constrain inputs using precondition filters or assume-guards to exclude values that would cause signed 64-bit integer overflow, ensuring that properties are verified only over the valid domain of each operation.

### Requirement 12: CMake Build System

**User Story:** As a build engineer, I want TICK to provide a standalone CMake build system with no external dependencies beyond the C++20 standard library, so that the library can be built and tested in isolation.

#### Acceptance Criteria

1. THE TICK build system SHALL use CMake version 3.21 or later, require the C++20 standard via target_compile_features (cxx_std_20), and disable compiler extensions.
2. THE TICK build system SHALL produce a static library target named tick with a namespace alias HELM::TICK, using target_include_directories and target_compile_features rather than global commands (include_directories, add_definitions).
3. THE TICK build system SHALL NOT require any external dependencies beyond the C++20 standard library for building the library target.
4. THE TICK build system SHALL provide a BUILD_TESTING option defaulting to OFF.
5. IF BUILD_TESTING is set to ON, THEN THE TICK build system SHALL locate Google Test using find_package(GTest REQUIRED) and build the test suite.
6. THE TICK build system SHALL export CMake configuration files (TICKConfig.cmake, TICKConfigVersion.cmake, and TICKTargets.cmake) so downstream projects can consume TICK via find_package(TICK), using SameMajorVersion compatibility for version matching.
7. THE TICK build system SHALL define a cache option TICK_RESOLUTION with allowed values SECONDS and NANOSECONDS, defaulting to NANOSECONDS, that passes a compile definition TICK_RESOLUTION_SECONDS or TICK_RESOLUTION_NANOSECONDS to the tick target and propagates it to consuming targets via PUBLIC compile definitions.
8. IF BUILD_TESTING is ON, THEN THE TICK build system SHALL also locate RapidCheck using find_package(rapidcheck REQUIRED) and link it to property-based test targets.
9. THE TICK build system SHALL install public headers to an include/tick subdirectory and the library artifact to a lib subdirectory relative to CMAKE_INSTALL_PREFIX.

### Requirement 13: Repository and Container Structure

**User Story:** As a build engineer, I want TICK to reside in a dedicated Git submodule within the HELM project, so that Tier 1 libraries maintain independent version histories and CI pipelines.

#### Acceptance Criteria

1. THE TICK source code SHALL reside in a dedicated Git repository that is added as a Git submodule at the path `libs/tick` within the HELM project workspace root.
2. THE TICK build and test workflow SHALL use the Docker image built from the HELM project Dockerfile as its development and CI container environment, launched via the `helm-dev` service defined in the HELM project `docker-compose.yml`.
3. WHEN the TICK repository CMakeLists.txt is configured and built inside the helm-project Docker container with only the TICK source tree present and BUILD_TESTING set to ON, THE build system SHALL produce the HELM::TICK library target and all test executables without errors using only the compilers and libraries pre-installed in the container (GCC-13, CMake, Google Test).
4. THE TICK repository SHALL include a README.md at its root containing at minimum: a prerequisites section listing the Docker container requirement, the exact `docker compose` command to launch the development container, the exact CMake configure and build commands to compile the library, and the exact CTest or test-runner command to execute the test suite.
5. WHEN the HELM project workspace is cloned with `--recurse-submodules`, THE TICK submodule SHALL be checked out at a pinned commit recorded in the HELM project `.gitmodules` and Git index so that the HELM project build is reproducible without additional manual steps.
6. THE TICK repository SHALL contain a GitHub Actions CI workflow that, on each push or pull request to the TICK repository, builds the library and runs the full test suite inside the helm-project Docker container, reporting pass or fail status.

### Requirement 14: Tier 1 Isolation Compliance

**User Story:** As an architect, I want TICK to have zero compile-time dependencies on other HELM Tier 1 libraries, so that the no-circular-dependency invariant of the HELM architecture is preserved.

#### Acceptance Criteria

1. THE TICK library SHALL NOT include any header files from HALO, LOGS, AXIS, AMIO, SPAN, or DAGR — identified by any `#include` directive whose path contains `halo/`, `logs/`, `axis/`, `amio/`, `span/`, or `dagr/` (case-insensitive) — in any source file, public header, or internal header.
2. THE TICK library SHALL NOT link against any other HELM library target (HELM::HALO, HELM::LOGS, HELM::AXIS, HELM::AMIO, HELM::SPAN, or HELM::DAGR) at build time.
3. THE TICK library public headers and internal headers SHALL only contain `#include` directives referencing C++ standard library headers (those defined by the ISO C++20 standard in namespace `std`).
4. THE TICK CMakeLists.txt SHALL NOT reference any HELM:: namespace targets other than HELM::TICK in its target_link_libraries, add_dependencies, or find_package directives.
5. WHEN the TICK CI pipeline runs, THE build system SHALL execute a static verification step that scans all TICK source and header files for `#include` directives whose paths match any other HELM component name pattern (`halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`) and fails the build with a non-zero exit code if any matching directive is found.

### Requirement 15: Date-Time Formatting and Parsing

**User Story:** As a model integrator, I want to convert between ISO 8601 string representations and Date_Time values, so that simulation configuration files and log output use human-readable timestamps.

#### Acceptance Criteria

1. WHEN a valid ISO 8601 date-time string in the format "YYYY-MM-DDThh:mm:ss" is provided, THE TICK parser SHALL produce a Date_Time value with the corresponding year, month, day, hour, minute, and second components, with the nanosecond field set to zero.
2. WHEN a Date_Time value is formatted, THE TICK formatter SHALL produce an ISO 8601 string in the format "YYYY-MM-DDThh:mm:ss" with zero-padded fields, discarding the nanosecond component.
3. FOR ALL valid Date_Time values whose nanosecond field is zero and whose year is in the range 0000-9999, formatting a Date_Time to a string and parsing it back SHALL produce the original Date_Time value (round-trip property).
4. IF a string does not conform to the "YYYY-MM-DDThh:mm:ss" format or contains trailing characters beyond the 19-character fixed-length representation, THEN THE TICK parser SHALL throw std::invalid_argument indicating the expected format and the zero-based byte offset of the first invalid character.
5. IF a parsed string contains valid format but invalid date-time component values (e.g., month 13 or hour 25), THEN THE TICK parser SHALL throw std::invalid_argument indicating which component is out of range.
6. THE TICK formatter SHALL NOT allocate heap memory for strings of 20 characters or fewer (covering all valid ISO 8601 date-time representations within the supported range).
7. THE TICK parser SHALL operate on the numeric string fields only and SHALL NOT require a Calendar parameter; calendar-specific validation of the resulting Date_Time component values is the responsibility of the Calendar engine.
