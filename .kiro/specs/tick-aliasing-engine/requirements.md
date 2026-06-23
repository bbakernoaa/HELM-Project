# Requirements Document

## Introduction

The Time Aliasing Engine extends the TICK micro-library to solve a fundamental problem in Earth System Models: translating unbounded simulation time into the bounded temporal range of forcing datasets. Forcing fields (e.g., Sea Surface Temperature) are typically available only for a finite historical period (e.g., 1990–2020). When a simulation's clock exceeds this range, a deterministic policy must map the out-of-bounds simulation time back into the dataset's valid temporal domain and compute an interpolation weight between two bounding snapshots. The Aliasing Engine is stateless, performs all internal arithmetic in integer nanoseconds, and outputs a single floating-point alpha weight as the sole non-integer value in the entire TICK library.

## Glossary

- **Aliasing_Engine**: A stateless, immutable value type that resolves an arbitrary simulation Time_Point into an AliasedWindow within a dataset's temporal coverage by applying a configured OutOfBoundsPolicy.
- **AliasedWindow**: A result struct containing a Time_Window (two bounding dataset Time_Points) and a floating-point interpolation weight (alpha) in the range [0.0, 1.0].
- **OutOfBoundsPolicy**: An enumeration defining the strategy used to remap simulation time when it falls outside the dataset's temporal coverage. Valid values are clamp_to_edge, cycle_last_year, pure_climatology, and leap_hold.
- **Dataset_Coverage**: The half-open temporal interval [start, end) representing the range of time instants for which a forcing dataset has valid snapshots.
- **Interpolation_Weight**: A double-precision floating-point value (alpha) in the closed interval [0.0, 1.0] representing the linear blend between the left and right bounding snapshots of a Time_Window.
- **Simulation_Calendar**: The Calendar type used by the running simulation (e.g., Gregorian_Calendar).
- **Dataset_Calendar**: The Calendar type used by the forcing dataset (e.g., NoLeap_Calendar).
- **Climatological_Year**: A fixed reference year (supplied at construction) used by the pure_climatology policy to anchor all date remapping.
- **Year_Boundary_Wrap**: The condition where an interpolation window spans a December-to-January transition, requiring modular year arithmetic to correctly identify the two bounding Time_Points.
- **Leap_Hold**: The behaviour of freezing interpolation state on February 29 when the simulation calendar is Gregorian but the dataset calendar is NoLeap (365-day), since the dataset has no February 29 snapshot.

## Requirements

### Requirement 1: Interpolation Weight Calculation

**User Story:** As a forcing data consumer, I want to compute a precise linear interpolation weight between two dataset snapshots, so that field values can be blended accurately at arbitrary sub-interval simulation times.

#### Acceptance Criteria

1. WHEN calculate_weight is called with a Time_Point current and a Time_Window window where current is within [window.start(), window.end()), THE Aliasing_Engine SHALL return a double-precision alpha equal to the integer nanosecond difference (current minus window.start()) cast to double, divided by the integer nanosecond difference (window.end() minus window.start()) cast to double.
2. WHEN calculate_weight is called with current equal to window.start(), THE Aliasing_Engine SHALL return alpha equal to 0.0.
3. WHEN calculate_weight is called with current approaching window.end() (one nanosecond before), THE Aliasing_Engine SHALL return an alpha strictly less than 1.0.
4. THE calculate_weight function SHALL perform all time difference computations using int64_t nanosecond arithmetic and SHALL produce exactly one floating-point division as the final operation.
5. IF calculate_weight is called with a current Time_Point outside the half-open interval [window.start(), window.end()), THEN THE Aliasing_Engine SHALL throw std::out_of_range indicating that the current time is not within the supplied window.
6. IF calculate_weight is called with a Time_Window whose duration is zero nanoseconds, THEN THE Aliasing_Engine SHALL throw std::invalid_argument indicating a degenerate window.

### Requirement 2: OutOfBoundsPolicy Enumeration

**User Story:** As a model integrator, I want to select from multiple remapping strategies when simulation time exceeds dataset coverage, so that different scientific use cases (clamping, cycling, climatology) are supported without code changes.

#### Acceptance Criteria

1. THE OutOfBoundsPolicy SHALL be a scoped enum (enum class) with exactly four enumerators: clamp_to_edge, cycle_last_year, pure_climatology, and leap_hold.
2. THE OutOfBoundsPolicy SHALL be defined in the tick namespace within a dedicated header file.
3. THE OutOfBoundsPolicy enumerators SHALL have an underlying type of std::uint8_t.

### Requirement 3: Clamp-to-Edge Policy

**User Story:** As a model integrator, I want simulation time clamped to the dataset boundary when it exceeds coverage, so that the last (or first) available forcing snapshot is held constant rather than extrapolating into unknown data.

#### Acceptance Criteria

1. WHILE the OutOfBoundsPolicy is clamp_to_edge and the simulation Time_Point exceeds or equals the Dataset_Coverage end, THE Aliasing_Engine SHALL resolve to an AliasedWindow whose Time_Window has t_left equal to the last dataset snapshot (coverage end minus one snapshot interval) and t_right equal to coverage end, with alpha equal to 0.0 (freezing at the last snapshot).
2. WHILE the OutOfBoundsPolicy is clamp_to_edge and the simulation Time_Point precedes the Dataset_Coverage start, THE Aliasing_Engine SHALL resolve to an AliasedWindow whose Time_Window contains the first two dataset snapshots (coverage start as t_left, coverage start plus one snapshot interval as t_right) and whose alpha equals 0.0.
3. WHEN the simulation Time_Point is within Dataset_Coverage bounds under clamp_to_edge policy, THE Aliasing_Engine SHALL compute the enclosing Time_Window from the dataset snapshot interval and return the standard interpolation weight.

### Requirement 4: Cycle-Last-Year Policy

**User Story:** As a climate scientist, I want out-of-bounds simulation time to repeat the last year of dataset coverage cyclically, so that boundary forcing remains physically plausible without discontinuities at the dataset edge.

#### Acceptance Criteria

1. WHEN the simulation Time_Point exceeds Dataset_Coverage under cycle_last_year policy, THE Aliasing_Engine SHALL decompose the simulation Time_Point into Date_Time components using the Simulation_Calendar, substitute the year component with the last complete year of Dataset_Coverage, preserve month, day, hour, minute, second, and nanosecond components, and convert the result back to a Time_Point using the Dataset_Calendar.
2. IF the substituted date is invalid in the Dataset_Calendar (e.g., February 29 in a NoLeap dataset year), THEN THE Aliasing_Engine SHALL clamp the day to the last valid day of that month in the Dataset_Calendar (e.g., February 28).
3. WHEN the remapped Time_Point under cycle_last_year falls within the last year of dataset coverage, THE Aliasing_Engine SHALL compute the enclosing Time_Window from the dataset snapshot interval and return the standard interpolation weight.
4. WHEN the simulation Time_Point precedes Dataset_Coverage start under cycle_last_year policy, THE Aliasing_Engine SHALL apply the same year-substitution logic using the first complete year of Dataset_Coverage.

### Requirement 5: Pure Climatology Policy

**User Story:** As a climate scientist, I want all simulation time locked to a fixed climatological year, so that perpetual-year boundary forcing experiments use a single repeating annual cycle regardless of simulation date.

#### Acceptance Criteria

1. WHEN resolve is called under pure_climatology policy, THE Aliasing_Engine SHALL decompose the simulation Time_Point into Date_Time components using the Simulation_Calendar, substitute the year component with the configured Climatological_Year, preserve month, day, hour, minute, second, and nanosecond components, and convert the result back to a Time_Point using the Dataset_Calendar.
2. IF the substituted date is invalid in the Dataset_Calendar (e.g., February 29 in a NoLeap dataset with a climatological year), THEN THE Aliasing_Engine SHALL clamp the day to the last valid day of that month in the Dataset_Calendar.
3. WHEN the remapped Time_Point under pure_climatology produces a Time_Window that spans a December-to-January year boundary (t_left in December and t_right in January), THE Aliasing_Engine SHALL construct t_left using the Climatological_Year and t_right using the Climatological_Year plus one, then compute the interpolation weight across the year boundary using the integer nanosecond delta between the two Time_Points.
4. WHEN the remapped Time_Point under pure_climatology produces a Time_Window entirely within a single month, THE Aliasing_Engine SHALL compute the enclosing Time_Window from the dataset snapshot interval and return the standard interpolation weight.
5. THE pure_climatology policy SHALL apply to all simulation Time_Points regardless of whether the simulation time is within or outside Dataset_Coverage bounds.

### Requirement 6: Leap-Hold Policy

**User Story:** As a model integrator, I want February 29 simulation dates handled gracefully when the dataset uses a NoLeap calendar, so that the simulation does not crash or produce undefined interpolation on leap day.

#### Acceptance Criteria

1. WHEN the Simulation_Calendar is Gregorian_Calendar and the Dataset_Calendar is NoLeap_Calendar and the simulation Time_Point falls on February 29 (any hour, minute, second, nanosecond) under leap_hold policy, THE Aliasing_Engine SHALL resolve to an AliasedWindow whose Time_Window has t_left equal to February 28 00:00:00.000000000 of the same year (in Dataset_Calendar) and t_right equal to March 1 00:00:00.000000000 of the same year (in Dataset_Calendar), with alpha equal to 0.0.
2. WHEN the simulation Time_Point does not fall on February 29, THE leap_hold policy SHALL behave identically to clamp_to_edge policy for out-of-bounds times and standard interpolation for in-bounds times.
3. THE leap_hold policy SHALL only activate its February 29 hold behaviour when the Simulation_Calendar is Gregorian_Calendar and the Dataset_Calendar is NoLeap_Calendar; for all other calendar combinations, leap_hold SHALL behave identically to clamp_to_edge.

### Requirement 7: Aliasing Engine Interface

**User Story:** As a forcing data consumer, I want a single entry-point class that encapsulates dataset coverage, calendar types, snapshot interval, and out-of-bounds policy, so that resolving aliased time is a single function call with no additional context.

#### Acceptance Criteria

1. THE Aliasing_Engine SHALL be constructed with the following immutable parameters: a Dataset_Coverage Time_Window (start and end Time_Points), a Duration representing the dataset snapshot interval, an OutOfBoundsPolicy, a Simulation_Calendar type, a Dataset_Calendar type, and (for pure_climatology policy) a Climatological_Year integer.
2. THE Aliasing_Engine SHALL be an immutable value type with no mutable internal state; all member functions SHALL be const-qualified and produce no side effects.
3. WHEN resolve is called with a simulation Time_Point, THE Aliasing_Engine SHALL return an AliasedWindow containing a Time_Window (t_left and t_right as dataset Time_Points) and a double-precision interpolation weight alpha in [0.0, 1.0].
4. THE Aliasing_Engine SHALL be a class template parameterized on the Simulation_Calendar and Dataset_Calendar types, constrained by the tick::Calendar concept.
5. IF the Aliasing_Engine is constructed with a snapshot interval Duration that does not evenly divide the Dataset_Coverage duration, THEN THE Aliasing_Engine SHALL throw std::invalid_argument indicating that the snapshot interval must evenly divide the coverage.
6. IF the Aliasing_Engine is constructed with a snapshot interval Duration of zero or negative value, THEN THE Aliasing_Engine SHALL throw std::invalid_argument indicating that the snapshot interval must be positive.
7. IF the Aliasing_Engine is constructed with pure_climatology policy and the Climatological_Year falls outside the Dataset_Coverage range, THEN THE Aliasing_Engine SHALL throw std::invalid_argument indicating that the climatological year must be within dataset coverage.
8. THE Aliasing_Engine SHALL provide const accessors for all construction parameters: coverage(), snapshot_interval(), policy(), and climatological_year().

### Requirement 8: AliasedWindow Result Type

**User Story:** As a forcing data consumer, I want a structured result containing both the bounding timestamps and the interpolation weight, so that I can directly index into the dataset and blend two snapshots in a single step.

#### Acceptance Criteria

1. THE AliasedWindow SHALL be an aggregate struct containing a Time_Window member (representing t_left and t_right) and a double member (representing the interpolation weight alpha).
2. THE AliasedWindow alpha member SHALL always be in the closed interval [0.0, 1.0].
3. THE AliasedWindow SHALL support equality comparison where two AliasedWindow values are equal if their Time_Window members are equal and their alpha members are bitwise identical.
4. THE AliasedWindow SHALL be trivially copyable and default-constructible.

### Requirement 9: Stateless Design Constraint

**User Story:** As a HELM architect, I want the Aliasing Engine to remain purely functional with no hidden mutable state, so that it is thread-safe by construction and consistent with TICK's zero-side-effect philosophy.

#### Acceptance Criteria

1. THE Aliasing_Engine SHALL contain no mutable member variables, no static mutable state, and no non-const member functions.
2. WHEN resolve is called multiple times with the same simulation Time_Point on the same Aliasing_Engine instance, THE Aliasing_Engine SHALL return bitwise-identical AliasedWindow results.
3. THE Aliasing_Engine SHALL be safe to call concurrently from multiple threads without external synchronization.
4. THE Aliasing_Engine SHALL not allocate heap memory during resolve calls; all computation SHALL use stack-local variables and the immutable construction parameters.

### Requirement 10: Integer Arithmetic Invariant

**User Story:** As a HELM architect, I want all internal time calculations to remain in integer nanoseconds, so that temporal drift is impossible and the aliasing engine inherits TICK's precision guarantees.

#### Acceptance Criteria

1. THE Aliasing_Engine SHALL perform all internal time difference, addition, modulo, and comparison operations using int64_t nanosecond values obtained from Time_Point and Duration accessors.
2. THE Aliasing_Engine SHALL produce floating-point values only in the final interpolation weight calculation (one double division per resolve call).
3. IF any internal integer arithmetic operation would overflow the int64_t range during resolve, THEN THE Aliasing_Engine SHALL throw std::overflow_error indicating the operation that caused the overflow.
4. THE Aliasing_Engine SHALL NOT use floating-point intermediate values for year remapping, day clamping, modulo wrap-around, or any calendar conversion.

### Requirement 11: Property-Based Testing for Aliasing Correctness

**User Story:** As a quality engineer, I want property-based tests proving the aliasing engine's mathematical invariants, so that edge cases in year-boundary wrapping, leap-hold logic, and weight calculation are exhaustively verified.

#### Acceptance Criteria

1. THE test suite SHALL contain a property-based test proving the weight-bounds invariant: for all valid simulation Time_Points resolved by the Aliasing_Engine, the returned alpha SHALL be greater than or equal to 0.0 and less than or equal to 1.0.
2. THE test suite SHALL contain a property-based test proving the idempotence property of clamp_to_edge: for all simulation Time_Points beyond Dataset_Coverage end, resolving the same Time_Point repeatedly SHALL produce bitwise-identical AliasedWindow results.
3. THE test suite SHALL contain a property-based test proving the cycle-consistency property: for all simulation Time_Points that map to the same month/day/time under cycle_last_year policy, the returned AliasedWindow Time_Window endpoints SHALL fall within the last year of Dataset_Coverage.
4. THE test suite SHALL contain a property-based test proving the pure-climatology year-lock property: for all simulation Time_Points resolved under pure_climatology policy, both t_left and t_right of the returned Time_Window SHALL have year components equal to the Climatological_Year or Climatological_Year plus one (for December-to-January wrap).
5. THE test suite SHALL contain a property-based test proving the leap-hold freeze property: for all Gregorian simulation Time_Points falling on February 29 resolved against a NoLeap dataset under leap_hold policy, the returned alpha SHALL equal 0.0 and t_left shall correspond to February 28.
6. THE test suite SHALL contain a property-based test proving the weight-monotonicity property: for all pairs of simulation Time_Points (A, B) within the same Time_Window where A is earlier than B, the alpha for A SHALL be less than or equal to the alpha for B.
7. THE test suite SHALL use RapidCheck integrated with Google Test and execute a minimum of 1000 generated cases per property.
