# TICK — Time Integration & Chronology Kernel

TICK is a Tier 1 C++20 micro-library in the HELM ecosystem providing stateless,
fixed-point time management for Earth system models. It replaces the legacy ESMF
Time Manager with zero-dependency, zero-drift temporal arithmetic.

**Key capabilities:**

- Fixed-point `int64_t` nanosecond arithmetic (no floating-point)
- Three calendar engines: Gregorian, NoLeap (365-day), and Cal360 (360-day)
- Interval and absolute alarm triggers
- GCD heartbeat and LCM coupling-period synchronization
- Temporal accumulation window computation

All core types (`Time_Point`, `Duration`, `Date_Time`, `Time_Window`) are
immutable value types with `constexpr` arithmetic.

## Prerequisites

TICK must be built inside the HELM project Docker container. The container
provides:

- GCC 13 (C++20 support)
- CMake 3.21+
- Google Test
- RapidCheck (property-based testing)

No additional dependencies are required beyond the C++20 standard library for
building the library itself.

## Getting Started

### 1. Launch the development container

From the HELM project root:

```bash
docker compose run --rm helm-dev bash
```

### 2. Configure

Inside the container:

```bash
cd /workspace/helm-project/libs/tick
mkdir -p build && cd build
cmake .. -DTICK_BUILD_TESTING=ON
```

### 3. Build

```bash
cmake --build . --parallel
```

### 4. Run tests

```bash
ctest --output-on-failure
```

## API Summary

Include the umbrella header for full access:

```cpp
#include <tick/tick.hpp>
```

### Core Types

| Header | Type | Description |
|--------|------|-------------|
| `tick/time_point.hpp` | `tick::Time_Point` | Immutable nanosecond offset from epoch |
| `tick/duration.hpp` | `tick::Duration` | Immutable nanosecond time interval |
| `tick/date_time.hpp` | `tick::Date_Time` | Decomposed year/month/day/h/m/s/ns |

### Calendar Engines

| Header | Type | Description |
|--------|------|-------------|
| `tick/gregorian_calendar.hpp` | `tick::Gregorian_Calendar` | Proleptic Gregorian with leap years |
| `tick/noleap_calendar.hpp` | `tick::NoLeap_Calendar` | 365-day year, no leap days |
| `tick/cal360_calendar.hpp` | `tick::Cal360_Calendar` | 360-day year, 30-day months |

### Alarms

| Header | Type | Description |
|--------|------|-------------|
| `tick/interval_alarm.hpp` | `tick::Interval_Alarm` | Periodic trigger at fixed intervals |
| `tick/absolute_alarm.hpp` | `tick::Absolute_Alarm` | Single-fire trigger at a precise time |

### Synchronization

| Header | Function | Description |
|--------|----------|-------------|
| `tick/sync.hpp` | `tick::compute_heartbeat()` | GCD of model timesteps |
| `tick/sync.hpp` | `tick::compute_sync_period()` | LCM coupling period |
| `tick/sync.hpp` | `tick::is_phase_aligned()` | Phase alignment check |

### Accumulation Windows

| Header | Type / Function | Description |
|--------|-----------------|-------------|
| `tick/time_window.hpp` | `tick::Time_Window` | Half-open interval [start, end) |
| `tick/time_window.hpp` | `tick::is_on_boundary()` | Window boundary detection |
| `tick/time_window.hpp` | `tick::compute_window()` | Current window for a given time |

### Factory Functions (in `tick/duration.hpp`)

```cpp
tick::nanoseconds(count)
tick::microseconds(count)
tick::milliseconds(count)
tick::seconds(count)
tick::minutes(count)
tick::hours(count)
tick::days(count)
```

## License

See the top-level [LICENSE](../../LICENSE) file in the HELM project root.
