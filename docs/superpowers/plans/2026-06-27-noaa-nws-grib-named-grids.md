# AXIS NOAA NWS Named GRIB Grids Support (G-Family) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the new NOAA G-family inside `NamedGridRegistry`, adding an extensible static array structure to easily register and generate GFS `grid3`, GFS `grid4`, and NAM `grid218` grids.

**Architecture:** Extend the `NamedGridRegistry::parse` and `generate` methods to support G-family prefixes, routing to a static, declarative configuration array of grid parameters.

**Tech Stack:** C++20, Kokkos 5.1.1, PROJ.

## Global Constraints
- Do not use dynamic memory allocations (`malloc`, `free`, `new`, `delete`) inside device kernels.
- Support double-precision calculations with $1.0 \times 10^{-12}$ relative tolerance.
- Adhere strictly to existing AXIS code patterns and file layouts.
- **CRITICAL:** All compile, run, or validation commands MUST be executed inside the running Docker container `helm-dev-env` using `docker exec`. Do NOT run local commands on the macOS host.

---

### Task 1: Extend NamedGridRegistry Parsing and G-Family Dispatcher

**Files:**
- Modify: `libs/axis/include/axis/topology/named_grid_registry.hpp`
- Modify: `libs/axis/src/topology/named_grid_registry.cpp`

**Interfaces:**
- Produces: `NamedGridRegistry::ParsedName` mapping G-family grid numbers

- [ ] **Step 1: Update `NamedGridRegistry::parse` to support case-insensitive `"grid"` names**

In `libs/axis/src/topology/named_grid_registry.cpp`, add case-insensitive `"grid"` prefix parsing to match family `'G'`:
```cpp
    std::string lower_name = name;
    for (char& c : lower_name) c = std::tolower(c);
    
    if (lower_name.rfind("grid", 0) == 0) {
        std::string num_str = name.substr(4);
        int grid_num = std::stoi(num_str);
        return ParsedName{'G', grid_num};
    }
```

- [ ] **Step 2: Add Case `'G'` in `NamedGridRegistry::generate` dispatcher**

Dispatch G-family grid numbers to `generate_noaa_grib_grid`:
```cpp
        case 'G':
            return generate_noaa_grib_grid<MemorySpace>(parsed.number);
```

- [ ] **Step 3: Run verify compile in Docker container**

Execute in `helm-dev-env` container:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8"
```

---

### Task 2: Implement Extensible G-Family Static Grid Definitions and Generators

**Files:**
- Modify: `libs/axis/src/topology/named_grid_registry.cpp`

**Interfaces:**
- Produces: `generate_noaa_grib_grid`

- [ ] **Step 1: Define `NoaaGribDefinition` and static configuration registry**

Define the extensible metadata structures inside `named_grid_registry.cpp`:
```cpp
struct NoaaGribDefinition {
    int number;
    std::size_t ni;
    std::size_t nj;
    double lon_start;
    double lat_start;
    double dlon;
    double dlat;
    const char* proj_string; // nullptr for regular lat-lon
};

static const NoaaGribDefinition NOAA_GRIB_GRIDS[] = {
    {3, 360, 181, -180.0, -90.0, 1.0, 1.0, nullptr}, // GFS 1.0 degree
    {4, 720, 361, -180.0, -90.0, 0.5, 0.5, nullptr}, // GFS 0.5 degree
    {218, 614, 428, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"} // NAM 12km
};
```

- [ ] **Step 2: Implement `generate_noaa_grib_grid` using the registry lookup**

Implement the generator matching `NOAA_GRIB_GRIDS` definitions. For regular grids (where `proj_string` is null), call coordinate arrays on-device. For projected grids, invoke `ProjectionBuilder::build`. Throw `std::invalid_argument` for unregistered grid numbers.

- [ ] **Step 3: Run compile check inside Docker**

Verify complete C++ compilation using `docker exec`.

---

### Task 3: Unit Testing & G-Family Verification

**Files:**
- Create: `libs/axis/tests/test_noaa_grib_named_grids.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Write `test_noaa_grib_named_grids.cpp`**

Write Google test cases asserting:
- `"grid3"` generates $360 \times 181 = 65,160$ cells.
- `"grid4"` generates $720 \times 361 = 259,920$ cells.
- `"grid218"` generates $614 \times 428 = 262,792$ cells (when PROJ is enabled).
- Unknown grid numbers (e.g. `"grid999"`) throw standard `std::invalid_argument`.

- [ ] **Step 2: Register test and execute in Docker container**

Add `test_noaa_grib_named_grids.cpp` to `AXIS_UNIT_TEST_SOURCES`. Run and verify using `docker exec`:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8 && ./tests/axis_unit_tests --gtest_filter=\"NoaaGribNamedGrids.*\""
```
