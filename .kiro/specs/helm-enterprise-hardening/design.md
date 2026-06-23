# Design Document: HELM Enterprise Hardening

## Overview

This design specifies the enterprise hardening deliverables for the HELM micro-library ecosystem: a BLEND test suite, five CI pipelines, four README files, a root `.clang-format`, and tier isolation scripts. All work replicates established patterns from DAGR/HALO/LOGS without introducing a super-build or top-level CMakeLists.txt.

## Architecture

### High-Level File Layout

All new files are created within the existing monorepo structure. No new top-level directories are introduced.

```
HELM-Project/
├── .clang-format                              # NEW — Root formatting config (replica of AMIO)
├── .github/workflows/
│   ├── blend-ci.yml                           # NEW — BLEND CI pipeline
│   ├── span-ci.yml                            # NEW — SPAN CI pipeline
│   ├── conf-ci.yml                            # NEW — CONF CI pipeline
│   ├── axis-ci.yml                            # NEW — AXIS CI pipeline
│   ├── tick-ci.yml                            # NEW — TICK CI pipeline
│   ├── dagr-ci.yml                            # EXISTS
│   ├── halo-ci.yml                            # EXISTS
│   └── logs-ci.yml                            # EXISTS
├── libs/
│   ├── blend/
│   │   ├── CMakeLists.txt                     # EXISTS (already has BUILD_TESTING gate)
│   │   ├── README.md                          # NEW
│   │   ├── include/blend/helm_math_blend.hpp  # EXISTS
│   │   ├── cmake/
│   │   │   └── check_tier1_isolation.sh       # NEW — Isolation script
│   │   └── tests/
│   │       ├── CMakeLists.txt                 # NEW — Test build configuration
│   │       ├── generators.hpp                 # NEW — RapidCheck generators
│   │       ├── test_linear_blend.cpp          # NEW — LinearBlendKernel unit tests
│   │       ├── test_step_blend.cpp            # NEW — StepBlendKernel unit tests
│   │       ├── prop_identity.cpp              # NEW — Identity property test
│   │       ├── prop_step_select.cpp           # NEW — Step-select property test
│   │       └── prop_boundedness.cpp           # NEW — Boundedness property test
│   ├── span/
│   │   ├── README.md                          # NEW
│   │   └── cmake/
│   │       └── check_tier1_isolation.sh       # NEW — Isolation script
│   ├── conf/
│   │   ├── README.md                          # NEW
│   │   └── cmake/
│   │       └── check_tier1_isolation.sh       # NEW — Isolation script
│   ├── axis/
│   │   └── cmake/
│   │       └── check_tier1_isolation.sh       # EXISTS
│   ├── tick/
│   │   └── cmake/
│   │       └── check_tier1_isolation.sh       # NEW — Isolation script
│   └── dagr/
│       ├── README.md                          # NEW
│       └── scripts/
│           └── check_zero_computation.sh      # EXISTS
```

## Components and Interfaces

### 1. BLEND Test Suite

BLEND is a header-only (INTERFACE target) library with two kernels: `LinearBlendKernel` and `StepBlendKernel`. The test suite validates correctness via GTest unit tests and RapidCheck property tests.

#### Test File Architecture

| File | Purpose | CTest Label |
|------|---------|-------------|
| `test_linear_blend.cpp` | Fixed-input verification of `target[i] = left[i]*(1-α) + right[i]*α`; extent mismatch throws; empty array correctness | `unit` |
| `test_step_blend.cpp` | Fixed-input verification of step-select behavior; extent mismatch throws; empty array correctness | `unit` |
| `prop_identity.cpp` | Identity property: α=0 → left, α=1 → right | `property` |
| `prop_step_select.cpp` | Step-select correctness: output == left OR output == right | `property` |
| `prop_boundedness.cpp` | Interpolation boundedness: output ∈ [min(left,right), max(left,right)] for α∈[0,1] | `property` |

#### `tests/CMakeLists.txt` Structure

```cmake
include(GoogleTest)

find_package(GTest REQUIRED)
find_package(rapidcheck REQUIRED)

# ── Unit test targets ─────────────────────────────────────────────────────────

set(BLEND_UNIT_TESTS
    test_linear_blend
    test_step_blend
)

foreach(test_name IN LISTS BLEND_UNIT_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            HELM::BLEND
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    gtest_discover_tests(${test_name})
endforeach()

# Apply "unit" label to all discovered unit tests
foreach(test_name IN LISTS BLEND_UNIT_TESTS)
    get_test_property(${test_name} TESTS test_list)
endforeach()

# ── Property test targets (RapidCheck + GTest via RC_GTEST_PROP) ──────────────

set(BLEND_PROPERTY_TESTS
    prop_identity
    prop_step_select
    prop_boundedness
)

foreach(test_name IN LISTS BLEND_PROPERTY_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            HELM::BLEND
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "property"
        TIMEOUT 120
        ENVIRONMENT "RC_PARAMS=max_success=100"
    )
endforeach()

# ── Apply "unit" label via add_test + set_tests_properties ────────────────────
# (gtest_discover_tests auto-registers; we override labels post-discovery)
foreach(test_name IN LISTS BLEND_UNIT_TESTS)
    set_tests_properties(${test_name} PROPERTIES LABELS "unit")
endforeach()
```

#### `generators.hpp` — RapidCheck Generators for BLEND

```cpp
#pragma once

#include <rapidcheck.h>
#include <Kokkos_Core.hpp>
#include <blend/helm_math_blend.hpp>
#include <vector>
#include <cstddef>

namespace blend::gen {

/// Generate a random array length in [1, max_len].
inline auto array_length(std::size_t max_len = 256) {
    return rc::gen::inRange<std::size_t>(1, max_len + 1);
}

/// Generate a vector of random doubles in [-1e6, 1e6].
inline auto field_data(std::size_t len) {
    return rc::gen::container<std::vector<double>>(
        len, rc::gen::inRange<double>(-1e6, 1e6));
}

/// Generate an alpha value uniformly in [0.0, 1.0].
inline auto alpha_unit() {
    return rc::gen::map(
        rc::gen::inRange<int>(0, 10001),
        [](int x) { return static_cast<double>(x) / 10000.0; });
}

/// Generate an alpha value across full double range [-10.0, 10.0].
inline auto alpha_wide() {
    return rc::gen::map(
        rc::gen::inRange<int>(-100000, 100001),
        [](int x) { return static_cast<double>(x) / 10000.0; });
}

} // namespace blend::gen
```

#### Property Test Pattern (RC_GTEST_PROP macro)

Each property test file follows the DAGR/HALO pattern:

```cpp
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <blend/helm_math_blend.hpp>
#include "generators.hpp"

RC_GTEST_PROP(BlendIdentity, AlphaZeroReturnsLeft, ()) {
    auto n = *blend::gen::array_length();
    auto left_data = *blend::gen::field_data(n);
    auto right_data = *blend::gen::field_data(n);
    // ... create Kokkos views, call kernel, assert output == left
}
```

### 2. CI Pipeline Template

All five new pipelines follow the LOGS-style single-job sequential pattern (DAGR is the only one with a two-job structure due to its unique zero-computation check).

#### Reusable CI Structure

```yaml
name: {LIB} CI

on:
  push:
    paths:
      - "libs/{lib}/**"
      - "DockerFile"
      - "docker-compose.yml"
      - ".github/workflows/{lib}-ci.yml"
  pull_request:
    paths:
      - "libs/{lib}/**"
      - "DockerFile"
      - "docker-compose.yml"
      - ".github/workflows/{lib}-ci.yml"
  workflow_dispatch: {}

jobs:
  {lib}:
    name: {LIB} build, test, sanitize
    runs-on: ubuntu-latest
    steps:
      - name: Checkout (with submodules)
        uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Bring up HELM container
        run: docker compose up -d --build

      # Stage 1: Tier isolation scan (fail-fast)
      - name: "Stage 1: Tier isolation scan"
        run: |
          docker compose exec -T helm-dev bash -lc '
            cd libs/{lib} && sh cmake/check_tier1_isolation.sh .
          '

      # Stage 2: Standalone CMake build
      - name: "Stage 2: Standalone build (produces HELM::{LIB})"
        run: |
          docker compose exec -T helm-dev bash -lc '
            set -e
            cd libs/{lib}
            rm -rf build-ci
            cmake -B build-ci -G Ninja \
              -DCMAKE_CXX_STANDARD=20 \
              -DBUILD_TESTING=ON {extra_cmake_flags}
            cmake --build build-ci --parallel $(nproc)
          '

      # Stage 2b: Downstream consumer test
      - name: "Stage 2b: Downstream HELM::{LIB} consumer"
        run: |
          docker compose exec -T helm-dev bash -lc '
            set -e
            rm -rf /tmp/{lib}_consumer && mkdir -p /tmp/{lib}_consumer
            cat > /tmp/{lib}_consumer/CMakeLists.txt <<EOF
          cmake_minimum_required(VERSION 3.21)
          project({lib}_consumer LANGUAGES CXX)
          set(CMAKE_CXX_STANDARD 20)
          add_subdirectory(/workspace/helm-project/libs/{lib} \${CMAKE_BINARY_DIR}/{lib}_build)
          add_executable(consumer main.cpp)
          target_link_libraries(consumer PRIVATE HELM::{LIB})
          EOF
            printf "#include <{lib}/{header}>\nint main() { return 0; }\n" > /tmp/{lib}_consumer/main.cpp
            cd /tmp/{lib}_consumer
            cmake -B build -G Ninja -DBUILD_TESTING=OFF
            cmake --build build --target consumer
          '

      # Stage 3: Unit tests
      - name: "Stage 3: Unit tests"
        run: |
          docker compose exec -T helm-dev bash -lc '
            cd libs/{lib}/build-ci && ctest -L unit --output-on-failure
          '

      # Stage 4: Property tests
      - name: "Stage 4: Property tests"
        run: |
          docker compose exec -T helm-dev bash -lc '
            cd libs/{lib}/build-ci && ctest -L property --output-on-failure
          '

      # Stage 5: Sanitizer build (ASan + UBSan)
      - name: "Stage 5: Sanitizer build (ASan + UBSan)"
        run: |
          docker compose exec -T helm-dev bash -lc '
            set -e
            cd libs/{lib}
            rm -rf build-asan
            cmake -B build-asan -G Ninja \
              -DCMAKE_CXX_STANDARD=20 \
              -DBUILD_TESTING=ON {sanitizer_cmake_flags} \
              -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
            cmake --build build-asan --parallel $(nproc)
            cd build-asan
            ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
              ctest -L property --output-on-failure
          '

      - name: Cleanup scratch build trees
        if: always()
        run: |
          docker compose exec -T helm-dev bash -lc '
            cd libs/{lib} && rm -rf build-ci build-asan
          ' || true

      - name: Tear down container
        if: always()
        run: docker compose down || true
```

#### Per-Library Customizations

| Library | `{extra_cmake_flags}` | `{sanitizer_cmake_flags}` | Additional Stages |
|---------|----------------------|--------------------------|-------------------|
| BLEND | (none) | (none) | — |
| SPAN | (none) | (none) | — |
| CONF | `-DBUILD_FORTRAN=ON` | `-DBUILD_FORTRAN=OFF` | Fortran test stage: `ctest -L fortran` |
| AXIS | (none) | (none) | — |
| TICK | (none) | (none) | — |

### 3. Isolation Script Structure

Each new isolation script follows the LOGS `check_tier1_isolation.sh` pattern exactly. The only variance per library is the set of forbidden path segments.

#### Script Template

```bash
#!/usr/bin/env bash
# check_tier1_isolation.sh — Tier Isolation Static Verification for {LIB}
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_ROOT="${1:-"${SCRIPT_DIR}/.."}"
LIB_ROOT="$(cd "${LIB_ROOT}" && pwd)"

INCLUDE_DIR="${LIB_ROOT}/include"
SRC_DIR="${LIB_ROOT}/src"

FILES=()
for dir in "${INCLUDE_DIR}" "${SRC_DIR}"; do
    if [[ -d "${dir}" ]]; then
        while IFS= read -r -d '' file; do
            FILES+=("${file}")
        done < <(find "${dir}" -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.h" \) -print0)
    fi
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "TIER_ISOLATION: No source/header files found to scan in ${LIB_ROOT}"
    echo "TIER_ISOLATION: PASSED (no files)"
    exit 0
fi

# Library-specific forbidden patterns
FORBIDDEN_PATTERN='#[[:space:]]*include[[:space:]]*[<"][^>"]*\b({FORBIDDEN_LIBS})/[^>"]*[>"]'

VIOLATIONS=()
VIOLATION_COUNT=0

for file in "${FILES[@]}"; do
    while IFS= read -r match; do
        VIOLATIONS+=("${file}:${match}")
        ((VIOLATION_COUNT++))
    done < <(grep -inE "${FORBIDDEN_PATTERN}" "${file}" 2>/dev/null || true)
done

if [[ ${VIOLATION_COUNT} -gt 0 ]]; then
    echo "TIER_ISOLATION: FAILED — ${VIOLATION_COUNT} forbidden #include directive(s) detected."
    for violation in "${VIOLATIONS[@]}"; do
        echo "  ${violation}"
    done
    exit 1
else
    echo "TIER_ISOLATION: PASSED — scanned ${#FILES[@]} file(s), no forbidden includes found."
    exit 0
fi
```

#### Forbidden Includes per Library

| Library | Forbidden Path Segments |
|---------|------------------------|
| BLEND | `halo`, `logs`, `axis`, `amio`, `span`, `dagr`, `conf`, `tick` |
| SPAN | `halo`, `logs`, `axis`, `amio`, `dagr`, `conf`, `tick`, `blend` |
| CONF | `halo`, `logs`, `axis`, `amio`, `span`, `dagr`, `tick`, `blend` |
| AXIS | `halo`, `logs`, `amio`, `span`, `dagr`, `conf`, `tick`, `blend` |
| TICK | `halo`, `logs`, `axis`, `amio`, `span`, `dagr`, `conf`, `blend` |

**Note:** The C++20 `<span>` header (bare, no trailing `/`) is explicitly permitted by the regex — only `span/` path segments are forbidden.

### 4. README Structure Template

Each README follows the TICK/AXIS format:

```markdown
# {LIB} — {Full Name}

{One-paragraph description of the library's purpose and tier placement.}

## Features

- {Feature bullet points}

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13 | Required |
| CMake | ≥ 3.21 | Required |
| {Library-specific deps} | ... | ... |
| GTest | any recent | Required when BUILD_TESTING=ON |
| RapidCheck | latest | Required when BUILD_TESTING=ON |

All prerequisites are pre-installed in the HELM Docker development container.

## Docker Container Launch

{Standard docker compose instructions}

## CMake Configure and Build

{Library-specific cmake commands}

### CMake Options

| Option | Default | Description |
|---|---|---|
| BUILD_TESTING | OFF | Build the test suite |
| {Other options} | ... | ... |

### Consuming {LIB} from a downstream project

{cmake find_package / target_link_libraries example}

## Running Tests

{ctest commands}

## License

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER).
```

### 5. Root `.clang-format`

An exact replica of `libs/AMIO/.clang-format`:

```yaml
BasedOnStyle: Google
ColumnLimit: 150
Standard: c++20
IndentWidth: 4
PointerAlignment: Right
AllowShortFunctionsOnASingleLine: Empty
AlignAfterOpenBracket: Align
BreakBeforeBraces: Attach
```

### 6. Interfaces

#### BLEND Test Interface with Kokkos

Tests must initialize Kokkos before running (GTest provides `main()` via `gtest_main`, but Kokkos needs `Kokkos::initialize`). The test infrastructure uses a GTest environment to manage Kokkos lifecycle:

```cpp
// In each test file or a shared test_main.cpp
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

// Register before RUN_ALL_TESTS (or use a global static registration)
```

Alternatively, since BLEND uses `gtest_main`, the Kokkos initialization can be handled via a global fixture registered in each test file.

#### CI Pipeline Interface with Docker

All CI stages interact with the build environment via:

```
docker compose exec -T helm-dev bash -lc '<commands>'
```

The container mounts the workspace at `/workspace/helm-project` and provides all toolchain dependencies (GCC-13, CMake, Ninja, Kokkos, GTest, RapidCheck, yaml-cpp, gfortran).

#### Isolation Script Interface

Each script accepts one optional argument — the library root path — defaulting to the parent of its own directory (`cmake/` or `scripts/`). It exits 0 on success and 1 on violation, printing human-readable diagnostics to stdout.

## Data Models

### BLEND Test Data

Property tests generate data using RapidCheck generators:

- **Array length**: `std::size_t` in `[1, 256]` (sufficient to exercise Kokkos parallel dispatch without excessive runtime)
- **Field values**: `double` in `[-1e6, 1e6]` (avoids infinity/NaN while covering wide numeric range)
- **Alpha (unit)**: `double` in `[0.0, 1.0]` discretized to 4 decimal places (avoids floating-point comparison issues at boundaries)
- **Alpha (wide)**: `double` in `[-10.0, 10.0]` for testing extrapolation behavior

Data is stored in `std::vector<double>` on the host, then wrapped in `Kokkos::View<double*, Kokkos::MemoryUnmanaged>` for kernel invocation.

## Error Handling

### BLEND Kernel Error Handling

Both kernels throw `std::invalid_argument` for extent mismatches. The test suite verifies this via `EXPECT_THROW(...)`. Zero-length arrays are valid input and produce no output (no-op).

### CI Pipeline Failure Modes

- **Isolation scan failure**: Pipeline terminates immediately (fail-fast before build)
- **Build failure**: Subsequent test stages are skipped
- **Test failure**: Other test stages still execute; sanitizer stage always runs
- **Container failure**: `docker compose down` in `if: always()` block ensures cleanup

### Isolation Script Error Handling

Scripts use `set -uo pipefail` for strict error handling. If no source files are found to scan (e.g., include/src directories missing), the script passes with a warning — this prevents false failures during library bootstrapping.

## Testing Strategy

### Unit Tests (GTest)

Unit tests cover specific examples and edge cases with deterministic inputs:

- **LinearBlendKernel**: Fixed-input formula verification, extent mismatch exception, empty array no-op
- **StepBlendKernel**: Fixed-input step-select verification (both sides of 0.5 threshold), extent mismatch exception, empty array no-op

### Property Tests (RapidCheck + GTest via RC_GTEST_PROP)

Property tests validate universal correctness guarantees over randomized inputs (minimum 100 iterations per property):

- **Identity property**: Boundary behavior at α=0 and α=1
- **Step-select property**: Output always equals exactly one input
- **Boundedness property**: Interpolation stays within element-wise bounds

### Integration / Smoke Tests

- **Downstream consumer test**: Verifies `HELM::BLEND` alias is consumable without hidden dependencies
- **Isolation scan**: Verifies no forbidden cross-tier includes
- **CI pipeline execution**: End-to-end validation inside Docker container

### Test Configuration

- Property tests use `RC_PARAMS=max_success=100` environment variable
- Test timeout: 120 seconds for property tests, 60 seconds for unit tests
- CTest labels: `"unit"` and `"property"` for selective execution

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: LinearBlendKernel Identity

*For any* array of doubles `left` and `right` of equal length N ≥ 1, `LinearBlendKernel::apply(left, right, target, 0.0)` SHALL produce `target[i] == left[i]` for all i, and `LinearBlendKernel::apply(left, right, target, 1.0)` SHALL produce `target[i] == right[i]` for all i.

**Validates: Requirements 1.7**

### Property 2: StepBlendKernel Step-Select Correctness

*For any* arrays `left` and `right` of equal length N ≥ 1 and *for any* alpha value, `StepBlendKernel::apply(left, right, target, alpha)` SHALL produce output where `target[i] == left[i]` for all i when alpha < 0.5, or `target[i] == right[i]` for all i when alpha ≥ 0.5. That is, the output always equals exactly one of the two inputs.

**Validates: Requirements 1.8**

### Property 3: LinearBlendKernel Interpolation Boundedness

*For any* arrays `left` and `right` of equal length N ≥ 1 and *for any* alpha ∈ [0.0, 1.0], `LinearBlendKernel::apply(left, right, target, alpha)` SHALL produce output where `min(left[i], right[i]) <= target[i] <= max(left[i], right[i])` for all i.

**Validates: Requirements 1.9**
