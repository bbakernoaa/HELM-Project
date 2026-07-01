# AXIS Advanced Regridding Suite (Vertical, Vector, and Tripolar) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a thread-safe, GPU-portable suite of advanced regridding solvers, covering:
1.  **AXIS-Vertical:** 1D vertical tension spline regridding based on the TSPACK algorithm, supporting both uniform (1D) and column-varying (2D) levels.
2.  **AXIS-Vector:** 2D vector field regridding with local frame rotations pre-assembled into unified remapping matrices.
3.  **AXIS-Tripolar:** Analytical coordinate fast-path for tripolar ORCA ocean grids to resolve polar folding seams.

**Architecture:**
- *Vertical:* Port Robert Renka's core TSPACK spline evaluation and tridiagonal solving routines into C++ device-resident inline functions (`KOKKOS_FUNCTION`) running column-parallel with static stack-allocated workspaces.
- *Vector:* Pre-assemble coordinate rotation matrices ($R$) with spatial interpolation weights ($W$) into coupled SpMV weights matrices ($W_u$, $W_v$).
- *Tripolar:* Identify tripolar land poles analytically and execute localized $O(1)$ reflection coordinate translation near folded boundaries.

**Tech Stack:** C++20, Kokkos 5.1.1, PROJ.

## Global Constraints
- Do not use dynamic memory allocations (`malloc`, `free`, `new`, `delete`) inside device kernels.
- Support double-precision calculations with $1.0 \times 10^{-12}$ relative tolerance.
- Adhere strictly to existing AXIS code patterns and file layouts.
- All compile and test execution validation checks MUST be executed inside the running Docker container `helm-dev-env` using `docker exec`.

---

### Task 1: TSPACK Core Mathematical Port

**Files:**
- Create: `libs/axis/include/axis/detail/tspack.hpp` (COMPLETED)
- Create: `libs/axis/src/detail/tspack.cpp` (COMPLETED)
- Modify: `libs/axis/CMakeLists.txt` (COMPLETED)

**Interfaces:**
- Produces: `axis::detail::tspack::evaluate_spline` and `axis::detail::tspack::solve_column_spline`

- [x] **Step 1: Write header declaration `tspack.hpp`** (Done)
- [x] **Step 2: Implement spline routines in `tspack.cpp`** (Done)
- [x] **Step 3: Register files in `libs/axis/CMakeLists.txt`** (Done)

---

### Task 2: Vertical Regridding Interface & Host-Device Dispatcher

**Files:**
- Create: `libs/axis/include/axis/solver/vertical_regridder.hpp`
- Create: `libs/axis/src/solver/vertical_regridder.cpp`
- Modify: `libs/axis/CMakeLists.txt`

**Interfaces:**
- Consumes: `axis::detail::tspack::solve_column_spline` and `evaluate_spline`
- Produces: `axis::solver::VerticalRegridder::interpolate`

- [ ] **Step 1: Write header `vertical_regridder.hpp`**

Write `libs/axis/include/axis/solver/vertical_regridder.hpp` with the static public interfaces:
```cpp
#ifndef AXIS_SOLVER_VERTICAL_REGRIDDER_HPP
#define AXIS_SOLVER_VERTICAL_REGRIDDER_HPP

#include <Kokkos_Core.hpp>

namespace axis::solver {

template <typename MemorySpace>
class VerticalRegridder {
public:
    static void interpolate(
        Kokkos::View<const double**, MemorySpace> src_field,
        Kokkos::View<double**, MemorySpace>       dst_field,
        Kokkos::View<const double*, MemorySpace>  src_levels,
        Kokkos::View<const double*, MemorySpace>  dst_levels,
        double tension = 0.0
    );

    static void interpolate(
        Kokkos::View<const double**, MemorySpace>  src_field,
        Kokkos::View<double**, MemorySpace>        dst_field,
        Kokkos::View<const double**, MemorySpace>  src_levels,
        Kokkos::View<const double**, MemorySpace>  dst_levels,
        double tension = 0.0
    );
};

} // namespace axis::solver

#endif // AXIS_SOLVER_VERTICAL_REGRIDDER_HPP
```

- [ ] **Step 2: Implement dispatcher in `vertical_regridder.cpp`**

Write `libs/axis/src/solver/vertical_regridder.cpp` executing `solve_column_spline` and `evaluate_spline` inside parallel threads:
```cpp
#include <axis/solver/vertical_regridder.hpp>
#include <axis/detail/tspack.hpp>
#include <stdexcept>

namespace axis::solver {

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(
    Kokkos::View<const double**, MemorySpace> src_field,
    Kokkos::View<double**, MemorySpace>       dst_field,
    Kokkos::View<const double*, MemorySpace>  src_levels,
    Kokkos::View<const double*, MemorySpace>  dst_levels,
    double tension) {

    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(0);
    const std::size_t n_dst = dst_levels.extent(0);

    constexpr std::size_t MAX_LEVELS = 256;
    if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
        throw std::invalid_argument("VerticalRegridder: Level count exceeds 256 cap");
    }

    Kokkos::parallel_for("VerticalInterpolate1D", Kokkos::RangePolicy<MemorySpace>(0, n_col),
        KOKKOS_LAMBDA(const std::size_t c) {
            double d[MAX_LEVELS];
            double scratch[MAX_LEVELS];
            double src_y[MAX_LEVELS];
            double src_x[MAX_LEVELS];

            for (std::size_t i = 0; i < n_src; ++i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(i);
            }

            axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch);

            for (std::size_t j = 0; j < n_dst; ++j) {
                double target = dst_levels(j);
                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx+1] < target) {
                    idx++;
                }
                dst_field(c, j) = axis::detail::tspack::evaluate_spline(
                    target, src_x[idx], src_x[idx+1], src_y[idx], src_y[idx+1],
                    d[idx], d[idx+1], tension);
            }
        });
}

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(
    Kokkos::View<const double**, MemorySpace>  src_field,
    Kokkos::View<double**, MemorySpace>        dst_field,
    Kokkos::View<const double**, MemorySpace>  src_levels,
    Kokkos::View<const double**, MemorySpace>  dst_levels,
    double tension) {

    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(1);
    const std::size_t n_dst = dst_levels.extent(1);

    constexpr std::size_t MAX_LEVELS = 256;
    if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
        throw std::invalid_argument("VerticalRegridder: Level count exceeds 256 cap");
    }

    Kokkos::parallel_for("VerticalInterpolate2D", Kokkos::RangePolicy<MemorySpace>(0, n_col),
        KOKKOS_LAMBDA(const std::size_t c) {
            double d[MAX_LEVELS];
            double scratch[MAX_LEVELS];
            double src_y[MAX_LEVELS];
            double src_x[MAX_LEVELS];

            for (std::size_t i = 0; i < n_src; ++i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(c, i);
            }

            axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch);

            for (std::size_t j = 0; j < n_dst; ++j) {
                double target = dst_levels(c, j);
                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx+1] < target) {
                    idx++;
                }
                dst_field(c, j) = axis::detail::tspack::evaluate_spline(
                    target, src_x[idx], src_x[idx+1], src_y[idx], src_y[idx+1],
                    d[idx], d[idx+1], tension);
            }
        });
}

template class VerticalRegridder<Kokkos::HostSpace>;

} // namespace axis::solver
```

- [ ] **Step 3: Register solver file in CMakeLists.txt**

Add `src/solver/vertical_regridder.cpp` to the `AXIS_SOURCES` list.

---

### Task 3: Vertical Unit Testing & Reference Verification

**Files:**
- Create: `libs/axis/tests/test_vertical_regridder.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Write `test_vertical_regridder.cpp`**

Write `libs/axis/tests/test_vertical_regridder.cpp` evaluating linear profiles:
```cpp
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/solver/vertical_regridder.hpp>

namespace axis::test {

TEST(VerticalRegridderTest, UniformLinearInterpolation) {
    const std::size_t n_col = 5;
    const std::size_t n_src = 11;
    const std::size_t n_dst = 6;

    Kokkos::View<double**, Kokkos::HostSpace> src_field("src_field", n_col, n_src);
    Kokkos::View<double**, Kokkos::HostSpace> dst_field("dst_field", n_col, n_dst);
    Kokkos::View<double*, Kokkos::HostSpace> src_levels("src_levels", n_src);
    Kokkos::View<double*, Kokkos::HostSpace> dst_levels("dst_levels", n_dst);

    for (std::size_t i = 0; i < n_src; ++i) src_levels(i) = static_cast<double>(i);
    for (std::size_t j = 0; j < n_dst; ++j) dst_levels(j) = static_cast<double>(j) * 2.0;

    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t i = 0; i < n_src; ++i) {
            src_field(c, i) = 2.0 * src_levels(i) + 5.0;
        }
    }

    axis::solver::VerticalRegridder<Kokkos::HostSpace>::interpolate(
        src_field, dst_field, src_levels, dst_levels, 0.0);

    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t j = 0; j < n_dst; ++j) {
            double expected = 2.0 * dst_levels(j) + 5.0;
            EXPECT_NEAR(dst_field(c, j), expected, 1e-12);
        }
    }
}

} // namespace axis::test
```

- [ ] **Step 2: Register test file in tests CMakeLists.txt**

Add `test_vertical_regridder.cpp` to `AXIS_UNIT_TEST_SOURCES`.

- [ ] **Step 3: Compile and run test in Docker**

Verify with `ctest --output-on-failure` inside build directory using `docker exec`.

---

### Task 4: Vector Grid Rotation & Weights Pre-Assembly

**Files:**
- Create: `libs/axis/include/axis/solver/vector_regridder.hpp`
- Create: `libs/axis/src/solver/vector_regridder.cpp`
- Modify: `libs/axis/CMakeLists.txt`

**Interfaces:**
- Produces: `axis::solver::VectorWeightGenerator::generate`

- [ ] **Step 1: Write header `vector_regridder.hpp`**

Define `VectorWeightGenerator` interface:
```cpp
#ifndef AXIS_SOLVER_VECTOR_REGRIDDER_HPP
#define AXIS_SOLVER_VECTOR_REGRIDDER_HPP

#include <axis/solver/interpolation_matrix.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/solver/regrid_config.hpp>
#include <Kokkos_Core.hpp>
#include <utility>

namespace axis::solver {

template <typename MemorySpace>
struct GridRotation {
    Kokkos::View<const double*, MemorySpace> alpha;
};

template <typename MemorySpace>
class VectorWeightGenerator {
public:
    static std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>>
    generate(
        const topology::UnstructuredMesh<MemorySpace>& src_mesh,
        const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
        const GridRotation<MemorySpace>&               src_rotation,
        const GridRotation<MemorySpace>&               dst_rotation,
        const RegridConfig&                            config
    );
};

} // namespace axis::solver

#endif // AXIS_SOLVER_VECTOR_REGRIDDER_HPP
```

- [ ] **Step 2: Implement coupled weights assembly in `vector_regridder.cpp`**

Write `libs/axis/src/solver/vector_regridder.cpp` mathematically coupling local grid rotations into $W_u$, $W_v$ matrix products.

- [ ] **Step 3: Register in parent `CMakeLists.txt`**

Add `src/solver/vector_regridder.cpp` to the build.

---

### Task 5: Vector Regridding Unit Tests

**Files:**
- Create: `libs/axis/tests/test_vector_regridder.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Write `test_vector_regridder.cpp`**

Assert solid-body rotation velocities conserve amplitude under curvilinear rotation transforms.

- [ ] **Step 2: Register in tests `CMakeLists.txt`**

Add `test_vector_regridder.cpp` to the build.

- [ ] **Step 3: Run verify compile in Docker**

---

### Task 6: Tripolar Grid Detection & Reflection Query Fast-Path

**Files:**
- Modify: `libs/axis/include/axis/detail/regular_grid_detector.hpp`
- Modify: `libs/axis/src/topology/projection_builder.cpp`

- [ ] **Step 1: Implement `detect_tripolar_grid()`**

Add logic to scan folded northern lines near the Canada/Siberia land pole boundaries.

- [ ] **Step 2: Integrate localized reflection fast-path**

Pre-calculate coordinate reflections near land poles to short-circuit index lookups.

- [ ] **Step 3: Compile and run test suite in Docker**

---

### Task 7: Tripolar Boundary Seam Unit Tests

**Files:**
- Create: `libs/axis/tests/test_tripolar_regridder.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Write `test_tripolar_regridder.cpp`**

Assert perfect mass conservation ($1.0 \times 10^{-12}$ error) over the folded seam.

- [ ] **Step 2: Register and run test in Docker**
