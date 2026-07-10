# NOAA Budget Interpolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the mass-conserving NOAA Budget Interpolation method in AXIS (compatible with UPP and NCEPLIBS-ip), offering configurable sub-grid resolution and min-valid-fraction gating.

**Architecture:** We will introduce a new `InterpolationMethod::Budget` option inside `RegridConfig`. The weight generation algorithm will be added directly within `libs/axis/src/solver/weight_generator.cpp` to natively reuse spatial indexing (ArborX), coordinate projection (Gnomonic), and point-in-cell mapping functions without duplicating code.

**Tech Stack:** C++20, Kokkos, ArborX, Google Test, RapidCheck, Docker.

## Global Constraints
- AXIS is a stateless, pure-data micro-library: no direct file I/O, no MPI communication (delegated to AMIO and HALO respectively).
- No new external dependencies can be introduced.
- Existing tests must remain completely unaffected.

---

## File Structure

The implementation will modify the following files:
1. `libs/axis/include/axis/solver/regrid_config.hpp`: Add `InterpolationMethod::Budget` and configuration parameters to `RegridConfig`.
2. `libs/axis/src/solver/weight_generator.cpp`: Implement `generate_budget` method, dispatch switch cases, and explicit template instantiations.
3. `libs/axis/tests/CMakeLists.txt`: Register the new unit and property test files.

And will create these new files:
1. `libs/axis/tests/test_budget.cpp`: Unit tests validating mathematical correctness and parity.
2. `libs/axis/tests/prop_budget.cpp`: Property-based correctness guarantees under randomized coordinate meshes.

---

### Task 1: Update Configuration and Enums

**Files:**
- Modify: `libs/axis/include/axis/solver/regrid_config.hpp:20-60`

**Interfaces:**
- Produces: `InterpolationMethod::Budget`, and `RegridConfig::budget_subgrid_size` and `RegridConfig::budget_min_valid_fraction` properties.

- [ ] **Step 1: Write code changes in regrid_config.hpp**

Add `Budget` to `InterpolationMethod`:
```cpp
enum class InterpolationMethod : std::uint8_t {
    Bilinear,
    NearestNeighbor,
    Bicubic,
    Patch,
    Conservative1stOrder,
    Conservative2ndOrder,
    Budget  ///< NOAA-EMC/UPP Budget interpolation: box-sampled bilinear averaging
};
```

Add fields to `RegridConfig` struct:
```cpp
struct RegridConfig {
    InterpolationMethod method = InterpolationMethod::Bilinear;
    NormType norm_type = NormType::DstArea;
    LineType line_type = LineType::GreatCircle;
    UnmappedAction unmapped = UnmappedAction::Ignore;
    bool use_limiter = false;
    ExtrapolationAction extrap_method = ExtrapolationAction::NearestWet;

    // Budget-specific configuration
    std::uint32_t budget_subgrid_size = 5;      ///< Resolution of sampling sub-grid (default: 5)
    double budget_min_valid_fraction = 0.5;   ///< Min fraction of valid subgrid points (default: 0.5)
};
```

- [ ] **Step 2: Run verification compilation**

Execute compiler test inside the Docker container to verify syntax correctness:
```bash
docker exec helm-dev-env make -C /workspace/helm-project/libs/axis/build -j4
```
Expected: Compilation passes.

- [ ] **Step 3: Commit changes**

```bash
git add libs/axis/include/axis/solver/regrid_config.hpp
git commit -m "feat(axis): add InterpolationMethod::Budget and config parameters"
```

---

### Task 2: Stub and Dispatch Routing in WeightGenerator

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:1990-2020` and end of file.

**Interfaces:**
- Consumes: `InterpolationMethod::Budget` configuration.
- Produces: `generate_budget` entry point.

- [ ] **Step 1: Declare forward declaration and dispatch case in weight_generator.cpp**

Add forward declaration in `libs/axis/src/solver/weight_generator.cpp` near other `generate_*` helpers:
```cpp
template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_budget(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                 const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                 const RegridConfig &config) {
    throw std::runtime_error("generate_budget not yet implemented");
}
```

Add case dispatch inside `WeightGenerator::generate`:
```cpp
        case InterpolationMethod::Budget:
            raw_matrix = generate_budget(src_mesh, dst_mesh, config);
            break;
```

- [ ] **Step 2: Run verification compilation**

```bash
docker exec helm-dev-env make -C /workspace/helm-project/libs/axis/build -j4
```
Expected: Compiles with no errors.

- [ ] **Step 3: Commit changes**

```bash
git add libs/axis/src/solver/weight_generator.cpp
git commit -m "feat(axis): stub and route generate_budget"
```

---

### Task 3: Implement General Mesh Host Path for Budget Interpolation

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp` (replaces the stub `generate_budget` function)

**Interfaces:**
- Produces: Complete `generate_budget` implementation computing the averaged bilinear weights of $N \times N$ subgrid points for each destination cell.

- [ ] **Step 1: Implement `generate_budget` on host**

Replace the `generate_budget` stub in `libs/axis/src/solver/weight_generator.cpp` with the active host path algorithm:
```cpp
template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_budget(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                 const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                 const RegridConfig &config) {
    using HostSpace = Kokkos::HostSpace;
    using Point2 = ArborX::Point<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();
    const auto csys = src_mesh.coord_system();
    const bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);

    const std::uint32_t N = config.budget_subgrid_size;
    const std::uint32_t N2 = N * N;

    // 1. Build ArborX BVH for Source Centroids
    Kokkos::View<double *, HostSpace> src_cx, src_cy;
    compute_cell_centroids_xy(src_mesh, src_cx, src_cy);

    Kokkos::View<Point2 *, HostSpace> src_points("src_points", n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_points(i) = Point2{static_cast<float>(src_cx(i)), static_cast<float>(src_cy(i))};
    }

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(host_exec, ArborX::Experimental::attach_indices(src_points));

    // Access source coords and connectivity
    const auto coords = src_mesh.node_coords();
    const auto conn_off = src_mesh.conn_offsets();
    const auto conn_idx = src_mesh.conn_indices();

    // Access destination coords and connectivity
    const auto dst_coords = dst_mesh.node_coords();
    const auto dst_off = dst_mesh.conn_offsets();
    const auto dst_idx = dst_mesh.conn_indices();

    std::vector<double> weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    // 2. Loop over each destination cell
    for (std::size_t j = 0; j < n_dst; ++j) {
        auto d_start = static_cast<std::size_t>(dst_off[j]);
        auto d_end = static_cast<std::size_t>(dst_off[j + 1]);
        if (d_start == d_end) continue;

        // Compute local destination cell bounding box
        double x_min = std::numeric_limits<double>::max();
        double x_max = -std::numeric_limits<double>::max();
        double y_min = std::numeric_limits<double>::max();
        double y_max = -std::numeric_limits<double>::max();

        for (std::size_t i = d_start; i < d_end; ++i) {
            auto ni = static_cast<std::size_t>(dst_idx[i]);
            double vx = dst_coords(ni, 0);
            double vy = dst_coords(ni, 1);
            x_min = std::min(x_min, vx);
            x_max = std::max(x_max, vx);
            y_min = std::min(y_min, vy);
            y_max = std::max(y_max, vy);
        }

        // Subgrid accumulation maps (source cell index -> accumulated weight)
        std::unordered_map<index_t, double> subgrid_weights;
        std::uint32_t valid_subpoints = 0;

        // 3. Generate and sample N x N subgrid points inside the bounding box
        for (std::uint32_t u = 0; u < N; ++u) {
            for (std::uint32_t v = 0; v < N; ++v) {
                double px = x_min + (static_cast<double>(u) + 0.5) * (x_max - x_min) / N;
                double py = y_min + (static_cast<double>(v) + 0.5) * (y_max - y_min) / N;

                double lon0 = px;
                double lat0 = py;
                if (csys == topology::CoordinateSystem::SphericalDeg) {
                    const double pi = 3.14159265358979323846;
                    lon0 = lon0 * pi / 180.0;
                    lat0 = lat0 * pi / 180.0;
                }

                double test_px = px;
                double test_py = py;
                if (is_spherical) {
                    test_px = 0.0;
                    test_py = 0.0;
                }

                // Query 8 nearest source cells to find containing elements
                constexpr int k_query = 8;
                Kokkos::View<decltype(ArborX::nearest(Point2{}, 1)) *, HostSpace> sub_queries("sub_q", 1);
                sub_queries(0) = ArborX::nearest(Point2{static_cast<float>(px), static_cast<float>(py)}, (int)std::min(n_src, (std::size_t)k_query));

                Kokkos::View<typename decltype(tree)::value_type *, HostSpace> sub_values("sub_v", 0);
                Kokkos::View<int *, HostSpace> sub_offsets_view("sub_o", 0);
                tree.query(host_exec, sub_queries, sub_values, sub_offsets_view);

                int begin = sub_offsets_view(0);
                int end = sub_offsets_view(1);
                int n_avail = end - begin;

                bool subpoint_mapped = false;

                // Find containing quadrilateral
                if (n_avail >= 4 && !subpoint_mapped) {
                    for (int a = 0; a < n_avail - 3 && !subpoint_mapped; ++a) {
                        for (int b = a + 1; b < n_avail - 2 && !subpoint_mapped; ++b) {
                            for (int c = b + 1; c < n_avail - 1 && !subpoint_mapped; ++c) {
                                for (int d = c + 1; d < n_avail && !subpoint_mapped; ++d) {
                                    std::size_t c0 = static_cast<std::size_t>(sub_values(begin + a).index);
                                    std::size_t c1 = static_cast<std::size_t>(sub_values(begin + b).index);
                                    std::size_t c2 = static_cast<std::size_t>(sub_values(begin + c).index);
                                    std::size_t c3 = static_cast<std::size_t>(sub_values(begin + d).index);

                                    Vec2 q0{src_cx(c0), src_cy(c0)};
                                    Vec2 q1{src_cx(c1), src_cy(c1)};
                                    Vec2 q2{src_cx(c2), src_cy(c2)};
                                    Vec2 q3{src_cx(c3), src_cy(c3)};

                                    if (is_spherical) {
                                        const double pi = 3.14159265358979323846;
                                        auto to_rad = [&](double deg) { return deg * pi / 180.0; };
                                        auto proj = [&](Vec2 q) {
                                            double mu, mv;
                                            double q_lon = q.x;
                                            double q_lat = q.y;
                                            if (csys == topology::CoordinateSystem::SphericalDeg) {
                                                q_lon = to_rad(q_lon);
                                                q_lat = to_rad(q_lat);
                                            }
                                            project_gnomonic(lon0, lat0, q_lon, q_lat, mu, mv);
                                            return Vec2{mu, mv};
                                        };
                                        q0 = proj(q0);
                                        q1 = proj(q1);
                                        q2 = proj(q2);
                                        q3 = proj(q3);
                                    }

                                    double xi_centroids = 0.0, eta_centroids = 0.0;
                                    if (map_to_reference_quad(test_px, test_py, q0, q1, q2, q3, xi_centroids, eta_centroids)) {
                                        xi_centroids = std::max(-1.0, std::min(1.0, xi_centroids));
                                        eta_centroids = std::max(-1.0, std::min(1.0, eta_centroids));

                                        subgrid_weights[c0] += 0.25 * (1.0 - xi_centroids) * (1.0 - eta_centroids);
                                        subgrid_weights[c1] += 0.25 * (1.0 + xi_centroids) * (1.0 - eta_centroids);
                                        subgrid_weights[c2] += 0.25 * (1.0 + xi_centroids) * (1.0 + eta_centroids);
                                        subgrid_weights[c3] += 0.25 * (1.0 - xi_centroids) * (1.0 + eta_centroids);

                                        subpoint_mapped = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // Find containing triangle fallback
                if (n_avail >= 3 && !subpoint_mapped) {
                    for (int a = 0; a < n_avail - 2 && !subpoint_mapped; ++a) {
                        for (int b = a + 1; b < n_avail - 1 && !subpoint_mapped; ++b) {
                            for (int c = b + 1; c < n_avail && !subpoint_mapped; ++c) {
                                std::size_t c0 = static_cast<std::size_t>(sub_values(begin + a).index);
                                std::size_t c1 = static_cast<std::size_t>(sub_values(begin + b).index);
                                std::size_t c2 = static_cast<std::size_t>(sub_values(begin + c).index);

                                Vec2 q0{src_cx(c0), src_cy(c0)};
                                Vec2 q1{src_cx(c1), src_cy(c1)};
                                Vec2 q2{src_cx(c2), src_cy(c2)};

                                if (is_spherical) {
                                    const double pi = 3.14159265358979323846;
                                    auto to_rad = [&](double deg) { return deg * pi / 180.0; };
                                    auto proj = [&](Vec2 q) {
                                        double mu, mv;
                                        double q_lon = q.x;
                                        double q_lat = q.y;
                                        if (csys == topology::CoordinateSystem::SphericalDeg) {
                                            q_lon = to_rad(q_lon);
                                            q_lat = to_rad(q_lat);
                                        }
                                        project_gnomonic(lon0, lat0, q_lon, q_lat, mu, mv);
                                        return Vec2{mu, mv};
                                    };
                                    q0 = proj(q0);
                                    q1 = proj(q1);
                                    q2 = proj(q2);
                                }

                                double l0 = 0.0, l1 = 0.0, l2 = 0.0;
                                if (barycentric_triangle(test_px, test_py, q0, q1, q2, l0, l1, l2)) {
                                    l0 = std::max(0.0, l0);
                                    l1 = std::max(0.0, l1);
                                    l2 = std::max(0.0, l2);
                                    double sum = l0 + l1 + l2;
                                    if (sum > 0.0) {
                                        l0 /= sum;
                                        l1 /= sum;
                                        l2 /= sum;
                                    } else {
                                        l0 = l1 = l2 = 1.0 / 3.0;
                                    }

                                    subgrid_weights[c0] += l0;
                                    subgrid_weights[c1] += l1;
                                    subgrid_weights[c2] += l2;

                                    subpoint_mapped = true;
                                }
                            }
                        }
                    }
                }

                if (subpoint_mapped) {
                    valid_subpoints++;
                }
            }
        }

        // 4. Threshold gating & normalization
        double fraction = static_cast<double>(valid_subpoints) / static_cast<double>(N2);
        if (fraction < config.budget_min_valid_fraction) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error("WeightGenerator::generate_budget: Destination cell " + std::to_string(j) + " lacks sufficient coverage.");
            }
            continue;
        }

        // Normalization factor
        double norm = 1.0 / static_cast<double>(valid_subpoints);
        for (const auto &[src_cell, accum_wt] : subgrid_weights) {
            weights_vec.push_back(accum_wt * norm);
            rows_vec.push_back(static_cast<index_t>(j));
            cols_vec.push_back(src_cell);
        }
    }

    // 5. Build and return InterpolationMatrix
    const std::size_t nnz = weights_vec.size();
    Kokkos::View<double *, MemorySpace> factor_list("factor_list", nnz);
    Kokkos::View<index_t *, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t *, MemorySpace> factor_col("factor_col", nnz);

    Kokkos::View<double *, Kokkos::HostSpace> h_factor_list("h_factor_list", nnz);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_factor_row("h_factor_row", nnz);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_factor_col("h_factor_col", nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k) = rows_vec[k];
        h_factor_col(k) = cols_vec[k];
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);

    Kokkos::View<double *, MemorySpace> frac_a("frac_a", n_src);
    Kokkos::View<double *, MemorySpace> frac_b("frac_b", n_dst);
    Kokkos::View<double *, MemorySpace> area_a("area_a", n_src);
    Kokkos::View<double *, MemorySpace> area_b("area_b", n_dst);

    Kokkos::deep_copy(frac_a, 1.0);
    Kokkos::deep_copy(frac_b, 1.0);
    Kokkos::deep_copy(area_a, 0.0); // ESMF bilinear/budget convention: area_a is 0

    auto dst_areas = compute_cell_areas_device(dst_mesh, false);
    Kokkos::deep_copy(area_b, dst_areas);

    return InterpolationMatrix<MemorySpace>(std::move(factor_list), std::move(factor_row), std::move(factor_col), std::move(frac_a),
                                            std::move(frac_b), std::move(area_a), std::move(area_b), n_src, n_dst);
}
```

- [ ] **Step 2: Run verification compilation**

```bash
docker exec helm-dev-env make -C /workspace/helm-project/libs/axis/build -j4
```
Expected: Compiles with zero errors.

- [ ] **Step 3: Commit changes**

```bash
git add libs/axis/src/solver/weight_generator.cpp
git commit -m "feat(axis): implement general mesh budget interpolation"
```

---

### Task 4: Implement Unit Tests for Budget Interpolation

**Files:**
- Create: `libs/axis/tests/test_budget.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt:75-88`

**Interfaces:**
- Consumes: `InterpolationMethod::Budget`, and `RegridConfig::budget_subgrid_size` and `RegridConfig::budget_min_valid_fraction` properties.

- [ ] **Step 1: Register test_budget.cpp in tests/CMakeLists.txt**

Append `test_budget.cpp` to the `AXIS_UNIT_TEST_SOURCES` variable in `libs/axis/tests/CMakeLists.txt` around line 85:
```cmake
    test_distributed_weight_generator.cpp
    test_bicubic.cpp
    test_budget.cpp
)
```

- [ ] **Step 2: Write test_budget.cpp file content**

Create a new file `libs/axis/tests/test_budget.cpp`:
```cpp
// SPDX-License-Identifier: Apache-2.0
// AXIS unit tests for InterpolationMethod::Budget

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static topology::UnstructuredMesh<MemSpace> make_simple_cartesian_mesh(std::size_t n, double size) {
    const double dx = size / static_cast<double>(n);
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }
    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = static_cast<double>(i) * dx;
            cry(i + j * nc) = static_cast<double>(j) * dx;
        }
    }
    return grid.to_unstructured(crx, cry);
}

TEST(BudgetInterpolationTest, ConstantFieldPreservation) {
    auto src = make_simple_cartesian_mesh(4, 1.0);
    auto dst = make_simple_cartesian_mesh(2, 1.0);

    RegridConfig config;
    config.method = InterpolationMethod::Budget;
    config.budget_subgrid_size = 5;
    config.budget_min_valid_fraction = 0.5;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src, dst, config);

    Kokkos::View<double *, MemSpace> src_field("src", src.n_cells());
    Kokkos::deep_copy(src_field, 1.0);

    Kokkos::View<double *, MemSpace> dst_field("dst", dst.n_cells());
    solver::apply(matrix, src_field, dst_field);

    for (std::size_t j = 0; j < dst.n_cells(); ++j) {
        EXPECT_NEAR(dst_field(j), 1.0, 1e-12);
    }
}

TEST(BudgetInterpolationTest, BilinearParityWithSubgridSize1) {
    auto src = make_simple_cartesian_mesh(4, 1.0);
    auto dst = make_simple_cartesian_mesh(2, 1.0);

    RegridConfig config_bilinear;
    config_bilinear.method = InterpolationMethod::Bilinear;
    auto matrix_bilin = solver::WeightGenerator::generate<MemSpace>(src, dst, config_bilinear);

    RegridConfig config_budget;
    config_budget.method = InterpolationMethod::Budget;
    config_budget.budget_subgrid_size = 1; // 1x1 subgrid behaves like pure bilinear
    auto matrix_budget = solver::WeightGenerator::generate<MemSpace>(src, dst, config_budget);

    ASSERT_EQ(matrix_bilin.nnz(), matrix_budget.nnz());

    auto b_row = matrix_bilin.factor_row_view();
    auto b_col = matrix_bilin.factor_col_view();
    auto b_val = matrix_bilin.factor_list_view();

    auto m_row = matrix_budget.factor_row_view();
    auto m_col = matrix_budget.factor_col_view();
    auto m_val = matrix_budget.factor_list_view();

    for (std::size_t k = 0; k < matrix_bilin.nnz(); ++k) {
        EXPECT_EQ(b_row(k), m_row(k));
        EXPECT_EQ(b_col(k), m_col(k));
        EXPECT_NEAR(b_val(k), m_val(k), 1e-12);
    }
}

} // namespace axis::test
```

- [ ] **Step 3: Build and execute tests in Docker**

Configure the build and run unit tests inside Docker:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake -DBUILD_TESTING=ON .. && make -j4 && ctest -R BudgetInterpolationTest"
```
Expected: The two test cases `ConstantFieldPreservation` and `BilinearParityWithSubgridSize1` compile and PASS.

- [ ] **Step 4: Commit changes**

```bash
git add libs/axis/tests/CMakeLists.txt libs/axis/tests/test_budget.cpp
git commit -m "test(axis): add unit tests for budget interpolation"
```

---

### Task 5: Implement Property-Based Tests for Budget Interpolation

**Files:**
- Create: `libs/axis/tests/prop_budget.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt:155-170`

**Interfaces:**
- Consumes: RapidCheck frameworks, random unstructured grids.

- [ ] **Step 1: Register prop_budget.cpp in tests/CMakeLists.txt**

Append `prop_budget.cpp` to the `AXIS_PROPERTY_TEST_SOURCES` variable in `libs/axis/tests/CMakeLists.txt` around line 168:
```cmake
        prop_bilinear_rect_unmapped.cpp
        prop_bilinear_rect_cell_location.cpp
        prop_budget.cpp
    )
```

- [ ] **Step 2: Write prop_budget.cpp file content**

Create a new file `libs/axis/tests/prop_budget.cpp`:
```cpp
// SPDX-License-Identifier: Apache-2.0
// Property-based testing for InterpolationMethod::Budget

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static topology::UnstructuredMesh<MemSpace> make_prop_mesh(std::size_t n, double size) {
    const double dx = size / static_cast<double>(n);
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }
    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = static_cast<double>(i) * dx;
            cry(i + j * nc) = static_cast<double>(j) * dx;
        }
    }
    return grid.to_unstructured(crx, cry);
}

RC_GTEST_PROP(PropBudget, RowSumsAreUnityAndIndicesAreInBounds, ()) {
    // Generate random grid size between 2 and 6
    const auto src_size = *rc::gen::inRange<std::size_t>(2, 7);
    const auto dst_size = *rc::gen::inRange<std::size_t>(2, 7);

    auto src = make_prop_mesh(src_size, 1.0);
    auto dst = make_prop_mesh(dst_size, 1.0);

    RegridConfig config;
    config.method = InterpolationMethod::Budget;
    config.budget_subgrid_size = *rc::gen::inRange<std::uint32_t>(2, 6);
    config.budget_min_valid_fraction = 0.5;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src, dst, config);

    const auto row = matrix.factor_row_view();
    const auto col = matrix.factor_col_view();
    const auto val = matrix.factor_list_view();
    const auto nnz = matrix.nnz();

    std::vector<double> row_sums(dst.n_cells(), 0.0);

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(row(k) < dst.n_cells());
        RC_ASSERT(col(k) < src.n_cells());
        row_sums[row(k)] += val(k);
    }

    for (std::size_t j = 0; j < dst.n_cells(); ++j) {
        if (row_sums[j] > 0.0) {
            RC_ASSERT(std::abs(row_sums[j] - 1.0) < 1e-12);
        }
    }
}

} // namespace axis::test
```

- [ ] **Step 3: Run property tests inside Docker**

```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && make -j4 && ctest -R PropBudget"
```
Expected: The property test executes multiple randomized trials successfully and PASSES.

- [ ] **Step 4: Commit changes**

```bash
git add libs/axis/tests/CMakeLists.txt libs/axis/tests/prop_budget.cpp
git commit -m "test(axis): add property tests for budget interpolation"
```
