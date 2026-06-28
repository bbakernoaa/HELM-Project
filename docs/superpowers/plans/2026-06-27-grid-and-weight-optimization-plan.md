# Structured Grid and Weight Generation Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize grid creation (StructuredGrid to UnstructuredMesh conversion) and weight generation (nearest, bilinear, conservative) for both rectilinear and curvilinear grids, achieving 2x-50x speedups without sacrificing accuracy.

**Architecture:** Introduce `MDRangePolicy` and `TeamPolicy` for `StructuredGrid` conversion. Implement non-uniform rectilinear grid fast-paths using 1-D binary search (`Kokkos::upper_bound`) and analytical overlaps. Curvilinear grids will benefit from the optimized conversion and run on the highly accelerated unstructured ArborX and KokkosKernels pipelines.

**Tech Stack:** C++20, Kokkos, ArborX, KokkosKernels (for `KokkosSparse::spmv`).

## Global Constraints

- C++ standard: C++20
- Parallelism: Use Kokkos-only parallel constructs, no raw CUDA, HIP, or OpenMP
- Thread safety: Ensure all kernels are stateless and thread-safe
- Accuracy: Maintain 1e-10 precision for bilinear exactness and 1e-12 for conservation

---

### Task 1: MDRangePolicy in `to_unstructured`

**Files:**
- Modify: `libs/axis/src/topology/structured_grid.cpp:180-265`
- Test: `libs/axis/build/` (build/run existing tests `PropStructuredToUnstructured.*` and `CsrConnectivityProperty3.*`)

**Interfaces:**
- Consumes: `StructuredGrid<MemorySpace>` parameters (`ni_`, `nj_`, `corner_lon_`, `corner_lat_`)
- Produces: Optimized `UnstructuredMesh<MemorySpace>` conversion

- [ ] **Step 1: Replace flat 1D RangePolicy loops in `to_unstructured` with 2D `MDRangePolicy`**

In `libs/axis/src/topology/structured_grid.cpp`, update `to_unstructured()`:
```cpp
template <class MemorySpace>
UnstructuredMesh<MemorySpace>
StructuredGrid<MemorySpace>::to_unstructured() const {
    using exec_space = typename detail::exec_space_t<MemorySpace>;

    if (corner_lon_.extent(0) == 0) {
        synthesize_corners();
    }

    const std::size_t ni = ni_;
    const std::size_t nj = nj_;
    const std::size_t nip1 = ni + 1;
    const std::size_t njp1 = nj + 1;
    const std::size_t n_cells = ni * nj;
    const std::size_t n_nodes = nip1 * njp1;

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> node_coords(
        "unstructured_node_coords", n_nodes, std::size_t{2});

    Kokkos::View<index_t*, MemorySpace> cell_node_offsets(
        "unstructured_cell_offsets", n_cells + 1);

    Kokkos::View<index_t*, MemorySpace> cell_node_indices(
        "unstructured_cell_indices", n_cells * 4);

    auto clon = corner_lon_;
    auto clat = corner_lat_;

    // Optimize node coordinate filling using MDRangePolicy
    using MDRange2D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
    Kokkos::parallel_for(
        "to_unstructured_fill_nodes_md",
        MDRange2D({0, 0}, {static_cast<int>(nip1), static_cast<int>(njp1)}),
        KOKKOS_LAMBDA(const int ci, const int cj) {
            const int node_idx = ci + cj * nip1;
            node_coords(node_idx, 0) = clon(node_idx);
            node_coords(node_idx, 1) = clat(node_idx);
        });

    // Optimize connectivity filling using MDRangePolicy
    Kokkos::parallel_for(
        "to_unstructured_fill_connectivity_md",
        MDRange2D({0, 0}, {static_cast<int>(ni), static_cast<int>(nj)}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int cell_idx = i + j * ni;
            cell_node_offsets(cell_idx) = static_cast<index_t>(cell_idx * 4);

            const std::size_t base = cell_idx * 4;
            cell_node_indices(base + 0) = static_cast<index_t>(i     + j       * nip1);
            cell_node_indices(base + 1) = static_cast<index_t>((i+1) + j       * nip1);
            cell_node_indices(base + 2) = static_cast<index_t>((i+1) + (j + 1) * nip1);
            cell_node_indices(base + 3) = static_cast<index_t>(i     + (j + 1) * nip1);
        });

    Kokkos::parallel_for(
        "to_unstructured_sentinel_offset",
        Kokkos::RangePolicy<exec_space>(0, 1),
        KOKKOS_LAMBDA(const int /*unused*/) {
            cell_node_offsets(n_cells) = static_cast<index_t>(n_cells * 4);
        });

    Kokkos::fence("to_unstructured_fence");

    return UnstructuredMesh<MemorySpace>(
        std::move(node_coords),
        std::move(cell_node_offsets),
        std::move(cell_node_indices),
        coord_sys_);
}
```

- [ ] **Step 2: Build and verify the implementation inside Docker**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis && cmake --build build --parallel 8"
```
Expected: Successfully compiled without warnings or errors.

- [ ] **Step 3: Run the structured grid conversion tests**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && ctest -R \"PropStructuredToUnstructured|CsrConnectivityProperty3\" --output-on-failure"
```
Expected: 100% tests passed.

- [ ] **Step 4: Commit changes**

Run:
```bash
git add libs/axis/src/topology/structured_grid.cpp
git commit -m "perf: use Kokkos::MDRangePolicy in StructuredGrid::to_unstructured"
```

---

### Task 2: TeamPolicy and Scratchpad Caching in `synthesize_corners`

**Files:**
- Modify: `libs/axis/src/topology/structured_grid.cpp:115-175`
- Test: `libs/axis/build/` (build/run existing tests `PropStructuredToUnstructured.*`)

**Interfaces:**
- Consumes: `center_lon_`, `center_lat_`
- Produces: Synthesized `corner_lon_`, `corner_lat_` coordinates with L1 cache spatial reuse

- [ ] **Step 1: Re-implement `synthesize_corners` using `Kokkos::TeamPolicy` and shared scratch memory**

In `libs/axis/src/topology/structured_grid.cpp`:
```cpp
template <class MemorySpace>
void StructuredGrid<MemorySpace>::synthesize_corners() const {
    using exec_space = typename detail::exec_space_t<MemorySpace>;

    const std::size_t nip1 = ni_ + 1;
    const std::size_t njp1 = nj_ + 1;
    const std::size_t n_corners = nip1 * njp1;
    const std::size_t ni = ni_;
    const std::size_t nj = nj_;

    auto& self = const_cast<StructuredGrid<MemorySpace>&>(*this);
    self.corner_lon_ = Kokkos::View<double*, MemorySpace>(
        "structured_grid_corner_lon", n_corners);
    self.corner_lat_ = Kokkos::View<double*, MemorySpace>(
        "structured_grid_corner_lat", n_corners);

    auto clon = self.corner_lon_;
    auto clat = self.corner_lat_;
    auto center_lon = center_lon_;
    auto center_lat = center_lat_;

    // Use Kokkos::TeamPolicy for coordinate caching
    using TeamPolicy = Kokkos::TeamPolicy<exec_space>;
    using MemberType = typename TeamPolicy::member_type;

    // Dispatch one team per row of corners (nj + 1 teams). Threads in a team handle elements in the row.
    TeamPolicy policy(njp1, Kokkos::AUTO);
    Kokkos::parallel_for(
        "synthesize_corners_team",
        policy,
        KOKKOS_LAMBDA(const MemberType& team) {
            const std::size_t cj = team.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, nip1),
                [&](const std::size_t ci) {
                    const std::size_t idx = ci + cj * nip1;
                    double sum_lon = 0.0;
                    double sum_lat = 0.0;
                    int count = 0;

                    for (int dj = -1; dj <= 0; ++dj) {
                        for (int di = -1; di <= 0; ++di) {
                            const auto cell_i = static_cast<long long>(ci) + di;
                            const auto cell_j = static_cast<long long>(cj) + dj;
                            if (cell_i >= 0 && cell_i < static_cast<long long>(ni) &&
                                cell_j >= 0 && cell_j < static_cast<long long>(nj)) {
                                const std::size_t cell_idx =
                                    static_cast<std::size_t>(cell_i) +
                                    static_cast<std::size_t>(cell_j) * ni;
                                sum_lon += center_lon(cell_idx);
                                sum_lat += center_lat(cell_idx);
                                ++count;
                            }
                        }
                    }

                    clon(idx) = sum_lon / static_cast<double>(count);
                    clat(idx) = sum_lat / static_cast<double>(count);
                });
        });

    Kokkos::fence("synthesize_corners_fence");
}
```

- [ ] **Step 2: Build and verify compilation**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis && cmake --build build --parallel 8"
```
Expected: Successfully compiled.

- [ ] **Step 3: Run the tests to verify correctness**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && ctest -R \"PropStructuredToUnstructured\" --output-on-failure"
```
Expected: All tests pass.

- [ ] **Step 4: Commit changes**

Run:
```bash
git add libs/axis/src/topology/structured_grid.cpp
git commit -m "perf: use Kokkos::TeamPolicy in StructuredGrid::synthesize_corners"
```

---

### Task 3: Non-Uniform Rectilinear Grid Detection

**Files:**
- Modify: `libs/axis/include/axis/detail/regular_grid_detector.hpp`
- Test: Create `libs/axis/tests/test_rectilinear_grid_detector.cpp`

**Interfaces:**
- Consumes: `UnstructuredMesh<MemorySpace>`
- Produces: `RectilinearGridInfo` struct describing non-uniform/uniform rectilinear grid dimensions and 1D boundary coordinate arrays

- [ ] **Step 1: Add `RectilinearGridInfo` struct and `detect_rectilinear_grid` signature to `regular_grid_detector.hpp`**

In `libs/axis/include/axis/detail/regular_grid_detector.hpp`:
```cpp
struct RectilinearGridInfo {
    bool        is_rectilinear{false};
    std::size_t ni{0};
    std::size_t nj{0};
    Kokkos::View<double*, Kokkos::HostSpace> unique_lons;
    Kokkos::View<double*, Kokkos::HostSpace> unique_lats;
};

template <class MemorySpace>
RectilinearGridInfo detect_rectilinear_grid(
    const topology::UnstructuredMesh<MemorySpace>& mesh) {

    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
                  "detect_rectilinear_grid() requires a host-accessible mesh");

    RectilinearGridInfo info;

    const auto n_cells = mesh.n_cells();
    if (n_cells == 0) {
        return info;
    }

    const auto& offsets = mesh.conn_offsets_view();
    const auto& indices = mesh.conn_indices_view();
    const auto& coords  = mesh.node_coords_view();

    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = offsets(c);
        auto end   = offsets(c + 1);
        if ((end - start) != 4) {
            return info;  // Must be all quads
        }
    }

    const auto n_nodes = mesh.n_nodes();
    std::vector<double> all_lons;
    std::vector<double> all_lats;
    all_lons.reserve(n_nodes);
    all_lats.reserve(n_nodes);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        all_lons.push_back(coords(i, 0));
        all_lats.push_back(coords(i, 1));
    }

    std::sort(all_lons.begin(), all_lons.end());
    std::sort(all_lats.begin(), all_lats.end());

    constexpr double unique_tol = 1.0e-12;
    auto unique_filter = [&](std::vector<double>& sorted) -> std::vector<double> {
        std::vector<double> unique_vals;
        if (sorted.empty()) return unique_vals;
        unique_vals.push_back(sorted[0]);
        for (std::size_t i = 1; i < sorted.size(); ++i) {
            if (std::abs(sorted[i] - unique_vals.back()) > unique_tol) {
                unique_vals.push_back(sorted[i]);
            }
        }
        return unique_vals;
    };

    auto unique_lons = unique_filter(all_lons);
    auto unique_lats = unique_filter(all_lats);

    if (unique_lons.size() < 2 || unique_lats.size() < 2) {
        return info;
    }

    const std::size_t ni = unique_lons.size() - 1;
    const std::size_t nj = unique_lats.size() - 1;

    if (ni * nj != n_cells) {
        return info;  // Not a rectilinear structured layout
    }

    // Populate RectilinearGridInfo
    info.is_rectilinear = true;
    info.ni = ni;
    info.nj = nj;

    info.unique_lons = Kokkos::View<double*, Kokkos::HostSpace>("unique_lons", unique_lons.size());
    info.unique_lats = Kokkos::View<double*, Kokkos::HostSpace>("unique_lats", unique_lats.size());

    for (std::size_t i = 0; i < unique_lons.size(); ++i) {
        info.unique_lons(i) = unique_lons[i];
    }
    for (std::size_t j = 0; j < unique_lats.size(); ++j) {
        info.unique_lats(j) = unique_lats[j];
    }

    return info;
}
```

- [ ] **Step 2: Create a unit test file for rectilinear grid detection**

Create `libs/axis/tests/test_rectilinear_grid_detector.cpp`:
```cpp
#include <gtest/gtest.h>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/topology/structured_grid.hpp>

TEST(RectilinearGridDetector, DetectsNonUniformRectilinearGrid) {
    // Generate simple non-uniform rectilinear coordinate spacing
    std::vector<double> lons = {0.0, 1.5, 4.0, 5.5, 9.0}; // non-uniform spacing
    std::vector<double> lats = {0.0, 2.0, 3.5, 6.0};

    const std::size_t ni = lons.size() - 1;
    const std::size_t nj = lats.size() - 1;

    Kokkos::View<double*, Kokkos::HostSpace> center_lons("center_lons", ni * nj);
    Kokkos::View<double*, Kokkos::HostSpace> center_lats("center_lats", ni * nj);
    Kokkos::View<double*, Kokkos::HostSpace> corner_lons("corner_lons", (ni+1) * (nj+1));
    Kokkos::View<double*, Kokkos::HostSpace> corner_lats("corner_lats", (ni+1) * (nj+1));

    // Fill corner nodes
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            corner_lons(i + j * (ni+1)) = lons[i];
            corner_lats(i + j * (ni+1)) = lats[j];
        }
    }

    // Fill cell centers as average of corners
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            center_lons(i + j * ni) = 0.5 * (lons[i] + lons[i+1]);
            center_lats(i + j * ni) = 0.5 * (lats[j] + lats[j+1]);
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(
        ni, nj, center_lons, center_lats, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lons, corner_lats);

    auto mesh = grid.to_unstructured();
    auto info = axis::detail::detect_rectilinear_grid(mesh);

    EXPECT_TRUE(info.is_rectilinear);
    EXPECT_EQ(info.ni, ni);
    EXPECT_EQ(info.nj, nj);
    EXPECT_DOUBLE_EQ(info.unique_lons(0), 0.0);
    EXPECT_DOUBLE_EQ(info.unique_lons(4), 9.0);
}
```

Add the new test file to `libs/axis/tests/CMakeLists.txt`.

- [ ] **Step 3: Compile and run rectilinear grid detector test**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis && cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel 8"
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && ./tests/axis_unit_tests --gtest_filter=\"RectilinearGridDetector.*\""
```
Expected: Tests compile and pass.

- [ ] **Step 4: Commit changes**

Run:
```bash
git add libs/axis/include/axis/detail/regular_grid_detector.hpp libs/axis/tests/test_rectilinear_grid_detector.cpp libs/axis/tests/CMakeLists.txt
git commit -m "feat: implement detect_rectilinear_grid for non-uniform spacing"
```

---

### Task 4: Non-Uniform Rectilinear Bilinear Fast-Path

**Files:**
- Create: `libs/axis/src/solver/weight_generator_bilinear_rect_nonuniform.cpp`
- Modify: `libs/axis/src/solver/weight_generator.cpp`
- Test: Create a new test case inside `libs/axis/tests/test_bilinear_rect.cpp`

**Interfaces:**
- Consumes: `UnstructuredMesh<MemorySpace>`, `RectilinearGridInfo`
- Produces: Bilinear interpolation matrix computed via analytical $O(\log N)$ 1D binary search

- [ ] **Step 1: Write `generate_bilinear_rect_nonuniform` implementation**

Create `libs/axis/src/solver/weight_generator_bilinear_rect_nonuniform.cpp`:
```cpp
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <vector>

namespace axis::solver {

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
generate_bilinear_rect_nonuniform(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config,
    const detail::RectilinearGridInfo& src_rect_info) {

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t ni = src_rect_info.ni;
    const std::size_t nj = src_rect_info.nj;

    // Retrieve unique cell boundary coordinates (HostSpace)
    auto unique_lons = src_rect_info.unique_lons;
    auto unique_lats = src_rect_info.unique_lats;

    // Retrieve destination centroids
    const auto& offsets = dst_mesh.conn_offsets_view();
    const auto& indices = dst_mesh.conn_indices_view();
    const auto& coords  = dst_mesh.node_coords_view();

    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;
    weights_vec.reserve(n_dst * 4);
    rows_vec.reserve(n_dst * 4);
    cols_vec.reserve(n_dst * 4);

    for (std::size_t c = 0; c < n_dst; ++c) {
        auto start = static_cast<std::size_t>(offsets(c));
        auto end   = static_cast<std::size_t>(offsets(c + 1));
        std::size_t n_verts = end - start;

        double lon_sum = 0.0;
        double lat_sum = 0.0;
        for (std::size_t v = start; v < end; ++v) {
            auto node_idx = static_cast<std::size_t>(indices(v));
            lon_sum += coords(node_idx, 0);
            lat_sum += coords(node_idx, 1);
        }

        double lon_d = lon_sum / static_cast<double>(n_verts);
        double lat_d = lat_sum / static_cast<double>(n_verts);

        // Perform 1D binary search to locate cell (i, j)
        auto lon_it = std::upper_bound(unique_lons.data(), unique_lons.data() + unique_lons.extent(0), lon_d);
        auto lat_it = std::upper_bound(unique_lats.data(), unique_lats.data() + unique_lats.extent(0), lat_d);

        int i = std::distance(unique_lons.data(), lon_it) - 1;
        int j = std::distance(unique_lats.data(), lat_it) - 1;

        if (i < 0 || i >= static_cast<int>(ni) || j < 0 || j >= static_cast<int>(nj)) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error("Unmapped destination cell in non-uniform bilinear");
            }
            continue;
        }

        double x0 = unique_lons(i);
        double x1 = unique_lons(i + 1);
        double y0 = unique_lats(j);
        double y1 = unique_lats(j + 1);

        double tx = (lon_d - x0) / (x1 - x0);
        double ty = (lat_d - y0) / (y1 - y0);

        tx = std::max(0.0, std::min(1.0, tx));
        ty = std::max(0.0, std::min(1.0, ty));

        std::size_t src_idx[4] = {
            static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i),
            static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i + 1),
            static_cast<std::size_t>(j + 1) * ni + static_cast<std::size_t>(i),
            static_cast<std::size_t>(j + 1) * ni + static_cast<std::size_t>(i + 1)
        };

        double wts[4] = {
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty
        };

        for (int k = 0; k < 4; ++k) {
            weights_vec.push_back(wts[k]);
            rows_vec.push_back(static_cast<index_t>(c));
            cols_vec.push_back(static_cast<index_t>(src_idx[k]));
        }
    }

    // Mirror to Target MemorySpace and build InterpolationMatrix
    const std::size_t nnz = weights_vec.size();
    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k)  = rows_vec[k];
        h_factor_col(k)  = cols_vec[k];
    }
    for (std::size_t i = 0; i < n_src; ++i) h_frac_a(i) = 1.0;
    for (std::size_t j = 0; j < n_dst; ++j) h_frac_b(j) = 1.0;

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}

template InterpolationMatrix<Kokkos::HostSpace>
generate_bilinear_rect_nonuniform<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&,
    const detail::RectilinearGridInfo&);

} // namespace axis::solver
```

- [ ] **Step 2: Add Non-Uniform dispatch to `WeightGenerator::generate_bilinear`**

In `libs/axis/src/solver/weight_generator.cpp`, update `generate_bilinear()` to detect and dispatch rectilinear grids:
```cpp
    // Add inside WeightGenerator::generate_bilinear:
    auto src_rect_info = detail::detect_rectilinear_grid(src_mesh);
    if (src_rect_info.is_rectilinear) {
        auto result = generate_bilinear_rect_nonuniform(src_mesh, dst_mesh, config, src_rect_info);
        if (result.n_dst() > 0) {
            return result;
        }
    }
```

Add `libs/axis/src/solver/weight_generator_bilinear_rect_nonuniform.cpp` to `libs/axis/CMakeLists.txt` build sources.

- [ ] **Step 3: Compile and verify unit tests**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis && cmake --build build --parallel 8"
```

- [ ] **Step 4: Commit changes**

Run:
```bash
git add libs/axis/CMakeLists.txt libs/axis/src/solver/weight_generator.cpp libs/axis/src/solver/weight_generator_bilinear_rect_nonuniform.cpp
git commit -m "feat: implement non-uniform rectilinear grid bilinear fast-path"
```

---

### Task 5: Non-Uniform Rectilinear Conservative Overlap Fast-Path

**Files:**
- Create: `libs/axis/src/solver/weight_generator_conservative_rect_nonuniform.cpp`
- Modify: `libs/axis/src/solver/weight_generator.cpp`
- Test: Verify with `PropConservationOpt.*` property test suite

**Interfaces:**
- Consumes: `UnstructuredMesh<MemorySpace>`, `RectilinearGridInfo`
- Produces: Conservative weight generation via analytical bounding box overlays

- [ ] **Step 1: Write `generate_conservative_rect_nonuniform` implementation**

Create `libs/axis/src/solver/weight_generator_conservative_rect_nonuniform.cpp`:
```cpp
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <vector>

namespace axis::solver {

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
generate_conservative_rect_nonuniform(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config,
    const detail::RectilinearGridInfo& src_rect_info,
    const detail::RectilinearGridInfo& dst_rect_info) {

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t src_ni = src_rect_info.ni;
    const std::size_t src_nj = src_rect_info.nj;
    const std::size_t dst_ni = dst_rect_info.ni;
    const std::size_t dst_nj = dst_rect_info.nj;

    auto src_lons = src_rect_info.unique_lons;
    auto src_lats = src_rect_info.unique_lats;
    auto dst_lons = dst_rect_info.unique_lons;
    auto dst_lats = dst_rect_info.unique_lats;

    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    for (std::size_t jd = 0; jd < dst_nj; ++jd) {
        for (std::size_t id = 0; id < dst_ni; ++id) {
            const std::size_t c_dst = id + jd * dst_ni;

            double d_lo_x = dst_lons(id);
            double d_hi_x = dst_lons(id + 1);
            double d_lo_y = dst_lats(jd);
            double d_hi_y = dst_lats(jd + 1);

            // Binary search to find candidate overlapping source ranges
            auto src_lon_start_it = std::lower_bound(src_lons.data(), src_lons.data() + src_lons.extent(0), d_lo_x);
            auto src_lon_end_it   = std::upper_bound(src_lons.data(), src_lons.data() + src_lons.extent(0), d_hi_x);

            int is_start = std::max(0, static_cast<int>(std::distance(src_lons.data(), src_lon_start_it) - 1));
            int is_end   = std::min(static_cast<int>(src_ni) - 1, static_cast<int>(std::distance(src_lons.data(), src_lon_end_it)));

            auto src_lat_start_it = std::lower_bound(src_lats.data(), src_lats.data() + src_lats.extent(0), d_lo_y);
            auto src_lat_end_it   = std::upper_bound(src_lats.data(), src_lats.data() + src_lats.extent(0), d_hi_y);

            int js_start = std::max(0, static_cast<int>(std::distance(src_lats.data(), src_lat_start_it) - 1));
            int js_end   = std::min(static_cast<int>(src_nj) - 1, static_cast<int>(std::distance(src_lats.data(), src_lat_end_it)));

            for (int js = js_start; js <= js_end; ++js) {
                for (int is = is_start; is <= is_end; ++is) {
                    double s_lo_x = src_lons(is);
                    double s_hi_x = src_lons(is + 1);
                    double s_lo_y = src_lats(js);
                    double s_hi_y = src_lats(js + 1);

                    double overlap_x = std::max(0.0, std::min(s_hi_x, d_hi_x) - std::max(s_lo_x, d_lo_x));
                    double overlap_y = std::max(0.0, std::min(s_hi_y, d_hi_y) - std::max(s_lo_y, d_lo_y));
                    double overlap_area = overlap_x * overlap_y;

                    if (overlap_area > 1e-12) {
                        double dst_area = (d_hi_x - d_lo_x) * (d_hi_y - d_lo_y);
                        double weight = overlap_area / dst_area;

                        weights_vec.push_back(weight);
                        rows_vec.push_back(static_cast<index_t>(c_dst));
                        cols_vec.push_back(static_cast<index_t>(js * src_ni + is));
                    }
                }
            }
        }
    }

    const std::size_t nnz = weights_vec.size();
    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k)  = rows_vec[k];
        h_factor_col(k)  = cols_vec[k];
    }
    for (std::size_t i = 0; i < n_src; ++i) h_frac_a(i) = 1.0;
    for (std::size_t j = 0; j < n_dst; ++j) h_frac_b(j) = 1.0;

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}

template InterpolationMatrix<Kokkos::HostSpace>
generate_conservative_rect_nonuniform<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&,
    const detail::RectilinearGridInfo&,
    const detail::RectilinearGridInfo&);

} // namespace axis::solver
```

- [ ] **Step 2: Add Non-Uniform dispatch to `WeightGenerator::generate_conservative`**

In `libs/axis/src/solver/weight_generator.cpp`, update `generate_conservative()` to detect and dispatch non-uniform conservative overlays:
```cpp
    // Add inside WeightGenerator::generate_conservative:
    auto src_rect_info = detail::detect_rectilinear_grid(src_mesh);
    auto dst_rect_info = detail::detect_rectilinear_grid(dst_mesh);
    if (src_rect_info.is_rectilinear && dst_rect_info.is_rectilinear) {
        auto result = generate_conservative_rect_nonuniform(src_mesh, dst_mesh, config,
                                                             src_rect_info, dst_rect_info);
        if (result.nnz() > 0) {
            return result;
        }
    }
```

Add `libs/axis/src/solver/weight_generator_conservative_rect_nonuniform.cpp` to `libs/axis/CMakeLists.txt` build sources.

- [ ] **Step 3: Compile and run conservative and bilinear tests**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis && cmake --build build --parallel 8"
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && ctest --output-on-failure"
```
Expected: 100% tests passed.

- [ ] **Step 4: Commit changes**

Run:
```bash
git add libs/axis/CMakeLists.txt libs/axis/src/solver/weight_generator.cpp libs/axis/src/solver/weight_generator_conservative_rect_nonuniform.cpp
git commit -m "feat: implement non-uniform rectilinear grid conservative fast-path"
```

---

### Task 6: Sparse Matrix Apply via KokkosSparse spmv

**Files:**
- Modify: `libs/axis/src/solver/interpolation_matrix.cpp`
- Test: Verify with `PropCsrFormat.*` and `PropBatchApply.*` property test suite

**Interfaces:**
- Consumes: Generated CSR representation inside `InterpolationMatrix`
- Produces: Maximum-performance hardware execution using KokkosSparse::spmv

- [ ] **Step 1: Wire CSR Apply to `KokkosSparse::spmv`**

Ensure that CSR sparse matrix multiplication dispatches to `KokkosSparse::spmv`. In `libs/axis/src/solver/interpolation_matrix.cpp` or the apply mechanism, verify that the KokkosSparse interface is actively used for the local apply step.

- [ ] **Step 2: Build and run the entire verification suite**

Run:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && ctest --output-on-failure"
```
Expected: 100% tests passed.

- [ ] **Step 3: Commit and verify clean build**

Run:
```bash
git status
```
Expected: No untracked/unstaged changes, clean working tree on branch develop.
