# AXIS Python Offloaded Core Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offload heavy structured-to-unstructured grid conversions, vector re-normalization & wind rotation, and MPAS unstructured poly triangulation entirely to optimized C++ parallel-on-device kernels.

**Architecture:**
1. Bind C++ `StructuredGrid` class directly via nanobind and integrate it into `grid.py` curvilinear and cubed-sphere fallbacks.
2. Implement C++ `vector_apply_rotate` for combined wind rotation and SpMV.
3. Implement C++ `triangulate_poly_cells` to perform fast cell-polygon triangulation.

**Tech Stack:** C++20, Kokkos, nanobind, Python, numpy, xarray, pytest.

## Global Constraints
- **Zero-Copy Performance**: All NumPy views must address raw pointers directly.
- **Kokkos HostSpace Compliance**: Executed entirely on Kokkos::HostSpace.
- **TDD / Verification first**: Write tests for each feature and ensure they pass completely.

---

### Task 1: Bind `StructuredGrid` and Offload Python Curvilinear Conversion

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/grid.py`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_curvilinear_to_unstructured.py`

**Interfaces:**
- Consumes: C++ `axis::topology::StructuredGrid`
- Produces: `axis_py.StructuredGrid`, C++-backed curvilinear mesh generation in `grid.py`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_curvilinear_to_unstructured.py`:
```python
import pytest
import numpy as np
import axis
from axis import axis_py

def test_structured_grid_binding():
    cx = np.array([0.5, 1.5, 0.5, 1.5])
    cy = np.array([0.5, 0.5, 1.5, 1.5])
    
    grid = axis_py.StructuredGrid(2, 2, cx, cy)
    mesh = grid.to_unstructured()
    assert mesh.n_cells == 4
    assert mesh.n_nodes == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_curvilinear_to_unstructured.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'StructuredGrid'"

- [ ] **Step 3: Implement C++ nanobind exposure of `StructuredGrid`**

In `libs/axis/python/axis_py.cpp`:
Add binding definition near the end of `NB_MODULE(axis_py, m)`:
```cpp
    // ─── StructuredGrid Class ────────────────────────────────────────────────
    nb::class_<axis::topology::StructuredGrid<Kokkos::HostSpace>>(m, "StructuredGrid")
        .def(
            nb::init([](std::size_t ni, std::size_t nj, nb::ndarray<const double, nb::ndim<1>> cx, nb::ndarray<const double, nb::ndim<1>> cy) {
                ensure_kokkos();
                Kokkos::View<double *, Kokkos::HostSpace> cx_v(const_cast<double*>(cx.data()), cx.shape(0));
                Kokkos::View<double *, Kokkos::HostSpace> cy_v(const_cast<double*>(cy.data()), cy.shape(0));
                return axis::topology::StructuredGrid<Kokkos::HostSpace>(ni, nj, cx_v, cy_v, axis::topology::CoordinateSystem::SphericalDeg);
            }),
            "ni"_a, "nj"_a, "cx"_a, "cy"_a)
        .def(
            "set_corners",
            [](axis::topology::StructuredGrid<Kokkos::HostSpace> &grid, nb::ndarray<const double, nb::ndim<1>> crx,
               nb::ndarray<const double, nb::ndim<1>> cry) {
                Kokkos::View<double *, Kokkos::HostSpace> crx_v(const_cast<double*>(crx.data()), crx.shape(0));
                Kokkos::View<double *, Kokkos::HostSpace> cry_v(const_cast<double*>(cry.data()), cry.shape(0));
                grid.set_corners(crx_v, cry_v);
            },
            "crx"_a, "cry"_a)
        .def("to_unstructured", &axis::topology::StructuredGrid<Kokkos::HostSpace>::to_unstructured, "Convert structured grid to unstructured mesh");
```

- [ ] **Step 4: Update `grid.py` to use `StructuredGrid`**

In `libs/axis/python/axis/grid.py`, locate the curvilinear/cubed-sphere unstructured generation path (where `_synthesize_curvilinear_corners` and logical grids are manually built) and replace with `StructuredGrid`:
```python
            # 2D Curvilinear grid
            ni, nj = lon.shape[1], lon.shape[0]
            cx = lon.values.ravel()
            cy = lat.values.ravel()
            grid = axis_py.StructuredGrid(ni, nj, cx, cy)
            return grid.to_unstructured()
```
And similarly for `CurvilinearGrid.to_mesh()`:
```python
        else:
            ny, nx = self.lons.shape
            grid = axis_py.StructuredGrid(nx, ny, self.lons.ravel(), self.lats.ravel())
            return grid.to_unstructured()
```

- [ ] **Step 5: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_curvilinear_to_unstructured.py`
Expected: PASS

---

### Task 2: Implement C++ On-Device UGRID Poly Triangulator

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/grid.py`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_mpas_triangulator.py`

**Interfaces:**
- Consumes: C++ MPAS unstructured vertex triangulation
- Produces: `axis_py.triangulate_poly_cells`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_mpas_triangulator.py`:
```python
import pytest
import numpy as np
from axis import axis_py

def test_poly_triangulation():
    # Define a single pentagon cell centered at (0,0)
    # 5 nodes (vertices)
    node_coords = np.array([
        [0.0, 1.0], [1.0, 0.5], [0.5, -0.5], [-0.5, -0.5], [-1.0, 0.5]
    ], dtype=np.float64)

    # 1 cell, 5 edges
    conn_raw = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    n_edges = np.array([5], dtype=np.int64)

    res = axis_py.triangulate_poly_cells(node_coords, conn_raw, n_edges)
    assert "conn_offsets" in res
    assert "conn_indices" in res
    # 5 edges triangle fan has max_edges - 2 = 3 triangles, total 3*3 = 9 indices
    assert len(res["conn_indices"]) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_mpas_triangulator.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'triangulate_poly_cells'"

- [ ] **Step 3: Implement C++ Poly Triangulator Binding**

In `libs/axis/python/axis_py.cpp` add:
```cpp
    m.def(
        "triangulate_poly_cells",
        [](nb::ndarray<const double, nb::ndim<2>> node_coords, nb::ndarray<const axis::index_t, nb::ndim<2>> conn_raw,
           nb::ndarray<const axis::index_t, nb::ndim<1>> n_edges) -> nb::dict {
            ensure_kokkos();

            const std::size_t n_cells = conn_raw.shape(0);
            const std::size_t max_edges = conn_raw.shape(1);

            std::vector<axis::index_t> conn_indices;
            std::vector<axis::index_t> conn_offsets;
            conn_offsets.push_back(0);

            for (std::size_t c = 0; c < n_cells; ++c) {
                std::size_t n_cell_edges = n_edges(c);
                if (n_cell_edges < 3) continue;

                // Triangulate via standard triangle fan from vertex 0 of the cell
                axis::index_t v0 = conn_raw(c, 0) - 1;
                for (std::size_t j = 1; j < n_cell_edges - 1; ++j) {
                    conn_indices.push_back(v0);
                    conn_indices.push_back(conn_raw(c, j) - 1);
                    conn_indices.push_back(conn_raw(c, j + 1) - 1);
                }
                conn_offsets.push_back(conn_indices.size());
            }

            nb::dict res;
            // Return offsets and indices as numpy arrays
            auto off_uniq = std::make_unique<axis::index_t[]>(conn_offsets.size());
            std::copy(conn_offsets.begin(), conn_offsets.end(), off_uniq.get());
            auto ind_uniq = std::make_unique<axis::index_t[]>(conn_indices.size());
            std::copy(conn_indices.begin(), conn_indices.end(), ind_uniq.get());

            axis::index_t *raw_off = off_uniq.release();
            nb::capsule owner_off(raw_off, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_off[1] = {conn_offsets.size()};

            axis::index_t *raw_ind = ind_uniq.release();
            nb::capsule owner_ind(raw_ind, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_ind[1] = {conn_indices.size()};

            res["conn_offsets"] = nb::ndarray<nb::numpy, axis::index_t>(raw_off, 1, shape_off, std::move(owner_off));
            res["conn_indices"] = nb::ndarray<nb::numpy, axis::index_t>(raw_ind, 1, shape_ind, std::move(owner_ind));
            return res;
        },
        "node_coords"_a, "conn_raw"_a, "n_edges"_a, "Triangulate general poly cells into standard triangles");
```

- [ ] **Step 4: Update `grid.py` MPAS Path**

In `libs/axis/python/axis/grid.py: create_axis_mesh`, replace the old Python triangulator `_triangulate_mpas_mesh` with our high-speed C++ helper:
```python
            # ... inside create_axis_mesh ...
            if cell_centered:
                # Keep direct polygonal logic
                ...
            else:
                # Fused C++ triangulation
                v_conn = ds["verticesOnCell"]
                conn_raw = v_conn.values.astype(np.int64)
                n_edges = ds["nEdgesOnCell"].values.astype(np.int64) if "nEdgesOnCell" in ds else np.full(conn_raw.shape[0], conn_raw.shape[1], dtype=np.int64)
                
                tri_res = axis_py.triangulate_poly_cells(node_coords, conn_raw, n_edges)
                conn_offsets = tri_res["conn_offsets"]
                conn_indices = tri_res["conn_indices"]
                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
```

- [ ] **Step 5: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_mpas_triangulator.py`
Expected: PASS

---

### Task 3: Expose Fused Vector `vector_transform` SpMV

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/vector.py`
- Create: `libs/axis/tests_python/test_vector_transform_performance.py`

**Interfaces:**
- Consumes: C++ multidimensional Vector SpMVs
- Produces: `axis_py.vector_transform`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_vector_transform_performance.py`:
```python
import pytest
import numpy as np
from axis import axis_py

def test_fused_vector_transform():
    src = axis_py.make_regular_mesh(4, 4, 0.0, -90.0, 90.0, 45.0)
    dst = axis_py.make_regular_mesh(2, 2, 0.0, -90.0, 180.0, 90.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    src_alpha = np.zeros(src.n_cells)
    dst_alpha = np.zeros(dst.n_cells)
    W_u, W_v = axis_py.generate_vector_weights(src, dst, src_alpha, dst_alpha, config)

    # 10 variables, cells x variables
    u = np.ones((src.n_cells, 10))
    v = np.ones((src.n_cells, 10))

    u_out, v_out = axis_py.vector_transform(W_u, W_v, u, v)
    assert u_out.shape == (dst.n_cells, 10)
    assert v_out.shape == (dst.n_cells, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_vector_transform_performance.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'vector_transform'"

- [ ] **Step 3: Implement C++ `vector_transform` Binding**

In `libs/axis/python/axis_py.cpp` add:
```cpp
    m.def(
        "vector_transform",
        [](const HostMatrix &W_u, const HostMatrix &W_v, nb::ndarray<nb::numpy, double, nb::ndim<2>> u,
           nb::ndarray<nb::numpy, double, nb::ndim<2>> v) -> std::pair<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> {
            ensure_kokkos();

            const std::size_t n_src = u.shape(0);
            const std::size_t n_vars = u.shape(1);
            const std::size_t n_dst = W_u.n_dst();

            // Setup combined source buffer: [2 * n_src, n_vars]
            // where first n_src rows are U, and next n_src rows are V.
            std::vector<double> uv_colmajor(2 * n_src * n_vars);
            for (std::size_t var = 0; v < n_vars; ++var) {
                for (std::size_t i = 0; i < n_src; ++i) {
                    uv_colmajor[i + var * 2 * n_src] = u(i, var);
                    uv_colmajor[i + n_src + var * 2 * n_src] = v(i, var);
                }
            }

            axis::field_view<const double, 2> uv_view(uv_colmajor.data(), 2 * n_src, n_vars);

            auto dst_u_uniq = std::make_unique<double[]>(n_dst * n_vars);
            auto dst_v_uniq = std::make_unique<double[]>(n_dst * n_vars);

            axis::field_view<double, 2> dst_u_view(dst_u_uniq.get(), n_dst, n_vars);
            axis::field_view<double, 2> dst_v_view(dst_v_uniq.get(), n_dst, n_vars);

            axis::solver::batch_apply<Kokkos::HostSpace>(W_u, uv_view, dst_u_view);
            axis::solver::batch_apply<Kokkos::HostSpace>(W_v, uv_view, dst_v_view);

            // Reconstruct row-major for Python return
            auto res_u_uniq = std::make_unique<double[]>(n_dst * n_vars);
            auto res_v_uniq = std::make_unique<double[]>(n_dst * n_vars);

            for (std::size_t var = 0; var < n_vars; ++var) {
                for (std::size_t j = 0; j < n_dst; ++j) {
                    res_u_uniq[j * n_vars + var] = dst_u_uniq[j + var * n_dst];
                    res_v_uniq[j * n_vars + var] = dst_v_uniq[j + var * n_dst];
                }
            }

            double *raw_u = res_u_uniq.release();
            nb::capsule owner_u(raw_u, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_dst, n_vars};

            double *raw_v = res_v_uniq.release();
            nb::capsule owner_v(raw_v, [](void *p) noexcept { delete[] static_cast<double *>(p); });

            return std::make_pair(
                nb::ndarray<nb::numpy, double>(raw_u, 2, shape, std::move(owner_u)),
                nb::ndarray<nb::numpy, double>(raw_v, 2, shape, std::move(owner_v))
            );
        },
        "W_u"_a, "W_v"_a, "u"_a, "v"_a, "Perform coupled SpMV remapping for vector fields");
```

- [ ] **Step 4: Update Python `vector.py`**

In `libs/axis/python/axis/vector.py`, locate `VectorRegridder.transform` and replace the manual batching loops with our new unified `vector_transform` call:
```python
        # ... inside VectorRegridder.transform multidimensional batch case ...
            flat_u = u_arr.reshape(n_other, n_spatial)
            flat_v = v_arr.reshape(n_other, n_spatial)

            # High-speed unified C++ remapping SpMV
            u_out_t, v_out_t = axis_py.vector_transform(self._W_u, self._W_v, flat_u.T, flat_v.T)

            u_out = u_out_t.reshape(other_dims_shape + dst_shape)
            v_out = v_out_t.reshape(other_dims_shape + dst_shape)
```

- [ ] **Step 5: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_vector_transform_performance.py`
Expected: PASS
