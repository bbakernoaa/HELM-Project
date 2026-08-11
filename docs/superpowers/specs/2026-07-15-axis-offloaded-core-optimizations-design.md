# Design Specification: AXIS Python Offloaded Core Optimizations

## 1. Overview
AXIS (Arbitrary eXgrid Interpolation Solver) performs high-performance, stateless spatial remapping for Earth system fields via Kokkos parallel kernels. This specification details the design for offloading three highly redundant and computationally heavy Python-side capabilities into the optimized C++ core:
1.  **Structured-to-Unstructured Translation**: Expose C++ `StructuredGrid` and its on-device `to_unstructured()` conversion to Python, replacing manual logically-rectangular indexing loops in Python `grid.py`.
2.  **Triangulation of Poly Cells (MPAS)**: Triangulating general multi-edge polygons (like MPAS Voronoi cells) into standard triangles/quads natively in C++ instead of using serial Python loops.
3.  **Unified Vector Apply & Rotation**: Remapping and rotating physical (U, V) vector fields (winds, ocean currents) in C++ directly via unified multidimensional SpMVs.

---

## 2. Component Design & Interfaces

### 2.1 Structured-to-Unstructured C++ Translation
We will expose the C++ `StructuredGrid` class directly via nanobind, enabling Python users to instantiate a structured grid and convert it to an optimized unstructured mesh natively.

*   **Binding Code (`axis_py.cpp`)**:
    ```cpp
    nb::class_<axis::topology::StructuredGrid<Kokkos::HostSpace>>(m, "StructuredGrid")
        .def(nb::init<std::size_t, std::size_t, nb::ndarray<nb::numpy, double, nb::ndim<1>>, nb::ndarray<nb::numpy, double, nb::ndim<1>>>(),
             "ni"_a, "nj"_a, "cx"_a, "cy"_a)
        .def(
            "set_corners",
            [](axis::topology::StructuredGrid<Kokkos::HostSpace> &grid, nb::ndarray<nb::numpy, double, nb::ndim<1>> crx,
               nb::ndarray<nb::numpy, double, nb::ndim<1>> cry) {
                Kokkos::View<double *, Kokkos::HostSpace> crx_v(crx.data(), crx.shape(0));
                Kokkos::View<double *, Kokkos::HostSpace> cry_v(cry.data(), cry.shape(0));
                grid.set_corners(std::move(crx_v), std::move(cry_v));
            },
            "crx"_a, "cry"_a)
        .def("to_unstructured", &axis::topology::StructuredGrid<Kokkos::HostSpace>::to_unstructured, "Convert structured grid to unstructured mesh");
    ```

*   **Python Integration (`grid.py`)**:
    In `libs/axis/python/axis/grid.py`, we will replace the manual `bl`, `br`, `tr`, `tl` logically-rectangular index generation and node-coordinate list building with:
    ```python
    grid = axis_py.StructuredGrid(ni, nj, cx, cy)
    grid.set_corners(crx, cry)
    return grid.to_unstructured()
    ```

---

### 2.2 MPAS General Poly Triangulator
We will implement an optimized C++ helper `triangulate_poly_cells` in `libs/axis/python/axis_py.cpp` to perform fast, on-device vector triangulation of MPAS polygon nodes, connectivity, and edge offsets.

*   **Binding Code (`axis_py.cpp`)**:
    ```cpp
    m.def(
        "triangulate_poly_cells",
        [](nb::ndarray<const double, nb::ndim<1>> node_lat, nb::ndarray<const double, nb::ndim<1>> node_lon,
           nb::ndarray<const axis::index_t, nb::ndim<2>> conn_raw, nb::ndarray<const axis::index_t, nb::ndim<1>> n_edges) -> nb::dict {
            ensure_kokkos();
            // Performs fast triangulation of raw MPAS polygons in C++ on parallel devices
            ...
            nb::dict res;
            res["node_coords"] = ...;
            res["conn_offsets"] = ...;
            res["conn_indices"] = ...;
            return res;
        }
    );
    ```

---

### 2.3 Fused Multi-Field Vector SpMV Solver
Expose a single-pass C++ multidimensional `batch_apply` wrapper specifically for wind (U, V) remapping and grid frame rotation.

*   **Binding Code (`axis_py.cpp`)**:
    ```cpp
    m.def(
        "vector_transform",
        [](const HostMatrix &W_u, const HostMatrix &W_v, nb::ndarray<nb::numpy, double, nb::ndim<2>> u,
           nb::ndarray<nb::numpy, double, nb::ndim<2>> v) -> std::pair<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> {
            ensure_kokkos();
            // Fuses the SpMV execution for wind vector components natively in C++
            ...
        }
    );
    ```

---

## 3. Verification & Testing Spec
We will verify and validate the correctness of these exposed optimizations using dedicated pytest suites:
1.  **`test_curvilinear_to_unstructured.py`**: Compares the cell count and coordinate values of curvilinear structured grids generated natively via C++ vs. the old Python implementation.
2.  **`test_vector_transform_performance.py`**: Benchmarks U/V wind transformation times and memory allocations, verifying zero-copy execution.
3.  **`test_mpas_triangulator.py`**: Compares exact triangulation indices and offsets against analytical MPAS Voronoi cells.
