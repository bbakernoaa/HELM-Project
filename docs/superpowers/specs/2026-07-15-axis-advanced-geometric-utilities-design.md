# Design Specification: AXIS Python Advanced Geometric Utilities

## 1. Overview
AXIS (Arbitrary eXgrid Interpolation Solver) utilizes highly robust, numerically stable spherical trigonometry and local coordinate projection algorithms in its core C++ engine. This specification details the design for exposing these key geocomputation utilities as structured sub-modules under `axis` in Python.

---

## 2. Component Design & Interfaces

### 2.1 `axis.spherical` (Robust Spherical Geometry)
Robust Cartesian/Spherical vector calculations on the unit sphere, using Knuth TwoSum / Dekker TwoProduct adaptive predicates when standard double-precision orientation calculations are close to zero.

*   **Data Structure: `Vec3`**:
    *   Exposed as a lightweight, copyable class representation in Python:
        ```cpp
        nb::class_<axis::detail::spherical::Vec3>(m, "Vec3")
            .def(nb::init<double, double, double>())
            .def_rw("x", &axis::detail::spherical::Vec3::x)
            .def_rw("y", &axis::detail::spherical::Vec3::y)
            .def_rw("z", &axis::detail::spherical::Vec3::z);
        ```
*   **Mathematical Utilities**:
    *   `lonlat_to_xyz(lon_rad, lat_rad)`: Converts spherical longitude/latitude coordinates (in radians) into a unit-sphere Cartesian `Vec3`.
    *   `xyz_to_lonlat(p)`: Converts a Cartesian `Vec3` point back to `(lon_rad, lat_rad)`.
    *   `robust_orient_sphere(a, b, c)`: Returns orientative sign ($>0$ for CCW, $<0$ for CW, $0$ for collinear) utilizing adaptive error-free transformations (EFT).
    *   `great_circle_arc_intersection(a1, a2, b1, b2)`: Returns the exact `Vec3` intersection point of two great-circle arcs on a sphere, or `None` if they do not intersect.

---

### 2.2 `axis.gnomonic` (Tangent Coordinate Projections)
Handles forward/inverse local tangent gnomonic projection, and computes exact sub-pixel bilinear interpolation coordinates inside any quadrilateral.

*   **Projections**:
    *   `forward(center, point)`: Projects a `Vec3` point on a sphere onto a tangent plane at a given `center`, returning `(u, v)`.
    *   `inverse(center, u, v)`: Reconstructs a unit-sphere Cartesian `Vec3` from tangent-plane coordinates.
*   **Quadrilateral Solver**:
    *   `bilinear_weights(quad_u, quad_v, pu, pv)`: Solves the inverse bilinear problem for point `(pu, pv)` inside a quadrilateral using high-speed Newton iteration. Returns a list of 4 float weights ($w_0..w_3$) if converged, or `None`.

---

### 2.3 `axis.polygon` (General Spherical Polygon Metrics)
Calculates exact physical areas of arbitrary polygon layouts (such as MPAS unstructured cells) directly in C++ using spherical excess (Girard's theorem).

*   **Structure: `SphericalPolygon`**:
    *   A fixed-capacity polygon class mapping to C++ `SphericalPolygon<32>`:
        ```cpp
        nb::class_<axis::detail::SphericalPolygon<32>>(m, "SphericalPolygon")
            .def(nb::init<>())
            .def_rw("n", &axis::detail::SphericalPolygon<32>::n)
            .def("area", &axis::detail::SphericalPolygon<32>::area)
            .def("add_vertex", [](axis::detail::SphericalPolygon<32> &poly, double x, double y, double z) {
                if (poly.n >= 32) throw std::runtime_error("Polygon vertex capacity (32) exceeded");
                poly.verts[poly.n++] = axis::detail::Vec3{x, y, z};
            });
        ```

---

## 3. Structural Integration under `axis/`
The newly exposed features will be cleanly structured into dedicated modular files:
*   `libs/axis/python/axis/spherical.py`: Exposes `Vec3`, `lonlat_to_xyz`, `xyz_to_lonlat`, `robust_orient_sphere`, and `great_circle_arc_intersection`.
*   `libs/axis/python/axis/gnomonic.py`: Exposes `GnomonicProjector` and gnomonic coordinate math.
*   `libs/axis/python/axis/polygon.py`: Exposes the `SphericalPolygon` class.

These sub-packages are registered, documented, and exposed dynamically in `libs/axis/python/axis/__init__.py`.

---

## 4. Verification & Testing Spec
We will verify and validate the mathematical correctness of these exposed bindings using dedicated pytest suites:
1.  **`test_spherical_geometry.py`**: Asserts that `robust_orient_sphere` correctly determines collinearity/chirality, and validates that `great_circle_arc_intersection` resolves exact intersection coordinates.
2.  **`test_gnomonic_projector.py`**: Projects spherical cells to flat tangent coordinates, reconstructs them back onto a sphere, and verifies that `bilinear_weights` Newton iteration resolves weights correctly.
3.  **`test_spherical_polygon.py`**: Constructs triangles/quads on a unit sphere, calculates areas, and compares results to analytical spherical excess formulas.
