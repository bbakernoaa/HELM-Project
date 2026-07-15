# AXIS Python Advanced Geometric Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose core C++ advanced geometric utilities (Spherical Geometry, Gnomonic Projector, and Spherical Polygon Area) to AXIS Python bindings via nanobind, structured into clean Python sub-packages.

**Architecture:** 
1. Map `axis::detail::Vec3` as a unified 3D Cartesian position vector.
2. Bind raw spherical utilities, forward/inverse gnomonic projections, Newton-based bilinear weight solvers, and fixed-capacity `SphericalPolygon` structures.
3. Expose them under `axis.spherical`, `axis.gnomonic`, and `axis.polygon` Python sub-modules.

**Tech Stack:** C++20, Kokkos, nanobind, Python, numpy, pytest.

## Global Constraints
- **Zero-Copy Performance**: Returned NumPy views must address raw pointers directly.
- **Kokkos HostSpace Compliance**: Executed entirely on Kokkos::HostSpace.
- **Header-Only Inversion**: Coordinate conversions must translate types cleanly at the C++ binding boundaries.

---

### Task 1: Bind Unified `Vec3` and `axis.spherical` Math Utilities

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Create: `libs/axis/python/axis/spherical.py`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_spherical_geometry.py`

**Interfaces:**
- Consumes: C++ `axis::detail::Vec3` and `axis::detail::spherical::robust_orient_sphere` / `great_circle_arc_intersection`
- Produces: `axis.spherical.Vec3`, `axis.spherical.lonlat_to_xyz`, `axis.spherical.xyz_to_lonlat`, `axis.spherical.robust_orient_sphere`, `axis.spherical.great_circle_arc_intersection`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_spherical_geometry.py`:
```python
import pytest
import numpy as np
import axis
from axis.spherical import Vec3, lonlat_to_xyz, xyz_to_lonlat, robust_orient_sphere, great_circle_arc_intersection

def test_spherical_conversions():
    # Convert 45 lon, 45 lat in radians to Cartesian
    p = lonlat_to_xyz(np.pi / 4.0, np.pi / 4.0)
    assert isinstance(p, Vec3)
    assert np.isclose(p.x, 0.5)
    assert np.isclose(p.y, 0.5)
    assert np.isclose(p.z, np.sqrt(2.0) / 2.0)

    lon, lat = xyz_to_lonlat(p)
    assert np.isclose(lon, np.pi / 4.0)
    assert np.isclose(lat, np.pi / 4.0)

def test_robust_orientation():
    a = Vec3(1.0, 0.0, 0.0)
    b = Vec3(0.0, 1.0, 0.0)
    # Point c is CCW (to the left)
    c_left = Vec3(0.5, 0.5, 0.5)
    val = robust_orient_sphere(a, b, c_left)
    assert val > 0.0

def test_arc_intersection():
    # Arc 1: Along equator (0 to 90 East)
    a1 = Vec3(1.0, 0.0, 0.0)
    a2 = Vec3(0.0, 1.0, 0.0)
    # Arc 2: Along prime meridian (-45 to 45 North)
    b1 = lonlat_to_xyz(0.0, -np.pi / 4.0)
    b2 = lonlat_to_xyz(0.0, np.pi / 4.0)

    p_inter = great_circle_arc_intersection(a1, a2, b1, b2)
    assert p_inter is not None
    assert np.isclose(p_inter.x, 1.0)
    assert np.isclose(p_inter.y, 0.0)
    assert np.isclose(p_inter.z, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_spherical_geometry.py`
Expected: FAIL with "ModuleNotFoundError: No module named 'axis.spherical'"

- [ ] **Step 3: Implement C++ nanobind exposure of Vec3 and Spherical Geometry**

In `libs/axis/python/axis_py.cpp`:
1. Include the header at the top:
```cpp
#include <axis/detail/spherical_geometry.hpp>
```
2. Near the bottom of `NB_MODULE(axis_py, m)`, add bindings for `Vec3` and the mathematical functions. Convert between `axis::detail::spherical::Vec3` and `axis::detail::Vec3` at the binding boundary seamlessly:
```cpp
    // ─── Unified Vec3 Class ──────────────────────────────────────────────────
    nb::class_<axis::detail::Vec3>(m, "Vec3")
        .def(nb::init<double, double, double>())
        .def_rw("x", &axis::detail::Vec3::x)
        .def_rw("y", &axis::detail::Vec3::y)
        .def_rw("z", &axis::detail::Vec3::z);

    // ─── Spherical Geometry math ─────────────────────────────────────────────
    m.def(
        "lonlat_to_xyz",
        [](double lon, double lat) -> axis::detail::Vec3 {
            auto v = axis::detail::spherical::lonlat_to_xyz(lon, lat);
            return axis::detail::Vec3{v.x, v.y, v.z};
        },
        "lon"_a, "lat"_a, "Convert lon/lat in radians to 3D Cartesian position vector");

    m.def(
        "xyz_to_lonlat",
        [](const axis::detail::Vec3 &p) -> std::pair<double, double> {
            axis::detail::spherical::Vec3 v{p.x, p.y, p.z};
            double lon, lat;
            axis::detail::spherical::xyz_to_lonlat(v, lon, lat);
            return {lon, lat};
        },
        "p"_a, "Convert 3D Cartesian position vector back to lon/lat in radians");

    m.def(
        "robust_orient_sphere",
        [](const axis::detail::Vec3 &a, const axis::detail::Vec3 &b, const axis::detail::Vec3 &c) -> double {
            axis::detail::spherical::Vec3 va{a.x, a.y, a.z};
            axis::detail::spherical::Vec3 vb{b.x, b.y, b.z};
            axis::detail::spherical::Vec3 vc{c.x, c.y, c.z};
            return axis::detail::spherical::robust_orient_sphere(va, vb, vc);
        },
        "a"_a, "b"_a, "c"_a, "Compute orientation sign of C relative to arc A->B using adaptive predicates");

    m.def(
        "great_circle_arc_intersection",
        [](const axis::detail::Vec3 &a1, const axis::detail::Vec3 &a2, const axis::detail::Vec3 &b1, const axis::detail::Vec3 &b2) -> nb::object {
            axis::detail::spherical::Vec3 va1{a1.x, a1.y, a1.z};
            axis::detail::spherical::Vec3 va2{a2.x, a2.y, a2.z};
            axis::detail::spherical::Vec3 vb1{b1.x, b1.y, b1.z};
            axis::detail::spherical::Vec3 vb2{b2.x, b2.y, b2.z};
            axis::detail::spherical::Vec3 vp;
            if (axis::detail::spherical::great_circle_arc_intersection(va1, va2, vb1, vb2, vp)) {
                return nb::cast(axis::detail::Vec3{vp.x, vp.y, vp.z});
            }
            return nb::none();
        },
        "a1"_a, "a2"_a, "b1"_a, "b2"_a, "Compute the exact intersection Vec3 of great-circle arcs A1->A2 and B1->B2, or None");
```

- [ ] **Step 4: Create high-level Python file `libs/axis/python/axis/spherical.py`**

```python
# SPDX-License-Identifier: Apache-2.0

from . import axis_py

Vec3 = axis_py.Vec3
lonlat_to_xyz = axis_py.lonlat_to_xyz
xyz_to_lonlat = axis_py.xyz_to_lonlat
robust_orient_sphere = axis_py.robust_orient_sphere
great_circle_arc_intersection = axis_py.great_circle_arc_intersection
```

- [ ] **Step 5: Register sub-module in `libs/axis/python/axis/__init__.py`**

```python
from . import spherical
# Add "spherical" to __all__
```

- [ ] **Step 6: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_spherical_geometry.py`
Expected: PASS

---

### Task 2: Expose and Wrap `GnomonicProjector` and Bilinear Newton Solver

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Create: `libs/axis/python/axis/gnomonic.py`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_gnomonic_projector.py`

**Interfaces:**
- Consumes: C++ `axis::detail::GnomonicProjector`
- Produces: `axis.gnomonic.forward`, `axis.gnomonic.inverse`, `axis.gnomonic.bilinear_weights`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_gnomonic_projector.py`:
```python
import pytest
import numpy as np
import axis
from axis.spherical import Vec3, lonlat_to_xyz
from axis.gnomonic import forward, inverse, bilinear_weights

def test_gnomonic_projections():
    center = Vec3(1.0, 0.0, 0.0)
    pt = lonlat_to_xyz(np.radians(10.0), np.radians(10.0))

    u, v = forward(center, pt)
    assert np.isclose(u, np.tan(np.radians(10.0)))
    
    reconstructed = inverse(center, u, v)
    assert np.isclose(reconstructed.x, pt.x)
    assert np.isclose(reconstructed.y, pt.y)
    assert np.isclose(reconstructed.z, pt.z)

def test_bilinear_solver():
    # Simple projected quad with resolution 2.0 (from -1 to 1)
    quad_u = np.array([-1.0, 1.0, 1.0, -1.0])
    quad_v = np.array([-1.0, -1.0, 1.0, 1.0])

    # Point at exact center
    w = bilinear_weights(quad_u, quad_v, 0.0, 0.0)
    assert w is not None
    assert np.allclose(w, 0.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_gnomonic_projector.py`
Expected: FAIL with "ModuleNotFoundError: No module named 'axis.gnomonic'"

- [ ] **Step 3: Implement C++ nanobind exposure of Gnomonic projection**

In `libs/axis/python/axis_py.cpp`:
1. Include header at top:
```cpp
#include <axis/detail/gnomonic_projector.hpp>
```
2. Bind the class and functions near the bottom of `NB_MODULE(axis_py, m)`:
```cpp
    // ─── Gnomonic Tangent Projection Math ────────────────────────────────────
    m.def(
        "gnomonic_forward",
        [](const axis::detail::Vec3 &center, const axis::detail::Vec3 &point) -> std::pair<double, double> {
            double u, v;
            axis::detail::GnomonicProjector::forward(center, point, u, v);
            return {u, v};
        },
        "center"_a, "point"_a, "Project point on the sphere onto the tangent plane at center");

    m.def(
        "gnomonic_inverse",
        [](const axis::detail::Vec3 &center, double u, double v) -> axis::detail::Vec3 {
            return axis::detail::GnomonicProjector::inverse(center, u, v);
        },
        "center"_a, "u"_a, "v"_a, "Reconstruct a unit-sphere point from tangent-plane coordinates");

    m.def(
        "bilinear_weights",
        [](nb::ndarray<const double, nb::ndim<1>> quad_u, nb::ndarray<const double, nb::ndim<1>> quad_v, double pu, double pv) -> nb::object {
            if (quad_u.shape(0) != 4 || quad_v.shape(0) != 4) {
                throw std::invalid_argument("quad_u and quad_v must have exactly 4 vertices");
            }
            double weights[4];
            bool ok = axis::detail::GnomonicProjector::bilinear_weights(quad_u.data(), quad_v.data(), pu, pv, weights);
            if (ok) {
                std::vector<double> out_weights(weights, weights + 4);
                return nb::cast(out_weights);
            }
            return nb::none();
        },
        "quad_u"_a, "quad_v"_a, "pu"_a, "pv"_a, "Solve the inverse bilinear problem for point pu, pv in quadrilateral");
```

- [ ] **Step 4: Create high-level Python file `libs/axis/python/axis/gnomonic.py`**

```python
# SPDX-License-Identifier: Apache-2.0

from . import axis_py

forward = axis_py.gnomonic_forward
inverse = axis_py.gnomonic_inverse
bilinear_weights = axis_py.bilinear_weights
```

- [ ] **Step 5: Register sub-module in `libs/axis/python/axis/__init__.py`**

```python
from . import gnomonic
# Add "gnomonic" to __all__
```

- [ ] **Step 6: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_gnomonic_projector.py`
Expected: PASS

---

### Task 3: Expose and Wrap `SphericalPolygon` Area Computations

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Create: `libs/axis/python/axis/polygon.py`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_spherical_polygon.py`

**Interfaces:**
- Consumes: C++ `axis::detail::SphericalPolygon<32>`
- Produces: `axis.polygon.SphericalPolygon`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_spherical_polygon.py`:
```python
import pytest
import numpy as np
import axis
from axis.spherical import lonlat_to_xyz
from axis.polygon import SphericalPolygon

def test_spherical_excess_area():
    # Create an octant on the unit sphere (longitude 0 to 90 East, equator to North Pole)
    # Area must be 1/8 of total sphere area = (4 * pi) / 8 = pi / 2 ≈ 1.570796
    poly = SphericalPolygon()
    
    # Vertices
    v0 = lonlat_to_xyz(0.0, 0.0)
    v1 = lonlat_to_xyz(np.pi / 2.0, 0.0)
    v2 = lonlat_to_xyz(0.0, np.pi / 2.0)
    
    poly.add_vertex(v0)
    poly.add_vertex(v1)
    poly.add_vertex(v2)
    
    assert poly.n == 3
    area_calc = poly.area()
    assert np.isclose(area_calc, np.pi / 2.0, rtol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_spherical_polygon.py`
Expected: FAIL with "ModuleNotFoundError: No module named 'axis.polygon'"

- [ ] **Step 3: Implement C++ nanobind exposure of SphericalPolygon**

In `libs/axis/python/axis_py.cpp`:
1. Include header at top:
```cpp
#include <axis/detail/spherical_clipper.hpp>
```
2. Bind the class near the bottom of `NB_MODULE(axis_py, m)`:
```cpp
    // ─── SphericalPolygon 32 Capacity Struct ─────────────────────────────────
    nb::class_<axis::detail::SphericalPolygon<32>>(m, "SphericalPolygon")
        .def(nb::init<>())
        .def_rw("n", &axis::detail::SphericalPolygon<32>::n)
        .def("area", &axis::detail::SphericalPolygon<32>::area)
        .def(
            "add_vertex",
            [](axis::detail::SphericalPolygon<32> &poly, const axis::detail::Vec3 &v) {
                if (poly.n >= 32) {
                    throw std::runtime_error("Polygon vertex capacity (32) exceeded");
                }
                poly.verts[poly.n++] = v;
            },
            "v"_a, "Add a unit-sphere Cartesian position vector vertex to the polygon");
```

- [ ] **Step 4: Create high-level Python file `libs/axis/python/axis/polygon.py`**

```python
# SPDX-License-Identifier: Apache-2.0

from . import axis_py

SphericalPolygon = axis_py.SphericalPolygon
```

- [ ] **Step 5: Register sub-module in `libs/axis/python/axis/__init__.py`**

```python
from . import polygon
# Add "polygon" to __all__
```

- [ ] **Step 6: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_spherical_polygon.py`
Expected: PASS
