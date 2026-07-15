# AXIS Python Advanced C++ Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose core C++ utilities (GmshWriter, RuleGenerator, GradientReconstructor) to AXIS Python bindings and wrap them in elegant, high-level Python interfaces.

**Architecture:** Implement the high-performance raw C++ bindings in `axis_py.cpp` via nanobind and construct high-level wrappers and accessors in Python to support clean `xarray` and object integration.

**Tech Stack:** C++20, Kokkos, nanobind, Python, xarray, numpy, pytest.

## Global Constraints
- **Preserve zero-copy bounds**: Ensure C++ views and NumPy arrays share memory buffers directly.
- **Maintain Kokkos HostSpace execution**: All bindings operate on Kokkos::HostSpace.
- **TDD / Verification first**: Write tests for each feature and ensure they pass completely.

---

### Task 1: Expose and Wrap `GmshWriter`

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_gmsh_writer.py`

**Interfaces:**
- Consumes: `axis_py.Mesh`
- Produces: `axis_py.write_gmsh(filepath, mesh)` and `axis.Mesh.to_gmsh(filepath)`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_gmsh_writer.py`:
```python
import os
import pytest
import numpy as np
import axis
from axis import axis_py

def test_gmsh_writer_export(tmp_path):
    mesh = axis_py.make_regular_mesh(5, 5, 0.0, -90.0, 72.0, 36.0)
    filepath = os.path.join(tmp_path, "test.msh")
    
    # Verify high-level wrapping
    axis_py.write_gmsh(filepath, mesh)
    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest libs/axis/tests_python/test_gmsh_writer.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'write_gmsh'"

- [ ] **Step 3: Implement C++ nanobind exposure**

In `libs/axis/python/axis_py.cpp` near other m.def calls:
```cpp
    m.def(
        "write_gmsh",
        [](const std::string &filepath, const HostMesh &mesh) {
            axis::topology::GmshWriter::write<Kokkos::HostSpace>(filepath, mesh);
        },
        "filepath"_a, "mesh"_a, "Write an UnstructuredMesh to a Gmsh .msh file");
```

- [ ] **Step 4: Update high-level `__init__.py`**

In `libs/axis/python/axis/__init__.py`, import and export `write_gmsh`:
```python
write_gmsh = axis_py.write_gmsh
__all__.append("write_gmsh")
```

- [ ] **Step 5: Run tests and verify success**

Run: `python -m pytest libs/axis/tests_python/test_gmsh_writer.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add libs/axis/python/axis_py.cpp libs/axis/python/axis/__init__.py libs/axis/tests_python/test_gmsh_writer.py
git commit -m "feat: expose and test core GmshWriter to Python bindings"
```

---

### Task 2: Expose and Wrap `RuleGenerator`

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/__init__.py`
- Modify: `libs/axis/python/axis/grid.py`
- Create: `libs/axis/tests_python/test_rule_generator.py`

**Interfaces:**
- Consumes: Python dictionary representing mathematical rules
- Produces: `axis_py.generate_mesh_from_rules(config)`, `axis.RuleGeometry(config)`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_rule_generator.py`:
```python
import pytest
import axis

def test_rule_generator_regular():
    config = {
        "kind": "RegularLatLon",
        "bbox_min_x": 0.0,
        "bbox_max_x": 360.0,
        "bbox_min_y": -90.0,
        "bbox_max_y": 90.0,
        "r_x": 36.0,
        "r_y": 36.0
    }
    geom = axis.grid.RuleGeometry(config)
    mesh = geom.to_mesh()
    assert mesh.n_cells == 50
    assert mesh.n_nodes == 66
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest libs/axis/tests_python/test_rule_generator.py`
Expected: FAIL with "AttributeError: module 'axis.grid' has no attribute 'RuleGeometry'"

- [ ] **Step 3: Implement C++ nanobind exposure**

In `libs/axis/python/axis_py.cpp`:
```cpp
    m.def("generate_mesh_from_rules", [](const nb::dict &config) -> HostMesh {
        ensure_kokkos();
        axis::ingest::GridRulesParams rules;
        
        std::string kind_str = nb.cast<std::string>(config["kind"]);
        if (kind_str == "RegularLatLon") {
            rules.kind = axis::ingest::GridRulesKind::RegularLatLon;
        } else if (kind_str == "GaussianRegular") {
            rules.kind = axis::ingest::GridRulesKind::GaussianRegular;
        } else if (kind_str == "GaussianReduced") {
            rules.kind = axis::ingest::GridRulesKind::GaussianReduced;
        } else if (kind_str == "Projected") {
            rules.kind = axis::ingest::GridRulesKind::Projected;
        } else {
            throw std::invalid_argument("Unknown GridRulesKind: " + kind_str);
        }

        if (config.contains("bbox_min_x")) rules.bbox_min_x = nb::cast<double>(config["bbox_min_x"]);
        if (config.contains("bbox_max_x")) rules.bbox_max_x = nb::cast<double>(config["bbox_max_x"]);
        if (config.contains("bbox_min_y")) rules.bbox_min_y = nb::cast<double>(config["bbox_min_y"]);
        if (config.contains("bbox_max_y")) rules.bbox_max_y = nb::cast<double>(config["bbox_max_y"]);
        if (config.contains("r_x")) rules.r_x = nb::cast<double>(config["r_x"]);
        if (config.contains("r_y")) rules.r_y = nb::cast<double>(config["r_y"]);
        if (config.contains("gaussian_n")) rules.gaussian_n = nb::cast<axis::index_t>(config["gaussian_n"]);
        if (config.contains("proj_string")) rules.proj_string = nb::cast<std::string>(config["proj_string"]);

        return axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(rules);
    }, "config"_a, "Generate an UnstructuredMesh using abstract mathematical rules");
```

- [ ] **Step 4: Update high-level `grid.py` and `__init__.py`**

In `libs/axis/python/axis/grid.py`:
```python
class RuleGeometry(Geometry):
    """Geometry generated purely from abstract mathematical rules."""
    def __init__(self, config: dict):
        self.config = config

    def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
        return axis_py.generate_mesh_from_rules(self.config)
```

In `libs/axis/python/axis/__init__.py`:
```python
from .grid import CurvilinearGrid, Geometry, GridFactory, RectilinearGrid, UnstructuredMesh, RuleGeometry
# Add to __all__
```

- [ ] **Step 5: Run tests and verify success**

Run: `python -m pytest libs/axis/tests_python/test_rule_generator.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add libs/axis/python/axis_py.cpp libs/axis/python/axis/grid.py libs/axis/python/axis/__init__.py libs/axis/tests_python/test_rule_generator.py
git commit -m "feat: expose and test RuleGenerator bindings"
```

---

### Task 3: Expose and Wrap `GradientReconstructor`

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Modify: `libs/axis/python/axis/__init__.py`
- Create: `libs/axis/tests_python/test_gradient_reconstructor.py`

**Interfaces:**
- Consumes: `cell_values`, `centroids`, `adj_offsets`, `adj_indices`, `use_limiter`
- Produces: `axis_py.reconstruct_gradient`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_gradient_reconstructor.py`:
```python
import pytest
import numpy as np
from axis import axis_py

def test_gradient_reconstruct_linear_field():
    # 4 grid cell mesh (2x2 structured grid centers)
    # Centroids of 2x2 cells with resolution 1.0 starting at 0.5:
    centroids = np.array([
        [0.5, 0.5, 0.0],  # Cell 0
        [1.5, 0.5, 0.0],  # Cell 1
        [0.5, 1.5, 0.0],  # Cell 2
        [1.5, 1.5, 0.0]   # Cell 3
    ], dtype=np.float64)
    
    # Linear field f(x, y) = 2x + 3y
    values = 2.0 * centroids[:, 0] + 3.0 * centroids[:, 1]
    
    # Adjacency CSR structure
    adj_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    adj_indices = np.array([
        1, 2,  # Cell 0 neighbors: 1, 2
        0, 3,  # Cell 1 neighbors: 0, 3
        0, 3,  # Cell 2 neighbors: 0, 3
        1, 2   # Cell 3 neighbors: 1, 2
    ], dtype=np.int64)
    
    grad = axis_py.reconstruct_gradient(values, centroids, adj_offsets, adj_indices, False)
    
    # Verify reconstructed gradients (Expected x=2, y=3, z=0)
    assert grad.shape == (4, 3)
    for c in range(4):
        assert np.isclose(grad[c, 0], 2.0, rtol=1e-5)
        assert np.isclose(grad[c, 1], 3.0, rtol=1e-5)
        assert np.isclose(grad[c, 2], 0.0, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest libs/axis/tests_python/test_gradient_reconstructor.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'reconstruct_gradient'"

- [ ] **Step 3: Implement C++ nanobind exposure**

In `libs/axis/python/axis_py.cpp`:
```cpp
    m.def(
        "reconstruct_gradient",
        [](nb::ndarray<const double, nb::ndim<1>> cell_values,
           nb::ndarray<const double, nb::ndim<2>> centroids,
           nb::ndarray<const axis::index_t, nb::ndim<1>> adj_offsets,
           nb::ndarray<const axis::index_t, nb::ndim<1>> adj_indices,
           bool use_limiter) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            std::size_t n_cells = cell_values.shape(0);
            if (centroids.shape(0) != n_cells || centroids.shape(1) != 3) {
                throw std::invalid_argument("centroids shape must be (n_cells, 3)");
            }
            if (adj_offsets.shape(0) != n_cells + 1) {
                throw std::invalid_argument("adj_offsets shape must be (n_cells + 1)");
            }

            auto out_grad_uniq = std::make_unique<double[]>(n_cells * 3);
            double *out_ptr = out_grad_uniq.get();

            Kokkos::View<const double *, Kokkos::HostSpace> val_view(cell_values.data(), n_cells);
            Kokkos::View<const double *[3], Kokkos::HostSpace> cent_view(centroids.data(), n_cells);
            Kokkos::View<const axis::index_t *, Kokkos::HostSpace> off_view(adj_offsets.data(), n_cells + 1);
            Kokkos::View<const axis::index_t *, Kokkos::HostSpace> ind_view(adj_indices.data(), adj_indices.shape(0));
            Kokkos::View<double *[3], Kokkos::HostSpace> grad_view(out_ptr, n_cells);

            axis::solver::GradientReconstructor<Kokkos::HostSpace>::compute(
                val_view, cent_view, off_view, ind_view, grad_view, use_limiter
            );

            double *raw_ptr = out_grad_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_cells, 3};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "cell_values"_a, "centroids"_a, "adj_offsets"_a, "adj_indices"_a, "use_limiter"_a = false,
        "Reconstruct cell-centered linear gradients via least-squares over CSR neighbors"
    );
```

- [ ] **Step 4: Update high-level `__init__.py`**

In `libs/axis/python/axis/__init__.py`:
```python
reconstruct_gradient = axis_py.reconstruct_gradient
__all__.append("reconstruct_gradient")
```

- [ ] **Step 5: Run tests and verify success**

Run: `python -m pytest libs/axis/tests_python/test_gradient_reconstructor.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add libs/axis/python/axis_py.cpp libs/axis/python/axis/__init__.py libs/axis/tests_python/test_gradient_reconstructor.py
git commit -m "feat: expose and test GradientReconstructor bindings"
```
