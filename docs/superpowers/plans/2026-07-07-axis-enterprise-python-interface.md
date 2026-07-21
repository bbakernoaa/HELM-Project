# AXIS Enterprise Python Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a general-purpose, scikit-learn-compliant, and high-performance Python API for AXIS to outclass `xesmf` and `cdo`, supporting explicit geometry classes, unified `fit`/`transform` pipelines, coastal masking, coupled vectors, and vertical splines.

**Architecture:** Introduce explicit Geometry subclasses in `grid.py` mapping to native C++ bindings, adapt `Regridder` to conform to Scikit-learn transformer models with separate weight generation (`fit`) and applying (`transform`) methods, and introduce `VectorRegridder` and `VerticalRegridder` classes for complex multidimensional physics.

**Tech Stack:** Python 3.10+, NumPy, xarray, Dask, nanobind, C++20.

## Global Constraints
- Python classes must maintain 100% backwards compatibility for existing callables (e.g. `regridder(da)` must work identically).
- All new Python modules must follow existing style conventions (PEP 8, numpy docstrings).
- No new binary/compiled Python dependencies may be introduced.
- Dask multi-node serialization pathways (worker local caching) must be preserved.

---

### Task 1: Scaffolding Explicit Geometry Abstractions

**Files:**
- Modify: `libs/axis/python/axis/grid.py`
- Create: `libs/axis/tests_python/test_grid_geometry.py`

**Interfaces:**
- Consumes: `axis_py.make_regular_mesh`, `axis_py.make_projected_mesh`, `axis_py.make_ugrid_mesh`
- Produces: `axis.RectilinearGrid`, `axis.CurvilinearGrid`, `axis.UnstructuredMesh`, `axis.GridFactory`

- [ ] **Step 1: Write explicit Geometry classes**
  Modify `libs/axis/python/axis/grid.py` to declare the object-oriented abstractions. Add the classes:

```python
class Geometry:
    def to_mesh(self):
        raise NotImplementedError()

class RectilinearGrid(Geometry):
    def __init__(self, lons, lats):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)

    def to_mesh(self):
        ni = len(self.lons)
        nj = len(self.lats)
        dlon = (self.lons[-1] - self.lons[0]) / (ni - 1) if ni > 1 else 1.0
        dlat = (self.lats[-1] - self.lats[0]) / (nj - 1) if nj > 1 else 1.0
        return axis_py.make_regular_mesh(ni, nj, self.lons[0], self.lats[0], dlon, dlat)

class CurvilinearGrid(Geometry):
    def __init__(self, lons, lats, proj_string=None):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.proj_string = proj_string

    def to_mesh(self):
        if self.proj_string:
            return axis_py.make_projected_mesh(
                self.lons.shape[1], self.lons.shape[0],
                self.proj_string, self.lons.ravel(), self.lats.ravel()
            )
        else:
            clon, clat = _synthesize_curvilinear_corners(self.lons, self.lats)
            ny, nx = self.lons.shape
            n_nodes = (nx + 1) * (ny + 1)
            n_cells = nx * ny

            node_coords = np.stack([clon.ravel(), clat.ravel()], axis=1)
            conn_offsets = np.arange(0, (n_cells + 1) * 4, 4, dtype=np.int32)

            conn_indices = []
            for j in range(ny):
                row_start = j * (nx + 1)
                next_row_start = (j + 1) * (nx + 1)
                for i in range(nx):
                    conn_indices.extend([
                        row_start + i,
                        row_start + i + 1,
                        next_row_start + i + 1,
                        next_row_start + i
                    ])
            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, np.array(conn_indices, dtype=np.int32))

class UnstructuredMesh(Geometry):
    def __init__(self, node_coords, connectivity_offsets, connectivity_indices):
        self.coords = np.asarray(node_coords, dtype=np.float64)
        self.offsets = np.asarray(connectivity_offsets, dtype=np.int32)
        self.indices = np.asarray(connectivity_indices, dtype=np.int32)

    def to_mesh(self):
        return axis_py.make_ugrid_mesh(self.coords, self.offsets, self.indices)

class GridFactory:
    @staticmethod
    def from_xarray(ds, lon_var=None, lat_var=None):
        if lon_var and lat_var:
            lons = ds[lon_var].values
            lats = ds[lat_var].values
            if lons.ndim == 1:
                return RectilinearGrid(lons, lats)
            return CurvilinearGrid(lons, lats)
        # Fallback to standard auto-detection in _get_mesh_info
        lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds)
        if is_unstructured:
            if "verticesOnCell" in ds and "latVertex" in ds:
                # MPAS
                lons, lats, indices = _triangulate_mpas_mesh(ds)
                offsets = np.arange(0, len(indices) + 3, 3, dtype=np.int32)
                return UnstructuredMesh(np.stack([lons, lats], axis=1), offsets, indices)
            elif "grid_corner_lon" in ds:
                # SCRIP
                lons, lats, offsets, indices = _parse_scrip_bounds(ds)
                return UnstructuredMesh(np.stack([lons, lats], axis=1), offsets, indices)
            # General quad-curvilinear fallback
            clon, clat = _synthesize_curvilinear_corners(lon.values, lat.values)
            ...
```

- [ ] **Step 2: Create unit test `test_grid_geometry.py`**
  Write explicit assertions verifying node and cell dimensions for Rectilinear, Curvilinear, and unstructured meshes generated from NumPy arrays.

- [ ] **Step 3: Run python tests to verify**
  Run: `pytest libs/axis/tests_python/test_grid_geometry.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add libs/axis/python/axis/grid.py libs/axis/tests_python/test_grid_geometry.py
git commit -m "feat(python): implement low-level explicit Geometry API classes"
```

---

### Task 2: High-Level Regridder with ML Transformer API & Coastal Masking

**Files:**
- Modify: `libs/axis/python/axis/regridder.py`
- Create: `libs/axis/tests_python/test_regridder_transformer.py`
- Create: `libs/axis/tests_python/test_coastal_masking.py`

**Interfaces:**
- Consumes: `Geometry`, `axis_py.generate_weights`, `axis_py.apply_weights`
- Produces: `Regridder.fit()`, `Regridder.transform()`, land-sea coastal masking overrides.

- [ ] **Step 1: Implement SKLearn-style fit/transform interface**
  Refactor `libs/axis/python/axis/regridder.py` to decouple constructor initialization from compiling/fitting weights. Add `fit`, `transform`, and update `__call__` for backwards compatibility.

```python
    def fit(self, src, dst, src_mask=None, dst_mask=None) -> "Regridder":
        self._src_geom = self._normalize_geometry(src)
        self._dst_geom = self._normalize_geometry(dst)

        src_mesh = self._src_geom.to_mesh()
        dst_mesh = self._dst_geom.to_mesh()

        config = {
            "method": self._axis_method,
            "periodic": self.periodic,
            "line_type": self._axis_line_type,
            "unmapped": axis_py.UnmappedAction.Ignore if self.unmapped == "ignore" else axis_py.UnmappedAction.Error
        }

        if src_mask is not None:
            config["src_mask"] = np.asarray(src_mask, dtype=np.int32)
        if dst_mask is not None:
            config["dst_mask"] = np.asarray(dst_mask, dtype=np.int32)

        self._weights_matrix = axis_py.generate_weights(src_mesh, dst_mesh, config)
        self._serialized_weights = self._weights_matrix.to_bytes()
        return self

    def transform(self, obj) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
        if self._weights_matrix is None:
            raise RuntimeError("Regridder must be fit to grids before calling transform().")
        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            return self._regrid_dataarray(obj) if isinstance(obj, xr.DataArray) else self._regrid_dataset(obj)
        return axis_py.apply_weights(self._weights_matrix, np.asarray(obj))
```

- [ ] **Step 2: Add support for coastal masking inputs**
  Support passing masks in `fit()` and pass them to the C++ weight generator config dictionary to verify fractional coastline cell scaling.

- [ ] **Step 3: Create transformer and coastal masking test cases**
  Verify that calling `fit()` followed by `transform()` yields floating point values identical to `__call__()`, and that passing dry masks alters coastal boundary weights.

- [ ] **Step 4: Run python test suite**
  Run: `pytest libs/axis/tests_python/ -v`
  Expected: All passing.

- [ ] **Step 5: Commit**
```bash
git add libs/axis/python/axis/regridder.py libs/axis/tests_python/test_regridder_transformer.py libs/axis/tests_python/test_coastal_masking.py
git commit -m "feat(python): implement standard fit/transform transformer API and coastal masking"
```

---

### Task 3: Vector & Vertical Spline physics solvers

**Files:**
- Create: `libs/axis/python/axis/vector.py`
- Create: `libs/axis/python/axis/vertical.py`
- Create: `libs/axis/tests_python/test_vector_regridder.py`
- Create: `libs/axis/tests_python/test_vertical_spline.py`
- Modify: `libs/axis/python/axis/__init__.py`

**Interfaces:**
- Consumes: C++ VectorWeightGenerator and VerticalRegridder bindings
- Produces: `axis.VectorRegridder`, `axis.VerticalRegridder`, and unified pipeline `axis.regrid_3d(...)`

- [ ] **Step 1: Write `axis.VectorRegridder` class**
  Implement in `libs/axis/python/axis/vector.py` to wrap parallel rotational SpMV vector computations.

- [ ] **Step 2: Write `axis.VerticalRegridder` class**
  Implement in `libs/axis/python/axis/vertical.py` to wrap device-resident TSPACK tension splines.

- [ ] **Step 3: Implement 3D Unified Pipeline**
  Implement `axis.regrid_3d(...)` to combine 2D horizontal bilinear/conservative remapping and 1D column-wise vertical tension spline interpolations in sequence.

- [ ] **Step 4: Create and run comprehensive unit tests**
  Write tests for wind vector rotations on rotated grids and column splines verifying non-overshooting constraints.
  Run: `pytest libs/axis/tests_python/test_vector_regridder.py libs/axis/tests_python/test_vertical_spline.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add libs/axis/python/axis/vector.py libs/axis/python/axis/vertical.py libs/axis/python/axis/__init__.py libs/axis/tests_python/test_vector_regridder.py libs/axis/tests_python/test_vertical_spline.py
git commit -m "feat(python): add coupled Vector and Vertical spline regridding interfaces"
```
