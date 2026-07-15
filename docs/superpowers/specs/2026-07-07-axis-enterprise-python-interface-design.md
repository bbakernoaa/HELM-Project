# Design Specification: AXIS Enterprise Python Interface

## 1. Overview
AXIS (Arbitrary eXgrid Interpolation Solver) delivers parallel-on-device spatial regridding for Earth system models. This design establishes a general-purpose, enterprise-grade Python API for AXIS, providing an exascale replacement for established tools like `xesmf`, `xregrid`, `earthkit-regrid`, and `cdo`.

The interface combines a **high-level, convenience-oriented interface** (matching standard climate science conventions) with a **low-level, modular object-oriented interface** conforming to the modern Scikit-Learn standard (`fit`/`transform` Transformer API).

---

## 2. Low-Level Explicit Geometry API
To support diverse input formats (NumPy arrays, Dask lazy chunks, pandas tables, or NetCDF/Zarr files), we decouple coordinate metadata extraction from the underlying storage container.

Explicit geometries are located in `libs/axis/python/axis/grid.py`.

### Grid and Mesh Classes:
```python
class Geometry:
    """Base abstract class representing physical coordinate structures."""
    def to_mesh(self) -> "UnstructuredMesh":
        raise NotImplementedError()

class RectilinearGrid(Geometry):
    """2D lat-lon grid with 1-D coordinate vectors."""
    def __init__(self, lons: np.ndarray, lats: np.ndarray):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)

    def to_mesh(self):
        ni = len(self.lons)
        nj = len(self.lats)
        dlon = (self.lons[-1] - self.lons[0]) / (ni - 1) if ni > 1 else 1.0
        dlat = (self.lats[-1] - self.lats[0]) / (nj - 1) if nj > 1 else 1.0
        from . import axis_py
        return axis_py.make_regular_mesh(ni, nj, self.lons[0], self.lats[0], dlon, dlat)

class CurvilinearGrid(Geometry):
    """2D curvilinear grid with 2-D coordinate matrices."""
    def __init__(self, lons: np.ndarray, lats: np.ndarray, proj_string: Optional[str] = None):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.proj_string = proj_string

    def to_mesh(self):
        from . import axis_py
        if self.proj_string:
            return axis_py.make_projected_mesh(
                self.lons.shape[1], self.lons.shape[0],
                self.proj_string, self.lons.ravel(), self.lats.ravel()
            )
        else:
            # Fallback: Synthesize corners and build unstructured mesh representation
            from .grid import _synthesize_curvilinear_corners
            clon, clat = _synthesize_curvilinear_corners(self.lons, self.lats)
            # Create CSR elements representing quadrilateral connectivity
            ...

class UnstructuredMesh(Geometry):
    """Arbitrary unstructured polygon grid (e.g. MPAS, FVCOM, SCRIP)."""
    def __init__(self, node_coords: np.ndarray, connectivity_offsets: np.ndarray, connectivity_indices: np.ndarray):
        self.coords = np.asarray(node_coords, dtype=np.float64)
        self.offsets = np.asarray(connectivity_offsets, dtype=np.int32)
        self.indices = np.asarray(connectivity_indices, dtype=np.int32)

    def to_mesh(self):
        from . import axis_py
        return axis_py.make_ugrid_mesh(self.coords, self.offsets, self.indices)
```

---

## 3. High-Level Regridder & ML Transformer API
The `axis.Regridder` provides both a scikit-learn-compliant `fit`/`transform` interface and a convenient xarray `__call__` interface with coastal masking.

Exposed in `libs/axis/python/axis/regridder.py`.

```python
class Regridder:
    def __init__(
        self,
        src: Union[Geometry, xr.Dataset, dict] = None,
        dst: Union[Geometry, xr.Dataset, dict] = None,
        method: str = "bilinear",
        src_mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
        dst_mask: Optional[Union[np.ndarray, xr.DataArray]] = None,
        norm_type: str = "fracarea",
        **kwargs: Any
    ):
        self.method = method
        self.norm_type = norm_type
        self.kwargs = kwargs
        self._weights_matrix = None
        self._serialized_weights = None

        # Call fit if geometries are passed during construction (xesmf style)
        if src is not None and dst is not None:
            self.fit(src, dst, src_mask=src_mask, dst_mask=dst_mask)

    def fit(self, src, dst, src_mask=None, dst_mask=None) -> "Regridder":
        """Generate interpolation matrices (weights) from source to destination."""
        self._src_geom = self._normalize_geometry(src)
        self._dst_geom = self._normalize_geometry(dst)

        src_mesh = self._src_geom.to_mesh()
        dst_mesh = self._dst_geom.to_mesh()

        config = {
            "method": self._map_method(self.method),
            "norm_type": self._map_norm_type(self.norm_type),
            "periodic": self.kwargs.get("periodic", False),
            "line_type": self.kwargs.get("line_type", "great_circle"),
        }

        # Support coastline/fraction area adjustments during weight generation
        if src_mask is not None:
            config["src_mask"] = np.asarray(src_mask, dtype=np.int32)
        if dst_mask is not None:
            config["dst_mask"] = np.asarray(dst_mask, dtype=np.int32)

        from . import axis_py
        self._weights_matrix = axis_py.generate_weights(src_mesh, dst_mesh, config)
        self._serialized_weights = self._weights_matrix.to_bytes()
        return self

    def transform(self, obj: Union[xr.DataArray, xr.Dataset, np.ndarray]) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
        """Apply the computed sparse matrix weights. Supports remote lazy Dask block execution."""
        if self._weights_matrix is None:
            raise RuntimeError("Regridder must be fit to grids before calling transform().")

        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            return self._apply_xarray(obj)
        else:
            from . import axis_py
            return axis_py.apply_weights(self._weights_matrix, np.asarray(obj))

    def __call__(self, obj: Union[xr.DataArray, xr.Dataset]) -> Union[xr.DataArray, xr.Dataset]:
        """XESMF compatibility caller."""
        return self.transform(obj)
```

---

## 4. Multi-Dimensional & Physical Solvers

### 4.1 Coupled Vector Regridding (`axis.VectorRegridder`)
Horizontal vectors (winds, currents, tides) are remapped and rotated simultaneously to maintain physical alignment relative to regional/rotated poles or curvilinear grids:

```python
class VectorRegridder:
    def __init__(self, src: Geometry, dst: Geometry, method: str = "bilinear", **kwargs: Any):
        self.src_mesh = src.to_mesh() if isinstance(src, Geometry) else src
        self.dst_mesh = dst.to_mesh() if isinstance(dst, Geometry) else dst

        from . import axis_py
        config = {"method": method, **kwargs}
        self._cpp_vector_regridder = axis_py.VectorWeightGenerator.generate(
            self.src_mesh, self.dst_mesh, config
        )

    def transform(self, u: xr.DataArray, v: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
        """Remap and rotate (u, v) vector components simultaneously."""
        # Delegates to C++ parallel solver: self._cpp_vector_regridder.apply(u, v)
        ...
```

### 4.2 Tension Spline Vertical Regridding (`axis.VerticalRegridder`)
Interpolates column vertical profiles (1D or spatially-varying 2D coordinates) utilizing non-overshooting tension splines (TSPACK), parallelized on devices via Kokkos `TeamPolicy` and Shared Scratch Memory.

```python
class VerticalRegridder:
    def __init__(self, tension: float = 0.0):
        self.tension = tension

    def interpolate(self, src_field, src_levels, dst_levels):
        """Remap columns of 3D/4D profiles from source vertical coordinate to target levels."""
        from . import axis_py
        return axis_py.VerticalRegridder.interpolate(
            src_field, src_levels, dst_levels, self.tension
        )
```

### 4.3 3D Unified Pipeline (`axis.regrid_3d`)
Provides an end-to-end multi-dimensional convenience wrapper:
```python
def regrid_3d(
    ds: xr.Dataset,
    target_grid_2d: Union[Geometry, xr.Dataset],
    target_levels_1d: np.ndarray,
    method_2d: str = "bilinear",
    tension_1d: float = 2.0,
    vertical_coord_name: str = "lev",
) -> xr.Dataset:
    ...
```

---

## 5. Testing & Verification
We will verify the enterprise API additions under the following test specifications:
1.  **`test_grid_geometry.py`**: Asserts explicit creation of Rectilinear, Curvilinear, and Unstructured geometries from multiple sources, verifying matching mesh structures.
2.  **`test_regridder_transformer.py`**: Verifies that standard ML `fit` and `transform` paradigms produce identical floating-point outcomes to the convenient callable interface.
3.  **`test_coastal_masking.py`**: Tests that passing land-sea binary masks to the generator shifts the relative weights conforming to conservative fractional area models (`fracarea` vs `dstarea`).
4.  **`test_vector_regridder.py`**: Evaluates coordinate rotational accuracy for vector wind profiles.
5.  **`test_vertical_spline.py`**: Asserts column tension interpolation accuracy.
