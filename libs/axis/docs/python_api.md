# Python API {#python_api}

AXIS provides Python bindings via nanobind, enabling interactive prototyping of
regridding workflows with numpy arrays. The Python module links only against AXIS
and nanobind — no AMIO, HALO, or other HELM dependency.

## Installation

Build AXIS with Python bindings enabled:

```bash
cmake -B build \
  -DAXIS_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# Install the module
pip install build/python/
```

## Module Overview

```python
import axis

# Core components
axis.MeshFactory          # Create meshes from numpy arrays
axis.NamedGridRegistry    # Generate standard grids by name
axis.WeightGenerator      # Generate interpolation weights
axis.InterpolationMatrix  # Sparse weight matrix
axis.apply()              # Single-field interpolation
axis.batch_apply()        # Multi-field interpolation
```

## Mesh Construction

### From Numpy Arrays (GridDescriptor)

```python
import axis
import numpy as np

# Cell center coordinates
lon = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
lat = np.array([45.0, 45.0, 45.0, 45.0], dtype=np.float64)

# Connectivity (vertex indices per cell, CSR format)
connectivity = np.array([0, 1, 5, 4, 1, 2, 6, 5, ...], dtype=np.int64)
offsets = np.array([0, 4, 8, 12, 16], dtype=np.int64)

mesh = axis.MeshFactory.from_descriptor(
    lon=lon,
    lat=lat,
    connectivity=connectivity,
    offsets=offsets,
    coord_system="spherical_deg"
)
```

### Named Grids

```python
# Octahedral reduced Gaussian grids
src = axis.NamedGridRegistry.generate("O48")
dst = axis.NamedGridRegistry.generate("O96")

# Regular Gaussian
mesh = axis.NamedGridRegistry.generate("N128")
```

## Weight Generation

```python
config = {
    "method": "bilinear",               # or "conservative_1st", "conservative_2nd"
    "normalization": "dst_area",         # or "frac_area"
    "unmapped": "ignore",               # or "zero", "error"
    "use_limiter": False,
}

matrix = axis.WeightGenerator.generate(src, dst, config)
```

## Applying Weights

### Single Field

```python
src_field = np.random.randn(src.n_cells)
dst_field = axis.apply(matrix, src_field)
```

### Batch Apply (Multiple Variables)

```python
# Shape: [n_cells, n_vars]
src_fields = np.random.randn(src.n_cells, 10)
dst_fields = axis.batch_apply(matrix, src_fields)

assert dst_fields.shape == (dst.n_cells, 10)
```

## Weight Caching

```python
# Serialize to bytes
blob = matrix.to_bytes()

# Save/load via standard Python I/O
with open("weights.bin", "wb") as f:
    f.write(blob)

with open("weights.bin", "rb") as f:
    matrix2 = axis.InterpolationMatrix.from_bytes(f.read())

# Results are bitwise identical
dst1 = axis.apply(matrix, src_field)
dst2 = axis.apply(matrix2, src_field)
assert np.array_equal(dst1, dst2)
```

## Memory Layout Requirements

AXIS expects Fortran-order (column-major) arrays internally. The Python bindings
handle this transparently:

- **Fortran-order arrays** — zero-copy (passed directly to AXIS)
- **C-order arrays** — accepted with internal transpose (may allocate)

To avoid copies, create arrays with Fortran order:

```python
# Optimal: Fortran-order
data = np.asfortranarray(np.random.randn(n_cells, n_vars))

# Also works but may copy internally
data = np.random.randn(n_cells, n_vars)  # C-order
```

If a strict zero-copy mode is needed, pass `strict_layout=True`:

```python
# Raises ValueError for C-order arrays
dst = axis.apply(matrix, c_order_array, strict_layout=True)
```

## Complete Example

```python
import axis
import numpy as np

# Create grids
src_mesh = axis.NamedGridRegistry.generate("O48")
dst_mesh = axis.NamedGridRegistry.generate("O96")

# Generate conservative weights
config = {"method": "conservative_1st", "normalization": "dst_area"}
matrix = axis.WeightGenerator.generate(src_mesh, dst_mesh, config)

# Create a test field (constant = 1.0, should be preserved)
src_field = np.ones(src_mesh.n_cells)
dst_field = axis.apply(matrix, src_field)

# Verify conservation (partition of unity for constant field)
print(f"Max deviation from 1.0: {np.max(np.abs(dst_field - 1.0)):.2e}")

# Batch apply for atmosphere state
n_vars = 5  # T, u, v, q, ps
src_state = np.random.randn(src_mesh.n_cells, n_vars)
dst_state = axis.batch_apply(matrix, src_state)

print(f"Regridded {n_vars} variables: {src_state.shape} -> {dst_state.shape}")
```

## Error Handling

Python exceptions are mapped from C++ exceptions:

| C++ Exception | Python Exception |
|---------------|-----------------|
| `std::invalid_argument` | `ValueError` |
| `std::runtime_error` | `RuntimeError` |
| Kokkos allocation failure | `MemoryError` |

```python
try:
    # Wrong size array
    bad = np.ones(42)
    axis.apply(matrix, bad)
except ValueError as e:
    print(f"Caught: {e}")
```
