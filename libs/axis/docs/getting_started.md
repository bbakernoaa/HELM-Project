# Getting Started with AXIS {#mainpage}

AXIS (Arbitrary eXgrid Interpolation Solver) is a Tier 1 C++20 micro-library within the
HELM ecosystem providing stateless spatial interpolation (regridding) for Earth-system fields.
AXIS is hardware-portable via Kokkos, opens no files, and links no file-format libraries.

## Key Features

- **Bilinear interpolation** — true bilinear on the sphere via gnomonic projection
- **Conservative remapping** — 1st and 2nd order with spherical polygon clipping
- **GPU-portable** — entire pipeline runs on host or device via Kokkos
- **CSR storage** — row-parallel SpMV with no atomics
- **Batch apply** — single kernel for multiple variables
- **Weight caching** — serialize/deserialize without recomputation
- **Cell masking** — land/ocean boundary handling
- **Python API** — numpy-based prototyping interface

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| Kokkos | Required | Parallelism + memory spaces |
| ArborX | Vendored | BVH spatial index |
| PROJ | Optional | Coordinate projection transforms |

## Building

```bash
cd libs/axis
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAXIS_BUILD_TESTING=ON \
  -DAXIS_ENABLE_PROJ=ON
cmake --build build --parallel $(nproc)
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```

## Quick Example (C++)

```cpp
#include <axis/axis.hpp>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    // Create source and destination meshes from descriptors
    auto src_mesh = axis::topology::MeshFactory::from_descriptor(src_descriptor);
    auto dst_mesh = axis::topology::MeshFactory::from_descriptor(dst_descriptor);

    // Generate interpolation weights
    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Bilinear;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);

    // Apply weights to a source field
    axis::solver::apply(matrix, src_field, dst_field);

    return 0;
}
```

## Quick Example (Python)

```python
import axis
import numpy as np

# Generate a named grid
src_mesh = axis.NamedGridRegistry.generate("O48")
dst_mesh = axis.NamedGridRegistry.generate("O96")

# Generate weights
config = {"method": "bilinear"}
matrix = axis.WeightGenerator.generate(src_mesh, dst_mesh, config)

# Apply to a field
src_field = np.ones(src_mesh.n_cells, dtype=np.float64)
dst_field = axis.apply(matrix, src_field)

# Verify partition of unity
assert np.allclose(dst_field, 1.0, atol=1e-14)
```

## Architecture Overview

AXIS is organized into four namespaces:

- `axis::topology` — Mesh construction and grid generation
- `axis::solver` — Weight generation, matrix storage, apply kernels
- `axis::detail` — GPU-portable primitives (spherical clipper, gnomonic projector)
- `axis::ingest` — Egress contracts (SCRIP format)

## Next Steps

- @ref mesh_construction — Learn about mesh creation from descriptors
- @ref weight_generation — Understand the interpolation methods
- @ref apply — Apply weights to fields (single, batch, distributed)
- @ref performance_tuning — Optimize for your hardware
