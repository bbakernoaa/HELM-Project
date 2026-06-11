# Weight Generation {#weight_generation}

Weight generation computes the interpolation operator (sparse matrix) that maps source
fields to destination fields. AXIS v2 provides three interpolation methods with GPU-portable
execution paths.

## Interpolation Methods

### Bilinear (True Bilinear on the Sphere)

The default method for smooth fields. AXIS v2 uses gnomonic projection into a local
tangent plane centered on each source cell, then computes bilinear shape function
weights via Newton iteration. This avoids the lon/lat distortion of traditional
Cartesian bilinear interpolation.

```cpp
axis::solver::RegridConfig config;
config.method = axis::solver::RegridMethod::Bilinear;

axis::solver::WeightGenerator gen;
auto matrix = gen.generate(src_mesh, dst_mesh, config);
```

**Properties:**
- Affine exactness: reproduces `f(x,y,z) = a·x + b·y + c·z` within 1e-10
- Partition of unity: weights are non-negative and sum to 1.0
- Falls back to inverse-distance weighting if Newton fails to converge (20 iterations)

### Conservative 1st Order

Area-weighted remapping that preserves the global integral of the source field.
Uses spherical polygon clipping (Greiner-Hormann on the unit sphere) for exact
overlap area computation.

```cpp
config.method = axis::solver::RegridMethod::Conservative1stOrder;
config.normalization = axis::solver::Normalization::DstArea;  // or FracArea

auto matrix = gen.generate(src_mesh, dst_mesh, config);
```

**Properties:**
- Conservation: `Σ src·area_a·frac_a == Σ dst·area_b·frac_b` within 1e-12
- Weights are non-negative
- 1st-order accurate (constant fields reproduced exactly)

### Conservative 2nd Order

Extends 1st-order conservative with a linear gradient reconstruction within each
source cell. Achieves 2nd-order accuracy for smooth fields while maintaining
conservation.

```cpp
config.method = axis::solver::RegridMethod::Conservative2ndOrder;
config.use_limiter = true;  // Optional Barth-Jespersen monotonicity limiter

auto matrix = gen.generate(src_mesh, dst_mesh, config);
```

**Properties:**
- Linear exactness: reproduces linear fields within 1e-8
- Conservation preserved
- Optional monotonicity limiter prevents new extrema in destination
- Falls back to 1st-order for cells with < 3 neighbors

## Configuration Options

The `RegridConfig` struct controls weight generation behavior:

```cpp
struct RegridConfig {
    RegridMethod method = RegridMethod::Bilinear;
    Normalization normalization = Normalization::DstArea;
    UnmappedAction unmapped = UnmappedAction::Ignore;
    bool use_limiter = false;
    double pole_threshold_deg = 85.0;
};
```

### Normalization Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `DstArea` | `w_ij = overlap_ij / dst_area_j` | Standard conservative |
| `FracArea` | `w_ij = overlap_ij / (dst_area_j * frac_b_j)` | Partial coverage grids |

### Unmapped Destination Handling

| Action | Behavior |
|--------|----------|
| `Ignore` | Leave unmapped cells at their initial value |
| `Zero` | Set unmapped cells to 0.0 |
| `Error` | Throw `std::runtime_error` for unmapped cells |

## GPU Pipeline

When the mesh is on a device `MemorySpace`, weight generation executes entirely on
the GPU without host round-trips:

```cpp
// Device-resident meshes
auto src_dev = axis::topology::MeshFactory::from_descriptor<Kokkos::CudaSpace>(src_desc);
auto dst_dev = axis::topology::MeshFactory::from_descriptor<Kokkos::CudaSpace>(dst_desc);

// Weight generation runs on device
auto matrix = gen.generate(src_dev, dst_dev, config);
// matrix lives in CudaSpace — ready for device-side apply
```

The GPU pipeline:
1. ArborX BVH construction + spatial queries on device
2. SphericalClipper kernels on device (KOKKOS_FUNCTION)
3. COO assembly on device
4. COO-to-CSR conversion via device Kokkos sort + scan

## Internal Components

### SphericalClipper

Greiner-Hormann polygon clipping on the unit sphere. Handles concave polygons,
dateline-crossing cells, and degenerate intersections (vertex-on-edge, coincident
edges). Fixed-capacity `SphericalPolygon<MaxVerts>` avoids device allocation.

### GnomonicProjector

Forward/inverse gnomonic projection for true bilinear interpolation. Projects
source cell vertices and the target point into a local tangent plane, avoiding
pole singularities.

### GradientReconstructor

Least-squares gradient computation for Conservative2ndOrder. Operates over
face-adjacent neighbors with optional Barth-Jespersen monotonicity limiter.

### DatelineHandler

Detects and normalizes cells straddling the 180°/-180° longitude discontinuity
and cells containing geographic poles.

### DegenerateCellHandler

Detects zero-area cells, collapsed edges, and self-intersecting boundaries.
Excluded cells are recorded in a diagnostic report attached to the
InterpolationMatrix.
