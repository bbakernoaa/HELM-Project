# Migration Guide: AXIS v1 to v2 {#migration_v1_to_v2}

This guide documents behavioral differences between AXIS v1 and v2, and how to
update your code for the new capabilities.

## Summary of Breaking Changes

| Component | v1 Behavior | v2 Behavior |
|-----------|-------------|-------------|
| Polygon clipping | Sutherland-Hodgman (host-only) | Greiner-Hormann on sphere (GPU-portable) |
| Bilinear interpolation | Cartesian lon/lat | True bilinear via gnomonic projection |
| Apply kernel | COO scatter-add only | CSR row-parallel (opt-in) |
| InterpolationMatrix | COO only | COO + CSR (`to_csr()`) |
| Batch apply | Not available | `batch_apply()` for rank-2 fields |
| Weight caching | Not available | `WeightCache::serialize/deserialize` |
| Cell masking | Not available | `set_mask()` on meshes |
| Degenerate cells | Silent failure | Detected + reported |

## Polygon Clipping: Sutherland-Hodgman → Greiner-Hormann

### What Changed

v1 used Sutherland-Hodgman polygon clipping in Cartesian lon/lat space, which is
host-only and inaccurate for cells near poles or straddling the dateline.

v2 uses Greiner-Hormann clipping directly on the unit sphere with great-circle arcs.
This is GPU-portable (all functions are `KOKKOS_FUNCTION`) and geometrically exact.

### Impact on Results

Conservative weight values will differ slightly from v1 due to:
- Correct spherical geometry (vs projected geometry)
- Proper handling of dateline-crossing cells
- Proper handling of polar cells

Differences are within the accuracy improvement tolerance — v2 is more correct.

### Migration Steps

No code changes required. The `WeightGenerator` uses the new clipper automatically.
If you compared against v1 reference values, expect differences up to ~1e-6 for
high-latitude cells.

## Bilinear Interpolation: Cartesian → True Bilinear

### What Changed

v1 computed bilinear interpolation weights in raw lon/lat Cartesian coordinates.
This is inaccurate for large cells and near poles due to coordinate distortion.

v2 uses gnomonic projection into a local tangent plane centered on the source cell
centroid, then performs bilinear interpolation in the projected space via Newton
iteration.

### Impact on Results

- Affine fields (`f(x,y,z) = a·x + b·y + c·z`) are now reproduced within 1e-10
- Near-pole interpolation is significantly more accurate
- Weights still satisfy partition of unity (sum to 1.0)
- Absolute weight values will differ from v1

### Migration Steps

No code changes required. Results are automatically more accurate. Update any
reference-value tests to use v2 baselines.

## CSR Storage and Apply

### What Changed

v1 stored interpolation weights in COO format only. The apply kernel used
scatter-add with atomic operations.

v2 adds optional CSR format via `to_csr()`. When CSR is active, `solver::apply`
dispatches to a row-parallel kernel with no atomics.

### Migration Steps

```cpp
// v1 (still works in v2)
auto matrix = gen.generate(src_mesh, dst_mesh, config);
axis::solver::apply(matrix, src_field, dst_field);

// v2 optimization (opt-in)
matrix.to_csr();  // One-time conversion
axis::solver::apply(matrix, src_field, dst_field);  // Faster apply
```

No API break — COO apply remains the default. CSR is opt-in.

## Batch Apply (New in v2)

### What Changed

v1 required looping over variables and calling `apply` individually.
v2 provides `batch_apply` for rank-2 field views.

### Migration Steps

```cpp
// v1: manual loop
for (int v = 0; v < n_vars; ++v) {
    auto src_v = Kokkos::subview(src_2d, Kokkos::ALL, v);
    auto dst_v = Kokkos::subview(dst_2d, Kokkos::ALL, v);
    axis::solver::apply(matrix, src_v, dst_v);
}

// v2: single call
axis::solver::batch_apply(matrix, src_2d, dst_2d);
```

## Weight Caching (New in v2)

### What Changed

v1 had no built-in weight persistence. Users relied on external serialization.
v2 provides `WeightCache` with a versioned binary format.

### Migration Steps

If you had custom weight serialization, consider switching to the built-in format:

```cpp
// Replace custom serialization with:
auto buf_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
std::vector<char> buf(buf_size);
axis::solver::WeightCache::serialize(matrix, buf.data(), buf.size());
```

Existing custom formats continue to work — `WeightCache` is additive.

## Cell Masking (New in v2)

### What Changed

v1 had no masking support. Land/ocean boundaries required pre-filtering meshes.
v2 supports per-cell masks that exclude cells from weight generation and apply.

### Migration Steps

```cpp
// v1: pre-filter mesh (complex, error-prone)
auto ocean_mesh = filter_cells(full_mesh, ocean_indices);

// v2: attach mask (cleaner)
full_mesh.set_mask(ocean_mask);
auto matrix = gen.generate(full_mesh, dst_mesh, config);
```

## Degenerate Cell Detection (New in v2)

### What Changed

v1 silently produced incorrect weights for degenerate cells (zero area, collapsed
edges, self-intersecting). v2 detects and excludes them with diagnostic reporting.

### Impact

If your mesh contains degenerate cells, v2 may produce different results because
those cells are now excluded. A warning is emitted when >1% of cells are degenerate.

### Migration Steps

No code changes required. Check diagnostic reports if results differ from v1:

```cpp
auto diag = matrix.diagnostic_report();
if (diag.n_degenerate > 0) {
    // Log excluded cells
    for (auto idx : diag.excluded_indices) { /* ... */ }
}
```

## GPU Pipeline (New in v2)

### What Changed

v1 weight generation was host-only. v2 supports full device-resident execution
via Kokkos MemorySpace templates.

### Migration Steps

```cpp
// v1: host-only
auto matrix = gen.generate(src_mesh, dst_mesh, config);

// v2: device-resident (opt-in via MemorySpace)
auto src_dev = MeshFactory::from_descriptor<Kokkos::CudaSpace>(src_desc);
auto dst_dev = MeshFactory::from_descriptor<Kokkos::CudaSpace>(dst_desc);
auto matrix = gen.generate(src_dev, dst_dev, config);
```

## Compatibility Summary

| Feature | v1 Code Works in v2? | Behavior Change? |
|---------|---------------------|------------------|
| `WeightGenerator::generate` | ✅ Yes | Slightly different weights (more accurate) |
| `solver::apply` (COO) | ✅ Yes | No change |
| `InterpolationMatrix` construction | ✅ Yes | No change |
| `MeshFactory::from_descriptor` | ✅ Yes | No change |
| Conservative reference values | ⚠️ Values differ | More accurate spherical geometry |
| Bilinear reference values | ⚠️ Values differ | True bilinear vs Cartesian |
