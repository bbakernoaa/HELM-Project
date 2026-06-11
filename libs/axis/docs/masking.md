# Cell Masking {#masking}

Cell masking allows source and destination cells to be marked as inactive, enabling
land/ocean boundary handling and exclusion of invalid data regions from interpolation.

## Mask Representation

A cell mask is a `Kokkos::View<int*, MemorySpace>` where:
- `0` — masked (inactive, excluded from interpolation)
- Non-zero — active (participates in interpolation)

## Setting a Mask

Masks are attached to `UnstructuredMesh` objects before weight generation:

```cpp
#include <axis/topology/unstructured_mesh.hpp>

// Create a mask (1 = ocean, 0 = land)
Kokkos::View<int*> src_mask("src_mask", src_mesh.n_cells());
Kokkos::parallel_for(src_mesh.n_cells(), KOKKOS_LAMBDA(int i) {
    src_mask(i) = is_ocean(i) ? 1 : 0;
});

// Attach mask to mesh
src_mesh.set_mask(src_mask);
```

## Effect on Weight Generation

### Source Masking

When source cells are masked:
1. Masked source cells are excluded from BVH queries
2. No weight entries reference masked source cells
3. Destination cells that partially overlapped masked sources get renormalized weights

```cpp
// Source mask: exclude land cells
src_mesh.set_mask(ocean_mask);

// Generate weights — land cells contribute nothing
auto matrix = gen.generate(src_mesh, dst_mesh, config);
```

### Destination Masking

When destination cells are masked:
1. No weight entries are generated for masked destination rows
2. The InterpolationMatrix contains zero entries for those rows
3. `solver::apply` skips writing to masked cells (values unchanged)

```cpp
// Destination mask: only interpolate to ocean cells
dst_mesh.set_mask(ocean_mask);

auto matrix = gen.generate(src_mesh, dst_mesh, config);
```

### Renormalization

When source masking causes partial coverage, destination weights are renormalized:

```
Original weights (full coverage):  w_j = overlap_j / dst_area
With masking (partial coverage):   w_j = overlap_j / (dst_area * coverage_fraction)
```

The `frac_b` array reflects the actual coverage fraction after masking.

## Interaction with Unmapped Handling

When masking causes an unmasked destination cell to have zero coverage:

| `RegridConfig::unmapped` | Behavior |
|--------------------------|----------|
| `Ignore` | Leave destination value unchanged |
| `Zero` | Set destination value to 0.0 |
| `Error` | Throw `std::runtime_error` |

Masked destination cells never trigger the unmapped error, regardless of the setting.

## Effect on Apply

```cpp
// Apply respects destination mask
axis::solver::apply(matrix, src_field, dst_field);
// Masked destination cells in dst_field are unchanged
```

## Example: Ocean-Only Regridding

```cpp
#include <axis/axis.hpp>

void regrid_sst(
    axis::topology::UnstructuredMesh<>& src_mesh,
    axis::topology::UnstructuredMesh<>& dst_mesh,
    Kokkos::View<int*> src_ocean_mask,
    Kokkos::View<int*> dst_ocean_mask,
    Kokkos::View<const double*> sst_src,
    Kokkos::View<double*> sst_dst)
{
    // Mask land on both grids
    src_mesh.set_mask(src_ocean_mask);
    dst_mesh.set_mask(dst_ocean_mask);

    // Generate weights — only ocean-to-ocean
    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Conservative1stOrder;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);

    // Apply — land cells in sst_dst retain their initial values
    axis::solver::apply(matrix, sst_src, sst_dst);
}
```

## Degenerate Cell Interaction

Cells flagged as degenerate by the `DegenerateCellHandler` are automatically excluded
from weight generation (similar to being masked). Their exclusion is recorded in the
InterpolationMatrix diagnostic report. This is independent of user-supplied masks.
