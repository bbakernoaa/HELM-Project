# Applying Interpolation Weights {#apply}

Once an `InterpolationMatrix` is generated, use `solver::apply` or `solver::batch_apply`
to interpolate fields from the source grid to the destination grid.

## Single-Field Apply

The basic apply function performs sparse matrix-vector multiplication (SpMV):

```cpp
#include <axis/solver/apply.hpp>

// src_field: Kokkos::View<const double*, MemorySpace> of size n_src
// dst_field: Kokkos::View<double*, MemorySpace> of size n_dst

axis::solver::apply(matrix, src_field, dst_field);
```

### COO vs CSR Dispatch

`solver::apply` automatically dispatches based on the matrix format:

- **COO path** — scatter-add with atomics (default after `generate()`)
- **CSR path** — row-parallel TeamPolicy, no atomics (after `to_csr()`)

```cpp
// Convert to CSR for better apply performance
matrix.to_csr();
assert(matrix.is_csr());

// CSR apply: row-parallel, no atomics
axis::solver::apply(matrix, src_field, dst_field);
```

Both paths produce identical results (agreement within 1e-15).

## Batch Apply (Multi-Variable)

When regridding multiple variables with the same grid pair, batch apply amortizes
kernel launch overhead:

```cpp
#include <axis/solver/apply.hpp>

// src_fields: Kokkos::View<const double**, MemorySpace> shape [n_src, n_vars]
// dst_fields: Kokkos::View<double**, MemorySpace> shape [n_dst, n_vars]

axis::solver::batch_apply(matrix, src_fields, dst_fields);
```

### Batch Apply Kernels

- **CSR path**: Outer TeamPolicy over destination rows, inner TeamVectorRange
  over variables — maximum parallelism for high variable counts
- **COO path**: parallel_for over nnz × n_vars

### Validation

`batch_apply` validates array extents at runtime:

```cpp
// Throws std::invalid_argument if:
//   src_fields.extent(0) != matrix.n_src()
//   dst_fields.extent(0) != matrix.n_dst()
//   src_fields.extent(1) != dst_fields.extent(1)
```

## Distributed Apply

For domain-decomposed grids, AXIS generates a `HaloPattern` that describes which
remote source cells are needed. The caller (HALO library) performs the MPI exchange,
and AXIS applies weights to the gathered buffer:

```cpp
#include <axis/solver/halo_pattern.hpp>

// 1. Generate the exchange plan
auto halo = axis::solver::build_halo_pattern(matrix, local_src_range, local_dst_range);

// 2. Caller gathers remote data into a buffer using HALO
//    (AXIS does not perform MPI — Tier 1 isolation)

// 3. Apply with the gathered source field
axis::solver::apply(matrix, gathered_src_field, local_dst_field);
```

## Example: Complete Workflow

```cpp
#include <axis/axis.hpp>
#include <Kokkos_Core.hpp>

void regrid_temperature(
    const axis::topology::UnstructuredMesh<>& src_mesh,
    const axis::topology::UnstructuredMesh<>& dst_mesh,
    Kokkos::View<const double*> temperature_src,
    Kokkos::View<double*> temperature_dst)
{
    // Generate bilinear weights
    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Bilinear;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);

    // Convert to CSR for efficient apply
    matrix.to_csr();

    // Apply
    axis::solver::apply(matrix, temperature_src, temperature_dst);
}
```

## Example: Batch Apply for Multiple Variables

```cpp
void regrid_atmos_state(
    const axis::solver::InterpolationMatrix<>& matrix,
    Kokkos::View<const double**> state_src,  // [n_src, 5] (T, u, v, q, ps)
    Kokkos::View<double**> state_dst)        // [n_dst, 5]
{
    // Single kernel launch for all 5 variables
    axis::solver::batch_apply(matrix, state_src, state_dst);
}
```

## Performance Notes

- Convert to CSR (`to_csr()`) before apply if reusing the matrix across timesteps
- Use `batch_apply` when regridding > 1 variable to amortize kernel overhead
- On GPU, ensure source fields are already in device memory to avoid transfers
- See @ref performance_tuning for detailed optimization guidance
