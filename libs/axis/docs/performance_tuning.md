# Performance Tuning {#performance_tuning}

This guide covers optimization strategies for getting the most out of AXIS on both
CPU and GPU hardware.

## CSR vs COO Apply

The single most impactful optimization is converting the InterpolationMatrix to CSR
format before apply:

```cpp
// After generation (COO format)
auto matrix = gen.generate(src_mesh, dst_mesh, config);

// Convert once — reuse across timesteps
matrix.to_csr();
```

**Why CSR is faster:**
- Row-parallel execution: one thread-team per destination row, no atomics
- Better cache locality: sequential memory access per row
- Reduced synchronization overhead vs atomic scatter in COO

**When to convert:**
- Always convert if the matrix will be used more than once
- Skip conversion for one-shot regridding (the `to_csr()` cost is O(nnz·log(nnz)))

## Batch Apply

When regridding multiple variables, always use `batch_apply`:

```cpp
// BAD: N kernel launches, N synchronization barriers
for (int v = 0; v < n_vars; ++v) {
    axis::solver::apply(matrix, src_fields[v], dst_fields[v]);
}

// GOOD: Single kernel launch for all variables
axis::solver::batch_apply(matrix, src_2d, dst_2d);
```

**Performance benefits:**
- Amortized kernel launch overhead (significant on GPU)
- Better memory bandwidth utilization (row metadata loaded once per destination cell)
- Improved thread occupancy on GPU

## Weight Caching

For multi-timestep simulations, cache computed weights:

```cpp
// Compute once
auto matrix = gen.generate(src_mesh, dst_mesh, config);
matrix.to_csr();

// Serialize for subsequent runs
auto buf_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
std::vector<char> cache(buf_size);
axis::solver::WeightCache::serialize(matrix, cache.data(), cache.size());
// Persist cache via AMIO or filesystem
```

**Savings:** Weight generation is typically 100-1000x more expensive than apply.
Caching eliminates regeneration cost for static grid pairs.

## GPU Pipeline

For GPU execution, ensure data residency:

```cpp
// Allocate meshes in device memory from the start
auto src_dev = MeshFactory::from_descriptor<Kokkos::CudaSpace>(src_desc);
auto dst_dev = MeshFactory::from_descriptor<Kokkos::CudaSpace>(dst_desc);

// Weight generation runs entirely on device
auto matrix = gen.generate(src_dev, dst_dev, config);

// Fields should already be on device
Kokkos::View<double*, Kokkos::CudaSpace> src_field("src", n_src);
Kokkos::View<double*, Kokkos::CudaSpace> dst_field("dst", n_dst);

// Apply runs on device — no host-device transfer
axis::solver::apply(matrix, src_field, dst_field);
```

**Avoid:**
- Allocating temporary host buffers
- Deep-copying fields between host and device per timestep
- Mixing host and device memory spaces in a single operation

## Memory Space Selection

| Scenario | Recommended MemorySpace |
|----------|------------------------|
| Single-node CPU | `Kokkos::HostSpace` |
| GPU-accelerated | `Kokkos::CudaSpace` / `Kokkos::HIPSpace` |
| Unified memory (debugging) | `Kokkos::CudaUVMSpace` |

## Kokkos Execution Policy Tuning

AXIS uses TeamPolicy for CSR apply. Default team sizes work well, but you can
influence Kokkos behavior via environment variables:

```bash
# Set number of threads for OpenMP backend
export OMP_NUM_THREADS=16

# Kokkos GPU block size (CUDA/HIP)
export KOKKOS_NUM_THREADS=256
```

## Grid Size Considerations

| Grid Resolution | Typical nnz | Weight Gen Time | Apply Time |
|----------------|-------------|-----------------|------------|
| O48 → O96 | ~100K | ~10 ms | ~0.1 ms |
| O96 → O320 | ~1M | ~100 ms | ~1 ms |
| O320 → O1280 | ~50M | ~5 s | ~50 ms |

## Profiling

Use Kokkos profiling tools to identify bottlenecks:

```bash
# Enable Kokkos simple kernel timing
export KOKKOS_PROFILE_LIBRARY=libkp_kernel_timer.so

# Run your application
./my_app

# View timing output
```

## Checklist for Production

1. ✅ Convert to CSR after weight generation
2. ✅ Use `batch_apply` for multiple variables
3. ✅ Cache weights for repeated grid pairs
4. ✅ Keep data on device if using GPU
5. ✅ Set appropriate thread/team counts for your hardware
6. ✅ Profile to identify actual bottlenecks before optimizing
