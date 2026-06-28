# Spec: AXIS Tripolar Reflection Queries & MPI Distributed Weight Assembly

## 1. Overview & Motivation
To scale AXIS regridding to massive planetary grids run on advanced supercomputing clusters, we are adding two critical HPC features:
1.  **Tripolar Reflection Queries:** Speeds up target cell location across ORCA-style ocean grids by enforcing analytical coordinate reflections across folded seams, avoiding spatial search tree anomalies at polar singularities.
2.  **MPI Distributed Weight Assembly:** Integrates with the `libs/halo` library to enable multi-node clustered execution, building partitioned distributed sparse weight matrices for global earth system models.

---

## 2. Tripolar Reflection Queries (On-Device Fast-Path)

### A. Core Mathematical Reflection
When `detect_tripolar_grid` validates a folded northern boundary seam on an unstructured mesh, we register the boundary latitude $\phi_{seam}$ and a longitudinal reflection center $\lambda_{seam\_center}$ (often $180^\circ$ or $0^\circ$).

If a destination coordinate query exceeds $\phi_{seam}$, the solver intercepts the spatial lookup and analytically reflects the coordinates:
$$ \phi_{reflected} = 2 \cdot \phi_{seam} - \phi $$
$$ \lambda_{reflected} = \lambda_{seam\_center} + (\lambda_{seam\_center} - \lambda) $$

### B. Fast-Path Query Injection
Within the core spatial search loop (`WeightGenerator::generate` inside device execution kernels):
*   **Without Tripolar Fast-Path:** The ArborX BVH tree struggles near poles because spherical bounding boxes overlap significantly at polar convergence.
*   **With Tripolar Fast-Path:** The reflected coordinates instantly map to the proper mirrored source cells in $O(1)$ time, skipping anomalous polar tree-traversals entirely and boosting performance up to $10\times$.

---

## 3. MPI Distributed Weight Assembly (AXIS + HALO)

### A. Distributed Mesh Partitioning
The destination mesh is partitioned into local domains across MPI ranks. Each MPI rank holds its local destination cell coordinates.

### B. Interface Extensions
A new namespace `axis::distributed` is added to handle inter-node assembly using MPI Communicators:

```cpp
namespace axis::distributed {

/// @brief Generate an InterpolationMatrix distributed across MPI ranks.
///
/// This uses the HELM `halo` library to exchange mesh boundaries across ranks.
///
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
solver::InterpolationMatrix<MemorySpace> generate_distributed_weights(
    const topology::UnstructuredMesh<MemorySpace>& local_src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& local_dst_mesh,
    const solver::RegridConfig& config,
    MPI_Comm comm
);

} // namespace axis::distributed
```

### C. The Assembly Pipeline
1.  **Halo Identification:** Ranks identify destination bounding boxes that overlap local domain boundaries.
2.  **Coordinate Exchange:** Using `libs/halo` exchange patterns, neighboring ranks swap bounding box coordinates for source meshes.
3.  **Local BVH Search:** Ranks search both their local source meshes and the exchanged "ghost" halo source meshes.
4.  **Distributed Matrix Construction:** The generated sparse matrix resolves local row indices to globally-mapped column indices, ready for execution via `KokkosSparse::spmv` across the distributed cluster.

---

## 4. Verification & Testing Strategy
1.  **Tripolar Seam Exactness:** A solid-body analytical field test over a tripolar mesh, verifying that interpolation errors near the seam are identical (within $1.0 \times 10^{-12}$) with or without the fast path.
2.  **MPI Halo Boundary Validity:** Generates a synthetic mesh split across 2 MPI ranks (using `mpiexec -n 2`). Validates that the sum of the partial distributed weight matrices exactly equals a single-node serial baseline matrix.
