# Design Document: Structured Grid Creation and Weight Generation Optimization

## Overview

This specification details the optimizations for grid creation (StructuredGrid to UnstructuredMesh conversion) and interpolation weight generation (nearest, bilinear, conservative) across both rectilinear (uniform and non-uniform) and curvilinear grids in the AXIS micro-library.

The goal is to deliver significant performance gains on both CPU and GPU platforms without compromising precision or mathematical accuracy. To do this, we leverage explicit Kokkos performance portability patterns, ArborX spatial acceleration trees, and KokkosKernels sparse matrix-vector multiplication (`SpMV`).

---

## 1. Grid Creation Optimizations (`StructuredGrid` ➔ `UnstructuredMesh`)

Converting logically rectangular grids to the internal unstructured finite-element format involves node coordinate generation and connectivity indexing.

### 1.1 Multi-Dimensional Range Policy in `to_unstructured`
*   **Current State:** Logical cells are indexed using a flat 1D range policy. This requires computing 2D $(i, j)$ indices via expensive integer division (`/`) and modulo (`%`) operations inside the parallel kernel on every thread.
*   **Optimization:** Replace the 1D range policy with `Kokkos::MDRangePolicy<Kokkos::Rank<2>>` over the grid dimensions $(ni, nj)$. This allows Kokkos to map threads directly to the logical grid dimensions and enables compilers to optimize index arithmetic, vectorizing loops and eliminating division/modulo overhead.

### 1.2 Team-Shared Memory for Corner Synthesis (`synthesize_corners`)
*   **Current State:** If cell corner coordinates are not provided, they are synthesized on the fly. Each corner coordinate is calculated by taking the average of up to 4 adjacent cell center coordinates. In a naive parallel implementation, this results in up to 4 redundant global memory loads for each cell center.
*   **Optimization:** Rewrite the synthesis kernel using a `Kokkos::TeamPolicy`. Threads within a team cooperatively load cell center coordinates into team-shared memory (scratchpad / L1 cache). Individual corner calculations then read from this low-latency shared cache, reducing global memory bandwidth consumption by up to $4\times$.

---

## 2. Non-Uniform Rectilinear Grid Weight Generation

Non-uniform rectilinear grids have straight coordinate lines parallel to the axes, but with varying grid spacing (e.g., Gaussian grids). They do not require global BVH search.

### 2.1 Fast-Path 1D Binary Search for Nearest & Bilinear
*   **Cell Location:** For any destination centroid $(lon_d, lat_d)$, locate the enclosing source cell $(i, j)$ by running two independent 1-D binary searches (`Kokkos::upper_bound`) on the 1-D source coordinate arrays ($lon^{src}$ and $lat^{src}$). This reduces cell location complexity to $O(\log ni + \log nj)$, bypassing the need for any global BVH construction.
*   **Bilinear Weight Arithmetic:** Compute bilinear weights analytically from logical fractional offsets inside the 1D cell bounds:
    $$tx = \frac{lon_d - lon_i}{lon_{i+1} - lon_i}, \quad ty = \frac{lat_d - lat_j}{lat_{j+1} - lat_j}$$
    $$w_{00} = (1-tx)(1-ty), \quad w_{10} = tx(1-ty), \quad w_{01} = (1-tx)ty, \quad w_{11} = tx\cdot ty$$
    This is highly vectorizable and completely bypasses local tangent-plane Newton iterations.

### 2.2 Analytical Overlap Conservative Weight Generation
*   **Overlapping Index Extents:** Find the overlapping source cell index bounds $[i_{min}, i_{max}] \times [j_{min}, j_{max}]$ for a given destination cell by binary-searching the destination cell's corner bounds against the source corner longitude and latitude coordinate arrays.
*   **Axis-Aligned Overlaps:** Run a localized parallel nested loop over this index bounding box to compute intersection areas analytically using 1-D interval overlaps:
    $$\text{overlap}_{ij} = \max(0, \min(lon^{src}_{i+1}, lon^{dst}_{d+1}) - \max(lon^{src}_i, lon^{dst}_d)) \times \max(0, \min(lat^{src}_{j+1}, lat^{dst}_{d+1}) - \max(lat^{src}_j, lat^{dst}_d))$$
*   **CSR Packing:** Gather the calculated overlaps, normalize according to the selected policy (`DstArea` or `FracArea`), and assemble into a CSR matrix.

---

## 3. Curvilinear Grid Weight Generation

Curvilinear grids have curved coordinate paths but retain logical rectangular connectivity. They require spatial indexing, where we leverage ArborX and KokkosKernels.

### 3.1 Parallel ArborX BVH Construction
*   **AABB Calculation:** In parallel on the target `MemorySpace` (CPU or GPU device), compute axis-aligned bounding boxes (AABBs) for each logically rectangular curvilinear cell $(i, j)$ using its four corner nodes.
*   **BVH Tree:** Construct an `ArborX::BoundingVolumeHierarchy` on-device over these AABBs. ArborX is fully Kokkos-backed and builds spatial trees with zero host-device round-trips.

### 3.2 Nearest and Bilinear with Stencil-Walking Fallback
*   **ArborX Query:** For each destination point, perform an ArborX `intersects` query to find candidate source cells whose bounding boxes overlap the point.
*   **Local Stencil Verification:** For the 1 or 2 returned candidate cells, perform localized bilinear quadrilateral containment checks.
*   **Gnomonic Projection & Newton Iteration:** For the containing cell, project coordinates into the local tangent plane and calculate exact bilinear shape-function weights using Newton iteration.

### 3.3 Conservative Overlap Area Computation
*   **Candidate Overlap Queries:** For each destination cell, query the ArborX BVH with its cell bounding box to retrieve overlapping source cell candidates.
*   **Spherical Polygon Clipping:** Clip the intersecting candidate pairs on the unit sphere using the `SphericalClipper` (Greiner-Hormann polygon clipping) to obtain exact overlap areas.
*   **Normalization & Matrix Assembly:** Normalize the overlap areas and construct the weight matrix.

---

## 4. Sparse Matrix Apply via KokkosKernels

*   **CSR Conversion:** Once interpolation weights are generated (in COO format), convert them to CSR format using parallel sorts and prefix-scans in the target `MemorySpace`.
*   **KokkosSparse spmv:** The parallel matrix apply step is dispatched directly to **`KokkosSparse::spmv`** (leveraging vendor-optimized BLAS libraries where available, such as cuSPARSE on CUDA, rocSPARSE on HIP, or highly optimized OpenMP threads on CPU).

---

## 5. Verification and Validation Plan

### 5.1 Analytical and Exactness Verification
*   **Partition of Unity:** Verify that row sums equal $1.0$ within roundoff for all mapped destination cells.
*   **Bilinear Exactness:** Verify that linear fields $f(x, y) = a\cdot x + b\cdot y + c$ are reconstructed exactly (within $10^{-10}$) on both non-uniform rectilinear and curvilinear grids.
*   **Conservation:** For conservative regridding, verify that the total source integral equals the destination integral within $10^{-12}$.

### 5.2 Performance Benchmarking
*   Compare performance (build time + weight generation time + apply time) against the unstructured mesh fallback path to ensure speedups of at least:
    *   $10\times - 50\times$ for non-uniform rectilinear grids (due to $O(\log N)$ binary search and analytical overlaps).
    *   $2\times - 5\times$ for curvilinear grids (due to localized candidate filtering and optimized grid conversion).
