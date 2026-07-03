# Spec: AXIS GPU-Accelerated Second-Order Conservative Remapping (Gap G)

## 1. Overview & Motivation
Currently, unstructured Second-Order Conservative Remapping (`WeightGenerator::generate_conservative_2nd_order`) executes completely on the host (`HostSpace`). When compiled for GPU memory spaces (like `CudaSpace` or `HIPSpace`), the entire weight generation—including face-adjacency graph building, least-squares gradient reconstruction, and second-order weight correction—is executed sequentially/parallel on the CPU, with the final views deep-copied to the GPU.

This specification defines the implementation of a **Host-Device Hybrid GPU Pipeline** for `generate_conservative_2nd_order`. This offloads the heavy numerical kernels—least-squares gradient reconstruction, tridiagonal normal equations, and second-order weight corrections—to run natively in parallel on the GPU, while leveraging the CPU for complex, single-pass graph construction, optimizing HPC performance.

---

## 2. Design & Architecture

1.  **Host-Side Graph Building:**
    *   Construct the mesh face-adjacency CSR structures (`adj_offsets`, `adj_indices`) and unit-sphere Cartesian centroid coordinates (`src_centroids`) on the CPU (`HostSpace`).
2.  **Device Memory Uplift:**
    *   Deep-copy the centroids and adjacency CSR Views to the target GPU `MemorySpace` (`CudaSpace` or `HIPSpace`).
3.  **GPU-Parallel Gradient Reconstruction:**
    *   Uplift the synthetic coordinate fields (`field_x`, `field_y`, `field_z`) to device.
    *   Invoke the natively parallelized **`GradientReconstructor<MemorySpace>::compute` on the GPU** to solve the $3\times3$ least-squares Cramer equations and execute the Barth-Jespersen slope limiter in parallel on device threads.
4.  **GPU-Parallel Weight Correction:**
    *   Assemble and apply the second-order geometric offset corrections to the weight entries directly in device memory using parallel Kokkos kernels.
    *   Build and return the final `InterpolationMatrix<MemorySpace>` directly on device, eliminating any CPU-GPU synchronization.

---

## 3. Verification & Testing Strategy

*   **Unit & Property Tests:**
    *   Confirm that running second-order conservative weight generation on device (`CudaSpace` or `HIPSpace`) yields bitwise-identical results to host execution.
    *   Ensure all 363 C++ tests and 33 Python tests compile clean and pass with 100% success.
