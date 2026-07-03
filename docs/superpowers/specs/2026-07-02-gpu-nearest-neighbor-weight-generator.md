# Spec: AXIS GPU-Resident Unstructured Nearest Neighbor Weight Generator

## 1. Overview & Motivation
Currently, unstructured Nearest Neighbor weight generation in AXIS (`WeightGenerator::generate_nearest`) is restricted to executing on the host (`Kokkos::HostSpace`). When a GPU execution/memory space (like `CudaSpace` or `HIPSpace`) is selected, the library falls back to computing centroids, constructing the ArborX BVH tree, querying neighbors, and assembling sparse COO factors on the CPU, before performing synchronous deep copies of the assembled arrays to the GPU.

This specification defines the native C++ GPU pipeline for unstructured Nearest Neighbor weight generation (`generate_nearest_device`). This avoids CPU-GPU round-trips and ensures end-to-end device-resident performance for unstructured grids.

---

## 2. Architecture & Implementation Strategy

We will introduce a new device-resident function `generate_nearest_device` in `libs/axis/src/solver/weight_generator.cpp`:

1.  **Device-Space Dispatch:**
    Inside `WeightGenerator::generate_nearest`, we will add a template dispatch check:
    ```cpp
    if constexpr (is_device_space_v<MemorySpace>) {
        return generate_nearest_device(src_mesh, dst_mesh, config);
    }
    ```

2.  **Centroid Calculation:**
    Compute cell centroids directly on the device using the existing, highly optimized `compute_cell_centroids_device` helper:
    ```cpp
    auto src_centroids = compute_cell_centroids_device(src_mesh);
    auto dst_centroids = compute_cell_centroids_device(dst_mesh);
    ```

3.  **ArborX Point Cloud & BVH Construction:**
    *   Construct a device View `src_points` containing unit-sphere 3D Cartesian coordinates ($X, Y, Z$) for spherical grids or 2D Cartesian coordinates for flat grids.
    *   Build the `ArborX::BoundingVolumeHierarchy` directly in the target `MemorySpace` on the device execution space:
        ```cpp
        ArborX::BoundingVolumeHierarchy tree(exec_inst, ArborX::Experimental::attach_indices(src_points));
        ```

4.  **Parallel Query Construction & Execution:**
    *   Construct `queries` View of size `n_dst` containing `ArborX::nearest(point, 1)` for each destination cell centroid.
    *   Execute the parallel query on the device execution space:
        ```cpp
        tree.query(exec_inst, queries, values, offsets);
        ```

5.  **COO Array Assembly & Compaction:**
    *   Allocate `factor_list`, `factor_row`, and `factor_col` Views of size `n_dst` on device.
    *   Execute a parallel kernel `assemble_nearest_coo_device` of size `n_dst`. For each destination cell $j$:
        *   If the cell has no neighbors (`offsets(j + 1) == offsets(j)`):
            *   If `config.unmapped == UnmappedAction::Error`, throw a runtime error or flag.
            *   Otherwise (Ignore), set row to a sentinel value or flag for compaction.
        *   If mapped, write $1.0$ weight to `factor_list`, $j$ to `factor_row`, and the nearest source index (`values(offsets(j)).index`) to `factor_col`.
    *   If any cell is unmapped, compact the Views to their final size.

---

## 3. Verification & Testing Strategy

*   **Unit & Regression Tests:**
    *   Ensure that running nearest neighbor weight generation on device (`CudaSpace` or `HIPSpace`) yields bitwise-identical interpolation results to those generated on host (`HostSpace`).
    *   Add a specific device-space nearest-neighbor verification test case.
*   **Property-Based Tests:**
    *   Verify that `PropHostDeviceEquivalence` and other property tests automatically cover and pass on the new device-resident nearest neighbor pipeline.
