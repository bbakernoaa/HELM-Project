# AXIS GPU-Accelerated Second-Order Conservative Remapping (Gap G) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Host-Device Hybrid GPU pipeline for unstructured second-order conservative weight generation, offloading least-squares gradient reconstructions and weight corrections to device-parallel Kokkos kernels.

**Architecture:** Update `WeightGenerator::generate_conservative_2nd_order` to check if `MemorySpace` is a device space. If so, build the face-adjacency graph on host, copy all centroids and adjacency Views to device, invoke `GradientReconstructor<MemorySpace>::compute` on device, apply second-order corrections in a parallel Kokkos kernel on device, and assemble the final `InterpolationMatrix<MemorySpace>`.

**Tech Stack:** C++20, Kokkos

## Global Constraints
- Gradient reconstruction and second-order weight corrections must execute natively on the GPU execution space without Host-Device deep copies or synchronizations in the hot path.
- Must produce results bitwise-identical to the host-only second-order conservative pipeline.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Implement Host-Device Hybrid GPU pipeline for `generate_conservative_2nd_order`

Refactor second-order conservative weight generation to support GPU execution via a Host-Device hybrid pipeline.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:3835-4300`

**Interfaces:**
- Consumes: `GradientReconstructor<MemorySpace>::compute`.
- Produces: GPU-accelerated `WeightGenerator::generate_conservative_2nd_order`.

- [ ] **Step 1: Implement the template branching in `generate_conservative_2nd_order`**
  Inside `WeightGenerator::generate_conservative_2nd_order`, split into two branches: a pure-host path (if `is_host_space_v<MemorySpace>`) and a hybrid GPU path (if `is_device_space_v<MemorySpace>`).

  For the hybrid GPU path:
  *   Construct `src_centroids`, `adj_offsets_kv`, `adj_indices_kv` and coordinate fields on host first (as we already read).
  *   Allocate device Views of type `MemorySpace`:
      ```cpp
      Kokkos::View<double *[3], MemorySpace> d_centroids("d_centroids", n_src);
      Kokkos::View<index_t *, MemorySpace> d_adj_offsets("d_adj_offsets", adj_offsets_kv.extent(0));
      Kokkos::View<index_t *, MemorySpace> d_adj_indices("d_adj_indices", adj_indices_kv.extent(0));
      Kokkos::View<double *, MemorySpace> d_field_x("d_field_x", n_src);
      Kokkos::View<double *, MemorySpace> d_field_y("d_field_y", n_src);
      Kokkos::View<double *, MemorySpace> d_field_z("d_field_z", n_src);

      Kokkos::deep_copy(d_centroids, src_centroids);
      Kokkos::deep_copy(d_adj_offsets, adj_offsets_kv);
      Kokkos::deep_copy(d_adj_indices, adj_indices_kv);
      Kokkos::deep_copy(d_field_x, field_x);
      Kokkos::deep_copy(d_field_y, field_y);
      Kokkos::deep_copy(d_field_z, field_z);
      ```
  *   Invoke the GPU-parallelized `GradientReconstructor<MemorySpace>::compute` on device!
      ```cpp
      Kokkos::View<double *[3], MemorySpace> d_grad_x("d_grad_x", n_src);
      Kokkos::View<double *[3], MemorySpace> d_grad_y("d_grad_y", n_src);
      Kokkos::View<double *[3], MemorySpace> d_grad_z("d_grad_z", n_src);

      GradientReconstructor<MemorySpace>::compute(d_field_x, d_centroids, d_adj_offsets, d_adj_indices, d_grad_x, config.use_limiter);
      GradientReconstructor<MemorySpace>::compute(d_field_y, d_centroids, d_adj_offsets, d_adj_indices, d_grad_y, config.use_limiter);
      GradientReconstructor<MemorySpace>::compute(d_field_z, d_centroids, d_adj_offsets, d_adj_indices, d_grad_z, config.use_limiter);
      ```
  *   Assemble and apply the second-order geometric offset weight corrections directly on device in a parallel Kokkos kernel!
      First, deep copy the assembled overlap entries to a device View of type `MemorySpace`:
      ```cpp
      const std::size_t nnz = overlap_entries.size();
      Kokkos::View<OverlapEntry *, MemorySpace> d_overlap_entries("d_overlap_entries", nnz);
      auto h_overlap_entries = Kokkos::create_mirror_view(d_overlap_entries);
      for (std::size_t idx = 0; idx < nnz; ++idx) {
          h_overlap_entries(idx) = overlap_entries[idx];
      }
      Kokkos::deep_copy(d_overlap_entries, h_overlap_entries);
      ```
      Then, run the Kokkos parallel kernel to calculate the corrected weights:
      ```cpp
      using exec_space = typename MemorySpace::execution_space;
      Kokkos::View<double *, MemorySpace> factor_list("factor_list", nnz);
      Kokkos::View<index_t *, MemorySpace> factor_row("factor_row", nnz);
      Kokkos::View<index_t *, MemorySpace> factor_col("factor_col", nnz);

      Kokkos::parallel_for(
          "ApplySecondOrderCorrection", Kokkos::RangePolicy<exec_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t idx) {
              const auto &entry = d_overlap_entries(idx);
              const auto j = entry.dst_cell;
              const auto src_i = entry.src_cell;
              const double overlap_area = entry.area;

              // Extract geometric centroid offsets
              double off_x = entry.off_x;
              double off_y = entry.off_y;
              double off_z = entry.off_z;

              // Read computed gradients from device
              double gx_x = d_grad_x(src_i, 0);
              double gx_y = d_grad_x(src_i, 1);
              double gx_z = d_grad_x(src_i, 2);

              double gy_x = d_grad_y(src_i, 0);
              double gy_y = d_grad_y(src_i, 1);
              double gy_z = d_grad_y(src_i, 2);

              double gz_x = d_grad_z(src_i, 0);
              double gz_y = d_grad_z(src_i, 1);
              double gz_z = d_grad_z(src_i, 2);

              // 2nd order correction factor: 1.0 + grad * offset
              double correction = 1.0 + (gx_x * off_x + gy_x * off_y + gz_x * off_z);
              
              // Apply correction to the base area fraction
              double w = (overlap_area / dst_areas_dev(j)) * correction;

              factor_list(idx) = w;
              factor_row(idx) = static_cast<index_t>(j);
              factor_col(idx) = static_cast<index_t>(src_i);
          });
      ```

- [ ] **Step 2: Run complete build and test suite**
  Verify everything compiles clean and all 363 C++ tests pass perfectly:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100%

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): implement Host-Device Hybrid GPU pipeline for Conservative2ndOrder remapping"
  ```
