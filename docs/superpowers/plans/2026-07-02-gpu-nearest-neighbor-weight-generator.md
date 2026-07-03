# AXIS GPU-Resident Unstructured Nearest Neighbor Weight Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a native, fully device-resident unstructured Nearest Neighbor weight generator (`generate_nearest_device`) in C++ to avoid CPU-GPU round-trips when running on GPUs.

**Architecture:** Create a device-space dispatcher inside `WeightGenerator::generate_nearest`. Build a fully device-parallel `generate_nearest_device_impl` that utilizes `compute_cell_centroids_device` to compute centroids on device, constructs an ArborX BVH tree on device, executes nearest-neighbor queries, and compiles COO factors directly on device via a parallel Kokkos kernel.

**Tech Stack:** C++20, Kokkos, ArborX, GoogleTest

## Global Constraints
- Must be fully device-resident: no Host-Device deep copies or synchronizations allowed inside the hot path of the query execution and factor assembly.
- Must produce results bitwise-identical to the host-resident nearest neighbor pipeline.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Declare and Dispatch `generate_nearest_device`

Add the forward declarations, the template dispatch checks, and explicit instantiations for the new `generate_nearest_device` pipeline.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:1800-1845`

**Interfaces:**
- Consumes: `WeightGenerator::generate_nearest`.
- Produces: Forward declarations and a compile-time device-space dispatch to `generate_nearest_device` when `is_device_space_v<MemorySpace>` is true.

- [ ] **Step 1: Declare the device-space functions**
  Add the forward declarations for `generate_nearest_device_impl` and `generate_nearest_device` at the top of the unstructured methods section in `libs/axis/src/solver/weight_generator.cpp`:
  ```cpp
  // Forward declaration: nearest-neighbor device pipeline
  template <int Dimension, class MemorySpace>
  InterpolationMatrix<MemorySpace> generate_nearest_device_impl(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                const RegridConfig &config);

  template <class MemorySpace>
  InterpolationMatrix<MemorySpace> generate_nearest_device(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                           const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                           const RegridConfig &config);
  ```

- [ ] **Step 2: Add device-space dispatch in `WeightGenerator::generate_nearest`**
  Modify `WeightGenerator::generate_nearest` in `libs/axis/src/solver/weight_generator.cpp` to route to the device pipeline when compiled with a device memory space:
  ```cpp
  template <class MemorySpace>
  InterpolationMatrix<MemorySpace> WeightGenerator::generate_nearest(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                     const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                     const RegridConfig &config) {
      // ── Device-space dispatch ──
      if constexpr (is_device_space_v<MemorySpace>) {
          return generate_nearest_device(src_mesh, dst_mesh, config);
      }

      // ── Host-space path ──
      auto src_reg = detail::detect_regular_grid(src_mesh);
      ...
  ```

- [ ] **Step 3: Run the build to verify compiling succeeds**
  Verify the modified file compiles successfully (though linking will fail until the definitions are added in Task 2):
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc)'`
  Expected: Linking/compiling errors for missing `generate_nearest_device` symbol if instantiated for device spaces, which is correct and defines our next task.

- [ ] **Step 4: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): add declarations and compile-time dispatch for generate_nearest_device"
  ```

---

### Task 2: Implement `generate_nearest_device_impl`

Implement the complete native C++ GPU pipeline for unstructured nearest-neighbor weight generation.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp`

**Interfaces:**
- Consumes: `compute_cell_centroids_device` and ArborX device query interfaces.
- Produces: Complete, device-resident `generate_nearest_device_impl` and `generate_nearest_device` functions.

- [ ] **Step 1: Write the device implementation block**
  Insert the `generate_nearest_device_impl` and `generate_nearest_device` definitions right after `generate_bilinear_device` or before `coastal_renormalize_and_extrapolate` in `libs/axis/src/solver/weight_generator.cpp`:
  ```cpp
  // ─────────────── Device-resident nearest-neighbor pipeline ───────────────────

  /// Device-space generate_nearest implementation.
  template <int Dimension, class MemorySpace>
  InterpolationMatrix<MemorySpace> generate_nearest_device_impl(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config) {
      using exec_space = execution_space_for_t<MemorySpace>;
      using Point = ArborX::Point<Dimension>;

      const auto n_src = static_cast<index_t>(src_mesh.n_cells());
      const auto n_dst = static_cast<index_t>(dst_mesh.n_cells());
      const auto csys = src_mesh.coord_system();

      // ── Compute source centroids on device ──
      auto src_centroids = compute_cell_centroids_device(src_mesh);
      auto dst_centroids = compute_cell_centroids_device(dst_mesh);

      // Build point cloud for BVH
      Kokkos::View<Point *, MemorySpace> src_points("src_points_device", n_src);
      Kokkos::parallel_for(
          "build_src_points_bvh", Kokkos::RangePolicy<exec_space>(0, n_src), KOKKOS_LAMBDA(const index_t i) {
              if constexpr (Dimension == 3) {
                  double lon = src_centroids(i, 0);
                  double lat = src_centroids(i, 1);
                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                      const double pi = 3.14159265358979323846;
                      lon = lon * pi / 180.0;
                      lat = lat * pi / 180.0;
                  }
                  double x = Kokkos::cos(lat) * Kokkos::cos(lon);
                  double y = Kokkos::cos(lat) * Kokkos::sin(lon);
                  double z = Kokkos::sin(lat);
                  src_points(i) = Point{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
              } else {
                  src_points(i) = Point{static_cast<float>(src_centroids(i, 0)), static_cast<float>(src_centroids(i, 1))};
              }
          });

      // Build ArborX BVH tree on device
      exec_space exec_inst;
      ArborX::BoundingVolumeHierarchy tree(exec_inst, ArborX::Experimental::attach_indices(src_points));

      // ── Build ArborX nearest queries ──
      Kokkos::View<decltype(ArborX::nearest(Point{}, 1)) *, MemorySpace> queries("queries", n_dst);
      if constexpr (Dimension == 3) {
          Kokkos::parallel_for(
              "build_nn_queries_3d", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const index_t j) {
                  double lon = dst_centroids(j, 0);
                  double lat = dst_centroids(j, 1);
                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                      const double pi = 3.14159265358979323846;
                      lon = lon * pi / 180.0;
                      lat = lat * pi / 180.0;
                  }
                  double x = Kokkos::cos(lat) * Kokkos::cos(lon);
                  double y = Kokkos::cos(lat) * Kokkos::sin(lon);
                  double z = Kokkos::sin(lat);
                  queries(j) = ArborX::nearest(Point{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}, 1);
              });
      } else {
          Kokkos::parallel_for(
              "build_nn_queries_2d", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const index_t j) {
                  queries(j) = ArborX::nearest(Point{static_cast<float>(dst_centroids(j, 0)), static_cast<float>(dst_centroids(j, 1))}, 1);
              });
      }

      // ── Execute query on device ──
      Kokkos::View<typename decltype(tree)::value_type *, MemorySpace> values("values", 0);
      Kokkos::View<int *, MemorySpace> query_offsets("offsets", 0);
      tree.query(exec_inst, queries, values, query_offsets);

      // ── Allocate COO Views ──
      Kokkos::View<double *, MemorySpace> factor_list("factor_list", n_dst);
      Kokkos::View<index_t *, MemorySpace> factor_row("factor_row", n_dst);
      Kokkos::View<index_t *, MemorySpace> factor_col("factor_col", n_dst);

      // ── Assemble COO factors on device ──
      Kokkos::parallel_for(
          "assemble_nn_weights_device", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const index_t j) {
              int begin = query_offsets(j);
              int end = query_offsets(j + 1);

              if (begin == end) {
                  factor_list(j) = 0.0;
                  factor_row(j) = static_cast<index_t>(-1); // Flag as unmapped
                  factor_col(j) = static_cast<index_t>(-1);
                  return;
              }

              factor_list(j) = 1.0;
              factor_row(j) = static_cast<index_t>(j);
              factor_col(j) = static_cast<index_t>(values(begin).index);
          });

      // ── Handle unmapped and compaction ──
      // Calculate final non-zero count
      Kokkos::View<index_t, MemorySpace> active_count("active_count");
      Kokkos::deep_copy(active_count, 0);

      Kokkos::parallel_for(
          "count_mapped_nn", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const index_t j) {
              if (factor_row(j) != static_cast<index_t>(-1)) {
                  Kokkos::atomic_increment(&active_count());
              }
          });

      index_t h_active_count = 0;
      Kokkos::deep_copy(h_active_count, active_count);

      // Throw if unmapped found and action is Error
      if (h_active_count < n_dst && config.unmapped == UnmappedAction::Error) {
          throw std::runtime_error("WeightGenerator::generate_nearest_device: unmapped destination cell detected.");
      }

      // Compact COO buffers on device if unmapped cells were skipped
      Kokkos::View<double *, MemorySpace> final_list;
      Kokkos::View<index_t *, MemorySpace> final_row;
      Kokkos::View<index_t *, MemorySpace> final_col;

      if (h_active_count == n_dst) {
          final_list = factor_list;
          final_row = factor_row;
          final_col = factor_col;
      } else {
          final_list = Kokkos::View<double *, MemorySpace>("factor_list_compact", h_active_count);
          final_row = Kokkos::View<index_t *, MemorySpace>("factor_row_compact", h_active_count);
          final_col = Kokkos::View<index_t *, MemorySpace>("factor_col_compact", h_active_count);

          Kokkos::View<index_t, MemorySpace> write_idx("write_idx");
          Kokkos::deep_copy(write_idx, 0);

          Kokkos::parallel_for(
              "compact_nn_weights", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const index_t j) {
                  if (factor_row(j) != static_cast<index_t>(-1)) {
                      auto idx = Kokkos::atomic_fetch_add(&write_idx(), 1);
                      final_list(idx) = factor_list(j);
                      final_row(idx) = factor_row(j);
                      final_col(idx) = factor_col(j);
                  }
              });
      }

      Kokkos::View<double *, MemorySpace> frac_a("frac_a", n_src);
      Kokkos::View<double *, MemorySpace> frac_b("frac_b", n_dst);
      Kokkos::View<double *, MemorySpace> area_a("area_a", n_src);
      Kokkos::View<double *, MemorySpace> area_b("area_b", n_dst);

      Kokkos::deep_copy(frac_a, 1.0);
      Kokkos::deep_copy(frac_b, 1.0);
      Kokkos::deep_copy(area_a, 0.0);

      auto dst_areas_dev = compute_cell_areas_device(dst_mesh, use_spherical);
      Kokkos::deep_copy(area_b, dst_areas_dev);

      return InterpolationMatrix<MemorySpace>(std::move(factor_list), std::move(factor_row), std::move(factor_col),
                                              std::move(frac_a), std::move(frac_b), std::move(area_a), std::move(area_b),
                                              n_src, n_dst);
  }

  template <class MemorySpace>
  InterpolationMatrix<MemorySpace> generate_nearest_device(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                           const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                           const RegridConfig &config) {
      const auto csys = src_mesh.coord_system();
      const bool use_spherical_nn = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);
      if (use_spherical_nn) {
          return generate_nearest_device_impl<3, MemorySpace>(src_mesh, dst_mesh, config);
      } else {
          return generate_nearest_device_impl<2, MemorySpace>(src_mesh, dst_mesh, config);
      }
  }
  ```

- [ ] **Step 2: Compile the complete C++ test suite and run**
  Verify the full suite compiles and passes cleanly:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100%

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): implement native unstructured generate_nearest_device pipeline on GPU"
  ```
