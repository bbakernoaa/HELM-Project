# AXIS Spherical Unit-Vector Centroid Calculations (Gap D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor both host-side and device-side unstructured cell centroid computation helper functions to perform 3D unit-vector spherical averaging on spherical coordinate meshes. This eliminates dateline-discontinuity errors and coordinates wrap-around errors.

**Architecture:** Modify `compute_cell_centroids_xy` (Host) and `compute_cell_centroids_device` (Device). If the coordinate system is `SphericalDeg` or `SphericalRad`, convert cell vertices to 3D unit-vectors $(X, Y, Z)$ on the sphere, sum and average them, and then convert the average back to geographical coordinates, wrapping the longitude safely to $[0, 360)$ degrees or $[0, 2\pi)$ radians.

**Tech Stack:** C++20, Kokkos

## Global Constraints
- Must use standard, thread-safe, allocation-free, and stack-safe C++/Kokkos math functions (`Kokkos::cos`, `Kokkos::sin`, `std::cos`, `std::sin`, etc.).
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Refactor Host-Side `compute_cell_centroids_xy`

Implement the 3D unit-vector spherical averaging algorithm for host-side centroid calculations.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:61-93`

**Interfaces:**
- Consumes: `UnstructuredMesh::coord_system`.
- Produces: Corrected, dateline-safe host-side centroids in `compute_cell_centroids_xy`.

- [ ] **Step 1: Replace host-side centroid implementation**
  Modify `compute_cell_centroids_xy` in `libs/axis/src/solver/weight_generator.cpp`:
  ```cpp
  template <class MemorySpace>
  void compute_cell_centroids_xy(const topology::UnstructuredMesh<MemorySpace> &mesh, Kokkos::View<double *, Kokkos::HostSpace> &cx_out,
                                 Kokkos::View<double *, Kokkos::HostSpace> &cy_out) {
      const auto n_cells = mesh.n_cells();
      const auto coords = mesh.node_coords();    // [n_nodes, ndim]
      const auto offsets = mesh.conn_offsets();  // [n_cells + 1]
      const auto indices = mesh.conn_indices();  // [nnz]
      const auto csys = mesh.coord_system();
      const bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);

      cx_out = Kokkos::View<double *, Kokkos::HostSpace>("cx", n_cells);
      cy_out = Kokkos::View<double *, Kokkos::HostSpace>("cy", n_cells);

      for (std::size_t c = 0; c < n_cells; ++c) {
          auto start = static_cast<std::size_t>(offsets[c]);
          auto end = static_cast<std::size_t>(offsets[c + 1]);
          auto n_verts = end - start;

          if (is_spherical) {
              double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
              for (std::size_t i = start; i < end; ++i) {
                  auto ni = static_cast<std::size_t>(indices[i]);
                  double lon = coords(ni, 0);
                  double lat = coords(ni, 1);
                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                      const double pi = 3.14159265358979323846;
                      lon = lon * pi / 180.0;
                      lat = lat * pi / 180.0;
                  }
                  sum_x += std::cos(lat) * std::cos(lon);
                  sum_y += std::cos(lat) * std::sin(lon);
                  sum_z += std::sin(lat);
              }
              double inv = (n_verts > 0) ? 1.0 / static_cast<double>(n_verts) : 0.0;
              double avg_x = sum_x * inv;
              double avg_y = sum_y * inv;
              double avg_z = sum_z * inv;

              double lat_avg = std::asin(avg_z);
              double lon_avg = std::atan2(avg_y, avg_x);
              if (lon_avg < 0.0) {
                  const double pi = 3.14159265358979323846;
                  lon_avg += 2.0 * pi;
              }

              if (csys == topology::CoordinateSystem::SphericalDeg) {
                  const double pi = 3.14159265358979323846;
                  lon_avg = lon_avg * 180.0 / pi;
                  lat_avg = lat_avg * 180.0 / pi;
              }
              cx_out(c) = lon_avg;
              cy_out(c) = lat_avg;
          } else {
              double sx = 0.0, sy = 0.0;
              for (std::size_t i = start; i < end; ++i) {
                  auto ni = static_cast<std::size_t>(indices[i]);
                  sx += coords(ni, 0);
                  sy += coords(ni, 1);
              }

              double inv = (n_verts > 0) ? 1.0 / static_cast<double>(n_verts) : 0.0;
              cx_out(c) = sx * inv;
              cy_out(c) = sy * inv;
          }
      }
  }
  ```

- [ ] **Step 2: Run build to verify compiling succeeds**
  Verify the modified file compiles successfully:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc)'`
  Expected: PASS

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): refactor compute_cell_centroids_xy with 3D unit-vector spherical averaging"
  ```

---

### Task 2: Refactor Device-Side `compute_cell_centroids_device`

Implement the 3D unit-vector spherical averaging algorithm for device-side centroid calculations.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:751-782`

**Interfaces:**
- Consumes: `UnstructuredMesh::coord_system`.
- Produces: Corrected, dateline-safe device-side centroids in `compute_cell_centroids_device`.

- [ ] **Step 1: Replace device-side centroid implementation**
  Modify `compute_cell_centroids_device` in `libs/axis/src/solver/weight_generator.cpp`:
  ```cpp
  template <class MemorySpace>
  Kokkos::View<double *[2], MemorySpace> compute_cell_centroids_device(const topology::UnstructuredMesh<MemorySpace> &mesh) {
      using exec_space = execution_space_for_t<MemorySpace>;

      const auto n_cells = mesh.n_cells();
      const auto coords = mesh.node_coords();
      const auto offsets = mesh.conn_offsets();
      const auto indices = mesh.conn_indices();
      const auto csys = mesh.coord_system();
      const bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);

      Kokkos::View<double *[2], MemorySpace> centroids("centroids_device", n_cells);

      Kokkos::parallel_for(
          "compute_centroids", Kokkos::RangePolicy<exec_space>(0, n_cells), KOKKOS_LAMBDA(const std::size_t c) {
              auto start = static_cast<std::size_t>(offsets[c]);
              auto end = static_cast<std::size_t>(offsets[c + 1]);
              auto n_verts = end - start;

              if (is_spherical) {
                  double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
                  for (std::size_t i = start; i < end; ++i) {
                      auto ni = static_cast<std::size_t>(indices[i]);
                      double lon = coords(ni, 0);
                      double lat = coords(ni, 1);
                      if (csys == topology::CoordinateSystem::SphericalDeg) {
                          const double pi = 3.14159265358979323846;
                          lon = lon * pi / 180.0;
                          lat = lat * pi / 180.0;
                      }
                      sum_x += Kokkos::cos(lat) * Kokkos::cos(lon);
                      sum_y += Kokkos::cos(lat) * Kokkos::sin(lon);
                      sum_z += Kokkos::sin(lat);
                  }
                  double inv = (n_verts > 0) ? 1.0 / static_cast<double>(n_verts) : 0.0;
                  double avg_x = sum_x * inv;
                  double avg_y = sum_y * inv;
                  double avg_z = sum_z * inv;

                  double lat_avg = Kokkos::asin(avg_z);
                  double lon_avg = Kokkos::atan2(avg_y, avg_x);
                  if (lon_avg < 0.0) {
                      const double pi = 3.14159265358979323846;
                      lon_avg += 2.0 * pi;
                  }

                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                      const double pi = 3.14159265358979323846;
                      lon_avg = lon_avg * 180.0 / pi;
                      lat_avg = lat_avg * 180.0 / pi;
                  }
                  centroids(c, 0) = lon_avg;
                  centroids(c, 1) = lat_avg;
              } else {
                  double sx = 0.0, sy = 0.0;
                  for (std::size_t i = start; i < end; ++i) {
                      auto ni = static_cast<std::size_t>(indices[i]);
                      sx += coords(ni, 0);
                      sy += coords(ni, 1);
                  }

                  double inv = (n_verts > 0) ? 1.0 / static_cast<double>(n_verts) : 0.0;
                  centroids(c, 0) = sx * inv;
                  centroids(c, 1) = sy * inv;
              }
          });

      return centroids;
  }
  ```

- [ ] **Step 2: Run complete C++ build and test suite**
  Run:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100% (or equivalent, confirming full stability and dateline-correctness).

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): refactor compute_cell_centroids_device with 3D unit-vector spherical averaging on GPU"
  ```
