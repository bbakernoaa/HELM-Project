# AXIS Gnomonic Tangent-Plane Polar Projection for Bilinear Interpolation (Gap C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a local gnomonic tangent-plane projection for spherical unstructured bilinear interpolation to eliminate polar coordinate singularities and longitude wrap-around (dateline) discontinuities on the host.

**Architecture:** Add a `project_gnomonic` helper function. Inside `WeightGenerator::generate_bilinear`, if the grid uses a spherical coordinate system, project all vertex coordinates of the containing cell and nearest centroids onto a local tangent plane centered on the destination point (mapping the destination point exactly to $(0.0, 0.0)$). Solve for reference coordinates using $(0.0, 0.0)$ on the tangent plane.

**Tech Stack:** C++20

## Global Constraints
- Must be numerically stable up to the pole peaks ($\pm 90^\circ$).
- Must handle dateline crossings ($180^\circ$ / $360^\circ$) automatically without boundary artifacts.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Implement Gnomonic Projection in Host Bilinear Interpolation

Implement the `project_gnomonic` projection helper and update `WeightGenerator::generate_bilinear` to apply it.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:2184-2450`

**Interfaces:**
- Consumes: `map_to_reference_quad` and `barycentric_triangle`.
- Produces: Robust, singularity-free spherical unstructured bilinear weight generation.

- [ ] **Step 1: Add `project_gnomonic` helper**
  Add the helper function inside `libs/axis/src/solver/weight_generator.cpp` right before the bilinear shape functions section (around line 537):
  ```cpp
  /// Project coordinate (lon, lat) gnomonically onto a tangent plane centered at (lon0, lat0).
  /// Coordinates are assumed to be in radians.
  inline void project_gnomonic(double lon0, double lat0, double lon, double lat, double &u, double &v) {
      double cos_c = std::sin(lat0) * std::sin(lat) + std::cos(lat0) * std::cos(lat) * std::cos(lon - lon0);
      if (cos_c <= 0.0) cos_c = 1e-15;
      u = (std::cos(lat) * std::sin(lon - lon0)) / cos_c;
      v = (std::sin(lat) * std::cos(lat0) - std::cos(lat) * std::sin(lat0) * std::cos(lon - lon0)) / cos_c;
  }
  ```

- [ ] **Step 2: Apply projection locally for spherical coordinates**
  Inside `WeightGenerator::generate_bilinear` (around line 2300), if the coordinate system is spherical, convert `px` and `py` to radians `lon0` and `lat0`, and project all cell vertices and nearest centroids to the local tangent plane centered at `(lon0, lat0)` before evaluating shape functions.
  
  Let's see: we convert to radians if `csys == SphericalDeg` or `SphericalRad`.
  ```cpp
          double px = dst_cx(j);
          double py = dst_cy(j);

          bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);
          double lon0 = px;
          double lat0 = py;
          if (csys == topology::CoordinateSystem::SphericalDeg) {
              const double pi = 3.14159265358979323846;
              lon0 = lon0 * pi / 180.0;
              lat0 = lat0 * pi / 180.0;
          }

          // Try point-in-cell on the nearest source cell
          auto nearest_src = static_cast<std::size_t>(values(begin).index);

          // Get the vertex ring of the nearest source cell
          auto poly = extract_cell_polygon(src_mesh, nearest_src);
          std::size_t n_verts = poly.size();

          bool used_shape_functions = false;

          // Convert polygon vertices to local tangent plane if spherical
          std::vector<Vec2> local_poly(n_verts);
          double test_px = px;
          double test_py = py;

          if (is_spherical) {
              test_px = 0.0;
              test_py = 0.0;
              for (std::size_t i = 0; i < n_verts; ++i) {
                  double v_lon = poly[i].x;
                  double v_lat = poly[i].y;
                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                      const double pi = 3.14159265358979323846;
                      v_lon = v_lon * pi / 180.0;
                      v_lat = v_lat * pi / 180.0;
                  }
                  project_gnomonic(lon0, lat0, v_lon, v_lat, local_poly[i].x, local_poly[i].y);
              }
          } else {
              for (std::size_t i = 0; i < n_verts; ++i) {
                  local_poly[i] = poly[i];
              }
          }

          if (point_in_polygon(test_px, test_py, local_poly)) {
              // Point is inside the nearest cell — use shape functions
              auto cell_start = static_cast<std::size_t>(conn_off[nearest_src]);

              if (n_verts == 4) {
                  // Quad cell: bilinear shape functions via Newton iteration
                  double xi = 0.0, eta = 0.0;
                  if (map_to_reference_quad(test_px, test_py, local_poly[0], local_poly[1], local_poly[2], local_poly[3], xi, eta)) {
                      // Clamp to [-1, 1] for safety
                      xi = std::max(-1.0, std::min(1.0, xi));
                      eta = std::max(-1.0, std::min(1.0, eta));

                      double w0 = 0.25 * (1.0 - xi) * (1.0 - eta);
                      double w1 = 0.25 * (1.0 + xi) * (1.0 - eta);
                      double w2 = 0.25 * (1.0 + xi) * (1.0 + eta);
                      double w3 = 0.25 * (1.0 - xi) * (1.0 + eta);

                      int n_avail = end - begin;
                      if (n_avail >= 4) {
                          std::size_t c0 = static_cast<std::size_t>(values(begin).index);
                          std::size_t c1 = static_cast<std::size_t>(values(begin + 1).index);
                          std::size_t c2 = static_cast<std::size_t>(values(begin + 2).index);
                          std::size_t c3 = static_cast<std::size_t>(values(begin + 3).index);

                          Vec2 q0{src_cx(c0), src_cy(c0)};
                          Vec2 q1{src_cx(c1), src_cy(c1)};
                          Vec2 q2{src_cx(c2), src_cy(c2)};
                          Vec2 q3{src_cx(c3), src_cy(c3)};

                          if (is_spherical) {
                              const double pi = 3.14159265358979323846;
                              auto to_rad = [&](double deg) { return deg * pi / 180.0; };
                              auto proj = [&](Vec2 q) {
                                  double u, v;
                                  double q_lon = q.x;
                                  double q_lat = q.y;
                                  if (csys == topology::CoordinateSystem::SphericalDeg) {
                                      q_lon = to_rad(q_lon);
                                      q_lat = to_rad(q_lat);
                                  }
                                  project_gnomonic(lon0, lat0, q_lon, q_lat, u, v);
                                  return Vec2{u, v};
                              };
                              q0 = proj(q0);
                              q1 = proj(q1);
                              q2 = proj(q2);
                              q3 = proj(q3);
                          }

                          double xi_centroids = 0.0, eta_centroids = 0.0;
                          if (map_to_reference_quad(test_px, test_py, q0, q1, q2, q3, xi_centroids, eta_centroids)) {
                              xi_centroids = std::max(-1.0, std::min(1.0, xi_centroids));
                              eta_centroids = std::max(-1.0, std::min(1.0, eta_centroids));

                              double ww0 = 0.25 * (1.0 - xi_centroids) * (1.0 - eta_centroids);
                              double ww1 = 0.25 * (1.0 + xi_centroids) * (1.0 - eta_centroids);
                              double ww2 = 0.25 * (1.0 + xi_centroids) * (1.0 + eta_centroids);
                              double ww3 = 0.25 * (1.0 - xi_centroids) * (1.0 + eta_centroids);

                              weights_vec.push_back(ww0);
                              rows_vec.push_back(static_cast<index_t>(j));
                              cols_vec.push_back(static_cast<index_t>(c0));

                              weights_vec.push_back(ww1);
                              rows_vec.push_back(static_cast<index_t>(j));
                              cols_vec.push_back(static_cast<index_t>(c1));

                              weights_vec.push_back(ww2);
                              rows_vec.push_back(static_cast<index_t>(j));
                              cols_vec.push_back(static_cast<index_t>(c2));

                              weights_vec.push_back(ww3);
                              rows_vec.push_back(static_cast<index_t>(j));
                              cols_vec.push_back(static_cast<index_t>(c3));

                              used_shape_functions = true;
                          }
                      }
                  }
              } else if (n_verts == 3) {
                  // Triangle: barycentric coordinates on the 3 nearest cell centroids
                  int n_avail = end - begin;
                  if (n_avail >= 3) {
                      std::size_t c0 = static_cast<std::size_t>(values(begin).index);
                      std::size_t c1 = static_cast<std::size_t>(values(begin + 1).index);
                      std::size_t c2 = static_cast<std::size_t>(values(begin + 2).index);

                      Vec2 q0{src_cx(c0), src_cy(c0)};
                      Vec2 q1{src_cx(c1), src_cy(c1)};
                      Vec2 q2{src_cx(c2), src_cy(c2)};

                      if (is_spherical) {
                          const double pi = 3.14159265358979323846;
                          auto to_rad = [&](double deg) { return deg * pi / 180.0; };
                          auto proj = [&](Vec2 q) {
                              double u, v;
                              double q_lon = q.x;
                              double q_lat = q.y;
                              if (csys == topology::CoordinateSystem::SphericalDeg) {
                                  q_lon = to_rad(q_lon);
                                  q_lat = to_rad(q_lat);
                              }
                              project_gnomonic(lon0, lat0, q_lon, q_lat, u, v);
                              return Vec2{u, v};
                          };
                          q0 = proj(q0);
                          q1 = proj(q1);
                          q2 = proj(q2);
                      }

                      double l0 = 0.0, l1 = 0.0, l2 = 0.0;
                      if (barycentric_triangle(test_px, test_py, q0, q1, q2, l0, l1, l2)) {
                          l0 = std::max(0.0, l0);
                          l1 = std::max(0.0, l1);
                          l2 = std::max(0.0, l2);
                          double sum = l0 + l1 + l2;
                          if (sum > 0.0) {
                              l0 /= sum;
                              l1 /= sum;
                              l2 /= sum;
                          } else {
                              l0 = l1 = l2 = 1.0 / 3.0;
                          }

                          weights_vec.push_back(l0);
                          rows_vec.push_back(static_cast<index_t>(j));
                          cols_vec.push_back(static_cast<index_t>(c0));

                          weights_vec.push_back(l1);
                          rows_vec.push_back(static_cast<index_t>(j));
                          cols_vec.push_back(static_cast<index_t>(c1));

                          weights_vec.push_back(l2);
                          rows_vec.push_back(static_cast<index_t>(j));
                          cols_vec.push_back(static_cast<index_t>(c2));

                          used_shape_functions = true;
                      }
                  }
              }
          }
  ```

- [ ] **Step 3: Compile and run test suite**
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100%

- [ ] **Step 4: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): implement local gnomonic projection for unstructured bilinear host weight generation near poles"
  ```
