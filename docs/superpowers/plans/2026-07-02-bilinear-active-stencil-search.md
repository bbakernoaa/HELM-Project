# AXIS Bilinear Active-Stencil Search Expansion (Gap H) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the active stencil search expansion for unstructured bilinear interpolation on host, querying 8 nearest centroids and evaluating quad and triangle combinations to find a valid enclosing stencil near boundaries.

**Architecture:** Update `WeightGenerator::generate_bilinear` on host. Increase ArborX nearest neighbor search count `k_query` to 8. For each destination cell, loop over combinations of 4 centroids (quads) and 3 centroids (triangles) from closest to furthest. Project coordinates, check if destination is enclosed, and if so, map reference coordinates to assemble weights and exit successfully. Fall back to IDW on the 4 closest neighbors if no stencil is found.

**Tech Stack:** C++20

## Global Constraints
- Stencil search must be highly performant (nested loop boundaries bounded by 8 neighbors max).
- If no bounding stencil is found, IDW must fall back cleanly using only the 4 closest neighbors to preserve historical boundary defaults.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Implement Active-Stencil Search in Host Bilinear Interpolation

Refactor unstructured bilinear weight generation to query 8 neighbors and dynamically search for enclosing quads or triangles.

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp:2260-2490`

**Interfaces:**
- Consumes: `map_to_reference_quad` and `barycentric_triangle`.
- Produces: Enhanced, high-accuracy unstructured bilinear stencil search.

- [ ] **Step 1: Increase query size `k_query` to 8**
  Modify `WeightGenerator::generate_bilinear` around line 2269 to query up to 8 neighbors instead of 4:
  ```cpp
  // Number of neighbors for stencil search query (up to 8)
  const int k_query = static_cast<int>(std::min(static_cast<std::size_t>(8), n_src));
  ```
  And update the query allocation View size to use `k_query` instead of `k_fallback`:
  ```cpp
  Kokkos::View<decltype(ArborX::nearest(Point2{}, 1)) *, HostSpace> queries("queries", n_dst);
  for (std::size_t j = 0; j < n_dst; ++j) {
      double px = dst_cx(j);
      double py = dst_cy(j);
      if constexpr (Dimension == 3) {
          queries(j) = ArborX::nearest(Point2{static_cast<float>(px), static_cast<float>(py), 0.0f}, k_query);
      } else {
          queries(j) = ArborX::nearest(Point2{static_cast<float>(px), static_cast<float>(py)}, k_query);
      }
  }
  ```

- [ ] **Step 2: Implement combinations loop for stencil search**
  Inside the destination cell loop of `WeightGenerator::generate_bilinear` (around line 2300), replace the point-in-polygon logic with a dynamic combination loop:
  *   Retrieve the `n_avail` available neighbor indices (which is $end - begin$, up to 8).
  *   **Phase A: Search for containing Quad.** Iterate over combinations of 4 distinct neighbors $a < b < c < d$ (up to $\binom{8}{4} = 70$ iterations). For each combination:
      - Get the 4 centroids $C_a, C_b, C_c, C_d$.
      - Project them to $(u, v)$ tangent plane if spherical.
      - Check if $(0,0)$ is inside the quad $C_a C_b C_c C_d$.
      - If inside, compute bilinear weights via `map_to_reference_quad`. If mapped successfully, save weights, set `used_shape_functions = true`, and BREAK both loops.
  *   **Phase B: Search for containing Triangle.** If no quad is found, iterate over combinations of 3 distinct neighbors $a < b < c$ (up to $\binom{8}{3} = 56$ iterations). For each combination:
      - Get the 3 centroids $C_a, C_b, C_c$.
      - Project them to $(u, v)$ tangent plane if spherical.
      - Check if $(0,0)$ is inside the triangle $C_a C_b C_c$.
      - If inside, compute barycentric weights via `barycentric_triangle`. If mapped successfully, save weights, set `used_shape_functions = true`, and BREAK both loops.
  *   **Phase C: IDW Fallback.** If `used_shape_functions` remains false, fall back to IDW on the 4 closest neighbors (indices $begin$ to $begin + \min(4, n\_avail)$).

  Let's write the C++ code block for this stencil search:
  ```cpp
          int n_avail = end - begin;
          bool used_shape_functions = false;

          // Phase A: Search for containing Quad
          if (n_avail >= 4 && !used_shape_functions) {
              for (int a = 0; a < n_avail - 3 && !used_shape_functions; ++a) {
                  for (int b = a + 1; b < n_avail - 2 && !used_shape_functions; ++b) {
                      for (int c = b + 1; c < n_avail - 1 && !used_shape_functions; ++c) {
                          for (int d = c + 1; d < n_avail && !used_shape_functions; ++d) {
                              std::size_t c0 = static_cast<std::size_t>(values(begin + a).index);
                              std::size_t c1 = static_cast<std::size_t>(values(begin + b).index);
                              std::size_t c2 = static_cast<std::size_t>(values(begin + c).index);
                              std::size_t c3 = static_cast<std::size_t>(values(begin + d).index);

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

                              std::vector<Vec2> poly_quad = {q0, q1, q2, q3};
                              if (point_in_polygon(test_px, test_py, poly_quad)) {
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
                      }
                  }
              }
          }

          // Phase B: Search for containing Triangle
          if (n_avail >= 3 && !used_shape_functions) {
              for (int a = 0; a < n_avail - 2 && !used_shape_functions; ++a) {
                  for (int b = a + 1; b < n_avail - 1 && !used_shape_functions; ++b) {
                      for (int c = b + 1; c < n_avail && !used_shape_functions; ++c) {
                          std::size_t c0 = static_cast<std::size_t>(values(begin + a).index);
                          std::size_t c1 = static_cast<std::size_t>(values(begin + b).index);
                          std::size_t c2 = static_cast<std::size_t>(values(begin + c).index);

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
          }

          // Phase C: IDW Fallback (uses up to 4 closest neighbors)
          if (!used_shape_functions) {
              double sum_inv_dist = 0.0;
              int n_idw = std::min(4, n_avail);
              std::vector<double> distances(n_idw);
              bool has_zero = false;
              int zero_idx = -1;

              for (int k = 0; k < n_idw; ++k) {
                  std::size_t src_idx = static_cast<std::size_t>(values(begin + k).index);
                  double dist = 0.0;
                  if (is_spherical) {
                      double lon_s = src_cx(src_idx);
                      double lat_s = src_cy(src_idx);
                      double lon_d = dst_cx(j);
                      double lat_d = dst_cy(j);
                      if (csys == topology::CoordinateSystem::SphericalDeg) {
                          const double pi = 3.14159265358979323846;
                          lon_s = lon_s * pi / 180.0;
                          lat_s = lat_s * pi / 180.0;
                          lon_d = lon_d * pi / 180.0;
                          lat_d = lat_d * pi / 180.0;
                      }
                      double sx = std::cos(lat_s) * std::cos(lon_s);
                      double sy = std::cos(lat_s) * std::sin(lon_s);
                      double sz = std::sin(lat_s);
                      double dx = std::cos(lat_d) * std::cos(lon_d) - sx;
                      double dy = std::cos(lat_d) * std::sin(lon_d) - sy;
                      double dz = std::sin(lat_d) - sz;
                      dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                  } else {
                      double dx = px - src_cx(src_idx);
                      double dy = py - src_cy(src_idx);
                      dist = std::sqrt(dx * dx + dy * dy);
                  }

                  if (dist <= 0.0) {
                      has_zero = true;
                      zero_idx = k;
                      break;
                  }
                  distances[k] = dist;
                  sum_inv_dist += 1.0 / dist;
              }

              if (has_zero) {
                  std::size_t src_idx = static_cast<std::size_t>(values(begin + zero_idx).index);
                  weights_vec.push_back(1.0);
                  rows_vec.push_back(static_cast<index_t>(j));
                  cols_vec.push_back(static_cast<index_t>(src_idx));
              } else {
                  for (int k = 0; k < n_idw; ++k) {
                      std::size_t src_idx = static_cast<std::size_t>(values(begin + k).index);
                      double w = (1.0 / distances[k]) / sum_inv_dist;
                      weights_vec.push_back(w);
                      rows_vec.push_back(static_cast<index_t>(j));
                      cols_vec.push_back(static_cast<index_t>(src_idx));
                  }
              }
          }
  ```

- [ ] **Step 3: Run complete build and test suite**
  Verify everything compiles clean and all 363 C++ tests pass perfectly:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100%

- [ ] **Step 4: Commit**
  ```bash
  git add libs/axis/src/solver/weight_generator.cpp
  git commit -m "feat(axis): implement 8-neighbor active stencil search for unstructured bilinear interpolation"
  ```
