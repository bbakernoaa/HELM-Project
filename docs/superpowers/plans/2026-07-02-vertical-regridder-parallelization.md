# AXIS Vertical Regridding 3D Parallelization (Gap B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor both 1D and 2D `VerticalRegridder` interpolation methods to use Kokkos `TeamPolicy` and Shared Scratch Memory, parallelizing level processing and avoiding GPU stack allocations.

**Architecture:** Use `Kokkos::TeamPolicy` with league size equal to `n_col`. Allocate shared scratch space for each team, and use `Kokkos::TeamThreadRange` to cooperatively load values and evaluate splines in parallel across team threads, with thread 0 solving the tridiagonal column spline.

**Tech Stack:** C++20, Kokkos

## Global Constraints
- Must be fully parallelized across levels cooperatively within each execution team.
- Must allocate temporary column workspaces on Kokkos PerTeam Scratch Space to avoid stack spillage and stack overflow.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Refactor 1D coordinate `VerticalRegridder::interpolate`

Refactor the 1D levels vertical regridding method to utilize TeamPolicy and Shared Scratch Memory.

**Files:**
- Modify: `libs/axis/src/solver/vertical_regridder.cpp:10-65`

**Interfaces:**
- Consumes: `solve_column_spline` and `evaluate_spline`.
- Produces: Optimized `VerticalRegridder<MemorySpace>::interpolate` (1D coordinates).

- [ ] **Step 1: Replace 1D implementation with TeamPolicy**
  Modify the 1D levels `interpolate` method in `libs/axis/src/solver/vertical_regridder.cpp`:
  ```cpp
  template <typename MemorySpace>
  void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                   Kokkos::View<const double *, MemorySpace> src_levels,
                                                   Kokkos::View<const double *, MemorySpace> dst_levels, double tension) {
      const std::size_t n_col = src_field.extent(0);
      const std::size_t n_src = src_levels.extent(0);
      const std::size_t n_dst = dst_levels.extent(0);

      if (n_col != dst_field.extent(0)) {
          throw std::invalid_argument("VerticalRegridder: Source and destination column extents mismatch");
      }
      if (src_field.extent(1) != n_src) {
          throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
      }
      if (dst_field.extent(1) != n_dst) {
          throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
      }
      if (tension < 0.0) {
          throw std::invalid_argument("VerticalRegridder: Tension parameter must be non-negative");
      }

      constexpr std::size_t MAX_LEVELS = 256;
      if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
          throw std::invalid_argument("VerticalRegridder: Vertical level dimensions exceed hard cap of 256 levels");
      }

      using execution_space = typename MemorySpace::execution_space;
      using policy_type = Kokkos::TeamPolicy<execution_space>;
      using member_type = typename policy_type::member_type;

      std::size_t scratch_bytes = 4 * n_src * sizeof(double);
      policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
      policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

      Kokkos::parallel_for(
          "VerticalInterpolate1D", policy, KOKKOS_LAMBDA(const member_type &team) {
              const std::size_t c = team.league_rank();

              auto scratch_space = team.team_scratch(0);
              double *src_x = reinterpret_cast<double *>(scratch_space.data());
              double *src_y = src_x + n_src;
              double *d = src_y + n_src;
              double *scratch_arr = d + n_src;

              // Cooperatively fill src_x and src_y
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                  src_y[i] = src_field(c, i);
                  src_x[i] = src_levels(i);
              });
              team.team_barrier();

              // Thread 0 solves the column spline
              if (team.team_rank() == 0) {
                  axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
              }
              team.team_barrier();

              // Cooperatively evaluate spline for each target destination level
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                  double target = dst_levels(j);

                  std::size_t idx = 0;
                  while (idx < n_src - 2 && src_x[idx + 1] < target) {
                      idx++;
                  }

                  dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                          d[idx + 1], tension);
              });
          });
  }
  ```

- [ ] **Step 2: Run build to verify compile and linking succeeds**
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc)'`
  Expected: PASS

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/vertical_regridder.cpp
  git commit -m "feat(axis): refactor 1D VerticalRegridder with Kokkos TeamPolicy and Shared Scratch Memory"
  ```

---

### Task 2: Refactor 2D coordinate `VerticalRegridder::interpolate`

Refactor the 2D levels vertical regridding method to utilize TeamPolicy and Shared Scratch Memory.

**Files:**
- Modify: `libs/axis/src/solver/vertical_regridder.cpp:66-125`

**Interfaces:**
- Consumes: `solve_column_spline` and `evaluate_spline`.
- Produces: Optimized `VerticalRegridder<MemorySpace>::interpolate` (2D coordinates).

- [ ] **Step 1: Replace 2D implementation with TeamPolicy**
  Modify the 2D levels `interpolate` method in `libs/axis/src/solver/vertical_regridder.cpp`:
  ```cpp
  template <typename MemorySpace>
  void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                   Kokkos::View<const double **, MemorySpace> src_levels,
                                                   Kokkos::View<const double **, MemorySpace> dst_levels, double tension) {
      const std::size_t n_col = src_field.extent(0);
      const std::size_t n_src = src_levels.extent(1);
      const std::size_t n_dst = dst_levels.extent(1);

      if (n_col != dst_field.extent(0) || n_col != src_levels.extent(0) || n_col != dst_levels.extent(0)) {
          throw std::invalid_argument("VerticalRegridder: Grid column dimensions mismatch across fields and level views");
      }
      if (src_field.extent(1) != n_src) {
          throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
      }
      if (dst_field.extent(1) != n_dst) {
          throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
      }
      if (tension < 0.0) {
          throw std::invalid_argument("VerticalRegridder: Tension parameter must be non-negative");
      }

      constexpr std::size_t MAX_LEVELS = 256;
      if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
          throw std::invalid_argument("VerticalRegridder: Vertical level dimensions exceed hard cap of 256 levels");
      }

      using execution_space = typename MemorySpace::execution_space;
      using policy_type = Kokkos::TeamPolicy<execution_space>;
      using member_type = typename policy_type::member_type;

      std::size_t scratch_bytes = 4 * n_src * sizeof(double);
      policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
      policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

      Kokkos::parallel_for(
          "VerticalInterpolate2D", policy, KOKKOS_LAMBDA(const member_type &team) {
              const std::size_t c = team.league_rank();

              auto scratch_space = team.team_scratch(0);
              double *src_x = reinterpret_cast<double *>(scratch_space.data());
              double *src_y = src_x + n_src;
              double *d = src_y + n_src;
              double *scratch_arr = d + n_src;

              // Cooperatively fill src_x and src_y
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                  src_y[i] = src_field(c, i);
                  src_x[i] = src_levels(c, i);
              });
              team.team_barrier();

              // Thread 0 solves the column spline
              if (team.team_rank() == 0) {
                  axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
              }
              team.team_barrier();

              // Cooperatively evaluate spline for each target destination level
              Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                  double target = dst_levels(c, j);

                  std::size_t idx = 0;
                  while (idx < n_src - 2 && src_x[idx + 1] < target) {
                      idx++;
                  }

                  dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                          d[idx + 1], tension);
              });
          });
  }
  ```

- [ ] **Step 2: Run complete C++ build and test suite**
  Run:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100% (and verified green, preserving bitwise equivalence).

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/vertical_regridder.cpp
  git commit -m "feat(axis): refactor 2D VerticalRegridder with Kokkos TeamPolicy and Shared Scratch Memory"
  ```
