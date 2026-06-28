# AXIS Detail Utilities Doxygen Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive, complete, and highly descriptive Doxygen comments (`///`) to six core AXIS Detail Utility headers to achieve high-quality professional documentation.

**Architecture:** Standardize file-level documentation, class, struct, and function-level doc blocks across the target headers. Use explicit semantic tags including `@tparam`, `@param`, `@return`, and `@throw` where applicable. Ensure compile-readiness and semantic equivalence.

**Tech Stack:** C++20, Kokkos, GoogleTest, Docker (`helm-dev-env` container)

## Global Constraints
- Strictly maintain existing C++20 formatting and compile-readiness.
- Do not alter any functional logic.
- Use standard `///` Doxygen blocks.
- Add file-level comments (`@file`, `@brief`).
- Include `@tparam`, `@param`, `@return`, and `@throw` where appropriate, using complete sentences and explicit types.

---

### Task 1: Document `libs/axis/include/axis/detail/memory_traits.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/memory_traits.hpp`

**Interfaces:**
- Consumes: Existing implementation of `memory_traits.hpp`
- Produces: Documented version of traits `is_device_space`, `is_device_space_v`, and `exec_space_t`.

- [ ] **Step 1: Edit `memory_traits.hpp` with Doxygen Comments**

Modify the file to include standard file-level doc, descriptive class-level docs for `is_device_space`, its specializations for host/device spaces, the variable shortcut `is_device_space_v`, and the type alias `exec_space_t`.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 1**

```bash
git add libs/axis/include/axis/detail/memory_traits.hpp
git commit -m "docs: add Doxygen comments to memory_traits.hpp"
```

---

### Task 2: Document `libs/axis/include/axis/detail/mdspan_interop.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/mdspan_interop.hpp`

**Interfaces:**
- Consumes: Existing implementation of `mdspan_interop.hpp`
- Produces: Documented version of `view_builder` specializations, `to_view` overloads, `make_mdspan` helper, and `to_mdspan`.

- [ ] **Step 1: Edit `mdspan_interop.hpp` with Doxygen Comments**

Add standard file-level doc, document the internal helper `impl::view_builder` templates/specializations, and provide fully annotated public function docs for `to_view` and `to_mdspan` with explicit `@tparam`, `@param`, and `@return` details.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 2**

```bash
git add libs/axis/include/axis/detail/mdspan_interop.hpp
git commit -m "docs: add Doxygen comments to mdspan_interop.hpp"
```

---

### Task 3: Document `libs/axis/include/axis/detail/morton_sort.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/morton_sort.hpp`

**Interfaces:**
- Consumes: Existing implementation of `morton_sort.hpp`
- Produces: Documented version of `morton_encode_2d`, `normalize_coord`, and `morton_sort_indices`.

- [ ] **Step 1: Edit `morton_sort.hpp` with Doxygen Comments**

Ensure file-level doc is complete, and document the algorithms and execution paths for `morton_encode_2d`, `normalize_coord`, and `morton_sort_indices` with extensive parameter/template descriptions.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 3**

```bash
git add libs/axis/include/axis/detail/morton_sort.hpp
git commit -m "docs: add Doxygen comments to morton_sort.hpp"
```

---

### Task 4: Document `libs/axis/include/axis/detail/spherical_cap_filter.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/spherical_cap_filter.hpp`

**Interfaces:**
- Consumes: Existing implementation of `spherical_cap_filter.hpp`
- Produces: Documented version of constants, `CapData` struct, `lonlat_to_xyz_device`, `compute_cell_centroid_xyz`, `compute_angular_radius`, `spherical_cap_rejects`, and `precompute_cap_data`.

- [ ] **Step 1: Edit `spherical_cap_filter.hpp` with Doxygen Comments**

Provide thorough file-level, namespace, struct, and function comments detailing the geometry and mechanics of the early-exit filter.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 4**

```bash
git add libs/axis/include/axis/detail/spherical_cap_filter.hpp
git commit -m "docs: add Doxygen comments to spherical_cap_filter.hpp"
```

---

### Task 5: Document `libs/axis/include/axis/detail/spherical_clipper.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/spherical_clipper.hpp`

**Interfaces:**
- Consumes: Existing implementation of `spherical_clipper.hpp`
- Produces: Documented version of `Vec3`, vector operations, `SphericalPolygon`, and `SphericalClipper`.

- [ ] **Step 1: Edit `spherical_clipper.hpp` with Doxygen Comments**

Document all geometrical operations, struct definitions, vertex clamping limits, the area computation via Girard's theorem, and the clipping algorithm steps.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 5**

```bash
git add libs/axis/include/axis/detail/spherical_clipper.hpp
git commit -m "docs: add Doxygen comments to spherical_clipper.hpp"
```

---

### Task 6: Document `libs/axis/include/axis/detail/spherical_geometry.hpp`

**Files:**
- Modify: `libs/axis/include/axis/detail/spherical_geometry.hpp`

**Interfaces:**
- Consumes: Existing implementation of `spherical_geometry.hpp`
- Produces: Documented version of EFT helpers, `Vec3`, coordinate conversion, predicates, and advanced spherical area/clipping algorithms.

- [ ] **Step 1: Edit `spherical_geometry.hpp` with Doxygen Comments**

Add a rich header-level comment introducing the references (Chen et al. papers), and document the precision predicates (`two_sum`, `split`, `two_product`), `robust_orient_sphere` (adaptive precision), `great_circle_arc_intersection`, `great_circle_const_lat_intersection`, `spherical_clip_polygon`, `spherical_polygon_area`, `spherical_polygon_overlap_area`, and conversion helpers.

- [ ] **Step 2: Compile and verify unit tests**

Run: `docker exec helm-dev-env cmake --build /workspace/helm-project/libs/axis/build-ci -j 4 && docker exec helm-dev-env /workspace/helm-project/libs/axis/build-ci/tests/axis_unit_tests`
Expected: Compilation succeeds and all tests pass.

- [ ] **Step 3: Commit changes for Task 6**

```bash
git add libs/axis/include/axis/detail/spherical_geometry.hpp
git commit -m "docs: add Doxygen comments to spherical_geometry.hpp"
```
