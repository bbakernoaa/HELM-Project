# AXIS Rectilinear Normal Lat-Lon Named Grids (R-Family) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support for a new global Rectilinear Normal Lat-Lon named grid family under prefix `'R'` (e.g., `"R360"` for a $1440 \times 720$ grid spanning $[-180^\circ, 180^\circ]$ in longitude and $[-90^\circ, 90^\circ]$ in latitude) inside the AXIS NamedGridRegistry.

**Architecture:** Extend the `NamedGridRegistry` parsing logic to recognize prefix `'R'` and its associated resolution parameter $N$. Add a dispatch case in the `generate` function that calculates the dimensions ($ni = 4N, nj = 2N$) and step sizes ($dlon, dlat$) to construct a global regular grid with boundaries starting at $(-180.0, -90.0)$ using the existing analytical host generator. Update property-based and unit test suites to incorporate the new family.

**Tech Stack:** C++20, Kokkos, GoogleTest, RapidCheck

## Global Constraints
- Grid generation is deterministic: two calls with the same name produce bitwise-identical results (Requirement 6.5).
- All changes must be fully isolation-compliant and pass the existing project isolation scans and linters.
- No external file I/O or external databases may be introduced for named grids.

---

### Task 1: NamedGridRegistry Parsing & Family Integration

Add support for the `'R'` family character prefix in `NamedGridRegistry` parsing and validation.

**Files:**
- Modify: `libs/axis/include/axis/topology/named_grid_registry.hpp`
- Modify: `libs/axis/src/topology/named_grid_registry.cpp:650-715`
- Test: `libs/axis/tests/prop_named_grid_registration.cpp`

**Interfaces:**
- Consumes: None (built-in string library and standard `tolower`/`toupper` utilities).
- Produces: `NamedGridRegistry::parse` supporting family `'R'`, and `NamedGridRegistry::registered_families()` returning `{'F', 'N', 'O', 'R'}`.

- [ ] **Step 1: Write failing tests in property-based suite**
  Modify `libs/axis/tests/prop_named_grid_registration.cpp` to include `'R'` in the valid families list and verify failure of the families enumeration assertion.
  
  In `libs/axis/tests/prop_named_grid_registration.cpp` at `genValidFamily()`:
  ```cpp
  /// Generate a valid family prefix character: one of 'O', 'F', 'N', or 'R'.
  rc::Gen<char> genValidFamily() {
      return rc::gen::element('O', 'F', 'N', 'R');
  }
  ```
  And update `genInvalidFamily()` to exclude `'R'`:
  ```cpp
  rc::Gen<char> genInvalidFamily() {
      return rc::gen::suchThat(rc::gen::inRange<char>('A', '['),  // 'A'..'Z'
                               [](char c) { return c != 'O' && c != 'F' && c != 'N' && c != 'R'; });
  }
  ```
  And update `RegisteredFamiliesAreFNO` to expect `{'F', 'N', 'O', 'R'}` and rename to `RegisteredFamiliesAreFNOR`:
  ```cpp
  RC_GTEST_PROP(PropNamedGridRegistration, RegisteredFamiliesAreFNOR, ()) {
      auto families = NamedGridRegistry::registered_families();

      // Must be sorted
      RC_ASSERT(std::is_sorted(families.begin(), families.end()));

      // Must contain exactly F, N, O, R
      const std::vector<char> expected = {'F', 'N', 'O', 'R'};
      RC_ASSERT(families == expected);
  }
  ```

- [ ] **Step 2: Run tests to verify the expected failure**
  Run the test suite inside the docker container:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis/build-ci && ctest -R PropNamedGridRegistration -V'`
  Expected: FAIL (assertion error on sorting/registered families, and parsing errors for 'R' if generated).

- [ ] **Step 3: Implement minimal registry parsing changes**
  Modify the parsing and validation logic in `libs/axis/src/topology/named_grid_registry.cpp`:
  At `parse` method:
  ```cpp
  char family = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));

  // Validate family
  if (family != 'O' && family != 'F' && family != 'N' && family != 'R') {
      throw std::invalid_argument("NamedGridRegistry::parse: unknown grid family '" + std::string(1, name[0]) + "' in name \"" + name +
                                  "\"; registered families are O, F, N, R, and grid<num>");
  }
  ```
  At `registered_families` method:
  ```cpp
  std::vector<char> NamedGridRegistry::registered_families() {
      return {'F', 'N', 'O', 'R'};
  }
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run the registration tests again:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis/build-ci && ctest -R PropNamedGridRegistration --output-on-failure'`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add libs/axis/include/axis/topology/named_grid_registry.hpp libs/axis/src/topology/named_grid_registry.cpp libs/axis/tests/prop_named_grid_registration.cpp
  git commit -m "feat(axis): support R prefix parsing and register family in NamedGridRegistry"
  ```

---

### Task 2: NamedGridRegistry R-Family Mesh Generation

Implement the analytical rectilinear lat-lon grid generation dispatcher for family `'R'`.

**Files:**
- Modify: `libs/axis/src/topology/named_grid_registry.cpp:715-740`
- Modify: `libs/axis/tests/prop_named_grid_deterministic.cpp`
- Modify: `libs/axis/tests/test_named_and_rules.cpp`

**Interfaces:**
- Consumes: `NamedGridRegistry::parse` and `generate_regular_grid`.
- Produces: `NamedGridRegistry::generate<Kokkos::HostSpace>(name)` returning `UnstructuredMesh` for family `'R'`.

- [ ] **Step 1: Write a failing unit test in test_named_and_rules.cpp**
  Add unit tests for `"R4"` inside `libs/axis/tests/test_named_and_rules.cpp`:
  ```cpp
  // Test: generate("R4") produces a rectilinear normal lat-lon grid with 16x8 cells
  TEST(NamedAndRules, R4GeneratesCorrectMesh) {
      auto mesh = topology::NamedGridRegistry::generate<MemSpace>("R4");

      // ni = 16, nj = 8
      // n_cells = 16 * 8 = 128
      // n_nodes = 17 * 9 = 153
      EXPECT_EQ(mesh.n_cells(), std::size_t(128));
      EXPECT_EQ(mesh.n_nodes(), std::size_t(153));

      // Verify coordinate range spans [-180, 180] in x and [-90, 90] in y
      auto coords = mesh.node_coords();
      double min_x = 999.0, max_x = -999.0;
      double min_y = 999.0, max_y = -999.0;
      for (std::size_t i = 0; i < mesh.n_nodes(); ++i) {
          double x = coords(i, 0);
          double y = coords(i, 1);
          if (x < min_x) min_x = x;
          if (x > max_x) max_x = x;
          if (y < min_y) min_y = y;
          if (y > max_y) max_y = y;
      }
      EXPECT_NEAR(min_x, -180.0, 1e-7);
      EXPECT_NEAR(max_x, 180.0, 1e-7);
      EXPECT_NEAR(min_y, -90.0, 1e-7);
      EXPECT_NEAR(max_y, 90.0, 1e-7);
  }
  ```

- [ ] **Step 2: Add 'R' to the deterministic property-based test generator**
  Modify `libs/axis/tests/prop_named_grid_deterministic.cpp`:
  ```cpp
  /// Generate a valid grid family character: 'O', 'F', 'N', or 'R'.
  rc::Gen<char> genFamily() {
      return rc::gen::element('O', 'F', 'N', 'R');
  }
  ```

- [ ] **Step 3: Run the test suite to verify the expected failure**
  Run:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis/build-ci && make -j$(nproc) && ctest -R "R4GeneratesCorrectMesh|PropNamedGridDeterministic" -V'`
  Expected: FAIL (unsupported family error in generation for R4/R family).

- [ ] **Step 4: Implement minimal dispatcher logic in generate**
  Modify `NamedGridRegistry::generate<Kokkos::HostSpace>` in `libs/axis/src/topology/named_grid_registry.cpp` to handle `'R'`:
  ```cpp
  template <>
  UnstructuredMesh<Kokkos::HostSpace> NamedGridRegistry::generate<Kokkos::HostSpace>(const std::string &name) {
      ParsedName parsed = parse(name);

      switch (parsed.family) {
          case 'G':
              return generate_noaa_grib_grid<Kokkos::HostSpace>(parsed.number);
          case 'O':
              return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
          case 'N':
              // N (reduced Gaussian) follows ECMWF octahedral convention
              return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
          case 'F':
              return generate_regular_gaussian<Kokkos::HostSpace>(parsed.number);
          case 'R': {
              const int N = parsed.number;
              const std::size_t ni = static_cast<std::size_t>(4 * N);
              const std::size_t nj = static_cast<std::size_t>(2 * N);
              const double dlon = 360.0 / static_cast<double>(ni);
              const double dlat = 180.0 / static_cast<double>(nj);
              return generate_regular_grid<Kokkos::HostSpace>(ni, nj, -180.0, -90.0, dlon, dlat);
          }
          default:
              // Should not reach here due to parse() validation
              throw std::invalid_argument("NamedGridRegistry::generate: unknown family '" + std::string(1, parsed.family) + "'");
      }
  }
  ```

- [ ] **Step 5: Run tests to verify they all pass**
  Compile and run tests:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis/build-ci && make -j$(nproc) && ctest --output-on-failure'`
  Expected: PASS 100% (318 tests passed, including the new unit test and property tests with R-family).

- [ ] **Step 6: Commit**
  ```bash
  git add libs/axis/src/topology/named_grid_registry.cpp libs/axis/tests/prop_named_grid_deterministic.cpp libs/axis/tests/test_named_and_rules.cpp
  git commit -m "feat(axis): implement rectilinear lat-lon named grid mesh generation for R-family"
  ```

---

### Task 3: Documentation and Verification

Update standard markdown documentation of grid construction and run complete CI verification checks.

**Files:**
- Modify: `libs/axis/docs/mesh_construction.md:65-75`

**Interfaces:** None.

- [ ] **Step 1: Update documentation file**
  Add `'R<N>'` entry to the list of available grid types table in `libs/axis/docs/mesh_construction.md`:
  ```markdown
  | Pattern | Description | Example |
  |---------|-------------|---------|
  | `O<N>` | Octahedral reduced Gaussian | O48, O96, O320 |
  | `N<N>` | Regular Gaussian | N48, N128, N256 |
  | `F<N>` | Full (regular lat-lon) | F90, F180 |
  | `R<N>` | Global Rectilinear Normal Lat-Lon | R90, R360 |
  ```

- [ ] **Step 2: Run isolation scan to check for no violations**
  Run:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && bash cmake/check_tier1_isolation.sh .'`
  Expected: PASS (zero isolation scan violations).

- [ ] **Step 3: Run comprehensive test suite**
  Run:
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis/build-ci && ctest --output-on-failure'`
  Expected: 100% tests passed.

- [ ] **Step 4: Commit**
  ```bash
  git add libs/axis/docs/mesh_construction.md
  git commit -m "docs(axis): document R-family named grid support in mesh_construction"
  ```
