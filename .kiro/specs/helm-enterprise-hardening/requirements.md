# Requirements Document

## Introduction

This specification defines the enterprise hardening deliverables for the HELM micro-library ecosystem. The scope covers five workstreams: (1) creation of a comprehensive test suite for the BLEND library, (2) full staged CI pipelines for BLEND, SPAN, CONF, AXIS, and TICK, (3) structured README documentation for DAGR, BLEND, SPAN, and CONF, (4) a root `.clang-format` configuration, and (5) tier isolation enforcement scripts for libraries that lack them. All libraries remain independently buildable — no super-build or top-level CMakeLists.txt is introduced.

## Glossary

- **BLEND**: Tier 1 header-only micro-library providing stateless array blending kernels (linear interpolation, step-select) via Kokkos.
- **SPAN**: Tier 2 header-only micro-library providing zero-copy Fortran/C++ boundary views with optional C-API shared library.
- **CONF**: Tier 1 compiled micro-library providing stateless RAII YAML parsing for HELM configuration.
- **AXIS**: Tier 1 compiled micro-library providing spatial interpolation (regridding) via Kokkos sparse-matrix operations.
- **TICK**: Tier 1 compiled micro-library providing fixed-point time arithmetic and calendar engines.
- **DAGR**: Tier 3 compiled micro-library providing directed acyclic graph routing and orchestration.
- **CI_Pipeline**: A GitHub Actions workflow file implementing staged build, test, and sanitizer verification for a single HELM library.
- **Isolation_Scan**: A shell script that scans source files for forbidden cross-HELM `#include` directives, enforcing tier boundary constraints.
- **Sanitizer_Build**: A CMake build configured with `-fsanitize=address,undefined -fno-omit-frame-pointer -g` flags for runtime error detection.
- **Property_Test**: A RapidCheck-based test that validates correctness properties over randomized inputs.
- **Unit_Test**: A GTest-based deterministic test validating specific behavior with fixed inputs.
- **Downstream_Consumer_Test**: A minimal CMake project that links against a library's exported alias to verify the alias is consumable without hidden dependencies.
- **Docker_Exec_Pattern**: The CI execution model using `docker compose exec -T helm-dev bash -lc '...'` to run commands inside the HELM development container.
- **Root_Clang_Format**: A `.clang-format` file placed at the repository root defining the unified formatting standard for all HELM libraries.

## Requirements

### Requirement 1: BLEND Test Suite Creation

**User Story:** As a BLEND developer, I want a comprehensive GTest and RapidCheck test suite for LinearBlendKernel and StepBlendKernel, so that I can verify correctness, detect regressions, and validate edge cases.

#### Acceptance Criteria

1. WHEN `BUILD_TESTING=ON` is set during CMake configuration of BLEND, THE BLEND build system SHALL produce test executables from a `tests/` subdirectory.
2. THE BLEND test suite SHALL include unit tests that verify LinearBlendKernel produces `target[i] = left[i] * (1.0 - alpha) + right[i] * alpha` for known fixed inputs.
3. THE BLEND test suite SHALL include unit tests that verify StepBlendKernel produces `target[i] = left[i]` when alpha < 0.5 and `target[i] = right[i]` when alpha >= 0.5.
4. THE BLEND test suite SHALL include a unit test verifying that LinearBlendKernel throws `std::invalid_argument` when input extents are mismatched.
5. THE BLEND test suite SHALL include a unit test verifying that StepBlendKernel throws `std::invalid_argument` when input extents are mismatched.
6. THE BLEND test suite SHALL include a unit test verifying correct behavior on zero-length (empty) arrays for both kernels.
7. THE BLEND test suite SHALL include a property test verifying the identity property: blending with alpha=0.0 produces the left input unchanged, and blending with alpha=1.0 produces the right input unchanged.
8. THE BLEND test suite SHALL include a property test verifying that for StepBlendKernel the output equals exactly one of the two inputs (step-select correctness) for any alpha value.
9. THE BLEND test suite SHALL include a property test verifying LinearBlendKernel output is bounded between element-wise min and max of left and right inputs when alpha is in [0.0, 1.0].
10. THE BLEND test suite SHALL label unit tests with CTest label "unit" and property tests with CTest label "property".
11. THE BLEND test CMakeLists.txt SHALL link against GTest, RapidCheck, and the `HELM::BLEND` target.

### Requirement 2: BLEND CI Pipeline

**User Story:** As a BLEND maintainer, I want a full staged CI pipeline matching the DAGR/HALO/LOGS pattern, so that every commit to BLEND is automatically verified for isolation, correctness, and memory safety.

#### Acceptance Criteria

1. THE BLEND CI_Pipeline SHALL be defined in `.github/workflows/blend-ci.yml` at the repository root.
2. THE BLEND CI_Pipeline SHALL trigger on push and pull_request events affecting `libs/blend/**`, `Dockerfile`, `docker-compose.yml`, or `.github/workflows/blend-ci.yml`.
3. THE BLEND CI_Pipeline SHALL support manual dispatch via `workflow_dispatch`.
4. THE BLEND CI_Pipeline SHALL execute all stages inside the HELM development container using the Docker_Exec_Pattern.
5. THE BLEND CI_Pipeline SHALL include a tier isolation scan stage that verifies BLEND source files contain no forbidden cross-HELM includes (halo, logs, axis, amio, span, dagr, conf, tick).
6. THE BLEND CI_Pipeline SHALL include a standalone CMake build stage producing the `HELM::BLEND` alias target with `BUILD_TESTING=ON`.
7. THE BLEND CI_Pipeline SHALL include a downstream consumer test stage that builds a minimal project linking `HELM::BLEND` to verify the exported alias is consumable.
8. THE BLEND CI_Pipeline SHALL include a unit test stage executing `ctest -L unit --output-on-failure`.
9. THE BLEND CI_Pipeline SHALL include a property test stage executing `ctest -L property --output-on-failure`.
10. THE BLEND CI_Pipeline SHALL include a sanitizer build stage using `-fsanitize=address,undefined -fno-omit-frame-pointer -g` and running tests under `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`.
11. THE BLEND CI_Pipeline SHALL execute the isolation scan before the build stage to fail fast on architectural violations.

### Requirement 3: SPAN CI Pipeline

**User Story:** As a SPAN maintainer, I want a full staged CI pipeline, so that SPAN's 13 existing tests and isolation constraints are verified on every commit.

#### Acceptance Criteria

1. THE SPAN CI_Pipeline SHALL be defined in `.github/workflows/span-ci.yml` at the repository root.
2. THE SPAN CI_Pipeline SHALL trigger on push and pull_request events affecting `libs/span/**`, `Dockerfile`, `docker-compose.yml`, or `.github/workflows/span-ci.yml`.
3. THE SPAN CI_Pipeline SHALL support manual dispatch via `workflow_dispatch`.
4. THE SPAN CI_Pipeline SHALL execute all stages inside the HELM development container using the Docker_Exec_Pattern.
5. THE SPAN CI_Pipeline SHALL include a tier isolation scan stage verifying SPAN source files contain no forbidden cross-HELM includes (halo, logs, axis, amio, dagr, conf, tick, blend).
6. THE SPAN CI_Pipeline SHALL include a standalone CMake build stage with `BUILD_TESTING=ON`.
7. THE SPAN CI_Pipeline SHALL include a downstream consumer test stage linking `HELM::SPAN`.
8. THE SPAN CI_Pipeline SHALL include a unit test stage executing `ctest -L unit --output-on-failure`.
9. THE SPAN CI_Pipeline SHALL include a property test stage executing `ctest -L property --output-on-failure`.
10. THE SPAN CI_Pipeline SHALL include a sanitizer build stage using the standard Sanitizer_Build flags and running property tests under sanitizer runtime options.

### Requirement 4: CONF CI Pipeline

**User Story:** As a CONF maintainer, I want a full staged CI pipeline, so that CONF's 16 existing tests and Fortran integration are verified on every commit.

#### Acceptance Criteria

1. THE CONF CI_Pipeline SHALL be defined in `.github/workflows/conf-ci.yml` at the repository root.
2. THE CONF CI_Pipeline SHALL trigger on push and pull_request events affecting `libs/conf/**`, `Dockerfile`, `docker-compose.yml`, or `.github/workflows/conf-ci.yml`.
3. THE CONF CI_Pipeline SHALL support manual dispatch via `workflow_dispatch`.
4. THE CONF CI_Pipeline SHALL execute all stages inside the HELM development container using the Docker_Exec_Pattern.
5. THE CONF CI_Pipeline SHALL include a tier isolation scan stage verifying CONF source files contain no forbidden cross-HELM includes.
6. THE CONF CI_Pipeline SHALL include a standalone CMake build stage with `BUILD_TESTING=ON` and `BUILD_FORTRAN=ON`.
7. THE CONF CI_Pipeline SHALL include a downstream consumer test stage linking `HELM::CONF`.
8. THE CONF CI_Pipeline SHALL include a unit test stage executing `ctest -L unit --output-on-failure`.
9. THE CONF CI_Pipeline SHALL include a property test stage executing `ctest -L property --output-on-failure`.
10. THE CONF CI_Pipeline SHALL include a Fortran integration test stage executing `ctest -L fortran --output-on-failure`.
11. THE CONF CI_Pipeline SHALL include a sanitizer build stage using the standard Sanitizer_Build flags (with `BUILD_FORTRAN=OFF` since Fortran code is excluded from sanitizer analysis).

### Requirement 5: AXIS CI Pipeline

**User Story:** As an AXIS maintainer, I want a full staged CI pipeline at the top level (beyond the existing benchmark workflow), so that AXIS's 66 tests are verified for isolation, correctness, and memory safety on every commit.

#### Acceptance Criteria

1. THE AXIS CI_Pipeline SHALL be defined in `.github/workflows/axis-ci.yml` at the repository root.
2. THE AXIS CI_Pipeline SHALL trigger on push and pull_request events affecting `libs/axis/**`, `Dockerfile`, `docker-compose.yml`, or `.github/workflows/axis-ci.yml`.
3. THE AXIS CI_Pipeline SHALL support manual dispatch via `workflow_dispatch`.
4. THE AXIS CI_Pipeline SHALL execute all stages inside the HELM development container using the Docker_Exec_Pattern.
5. THE AXIS CI_Pipeline SHALL include a tier isolation scan stage verifying AXIS source files contain no forbidden cross-HELM includes.
6. THE AXIS CI_Pipeline SHALL include a standalone CMake build stage with `BUILD_TESTING=ON`.
7. THE AXIS CI_Pipeline SHALL include a downstream consumer test stage linking `HELM::AXIS`.
8. THE AXIS CI_Pipeline SHALL include a unit test stage executing `ctest -L unit --output-on-failure`.
9. THE AXIS CI_Pipeline SHALL include a property test stage executing `ctest -L property --output-on-failure`.
10. THE AXIS CI_Pipeline SHALL include a sanitizer build stage using the standard Sanitizer_Build flags and running property tests under sanitizer runtime options.

### Requirement 6: TICK Top-Level CI Pipeline

**User Story:** As a TICK maintainer, I want a top-level CI workflow that uses the Docker_Exec_Pattern (matching DAGR/HALO/LOGS), so that TICK CI runs consistently with the rest of the monorepo pipelines.

#### Acceptance Criteria

1. THE TICK CI_Pipeline SHALL be defined in `.github/workflows/tick-ci.yml` at the repository root.
2. THE TICK CI_Pipeline SHALL trigger on push and pull_request events affecting `libs/tick/**`, `Dockerfile`, `docker-compose.yml`, or `.github/workflows/tick-ci.yml`.
3. THE TICK CI_Pipeline SHALL support manual dispatch via `workflow_dispatch`.
4. THE TICK CI_Pipeline SHALL execute all stages inside the HELM development container using the Docker_Exec_Pattern (`docker compose exec -T helm-dev bash -lc '...'`).
5. THE TICK CI_Pipeline SHALL include a tier isolation scan stage verifying TICK source files contain no forbidden cross-HELM includes.
6. THE TICK CI_Pipeline SHALL include a standalone CMake build stage with `BUILD_TESTING=ON`.
7. THE TICK CI_Pipeline SHALL include a downstream consumer test stage linking `HELM::TICK`.
8. THE TICK CI_Pipeline SHALL include a unit test stage executing `ctest -L unit --output-on-failure`.
9. THE TICK CI_Pipeline SHALL include a property test stage executing `ctest -L property --output-on-failure`.
10. THE TICK CI_Pipeline SHALL include a sanitizer build stage using the standard Sanitizer_Build flags and running tests under sanitizer runtime options.

### Requirement 7: README Documentation

**User Story:** As a HELM contributor, I want structured README files for DAGR, BLEND, SPAN, and CONF (matching the TICK/AXIS README format), so that each library has consistent onboarding documentation.

#### Acceptance Criteria

1. THE DAGR README SHALL be located at `libs/dagr/README.md` and follow the structure of the existing TICK and AXIS READMEs.
2. THE BLEND README SHALL be located at `libs/blend/README.md` and follow the structure of the existing TICK and AXIS READMEs.
3. THE SPAN README SHALL be located at `libs/span/README.md` and follow the structure of the existing TICK and AXIS READMEs.
4. THE CONF README SHALL be located at `libs/conf/README.md` and follow the structure of the existing TICK and AXIS READMEs.
5. WHEN a README is created, THE README SHALL include sections for: library description, features, prerequisites table, Docker container launch instructions, CMake configure and build instructions, CMake options table, consuming the library from a downstream project, running tests, and license reference.
6. THE README prerequisites tables SHALL list only the actual dependencies of each library (Kokkos for BLEND and SPAN, yaml-cpp for CONF, GTest and RapidCheck for testing).
7. THE README CMake options tables SHALL document all user-facing CMake options defined in the library's CMakeLists.txt.

### Requirement 8: Root Clang-Format Configuration

**User Story:** As a HELM developer, I want a single root `.clang-format` file derived from the existing AMIO configuration, so that all libraries share a consistent formatting standard.

#### Acceptance Criteria

1. THE Root_Clang_Format SHALL be placed at the repository root as `.clang-format`.
2. THE Root_Clang_Format SHALL use `BasedOnStyle: Google` as the base style.
3. THE Root_Clang_Format SHALL set `ColumnLimit` to 150.
4. THE Root_Clang_Format SHALL set `IndentWidth` to 4.
5. THE Root_Clang_Format SHALL set `Standard` to `c++20`.
6. THE Root_Clang_Format SHALL preserve the `PointerAlignment`, `AllowShortFunctionsOnASingleLine`, `AlignAfterOpenBracket`, and `BreakBeforeBraces` settings from the AMIO `.clang-format`.

### Requirement 9: Tier Isolation Enforcement Scripts

**User Story:** As an architect, I want each library that lacks an isolation scan script to receive one, so that tier boundary violations are detectable by CI before compilation.

#### Acceptance Criteria

1. WHEN a HELM library does not already have an isolation scan script, THE hardening process SHALL create a `check_tier1_isolation.sh` (or equivalent) script in that library's `cmake/` or `scripts/` directory.
2. THE Isolation_Scan script for BLEND SHALL reject includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `conf/`, or `tick/` path segments in BLEND source files.
3. THE Isolation_Scan script for SPAN SHALL reject includes matching `halo/`, `logs/`, `axis/`, `amio/`, `dagr/`, `conf/`, `tick/`, or `blend/` path segments in SPAN source files.
4. THE Isolation_Scan script for CONF SHALL reject includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `tick/`, or `blend/` path segments in CONF source files.
5. THE Isolation_Scan script for AXIS SHALL reject includes matching `halo/`, `logs/`, `amio/`, `span/`, `dagr/`, `conf/`, `tick/`, or `blend/` path segments in AXIS source files.
6. THE Isolation_Scan script for TICK SHALL reject includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `conf/`, or `blend/` path segments in TICK source files.
7. WHEN an Isolation_Scan script detects a forbidden include, THE script SHALL print the offending file and line number and exit with a non-zero status code.
8. WHEN an Isolation_Scan script detects no violations, THE script SHALL print a success message and exit with status code 0.
