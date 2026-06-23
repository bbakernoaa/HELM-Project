# Implementation Plan: HELM Enterprise Hardening

## Overview

This plan implements five workstreams of enterprise hardening for the HELM micro-library ecosystem: a comprehensive BLEND test suite, five CI pipelines, four README files, a root `.clang-format`, and tier isolation scripts. All work replicates established patterns from DAGR/HALO/LOGS without introducing a super-build or top-level CMakeLists.txt. Tasks are ordered so independent artifacts can be created in parallel, with dependencies (e.g., BLEND tests before BLEND CI) respected.

## Tasks

- [ ] 1. Root configuration and isolation scripts
  - [ ] 1.1 Create root `.clang-format` configuration file
    - Create `.clang-format` at the repository root replicating the AMIO configuration: `BasedOnStyle: Google`, `ColumnLimit: 150`, `Standard: c++20`, `IndentWidth: 4`, `PointerAlignment: Right`, `AllowShortFunctionsOnASingleLine: Empty`, `AlignAfterOpenBracket: Align`, `BreakBeforeBraces: Attach`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 1.2 Create BLEND isolation script `libs/blend/cmake/check_tier1_isolation.sh`
    - Create the `cmake/` directory under `libs/blend/`
    - Implement the isolation script following the LOGS pattern, rejecting includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `conf/`, or `tick/` path segments
    - Make the script executable (`chmod +x`)
    - _Requirements: 9.1, 9.2, 9.7, 9.8_

  - [ ] 1.3 Create SPAN isolation script `libs/span/cmake/check_tier1_isolation.sh`
    - Create the `cmake/` directory under `libs/span/` if it doesn't exist
    - Implement the isolation script rejecting includes matching `halo/`, `logs/`, `axis/`, `amio/`, `dagr/`, `conf/`, `tick/`, or `blend/` path segments
    - Make the script executable
    - _Requirements: 9.1, 9.3, 9.7, 9.8_

  - [ ] 1.4 Create CONF isolation script `libs/conf/cmake/check_tier1_isolation.sh`
    - Create the `cmake/` directory under `libs/conf/` if it doesn't exist
    - Implement the isolation script rejecting includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `tick/`, or `blend/` path segments
    - Make the script executable
    - _Requirements: 9.1, 9.4, 9.7, 9.8_

  - [ ] 1.5 Create TICK isolation script `libs/tick/cmake/check_tier1_isolation.sh`
    - Create the `cmake/` directory under `libs/tick/` if it doesn't exist
    - Implement the isolation script rejecting includes matching `halo/`, `logs/`, `axis/`, `amio/`, `span/`, `dagr/`, `conf/`, or `blend/` path segments
    - Make the script executable
    - _Requirements: 9.1, 9.6, 9.7, 9.8_

- [ ] 2. BLEND test suite
  - [ ] 2.1 Create BLEND `tests/CMakeLists.txt` build configuration
    - Create `libs/blend/tests/` directory
    - Write `CMakeLists.txt` that finds GTest and RapidCheck, defines unit test targets (`test_linear_blend`, `test_step_blend`) and property test targets (`prop_identity`, `prop_step_select`, `prop_boundedness`), links against `HELM::BLEND`, `GTest::gtest_main`, and `rapidcheck`, and assigns CTest labels `"unit"` and `"property"` respectively
    - Wire into the parent BLEND `CMakeLists.txt` via the existing `BUILD_TESTING` gate
    - _Requirements: 1.1, 1.10, 1.11_

  - [ ] 2.2 Create BLEND `tests/generators.hpp` RapidCheck generators
    - Implement `blend::gen::array_length()`, `blend::gen::field_data()`, `blend::gen::alpha_unit()`, and `blend::gen::alpha_wide()` generators per the design specification
    - _Requirements: 1.7, 1.8, 1.9_

  - [ ] 2.3 Implement `test_linear_blend.cpp` unit tests
    - Write GTest cases verifying: fixed-input formula correctness (`target[i] = left[i]*(1-α) + right[i]*α`), `std::invalid_argument` thrown on extent mismatch, correct no-op behavior on empty arrays
    - Include Kokkos initialization via GTest environment fixture
    - _Requirements: 1.2, 1.4, 1.6_

  - [ ] 2.4 Implement `test_step_blend.cpp` unit tests
    - Write GTest cases verifying: step-select behavior (left when α<0.5, right when α≥0.5), `std::invalid_argument` thrown on extent mismatch, correct no-op behavior on empty arrays
    - Include Kokkos initialization via GTest environment fixture
    - _Requirements: 1.3, 1.5, 1.6_

  - [ ] 2.5 Implement `prop_identity.cpp` property test
    - **Property 1: LinearBlendKernel Identity**
    - Verify α=0.0 produces left unchanged, α=1.0 produces right unchanged, using RapidCheck `RC_GTEST_PROP` macro and generators from `generators.hpp`
    - **Validates: Requirements 1.7**

  - [ ] 2.6 Implement `prop_step_select.cpp` property test
    - **Property 2: StepBlendKernel Step-Select Correctness**
    - Verify output always equals exactly one of the two inputs for any alpha value, using RapidCheck `RC_GTEST_PROP` macro
    - **Validates: Requirements 1.8**

  - [ ] 2.7 Implement `prop_boundedness.cpp` property test
    - **Property 3: LinearBlendKernel Interpolation Boundedness**
    - Verify output is element-wise bounded between min and max of left/right for α∈[0,1], using RapidCheck `RC_GTEST_PROP` macro
    - **Validates: Requirements 1.9**

- [ ] 3. Checkpoint — BLEND test build verification
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. README documentation
  - [ ] 4.1 Create `libs/blend/README.md`
    - Follow the TICK/AXIS README structure: description (Tier 1 header-only math kernels), features, prerequisites table (Kokkos, GTest, RapidCheck), Docker launch, CMake configure/build, CMake options (`BUILD_TESTING`), downstream consumption example, running tests, license
    - _Requirements: 7.2, 7.5, 7.6, 7.7_

  - [ ] 4.2 Create `libs/span/README.md`
    - Follow the TICK/AXIS README structure: description (Tier 2 zero-copy boundary views), features, prerequisites table (Kokkos, GTest, RapidCheck), Docker launch, CMake configure/build, CMake options, downstream consumption, running tests, license
    - _Requirements: 7.3, 7.5, 7.6, 7.7_

  - [ ] 4.3 Create `libs/conf/README.md`
    - Follow the TICK/AXIS README structure: description (Tier 1 RAII YAML parsing), features, prerequisites table (yaml-cpp, GTest, RapidCheck, gfortran), Docker launch, CMake configure/build, CMake options (`BUILD_TESTING`, `BUILD_FORTRAN`), downstream consumption, running tests, license
    - _Requirements: 7.4, 7.5, 7.6, 7.7_

  - [ ] 4.4 Create `libs/dagr/README.md`
    - Follow the TICK/AXIS README structure: description (Tier 3 directed acyclic graph router), features, prerequisites table, Docker launch, CMake configure/build, CMake options, downstream consumption, running tests, license
    - _Requirements: 7.1, 7.5, 7.6, 7.7_

- [ ] 5. CI pipelines
  - [ ] 5.1 Create `.github/workflows/blend-ci.yml`
    - Implement the full LOGS-style staged pipeline: checkout with submodules, bring up container, Stage 1 tier isolation scan, Stage 2 standalone build (`BUILD_TESTING=ON`), Stage 2b downstream consumer test (link `HELM::BLEND`), Stage 3 unit tests (`ctest -L unit`), Stage 4 property tests (`ctest -L property`), Stage 5 sanitizer build (ASan+UBSan), cleanup and teardown
    - Trigger on push/PR to `libs/blend/**`, `DockerFile`, `docker-compose.yml`, `.github/workflows/blend-ci.yml`, plus `workflow_dispatch`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_

  - [ ] 5.2 Create `.github/workflows/span-ci.yml`
    - Implement the full staged pipeline matching BLEND CI structure
    - Trigger on push/PR to `libs/span/**`, `DockerFile`, `docker-compose.yml`, `.github/workflows/span-ci.yml`, plus `workflow_dispatch`
    - Consumer test includes `span/helm_span.hpp` (or equivalent primary header)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

  - [ ] 5.3 Create `.github/workflows/conf-ci.yml`
    - Implement the full staged pipeline with CONF-specific flags: `BUILD_FORTRAN=ON` in main build, additional `ctest -L fortran` stage, sanitizer build with `BUILD_FORTRAN=OFF`
    - Trigger on push/PR to `libs/conf/**`, `DockerFile`, `docker-compose.yml`, `.github/workflows/conf-ci.yml`, plus `workflow_dispatch`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11_

  - [ ] 5.4 Create `.github/workflows/axis-ci.yml`
    - Implement the full staged pipeline matching BLEND CI structure
    - Trigger on push/PR to `libs/axis/**`, `DockerFile`, `docker-compose.yml`, `.github/workflows/axis-ci.yml`, plus `workflow_dispatch`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

  - [ ] 5.5 Create `.github/workflows/tick-ci.yml`
    - Implement the full staged pipeline matching BLEND CI structure
    - Trigger on push/PR to `libs/tick/**`, `DockerFile`, `docker-compose.yml`, `.github/workflows/tick-ci.yml`, plus `workflow_dispatch`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10_

- [ ] 6. Final checkpoint — Full verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- All CI workflows are YAML files that don't need to be "run" during task execution — just created correctly
- Isolation scripts are shell scripts that should be created and made executable
- The BLEND test suite must be built and verified inside the Docker container via `docker exec helm-dev-env` or `docker compose exec -T helm-dev bash -lc '...'`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5", "4.1", "4.2", "4.3", "4.4"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "2.5", "2.6", "2.7"] },
    { "id": 3, "tasks": ["5.1", "5.2", "5.3", "5.4", "5.5"] }
  ]
}
```
