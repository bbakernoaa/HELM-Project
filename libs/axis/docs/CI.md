# AXIS CI Pipeline

This document describes the Continuous Integration pipeline for the AXIS micro-library. The pipeline runs on every push and pull request to ensure code quality, correctness, and standalone buildability.

## Pipeline Stages

### 1. Static Analysis — Tier 1 Isolation Scan

**Script:** `cmake/check_tier1_isolation.sh`

Scans all AXIS source and header files for forbidden `#include` directives or Fortran `use` statements. AXIS is a Tier 1 HELM utility and must not depend on HALO, AMIO, TICK, LOGS, SPAN, DAGR, eckit, or any domain-science headers.

Forbidden patterns include:
- `#include` of any HELM library header (HALO, AMIO, TICK, LOGS, SPAN, DAGR)
- `#include` of eckit, NetCDF, HDF5, ecCodes, yaml-cpp, or TensorStore headers
- Any `MPI_Comm`, `nc_*`, `codes_handle*`, or `eckit::*` type in public signatures
- Fortran `use` directives referencing forbidden modules

This stage **fails the build** if any violation is found.

### 2. Standalone CMake Build (Docker)

**Container:** `helm-dev-env` (Docker image from project `DockerFile`)

Verifies that AXIS builds as a standalone library using only its declared dependencies (Kokkos + optional PROJ), without requiring other HELM source trees present in the workspace.

```bash
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DAXIS_ENABLE_PROJ=ON && \
  cmake --build build --parallel 4"
```

The build must produce the `HELM::AXIS` CMake target. If other HELM libraries (HALO, AMIO, etc.) are not present, the build must still succeed — this confirms the zero-dependency guarantee.

### 3. Unit Tests (GTest)

Runs the `axis_unit_tests` executable, containing deterministic example-based GTest tests that verify:

- First-order conservation (Σ src ≈ Σ dst)
- Constant-field preservation (partition of unity)
- Bilinear exactness on affine fields
- SpMV apply correctness against reference scalar loop
- Descriptor round-trip (mesh → UGRID descriptor → mesh)
- Producer equivalence (identical descriptors from different producers)
- Descriptor validation (malformed inputs throw correctly)
- Named-grid and rule-based generation
- Gmsh export/reimport round-trip
- Distributed apply equivalence (HaloPattern + gathered buffer)
- DstArea vs FracArea normalization semantics
- RAII handle resource management

```bash
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis/build && \
  ctest --output-on-failure -L unit"
```

### 4. Property Tests (RapidCheck)

Runs the `axis_property_tests` executable with a minimum of **100 iterations** per property. These tests verify universal correctness properties across randomized inputs:

- Field view round-trip identity
- Structured-to-unstructured geometry preservation
- CSR connectivity validity
- Named-grid determinism
- Rule-based cell count matches expected formula
- Descriptor validation rejects all malformed inputs
- Producer equivalence for all valid descriptors
- Sparse matrix index bounds
- Conservative weight non-negativity
- Partition of unity
- First-order conservation
- Bilinear exactness
- SpMV equals reference loop
- DstArea / FracArea normalization
- Unmapped destination handling
- Coordinate-system consistency enforcement
- HaloPattern completeness
- Gmsh round-trip
- Descriptor round-trip
- Host/device result equivalence

```bash
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis/build && \
  ctest --output-on-failure -L property"
```

RapidCheck is configured with `RC_PARAMS="max_size=100 max_success=100"` to ensure statistical coverage.

### 5. Sanitizer Builds (ASan + UBSan)

Compiles and runs the full test suite under AddressSanitizer and UndefinedBehaviorSanitizer to detect memory errors, buffer overflows, use-after-free, integer overflow, and other undefined behavior.

```bash
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  cmake -B build-sanitizers \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
    -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
    -DBUILD_TESTING=ON && \
  cmake --build build-sanitizers --parallel 4 && \
  cd build-sanitizers && \
  ctest --output-on-failure"
```

Any sanitizer finding causes the pipeline to fail.

## Running Locally via Docker

To reproduce the full CI pipeline locally:

```bash
# Start the development container
docker compose up -d helm-dev-env

# Run isolation scan
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  bash cmake/check_tier1_isolation.sh"

# Configure and build
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  cmake -B build -DBUILD_TESTING=ON -DAXIS_ENABLE_PROJ=ON && \
  cmake --build build --parallel 4"

# Run all tests
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis/build && \
  ctest --output-on-failure"

# Run sanitizer build
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  cmake -B build-san \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
    -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined' \
    -DBUILD_TESTING=ON && \
  cmake --build build-san --parallel 4 && \
  cd build-san && ctest --output-on-failure"
```

## Verifying Standalone Build

The standalone build verification confirms that `HELM::AXIS` can be built and consumed without any other HELM library source tree present:

```bash
docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis && \
  cmake --build build --parallel 4 2>&1 | grep -c error"
# Expected output: 0

docker exec helm-dev-env bash -c "\
  cd /workspace/helm-project/libs/axis/build && \
  ctest --output-on-failure 2>&1 | grep 'tests passed'"
```

The `find_package(AXIS)` step in downstream consumers should resolve `HELM::AXIS` with Kokkos transitively linked. No `find_package(HALO)`, `find_package(AMIO)`, or any other HELM library should appear in the dependency chain.

## Pipeline Summary

| Stage | Tool | Failure Condition |
|-------|------|-------------------|
| Isolation scan | `check_tier1_isolation.sh` | Any forbidden include found |
| CMake build | `cmake --build` | Any compile error |
| Unit tests | GTest via CTest `-L unit` | Any test failure |
| Property tests | RapidCheck via CTest `-L property` | Any counterexample found |
| Sanitizers | ASan + UBSan | Any sanitizer report |
