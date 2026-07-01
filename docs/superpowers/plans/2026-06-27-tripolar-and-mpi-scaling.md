# AXIS Tripolar Reflection & MPI Distributed Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate target cell location on ORCA-style ocean grids using analytical coordinate reflections, and enable multi-node clustered execution by assembling distributed sparse matrices via the `libs/halo` boundary exchange library.

**Architecture:**
- *Tripolar Reflections:* Intercept coordinate lookups near folded seams on-device. If $\phi > \phi_{seam}$, analytically reflect $\lambda$ and $\phi$ and resolve the destination cell in $O(1)$ time, skipping BVH queries.
- *MPI Distribution:* Introduce `axis::distributed::generate_distributed_weights`. Partition local destination meshes, execute `halo` boundary exchanges to acquire "ghost" source cells, and build a distributed `InterpolationMatrix`.

**Tech Stack:** C++20, Kokkos 5.1.1, MPI, libs/halo.

## Global Constraints
- Do not use dynamic memory allocations (`malloc`, `free`, `new`, `delete`) inside device kernels.
- Support double-precision calculations with $1.0 \times 10^{-12}$ relative tolerance.
- Adhere strictly to existing AXIS code patterns and file layouts.
- **CRITICAL:** All compile, run, or validation commands MUST be executed inside the running Docker container `helm-dev-env` using `docker exec`. Do NOT run local commands on the macOS host.

---

### Task 1: On-Device Tripolar Coordinate Reflection Fast-Path

**Files:**
- Modify: `libs/axis/src/solver/weight_generator_bilinear.cpp` (or corresponding search routine)
- Modify: `libs/axis/src/solver/weight_generator.cpp`
- Modify: `libs/axis/include/axis/detail/regular_grid_detector.hpp`

**Interfaces:**
- Consumes: `TripolarGridInfo` from `detect_tripolar_grid`

- [ ] **Step 1: Expand `TripolarGridInfo`**

In `regular_grid_detector.hpp`, add fields for reflection centers:
```cpp
struct TripolarGridInfo {
    bool is_tripolar{false};
    std::size_t ni{0};
    std::size_t nj{0};
    double seam_lat{0.0};
    double seam_lon_center{0.0};
};
```

- [ ] **Step 2: Inject Reflection Logic in Spatial Searches**

In `weight_generator.cpp` (and associated fast-paths), check if the source mesh is tripolar. If so, copy the `TripolarGridInfo` struct to the device. Inside the `KOKKOS_LAMBDA` query loops, apply the transformation before querying the ArborX BVH:
```cpp
double query_lon = dst_cx(j);
double query_lat = dst_cy(j);

if (tripolar_info.is_tripolar && query_lat > tripolar_info.seam_lat) {
    // Analytically reflect coordinate
    query_lat = 2.0 * tripolar_info.seam_lat - query_lat;
    query_lon = tripolar_info.seam_lon_center + (tripolar_info.seam_lon_center - query_lon);

    // Normalize longitude
    while (query_lon >= 360.0) query_lon -= 360.0;
    while (query_lon < 0.0) query_lon += 360.0;
}

// Proceed with ArborX query using (query_lon, query_lat)
```

- [ ] **Step 3: Run verify compile in Docker container**

Execute in `helm-dev-env` container:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8"
```

---

### Task 2: Tripolar Reflection Unit Testing

**Files:**
- Modify: `libs/axis/tests/test_tripolar_regridder.cpp`

- [ ] **Step 1: Add Field Remapping Verification Test**

Write a Google Test that builds a tripolar mesh, sets up a smooth field ($y = \sin(\lambda) \cdot \cos(\phi)$), and verifies that remapping onto a regular destination grid spanning across the northern pole produces a continuous, perfectly symmetric field within $1.0 \times 10^{-12}$ tolerance.

- [ ] **Step 2: Run test in Docker container**

Execute in `helm-dev-env` container:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8 && ./tests/axis_unit_tests --gtest_filter=\"TripolarGridTest.*\""
```

---

### Task 3: MPI Distributed Weight Assembly Header & Config

**Files:**
- Create: `libs/axis/include/axis/distributed/distributed_weight_generator.hpp`
- Modify: `libs/axis/CMakeLists.txt`

- [ ] **Step 1: Write header `distributed_weight_generator.hpp`**

```cpp
// SPDX-License-Identifier: Apache-2.0
#ifndef AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP
#define AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP

#include <axis/solver/interpolation_matrix.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/solver/regrid_config.hpp>
#include <mpi.h>

namespace axis::distributed {

template <typename MemorySpace>
solver::InterpolationMatrix<MemorySpace> generate_distributed_weights(
    const topology::UnstructuredMesh<MemorySpace>& local_src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& local_dst_mesh,
    const solver::RegridConfig& config,
    MPI_Comm comm
);

} // namespace axis::distributed
#endif // AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP
```

- [ ] **Step 2: Update CMakeLists to link MPI and HALO**

In `libs/axis/CMakeLists.txt`, add the new source file (which will be created in Task 4) and link against `MPI::MPI_CXX` and `HALO::halo`.

---

### Task 4: Implement MPI Boundary Exchange and Assembly

**Files:**
- Create: `libs/axis/src/distributed/distributed_weight_generator.cpp`

- [ ] **Step 1: Write `distributed_weight_generator.cpp`**

Implement `generate_distributed_weights`.
*   Use `libs/halo` to determine bounding boxes for `local_dst_mesh`.
*   Execute an MPI exchange to fetch coordinates and cell variables of overlapping "ghost" source cells from neighboring ranks.
*   Concatenate `local_src_mesh` and "ghost" source cells into an extended temporary unstructured mesh.
*   Call the standard `axis::solver::WeightGenerator::generate` on the extended local mesh.
*   Map local CSR column indices back to Global MPI indices.

- [ ] **Step 2: Compile verification in Docker container**

Execute in `helm-dev-env` container:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8"
```

---

### Task 5: MPI Distributed Assembly Unit Tests

**Files:**
- Create: `libs/axis/tests/test_distributed_weight_generator.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Write `test_distributed_weight_generator.cpp`**

Write a Google Test that mocks a two-rank MPI environment. Partition a simple $10 \times 10$ structured grid into two $5 \times 10$ halves on ranks 0 and 1. Execute `generate_distributed_weights`. Verify that the sum of non-zero entries on both ranks equals the exact same number of non-zero entries produced by running a serial `WeightGenerator::generate` over the full $10 \times 10$ grid.

- [ ] **Step 2: Execute MPI tests in Docker**

Add to `AXIS_UNIT_TEST_SOURCES`. Run in the container using `mpiexec`:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8 && mpiexec --allow-run-as-root -n 2 ./tests/axis_unit_tests --gtest_filter=\"DistributedWeightGeneratorTest.*\""
```
