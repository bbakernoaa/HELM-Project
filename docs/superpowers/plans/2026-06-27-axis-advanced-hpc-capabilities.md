# AXIS Advanced HPC Capabilities (Coastal, NetCDF ESMF, and Warp-Cooperative GPU Clipping) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a suite of advanced high-performance features in AXIS including Coastal Mask Renormalization, optional ESMF-compliant NetCDF Weight I/O, and Warp-Cooperative GPU clipping.

**Architecture:** 
- *Coastal:* Dynamically filter masked cells, renormalize row sums to exactly 1.0, and perform nearest-neighbor extrapolation via ArborX for completely land-bound points.
- *NetCDF/ESMF:* Compile an optional C++ `axis_io` target when NetCDF is detected, translating sparse matrices to 1-based, ESMF-compliant formats.
- *Warp-Cooperative:* Auto-dispatch warp-team parallel clipping for high-order cells (vertices >= 5) using Kokkos shared scratchpad memory.

**Tech Stack:** C++20, Kokkos 5.1.1, NetCDF-C, ArborX, PROJ.

## Global Constraints
- Do not use dynamic memory allocations (`malloc`, `free`, `new`, `delete`) inside device kernels.
- Support double-precision calculations with $1.0 \times 10^{-12}$ relative tolerance.
- Adhere strictly to existing AXIS code patterns and file layouts.
- **CRITICAL:** All compile, run, or validation commands MUST be executed inside the running Docker container `helm-dev-env` using `docker exec`. Do NOT run local commands on the macOS host.

---

### Task 1: Coastal Mask Renormalization & Extrapolation

**Files:**
- Modify: `libs/axis/include/axis/solver/regrid_config.hpp`
- Modify: `libs/axis/src/solver/weight_generator.cpp`
- Create: `libs/axis/tests/test_coastal_renormalization.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

**Interfaces:**
- Produces: `axis::solver::ExtrapolationAction` and updated weight generation loops

- [ ] **Step 1: Declare Extrapolation enums in `regrid_config.hpp`**

Update `libs/axis/include/axis/solver/regrid_config.hpp`:
```cpp
namespace axis::solver {

enum class ExtrapolationAction : std::uint8_t {
    None,
    NearestWet
};

struct RegridConfig {
    // Existing config...
    ExtrapolationAction extrap_method{ExtrapolationAction::NearestWet};
};

} // namespace axis::solver
```

- [ ] **Step 2: Implement filter, renormalize, and extrapolate loops in `weight_generator.cpp`**

In the weight generation step, scale non-zero entries based on the source mask. If dry, weight becomes 0.0. Renormalize remaining wet weights to sum to 1.0. If all dry, run ArborX k=1 nearest-neighbor query to find the closest unmasked source center and write a weight of 1.0.

- [ ] **Step 3: Write test case in `test_coastal_renormalization.cpp`**

Add tests verifying wet row-sum re-scaling and nearest-wet neighbor extrapolation fallbacks.

- [ ] **Step 4: Register test and verify in Docker container**

Execute in `helm-dev-env` container:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8 && ./tests/axis_unit_tests --gtest_filter=\"CoastalRenormalization.*\""
```

---

### Task 2: Optional ESMF NetCDF Weight Adapter (AXIS-IO)

**Files:**
- Create: `libs/axis/include/axis/io/esmf_weight_io.hpp`
- Create: `libs/axis/src/io/esmf_weight_io.cpp`
- Modify: `libs/axis/CMakeLists.txt`
- Create: `libs/axis/tests/test_esmf_weight_io.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

**Interfaces:**
- Produces: `axis::io::EsmfWeightIO::write_esmf` and `read_esmf`

- [ ] **Step 1: Write header `esmf_weight_io.hpp`**

Write the `EsmfWeightIO` template class declaration wrapped in `#ifdef AXIS_HAVE_NETCDF`.

- [ ] **Step 2: Implement NetCDF writer and reader in `esmf_weight_io.cpp`**

Convert 0-based indices to 1-based indices during write. Convert 1-based indices back to 0-based during read. Use NetCDF-C library calls (`nc_create`, `nc_def_dim`, `nc_def_var`, `nc_put_var_double`, `nc_close`).

- [ ] **Step 3: Modify `CMakeLists.txt` to find NetCDF and add `axis_io` target**

Only add `axis_io` static library and register `test_esmf_weight_io.cpp` if `NetCDF_FOUND` is true.

- [ ] **Step 4: Compile and execute roundtrip verification test in Docker container**

Execute:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake -DAXIS_ENABLE_NETCDF=ON .. && cmake --build . --parallel 8 && ./tests/axis_unit_tests --gtest_filter=\"EsmfWeightIOTest.*\""
```
Verify that written and re-deserialized weights matrices are bitwise-identical.

---

### Task 3: Warp-Cooperative GPU Clipping

**Files:**
- Modify: `libs/axis/src/solver/weight_generator.cpp` (Heuristic dispatcher)
- Modify: `libs/axis/include/axis/detail/planar_clipper.hpp`
- Create: `libs/axis/tests/test_warp_cooperative_clipping.cpp`
- Modify: `libs/axis/tests/CMakeLists.txt`

- [ ] **Step 1: Implement dispatch heuristic in `weight_generator.cpp`**

If `max_vertices >= 5`, configure Kokkos `TeamPolicy` and dispatch the cooperative GPU clipping kernel.

- [ ] **Step 2: Implement `clip_spherical_polygon_cooperative` in `planar_clipper.hpp`**

Using Kokkos scratchpad memory, have the 32 threads in a warp load, intersect, and reduce spherical overlap areas cooperatively.

- [ ] **Step 3: Compile and run verification inside Docker container**

Execute:
```bash
docker exec helm-dev-env bash -c "cd /workspace/helm-project/libs/axis/build && cmake --build . --parallel 8 && ./tests/axis_unit_tests --gtest_filter=\"WarpCooperativeTest.*\""
```
Verify matching results between standard thread-parallel and warp-cooperative GPU clipping.
