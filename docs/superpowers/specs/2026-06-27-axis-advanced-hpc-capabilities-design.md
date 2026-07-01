# Spec: AXIS Advanced HPC Capabilities (Coastal Renormalization, ESMF NetCDF IO, and Warp-Cooperative GPU Clipping)

## 1. Overview & Motivation
This specification defines the architectural and interface designs for three highly advanced HPC features within the AXIS micro-library:
1.  **Coastal Mask Renormalization & Extrapolation:** Ensures correct partition of unity and flux conservation near land boundaries, with optional geographic nearest-wet-neighbor extrapolation for completely land-bound points.
2.  **Optional ESMF-Compliant NetCDF Weight Egress (AXIS-IO):** Adds an optional CMake-compiled target that reads and writes 1-based, ESMF-compliant NetCDF weights files, allowing AXIS to serve as a drop-in replacement for `ESMF_RegridWeightGen`.
3.  **Warp-Cooperative GPU Clipping:** Accelerates conservative remapping of high-order unstructured meshes (such as hexagonal/Voronoi MPAS grids) by assigning a cooperative Kokkos warp/team to execute parallel polygon clipping.

---

## 2. Coastal Mask Renormalization & Extrapolation

### Interface Extensions
We extend the `solver` namespace with extrapolation configurations:

```cpp
namespace axis::solver {

/// @brief Controls extrapolation behavior for completely masked destination points.
enum class ExtrapolationAction : std::uint8_t {
    None,       ///< Leave row zero (unmapped) or trigger error policy.
    NearestWet  ///< Find closest wet source cell and map it with a weight of 1.0 (Default).
};

struct RegridConfig {
    // Existing fields (method, line_type, etc.)...

    /// @brief Extrapolation method when a destination cell has zero wet source overlaps.
    ExtrapolationAction extrap_method{ExtrapolationAction::NearestWet};
};

} // namespace axis::solver
```

### The Renormalization Algorithm
During weights generation (bilinear or conservative):
1.  **Mask Check:** For each non-zero entry $W_{ji}$ from source cell $i$ to destination cell $j$, if `src_mesh.mask[i] == 0` (indicating land/dry), set $W_{ji} = 0.0$.
2.  **Row-Sum Calculation:** Calculate the sum of active wet weights for each destination row:
    $$S_j = \sum_{i \in \text{wet}} W_{ji}$$
3.  **Normalization Sweep:** If $S_j > 0.0$ and $S_j < 1.0$, re-scale the weights to sum exactly to $1.0$:
    $$W_{ji}' = \frac{W_{ji}}{S_j}$$
4.  **Extrapolation Fallback:** If $S_j == 0.0$ and `extrap_method == ExtrapolationAction::NearestWet`:
    *   Execute an on-device/host nearest-neighbor query using ArborX to find the closest unmasked ("wet") source cell center $i_{wet}$.
    *   Write a single weight of $1.0$ at index $(j, i_{wet})$.

---

## 3. Optional ESMF NetCDF Weight Adapter (AXIS-IO)

### CMake Modularity
The NetCDF I/O adapter is packaged as an optional compile-time target, `axis_io`, compiled only if NetCDF is found on the system during configure:

```cmake
# in libs/axis/CMakeLists.txt
option(AXIS_ENABLE_NETCDF "Enable NetCDF I/O and ESMF weight egress support" ON)

if(AXIS_ENABLE_NETCDF)
    find_package(NetCDF COMPONENTS C QUIET)
    if(NetCDF_FOUND)
        set(AXIS_HAVE_NETCDF 1)
        add_library(axis_io STATIC src/io/esmf_weight_io.cpp)
        target_link_libraries(axis_io PUBLIC axis NetCDF::NetCDF_C)
        target_compile_definitions(axis_io PUBLIC AXIS_HAVE_NETCDF=1)
    endif()
endif()
```

### Public API Interface
We introduce a dedicated `axis::io` namespace with the `EsmfWeightIO` template class:

```cpp
#ifdef AXIS_HAVE_NETCDF

#ifndef AXIS_IO_ESMF_WEIGHT_IO_HPP
#define AXIS_IO_ESMF_WEIGHT_IO_HPP

#include <axis/solver/interpolation_matrix.hpp>
#include <string>

namespace axis::io {

template <typename MemorySpace>
class EsmfWeightIO {
public:
    /// @brief Write an InterpolationMatrix to an ESMF-compliant NetCDF weights file (1-based indices).
    static void write_esmf(
        const std::string& filepath,
        const solver::InterpolationMatrix<MemorySpace>& matrix
    );

    /// @brief Read an ESMF-compliant NetCDF weights file into a C++ InterpolationMatrix (translating 1-based to 0-based).
    static solver::InterpolationMatrix<MemorySpace> read_esmf(
        const std::string& filepath
    );
};

} // namespace axis::io

#endif // AXIS_IO_ESMF_WEIGHT_IO_HPP
#endif // AXIS_HAVE_NETCDF
```

### NetCDF Variables Layout (1-Based)
The written NetCDF file will contain variables structured to conform to ESMF's standard format:
- `num_wgts` dimension (number of non-zero entries).
- `S(num_wgts)` — double-precision weights.
- `col_idx(num_wgts)` — 32-bit integers representing 1-based source indices ($i_{src} + 1$).
- `row_idx(num_wgts)` — 32-bit integers representing 1-based destination indices ($j_{dst} + 1$).
- Global attributes: `n_a` (source cells count), `n_b` (destination cells count).

---

## 4. Warp-Cooperative GPU Clipping

### Automatic Dispatch Heuristic
When generating weights on GPUs, AXIS evaluates the maximum vertices per cell:
- **Vertices $\le 4$ (Triangles / Quads):** Dispatched via standard thread-parallel clipping (1 thread per cell-cell overlap).
- **Vertices $\ge 5$ (Pentagons, Hexagonal/Voronoi MPAS):** Dispatched via warp-cooperative team clipping.

### Parallel Cooperative Clipping Design
Assigns a Kokkos `TeamPolicy` (representing a GPU warp of 32 threads) to cooperatively solve a single cell-cell overlap:
1.  **Cooperative Load:** All threads in the team load polygon vertices from global memory into fast GPU Shared Memory (Scratchpad).
2.  **Parallel Edge Intersections:** Threads check multiple edge intersections in parallel using warp-shuffle operations, completely removing sequential loop branching.
3.  **Parallel Area Reduction:** The overlapping polygon area is integrated using `Kokkos::parallel_reduce` across the team and returned.

---

## 5. Verification & Testing Strategy
1.  **Coastal Mask Renormalization Tests:** Remaps a sea surface temperature field over a coast, verifying that remaining wet weights sum to exactly $1.0$, and that dry cells map to the nearest-wet neighbor under the default extrapolation policy.
2.  **ESMF-NetCDF IO Roundtrip Tests:** Generates a sparse weight matrix, writes it to NetCDF format using the optional `axis-io` target, deserializes it back, and asserts that the reloaded matrix is bitwise-identical to the original.
3.  **Warp-Cooperative Clipping Parallel Equivalence Tests:** Remaps a hexagonal grid to a regular destination grid on both CPU (OpenMP) and GPU (Cuda/HIP) spaces, verifying that warp-cooperative GPU clipping results match serial CPU results to within a $1.0 \times 10^{-12}$ tolerance.
