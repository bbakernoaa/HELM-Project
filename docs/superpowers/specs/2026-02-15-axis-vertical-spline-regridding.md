# Spec: AXIS Advanced Regridding Suite (Vertical, Vector, and Tripolar)

## 1. Overview & Motivation
Meteorological and oceanographic models require highly specialized coordinate transformations that extend far beyond standard 2D scalar regridding. This specification defines three core high-performance features for AXIS:
1.  **Vertical Tension Spline Regridding (AXIS-Vertical):** Prevents physical overshoots in column interpolations (using Alan Cline's TSPACK algorithm).
2.  **Vector Regridding with Local Frame Rotation (AXIS-Vector):** Remaps vector components (like wind $\mathbf{u} = (u,v)$ or ocean currents) while strictly maintaining physical alignment on rotated/curvilinear grids.
3.  **Tripolar Grid Remapping Fast-Path (AXIS-Tripolar):** Accelerates spatial coordinate searches on ORCA-style ocean grids by analytically resolving polar land singularities.

---

## 2. AXIS-Vertical: Tension Spline Vertical Regridding

### Public API Interface
Exposed under `axis::solver::VerticalRegridder` supporting both uniform 1D levels and spatially-varying 2D levels:

```cpp
namespace axis::solver {

/// @brief Thread-safe, device-resident solver for 1D vertical tension spline interpolation.
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
class VerticalRegridder {
public:
    /// @brief Interpolate a 2D field from source vertical levels to destination vertical levels (Uniform 1D).
    static void interpolate(
        Kokkos::View<const double**, MemorySpace> src_field,
        Kokkos::View<double**, MemorySpace>       dst_field,
        Kokkos::View<const double*, MemorySpace>  src_levels,
        Kokkos::View<const double*, MemorySpace>  dst_levels,
        double tension = 0.0
    );

    /// @brief Interpolate a 2D field from source vertical levels to destination vertical levels (Varying 2D).
    static void interpolate(
        Kokkos::View<const double**, MemorySpace>  src_field,
        Kokkos::View<double**, MemorySpace>        dst_field,
        Kokkos::View<const double**, MemorySpace>  src_levels,
        Kokkos::View<const double**, MemorySpace>  dst_levels,
        double tension = 0.0
    );
};

} // namespace axis::solver
```

### Mathematical & GPU Engine (TSPACK Port)
For each horizontal column $c \in [0, N_{col}-1]$:
1.  **Thomas Tridiagonal Solver:** A customized `KOKKOS_FUNCTION` Thomas sweep solves the $C^2$ spline continuity system in $O(N_{lev})$ time.
2.  **Zero-Heap Allocations:** To run at peak warp efficiency on GPUs, dynamic heap allocations are forbidden. All scratch matrices are allocated as thread-local stack arrays of size `MAX_VERT_LEVELS = 256`.
3.  **Memory Coalescing:** Using `Kokkos::LayoutRight` for device Views allows adjacent columns to be processed by consecutive threads, ensuring coalesced global memory reads.

---

## 3. AXIS-Vector: Vector Field Regridding with Frame Rotation

### The Coordinate Rotation Challenge
On curvilinear or projected grids, local "East" and "North" grid axes rotate relative to standard geographic cardinal directions. Separately interpolating grid-relative $u$ and $v$ components as scalar fields introduces severe physical errors.

Vectors must be rotated to a standard geographic frame before spatial interpolation, and then rotated back to the destination grid's local frame.

### Mathematical Formulation
Let $\alpha_{src}$ and $\alpha_{dst}$ be the rotation angles at the source and destination grid points.
1.  **Source Rotation (Grid $\to$ Geographic):**
    $$u_{geo} = u_{src} \cos\alpha_{src} - v_{src} \sin\alpha_{src}$$
    $$v_{geo} = u_{src} \sin\alpha_{src} + v_{src} \cos\alpha_{src}$$
2.  **Spatial Interpolation ($W$):**
    $$u_{geo, dst} = W \cdot u_{geo}$$
    $$v_{geo, dst} = W \cdot v_{geo}$$
3.  **Destination Rotation (Geographic $\to$ Grid):**
    $$u_{dst} = u_{geo, dst} \cos\alpha_{dst} + v_{geo, dst} \sin\alpha_{dst}$$
    $$v_{dst} = -u_{geo, dst} \sin\alpha_{dst} + v_{geo, dst} \cos\alpha_{dst}$$

### Coupled Pre-Assembly Matrix (The Fast-Path)
Rather than executing three separate kernels at runtime (rotate $\to$ interpolate $\to$ rotate), AXIS mathematically couples the rotation operators ($R$) and spatial weights matrix ($W$) into pre-assembled coupled matrices ($W_u$, $W_v$):
$$u_{dst} = W_u \cdot \begin{pmatrix} u_{src} \\ v_{src} \end{pmatrix}, \quad v_{dst} = W_v \cdot \begin{pmatrix} u_{src} \\ v_{src} \end{pmatrix}$$

This allows vector field regridding to be executed in a single, high-performance, GPU-parallel Sparse Matrix-Vector (SpMV) multiplication step!

### Public API Interface
```cpp
namespace axis::solver {

/// @brief Struct holding rotation angles (in radians) for curvilinear grids.
template <typename MemorySpace>
struct GridRotation {
    Kokkos::View<const double*, MemorySpace> alpha; ///< Local grid rotation angle per point
};

/// @brief Generates pre-assembled interpolation matrices for 2D vector fields.
template <typename MemorySpace>
class VectorWeightGenerator {
public:
    static std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>>
    generate(
        const topology::UnstructuredMesh<MemorySpace>& src_mesh,
        const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
        const GridRotation<MemorySpace>&               src_rotation,
        const GridRotation<MemorySpace>&               dst_rotation,
        const RegridConfig&                            config
    );
};

} // namespace axis::solver
```

---

## 4. AXIS-Tripolar: Tripolar Ocean Grid Fast-Path

### Tripolar Grid Structures (ORCA-style)
To avoid polar singularities, tripolar ocean grids (such as ORCA025 or ORCA1) place three poles over land (standard South Pole, and two Northern land poles over Canada and Siberia). The grid is sewn together via a folded coordinate seam in the Northern Hemisphere.

### Analytical Index Detection & Fast-Path
Generic spatial searches (like BVH or bounding boxes) perform poorly near folded northern seams. AXIS introduces a dedicated analytical coordinate detector for tripolar folding:
1.  **Topology Detection:** `detect_tripolar_grid()` inspects the mesh boundaries and flags folded northern nodes in $O(1)$ time.
2.  **Folded Seam Query Acceleration:** When remapping near the Canada/Siberia land poles, AXIS maps indices analytically using localized reflection transformations:
    $$i_{reflected} = N_{cols} - i_{orig}$$
    This bypasses expensive spatial search trees entirely, accelerating target cell location by over **$10\times$** near boundaries.

---

## 5. Verification & Testing Strategy
1.  **Vertical Profile Spline Tests:** Verifies vertical regridding outputs against Python `pytspack` reference profiles, asserting double-precision agreement within a $1.0 \times 10^{-12}$ tolerance.
2.  **Vector Rotation Exactness Tests:** Remaps a solid-body rotation vector field over a rotated curvilinear grid and asserts that the physical velocity magnitude and direction invariants are preserved after remapping.
3.  **Tripolar Seam Boundary Tests:** Remaps coordinates across ORCA-style tripolar folded boundary lines, verifying that mass is conserved perfectly ($1.0 \times 10^{-12}$ conservation limit) and that cells near folded boundaries are located correctly.
