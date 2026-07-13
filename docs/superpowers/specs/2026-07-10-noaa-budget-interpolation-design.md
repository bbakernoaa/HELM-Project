# NOAA Budget Interpolation Design Specification

- **Title:** NOAA Budget Interpolation Method
- **Date:** 2026-07-10
- **Author:** Gemini CLI Agent / HELM Project Contributors
- **Status:** Approved

---

## 1. Overview & Purpose
This specification details the design for adding the mass-conserving **Budget Interpolation Method** (inspired by the NOAA-EMC `NCEPLIBS-ip` library and used in the Unified Post-Processor (UPP)) into the AXIS spatial regridding library.

Standard bilinear interpolation can lose small-scale or highly intense features (such as heavy precipitation bands) during grid-to-grid remapping. The budget interpolation method mitigates this by defining a subgrid "sampling box" around each destination grid cell, performing high-resolution bilinear sampling inside the box, and computing the weighted average of the sampled values. This preserves the overall field "budget" (mass/accumulation) over the destination cells.

---

## 2. Configuration Settings (`RegridConfig`)
We will add `InterpolationMethod::Budget` to the list of supported interpolation methods and provide specific configuration settings for tuning the budget algorithm's sub-grid resolution and domain-boundary validation.

```cpp
namespace axis::solver {

enum class InterpolationMethod : std::uint8_t {
    Bilinear,
    NearestNeighbor,
    Bicubic,
    Patch,
    Conservative1stOrder,
    Conservative2ndOrder,
    Budget  ///< NOAA-EMC/UPP Budget interpolation: box-sampled bilinear averaging
};

struct RegridConfig {
    InterpolationMethod method = InterpolationMethod::Bilinear;
    NormType norm_type = NormType::DstArea;
    LineType line_type = LineType::GreatCircle;
    UnmappedAction unmapped = UnmappedAction::Ignore;
    bool use_limiter = false;
    ExtrapolationAction extrap_method = ExtrapolationAction::NearestWet;

    // Budget-specific configuration
    std::uint32_t budget_subgrid_size = 5;      ///< Sub-grid resolution inside each destination box (default: 5x5 = 25 points)
    double budget_min_valid_fraction = 0.5;   ///< Minimum fraction of valid subgrid points required to map a cell (default: 50%)
};

} // namespace axis::solver
```

---

## 3. Mathematical Formulation

For any destination cell $j$, we define its spatial bounds $[x_{\text{min}}, x_{\text{max}}] \times [y_{\text{min}}, y_{\text{max}}]$ based on its geometry:

### 3.1 Bounding Box Definition
- **Structured/Rectilinear Fast-Path:** The bounding box matches the cell's physical coordinate spacing exactly.
- **Unstructured Fallback:** The bounding box is determined by the minimum and maximum coordinates of the cell's vertices:
  $$x_{\text{min}} = \min_{v \in \text{vertices}(j)} x_v, \quad x_{\text{max}} = \max_{v \in \text{vertices}(j)} x_v$$
  $$y_{\text{min}} = \min_{v \in \text{vertices}(j)} y_v, \quad y_{\text{max}} = \max_{v \in \text{vertices}(j)} y_v$$

### 3.2 Sub-grid Sampling coordinates
We divide the bounding box into $N \times N$ sub-cells (where $N = \text{budget\_subgrid\_size}$). The coordinates of each sub-grid sampling point $p_{(u, v)}$ are:
$$x_u = x_{\text{min}} + \left(u + 0.5\right) \frac{x_{\text{max}} - x_{\text{min}}}{N}, \quad u \in [0, N-1]$$
$$y_v = y_{\text{min}} + \left(v + 0.5\right) \frac{y_{\text{max}} - y_{\text{min}}}{N}, \quad v \in [0, N-1]$$

### 3.3 Bilinear Weight Evaluation
For each subgrid point $p_k = (x_u, y_v)$, standard bilinear interpolation from the source mesh is executed, yielding a list of source cell indices $s$ and their corresponding bilinear weights $w_{k, s}$ such that:
$$\sum_s w_{k, s} = 1.0$$

We accumulate these sub-point weights into a running total for the destination cell $j$:
$$\text{AccumulatedWeight}_{j, s} = \sum_{k=1}^{N^2} \frac{w_{k, s}}{N^2} \cdot \delta_k$$
where $\delta_k = 1$ if subgrid point $p_k$ is successfully mapped inside the source domain, and $\delta_k = 0$ otherwise.

### 3.4 Threshold Gating & Renormalization
Let $V_j = \sum_{k=1}^{N^2} \delta_k$ be the count of successfully mapped sub-points for destination cell $j$.
1. If $V_j / N^2 < \text{budget\_min\_valid\_fraction}$, the destination cell $j$ is considered unmapped. Its row will either be ignored or cause an error depending on `UnmappedAction`.
2. Otherwise, we renormalize the weights over the valid mapped sub-points to ensure strict conservation (partition of unity):
   $$\text{FinalWeight}_{j, s} = \text{AccumulatedWeight}_{j, s} \times \frac{N^2}{V_j}$$

---

## 4. Architectural & Implementation Strategy

### 4.1 Dispatch Routing
The stateless generator `WeightGenerator::generate` will dispatch to a new dedicated generator function:

```cpp
template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_budget(
    const topology::UnstructuredMesh<MemorySpace> &src_mesh,
    const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
    const RegridConfig &config);
```

### 4.2 Optimized Fast-Paths
- **Rectilinear Fast-Path:** If the source and destination meshes are detected as regular/rectilinear, coordinate-to-index projection will bypass spatial searches (ArborX) completely, mapping coordinates directly to cell indices in $O(1)$ time.
- **Unstructured Fallback:** If the grids are general unstructured meshes, we build an ArborX spatial query tree of the source mesh cells and execute queries in parallel on the Host or Device memory space.

---

## 5. Verification & Testing Plan

### 5.1 Unit Tests (`libs/axis/tests/test_budget.cpp`)
- **Constant Field Preservation:** Remapping a constant field of $1.0$ using any sub-grid size must produce exactly $1.0$ at all mapped destination points.
- **Bilinear Parity:** With `budget_subgrid_size = 1`, the produced `InterpolationMatrix` must be bitwise identical to the one generated by `InterpolationMethod::Bilinear`.
- **Min Valid Thresholding:** Verification that boundary cells are correctly unmapped or normalized according to `budget_min_valid_fraction`.

### 5.2 Property-Based Tests (`libs/axis/tests/prop_budget.cpp`)
- **Row Sums Unity (Partition of Unity):** For randomly generated meshes, every non-zero row in the matrix must sum to exactly $1.0$ (within double-precision epsilon).
- **Index Soundness:** Verification that all generated row and column indices are within their respective grid boundaries.
