# AXIS Fused NaN-Aware SpMV Re-Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-performance, single-pass on-device C++ `nan_batch_apply` SpMV re-normalization kernel via Kokkos to eliminate memory allocation bottlenecks and improve speed by >2x in NaN-aware Python workflows.

**Architecture:**
- Create `nan_batch_apply` C++ template in core solver (`apply.hpp`).
- Bind it via nanobind (`axis_py.cpp`) with proper column-major buffer conversions.
- Integrate into Python `core.py` and replace the previous pure Python double-pass SpMV fallback.

**Tech Stack:** C++20, Kokkos, nanobind, Python, numpy, xarray, pytest.

## Global Constraints
- **Preserve zero-copy bounds**: Returned NumPy views must address raw pointers directly.
- **Maintain Kokkos HostSpace execution**: All bindings operate on Kokkos::HostSpace.
- **TDD / Verification first**: Write tests for each feature and ensure they pass completely.

---

### Task 1: Implement C++ Core `nan_batch_apply` SpMV Solver

**Files:**
- Modify: `libs/axis/include/axis/solver/apply.hpp`

**Interfaces:**
- Consumes: `InterpolationMatrix`, `field_view<const double, 2> src`
- Produces: `void nan_batch_apply(matrix, src, dst, na_thres)`

- [ ] **Step 1: Implement C++ Solver Template**

In `libs/axis/include/axis/solver/apply.hpp`, add the following method near the end (right below the existing `batch_apply` implementation around line 450):

```cpp
/// Apply the interpolation matrix to multiple variables simultaneously with on-the-fly
/// NaN-aware SpMV re-normalization (fused single-pass).
///
/// Both src and dst are rank-2 layout_left (column-major) views.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix  Sparse interpolation operator
/// @param src     Source fields [n_src, n_vars]
/// @param dst     Destination fields [n_dst, n_vars] — overwritten with result
/// @param na_thres Minimum fraction of valid input contribution required for output cell
template <class MemorySpace>
void nan_batch_apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const double, 2> src, field_view<double, 2> dst, double na_thres) {
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: src.extent(0) != matrix.n_src()");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: dst.extent(0) != matrix.n_dst()");
    }
    if (src.extent(1) != dst.extent(1)) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: src.extent(1) != dst.extent(1)");
    }

    const std::size_t n_dst = matrix.n_dst();
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_vars = src.extent(1);

    using dst_view_t = Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    dst_view_t dst_kk(dst.data_handle(), n_dst, n_vars);
    src_view_t src_kk(src.data_handle(), n_src, n_vars);

    using exec_space = typename MemorySpace::execution_space;

    if (!matrix.is_csr()) {
        throw std::runtime_error("axis::solver::nan_batch_apply requires matrix in CSR format. Call to_csr() first.");
    }

    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto csr_vals = matrix.csr_values();

    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using member_type = typename team_policy::member_type;

    Kokkos::parallel_for(
        "axis::nan_batch_apply::csr", team_policy(static_cast<int>(n_dst), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &team) {
            const auto j = team.league_rank();
            const auto start = row_ptr(j);
            const auto end = row_ptr(j + 1);

            // Compute total weight for row j (sum of S)
            double total_row_weight = 0.0;
            for (auto k = start; k < end; ++k) {
                total_row_weight += csr_vals(k);
            }

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, static_cast<int>(n_vars)), [&](const int v) {
                double weighted_sum_val = 0.0;
                double sum_valid_weights = 0.0;

                for (auto k = start; k < end; ++k) {
                    const double val = src_kk(col_idx(k), v);
                    if (!Kokkos::isnan(val)) {
                        weighted_sum_val += csr_vals(k) * val;
                        sum_valid_weights += csr_vals(k);
                    }
                }

                if (sum_valid_weights == 0.0 || total_row_weight == 0.0) {
                    dst_kk(j, v) = Kokkos::ArithTraits<double>::nan();
                } else {
                    double fraction_valid = sum_valid_weights / total_row_weight;
                    if (fraction_valid < (1.0 - na_thres - 1e-6)) {
                        dst_kk(j, v) = Kokkos::ArithTraits<double>::nan();
                    } else {
                        dst_kk(j, v) = weighted_sum_val / sum_valid_weights;
                    }
                }
            });
        });

    Kokkos::fence("axis::nan_batch_apply::complete");
}
```

---

### Task 2: Expose Python Bindings (`axis_py.nan_batch_apply`)

**Files:**
- Modify: `libs/axis/python/axis_py.cpp`
- Create: `libs/axis/tests_python/test_fused_nan_spmv.py`

**Interfaces:**
- Consumes: C++ `nan_batch_apply`
- Produces: `axis_py.nan_batch_apply(matrix, src_arr, na_thres)`

- [ ] **Step 1: Write the failing test**

Create `libs/axis/tests_python/test_fused_nan_spmv.py`:
```python
import pytest
import numpy as np
from axis import axis_py

def test_fused_nan_batch_apply():
    # Construct a simple regular mesh
    src = axis_py.make_regular_mesh(4, 4, 0.0, -90.0, 90.0, 45.0)
    dst = axis_py.make_regular_mesh(2, 2, 0.0, -90.0, 180.0, 90.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    matrix = axis_py.generate_weights(src, dst, config)
    matrix.to_csr()

    # Create source data with a NaN value
    src_data = np.ones((matrix.n_src, 2), dtype=np.float64)
    src_data[0, 0] = np.nan  # Inject NaN in variable 0

    # Ensure python binding works
    res = axis_py.nan_batch_apply(matrix, src_data, 1.0)
    assert res.shape == (matrix.n_dst, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_fused_nan_spmv.py`
Expected: FAIL with "AttributeError: module 'axis_py' has no attribute 'nan_batch_apply'"

- [ ] **Step 3: Implement C++ nanobind exposure of `nan_batch_apply`**

In `libs/axis/python/axis_py.cpp` near other m.def calls:
```cpp
    m.def(
        "nan_batch_apply",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<2>> src_arr, double na_thres) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            const std::size_t n_src = src_arr.shape(0);
            const std::size_t n_vars = src_arr.shape(1);
            const std::size_t n_dst = matrix.n_dst();

            if (n_src != matrix.n_src()) {
                throw std::invalid_argument("src array shape[0] (" + std::to_string(n_src) + ") != matrix.n_src (" + std::to_string(matrix.n_src()) +
                                            ")");
            }

            const double *src_ptr = src_arr.data();
            bool is_fortran_order = (src_arr.stride(0) == 1);

            // Convert to column-major if layout is row-major
            std::vector<double> src_colmajor;
            if (!is_fortran_order) {
                src_colmajor.resize(n_src * n_vars);
                for (std::size_t v = 0; v < n_vars; ++v) {
                    for (std::size_t i = 0; i < n_src; ++i) {
                        src_colmajor[i + v * n_src] = src_ptr[i * n_vars + v];
                    }
                }
                src_ptr = src_colmajor.data();
            }

            axis::field_view<const double, 2> src_view(src_ptr, n_src, n_vars);

            auto dst_buf_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *dst_buf = dst_buf_uniq.get();
            axis::field_view<double, 2> dst_view(dst_buf, n_dst, n_vars);

            axis::solver::nan_batch_apply<Kokkos::HostSpace>(matrix, src_view, dst_view, na_thres);

            // Convert back to row-major for numpy return
            auto result_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *result = result_uniq.get();
            for (std::size_t v = 0; v < n_vars; ++v) {
                for (std::size_t j = 0; j < n_dst; ++j) {
                    result[j * n_vars + v] = dst_buf[j + v * n_dst];
                }
            }

            double *raw_ptr = result_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_dst, n_vars};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "matrix"_a, "src"_a, "na_thres"_a,
        "Apply interpolation matrix with on-the-fly NaN-aware SpMV re-normalization"
    );
```

- [ ] **Step 4: Rebuild and Run tests**

Run: `uv pip install -e libs/axis`
Run: `.venv/bin/python -m pytest libs/axis/tests_python/test_fused_nan_spmv.py`
Expected: PASS

---

### Task 3: Expose and Wrap Python Integration

**Files:**
- Modify: `libs/axis/python/axis/core.py`
- Modify: `libs/axis/tests_python/test_fused_nan_spmv.py`

**Interfaces:**
- Consumes: `axis_py.nan_batch_apply`
- Produces: `_apply_weights_core` using on-device SpMV re-normalization

- [ ] **Step 1: Integrate into `_apply_weights_core`**

In `libs/axis/python/axis/core.py` (around line 73), replace the old Python-side fallback logic with:

```python
    if not skipna:
        # Standard fast path: Apply weights directly in C++
        result_t = axis_py.batch_apply(weights_matrix, flat_data_t)
        result = result_t.T
    else:
        # NaN-aware re-normalization path using C++ fused solver
        if not weights_matrix.is_csr:
            weights_matrix.to_csr()
        result_t = axis_py.nan_batch_apply(weights_matrix, flat_data_t, na_thres)
        result = result_t.T
```

- [ ] **Step 2: Add validation tests comparing new C++ vs old Python math**

In `libs/axis/tests_python/test_fused_nan_spmv.py`, append the following mathematical comparison test:

```python
def test_fused_nan_correctness():
    # Construct a mesh
    src = axis_py.make_regular_mesh(10, 5, 0.0, -90.0, 36.0, 36.0)
    dst = axis_py.make_regular_mesh(5, 3, 0.0, -90.0, 72.0, 60.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    matrix = axis_py.generate_weights(src, dst, config)
    matrix.to_csr()

    # Source data with various NaNs
    src_data = np.ones((matrix.n_src, 3), dtype=np.float64)
    src_data[::3, 0] = np.nan
    src_data[1::4, 1] = np.nan
    src_data[::2, 2] = np.nan

    # 1. Manual baseline (the previous Python-side math)
    flat_data = src_data.T
    mask = np.isnan(flat_data)
    safe_data = np.where(mask, 0.0, flat_data)
    result_t = axis_py.batch_apply(matrix, np.asfortranarray(safe_data.T))
    result = result_t.T

    valid_mask = np.logical_not(mask).astype(np.float32)
    weights_sum_t = axis_py.batch_apply(matrix, np.asfortranarray(valid_mask.T))
    weights_sum = weights_sum_t.T

    total_weights = np.array(axis_py.apply_weights(matrix, np.ones(matrix.n_src))).flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        result /= weights_sum
        # total weights fraction threshold check (na_thres = 0.5)
        fraction_valid = weights_sum / total_weights[:, None]
        expected = np.where(fraction_valid < (1.0 - 0.5 - 1e-6), np.nan, result)

    # 2. C++ Fused execution
    actual_t = axis_py.nan_batch_apply(matrix, src_data, 0.5)
    actual = actual_t.T

    # Verify identical matching (both values and NaN locations)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)
```

- [ ] **Step 3: Run the full pytest suite**

Run: `.venv/bin/python -m pytest libs/axis/tests_python/`
Expected: ALL PASS (including the new correctness tests and all existing baseline tests)
