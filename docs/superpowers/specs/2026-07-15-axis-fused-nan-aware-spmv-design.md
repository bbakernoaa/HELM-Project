# Design Specification: AXIS Fused NaN-Aware SpMV Re-Normalization

## 1. Overview
In standard climate and weather models, datasets often contain missing values or land/sea masks represented as `NaN` (Not a Number) values. During spatial regridding, if any source cell contributing to a destination cell is `NaN`, a standard Sparse Matrix-Vector Multiply (SpMV) would propagate that `NaN` across the destination grid. To prevent this, AXIS provides a **NaN-aware re-normalization** pathway (`skipna=True`).

Previously, this was implemented in pure Python by making multiple passes over the dataset and calling standard `batch_apply` twice (once on a zero-substituted field, once on a boolean validation mask), resulting in high memory allocations and slow execution. This specification details the design of a **unified, fused C++ `nan_batch_apply` SpMV kernel** executed entirely on device spaces (Kokkos) in a single pass.

---

## 2. Component Design & Interfaces

### 2.1 C++ Core `nan_batch_apply` SpMV Solver
We will add `nan_batch_apply` directly to `libs/axis/include/axis/solver/apply.hpp`. This kernel executes a single-pass SpMV and on-the-fly re-normalization using a CSR (Compressed Sparse Row) layout.

*   **Signature**:
    ```cpp
    template <class MemorySpace>
    void nan_batch_apply(
        const InterpolationMatrix<MemorySpace> &matrix,
        field_view<const double, 2> src,
        field_view<double, 2> dst,
        double na_thres
    );
    ```

*   **Mathematical Formula (Single-Pass Fusion)**:
    For each destination cell $j$ and variable $v$:
    *   $\text{weighted\_sum} = \sum_{k \in \text{nonzeros}(j)} S_k \cdot \text{src}(\text{col}_k, v) \quad \text{for non-NaN values}$
    *   $\text{sum\_valid\_weights} = \sum_{k \in \text{nonzeros}(j)} S_k \quad \text{for non-NaN values}$
    *   $\text{total\_row\_weight} = \sum_{k \in \text{nonzeros}(j)} S_k \quad \text{all weights}$
    *   If $\text{sum\_valid\_weights} = 0$:
        $$\text{dst}(j, v) = \text{NaN}$$
    *   Else if $\frac{\text{sum\_valid\_weights}}{\text{total\_row\_weight}} < (1.0 - \text{na\_thres} - 10^{-6})$:
        $$\text{dst}(j, v) = \text{NaN}$$
    *   Else:
        $$\text{dst}(j, v) = \frac{\text{weighted\_sum}}{\text{sum\_valid\_weights}}$$

---

### 2.2 `nan_batch_apply` Python Bindings
We will expose this new C++ solver directly via `nanobind` in `libs/axis/python/axis_py.cpp`.

*   **Binding Code**:
    ```cpp
    m.def(
        "nan_batch_apply",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<2>> src_arr, double na_thres) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            const std::size_t n_src = src_arr.shape(0);
            const std::size_t n_vars = src_arr.shape(1);
            const std::size_t n_dst = matrix.n_dst();

            if (n_src != matrix.n_src()) {
                throw std::invalid_argument("src array shape[0] != matrix.n_src");
            }

            const double *src_ptr = src_arr.data();
            bool is_fortran_order = (src_arr.stride(0) == 1);

            // Handle column-major conversion if layout is row-major
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

            // Execute fused C++ nan_batch_apply SpMV
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

---

### 2.3 High-Level Python Layer Integration
In `libs/axis/python/axis/core.py`, we will completely replace the multi-pass, memory-intensive Python-side NaN re-normalization with our new C++ fused SpMV kernel:

```python
    # ... inside _apply_weights_core ...
    if not skipna:
        result_t = axis_py.batch_apply(weights_matrix, flat_data_t)
        result = result_t.T
    else:
        # Call single-pass fused C++ SpMV kernel
        if not weights_matrix.is_csr:
            weights_matrix.to_csr()
        result_t = axis_py.nan_batch_apply(weights_matrix, flat_data_t, na_thres)
        result = result_t.T
```

---

## 3. Verification & Testing Spec
We will verify both correctness and performance gains using a dedicated unit-test suite:
*   **`test_fused_nan_spmv.py`**:
    *   **Test Case 1: Simple NaN Masking**: Checks that if an entire source row is `NaN` or doesn't meet the `na_thres`, the output correctly becomes `NaN`.
    *   **Test Case 2: Multi-Variable SpMV**: Asserts that `nan_batch_apply` produces mathematically identical floating-point values to the old two-pass python re-normalization, within precision limits.
    *   **Test Case 3: Performance & Memory Benchmark**: Validates that no temporary copies are allocated during NaN-aware SpMVs, showing at least a 2x memory usage improvement.
