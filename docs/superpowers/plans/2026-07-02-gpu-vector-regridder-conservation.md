# AXIS GPU Vector Regridding & Conservation Accounting (Gaps I & J) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement device-resident parallel coupled vector weight generation, and introduce CUDA/HIP explicit template instantiations for `VectorWeightGenerator` and the climate `conservation` system to eliminate GPU link errors and CPU-GPU bottlenecks.

**Architecture:** Use `if constexpr` compile-time branching in `VectorWeightGenerator::generate`. If `MemorySpace` is a device space, run a parallel Kokkos kernel on device to compute coupled weights and indexes natively on GPU, and build device `InterpolationMatrix` objects. Add CUDA and HIP template instantiations in `vector_regridder.cpp` and `conservation.cpp`.

**Tech Stack:** C++20, Kokkos

## Global Constraints
- Coupled vector weight assembly must execute natively on the GPU execution space without Host-Device deep copies or synchronizations.
- Must produce results bitwise-identical to the host-only vector regridder.
- All code must pass Tier 1 isolation scans and compile clean.

---

### Task 1: Refactor `VectorWeightGenerator::generate` and add GPU instantiations

Optimize the vector coupled weight generator for GPU and add explicit instantiations for `CudaSpace` and `HIPSpace`.

**Files:**
- Modify: `libs/axis/src/solver/vector_regridder.cpp`

**Interfaces:**
- Consumes: Standard scalar `W_scalar` factor views and alpha rotations.
- Produces: Optimized `VectorWeightGenerator<MemorySpace>::generate` supporting device parallel coupled operations and compiling clean on CUDA/HIP.

- [ ] **Step 1: Implement compile-time branching in `VectorWeightGenerator::generate`**
  Modify the `generate` function in `libs/axis/src/solver/vector_regridder.cpp`:
  *   If `is_host_space_v<MemorySpace>`, execute the existing CPU-vector path (retains full compatibility).
  *   If `is_device_space_v<MemorySpace>` (GPU path):
      - Allocate device output Views directly in `MemorySpace` of size `nnz_vector`:
        ```cpp
        Kokkos::View<index_t *, MemorySpace> dev_u_rows("dev_u_rows", nnz_vector);
        Kokkos::View<index_t *, MemorySpace> dev_u_cols("dev_u_cols", nnz_vector);
        Kokkos::View<double *, MemorySpace> dev_u_vals("dev_u_vals", nnz_vector);

        Kokkos::View<index_t *, MemorySpace> dev_v_rows("dev_v_rows", nnz_vector);
        Kokkos::View<index_t *, MemorySpace> dev_v_cols("dev_v_cols", nnz_vector);
        Kokkos::View<double *, MemorySpace> dev_v_vals("dev_v_vals", nnz_vector);

        Kokkos::View<double *, MemorySpace> dev_frac_a("dev_frac_a", n_src * 2);
        Kokkos::View<double *, MemorySpace> dev_area_a("dev_area_a", n_src * 2);
        ```
      - Execute parallel Kokkos kernel to compute rotated weights and stacked indexes:
        ```cpp
        using exec_space = typename MemorySpace::execution_space;
        Kokkos::parallel_for(
            "AssembleVectorWeightsDevice", Kokkos::RangePolicy<exec_space>(0, nnz_scalar), KOKKOS_LAMBDA(const std::size_t k) {
                index_t j = rows(k);  // dst cell index
                index_t i = cols(k);  // src cell index
                double w = vals(k);

                double a_src = src_rotation.alpha(i);
                double a_dst = dst_rotation.alpha(j);
                double diff_alpha = a_dst - a_src;

                double cos_d = Kokkos::cos(diff_alpha);
                double sin_d = Kokkos::sin(diff_alpha);

                std::size_t idx0 = k * 2;
                std::size_t idx1 = k * 2 + 1;

                // --- Coupled Weights for W_u ---
                dev_u_rows(idx0) = j;
                dev_u_cols(idx0) = i;
                dev_u_vals(idx0) = w * cos_d;

                dev_u_rows(idx1) = j;
                dev_u_cols(idx1) = i + n_src;  // Stacked v component index
                dev_u_vals(idx1) = w * sin_d;

                // --- Coupled Weights for W_v ---
                dev_v_rows(idx0) = j;
                dev_v_cols(idx0) = i;
                dev_v_vals(idx0) = -w * sin_d;

                dev_v_rows(idx1) = j;
                dev_v_cols(idx1) = i + n_src;  // Stacked v component index
                dev_v_vals(idx1) = w * cos_d;
            });
        ```
      - Expand fraction and area Views on GPU:
        ```cpp
        auto dev_orig_frac_a = W_scalar.frac_a_view();
        auto dev_orig_area_a = W_scalar.area_a_view();

        Kokkos::parallel_for(
            "CopyFractionsDevice", Kokkos::RangePolicy<exec_space>(0, n_src), KOKKOS_LAMBDA(const std::size_t i) {
                dev_frac_a(i) = dev_orig_frac_a(i);
                dev_frac_a(i + n_src) = dev_orig_frac_a(i);
                dev_area_a(i) = dev_orig_area_a(i);
                dev_area_a(i + n_src) = dev_orig_area_a(i);
            });
        ```
      - Assemble and return `W_u` and `W_v` `InterpolationMatrix` objects directly in `MemorySpace`.

- [ ] **Step 2: Add GPU template instantiations at the bottom of the file**
  ```cpp
  #ifdef KOKKOS_ENABLE_CUDA
  template class VectorWeightGenerator<Kokkos::CudaSpace>;
  #endif

  #ifdef KOKKOS_ENABLE_HIP
  template class VectorWeightGenerator<Kokkos::HIPSpace>;
  #endif
  ```

- [ ] **Step 3: Compile to verify syntax**
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc)'`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```bash
  git add libs/axis/src/solver/vector_regridder.cpp
  git commit -m "feat(axis): implement native GPU pipeline and CUDA/HIP instantiations for VectorWeightGenerator"
  ```

---

### Task 2: Add CUDA/HIP template instantiations in `conservation.cpp`

Add the missing explicit template instantiations for `CudaSpace` and `HIPSpace` to `conservation.cpp` to eliminate linker errors on GPUs.

**Files:**
- Modify: `libs/axis/src/solver/conservation.cpp:115-125`

**Interfaces:** None.

- [ ] **Step 1: Append GPU explicit instantiations**
  Add the explicit template instantiations at the bottom of `libs/axis/src/solver/conservation.cpp`:
  ```cpp
  #ifdef KOKKOS_ENABLE_CUDA
  template double source_integral<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>);

  template double destination_integral<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>,
                                                          NormType);

  template ConservationReport check_conservation<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>,
                                                                    const InterpolationMatrix<Kokkos::CudaSpace> &, NormType);

  template void adjust_by_fraction<Kokkos::CudaSpace>(field_view<double, 1>, field_view<const double, 1>);
  #endif

  #ifdef KOKKOS_ENABLE_HIP
  template double source_integral<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>);

  template double destination_integral<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>,
                                                          NormType);

  template ConservationReport check_conservation<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>,
                                                                    const InterpolationMatrix<Kokkos::HIPSpace> &, NormType);

  template void adjust_by_fraction<Kokkos::HIPSpace>(field_view<double, 1>, field_view<const double, 1>);
  #endif
  ```

- [ ] **Step 2: Run complete C++ build and test suite**
  `docker exec helm-dev-env bash -lc 'cd /workspace/helm-project/libs/axis && cmake --build build-ci --parallel $(nproc) && cd build-ci && ctest --output-on-failure'`
  Expected: PASS 100%

- [ ] **Step 3: Commit**
  ```bash
  git add libs/axis/src/solver/conservation.cpp
  git commit -m "feat(axis): add CUDA and HIP explicit template instantiations in conservation.cpp"
  ```
