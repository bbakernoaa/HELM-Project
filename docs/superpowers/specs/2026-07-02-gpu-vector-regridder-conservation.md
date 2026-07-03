# Spec: AXIS GPU Vector Regridding & Conservation Accounting (Gaps I & J)

## 1. Overview & Motivation
This specification defines the resolution of Gaps I and J:
- **Gap I:** Native GPU Vector Coupled Weight Generation and GPU Linker Instantiations.
- **Gap J:** Missing GPU Linker Instantiations for Climate Conservation Checks.

Currently, vector coupled regridding is evaluated sequentially on the Host CPU, and climate conservation checks (`source_integral`, etc.) are only instantiated for `HostSpace`. This leads to compile/link errors when compiling for GPUs (`CudaSpace`/`HIPSpace`) and introduces major host-side bottlenecks.

---

## 2. Design & Architecture

### A. GPU Parallelization of Vector coupled weights:
Inside `VectorWeightGenerator<MemorySpace>::generate`, we split into two compile-time branches:
- **Host Path (`is_host_space_v`):** Standard serial CPU implementation utilizing `std::vector` (retains full backward compatibility).
- **Device Path (`is_device_space_v`):**
  - Allocate device Views of size `nnz_scalar * 2` directly in the target GPU `MemorySpace`.
  - Execute a device-parallel `AssembleVectorWeightsDevice` kernel of size `nnz_scalar` to compute coupled trigonometric weights and indexes natively on the GPU:
    $$W_{u0} = w \cdot \cos(\alpha_d - \alpha_s), \quad W_{u1} = w \cdot \sin(\alpha_d - \alpha_s)$$
    $$W_{v0} = -w \cdot \sin(\alpha_d - \alpha_s), \quad W_{v1} = w \cdot \cos(\alpha_d - \alpha_s)$$

### B. GPU Explicit Template Instantiations:
- At the bottom of `libs/axis/src/solver/vector_regridder.cpp`, add the guarded CUDA and HIP explicit template instantiations:
  ```cpp
  #ifdef KOKKOS_ENABLE_CUDA
  template class VectorWeightGenerator<Kokkos::CudaSpace>;
  #endif
  #ifdef KOKKOS_ENABLE_HIP
  template class VectorWeightGenerator<Kokkos::HIPSpace>;
  #endif
  ```
- At the bottom of `libs/axis/src/solver/conservation.cpp`, add the guarded CUDA and HIP explicit template instantiations for all four core conservation functions:
  ```cpp
  #ifdef KOKKOS_ENABLE_CUDA
  template double source_integral<Kokkos::CudaSpace>(...);
  template double destination_integral<Kokkos::CudaSpace>(...);
  template ConservationReport check_conservation<Kokkos::CudaSpace>(...);
  template void adjust_by_fraction<Kokkos::CudaSpace>(...);
  #endif
  ```

This completely eliminates C++ GPU linker errors and ensures fully device-resident high-performance vector regridding and conservation audits.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Confirm all vector regridder unit tests and conservation property tests compile cleanly, link without errors, and pass with 100% success on CPU and GPU.
