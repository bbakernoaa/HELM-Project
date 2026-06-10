// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_MEMORY_TRAITS_HPP
#define AXIS_DETAIL_MEMORY_TRAITS_HPP

/// @file axis/detail/memory_traits.hpp
/// @brief Compile-time memory-space dispatch traits.
///
/// The AXIS solver and topology generators are templated on a Kokkos
/// MemorySpace. These traits select host vs device code paths at compile
/// time with no virtual dispatch and no Unified Virtual Memory reliance
/// (HELM Law #2).

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// is_device_space<MemorySpace> — true when MemorySpace is a GPU/device space
// (data not directly host-addressable).
// ─────────────────────────────────────────────────────────────────────────────

/// Primary template: default to false (host-like space).
template <class MemorySpace>
struct is_device_space : std::false_type {};

/// HostSpace is explicitly host-accessible.
template <>
struct is_device_space<Kokkos::HostSpace> : std::false_type {};

#ifdef KOKKOS_ENABLE_CUDA
/// CudaSpace is a device space (dedicated GPU DRAM).
template <>
struct is_device_space<Kokkos::CudaSpace> : std::true_type {};

/// CudaUVMSpace is technically device-accessible but page-migrating; treat
/// as device for AXIS's purposes (no UVM reliance policy).
template <>
struct is_device_space<Kokkos::CudaUVMSpace> : std::true_type {};
#endif

#ifdef KOKKOS_ENABLE_HIP
/// HIPSpace is a device space (AMD GPU DRAM).
template <>
struct is_device_space<Kokkos::HIPSpace> : std::true_type {};

/// HIPManagedSpace — treated as device (same policy as CudaUVMSpace).
template <>
struct is_device_space<Kokkos::HIPManagedSpace> : std::true_type {};
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Variable template shortcut
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience variable template for is_device_space.
template <class MemorySpace>
inline constexpr bool is_device_space_v = is_device_space<MemorySpace>::value;

// ─────────────────────────────────────────────────────────────────────────────
// exec_space_t<MemorySpace> — maps a MemorySpace to its default execution space
// ─────────────────────────────────────────────────────────────────────────────

/// The execution space paired with a memory space (Kokkos default mapping).
/// For Kokkos::HostSpace this is typically Kokkos::Serial or Kokkos::OpenMP;
/// for CudaSpace it is Kokkos::Cuda; for HIPSpace it is Kokkos::HIP.
template <class MemorySpace>
using exec_space_t = typename MemorySpace::execution_space;

} // namespace axis::detail

#endif // AXIS_DETAIL_MEMORY_TRAITS_HPP
