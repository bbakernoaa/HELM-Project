#ifndef HALO_DETAIL_MEMORY_TRAITS_HPP
#define HALO_DETAIL_MEMORY_TRAITS_HPP

/// @file halo/detail/memory_traits.hpp
/// @brief Compile-time memory space traits for GPU-aware dispatch.
///
/// Provides type traits to detect whether a Kokkos::View resides in device
/// memory and whether host staging is required for MPI operations based on
/// the HALO_GPU_AWARE_MPI compile-time flag.

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace halo::detail {

// ─── Device Space Detection ─────────────────────────────────────────────────

/// Primary template: assume memory space is NOT a device space.
template <typename MemorySpace>
struct is_device_space : std::false_type {};

// Specializations for known device memory spaces, guarded by Kokkos backend macros.

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct is_device_space<Kokkos::CudaSpace> : std::true_type {};

template <>
struct is_device_space<Kokkos::CudaUVMSpace> : std::true_type {};
#endif

#ifdef KOKKOS_ENABLE_HIP
template <>
struct is_device_space<Kokkos::HIPSpace> : std::true_type {};
#endif

#ifdef KOKKOS_ENABLE_OPENACC
template <>
struct is_device_space<Kokkos::Experimental::OpenACCSpace> : std::true_type {};
#endif

/// Variable template shorthand for is_device_space.
template <typename MemorySpace>
inline constexpr bool is_device_space_v = is_device_space<MemorySpace>::value;

// ─── View Memory Space Extraction ───────────────────────────────────────────

/// Extract the memory space type from a Kokkos::View type.
template <typename ViewType>
using view_memory_space_t = typename ViewType::memory_space;

// ─── Staging Requirement ────────────────────────────────────────────────────

/// Determine at compile time whether a view requires host staging for MPI.
///
/// A view requires staging when:
///   1. It resides in a device memory space, AND
///   2. GPU-aware MPI is NOT enabled (HALO_GPU_AWARE_MPI not defined).
///
/// When HALO_GPU_AWARE_MPI is defined, device pointers can be passed directly
/// to MPI, so staging is never required.
template <typename ViewType>
inline constexpr bool requires_staging_v =
    is_device_space_v<view_memory_space_t<ViewType>>
#ifdef HALO_GPU_AWARE_MPI
    && false;  // GPU-aware MPI available: never stage
#else
    ;          // No GPU-aware MPI: always stage device views
#endif

// ─── Host Mirror Type ───────────────────────────────────────────────────────

/// Host mirror type alias for creating staging buffers.
template <typename ViewType>
using host_mirror_t = typename ViewType::HostMirror;

} // namespace halo::detail

#endif // HALO_DETAIL_MEMORY_TRAITS_HPP
