#ifndef HALO_DETAIL_GPU_AWARE_PROBE_HPP
#define HALO_DETAIL_GPU_AWARE_PROBE_HPP

/// @file halo/detail/gpu_aware_probe.hpp
/// @brief Runtime detection of GPU-aware MPI support.
///
/// Probes the MPI implementation at runtime using MPIX_Query_cuda_support()
/// or MPIX_Query_rocm_support() when available. Falls back to the compile-time
/// HALO_GPU_AWARE_MPI flag when the MPIX extensions are not present.
///
/// The result is cached during Environment::initialize() and exposed via
/// Environment::is_gpu_aware_mpi().

#include <Kokkos_Core.hpp>

// The MPIX extensions for GPU-aware queries are typically declared in
// <mpi-ext.h> (OpenMPI) or available as symbols in the MPI library.
// We guard inclusion and usage with the standard feature-test macros.
#if defined(MPIX_CUDA_AWARE_SUPPORT) || defined(MPIX_ROCM_AWARE_SUPPORT)
// Already available from a prior mpi-ext.h include
#elif __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

namespace halo::detail {

/// @brief Probe the MPI implementation for GPU-aware (device pointer) support.
///
/// Strategy:
/// - CUDA backend: use MPIX_Query_cuda_support() if available, else fall back
///   to the HALO_GPU_AWARE_MPI compile-time flag.
/// - HIP backend: use MPIX_Query_rocm_support() if available, else fall back
///   to the HALO_GPU_AWARE_MPI compile-time flag.
/// - No device backend enabled: return false (no GPU pointers involved).
///
/// @return true if the MPI implementation reports support for device pointers.
inline bool gpu_aware_probe() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
        return MPIX_Query_cuda_support() != 0;
    #else
        #if defined(HALO_GPU_AWARE_MPI)
            return true;
        #else
            return false;
        #endif
    #endif
#elif defined(KOKKOS_ENABLE_HIP)
    #if defined(MPIX_ROCM_AWARE_SUPPORT)
        return MPIX_Query_rocm_support() != 0;
    #else
        #if defined(HALO_GPU_AWARE_MPI)
            return true;
        #else
            return false;
        #endif
    #endif
#else
    // No device backend enabled — GPU-aware MPI is irrelevant.
    #if defined(HALO_GPU_AWARE_MPI)
        return true;
    #else
        return false;
    #endif
#endif
}

} // namespace halo::detail

#endif // HALO_DETAIL_GPU_AWARE_PROBE_HPP
