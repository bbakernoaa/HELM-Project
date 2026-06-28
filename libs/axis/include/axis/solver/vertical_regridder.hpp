// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_VERTICAL_REGRIDDER_HPP
#define AXIS_SOLVER_VERTICAL_REGRIDDER_HPP

#include <Kokkos_Core.hpp>

namespace axis::solver {

/// @class VerticalRegridder
/// @brief Thread-safe, device-resident solver for 1D vertical tension spline interpolation.
///
/// This class handles vertical regridding of meteorology and oceanography fields. It supports
/// both uniform 1D vertical coordinates and column-varying 2D coordinates (e.g. terrain-following
/// sigma levels). It leverages parallel execution over horizontal columns on Host or Device.
///
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
class VerticalRegridder {
   public:
    /// @brief Interpolate a 2D field from source vertical levels to destination vertical levels (Uniform 1D).
    /// @param src_field   Source field of shape (N_col, N_src_lev).
    /// @param dst_field   Destination field of shape (N_col, N_dst_lev).
    /// @param src_levels  Source level coordinates of shape (N_src_lev).
    /// @param dst_levels  Destination level coordinates of shape (N_dst_lev).
    /// @param tension     Tension parameter (>= 0.0). 0.0 corresponds to standard cubic splines.
    static void interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                            Kokkos::View<const double *, MemorySpace> src_levels, Kokkos::View<const double *, MemorySpace> dst_levels,
                            double tension = 0.0);

    /// @brief Interpolate a 2D field from source vertical levels to destination vertical levels (Varying 2D).
    /// @param src_field   Source field of shape (N_col, N_src_lev).
    /// @param dst_field   Destination field of shape (N_col, N_dst_lev).
    /// @param src_levels  Source level coordinates of shape (N_col, N_src_lev).
    /// @param dst_levels  Destination level coordinates of shape (N_col, N_dst_lev).
    /// @param tension     Tension parameter (>= 0.0). 0.0 corresponds to standard cubic splines.
    static void interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                            Kokkos::View<const double **, MemorySpace> src_levels, Kokkos::View<const double **, MemorySpace> dst_levels,
                            double tension = 0.0);
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_VERTICAL_REGRIDDER_HPP
