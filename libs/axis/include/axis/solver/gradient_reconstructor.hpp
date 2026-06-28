// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_GRADIENT_RECONSTRUCTOR_HPP
#define AXIS_SOLVER_GRADIENT_RECONSTRUCTOR_HPP

/// @file axis/solver/gradient_reconstructor.hpp
/// @brief Least-squares gradient reconstruction for Conservative2ndOrder.
///
/// Computes per-cell gradient vectors from a scalar field using a least-squares
/// fit over face-adjacent neighbors (CSR adjacency). Cells with fewer than 3
/// neighbors fall back to zero gradient (first-order accuracy). An optional
/// Barth-Jespersen monotonicity limiter prevents introduction of new extrema.
///
/// Templated on MemorySpace for device portability (HELM Law #2).

#include <Kokkos_Core.hpp>
#include <axis/types.hpp>

namespace axis::solver {

/// @brief Stateless gradient reconstruction utility for Conservative2ndOrder.
///
/// Reconstructs a linear gradient within each cell via least-squares fitting
/// over the cell's face-adjacent neighbors. The adjacency is provided in CSR
/// format (offsets + indices arrays). The caller (WeightGenerator) builds the
/// adjacency from mesh connectivity — GradientReconstructor only consumes it.
///
/// @tparam MemorySpace Kokkos memory space (HostSpace, CudaSpace, HIPSpace).
template <class MemorySpace = Kokkos::HostSpace>
struct GradientReconstructor {
    /// @brief Compute per-cell gradients via least-squares reconstruction.
    ///
    /// For each cell i with ≥ 3 face-adjacent neighbors, solves the 3×3 normal
    /// equations (A^T A) g = A^T b where:
    ///   - Each row of A is [dx_j, dy_j, dz_j] (centroid difference)
    ///   - Each element of b is (value_j - value_i)
    ///
    /// Uses Cramer's rule for the 3×3 solve (device-portable, no LAPACK).
    /// Falls back to zero gradient when:
    ///   - Cell has < 3 neighbors
    ///   - Normal matrix is singular (|det| < eps)
    ///
    /// @param cell_values  Scalar field values per cell [n_cells]
    /// @param centroids    Cell centroid coordinates [n_cells, 3] (x, y, z)
    /// @param adj_offsets  CSR row-pointer array [n_cells + 1]
    /// @param adj_indices  CSR column-index array [nnz_adjacency]
    /// @param grad         Output gradient vectors [n_cells, 3] (gx, gy, gz)
    /// @param use_limiter  If true, apply Barth-Jespersen monotonicity limiter
    static void compute(Kokkos::View<const double *, MemorySpace> cell_values, Kokkos::View<const double *[3], MemorySpace> centroids,
                        Kokkos::View<const index_t *, MemorySpace> adj_offsets, Kokkos::View<const index_t *, MemorySpace> adj_indices,
                        Kokkos::View<double *[3], MemorySpace> grad, bool use_limiter = false);
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_GRADIENT_RECONSTRUCTOR_HPP
