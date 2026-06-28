// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_VECTOR_REGRIDDER_HPP
#define AXIS_SOLVER_VECTOR_REGRIDDER_HPP

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <utility>

namespace axis::solver {

/// @struct GridRotation
/// @brief Struct holding local grid rotation angles (in radians) for curvilinear grids.
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
struct GridRotation {
    Kokkos::View<const double *, MemorySpace> alpha;  ///< Local grid rotation angle per grid cell
};

/// @class VectorWeightGenerator
/// @brief Generates pre-assembled, coupled interpolation matrices for 2D vector fields.
///
/// This generator mathematically couples local grid-relative coordinate frame rotations
/// with standard scalar spatial interpolation weights into pre-assembled coupled matrices,
/// allowing vector remapping to be performed in a single high-performance SpMV step.
///
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
class VectorWeightGenerator {
   public:
    /// @brief Generate coupled weight matrices for u and v vector components.
    /// @param src_mesh      Source unstructured mesh.
    /// @param dst_mesh      Destination unstructured mesh.
    /// @param src_rotation  Rotation angles at source grid cells.
    /// @param dst_rotation  Rotation angles at destination grid cells.
    /// @param config        Scalar regridding configuration.
    /// @return A pair of InterpolationMatrix objects: first is W_u, second is W_v.
    static std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>> generate(
        const topology::UnstructuredMesh<MemorySpace> &src_mesh, const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
        const GridRotation<MemorySpace> &src_rotation, const GridRotation<MemorySpace> &dst_rotation, const RegridConfig &config);
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_VECTOR_REGRIDDER_HPP
