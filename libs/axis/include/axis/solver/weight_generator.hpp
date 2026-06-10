// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_WEIGHT_GENERATOR_HPP
#define AXIS_SOLVER_WEIGHT_GENERATOR_HPP

/// @file axis/solver/weight_generator.hpp
/// @brief Static interface for computing interpolation weights from source and
///        destination meshes.
///
/// WeightGenerator is a stateless facade (no instance state). It provides:
///   - generate() (single-rank): produces an InterpolationMatrix from two
///     UnstructuredMesh instances and a RegridConfig.
///   - generate() (distributed): additionally accepts global_ids, owner maps,
///     and outputs a HaloPattern for off-rank source gather.
///
/// Supported interpolation methods:
///   - Bilinear: inverse-distance / barycentric weights within the containing
///     source cell. Source cell areas (area_a) are set to 0.0 (ESMF convention).
///   - Conservative1stOrder: area-weighted overlap. Weights satisfy first-order
///     conservation and are non-negative. Supports DstArea and FracArea
///     normalization.
///
/// @tparam MemorySpace Kokkos memory space for internal computation and output.

#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace axis::solver {

/// Stateless weight-generation facade.
///
/// All methods are static — no instance state is needed. This mirrors the
/// factory pattern used by MeshFactory in the topology namespace.
class WeightGenerator {
public:
    WeightGenerator() = delete;

    // ─────────────────────────────────────────────────────────────────────────
    // Single-rank generate
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute interpolation weights between a source and destination mesh.
    ///
    /// @tparam MemorySpace Kokkos memory space (default Kokkos::HostSpace)
    /// @param src_mesh  Source unstructured mesh
    /// @param dst_mesh  Destination unstructured mesh
    /// @param config    Regridding configuration (method, norm, unmapped, etc.)
    /// @return          Sparse interpolation matrix in COO form
    ///
    /// @throws std::invalid_argument if src/dst CoordinateSystem values differ
    /// @throws std::runtime_error    if unmapped==Error and any dst cell is uncovered
    template <class MemorySpace = Kokkos::HostSpace>
    static InterpolationMatrix<MemorySpace>
    generate(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
             const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
             const RegridConfig& config);

    // ─────────────────────────────────────────────────────────────────────────
    // Distributed generate (publishes HaloPattern for off-rank gather)
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute interpolation weights in distributed mode.
    ///
    /// In addition to producing a local InterpolationMatrix, this overload
    /// inspects factor columns against local ownership and publishes a
    /// HaloPattern describing which off-rank source cells are needed.
    ///
    /// @tparam MemorySpace   Kokkos memory space
    /// @param src_mesh       Local partition of the source mesh
    /// @param dst_mesh       Local partition of the destination mesh
    /// @param config         Regridding configuration
    /// @param src_global_ids Global cell IDs for local source cells [n_src_local]
    /// @param dst_global_ids Global cell IDs for local destination cells [n_dst_local]
    /// @param owner_of_src   Mapping from global source cell ID to owning rank
    ///
    /// @return Pair of (InterpolationMatrix, HaloPattern)
    ///
    /// @throws std::invalid_argument if src/dst CoordinateSystem values differ
    /// @throws std::runtime_error    if unmapped==Error and any dst cell is uncovered
    template <class MemorySpace = Kokkos::HostSpace>
    static std::pair<InterpolationMatrix<MemorySpace>, HaloPattern>
    generate(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
             const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
             const RegridConfig& config,
             Kokkos::View<const index_t*, MemorySpace> src_global_ids,
             Kokkos::View<const index_t*, MemorySpace> dst_global_ids,
             const std::vector<int>& owner_of_src);

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal dispatch helpers
    // ─────────────────────────────────────────────────────────────────────────

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace>
    generate_bilinear(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                      const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                      const RegridConfig& config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace>
    generate_nearest(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                     const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                     const RegridConfig& config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace>
    generate_bicubic(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                     const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                     const RegridConfig& config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace>
    generate_patch(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                   const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                   const RegridConfig& config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace>
    generate_conservative(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                          const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                          const RegridConfig& config);
};

} // namespace axis::solver

#endif // AXIS_SOLVER_WEIGHT_GENERATOR_HPP
