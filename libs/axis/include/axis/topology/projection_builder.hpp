// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_PROJECTION_BUILDER_HPP
#define AXIS_TOPOLOGY_PROJECTION_BUILDER_HPP

/// @file axis/topology/projection_builder.hpp
/// @brief ProjectionBuilder — PROJ-based coordinate transform for Projected grids.
///
/// Transforms a regular grid laid out in projection space (given by center_x/y
/// coordinates in the descriptor's BufferViews) to geographic coordinates
/// (lon/lat in degrees) via PROJ. This is AXIS's single optional third-party
/// dependency; projection math has no equivalent in AMIO's stack, so it
/// genuinely belongs to AXIS.
///
/// Invoked by MeshFactory::from_descriptor for ConventionKind::Projected and
/// optionally by RuleGenerator for GridRulesParams of kind "Projected".
///
/// When AXIS_ENABLE_PROJ is OFF, the header is still includable but the build
/// function throws std::runtime_error("AXIS built without PROJ support").

#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/structured_grid.hpp>

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>

namespace axis::topology {

/// @class ProjectionBuilder
/// @brief PROJ-based projection builder for geographic coordinate transformation.
///
/// This class transforms regular grid coordinates specified in projection space (such as a bounding
/// box and grid resolutions) into geographic coordinates (longitude and latitude in degrees) using the
/// PROJ library via an internal handle.
/// This is the only helper in AXIS guarded by the optional AXIS_ENABLE_PROJ compilation dependency.
/// It is invoked by MeshFactory::from_descriptor when processing ConventionKind::Projected grids.
class ProjectionBuilder {
public:
    /// @brief Build a StructuredGrid by transforming projection-space coordinates to geographic (lon/lat) coordinates using PROJ.
    ///
    /// For GPU builds targeting a device memory space, the PROJ transformation is executed on the host CPU
    /// (since the third-party PROJ library is CPU-only), and the resulting coordinate views are subsequently
    /// transferred to the target device memory space using a Kokkos::deep_copy operation.
    ///
    /// @tparam MemorySpace The Kokkos memory space in which the output StructuredGrid's data arrays should be allocated. Defaults to Kokkos::HostSpace.
    /// @param params The ingest::ProjectedParams structure containing projection information, such as the PROJ string.
    /// @param buffers The ingest::BufferViews structure containing grid dimension sizes (ni, nj) and source coordinate buffers (center_x, center_y).
    /// @return StructuredGrid<MemorySpace> A StructuredGrid containing longitude and latitude coordinates in degrees, allocated in the specified MemorySpace.
    /// @throw std::runtime_error If AXIS was compiled without PROJ support (AXIS_ENABLE_PROJ is OFF), or if the underlying PROJ coordinate transformation library encounters an error.
    /// @throw std::invalid_argument If the provided projection string is empty, or if the input buffer views are invalid or of inconsistent sizes.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static StructuredGrid<MemorySpace>
    build(const ingest::ProjectedParams& params,
          const ingest::BufferViews& buffers);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_PROJECTION_BUILDER_HPP
