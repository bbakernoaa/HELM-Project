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

/// PROJ projection builder. Transforms a regular grid laid out in projection
/// space (bbox + resolution) to geographic coordinates via detail::Proj_Handle.
/// This is the ONLY helper guarded by an optional dependency (AXIS_ENABLE_PROJ);
/// projection math has no equivalent in AMIO's stack, so it genuinely belongs
/// to AXIS. Invoked by MeshFactory::from_descriptor for ConventionKind::Projected.
class ProjectionBuilder {
public:
    /// Build a StructuredGrid by transforming projection-space coordinates to
    /// geographic (lon/lat) coordinates using PROJ.
    ///
    /// @tparam MemorySpace  Target Kokkos memory space for the output grid.
    /// @param params        The projected params (contains the proj_string).
    /// @param buffers       The descriptor's buffer views (center_x, center_y, ni, nj).
    /// @return A StructuredGrid with coordinates in geographic lon/lat (degrees).
    ///
    /// @throws std::runtime_error if AXIS was built without PROJ support.
    /// @throws std::runtime_error if the PROJ transformation fails.
    /// @throws std::invalid_argument if the proj_string is empty or buffers are invalid.
    ///
    /// For GPU builds (device MemorySpace): the PROJ transform executes on the
    /// host first (PROJ is CPU-only), then the result is deep_copied to the
    /// target device memory space.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static StructuredGrid<MemorySpace>
    build(const ingest::ProjectedParams& params,
          const ingest::BufferViews& buffers);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_PROJECTION_BUILDER_HPP
