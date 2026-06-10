// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_ENUMS_HPP
#define AXIS_TOPOLOGY_ENUMS_HPP

/// @file axis/topology/enums.hpp
/// @brief Topology-level enumerations: ElementType, StaggerLoc, CoordinateSystem.
///
/// CoordinateSystem is also defined in axis/ingest/grid_descriptor.hpp so the
/// ingest contract remains self-contained (no topology dependency). The two
/// definitions share identical values and semantics. Topology code should use
/// axis::topology::CoordinateSystem; ingest code uses axis::ingest::CoordinateSystem.

#include <cstdint>

namespace axis::topology {

// ─────────────────────────────────────────────────────────────────────────────
// ElementType — cell topology classification for mixed-element CSR meshes.
// ─────────────────────────────────────────────────────────────────────────────

/// Cell element type (2-D surface meshes primarily). Mixed-element meshes store
/// per-cell types alongside the CSR connectivity; the Polygon value covers
/// general polygonal cells with arbitrary vertex count.
enum class ElementType : std::uint8_t {
    Triangle,       ///< 3-node triangle
    Quadrilateral,  ///< 4-node quadrilateral
    Polygon         ///< General polygon (n-node, mixed-element CSR)
};

// ─────────────────────────────────────────────────────────────────────────────
// StaggerLoc — stagger location (center vs corner vs edge) within a cell.
// ─────────────────────────────────────────────────────────────────────────────

/// Field stagger location relative to cells.
enum class StaggerLoc : std::uint8_t {
    Center,  ///< Cell centroid (default for FV fields)
    Corner,  ///< Cell vertex/corner
    Edge     ///< Cell edge midpoint
};

// ─────────────────────────────────────────────────────────────────────────────
// CoordinateSystem — coordinate system for node/cell coordinates.
//
// Also defined identically in axis::ingest (grid_descriptor.hpp) to keep the
// ingest contract self-contained. Both definitions have the same underlying
// values (SphericalDeg=0, SphericalRad=1, Cartesian3D=2).
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinate system for node/cell coordinates.
enum class CoordinateSystem : std::uint8_t {
    SphericalDeg,   ///< Geographic lon/lat in degrees
    SphericalRad,   ///< Geographic lon/lat in radians
    Cartesian3D     ///< 3-D Cartesian (x, y, z)
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_ENUMS_HPP
