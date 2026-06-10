// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_STRUCTURED_GRID_HPP
#define AXIS_TOPOLOGY_STRUCTURED_GRID_HPP

/// @file axis/topology/structured_grid.hpp
/// @brief StructuredGrid<MemorySpace> — logically rectangular grid.
///
/// Represents a rectilinear or curvilinear structured grid stored as 1-D
/// center/corner coordinate arrays of size ni*nj (centers) and (ni+1)*(nj+1)
/// (corners). Provides to_unstructured() which converts each logical cell
/// into a quadrilateral element in the common internal FEM format.
///
/// Template parameter: Kokkos MemorySpace (HELM Law #2: explicit placement).

#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace axis::topology {

/// Logically rectangular (rectilinear or curvilinear) grid. Internally stores
/// center lon/lat arrays [ni*nj] and optionally corner arrays [(ni+1)*(nj+1)].
/// to_unstructured() converts to the common FEM format for the solver.
template <class MemorySpace = Kokkos::HostSpace>
class StructuredGrid {
public:
    using memory_space = MemorySpace;

    /// Construct from dimensions and 1-D center coordinate arrays.
    /// @param ni        Number of cells in i-direction (longitude-like)
    /// @param nj        Number of cells in j-direction (latitude-like)
    /// @param center_lon  Center longitudes as 1-D array [ni*nj], column-major
    /// @param center_lat  Center latitudes as 1-D array [ni*nj], column-major
    /// @param coord_sys   Coordinate system of the coordinates
    StructuredGrid(std::size_t ni, std::size_t nj,
                   Kokkos::View<double*, MemorySpace> center_lon,
                   Kokkos::View<double*, MemorySpace> center_lat,
                   CoordinateSystem coord_sys);

    // ── Dimension queries ────────────────────────────────────────────────────

    /// Number of cells in the i-direction (fastest-varying dimension).
    [[nodiscard]] std::size_t ni() const noexcept { return ni_; }

    /// Number of cells in the j-direction.
    [[nodiscard]] std::size_t nj() const noexcept { return nj_; }

    /// Coordinate system.
    [[nodiscard]] CoordinateSystem coord_system() const noexcept { return coord_sys_; }

    // ── Coordinate accessors (1-D flat arrays) ───────────────────────────────

    /// Center longitudes [ni*nj].
    [[nodiscard]] field_view<const double, 1> center_lon() const noexcept;

    /// Center latitudes [ni*nj].
    [[nodiscard]] field_view<const double, 1> center_lat() const noexcept;

    /// Corner longitudes [(ni+1)*(nj+1)]. Empty if corners not set.
    [[nodiscard]] field_view<const double, 1> corner_lon() const noexcept;

    /// Corner latitudes [(ni+1)*(nj+1)]. Empty if corners not set.
    [[nodiscard]] field_view<const double, 1> corner_lat() const noexcept;

    // ── Mutators ─────────────────────────────────────────────────────────────

    /// Set vertex (corner) coordinates for conservative methods.
    /// @param corner_lon  Corner longitudes [(ni+1)*(nj+1)]
    /// @param corner_lat  Corner latitudes [(ni+1)*(nj+1)]
    void set_corners(Kokkos::View<double*, MemorySpace> corner_lon,
                     Kokkos::View<double*, MemorySpace> corner_lat);

    // ── Conversion ───────────────────────────────────────────────────────────

    /// Convert to the common internal FEM unstructured mesh format.
    /// Produces exactly ni*nj quadrilateral cells. Each cell (i,j) has 4 corner
    /// nodes. If corners are not set, they are synthesized from centers using
    /// midpoints between adjacent centers.
    ///
    /// The conversion executes via a Kokkos parallel kernel for hardware
    /// portability (Requirement 18.3).
    [[nodiscard]] UnstructuredMesh<MemorySpace> to_unstructured() const;

private:
    std::size_t ni_{0};
    std::size_t nj_{0};
    Kokkos::View<double*, MemorySpace> center_lon_;
    Kokkos::View<double*, MemorySpace> center_lat_;
    Kokkos::View<double*, MemorySpace> corner_lon_;
    Kokkos::View<double*, MemorySpace> corner_lat_;
    CoordinateSystem coord_sys_{CoordinateSystem::SphericalDeg};

    /// Internal: synthesize corner coordinates from centers when corners not
    /// explicitly provided.
    void synthesize_corners() const;
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_STRUCTURED_GRID_HPP
