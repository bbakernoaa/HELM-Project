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

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>

namespace axis::topology {

/// @brief Represents a logically rectangular (rectilinear or curvilinear) structured grid.
///
/// This class represents a structured grid in logical 2D space. Internally, it stores the
/// coordinate arrays for cell centers and cell corners as flat 1-D Kokkos::Views in a specific
/// memory space. The center arrays contain `ni * nj` elements, while the corner arrays contain
/// `(ni + 1) * (nj + 1)` elements. It supports conservative methods by allowing the explicit
/// configuration of corner coordinates, and can be converted into the common internal finite-element
/// `UnstructuredMesh` format via the `to_unstructured()` member function.
///
/// @tparam MemorySpace The Kokkos memory space used for internal array storage (e.g., Kokkos::HostSpace, Kokkos::CudaSpace, Kokkos::HIPSpace).
template <class MemorySpace = Kokkos::HostSpace>
class StructuredGrid {
   public:
    /// @brief Type alias for the memory space template parameter.
    using memory_space = MemorySpace;

    /// @brief Constructs a StructuredGrid from dimensions and 1-D center coordinate arrays.
    ///
    /// This constructor adopts the provided center coordinate Kokkos::View objects, moving them in
    /// without performing deep copies.
    ///
    /// @param ni The number of cells in the i-direction (longitude-like, fastest-varying dimension) as a std::size_t.
    /// @param nj The number of cells in the j-direction (latitude-like) as a std::size_t.
    /// @param center_lon A rank-1 Kokkos::View of size [ni*nj] containing cell center longitudes in column-major order.
    /// @param center_lat A rank-1 Kokkos::View of size [ni*nj] containing cell center latitudes in column-major order.
    /// @param coord_sys The CoordinateSystem enum value specifying the coordinate system (e.g., SphericalDeg, Cartesian).
    StructuredGrid(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> center_lon, Kokkos::View<double *, MemorySpace> center_lat,
                   CoordinateSystem coord_sys);

    // ── Dimension queries ────────────────────────────────────────────────────

    /// @brief Gets the number of cells in the i-direction (fastest-varying dimension).
    /// @return The number of cells in the i-direction as a std::size_t.
    [[nodiscard]] std::size_t ni() const noexcept {
        return ni_;
    }

    /// @brief Gets the number of cells in the j-direction.
    /// @return The number of cells in the j-direction as a std::size_t.
    [[nodiscard]] std::size_t nj() const noexcept {
        return nj_;
    }

    /// @brief Gets the coordinate system of the grid's coordinates.
    /// @return The CoordinateSystem enum value representing the grid's coordinate system.
    [[nodiscard]] CoordinateSystem coord_system() const noexcept {
        return coord_sys_;
    }

    // ── Coordinate accessors (1-D flat arrays) ───────────────────────────────

    /// @brief Gets the center longitudes as a flat 1-D field_view.
    /// @return A non-owning rank-1 field_view of size [ni*nj].
    [[nodiscard]] field_view<const double, 1> center_lon() const noexcept;

    /// @brief Gets the center latitudes as a flat 1-D field_view.
    /// @return A non-owning rank-1 field_view of size [ni*nj].
    [[nodiscard]] field_view<const double, 1> center_lat() const noexcept;

    /// @brief Gets the corner longitudes as a flat 1-D field_view.
    ///
    /// If corner coordinates were not explicitly set via `set_corners()`, this returns
    /// an empty view of size 0.
    ///
    /// @return A non-owning rank-1 field_view of size [(ni+1)*(nj+1)] or empty.
    [[nodiscard]] field_view<const double, 1> corner_lon() const noexcept;

    /// @brief Gets the corner latitudes as a flat 1-D field_view.
    ///
    /// If corner coordinates were not explicitly set via `set_corners()`, this returns
    /// an empty view of size 0.
    ///
    /// @return A non-owning rank-1 field_view of size [(ni+1)*(nj+1)] or empty.
    [[nodiscard]] field_view<const double, 1> corner_lat() const noexcept;

    // ── Mutators ─────────────────────────────────────────────────────────────

    /// @brief Sets the vertex (corner) coordinates for conservative methods.
    ///
    /// This method moves the provided corner Views into the grid instance without copying.
    ///
    /// @param corner_lon A rank-1 Kokkos::View of size [(ni+1)*(nj+1)] containing corner longitudes.
    /// @param corner_lat A rank-1 Kokkos::View of size [(ni+1)*(nj+1)] containing corner latitudes.
    void set_corners(Kokkos::View<double *, MemorySpace> corner_lon, Kokkos::View<double *, MemorySpace> corner_lat);

    // ── Conversion ───────────────────────────────────────────────────────────

    /// @brief Converts the StructuredGrid into the common internal finite-element UnstructuredMesh format.
    ///
    /// Produces exactly `ni * nj` quadrilateral cells. Each logical cell (i, j) is mapped to a
    /// quadrilateral element with 4 corner nodes in the resulting unstructured mesh. If corner
    /// coordinates have not been explicitly provided via `set_corners()`, they are dynamically
    /// synthesized from the cell center coordinates using a midpoint interpolation scheme.
    ///
    /// This conversion is executed via a highly parallelized Kokkos kernel on the device or host
    /// associated with the template's `MemorySpace`.
    ///
    /// @return A complete UnstructuredMesh<MemorySpace> instance representing the same grid.
    [[nodiscard]] UnstructuredMesh<MemorySpace> to_unstructured() const;

   private:
    std::size_t ni_{0};
    std::size_t nj_{0};
    Kokkos::View<double *, MemorySpace> center_lon_;
    Kokkos::View<double *, MemorySpace> center_lat_;
    Kokkos::View<double *, MemorySpace> corner_lon_;
    Kokkos::View<double *, MemorySpace> corner_lat_;
    CoordinateSystem coord_sys_{CoordinateSystem::SphericalDeg};

    /// Internal: synthesize corner coordinates from centers when corners not
    /// explicitly provided.
    void synthesize_corners() const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Global-context (halo-aware) corner synthesis — the single source of truth for
// how a latitude band's Cell_Corners are derived.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Synthesize globally-consistent Cell_Corners for a latitude band.
///
/// A distributed driver partitions a destination grid into contiguous latitude
/// bands [j0, j1). A naive implementation builds a band `StructuredGrid` from a
/// band-local center slice and lets `to_unstructured()` synthesize corners from
/// it; that extrapolates the band's outer edges one-sidedly, which is exact only
/// on a uniform grid and wrong (and rank-seam-inconsistent) on non-uniform or
/// curvilinear grids.
///
/// This function is the halo-aware alternative: it runs the SAME corner-synthesis
/// kernel `StructuredGrid::synthesize_corners()` uses for a full grid — the 2×2
/// midpoint average of surrounding centers, with the identical periodic-longitude
/// wrap and one-sided boundary convention — over the FULL global center arrays,
/// and returns only the corner rows `[j0, j1]` (i.e. `j1 - j0 + 1` rows) that
/// bound the band. Because it is the same kernel on the same global centers, the
/// result is provably identical to rows `[j0, j1]` of the global mesh's
/// synthesized corners, so a band's boundary edges are globally consistent and a
/// single-rank (whole-grid) band is byte-for-byte the global mesh.
///
/// Works for both rectilinear and curvilinear grids: pass the full global center
/// arrays (rectilinear grids are expanded to full `ni*nj` centers by the caller).
///
/// @tparam MemorySpace Kokkos memory space of the input/output views.
/// @param ni         Number of cells in the i (longitude) direction (> 0).
/// @param nj_global  Total number of rows in the FULL global grid (>= j1).
/// @param center_lon Full global center longitudes, size ni*nj_global, column-major
///                   (index = i + j*ni). Periodicity is detected from these.
/// @param center_lat Full global center latitudes, size ni*nj_global.
/// @param j0         First destination row of the band (>= 0).
/// @param j1         One-past-last destination row of the band (j0 <= j1 <= nj_global).
/// @param corner_lon Output view, allocated here, size (ni+1)*(j1-j0+1); corner
///                   (ci, cj) is at index ci + (cj - j0)*(ni+1).
/// @param corner_lat Output view, allocated here, same layout.
///
/// @throws std::invalid_argument if ni == 0, j1 < j0, or j1 > nj_global.
template <class MemorySpace = Kokkos::HostSpace>
void synthesize_band_corners(std::size_t ni, std::size_t nj_global, Kokkos::View<double *, MemorySpace> center_lon,
                             Kokkos::View<double *, MemorySpace> center_lat, std::size_t j0, std::size_t j1,
                             Kokkos::View<double *, MemorySpace> &corner_lon, Kokkos::View<double *, MemorySpace> &corner_lat);

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_STRUCTURED_GRID_HPP
