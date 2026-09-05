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

    /// @brief Ensure corner coordinates exist, synthesizing them from centers if unset.
    ///
    /// Idempotent. After this call, `corner_lon()`/`corner_lat()` return the
    /// `[(ni+1)*(nj+1)]` shared-vertex arrays. Lets callers inspect or slice the
    /// synthesized corners without building the full unstructured mesh.
    void ensure_corners() const;

    /// @brief Convert a contiguous range of j-rows into an UnstructuredMesh.
    ///
    /// Emits `ni * nrows` quadrilateral cells for the global row range
    /// `[j0, j0 + nrows)`, referencing the globally-synthesized shared corners so
    /// band boundaries are bit-identical to those of adjacent bands. This enables
    /// seam-free MPI row-band decomposition of conservative regridding: build one
    /// global grid, then extract each rank's band. Corners are synthesized once on
    /// the full grid (see `ensure_corners()`), so periodicity and pole handling
    /// use global context rather than the truncated band.
    ///
    /// @param j0    First global j-row of the band (0-based).
    /// @param nrows Number of j-rows in the band; requires `j0 + nrows <= nj`.
    /// @return An UnstructuredMesh<MemorySpace> for the band's cells.
    [[nodiscard]] UnstructuredMesh<MemorySpace> to_unstructured_band(std::size_t j0, std::size_t nrows) const;

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

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_STRUCTURED_GRID_HPP
