// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_INGEST_EGRESS_HPP
#define AXIS_INGEST_EGRESS_HPP

/// @file axis/ingest/egress.hpp
/// @brief The public plain-data egress contract for AXIS.
///
/// WeightEgress and MeshEgress are the symmetric output counterparts of
/// GridDescriptor. AXIS exposes its weight and mesh results as non-owning
/// layout_left field_view members; a consumer (AMIO today, Python tomorrow)
/// serializes them to SCRIP/ESMF weight files, NetCDF, Zarr, etc.
///
/// The egress types own no resources, name no Kokkos/AMIO/eckit type, and
/// require no DAGR to reach a consumer. Header-only: no associated .cpp
/// compilation unit.

#include <axis/ingest/grid_descriptor.hpp>  // CoordinateSystem
#include <axis/types.hpp>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations for types used in the free-function signatures.
// These live in axis::topology and axis::solver and will be defined in their
// respective headers (unstructured_mesh.hpp, interpolation_matrix.hpp).
// ─────────────────────────────────────────────────────────────────────────────

namespace axis::topology {
template <class MemorySpace>
class UnstructuredMesh;
}  // namespace axis::topology

namespace axis::solver {
template <class MemorySpace>
class InterpolationMatrix;
}  // namespace axis::solver

namespace axis::ingest {

// ─────────────────────────────────────────────────────────────────────────────
// WeightEgress — plain-data view over an InterpolationMatrix's weight buffers
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Plain-data, non-owning view over an InterpolationMatrix's weight buffers.
///
/// Laid out exactly as ESMF / SCRIP weight files express them. A consumer
/// (such as AMIO or a Python wrapper) serializes these to a SCRIP/ESMF weight file.
/// AXIS owns the memory; this struct only views it. Mirror of GridDescriptor for output.
struct WeightEgress {
    /// @brief Interpolation weights list (often labeled 'S' in SCRIP), size [n_s].
    field_view<const double, 1> factor_list{};

    /// @brief Source cell index per nonzero weight factor (0-based, column indices), size [n_s].
    field_view<const index_t, 1> factor_col{};

    /// @brief Destination cell index per nonzero weight factor (0-based, row indices), size [n_s].
    field_view<const index_t, 1> factor_row{};

    /// @brief Source cell fractions of area active in interpolation, size [n_a].
    field_view<const double, 1> frac_a{};

    /// @brief Destination cell fractions of area active in interpolation, size [n_b].
    field_view<const double, 1> frac_b{};

    /// @brief Source cell areas, size [n_a].
    field_view<const double, 1> area_a{};

    /// @brief Destination cell areas, size [n_b].
    field_view<const double, 1> area_b{};

    /// @brief Number of nonzero interpolation weight entries (size of factor_list, factor_col, factor_row).
    std::size_t n_s{0};

    /// @brief Number of cells in the source grid.
    std::size_t n_a{0};

    /// @brief Number of cells in the destination grid.
    std::size_t n_b{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// MeshEgress — plain-data view over a mesh's nodes + CSR connectivity
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Plain-data, non-owning view over a mesh's node coordinates + CSR connectivity.
///
/// Used for a consumer to serialize a mesh to NetCDF/UGRID or a SCRIP grid file.
/// This functions as the structural mirror of GridDescriptor::BufferViews on the egress side.
struct MeshEgress {
    /// @brief Node coordinates of shape [n_nodes, ndim] where ndim is the coordinate system dimensionality.
    field_view<const double, 2> node_coords{};

    /// @brief CSR offsets of size [n_cells + 1], pointing to the start of each cell's nodes in conn_indices.
    field_view<const index_t, 1> conn_offsets{};

    /// @brief CSR indices mapping cells to their constituent nodes.
    field_view<const index_t, 1> conn_indices{};

    /// @brief Optional precomputed areas per cell, size [n_cells].
    field_view<const double, 1> cell_areas{};

    /// @brief Optional cell active mask: 0 for masked/inactive, 1 for active, size [n_cells].
    field_view<const int, 1> cell_mask{};

    /// @brief Coordinate system of the node coordinates (e.g. spherical degrees, radians, or 3-D Cartesian).
    CoordinateSystem coord_system{CoordinateSystem::SphericalDeg};

    /// @brief Number of nodes in the mesh.
    std::size_t n_nodes{0};

    /// @brief Number of cells in the mesh.
    std::size_t n_cells{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// Free functions — build egress views without copying
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Build a WeightEgress view over an InterpolationMatrix.
///
/// This operation is zero-copy. Host-resident matrices are viewed directly in place.
/// Device-resident matrices must be mirrored to the host space first by the caller
/// via an explicit deep copy (never Unified Virtual Memory / UVM).
///
/// @tparam MemorySpace The Kokkos memory space of the underlying InterpolationMatrix.
/// @param m The InterpolationMatrix to view.
/// @return A @c WeightEgress struct viewing the underlying matrix data.
template <class MemorySpace>
[[nodiscard]] inline WeightEgress weight_egress(const solver::InterpolationMatrix<MemorySpace> &m) {
    WeightEgress eg;
    eg.factor_list = m.factor_list();
    eg.factor_col = m.factor_col();
    eg.factor_row = m.factor_row();
    eg.frac_a = m.frac_a();
    eg.frac_b = m.frac_b();
    eg.area_a = m.area_a();
    eg.area_b = m.area_b();
    eg.n_s = m.nnz();
    eg.n_a = m.n_src();
    eg.n_b = m.n_dst();
    return eg;
}

/// @brief Build a MeshEgress view over an UnstructuredMesh.
///
/// This operation is zero-copy. Host-resident meshes are viewed directly in place.
/// Device-resident meshes must be mirrored to the host space first by the caller
/// via an explicit deep copy (never UVM).
///
/// @tparam MemorySpace The Kokkos memory space of the underlying UnstructuredMesh.
/// @param mesh The UnstructuredMesh to view.
/// @return A @c MeshEgress struct viewing the underlying mesh topology and coordinates.
template <class MemorySpace>
[[nodiscard]] inline MeshEgress mesh_egress(const topology::UnstructuredMesh<MemorySpace> &mesh) {
    MeshEgress eg;
    eg.node_coords = mesh.node_coords();
    eg.conn_offsets = mesh.cell_node_offsets();
    eg.conn_indices = mesh.cell_node_indices();
    eg.cell_areas = mesh.cell_areas();
    eg.cell_mask = mesh.cell_mask();
    eg.coord_system = mesh.coord_system();
    eg.n_nodes = mesh.num_nodes();
    eg.n_cells = mesh.num_cells();
    return eg;
}

}  // namespace axis::ingest

#endif  // AXIS_INGEST_EGRESS_HPP
