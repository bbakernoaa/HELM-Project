// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP
#define AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP

/// @file axis/topology/unstructured_mesh.hpp
/// @brief The common internal finite-element unstructured mesh representation.
///
/// UnstructuredMesh is the single internal format that ALL source grids convert
/// into and that the solver operates on exclusively. It owns Kokkos Views in a
/// single memory space; all public accessors return field_view (non-owning
/// std::mdspan<layout_left>) so consumers never copy.
///
/// Templated on a Kokkos MemorySpace (HELM Law #2: explicit placement, no UVM).
/// Light inline accessors live in this header; the heavy compute_areas kernel
/// is implemented in the .cpp with explicit template instantiations.

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <axis/types.hpp>
#include <axis/topology/enums.hpp>

namespace axis::topology {

/// Finite-element unstructured mesh: nodes (vertices) + elements (cells) with
/// arbitrary mixed element types, stored CSR-style for cell→node connectivity.
///
/// @tparam MemorySpace Kokkos memory space for internal array storage
///         (Kokkos::HostSpace, CudaSpace, HIPSpace, etc.)
template <class MemorySpace = Kokkos::HostSpace>
class UnstructuredMesh {
public:
    using memory_space = MemorySpace;

    /// Default-construct an empty mesh.
    UnstructuredMesh() = default;

    /// Construct by adopting pre-built Kokkos arrays (moved in, no copy).
    ///
    /// @param node_coords    Node coordinates [n_nodes, ndim] (LayoutLeft)
    /// @param conn_offsets   CSR offsets [n_cells + 1]
    /// @param conn_indices   CSR node indices [nnz]
    /// @param coord_sys      Coordinate system for the node data
    /// @param areas          Optional precomputed cell areas [n_cells]
    /// @param mask           Optional cell mask [n_cells] (0=masked, 1=active)
    UnstructuredMesh(
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> node_coords,
        Kokkos::View<index_t*, MemorySpace>                     conn_offsets,
        Kokkos::View<index_t*, MemorySpace>                     conn_indices,
        CoordinateSystem                                        coord_sys,
        Kokkos::View<double*, MemorySpace>                      areas = {},
        Kokkos::View<int*, MemorySpace>                         mask  = {})
        : node_coords_(std::move(node_coords))
        , conn_offsets_(std::move(conn_offsets))
        , conn_indices_(std::move(conn_indices))
        , coord_sys_(coord_sys)
        , cell_areas_(std::move(areas))
        , cell_mask_(std::move(mask))
    {}

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Number of nodes (vertices) in the mesh.
    [[nodiscard]] std::size_t n_nodes() const noexcept {
        return node_coords_.extent(0);
    }

    /// Number of cells (elements) in the mesh.
    [[nodiscard]] std::size_t n_cells() const noexcept {
        // CSR offsets array has length n_cells + 1
        return conn_offsets_.extent(0) > 0
            ? conn_offsets_.extent(0) - 1
            : 0;
    }

    /// Coordinate system for this mesh's node data.
    [[nodiscard]] CoordinateSystem coord_system() const noexcept {
        return coord_sys_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Non-owning field_view accessors (zero-copy, layout_left)
    //
    // These return std::mdspan<const T, dextents, layout_left> directly
    // wrapping the internal Kokkos::View data pointers. No allocation or copy.
    // ─────────────────────────────────────────────────────────────────────────

    /// Node coordinates as a rank-2 view: [n_nodes, ndim].
    [[nodiscard]] field_view<const double, 2> node_coords() const noexcept {
        return field_view<const double, 2>{
            node_coords_.data(),
            node_coords_.extent(0),
            node_coords_.extent(1)};
    }

    /// CSR connectivity offsets: [n_cells + 1]. Node indices of cell c are
    /// conn_indices[conn_offsets[c] .. conn_offsets[c+1]).
    [[nodiscard]] field_view<const index_t, 1> conn_offsets() const noexcept {
        return field_view<const index_t, 1>{
            conn_offsets_.data(),
            conn_offsets_.extent(0)};
    }

    /// CSR connectivity indices (flattened node indices for all cells): [nnz].
    [[nodiscard]] field_view<const index_t, 1> conn_indices() const noexcept {
        return field_view<const index_t, 1>{
            conn_indices_.data(),
            conn_indices_.extent(0)};
    }

    /// Per-cell areas (computed via compute_areas()). Returns an empty view
    /// (extent 0) until compute_areas() has been called or areas were provided
    /// at construction.
    [[nodiscard]] field_view<const double, 1> cell_areas() const noexcept {
        return field_view<const double, 1>{
            cell_areas_.data(),
            cell_areas_.extent(0)};
    }

    /// Optional per-cell mask (0 = masked, 1 = active). Empty view if no mask
    /// was provided at construction.
    [[nodiscard]] field_view<const int, 1> cell_mask() const noexcept {
        return field_view<const int, 1>{
            cell_mask_.data(),
            cell_mask_.extent(0)};
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Kokkos::View accessors (for topology builders that need
    // direct View access, e.g., MeshFactory, GmshWriter, WeightGenerator)
    // ─────────────────────────────────────────────────────────────────────────

    [[nodiscard]] const auto& node_coords_view() const noexcept { return node_coords_; }
    [[nodiscard]] const auto& conn_offsets_view() const noexcept { return conn_offsets_; }
    [[nodiscard]] const auto& conn_indices_view() const noexcept { return conn_indices_; }
    [[nodiscard]] const auto& cell_areas_view() const noexcept { return cell_areas_; }
    [[nodiscard]] const auto& cell_mask_view() const noexcept { return cell_mask_; }

    // ─────────────────────────────────────────────────────────────────────────
    // Area computation (heavy kernel — implemented in .cpp)
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute cell areas via a Kokkos parallel kernel.
    ///
    /// For SphericalDeg/SphericalRad: spherical excess formula (Girard's theorem
    /// for triangles, generalized polygon via spherical excess summation).
    /// Areas are in steradians on the unit sphere.
    ///
    /// For Cartesian3D: standard planar polygon area via cross-product summation
    /// (shoelace formula generalized to 3D-embedded polygons).
    ///
    /// Results are stored internally and accessible via cell_areas().
    void compute_areas();

private:
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> node_coords_;
    Kokkos::View<index_t*, MemorySpace>                     conn_offsets_;
    Kokkos::View<index_t*, MemorySpace>                     conn_indices_;
    CoordinateSystem                                        coord_sys_{CoordinateSystem::SphericalDeg};
    Kokkos::View<double*, MemorySpace>                      cell_areas_;
    Kokkos::View<int*, MemorySpace>                         cell_mask_;
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP
