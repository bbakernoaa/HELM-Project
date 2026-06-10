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

#include <cstddef>

#include <axis/types.hpp>
#include <axis/ingest/grid_descriptor.hpp> // CoordinateSystem

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations for types used in the free-function signatures.
// These live in axis::topology and axis::solver and will be defined in their
// respective headers (unstructured_mesh.hpp, interpolation_matrix.hpp).
// ─────────────────────────────────────────────────────────────────────────────

namespace axis::topology {
template <class MemorySpace>
class UnstructuredMesh;
} // namespace axis::topology

namespace axis::solver {
template <class MemorySpace>
class InterpolationMatrix;
} // namespace axis::solver

namespace axis::ingest {

// ─────────────────────────────────────────────────────────────────────────────
// WeightEgress — plain-data view over an InterpolationMatrix's weight buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-data, non-owning view over an InterpolationMatrix's weight buffers,
/// laid out exactly as ESMF / SCRIP weight files express them. A consumer
/// (AMIO, Python) serializes these to a SCRIP/ESMF weight file. AXIS owns the
/// memory; this struct only views it. Mirror of GridDescriptor for output.
struct WeightEgress {
    field_view<const double, 1>  factor_list{};   ///< S(n_s) — interpolation weights
    field_view<const index_t, 1> factor_col{};    ///< col(n_s) — source index per nonzero
    field_view<const index_t, 1> factor_row{};    ///< row(n_s) — destination index per nonzero
    field_view<const double, 1>  frac_a{};        ///< source fractions [n_a]
    field_view<const double, 1>  frac_b{};        ///< destination fractions [n_b]
    field_view<const double, 1>  area_a{};        ///< source cell areas [n_a]
    field_view<const double, 1>  area_b{};        ///< destination cell areas [n_b]
    std::size_t n_s{0};                           ///< number of nonzero weights
    std::size_t n_a{0};                           ///< number of source cells
    std::size_t n_b{0};                           ///< number of destination cells
};

// ─────────────────────────────────────────────────────────────────────────────
// MeshEgress — plain-data view over a mesh's nodes + CSR connectivity
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-data, non-owning view over a mesh's node coordinates + CSR connectivity
/// for a consumer to serialize (e.g. to NetCDF/UGRID or a SCRIP grid file). The
/// structural mirror of GridDescriptor::BufferViews on the output side.
struct MeshEgress {
    field_view<const double, 2>  node_coords{};   ///< [n_nodes, ndim]
    field_view<const index_t, 1> conn_offsets{};  ///< CSR offsets, length n_cells + 1
    field_view<const index_t, 1> conn_indices{};  ///< CSR node indices
    field_view<const double, 1>  cell_areas{};    ///< optional per-cell areas
    field_view<const int, 1>     cell_mask{};     ///< optional 0/1 mask
    CoordinateSystem             coord_system{CoordinateSystem::SphericalDeg};
    std::size_t n_nodes{0};                       ///< number of mesh nodes
    std::size_t n_cells{0};                       ///< number of mesh cells
};

// ─────────────────────────────────────────────────────────────────────────────
// Free functions — build egress views without copying
// ─────────────────────────────────────────────────────────────────────────────

/// Build a WeightEgress view over an InterpolationMatrix (no copy; host-resident
/// matrices view in place, device-resident matrices are mirrored to host first
/// by the caller via explicit deep_copy — never UVM).
template <class MemorySpace>
[[nodiscard]] inline WeightEgress
weight_egress(const solver::InterpolationMatrix<MemorySpace>& m) {
    WeightEgress eg;
    eg.factor_list = m.factor_list();
    eg.factor_col  = m.factor_col();
    eg.factor_row  = m.factor_row();
    eg.frac_a      = m.frac_a();
    eg.frac_b      = m.frac_b();
    eg.area_a      = m.area_a();
    eg.area_b      = m.area_b();
    eg.n_s         = m.nnz();
    eg.n_a         = m.n_src();
    eg.n_b         = m.n_dst();
    return eg;
}

/// Build a MeshEgress view over an UnstructuredMesh (same host/device note).
template <class MemorySpace>
[[nodiscard]] inline MeshEgress
mesh_egress(const topology::UnstructuredMesh<MemorySpace>& mesh) {
    MeshEgress eg;
    eg.node_coords  = mesh.node_coords();
    eg.conn_offsets = mesh.cell_node_offsets();
    eg.conn_indices = mesh.cell_node_indices();
    eg.cell_areas   = mesh.cell_areas();
    eg.cell_mask    = mesh.cell_mask();
    eg.coord_system = mesh.coord_system();
    eg.n_nodes      = mesh.num_nodes();
    eg.n_cells      = mesh.num_cells();
    return eg;
}

} // namespace axis::ingest

#endif // AXIS_INGEST_EGRESS_HPP
