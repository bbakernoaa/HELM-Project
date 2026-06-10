// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/gmsh_writer.cpp
/// @brief GmshWriter implementation — Gmsh .msh v2.2 ASCII serialization.
///
/// Implements device→host mirroring via Kokkos::deep_copy and file I/O via
/// the RAII File_Handle wrapper. Element types are mapped from per-cell node
/// count: 3→triangle (type 2), 4→quadrilateral (type 3).

#include <axis/topology/gmsh_writer.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include <axis/detail/memory_traits.hpp>
#include <axis/detail/raii_handles.hpp>
#include <axis/types.hpp>

namespace axis::topology {

namespace {

/// Map number of nodes per cell to Gmsh MSH v2.2 element type.
/// Returns 0 if the element cannot be represented.
inline int gmsh_element_type(int n_nodes_per_cell) noexcept {
    switch (n_nodes_per_cell) {
        case 3: return 2;  // 3-node triangle
        case 4: return 3;  // 4-node quadrilateral
        default: return 0; // not representable in MSH v2.2
    }
}

} // anonymous namespace

template <class MemorySpace>
void GmshWriter::write(const std::string& filepath,
                       const UnstructuredMesh<MemorySpace>& mesh)
{
    // ─────────────────────────────────────────────────────────────────────────
    // Step 1: Mirror device arrays to host if needed (HELM Law #2: explicit
    // Kokkos::deep_copy, no UVM reliance).
    // ─────────────────────────────────────────────────────────────────────────

    using host_space = Kokkos::HostSpace;

    // Host-accessible views for node_coords, conn_offsets, conn_indices.
    Kokkos::View<double**, Kokkos::LayoutLeft, host_space> h_coords;
    Kokkos::View<index_t*, host_space> h_offsets;
    Kokkos::View<index_t*, host_space> h_indices;

    if constexpr (detail::is_device_space_v<MemorySpace>) {
        // Device mesh: allocate host mirrors and deep_copy.
        const auto& d_coords  = mesh.node_coords_view();
        const auto& d_offsets = mesh.conn_offsets_view();
        const auto& d_indices = mesh.conn_indices_view();

        h_coords  = Kokkos::View<double**, Kokkos::LayoutLeft, host_space>(
            "gmsh_h_coords", d_coords.extent(0), d_coords.extent(1));
        h_offsets = Kokkos::View<index_t*, host_space>(
            "gmsh_h_offsets", d_offsets.extent(0));
        h_indices = Kokkos::View<index_t*, host_space>(
            "gmsh_h_indices", d_indices.extent(0));

        Kokkos::deep_copy(h_coords, d_coords);
        Kokkos::deep_copy(h_offsets, d_offsets);
        Kokkos::deep_copy(h_indices, d_indices);
    } else {
        // Host mesh: directly reference internal views (no copy).
        h_coords  = mesh.node_coords_view();
        h_offsets = mesh.conn_offsets_view();
        h_indices = mesh.conn_indices_view();
    }

    const std::size_t n_nodes = h_coords.extent(0);
    const std::size_t ndim    = h_coords.extent(1);
    const std::size_t n_cells = h_offsets.extent(0) > 0
                                    ? h_offsets.extent(0) - 1
                                    : 0;

    // ─────────────────────────────────────────────────────────────────────────
    // Step 2: Open file via File_Handle RAII (throws on failure).
    // ─────────────────────────────────────────────────────────────────────────

    detail::File_Handle fh(filepath.c_str(), "w");
    std::FILE* fp = fh.get();

    // ─────────────────────────────────────────────────────────────────────────
    // Step 3: $MeshFormat section
    // ─────────────────────────────────────────────────────────────────────────

    std::fprintf(fp, "$MeshFormat\n");
    std::fprintf(fp, "2.2 0 8\n");
    std::fprintf(fp, "$EndMeshFormat\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Step 4: $Nodes section
    // Node IDs are 1-based in Gmsh. Coordinates are x y z (z=0 for 2D).
    // ─────────────────────────────────────────────────────────────────────────

    std::fprintf(fp, "$Nodes\n");
    std::fprintf(fp, "%zu\n", n_nodes);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        const double x = h_coords(i, 0);
        const double y = (ndim > 1) ? h_coords(i, 1) : 0.0;
        const double z = (ndim > 2) ? h_coords(i, 2) : 0.0;
        // Node ID is 1-based
        std::fprintf(fp, "%zu %.17g %.17g %.17g\n", i + 1, x, y, z);
    }

    std::fprintf(fp, "$EndNodes\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Step 5: $Elements section
    // First pass: count writable elements (those with a valid Gmsh type).
    // Second pass: write them.
    // ─────────────────────────────────────────────────────────────────────────

    // Count elements that can be represented in MSH v2.2.
    std::size_t n_writable = 0;
    for (std::size_t c = 0; c < n_cells; ++c) {
        const auto start = static_cast<std::size_t>(h_offsets(c));
        const auto end   = static_cast<std::size_t>(h_offsets(c + 1));
        const int  n_cell_nodes = static_cast<int>(end - start);
        if (gmsh_element_type(n_cell_nodes) != 0) {
            ++n_writable;
        }
    }

    std::fprintf(fp, "$Elements\n");
    std::fprintf(fp, "%zu\n", n_writable);

    std::size_t elm_id = 1; // 1-based element numbering
    for (std::size_t c = 0; c < n_cells; ++c) {
        const auto start = static_cast<std::size_t>(h_offsets(c));
        const auto end   = static_cast<std::size_t>(h_offsets(c + 1));
        const int  n_cell_nodes = static_cast<int>(end - start);
        const int  elm_type = gmsh_element_type(n_cell_nodes);

        if (elm_type == 0) {
            continue; // Skip elements not representable in MSH v2.2
        }

        // Format: elm-number elm-type number-of-tags <tags> node-list
        // Use 1 tag with value 0 (physical entity = 0).
        std::fprintf(fp, "%zu %d 1 0", elm_id, elm_type);

        // Write node indices (convert from AXIS 0-based to Gmsh 1-based).
        for (auto j = start; j < end; ++j) {
            const auto node_idx = static_cast<std::size_t>(h_indices(j));
            std::fprintf(fp, " %zu", node_idx + 1);
        }
        std::fprintf(fp, "\n");
        ++elm_id;
    }

    std::fprintf(fp, "$EndElements\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Step 6: File is closed by File_Handle destructor (RAII).
    // ─────────────────────────────────────────────────────────────────────────
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template void GmshWriter::write<Kokkos::HostSpace>(
    const std::string&,
    const UnstructuredMesh<Kokkos::HostSpace>&);

#ifdef KOKKOS_ENABLE_CUDA
template void GmshWriter::write<Kokkos::CudaSpace>(
    const std::string&,
    const UnstructuredMesh<Kokkos::CudaSpace>&);
#endif

#ifdef KOKKOS_ENABLE_HIP
template void GmshWriter::write<Kokkos::HIPSpace>(
    const std::string&,
    const UnstructuredMesh<Kokkos::HIPSpace>&);
#endif

} // namespace axis::topology
