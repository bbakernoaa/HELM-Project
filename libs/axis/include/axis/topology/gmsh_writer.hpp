// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_GMSH_WRITER_HPP
#define AXIS_TOPOLOGY_GMSH_WRITER_HPP

/// @file axis/topology/gmsh_writer.hpp
/// @brief Native Gmsh .msh v2.2 ASCII exporter for UnstructuredMesh.
///
/// GmshWriter is AXIS's only native file writer. It serializes an
/// UnstructuredMesh to the Gmsh .msh v2.2 ASCII format using plain stdio
/// (std::FILE* via detail::File_Handle) with no third-party library dependency.
///
/// All other output formats (SCRIP, ESMF weight files, NetCDF, Zarr) are
/// produced by a consumer (AMIO or Python) reading the egress views.
///
/// When the mesh resides in device memory, GmshWriter mirrors node/connectivity
/// arrays to host via explicit Kokkos::deep_copy before serialization
/// (HELM Law #2 — no UVM reliance).

#include <string>

#include <axis/topology/unstructured_mesh.hpp>

namespace axis::topology {

/// @brief Exports an UnstructuredMesh to Gmsh .msh v2.2 ASCII format.
///
/// Usage:
/// @code
///   GmshWriter::write("output.msh", mesh);
/// @endcode
///
/// @throws std::runtime_error if the file cannot be opened or a write error occurs.
class GmshWriter {
public:
    /// Serialize @p mesh to Gmsh .msh v2.2 ASCII at @p filepath.
    ///
    /// If the mesh resides in device memory (CudaSpace, HIPSpace, etc.),
    /// internal arrays are mirrored to host via Kokkos::deep_copy before
    /// writing. The file is managed via detail::File_Handle RAII — guaranteed
    /// to be closed on all exit paths including exceptions.
    ///
    /// Element types are determined from per-cell node count:
    ///   - 3 nodes → Gmsh type 2 (3-node triangle)
    ///   - 4 nodes → Gmsh type 3 (4-node quadrilateral)
    ///   - Other   → skipped (not representable in MSH v2.2)
    ///
    /// Node indices are converted from AXIS 0-based to Gmsh 1-based.
    ///
    /// @param filepath  Path to the output .msh file.
    /// @param mesh      The mesh to serialize.
    /// @throws std::runtime_error if fopen fails or a write error occurs.
    template <class MemorySpace>
    static void write(const std::string& filepath,
                      const UnstructuredMesh<MemorySpace>& mesh);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_GMSH_WRITER_HPP
