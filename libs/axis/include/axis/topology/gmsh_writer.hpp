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

/// @class GmshWriter
/// @brief Native exporter for serializing an UnstructuredMesh to Gmsh .msh v2.2 ASCII format.
///
/// Provides a clean static function interface for exporting unstructured mesh data.
/// Usage:
/// @code
///   GmshWriter::write("output.msh", mesh);
/// @endcode
class GmshWriter {
public:
    /// @brief Serialize an UnstructuredMesh to Gmsh .msh v2.2 ASCII format at the specified file path.
    ///
    /// If the mesh resides in device memory (e.g., CudaSpace or HIPSpace), internal arrays
    /// are automatically mirrored to the host memory space via explicit Kokkos::deep_copy prior to serialization.
    /// File streams are managed safely using a detail::File_Handle RAII container to guarantee files
    /// are closed properly on all function exits, including exception unwinding.
    ///
    /// Element types are determined from per-cell node count as follows:
    ///   - Cells with 3 nodes are exported as Gmsh element type 2 (3-node triangle).
    ///   - Cells with 4 nodes are exported as Gmsh element type 3 (4-node quadrilateral).
    ///   - Cells with other node counts are skipped, as they are not natively representable in standard MSH v2.2.
    ///
    /// Node indexing is converted from AXIS 0-based indexing to Gmsh 1-based indexing during serialization.
    ///
    /// @tparam MemorySpace The Kokkos memory space of the mesh being serialized.
    /// @param filepath The std::string representing the path to the output .msh file on the file system.
    /// @param mesh The UnstructuredMesh<MemorySpace> instance to serialize.
    /// @throw std::runtime_error If the target output file cannot be opened, or if a write error occurs during serialization.
    template <class MemorySpace>
    static void write(const std::string& filepath,
                      const UnstructuredMesh<MemorySpace>& mesh);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_GMSH_WRITER_HPP
