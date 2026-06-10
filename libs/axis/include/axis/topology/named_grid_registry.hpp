// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP
#define AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP

/// @file axis/topology/named_grid_registry.hpp
/// @brief NamedGridRegistry — on-the-fly generation of standard global weather
///        grids (O-octahedral, F-regular, N-reduced Gaussian families) via
///        Kokkos parallel kernels with zero file I/O.
///
/// Generates standard ECMWF-style Gaussian grids by name string (e.g.
/// "O1280", "F128", "N320"). Each family has a registered generator that
/// builds an UnstructuredMesh entirely in memory using Kokkos parallel
/// kernels — no file access, no third-party parsing.
///
/// Grid families:
///   O — Octahedral reduced Gaussian (ECMWF convention): 2N latitude circles
///       total (N per hemisphere). Latitude circle j (0-indexed from nearest
///       pole) has 20 + 4*j points. Total points = 4*N*(N+9).
///   F — Regular (full) Gaussian: 2N latitude circles, each with 4N points.
///       Total points = 2N * 4N = 8N^2.
///   N — Reduced Gaussian: same as O (octahedral pattern, ECMWF convention).
///
/// All generation is deterministic: two calls with the same name produce
/// bitwise-identical results (Requirement 6.5).

#include <string>
#include <vector>

#include <axis/topology/unstructured_mesh.hpp>

namespace axis::topology {

/// Registry of hardcoded mathematical generators for standard global weather
/// grids. Purely static interface — no instance state.
class NamedGridRegistry {
public:
    /// Parsed grid name, e.g. "O1280" → {family='O', number=1280}.
    struct ParsedName {
        char family;   ///< Family prefix: 'O', 'F', or 'N'
        int  number;   ///< Grid number (positive integer, e.g. Gaussian N)
    };

    /// Parse and validate a named-grid string.
    ///
    /// @param name  Grid name string (e.g. "O1280", "F128", "N320")
    /// @return ParsedName with family character and number
    /// @throws std::invalid_argument if family is unknown or number is
    ///         non-positive or the string is otherwise malformed
    [[nodiscard]] static ParsedName parse(const std::string& name);

    /// Check whether a name string corresponds to a registered grid generator.
    /// Returns false (never throws) if the name is malformed or the family
    /// is not registered.
    [[nodiscard]] static bool is_registered(const std::string& name) noexcept;

    /// Generate the named grid as an UnstructuredMesh in the given MemorySpace.
    ///
    /// The mesh is built entirely via Kokkos parallel kernels with zero file
    /// I/O. Generation is deterministic: two calls with the same name produce
    /// bitwise-identical results.
    ///
    /// @tparam MemorySpace  Kokkos memory space for the generated mesh
    /// @param name  Valid grid name string (e.g. "O1280")
    /// @return UnstructuredMesh with quadrilateral cells
    /// @throws std::invalid_argument if the name is unknown/malformed
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace> generate(const std::string& name);

    /// Enumerate all registered family prefixes.
    /// @return Vector of family characters (currently {'F', 'N', 'O'} sorted)
    [[nodiscard]] static std::vector<char> registered_families();
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP
