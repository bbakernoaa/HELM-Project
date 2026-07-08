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

#include <axis/topology/unstructured_mesh.hpp>
#include <string>
#include <vector>

namespace axis::topology {

/// @class NamedGridRegistry
/// @brief Registry of hardcoded mathematical generators for standard global weather
///        grids. Purely static interface — no instance state.
class NamedGridRegistry {
   public:
    /// @struct ParsedName
    /// @brief Parsed grid name containing family and grid number.
    ///
    /// For example, "O1280" parses into family 'O' and number 1280.
    struct ParsedName {
        char family;  ///< Family prefix: 'O', 'F', 'N', or 'R' (standard weather grid families).
        int number;   ///< Grid number (a positive integer, typically representing the Gaussian number N).
    };

    /// @brief Parse and validate a named-grid string.
    ///
    /// @param name The std::string representing the grid name to parse (e.g., "O1280", "F128", "N320").
    /// @return NamedGridRegistry::ParsedName A structure containing the parsed family character and grid number.
    /// @throw std::invalid_argument If the family prefix is unknown, the grid number is non-positive,
    ///                              or the string is otherwise malformed.
    [[nodiscard]] static ParsedName parse(const std::string &name);

    /// @brief Check whether a name string corresponds to a registered grid generator.
    ///
    /// This method returns false and does not throw if the grid name is malformed or if
    /// the grid family prefix is unregistered.
    ///
    /// @param name The std::string representing the grid name to check.
    /// @return bool True if a grid generator is registered for the specified name, false otherwise.
    [[nodiscard]] static bool is_registered(const std::string &name) noexcept;

    /// @brief Generate the named grid as an UnstructuredMesh in the specified MemorySpace.
    ///
    /// The mesh is built entirely via Kokkos parallel kernels with zero file
    /// I/O. Generation is deterministic: two calls with the same name produce
    /// bitwise-identical results.
    ///
    /// @tparam MemorySpace The Kokkos memory space in which the generated mesh should reside. Defaults to Kokkos::HostSpace.
    /// @param name The std::string representing the valid grid name string (e.g., "O1280").
    /// @return UnstructuredMesh<MemorySpace> The generated UnstructuredMesh consisting of quadrilateral cells.
    /// @throw std::invalid_argument If the grid name is unknown or malformed.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace> generate(const std::string &name);

    /// @brief Enumerate all registered family prefixes.
    ///
    /// @return std::vector<char> A sorted vector of registered grid family characters (currently {'F', 'G', 'N', 'O', 'R'}).
    [[nodiscard]] static std::vector<char> registered_families();
};

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP
