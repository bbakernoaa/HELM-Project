// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_MESH_FACTORY_HPP
#define AXIS_TOPOLOGY_MESH_FACTORY_HPP

/// @file axis/topology/mesh_factory.hpp
/// @brief MeshFactory — the single funnel through which all grids enter AXIS.
///
/// MeshFactory is a static interface (no instance state) providing three entry
/// points:
///
///   from_descriptor<MemorySpace>(const GridDescriptor&)
///     THE single funnel for all file-backed grid sources. Branches on
///     descriptor.kind ONCE (switch on ConventionKind). NEVER branches on
///     producer identity. Validates the descriptor thoroughly before building:
///       * Unknown kind → throw std::invalid_argument naming the kind
///       * Missing required field → throw std::invalid_argument naming the field
///       * Inconsistent buffer extents → throw std::invalid_argument describing mismatch
///       * Null/empty required buffer → throw std::invalid_argument naming the buffer
///     When buffer views already address the target MemorySpace: adopt without copy.
///     When in a different space: explicit Kokkos::deep_copy.
///
///   from_named<MemorySpace>(const std::string& name)
///     Shortcut: delegates to NamedGridRegistry::generate.
///
///   from_rules<MemorySpace>(const GridRulesParams& params)
///     Shortcut: delegates to RuleGenerator::generate.
///
/// This is the replacement for ESMF Grid/Mesh/LocStream factory functions.
/// No producer-specific branching ever appears — only ConventionKind dispatch.

#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/unstructured_mesh.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace axis::topology {

/// Static interface factory for building UnstructuredMesh from descriptors,
/// named-grid tokens, or rule parameters.
///
/// All methods are static — MeshFactory has no instance state.
class MeshFactory {
public:
    MeshFactory() = delete;

    /// Build an UnstructuredMesh from a GridDescriptor — THE single entry point
    /// for all file-backed grid sources.
    ///
    /// Branches on descriptor.kind ONCE. NEVER branches on producer identity.
    /// Validates thoroughly before building (strong guarantee: all-or-nothing).
    ///
    /// @tparam MemorySpace  Target Kokkos memory space for the resulting mesh.
    /// @param descriptor    The populated GridDescriptor from a producer.
    /// @return A complete UnstructuredMesh in the target MemorySpace.
    ///
    /// @throws std::invalid_argument on unknown kind, missing field,
    ///         inconsistent extents, or null/empty required buffers.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_descriptor(const ingest::GridDescriptor& descriptor);

    /// Shortcut: generate a named grid by token. Delegates directly to
    /// NamedGridRegistry::generate.
    ///
    /// @tparam MemorySpace  Target Kokkos memory space.
    /// @param name          Grid name string (e.g. "O1280", "F128", "N320").
    /// @return A complete UnstructuredMesh.
    /// @throws std::invalid_argument if the name is unknown/malformed.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_named(const std::string& name);

    /// Shortcut: generate a mesh from rule parameters. Delegates directly to
    /// RuleGenerator::generate.
    ///
    /// @tparam MemorySpace  Target Kokkos memory space.
    /// @param params        GridRulesParams specifying kind, bbox, resolution, etc.
    /// @return A complete UnstructuredMesh.
    /// @throws std::invalid_argument on invalid/inconsistent parameters.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_rules(const ingest::GridRulesParams& params);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_MESH_FACTORY_HPP
