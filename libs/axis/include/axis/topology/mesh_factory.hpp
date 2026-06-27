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

/// @brief Static factory class that serves as the single entry funnel for all grids into AXIS.
///
/// MeshFactory provides a static interface (no instance state) to construct `UnstructuredMesh`
/// instances from various sources:
///   1. `from_descriptor`: The primary unified entry point for all file-backed grid sources described via `GridDescriptor`.
///   2. `from_named`: A shortcut entry point that resolves pre-defined or registered named grids by string tokens.
///   3. `from_rules`: A shortcut entry point that generates mesh topologies procedurally from rules-based parameters.
///
/// Because all methods are static, this class cannot be instantiated (the default constructor is deleted).
class MeshFactory {
public:
    /// @brief Deleted default constructor to prevent instantiation of this static factory class.
    MeshFactory() = delete;

    /// @brief Builds an UnstructuredMesh from a GridDescriptor.
    ///
    /// This is the single, unified entry point for all file-backed grid sources. It branches
    /// on `descriptor.kind` exactly once to dispatch to the appropriate convention builder, and
    /// never branches based on producer identity.
    ///
    /// Thorough validation of the descriptor is performed prior to construction (providing a strong
    /// all-or-nothing guarantee). If any validation check fails, an exception is thrown and no mesh
    /// is constructed.
    ///
    /// If the buffer views in the descriptor already address the target `MemorySpace`, they are adopted
    /// directly without performing any copy. Otherwise, an explicit deep copy using `Kokkos::deep_copy`
    /// is executed to place the data in the target memory space.
    ///
    /// @tparam MemorySpace The target Kokkos memory space for the resulting unstructured mesh. Defaults to Kokkos::HostSpace.
    /// @param descriptor The populated ingest::GridDescriptor instance from a file or grid producer.
    /// @return A complete UnstructuredMesh instance residing in the specified MemorySpace.
    /// @throws std::invalid_argument If the descriptor has an unknown kind, missing required fields, inconsistent buffer extents, or null/empty required buffers.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_descriptor(const ingest::GridDescriptor& descriptor);

    /// @brief Generates an UnstructuredMesh for a registered named grid token.
    ///
    /// This method is a shortcut helper that delegates directly to `NamedGridRegistry::generate`.
    /// Typical named grid tokens include standard meteorological grids such as "O1280", "F128", "N320", etc.
    ///
    /// @tparam MemorySpace The target Kokkos memory space for the resulting unstructured mesh. Defaults to Kokkos::HostSpace.
    /// @param name The named grid token string.
    /// @return A complete UnstructuredMesh instance residing in the specified MemorySpace.
    /// @throws std::invalid_argument If the provided name is unknown, malformed, or unregistered.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_named(const std::string& name);

    /// @brief Generates an UnstructuredMesh procedurally from a set of rule parameters.
    ///
    /// This method is a shortcut helper that delegates directly to `RuleGenerator::generate` to build
    /// structured/unstructured meshes based on analytical specifications (such as bounding boxes and resolutions).
    ///
    /// @tparam MemorySpace The target Kokkos memory space for the resulting unstructured mesh. Defaults to Kokkos::HostSpace.
    /// @param params The ingest::GridRulesParams structure specifying the procedural rules.
    /// @return A complete UnstructuredMesh instance residing in the specified MemorySpace.
    /// @throws std::invalid_argument If the parameters are inconsistent or invalid.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
    from_rules(const ingest::GridRulesParams& params);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_MESH_FACTORY_HPP
