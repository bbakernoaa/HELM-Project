// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_RULE_GENERATOR_HPP
#define AXIS_TOPOLOGY_RULE_GENERATOR_HPP

/// @file axis/topology/rule_generator.hpp
/// @brief Rule-based mesh generator from GridRulesParams.
///
/// Builds a mesh purely from the GridRules parameters in a descriptor (kind,
/// bbox, resolution, gaussian_n) using Kokkos parallel kernels — NO file
/// enumeration and NO YAML/JSON parsing. A GridRules YAML FILE is parsed by the
/// PRODUCER (AMIO via eckit, or Python) into GridRulesParams; AXIS only
/// generates the mesh from those fields.
///
/// Supported rule kinds:
///   - RegularLatLon:     regular lat-lon grid from bbox + resolution
///   - GaussianRegular:   regular Gaussian grid from gaussian_n
///   - GaussianReduced:   reduced (octahedral) Gaussian grid from gaussian_n
///   - Projected:         regular grid in projection space, transformed via PROJ
///
/// Invoked by MeshFactory::from_descriptor for ConventionKind::GridRules.

#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/unstructured_mesh.hpp>

namespace axis::topology {

/// Rule-based generator. Builds a mesh purely from the GridRules parameters in a
/// descriptor (kind, bbox, resolution, gaussian_n) using a Kokkos parallel
/// kernel — NO file enumeration and NO YAML/JSON parsing.
///
/// Validation:
///   - If kind is unrecognized → throw std::invalid_argument
///   - If resolution r_x or r_y <= 0 → throw std::invalid_argument
///   - If bbox max < min → throw std::invalid_argument
///   - If gaussian_n <= 0 for Gaussian kinds → throw std::invalid_argument
class RuleGenerator {
public:
    /// Generate an UnstructuredMesh from rule parameters.
    ///
    /// @tparam MemorySpace  Kokkos memory space for the resulting mesh
    /// @param rules  GridRulesParams specifying kind, bbox, resolution, etc.
    /// @return A complete UnstructuredMesh in the target MemorySpace
    /// @throws std::invalid_argument on invalid/inconsistent parameters
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
        generate(const ingest::GridRulesParams& rules);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_RULE_GENERATOR_HPP
