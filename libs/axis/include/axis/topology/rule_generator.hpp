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

/// @class RuleGenerator
/// @brief Rule-based grid generator from abstract mathematical rules.
///
/// Generates a mesh in a deterministic manner using Kokkos parallel kernels, fully derived from
/// GridRules configurations (such as grid kind, bounding boxes, resolutions, and Gaussian number N).
/// It performs generation with zero file interactions or third-party file format parsing.
///
/// Validation rules:
///   - If the grid kind is unrecognized, an exception is thrown.
///   - If the grid resolutions (r_x, r_y) are non-positive, an exception is thrown.
///   - If the bounding box maximum coordinate is less than its minimum coordinate, an exception is thrown.
///   - If the Gaussian grid parameter gaussian_n is non-positive for Gaussian-style grids, an exception is thrown.
class RuleGenerator {
public:
    /// @brief Generate an UnstructuredMesh from specified grid rules parameters.
    ///
    /// Constructs a fully populated UnstructuredMesh structure entirely within the target memory space
    /// based on mathematical rule formulations.
    ///
    /// @tparam MemorySpace The Kokkos memory space in which the resulting mesh's data views should be allocated. Defaults to Kokkos::HostSpace.
    /// @param rules The ingest::GridRulesParams structure containing parameters for mesh generation (e.g., kind, bbox, resolution, and gaussian_n).
    /// @return UnstructuredMesh<MemorySpace> A completed UnstructuredMesh object located in the specified MemorySpace.
    /// @throw std::invalid_argument If the grid rule kind is unrecognized, if coordinate bounding box constraints
    ///                              are violated (max < min), if resolution spacing is non-positive, or if the Gaussian N parameter
    ///                              is invalid (<= 0) for Gaussian-based grids.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace>
        generate(const ingest::GridRulesParams& rules);
};

} // namespace axis::topology

#endif // AXIS_TOPOLOGY_RULE_GENERATOR_HPP
