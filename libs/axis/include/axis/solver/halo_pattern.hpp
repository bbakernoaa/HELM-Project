// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_HALO_PATTERN_HPP
#define AXIS_SOLVER_HALO_PATTERN_HPP

/// @file axis/solver/halo_pattern.hpp
/// @brief Off-rank source-gather pattern for distributed SpMV apply.
///
/// HaloPattern is the distributed communication seam: a pure-data description
/// of the off-rank source values a local sparse-matrix apply needs. Produced by
/// WeightGenerator in distributed mode; handed directly to HALO (or a Python
/// layer) by the caller — NO DAGR, NO MPI/HALO/AMIO types, only integer index
/// arrays. AXIS performs the analysis; HALO performs the exchange.
///
/// Off-rank global source ids are grouped by owning neighbor rank in CSR form:
///   ids needed from source_ranks[r] are
///     needed_global_src_ids[rank_offsets[r] .. rank_offsets[r+1]).
/// For each needed id k, gather_slot[k] is the local index in the gathered
/// "halo" source buffer where HALO must deposit that cell's value.

#include <cstddef>
#include <vector>

#include <axis/types.hpp>

namespace axis::solver {

/// Plain-data exchange-plan describing the off-rank source cells a local
/// distributed apply requires.
///
/// Contains NO MPI types, NO HALO types, NO AMIO types — only integer arrays.
struct HaloPattern {
    std::vector<int>     source_ranks;          ///< Distinct neighbor ranks to gather from
    std::vector<index_t> rank_offsets;          ///< CSR offsets [source_ranks.size() + 1]
    std::vector<index_t> needed_global_src_ids; ///< Off-rank global source ids, grouped by neighbor
    std::vector<index_t> gather_slot;           ///< Local slot in gathered halo buffer per id

    /// Total number of off-rank source values to gather (== gathered-buffer size).
    [[nodiscard]] std::size_t num_remote() const noexcept {
        return needed_global_src_ids.size();
    }

    /// Number of distinct neighbor ranks involved.
    [[nodiscard]] std::size_t num_ranks() const noexcept {
        return source_ranks.size();
    }
};

} // namespace axis::solver

#endif // AXIS_SOLVER_HALO_PATTERN_HPP
