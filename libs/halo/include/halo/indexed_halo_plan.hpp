#ifndef HALO_INDEXED_HALO_PLAN_HPP
#define HALO_INDEXED_HALO_PLAN_HPP

/// @file indexed_halo_plan.hpp
/// @brief Precomputed indexed (gather/scatter) halo exchange communication plan.
///
/// Indexed_Halo_Plan stores, per neighbor and per halo layer, an ordered list of
/// local element indices to gather (send) or scatter into (recv). Unlike the
/// contiguous Halo_Plan (which assumes a flat `[send | owned | recv]` layout and
/// stores only `{rank, count}` per neighbor), the indexed plan targets fields
/// stored in an `[owned | halo]` layout with arbitrary per-neighbor index lists,
/// such as those produced by an unstructured-mesh partition decomposition.
///
/// The plan is domain-agnostic: it depends only on the HALO Communicator,
/// neighbor ranks, and index lists. It carries no application-specific types.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "halo/communicator.hpp"

namespace halo {

/// @brief The mesh element type a field (and its plan) is defined on.
///
/// `generic` lets non-mesh clients ignore the concept entirely.
enum class Element_Kind { cell, edge, vertex, generic };

/// @brief One neighbor's send or receive index lists, organized by halo layer.
///
/// `layers[l]` holds the ordered list of 0-based local indices belonging to halo
/// layer `l` (layer index `l` corresponds to the application's halo layer `l+1`
/// when a 1-based layer numbering is used upstream). A layer list may be empty,
/// meaning no elements are exchanged with this neighbor on that layer.
struct Indexed_Neighbor {
    int rank{0};                                    ///< Neighbor MPI rank.
    std::vector<std::vector<std::size_t>> layers;   ///< layers[l] = local indices for layer l.

    /// @brief Total number of indices across all layers for this neighbor.
    [[nodiscard]] std::size_t total_indices() const noexcept {
        std::size_t total = 0;
        for (const auto &layer : layers) {
            total += layer.size();
        }
        return total;
    }

    /// @brief Concatenate the indices for a subset of layers, in ascending
    ///        layer order.
    ///
    /// @param layer_subset 0-based layer indices to include. Entries that are
    ///        out of range for this neighbor's layer count are ignored.
    /// @return The concatenation of `layers[l]` for every valid `l` in the
    ///         subset, visited in ascending `l` order regardless of the order in
    ///         which the subset lists them (so results are order-independent).
    [[nodiscard]] std::vector<std::size_t> indices_for(std::span<const int> layer_subset) const {
        std::vector<std::size_t> result;

        // Determine which layers are selected, then walk them in ascending order
        // so the concatenation is deterministic and independent of the order in
        // which the caller lists the subset.
        for (std::size_t l = 0; l < layers.size(); ++l) {
            const bool selected =
                std::any_of(layer_subset.begin(), layer_subset.end(),
                            [l](int layer) { return layer >= 0 && static_cast<std::size_t>(layer) == l; });
            if (selected) {
                result.insert(result.end(), layers[l].begin(), layers[l].end());
            }
        }
        return result;
    }
};

/// @brief Precomputed, immutable indexed halo exchange metadata for one
///        Element_Kind.
///
/// Stores per-neighbor, per-layer send and receive index lists. Validates ranks
/// at construction time (rejecting out-of-range and duplicate ranks) and
/// precomputes total send/receive index counts and the halo-layer count.
/// Copyable and movable for storage in containers and reuse across exchanges.
class Indexed_Halo_Plan {
   public:
    /// @brief Construct an indexed halo plan with validated neighbor lists.
    ///
    /// @param comm Reference to the communicator used for exchanges. The
    ///        referenced Communicator must outlive this plan.
    /// @param kind The element kind this plan applies to.
    /// @param send_neighbors Per-neighbor, per-layer local indices to gather
    ///        (owned elements to send).
    /// @param recv_neighbors Per-neighbor, per-layer local indices to scatter
    ///        into (halo elements to fill).
    ///
    /// @throws std::invalid_argument if any rank is negative or >= comm.size().
    /// @throws std::invalid_argument if any rank appears more than once in the
    ///         same send or receive list.
    ///
    /// Empty index lists (empty layers, or empty per-layer lists) are valid and
    /// represent zero exchanged elements for that neighbor/layer.
    Indexed_Halo_Plan(const Communicator &comm, Element_Kind kind, std::vector<Indexed_Neighbor> send_neighbors,
                      std::vector<Indexed_Neighbor> recv_neighbors)
        : comm_{&comm},
          kind_{kind},
          send_neighbors_{std::move(send_neighbors)},
          recv_neighbors_{std::move(recv_neighbors)} {
        const int comm_size = comm.size();

        validate_neighbors(send_neighbors_, comm_size, "send");
        validate_neighbors(recv_neighbors_, comm_size, "recv");

        total_send_ = sum_indices(send_neighbors_);
        total_recv_ = sum_indices(recv_neighbors_);
        num_layers_ = std::max(max_layers(send_neighbors_), max_layers(recv_neighbors_));
    }

    // Copyable and movable (defaulted special members).
    Indexed_Halo_Plan(const Indexed_Halo_Plan &) = default;
    Indexed_Halo_Plan &operator=(const Indexed_Halo_Plan &) = default;
    Indexed_Halo_Plan(Indexed_Halo_Plan &&) noexcept = default;
    Indexed_Halo_Plan &operator=(Indexed_Halo_Plan &&) noexcept = default;

    /// @brief The element kind this plan applies to.
    [[nodiscard]] Element_Kind element_kind() const noexcept { return kind_; }

    /// @brief Immutable view of send neighbor information.
    [[nodiscard]] std::span<const Indexed_Neighbor> send_info() const noexcept { return send_neighbors_; }

    /// @brief Immutable view of receive neighbor information.
    [[nodiscard]] std::span<const Indexed_Neighbor> recv_info() const noexcept { return recv_neighbors_; }

    /// @brief Number of distinct send neighbors.
    [[nodiscard]] std::size_t num_send_neighbors() const noexcept { return send_neighbors_.size(); }

    /// @brief Number of distinct receive neighbors.
    [[nodiscard]] std::size_t num_recv_neighbors() const noexcept { return recv_neighbors_.size(); }

    /// @brief Total indices to send (sum over all neighbors and layers).
    [[nodiscard]] std::size_t total_send_indices() const noexcept { return total_send_; }

    /// @brief Total indices to receive (sum over all neighbors and layers).
    [[nodiscard]] std::size_t total_recv_indices() const noexcept { return total_recv_; }

    /// @brief Number of halo layers (max layer count over all neighbors).
    [[nodiscard]] std::size_t num_layers() const noexcept { return num_layers_; }

    /// @brief Reference to the communicator used for exchanges.
    /// @note The referenced Communicator must outlive this plan.
    [[nodiscard]] const Communicator &communicator() const noexcept { return *comm_; }

   private:
    /// @brief Validate a neighbor list against the communicator size.
    /// @throws std::invalid_argument for out-of-range or duplicate ranks.
    static void validate_neighbors(const std::vector<Indexed_Neighbor> &neighbors, int comm_size,
                                   const char *direction) {
        std::unordered_set<int> seen;
        seen.reserve(neighbors.size());

        for (const auto &info : neighbors) {
            if (info.rank < 0 || info.rank >= comm_size) {
                std::ostringstream oss;
                oss << "Indexed_Halo_Plan: invalid " << direction << " neighbor rank " << info.rank
                    << "; valid range is [0, " << comm_size << ")";
                throw std::invalid_argument(oss.str());
            }

            auto [it, inserted] = seen.insert(info.rank);
            if (!inserted) {
                std::ostringstream oss;
                oss << "Indexed_Halo_Plan: duplicate " << direction << " neighbor rank " << info.rank;
                throw std::invalid_argument(oss.str());
            }
        }
    }

    /// @brief Sum of index counts across all neighbors and layers.
    static std::size_t sum_indices(const std::vector<Indexed_Neighbor> &neighbors) noexcept {
        return std::accumulate(neighbors.begin(), neighbors.end(), std::size_t{0},
                               [](std::size_t acc, const Indexed_Neighbor &info) { return acc + info.total_indices(); });
    }

    /// @brief Maximum number of layers present across all neighbors.
    static std::size_t max_layers(const std::vector<Indexed_Neighbor> &neighbors) noexcept {
        std::size_t max_l = 0;
        for (const auto &info : neighbors) {
            max_l = std::max(max_l, info.layers.size());
        }
        return max_l;
    }

    const Communicator *comm_;                        ///< Non-owning pointer to communicator.
    Element_Kind kind_;                               ///< Element kind this plan applies to.
    std::vector<Indexed_Neighbor> send_neighbors_;    ///< Send neighbor list (immutable after construction).
    std::vector<Indexed_Neighbor> recv_neighbors_;    ///< Receive neighbor list (immutable after construction).
    std::size_t total_send_{0};                       ///< Precomputed sum of send index counts.
    std::size_t total_recv_{0};                       ///< Precomputed sum of recv index counts.
    std::size_t num_layers_{0};                       ///< Precomputed halo-layer count.
};

}  // namespace halo

#endif  // HALO_INDEXED_HALO_PLAN_HPP
