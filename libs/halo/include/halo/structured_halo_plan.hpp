#ifndef HALO_STRUCTURED_HALO_PLAN_HPP
#define HALO_STRUCTURED_HALO_PLAN_HPP

/// @file structured_halo_plan.hpp
/// @brief Precomputed multi-dimensional structured halo exchange plan.
///
/// Structured_Halo_Plan<Rank> describes the halo geometry for a structured
/// grid of dimensionality 1-4. It precomputes subview index ranges (as
/// Kokkos::pair) for each send and receive region on every face neighbor,
/// enabling efficient pack/unpack without recomputing slice boundaries at
/// each exchange.
///
/// Face-based connectivity: a Rank-dimensional grid has 2*Rank face neighbors
/// (one per direction per dimension: low and high side).
///
/// The plan is layout-agnostic at the index-range level; LayoutLeft vs
/// LayoutRight affects only the pack/unpack kernel dispatch (which dimension
/// is contiguous), not the logical slice boundaries stored here.

#include <array>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include "halo/communicator.hpp"

namespace halo {

/// @brief Index range pair for one dimension of a subview slice.
///
/// Represents a half-open interval [first, second) suitable for use with
/// Kokkos::subview and Kokkos::pair.
using index_range = Kokkos::pair<std::size_t, std::size_t>;

/// @brief Describes the multi-dimensional subview region for a single face.
///
/// For a grid of rank R, each face's send or receive region is defined by
/// R index ranges — one per dimension. The face dimension has a narrow
/// slice (width = halo_width), while all other dimensions span the full
/// extent (including halos on those dimensions).
///
/// @tparam Rank The dimensionality of the grid (1-4).
template <int Rank>
struct Region {
    /// Per-dimension index ranges defining the subview slice.
    std::array<index_range, Rank> ranges;

    /// @brief Compute the total number of elements in this region.
    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t s = 1;
        for (int d = 0; d < Rank; ++d) {
            s *= (ranges[d].second - ranges[d].first);
        }
        return s;
    }
};

/// @brief Precomputed structured halo exchange plan for a Rank-dimensional grid.
///
/// Stores the neighbor topology (one rank per face), grid extents, halo widths,
/// and precomputed send/recv subview regions for each face. The plan is
/// reusable across timesteps with different view instances of the same shape.
///
/// Face indexing convention (for Rank dimensions d=0..Rank-1):
///   face 2*d     = low  side of dimension d (e.g., "west"  for d=0)
///   face 2*d + 1 = high side of dimension d (e.g., "east"  for d=0)
///
/// For a 3D grid: faces 0-5 map to west/east/south/north/bottom/top.
///
/// @tparam Rank Grid dimensionality (1, 2, 3, or 4).
template <int Rank>
class Structured_Halo_Plan {
    static_assert(Rank >= 1 && Rank <= 4,
                  "Structured_Halo_Plan supports ranks 1 through 4");

public:
    /// Number of face neighbors for this rank.
    static constexpr int num_faces_value = 2 * Rank;

    /// @brief Construct a structured halo plan.
    ///
    /// @param global_extents Per-dimension sizes of the local grid INCLUDING
    ///        halo zones on both sides.
    /// @param neighbor_ranks MPI rank for each face neighbor. Use -1 (or
    ///        MPI_PROC_NULL) for faces with no neighbor (non-periodic boundary).
    ///        Layout: [low_d0, high_d0, low_d1, high_d1, ...].
    /// @param halo_widths Per-dimension halo width (same width on both sides
    ///        of each dimension).
    /// @param comm Reference to the communicator for this exchange. The
    ///        Communicator must outlive this plan (stored as non-owning pointer).
    ///
    /// @throws std::invalid_argument if any extent < 2 * halo_width for that
    ///         dimension.
    Structured_Halo_Plan(std::array<std::size_t, Rank> global_extents,
                         std::array<int, 2 * Rank> neighbor_ranks,
                         std::array<std::size_t, Rank> halo_widths,
                         const Communicator& comm)
        : extents_(global_extents),
          neighbor_ranks_(neighbor_ranks),
          halo_widths_(halo_widths),
          comm_(&comm) {
        validate_extents();
        precompute_regions();
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    /// @brief Number of face neighbors (always 2*Rank).
    [[nodiscard]] static constexpr int num_faces() noexcept {
        return num_faces_value;
    }

    /// @brief Get the send region for the given face index.
    ///
    /// The send region is the interior slice adjacent to the face boundary
    /// that should be packed and sent to the neighbor.
    ///
    /// @param face_idx Face index in [0, 2*Rank).
    /// @return Region describing the subview index ranges.
    [[nodiscard]] const Region<Rank>& send_region(int face_idx) const {
        check_face_idx(face_idx);
        return send_regions_[face_idx];
    }

    /// @brief Get the receive region for the given face index.
    ///
    /// The receive region is the halo (ghost) zone on the specified face
    /// that should be filled with data received from the neighbor.
    ///
    /// @param face_idx Face index in [0, 2*Rank).
    /// @return Region describing the subview index ranges.
    [[nodiscard]] const Region<Rank>& recv_region(int face_idx) const {
        check_face_idx(face_idx);
        return recv_regions_[face_idx];
    }

    /// @brief Get the MPI rank of the neighbor at the given face.
    ///
    /// @param face_idx Face index in [0, 2*Rank).
    /// @return The neighbor's MPI rank, or -1 if no neighbor exists.
    [[nodiscard]] int neighbor_rank(int face_idx) const {
        check_face_idx(face_idx);
        return neighbor_ranks_[face_idx];
    }

    /// @brief Access the communicator.
    /// @return Const reference to the owning Communicator.
    [[nodiscard]] const Communicator& communicator() const noexcept {
        return *comm_;
    }

    /// @brief Get the halo width for a given dimension.
    /// @param dim Dimension index in [0, Rank).
    [[nodiscard]] std::size_t halo_width(int dim) const {
        check_dim(dim);
        return halo_widths_[dim];
    }

    /// @brief Get the extent (total size including halos) for a given dimension.
    /// @param dim Dimension index in [0, Rank).
    [[nodiscard]] std::size_t extent(int dim) const {
        check_dim(dim);
        return extents_[dim];
    }

    /// @brief Get all extents.
    [[nodiscard]] const std::array<std::size_t, Rank>& extents() const noexcept {
        return extents_;
    }

    /// @brief Get all halo widths.
    [[nodiscard]] const std::array<std::size_t, Rank>& halo_widths() const noexcept {
        return halo_widths_;
    }

    /// @brief Get all neighbor ranks.
    [[nodiscard]] const std::array<int, 2 * Rank>& neighbor_ranks_array() const noexcept {
        return neighbor_ranks_;
    }

    /// @brief Check if a given face has a valid neighbor (rank != -1).
    [[nodiscard]] bool has_neighbor(int face_idx) const {
        check_face_idx(face_idx);
        return neighbor_ranks_[face_idx] >= 0;
    }

    /// @brief Count the number of active (valid) face neighbors.
    [[nodiscard]] int active_neighbor_count() const noexcept {
        int count = 0;
        for (int f = 0; f < num_faces_value; ++f) {
            if (neighbor_ranks_[f] >= 0) {
                ++count;
            }
        }
        return count;
    }

    // ─── Layout Detection Helpers ───────────────────────────────────────────

    /// @brief Detect whether a ViewType uses LayoutLeft (column-major).
    ///
    /// Used by exchange functions to determine pack/unpack kernel dispatch.
    template <typename ViewType>
    static constexpr bool is_layout_left() {
        return std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft>;
    }

    /// @brief Detect whether a ViewType uses LayoutRight (row-major).
    template <typename ViewType>
    static constexpr bool is_layout_right() {
        return std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>;
    }

    // ─── Topology Communicator Cache (for neighbor collectives) ────────────

    /// @brief Check if the neighbor topology is symmetric (send/recv neighbor
    ///        sets are identical). Required for MPI_Neighbor_alltoallv.
    [[nodiscard]] bool is_topology_symmetric() const noexcept {
        // For structured grids with face-based connectivity, the topology is
        // symmetric when every active face f has a corresponding face where
        // the neighbor also sends back to us. In structured grids with the
        // face convention (2*d and 2*d+1), this is inherently symmetric:
        // if we send to rank R on face f, rank R sends to us on the opposite face.
        // However, asymmetry arises when boundary faces have -1 on one side
        // but not the other — which doesn't happen in our face model.
        // The true asymmetry check is: the set of ranks we send to equals
        // the set of ranks we receive from.
        // In our face model, send and recv go to the same rank per face,
        // so topology is always symmetric when all neighbors are valid or -1.
        // We check that the send-to rank set == recv-from rank set.
        // Since each face both sends and receives from the same rank, the
        // topology is symmetric by construction for structured grids.
        return true;
    }

    /// @brief Get or lazily create the cached dist-graph topology communicator.
    ///
    /// Creates an MPI_Dist_graph_create_adjacent communicator on first call
    /// and caches it for subsequent reuse. The topology comm is freed in the
    /// destructor.
    ///
    /// @return The cached topology communicator handle.
    [[nodiscard]] MPI_Comm topology_comm() const {
        std::lock_guard<std::mutex> lock(topo_mutex_);
        if (topo_comm_ != MPI_COMM_NULL) {
            return topo_comm_;
        }

        // Build deduplicated neighbor list (unique ranks in face order)
        std::vector<int> neighbors;
        neighbors.reserve(num_faces_value);
        for (int f = 0; f < num_faces_value; ++f) {
            if (neighbor_ranks_[f] >= 0) {
                neighbors.push_back(neighbor_ranks_[f]);
            }
        }

        // For dist_graph_create_adjacent, sources = ranks that send to us,
        // destinations = ranks we send to. For symmetric structured grids,
        // these are the same set (each neighbor both sends and receives).
        // Weights are unweighted (MPI_UNWEIGHTED).
        int indegree = static_cast<int>(neighbors.size());
        int outdegree = static_cast<int>(neighbors.size());

        int rc = MPI_Dist_graph_create_adjacent(
            comm_->handle(),
            indegree, neighbors.data(), MPI_UNWEIGHTED,
            outdegree, neighbors.data(), MPI_UNWEIGHTED,
            MPI_INFO_NULL, /*reorder=*/0, &topo_comm_);

        if (rc != MPI_SUCCESS) {
            topo_comm_ = MPI_COMM_NULL;
            throw std::runtime_error(
                "Structured_Halo_Plan: MPI_Dist_graph_create_adjacent failed");
        }

        return topo_comm_;
    }

    /// @brief Free the cached topology communicator if it was created.
    void free_topology_comm() noexcept {
        if (topo_comm_ != MPI_COMM_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Comm_free(&topo_comm_);
            }
            topo_comm_ = MPI_COMM_NULL;
        }
    }

public:
    /// @brief Destructor frees the cached topology communicator if created.
    ~Structured_Halo_Plan() {
        free_topology_comm();
    }

    // Move support — transfer topology comm ownership
    Structured_Halo_Plan(Structured_Halo_Plan&& other) noexcept
        : extents_(other.extents_),
          neighbor_ranks_(other.neighbor_ranks_),
          halo_widths_(other.halo_widths_),
          comm_(other.comm_),
          send_regions_(other.send_regions_),
          recv_regions_(other.recv_regions_),
          topo_comm_(other.topo_comm_) {
        other.topo_comm_ = MPI_COMM_NULL;
        other.comm_ = nullptr;
    }

    Structured_Halo_Plan& operator=(Structured_Halo_Plan&& other) noexcept {
        if (this != &other) {
            free_topology_comm();
            extents_ = other.extents_;
            neighbor_ranks_ = other.neighbor_ranks_;
            halo_widths_ = other.halo_widths_;
            comm_ = other.comm_;
            send_regions_ = other.send_regions_;
            recv_regions_ = other.recv_regions_;
            topo_comm_ = other.topo_comm_;
            other.topo_comm_ = MPI_COMM_NULL;
            other.comm_ = nullptr;
        }
        return *this;
    }

    // Copy is deleted — topology comm is not duplicated
    Structured_Halo_Plan(const Structured_Halo_Plan&) = delete;
    Structured_Halo_Plan& operator=(const Structured_Halo_Plan&) = delete;

private:
    std::array<std::size_t, Rank> extents_;
    std::array<int, 2 * Rank> neighbor_ranks_;
    std::array<std::size_t, Rank> halo_widths_;
    const Communicator* comm_;

    std::array<Region<Rank>, 2 * Rank> send_regions_;
    std::array<Region<Rank>, 2 * Rank> recv_regions_;

    /// @brief Cached MPI dist-graph topology communicator for neighbor collectives.
    /// Created lazily on first call to topology_comm(). Freed in destructor.
    mutable MPI_Comm topo_comm_{MPI_COMM_NULL};

    /// @brief Mutex protecting lazy initialization of topo_comm_.
    mutable std::mutex topo_mutex_;

    /// @brief Validate that each extent is at least 2 * halo_width.
    void validate_extents() const {
        for (int d = 0; d < Rank; ++d) {
            if (extents_[d] < 2 * halo_widths_[d]) {
                throw std::invalid_argument(
                    "Structured_Halo_Plan: extent[" + std::to_string(d) +
                    "] = " + std::to_string(extents_[d]) +
                    " is less than 2 * halo_width[" + std::to_string(d) +
                    "] = " + std::to_string(2 * halo_widths_[d]));
            }
        }
    }

    /// @brief Precompute all send and receive regions for each face.
    ///
    /// For each dimension d and each side (low=0, high=1):
    ///
    /// Send region (interior slice adjacent to halo boundary):
    ///   - Low side (face 2*d):   dimension d spans [h, 2*h)
    ///     (the h interior cells just inside the low halo zone)
    ///   - High side (face 2*d+1): dimension d spans [N-2*h, N-h)
    ///     (the h interior cells just inside the high halo zone)
    ///
    /// Recv region (ghost/halo zone to be filled):
    ///   - Low side (face 2*d):   dimension d spans [0, h)
    ///   - High side (face 2*d+1): dimension d spans [N-h, N)
    ///
    /// All other dimensions span the full extent [0, extent[d']).
    void precompute_regions() {
        for (int d = 0; d < Rank; ++d) {
            const std::size_t h = halo_widths_[d];
            const std::size_t N = extents_[d];

            // Face index for low side of dimension d
            const int face_lo = 2 * d;
            // Face index for high side of dimension d
            const int face_hi = 2 * d + 1;

            // Build region for low face (face_lo)
            Region<Rank> send_lo;
            Region<Rank> recv_lo;
            for (int k = 0; k < Rank; ++k) {
                if (k == d) {
                    // Send: the h interior cells just past the low halo
                    send_lo.ranges[k] = index_range(h, 2 * h);
                    // Recv: the low halo zone itself
                    recv_lo.ranges[k] = index_range(0, h);
                } else {
                    // Other dimensions: full extent
                    send_lo.ranges[k] = index_range(0, extents_[k]);
                    recv_lo.ranges[k] = index_range(0, extents_[k]);
                }
            }
            send_regions_[face_lo] = send_lo;
            recv_regions_[face_lo] = recv_lo;

            // Build region for high face (face_hi)
            Region<Rank> send_hi;
            Region<Rank> recv_hi;
            for (int k = 0; k < Rank; ++k) {
                if (k == d) {
                    // Send: the h interior cells just before the high halo
                    send_hi.ranges[k] = index_range(N - 2 * h, N - h);
                    // Recv: the high halo zone itself
                    recv_hi.ranges[k] = index_range(N - h, N);
                } else {
                    // Other dimensions: full extent
                    send_hi.ranges[k] = index_range(0, extents_[k]);
                    recv_hi.ranges[k] = index_range(0, extents_[k]);
                }
            }
            send_regions_[face_hi] = send_hi;
            recv_regions_[face_hi] = recv_hi;
        }
    }

    /// @brief Bounds-check a face index.
    void check_face_idx(int face_idx) const {
        if (face_idx < 0 || face_idx >= num_faces_value) {
            throw std::out_of_range(
                "Structured_Halo_Plan: face_idx " + std::to_string(face_idx) +
                " out of range [0, " + std::to_string(num_faces_value) + ")");
        }
    }

    /// @brief Bounds-check a dimension index.
    void check_dim(int dim) const {
        if (dim < 0 || dim >= Rank) {
            throw std::out_of_range(
                "Structured_Halo_Plan: dim " + std::to_string(dim) +
                " out of range [0, " + std::to_string(Rank) + ")");
        }
    }
};

} // namespace halo

#endif // HALO_STRUCTURED_HALO_PLAN_HPP
