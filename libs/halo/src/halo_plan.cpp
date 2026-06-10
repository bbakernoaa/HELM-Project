#include "halo/halo_plan.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace halo {

namespace {

/// @brief Validate a neighbor list against the communicator size.
/// @throws std::invalid_argument for out-of-range or duplicate ranks.
void validate_neighbors(const std::vector<Neighbor_Info>& neighbors,
                        int comm_size,
                        const char* direction) {
    std::unordered_set<int> seen;
    seen.reserve(neighbors.size());

    for (const auto& info : neighbors) {
        // Range check
        if (info.rank < 0 || info.rank >= comm_size) {
            std::ostringstream oss;
            oss << "Halo_Plan: invalid " << direction << " neighbor rank "
                << info.rank << "; valid range is [0, " << comm_size << ")";
            throw std::invalid_argument(oss.str());
        }

        // Duplicate check
        auto [it, inserted] = seen.insert(info.rank);
        if (!inserted) {
            std::ostringstream oss;
            oss << "Halo_Plan: duplicate " << direction << " neighbor rank "
                << info.rank;
            throw std::invalid_argument(oss.str());
        }
    }
}

/// @brief Compute the sum of element counts across all neighbors.
std::size_t sum_counts(const std::vector<Neighbor_Info>& neighbors) noexcept {
    return std::accumulate(
        neighbors.begin(), neighbors.end(), std::size_t{0},
        [](std::size_t acc, const Neighbor_Info& info) {
            return acc + info.count;
        });
}

} // anonymous namespace

Halo_Plan::Halo_Plan(const Communicator& comm,
                     std::vector<Neighbor_Info> send_neighbors,
                     std::vector<Neighbor_Info> recv_neighbors)
    : comm_{&comm},
      send_neighbors_{std::move(send_neighbors)},
      recv_neighbors_{std::move(recv_neighbors)} {

    const int comm_size = comm.size();

    // Validate send and receive neighbor lists
    validate_neighbors(send_neighbors_, comm_size, "send");
    validate_neighbors(recv_neighbors_, comm_size, "recv");

    // Precompute totals
    total_send_ = sum_counts(send_neighbors_);
    total_recv_ = sum_counts(recv_neighbors_);
}

std::size_t Halo_Plan::num_send_neighbors() const noexcept {
    return send_neighbors_.size();
}

std::size_t Halo_Plan::num_recv_neighbors() const noexcept {
    return recv_neighbors_.size();
}

std::span<const Neighbor_Info> Halo_Plan::send_info() const noexcept {
    return send_neighbors_;
}

std::span<const Neighbor_Info> Halo_Plan::recv_info() const noexcept {
    return recv_neighbors_;
}

std::size_t Halo_Plan::total_send_elements() const noexcept {
    return total_send_;
}

std::size_t Halo_Plan::total_recv_elements() const noexcept {
    return total_recv_;
}

const Communicator& Halo_Plan::communicator() const noexcept {
    return *comm_;
}

} // namespace halo
