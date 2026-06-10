#ifndef HALO_HALO_PLAN_HPP
#define HALO_HALO_PLAN_HPP

/// @file halo_plan.hpp
/// @brief Precomputed halo exchange communication plan.
///
/// Halo_Plan stores immutable send/receive neighbor topology and buffer geometry
/// for reuse across multiple halo exchange invocations. Analogous to the legacy
/// ESMF RouteHandle, it precomputes neighbor relationships once and amortizes
/// setup cost over thousands of exchanges.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "halo/communicator.hpp"

namespace halo {

/// @brief Describes a single neighbor in a halo exchange.
struct Neighbor_Info {
    int rank;             ///< MPI rank of the neighbor process.
    std::size_t count;    ///< Number of elements to send/receive.
};

/// @brief Precomputed, immutable halo exchange metadata.
///
/// Stores send and receive neighbor lists with per-neighbor element counts.
/// Validates ranks at construction time and precomputes total element counts.
/// Copyable and movable for storage in containers and reuse across exchanges.
class Halo_Plan {
public:
    /// @brief Construct a halo plan with validated neighbor lists.
    ///
    /// @param comm Reference to the communicator used for exchanges.
    /// @param send_neighbors List of send neighbors (rank + element count).
    /// @param recv_neighbors List of receive neighbors (rank + element count).
    ///
    /// @throws std::invalid_argument if any rank is negative or >= comm.size().
    /// @throws std::invalid_argument if any rank appears more than once in the
    ///         same send or receive list.
    ///
    /// Empty neighbor lists are valid (zero neighbors in that direction).
    Halo_Plan(const Communicator& comm,
              std::vector<Neighbor_Info> send_neighbors,
              std::vector<Neighbor_Info> recv_neighbors);

    // Copyable and movable (defaulted special members)
    Halo_Plan(const Halo_Plan&) = default;
    Halo_Plan& operator=(const Halo_Plan&) = default;
    Halo_Plan(Halo_Plan&&) noexcept = default;
    Halo_Plan& operator=(Halo_Plan&&) noexcept = default;

    /// @brief Number of distinct send neighbors.
    [[nodiscard]] std::size_t num_send_neighbors() const noexcept;

    /// @brief Number of distinct receive neighbors.
    [[nodiscard]] std::size_t num_recv_neighbors() const noexcept;

    /// @brief Immutable view of send neighbor information.
    [[nodiscard]] std::span<const Neighbor_Info> send_info() const noexcept;

    /// @brief Immutable view of receive neighbor information.
    [[nodiscard]] std::span<const Neighbor_Info> recv_info() const noexcept;

    /// @brief Total elements to send (sum of all send counts).
    [[nodiscard]] std::size_t total_send_elements() const noexcept;

    /// @brief Total elements to receive (sum of all recv counts).
    [[nodiscard]] std::size_t total_recv_elements() const noexcept;

    /// @brief Reference to the communicator used for exchanges.
    /// @note The referenced Communicator must outlive this Halo_Plan.
    [[nodiscard]] const Communicator& communicator() const noexcept;

private:
    const Communicator* comm_;                    ///< Non-owning pointer to communicator.
    std::vector<Neighbor_Info> send_neighbors_;   ///< Send neighbor list (immutable after construction).
    std::vector<Neighbor_Info> recv_neighbors_;   ///< Receive neighbor list (immutable after construction).
    std::size_t total_send_{0};                   ///< Precomputed sum of send counts.
    std::size_t total_recv_{0};                   ///< Precomputed sum of recv counts.
};

} // namespace halo

#endif // HALO_HALO_PLAN_HPP
