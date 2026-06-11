#ifndef HALO_DETAIL_COMPUTE_TAG_HPP
#define HALO_DETAIL_COMPUTE_TAG_HPP

/// @file halo/detail/compute_tag.hpp
/// @brief Deterministic MPI tag computation for halo exchanges.
///
/// Computes a unique MPI tag from sender, receiver, and communicator size
/// using modular arithmetic. The tag is deterministic and reproducible for
/// matching send/recv pairs.

#include <mpi.h>

namespace halo::detail {

/// @brief Compute a deterministic MPI tag from sender, receiver, and comm size.
///
/// tag = (sender_rank * comm_size + receiver_rank) % MPI_TAG_UB_VALUE
///
/// @param sender   The rank of the sending process.
/// @param receiver The rank of the receiving process.
/// @param comm_size Total number of processes in the communicator.
/// @return A tag value in [0, MPI_TAG_UB).
inline int compute_tag(int sender, int receiver, int comm_size) noexcept {
    // Query MPI_TAG_UB from MPI_COMM_WORLD
    int* tag_ub_ptr = nullptr;
    int flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_ptr, &flag);

    // MPI standard guarantees MPI_TAG_UB >= 32767
    int tag_ub = (flag && tag_ub_ptr != nullptr) ? *tag_ub_ptr : 32767;

    // Compute deterministic tag using modular arithmetic
    long long product = static_cast<long long>(sender) * comm_size + receiver;
    return static_cast<int>(product % tag_ub);
}

} // namespace halo::detail

#endif // HALO_DETAIL_COMPUTE_TAG_HPP
