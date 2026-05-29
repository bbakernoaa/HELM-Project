#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

/// @file exchange.hpp
/// @brief Blocking and non-blocking halo exchange function templates.
///
/// Provides exchange_blocking<ViewType> which executes a synchronous halo
/// exchange using a precomputed Halo_Plan. Posts all MPI_Irecv before any
/// MPI_Isend, uses deterministic tag computation, and dispatches between
/// GPU-direct and host-staged paths at compile time via requires_staging_v.
///
/// Thread safety is ensured via detail::Serialized_MPI_Guard when the MPI
/// thread level is below MPI_THREAD_MULTIPLE.

#include <mpi.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <halo/communicator.hpp>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/staging.hpp>
#include <halo/environment.hpp>
#include <halo/halo_handle.hpp>
#include <halo/halo_plan.hpp>

namespace halo {

namespace detail {

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

/// @brief Throw a std::runtime_error with MPI error string and failing rank.
///
/// @param mpi_error_code The MPI error code returned by the failing call.
/// @param neighbor_rank  The rank of the neighbor involved in the failure.
/// @param operation      Description of the failing operation (e.g., "MPI_Irecv").
[[noreturn]] inline void throw_mpi_error(int mpi_error_code,
                                         int neighbor_rank,
                                         const char* operation) {
    char error_string[MPI_MAX_ERROR_STRING];
    int resultlen = 0;
    MPI_Error_string(mpi_error_code, error_string, &resultlen);

    std::string msg = std::string(operation) + " failed for rank " +
                      std::to_string(neighbor_rank) + ": " +
                      std::string(error_string, static_cast<std::size_t>(resultlen));
    throw std::runtime_error(msg);
}

} // namespace detail

/// @brief Blocking halo exchange.
///
/// Posts all MPI_Irecv before any MPI_Isend, then calls MPI_Waitall.
/// Dispatches GPU-direct or host-staged path at compile time based on
/// requires_staging_v<ViewType>.
///
/// @tparam ViewType A Kokkos::View type (1D).
/// @param plan The precomputed halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
///
/// @throws std::runtime_error if any MPI operation fails.
///
/// @note Uses detail::Serialized_MPI_Guard for thread safety when MPI
///       thread level < MPI_THREAD_MULTIPLE.
template <typename ViewType>
void exchange_blocking(const Halo_Plan& plan, ViewType& view) {
    // Early return for empty plans (no neighbors in either direction)
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return;
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto& comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();

    // Determine MPI datatype from the view's value type
    using value_type = typename ViewType::value_type;
    MPI_Datatype mpi_dtype;
    if constexpr (std::is_same_v<value_type, double>) {
        mpi_dtype = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<value_type, float>) {
        mpi_dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<value_type, int>) {
        mpi_dtype = MPI_INT;
    } else if constexpr (std::is_same_v<value_type, long>) {
        mpi_dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<value_type, long long>) {
        mpi_dtype = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned int>) {
        mpi_dtype = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<value_type, char>) {
        mpi_dtype = MPI_CHAR;
    } else {
        // Fallback: treat as raw bytes
        mpi_dtype = MPI_BYTE;
    }

    const auto recv_info = plan.recv_info();
    const auto send_info = plan.send_info();

    const std::size_t num_recv = recv_info.size();
    const std::size_t num_send = send_info.size();
    const std::size_t total_requests = num_recv + num_send;

    // Allocate request array for MPI_Waitall
    std::vector<MPI_Request> requests(total_requests, MPI_REQUEST_NULL);

    if constexpr (detail::requires_staging_v<ViewType>) {
        // ─── Staged path: device view requires host buffers ─────────────────

        // Allocate host receive buffers
        std::vector<detail::host_mirror_t<ViewType>> recv_buffers;
        recv_buffers.reserve(num_recv);

        // Post all MPI_Irecv first (into host buffers)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto& neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            // Allocate host buffer for this receive
            auto host_buf = Kokkos::View<value_type*, Kokkos::HostSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "recv_buf"),
                neighbor.count);
            recv_buffers.push_back(host_buf);

            int rc = MPI_Irecv(
                host_buf.data(),
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &requests[i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }
        }

        // Stage send data from device to host, then post MPI_Isend
        std::vector<detail::host_mirror_t<ViewType>> send_buffers;
        send_buffers.reserve(num_send);

        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto& neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            // Deep-copy send region from device to host
            auto host_buf = detail::stage_send(view, send_offset, neighbor.count);
            send_buffers.push_back(host_buf);

            int rc = MPI_Isend(
                host_buf.data(),
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &requests[num_recv + i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            send_offset += neighbor.count;
        }

        // Wait for all operations to complete
        int rc = MPI_Waitall(
            static_cast<int>(total_requests),
            requests.data(),
            MPI_STATUSES_IGNORE);

        if (rc != MPI_SUCCESS) {
            detail::throw_mpi_error(rc, my_rank, "MPI_Waitall");
        }

        // Deep-copy received data from host buffers back to device view
        recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto& neighbor = recv_info[i];
            detail::stage_recv<ViewType>(recv_buffers[i], view, recv_offset, neighbor.count);
            recv_offset += neighbor.count;
        }

    } else {
        // ─── Direct path: host view or GPU-aware MPI ────────────────────────

        // Post all MPI_Irecv first (directly into view)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto& neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            int rc = MPI_Irecv(
                view.data() + recv_offset,
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &requests[i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }

            recv_offset += neighbor.count;
        }

        // Post all MPI_Isend (directly from view)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto& neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            int rc = MPI_Isend(
                view.data() + send_offset,
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &requests[num_recv + i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            send_offset += neighbor.count;
        }

        // Wait for all operations to complete
        int rc = MPI_Waitall(
            static_cast<int>(total_requests),
            requests.data(),
            MPI_STATUSES_IGNORE);

        if (rc != MPI_SUCCESS) {
            detail::throw_mpi_error(rc, my_rank, "MPI_Waitall");
        }
    }
}

/// @brief Non-blocking asynchronous halo exchange.
///
/// Posts all MPI_Irecv before any MPI_Isend, wraps each request in a
/// Request_Guard, and returns a Halo_Handle that owns all pending operations.
/// Dispatches GPU-direct or host-staged path at compile time based on
/// requires_staging_v<ViewType>.
///
/// @tparam ViewType A Kokkos::View type (1D).
/// @param plan The precomputed halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
/// @return A Halo_Handle owning all pending Request_Guard objects.
///
/// @throws std::runtime_error if any MPI operation fails.
///
/// @note Uses detail::Serialized_MPI_Guard for thread safety when MPI
///       thread level < MPI_THREAD_MULTIPLE.
template <typename ViewType>
[[nodiscard]] Halo_Handle exchange_async(const Halo_Plan& plan, ViewType& view) {
    // Early return for empty plans (no neighbors in either direction)
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return Halo_Handle{};
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto& comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();

    // Determine MPI datatype from the view's value type
    using value_type = typename ViewType::value_type;
    MPI_Datatype mpi_dtype;
    if constexpr (std::is_same_v<value_type, double>) {
        mpi_dtype = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<value_type, float>) {
        mpi_dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<value_type, int>) {
        mpi_dtype = MPI_INT;
    } else if constexpr (std::is_same_v<value_type, long>) {
        mpi_dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<value_type, long long>) {
        mpi_dtype = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned int>) {
        mpi_dtype = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<value_type, char>) {
        mpi_dtype = MPI_CHAR;
    } else {
        // Fallback: treat as raw bytes
        mpi_dtype = MPI_BYTE;
    }

    const auto recv_info = plan.recv_info();
    const auto send_info = plan.send_info();

    const std::size_t num_recv = recv_info.size();
    const std::size_t num_send = send_info.size();

    // Build the Halo_Handle that will own all Request_Guards
    Halo_Handle handle;
    handle.requests_.reserve(num_recv + num_send);

    if constexpr (detail::requires_staging_v<ViewType>) {
        // ─── Staged path: device view requires host buffers ─────────────────

        // Allocate host receive buffers
        std::vector<detail::host_mirror_t<ViewType>> recv_buffers;
        recv_buffers.reserve(num_recv);

        // Post all MPI_Irecv first (into host buffers)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto& neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            // Allocate host buffer for this receive
            auto host_buf = Kokkos::View<value_type*, Kokkos::HostSpace>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, "recv_buf"),
                neighbor.count);
            recv_buffers.push_back(host_buf);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Irecv(
                host_buf.data(),
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }

            handle.requests_.emplace_back(req);
        }

        // Stage send data from device to host, then post MPI_Isend
        std::vector<detail::host_mirror_t<ViewType>> send_buffers;
        send_buffers.reserve(num_send);

        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto& neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            // Deep-copy send region from device to host
            auto host_buf = detail::stage_send(view, send_offset, neighbor.count);
            send_buffers.push_back(host_buf);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Isend(
                host_buf.data(),
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            handle.requests_.emplace_back(req);
            send_offset += neighbor.count;
        }

        // Attach staged receive state: captures recv buffers and device view
        // reference for post-receive deep-copy upon completion.
        auto staged = std::make_unique<Halo_Handle::Staged_Recv>();
        staged->post_recv_copy = [recv_buffers = std::move(recv_buffers),
                                  send_buffers = std::move(send_buffers),
                                  &view,
                                  recv_info,
                                  total_send = plan.total_send_elements()]() mutable {
            std::size_t offset = total_send;
            for (std::size_t i = 0; i < recv_info.size(); ++i) {
                const auto& neighbor = recv_info[i];
                detail::stage_recv<ViewType>(recv_buffers[i], view, offset, neighbor.count);
                offset += neighbor.count;
            }
        };
        handle.staged_recv_ = std::move(staged);

    } else {
        // ─── Direct path: host view or GPU-aware MPI ────────────────────────

        // Post all MPI_Irecv first (directly into view)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto& neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Irecv(
                view.data() + recv_offset,
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }

            handle.requests_.emplace_back(req);
            recv_offset += neighbor.count;
        }

        // Post all MPI_Isend (directly from view)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto& neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Isend(
                view.data() + send_offset,
                static_cast<int>(neighbor.count),
                mpi_dtype,
                neighbor.rank,
                tag,
                mpi_comm,
                &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            handle.requests_.emplace_back(req);
            send_offset += neighbor.count;
        }
    }

    return handle;
}

} // namespace halo

#endif // HALO_EXCHANGE_HPP
