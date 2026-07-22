#ifndef HALO_PERSISTENT_HALO_HANDLE_HPP
#define HALO_PERSISTENT_HALO_HANDLE_HPP

/// @file persistent_halo_handle.hpp
/// @brief RAII wrapper for MPI persistent communication on halo exchanges.
///
/// Persistent_Halo_Handle binds a Halo_Plan to a specific Kokkos::View,
/// calling MPI_Send_init / MPI_Recv_init once at construction time. Subsequent
/// exchanges reuse the same persistent requests via start() → wait() cycles,
/// eliminating per-call MPI setup overhead for repeated halo exchanges with
/// the same buffer geometry.
///
/// The handle is move-only and RAII: the destructor calls MPI_Request_free
/// on all persistent requests. For device views without GPU-aware MPI,
/// host staging buffers are allocated at construction and pack/unpack is
/// performed around start()/wait().

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <functional>
#include <halo/communicator.hpp>
#include <halo/detail/compute_tag.hpp>
#include <halo/detail/gpu_aware_probe.hpp>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/mpi_datatype.hpp>
#include <halo/detail/staging.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/halo_plan.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace halo {

/// @brief RAII persistent communication handle for repeated halo exchanges.
///
/// Binds a Halo_Plan to a specific view at construction via MPI_Send_init /
/// MPI_Recv_init. The start() / wait() / test() interface allows repeated
/// exchanges without per-call setup overhead. Useful for NWP models that
/// perform the same halo exchange pattern every timestep.
///
/// Move-only: copy construction and copy assignment are deleted.
/// Destructor calls MPI_Request_free on all persistent requests.
///
/// For device views without GPU-aware MPI, the handle allocates host staging
/// buffers at construction. start() packs device→host before MPI_Startall,
/// and wait() unpacks host→device after MPI_Waitall.
class Persistent_Halo_Handle {
   public:
    /// @brief Construct a persistent handle by binding a plan to a view.
    ///
    /// Calls MPI_Send_init for each send neighbor and MPI_Recv_init for each
    /// receive neighbor, creating persistent MPI requests that can be started
    /// and completed repeatedly.
    ///
    /// For device views without GPU-aware MPI, host staging buffers are
    /// allocated here and reused across start/wait cycles.
    ///
    /// @tparam ViewType A Kokkos::View type (1D flat buffer).
    /// @param plan The precomputed halo exchange plan.
    /// @param view The Kokkos view to bind for persistent exchange.
    ///
    /// @throws std::runtime_error if any MPI_Send_init or MPI_Recv_init fails.
    template <typename ViewType>
    Persistent_Halo_Handle(const Halo_Plan &plan, ViewType &view);

    /// @brief Destructor. Calls MPI_Request_free on all persistent requests.
    ~Persistent_Halo_Handle();

    /// @brief Default construct an empty handle (no persistent requests).
    Persistent_Halo_Handle() noexcept = default;

    /// @brief Move constructor. Transfers ownership; source becomes empty.
    Persistent_Halo_Handle(Persistent_Halo_Handle &&other) noexcept;

    /// @brief Move assignment. Frees current requests, transfers from other.
    Persistent_Halo_Handle &operator=(Persistent_Halo_Handle &&other) noexcept;

    /// @brief Copy construction is deleted (unique ownership of persistent requests).
    Persistent_Halo_Handle(const Persistent_Halo_Handle &) = delete;

    /// @brief Copy assignment is deleted (unique ownership of persistent requests).
    Persistent_Halo_Handle &operator=(const Persistent_Halo_Handle &) = delete;

    /// @brief Initiate all persistent communication operations.
    ///
    /// For staged (device) views, packs the send regions from device to host
    /// staging buffers before calling MPI_Startall.
    ///
    /// @throws std::runtime_error if MPI_Startall fails.
    void start();

    /// @brief Block until all persistent communication operations complete.
    ///
    /// For staged (device) views, unpacks received data from host staging
    /// buffers back to the device view after MPI_Waitall completes.
    ///
    /// @throws std::runtime_error if MPI_Waitall fails.
    void wait();

    /// @brief Non-blocking test for completion of all persistent operations.
    ///
    /// If all operations have completed and staging is active, performs the
    /// host→device unpack. If any operation is still pending, returns false.
    ///
    /// @return true if all operations are complete; false if any are pending.
    [[nodiscard]] bool test();

    /// @brief Check if this handle has no persistent requests.
    /// @return true if default-constructed or moved-from.
    [[nodiscard]] bool empty() const noexcept;

   private:
    /// @brief Persistent MPI request handles (recv requests first, then sends).
    std::vector<MPI_Request> requests_;

    /// @brief Number of receive requests (for indexing into requests_).
    std::size_t num_recv_requests_{0};

    /// @brief Type-erased pack callback (device→host before start).
    /// Null when staging is not needed.
    std::function<void()> pack_fn_;

    /// @brief Type-erased unpack callback (host→device after wait/test).
    /// Null when staging is not needed.
    std::function<void()> unpack_fn_;

    /// @brief Free all persistent requests. Called by destructor and move-assign.
    void free_requests_() noexcept;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════════════

template <typename ViewType>
Persistent_Halo_Handle::Persistent_Halo_Handle(const Halo_Plan &plan, ViewType &view) {
    // Early return for empty plans
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return;
    }

    // Acquire serialization guard for thread safety during init
    detail::Serialized_MPI_Guard guard;

    const auto &comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();

    using value_type = typename ViewType::value_type;
    const MPI_Datatype mpi_dtype = detail::mpi_datatype_for<value_type>();

    const auto recv_info = plan.recv_info();
    const auto send_info = plan.send_info();

    const std::size_t num_recv = recv_info.size();
    const std::size_t num_send = send_info.size();

    num_recv_requests_ = num_recv;
    requests_.resize(num_recv + num_send, MPI_REQUEST_NULL);

    // Determine if staging is needed: device view without GPU-aware MPI
    const bool needs_staging = [&]() {
        if constexpr (detail::requires_staging_v<ViewType>) {
            return !Environment::is_gpu_aware_mpi();
        } else {
            return false;
        }
    }();

    if (needs_staging) {
        // ─── Staged path: allocate host buffers and bind persistent requests ─

        // Allocate persistent host send buffers
        using host_view_t = Kokkos::View<value_type *, Kokkos::HostSpace>;
        auto send_buffers = std::make_shared<std::vector<host_view_t>>();
        send_buffers->reserve(num_send);

        // Allocate persistent host receive buffers
        auto recv_buffers = std::make_shared<std::vector<host_view_t>>();
        recv_buffers->reserve(num_recv);

        // MPI_Recv_init for each receive neighbor (into host buffers)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto &neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            auto host_buf = host_view_t(Kokkos::view_alloc(std::string("persistent_recv_buf"), Kokkos::WithoutInitializing), neighbor.count);
            recv_buffers->push_back(host_buf);

            int rc = MPI_Recv_init(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests_[i]);

            if (rc != MPI_SUCCESS) {
                detail::handle_mpi_error(rc, neighbor.rank, "MPI_Recv_init");
            }

            recv_offset += neighbor.count;
        }

        // MPI_Send_init for each send neighbor (from host buffers)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto &neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            auto host_buf = host_view_t(Kokkos::view_alloc(std::string("persistent_send_buf"), Kokkos::WithoutInitializing), neighbor.count);
            send_buffers->push_back(host_buf);

            int rc =
                MPI_Send_init(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests_[num_recv + i]);

            if (rc != MPI_SUCCESS) {
                detail::handle_mpi_error(rc, neighbor.rank, "MPI_Send_init");
            }

            send_offset += neighbor.count;
        }

        // Capture pack function: device view → host send buffers
        const std::size_t total_send_elems = plan.total_send_elements();
        pack_fn_ = [send_buffers, &view, send_info, total_send_elems]() {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < send_info.size(); ++i) {
                const auto &neighbor = send_info[i];
                auto device_subview = Kokkos::subview(view, Kokkos::make_pair(offset, offset + neighbor.count));
                Kokkos::deep_copy((*send_buffers)[i], device_subview);
                offset += neighbor.count;
            }
        };

        // Capture unpack function: host recv buffers → device view
        unpack_fn_ = [recv_buffers, &view, recv_info, total_send_elems]() {
            std::size_t offset = total_send_elems;
            for (std::size_t i = 0; i < recv_info.size(); ++i) {
                const auto &neighbor = recv_info[i];
                auto device_subview = Kokkos::subview(view, Kokkos::make_pair(offset, offset + neighbor.count));
                Kokkos::deep_copy(device_subview, (*recv_buffers)[i]);
                offset += neighbor.count;
            }
        };
    } else {
        // ─── Direct path: bind persistent requests directly to view memory ──

        // MPI_Recv_init for each receive neighbor (directly into view)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto &neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            int rc =
                MPI_Recv_init(view.data() + recv_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests_[i]);

            if (rc != MPI_SUCCESS) {
                detail::handle_mpi_error(rc, neighbor.rank, "MPI_Recv_init");
            }

            recv_offset += neighbor.count;
        }

        // MPI_Send_init for each send neighbor (directly from view)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto &neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            int rc = MPI_Send_init(view.data() + send_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm,
                                   &requests_[num_recv + i]);

            if (rc != MPI_SUCCESS) {
                detail::handle_mpi_error(rc, neighbor.rank, "MPI_Send_init");
            }

            send_offset += neighbor.count;
        }
    }
}

inline Persistent_Halo_Handle::~Persistent_Halo_Handle() {
    free_requests_();
}

inline Persistent_Halo_Handle::Persistent_Halo_Handle(Persistent_Halo_Handle &&other) noexcept
    : requests_(std::move(other.requests_)),
      num_recv_requests_(other.num_recv_requests_),
      pack_fn_(std::move(other.pack_fn_)),
      unpack_fn_(std::move(other.unpack_fn_)) {
    other.requests_.clear();
    other.num_recv_requests_ = 0;
    other.pack_fn_ = nullptr;
    other.unpack_fn_ = nullptr;
}

inline Persistent_Halo_Handle &Persistent_Halo_Handle::operator=(Persistent_Halo_Handle &&other) noexcept {
    if (this != &other) {
        free_requests_();

        requests_ = std::move(other.requests_);
        num_recv_requests_ = other.num_recv_requests_;
        pack_fn_ = std::move(other.pack_fn_);
        unpack_fn_ = std::move(other.unpack_fn_);

        other.requests_.clear();
        other.num_recv_requests_ = 0;
        other.pack_fn_ = nullptr;
        other.unpack_fn_ = nullptr;
    }
    return *this;
}

inline void Persistent_Halo_Handle::start() {
    if (requests_.empty()) {
        return;
    }

    // Pack device→host if staging is active
    if (pack_fn_) {
        pack_fn_();
    }

    detail::Serialized_MPI_Guard guard;

    int rc = MPI_Startall(static_cast<int>(requests_.size()), requests_.data());

    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, -1, "MPI_Startall");
    }
}

inline void Persistent_Halo_Handle::wait() {
    if (requests_.empty()) {
        return;
    }

    detail::Serialized_MPI_Guard guard;

    int rc = MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);

    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, -1, "MPI_Waitall");
    }

    // Unpack host→device if staging is active
    if (unpack_fn_) {
        unpack_fn_();
    }
}

inline bool Persistent_Halo_Handle::test() {
    if (requests_.empty()) {
        return true;
    }

    detail::Serialized_MPI_Guard guard;

    int flag = 0;
    int rc = MPI_Testall(static_cast<int>(requests_.size()), requests_.data(), &flag, MPI_STATUSES_IGNORE);

    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, -1, "MPI_Testall");
    }

    if (flag) {
        // All complete — unpack if staging is active
        if (unpack_fn_) {
            unpack_fn_();
        }
        return true;
    }

    return false;
}

inline bool Persistent_Halo_Handle::empty() const noexcept {
    return requests_.empty();
}

inline void Persistent_Halo_Handle::free_requests_() noexcept {
    for (auto &req : requests_) {
        if (req != MPI_REQUEST_NULL) {
            MPI_Request_free(&req);
        }
    }
    requests_.clear();
}

}  // namespace halo

#endif  // HALO_PERSISTENT_HALO_HANDLE_HPP
