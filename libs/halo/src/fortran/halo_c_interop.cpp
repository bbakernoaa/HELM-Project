/// @file halo_c_interop.cpp
/// @brief Extern "C" interop functions for Fortran iso_c_binding consumption.
///
/// This file implements the C interop layer that bridges the Fortran halo_mod
/// module to the C++ HALO core library. All functions follow a uniform contract:
///   - Return int error code: 0 = success, non-zero = error
///   - Output parameters passed as pointers (e.g., int* handle_out)
///   - All C++ exceptions caught at the boundary via the HALO_C_TRY macro
///   - MPI communicators accepted as int (Fortran integer compatible)
///   - Opaque handles managed via the Handle_Registry singleton
///
/// @note MPI_Comm_f2c is used to convert Fortran integer communicator values
/// to C MPI_Comm handles. This is compatible with both the Fortran 2008 MPI
/// binding (type(MPI_Comm)%mpi_val) and the legacy ESMF integer convention.
///
/// Requirements: 14.1, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.13

#include "handle_registry.hpp"

#include <halo/halo.hpp>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

/// @brief Error codes returned to Fortran callers.
///
/// These values match the constants defined in halo_mod.f90.
enum Halo_Error : int {
    HALO_SUCCESS         = 0,   ///< Operation completed successfully.
    HALO_ERR_INVALID_ARG = 1,   ///< Invalid argument (e.g., bad rank, null pointer).
    HALO_ERR_MPI         = 2,   ///< MPI operation failed.
    HALO_ERR_RUNTIME     = 3,   ///< Runtime error (e.g., MPI not initialized).
    HALO_ERR_BAD_HANDLE  = 4,   ///< Invalid or expired opaque handle token.
    HALO_ERR_UNKNOWN     = 99   ///< Unknown/unexpected error.
};

} // anonymous namespace

/// @brief Macro wrapping function bodies with exception-to-error-code translation.
///
/// Catches all C++ exceptions at the language boundary and maps them to integer
/// error codes. This ensures no C++ exception ever propagates into Fortran code,
/// which would cause undefined behavior.
///
/// Usage:
/// @code
///   int my_func_c(int arg, int* out) {
///       HALO_C_TRY(
///           // C++ code that may throw
///           *out = do_something(arg);
///       )
///   }
/// @endcode
#define HALO_C_TRY(body)                                    \
    try {                                                   \
        body;                                               \
        return HALO_SUCCESS;                                \
    } catch (const std::invalid_argument&) {                \
        return HALO_ERR_INVALID_ARG;                        \
    } catch (const std::runtime_error&) {                   \
        return HALO_ERR_RUNTIME;                            \
    } catch (...) {                                         \
        return HALO_ERR_UNKNOWN;                            \
    }

extern "C" {

/// @brief Initialize HALO and create root communicator from Fortran MPI_Comm integer.
///
/// Calls Environment::initialize() to query MPI thread support level, then
/// constructs a Communicator from the provided Fortran MPI communicator handle
/// (converted via MPI_Comm_f2c). The Communicator is registered in the handle
/// registry and its opaque token is returned to the Fortran caller.
///
/// @param mpi_comm_int    The integer value of the MPI communicator
///                        (from MPI_Comm%mpi_val or ESMF_VMGet).
/// @param comm_handle_out Output: opaque handle token for the created Communicator.
/// @return 0 on success, non-zero error code on failure.
int halo_init_c(int mpi_comm_int, int* comm_handle_out) {
    HALO_C_TRY(
        halo::Environment::initialize();
        MPI_Comm comm = MPI_Comm_f2c(mpi_comm_int);
        auto* c = new halo::Communicator(comm);
        *comm_handle_out = halo::fortran::Handle_Registry::instance()
                               .register_handle(static_cast<void*>(c));
    )
}

/// @brief Create a sub-communicator via MPI_Comm_split.
///
/// Looks up the parent Communicator by its opaque handle, calls split(color, key),
/// and registers the resulting child Communicator in the handle registry.
///
/// @param parent_handle    Opaque handle of the parent Communicator.
/// @param color            Split color (MPI_UNDEFINED yields NULL communicator).
/// @param key              Split key for rank ordering.
/// @param child_handle_out Output: opaque handle for the new sub-communicator.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if parent_handle is invalid.
int halo_comm_create_c(int parent_handle, int color, int key,
                       int* child_handle_out) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* parent = static_cast<halo::Communicator*>(reg.lookup(parent_handle));
        if (!parent) return HALO_ERR_BAD_HANDLE;
        auto child = parent->split(color, key);
        auto* c = new halo::Communicator(std::move(child));
        *child_handle_out = reg.register_handle(static_cast<void*>(c));
    )
}

/// @brief Create a Halo_Plan from neighbor rank and count arrays.
///
/// Constructs Neighbor_Info vectors from the provided C arrays and creates a
/// Halo_Plan validated against the communicator's size. The plan is registered
/// in the handle registry.
///
/// @param comm_handle     Opaque handle of the Communicator.
/// @param send_ranks      Array of send-neighbor ranks (length num_send).
/// @param send_counts     Array of per-neighbor send element counts (length num_send).
/// @param num_send        Number of send neighbors.
/// @param recv_ranks      Array of receive-neighbor ranks (length num_recv).
/// @param recv_counts     Array of per-neighbor receive element counts (length num_recv).
/// @param num_recv        Number of receive neighbors.
/// @param plan_handle_out Output: opaque handle for the created Halo_Plan.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if comm_handle is invalid,
///         HALO_ERR_INVALID_ARG if ranks are invalid or duplicated.
int halo_plan_create_c(int comm_handle,
                       const int* send_ranks, const int* send_counts, int num_send,
                       const int* recv_ranks, const int* recv_counts, int num_recv,
                       int* plan_handle_out) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* comm = static_cast<halo::Communicator*>(reg.lookup(comm_handle));
        if (!comm) return HALO_ERR_BAD_HANDLE;

        std::vector<halo::Neighbor_Info> sends(static_cast<std::size_t>(num_send));
        for (int i = 0; i < num_send; ++i) {
            sends[static_cast<std::size_t>(i)] = {
                send_ranks[i],
                static_cast<std::size_t>(send_counts[i])
            };
        }

        std::vector<halo::Neighbor_Info> recvs(static_cast<std::size_t>(num_recv));
        for (int i = 0; i < num_recv; ++i) {
            recvs[static_cast<std::size_t>(i)] = {
                recv_ranks[i],
                static_cast<std::size_t>(recv_counts[i])
            };
        }

        auto* plan = new halo::Halo_Plan(*comm, std::move(sends), std::move(recvs));
        *plan_handle_out = reg.register_handle(static_cast<void*>(plan));
    )
}

/// @brief Execute a blocking halo exchange on a contiguous Fortran array.
///
/// Constructs a non-owning Kokkos::View<char*, HostSpace, Unmanaged> over the
/// raw pointer from Fortran (obtained via c_loc). The view spans exactly
/// num_elements * element_size bytes. Then calls exchange_blocking with the
/// looked-up Halo_Plan.
///
/// @param plan_handle   Opaque handle of the Halo_Plan.
/// @param data          Pointer to contiguous array data (from Fortran c_loc).
/// @param num_elements  Total number of elements in the array.
/// @param element_size  Size of each element in bytes (e.g., 8 for real(8)).
/// @return 0 on success, HALO_ERR_BAD_HANDLE if plan_handle is invalid.
int halo_exchange_blocking_c(int plan_handle, void* data,
                             int num_elements, int element_size) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* plan = static_cast<halo::Halo_Plan*>(reg.lookup(plan_handle));
        if (!plan) return HALO_ERR_BAD_HANDLE;

        // Construct a non-owning Kokkos::View over the Fortran contiguous array.
        // The view treats the data as raw bytes (char*) with total size =
        // num_elements * element_size. This allows the exchange functions to
        // operate on the data without knowing the Fortran element type.
        auto view = Kokkos::View<char*, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            static_cast<char*>(data),
            static_cast<std::size_t>(num_elements) * static_cast<std::size_t>(element_size));

        halo::exchange_blocking(*plan, view);
    )
}

/// @brief Initiate a non-blocking halo exchange on a contiguous Fortran array.
///
/// Constructs a non-owning Kokkos::View over the raw pointer, initiates the
/// async exchange, and registers the resulting Halo_Handle in the registry.
///
/// @param plan_handle   Opaque handle of the Halo_Plan.
/// @param data          Pointer to contiguous array data (from Fortran c_loc).
/// @param num_elements  Total number of elements in the array.
/// @param element_size  Size of each element in bytes.
/// @param handle_out    Output: opaque handle for the Halo_Handle.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if plan_handle is invalid.
int halo_exchange_async_c(int plan_handle, void* data,
                          int num_elements, int element_size,
                          int* handle_out) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* plan = static_cast<halo::Halo_Plan*>(reg.lookup(plan_handle));
        if (!plan) return HALO_ERR_BAD_HANDLE;

        // Construct a non-owning Kokkos::View over the Fortran contiguous array.
        auto view = Kokkos::View<char*, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            static_cast<char*>(data),
            static_cast<std::size_t>(num_elements) * static_cast<std::size_t>(element_size));

        auto handle = halo::exchange_async(*plan, view);
        auto* h = new halo::Halo_Handle(std::move(handle));
        *handle_out = reg.register_handle(static_cast<void*>(h));
    )
}

/// @brief Wait for an async halo exchange to complete.
///
/// Looks up the Halo_Handle by its opaque token and calls wait(), which blocks
/// until all pending MPI operations complete and performs any post-receive
/// deep-copy if host-staged buffers exist.
///
/// @param handle  Opaque handle of the Halo_Handle.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if handle is invalid.
int halo_wait_c(int handle) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* h = static_cast<halo::Halo_Handle*>(reg.lookup(handle));
        if (!h) return HALO_ERR_BAD_HANDLE;
        h->wait();
    )
}

/// @brief Test if an async halo exchange has completed (non-blocking).
///
/// Looks up the Halo_Handle and calls test(). If all operations are complete,
/// sets complete_out to 1 and performs any post-receive deep-copy. Otherwise
/// sets complete_out to 0.
///
/// @param handle       Opaque handle of the Halo_Handle.
/// @param complete_out Output: 1 if all operations complete, 0 if pending.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if handle is invalid.
int halo_test_c(int handle, int* complete_out) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* h = static_cast<halo::Halo_Handle*>(reg.lookup(handle));
        if (!h) return HALO_ERR_BAD_HANDLE;
        *complete_out = h->test() ? 1 : 0;
    )
}

/// @brief Destroy a Halo_Plan and invalidate its opaque handle.
///
/// Releases the handle from the registry and deletes the underlying Halo_Plan
/// object. After this call, the handle token is no longer valid and any
/// subsequent use will return HALO_ERR_BAD_HANDLE.
///
/// @param plan_handle  Opaque handle of the Halo_Plan to destroy.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if plan_handle is invalid.
int halo_destroy_plan_c(int plan_handle) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* ptr = reg.release(plan_handle);
        if (!ptr) return HALO_ERR_BAD_HANDLE;
        delete static_cast<halo::Halo_Plan*>(ptr);
    )
}

/// @brief Destroy a Communicator and invalidate its opaque handle.
///
/// Releases the handle from the registry and deletes the underlying
/// Communicator object. The Communicator destructor will call MPI_Comm_free
/// if the handle is not predefined and MPI has not been finalized.
///
/// @param comm_handle  Opaque handle of the Communicator to destroy.
/// @return 0 on success, HALO_ERR_BAD_HANDLE if comm_handle is invalid.
int halo_destroy_comm_c(int comm_handle) {
    HALO_C_TRY(
        auto& reg = halo::fortran::Handle_Registry::instance();
        auto* ptr = reg.release(comm_handle);
        if (!ptr) return HALO_ERR_BAD_HANDLE;
        delete static_cast<halo::Communicator*>(ptr);
    )
}

} // extern "C"
