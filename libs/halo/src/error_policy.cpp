#include "halo/error_policy.hpp"
#include "halo/environment.hpp"

#include <mpi.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace halo::detail {

namespace {

/// @brief Query the communicator name via MPI_Comm_get_name.
/// @return The communicator name if set and non-empty, or empty string.
std::string query_comm_name(MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) {
        return {};
    }
    char name[MPI_MAX_OBJECT_NAME];
    int resultlen = 0;
    int rc = MPI_Comm_get_name(comm, name, &resultlen);
    if (rc == MPI_SUCCESS && resultlen > 0) {
        return std::string(name, static_cast<std::size_t>(resultlen));
    }
    return {};
}

/// @brief Core error formatting and policy dispatch.
///
/// Builds the full diagnostic message from components and either throws
/// or aborts depending on the active ErrorPolicy.
[[noreturn]] void dispatch_error(int mpi_error_code,
                                 int neighbor_rank,
                                 const char* operation,
                                 const std::string& comm_name,
                                 const std::string& neighbor_summary) {
    // Format the MPI error string.
    char error_string[MPI_MAX_ERROR_STRING];
    int resultlen = 0;
    MPI_Error_string(mpi_error_code, error_string, &resultlen);

    // Query local rank for diagnostics context.
    int local_rank = -1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    }

    // Build the core message.
    std::string msg = std::string(operation) + " failed for rank " +
                      std::to_string(neighbor_rank) + ": " +
                      std::string(error_string, static_cast<std::size_t>(resultlen));

    // Prepend local rank prefix.
    if (local_rank >= 0) {
        msg = "[rank " + std::to_string(local_rank) + "] " + msg;
    }

    // Append communicator name if available.
    if (!comm_name.empty()) {
        msg += " | comm=\"" + comm_name + "\"";
    }

    // Append plan neighbor summary if available.
    if (!neighbor_summary.empty()) {
        msg += " | neighbors={" + neighbor_summary + "}";
    }

    const ErrorPolicy policy = Environment::error_policy();

    if (policy == ErrorPolicy::abort_with_diagnostics) {
        // Write full diagnostic context to stderr then abort.
        std::fprintf(stderr, "HALO FATAL: %s\n", msg.c_str());
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, mpi_error_code);
        // MPI_Abort may not return, but the compiler needs [[noreturn]] satisfied.
        std::abort();
    }

    // Default: throw_on_error
    throw std::runtime_error(msg);
}

} // anonymous namespace

[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation) {
    dispatch_error(mpi_error_code, neighbor_rank, operation, {}, {});
}

[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation,
                                   MPI_Comm comm) {
    std::string comm_name = query_comm_name(comm);
    dispatch_error(mpi_error_code, neighbor_rank, operation, comm_name, {});
}

[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation,
                                   MPI_Comm comm,
                                   const std::string& neighbor_summary) {
    std::string comm_name = query_comm_name(comm);
    dispatch_error(mpi_error_code, neighbor_rank, operation, comm_name, neighbor_summary);
}

std::string format_neighbor_summary(const int* send_ranks, std::size_t num_send,
                                    const int* recv_ranks, std::size_t num_recv) {
    std::string result = "send-to:[";
    for (std::size_t i = 0; i < num_send; ++i) {
        if (i > 0) result += ',';
        result += std::to_string(send_ranks[i]);
    }
    result += "] recv-from:[";
    for (std::size_t i = 0; i < num_recv; ++i) {
        if (i > 0) result += ',';
        result += std::to_string(recv_ranks[i]);
    }
    result += ']';
    return result;
}

} // namespace halo::detail
