#ifndef HALO_ERROR_POLICY_HPP
#define HALO_ERROR_POLICY_HPP

/// @file error_policy.hpp
/// @brief Configurable error policy for HALO MPI error handling.
///
/// Provides the ErrorPolicy enum and the detail::handle_mpi_error() function
/// that replaces the old detail::throw_mpi_error(). The active policy is
/// stored in Environment and defaults to throw_on_error (backward compatible).
///
/// When policy is throw_on_error: throws std::runtime_error (existing behavior).
/// When policy is abort_with_diagnostics: writes context to stderr then calls
/// MPI_Abort on MPI_COMM_WORLD (for Fortran callers where exceptions cannot
/// propagate across the language boundary).

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace halo {

/// @brief Error handling policy for MPI failures detected by HALO.
///
/// The default policy (throw_on_error) preserves backward compatibility with
/// existing C++ callers. The abort_with_diagnostics policy is required for the
/// Fortran C-interop layer where C++ exceptions cannot propagate.
enum class ErrorPolicy {
    throw_on_error,          ///< Throw std::runtime_error (default, C++ callers).
    abort_with_diagnostics   ///< Write diagnostics to stderr, then MPI_Abort.
};

// Forward declaration — Environment stores and exposes the active policy.
class Environment;

namespace detail {

/// @brief Handle an MPI error according to the active error policy.
///
/// When the policy is throw_on_error, throws std::runtime_error with the
/// formatted message. When abort_with_diagnostics, writes the message to
/// stderr and calls MPI_Abort(MPI_COMM_WORLD, mpi_error_code).
///
/// @param mpi_error_code The MPI error code returned by the failing call.
/// @param neighbor_rank  The rank of the neighbor involved in the failure.
/// @param operation      Description of the failing operation (e.g., "MPI_Irecv").
[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation);

/// @brief Handle an MPI error with communicator context.
///
/// Enhanced overload that queries MPI_Comm_get_name for the communicator and
/// includes it in the error message. When comm is MPI_COMM_NULL, behaves
/// identically to the 3-parameter version.
///
/// @param mpi_error_code The MPI error code returned by the failing call.
/// @param neighbor_rank  The rank of the neighbor involved in the failure.
/// @param operation      Description of the failing operation (e.g., "MPI_Irecv").
/// @param comm           The communicator on which the failure occurred.
[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation,
                                   MPI_Comm comm);

/// @brief Handle an MPI error with communicator and plan neighbor context.
///
/// Full-context overload that includes communicator name and a plan neighbor
/// summary (send-to and recv-from rank lists) in the diagnostic output.
///
/// @param mpi_error_code    The MPI error code returned by the failing call.
/// @param neighbor_rank     The rank of the neighbor involved in the failure.
/// @param operation         Description of the failing operation.
/// @param comm              The communicator on which the failure occurred.
/// @param neighbor_summary  A string summarizing plan neighbors, e.g.,
///                          "send-to:[1,2,3] recv-from:[0,2,3]"
[[noreturn]] void handle_mpi_error(int mpi_error_code,
                                   int neighbor_rank,
                                   const char* operation,
                                   MPI_Comm comm,
                                   const std::string& neighbor_summary);

/// @brief Format a plan neighbor summary string from send/recv rank lists.
///
/// Produces a string like "send-to:[1,2,3] recv-from:[0,2,3]" suitable for
/// inclusion in error messages.
///
/// @param send_ranks  Array of send-neighbor ranks.
/// @param num_send    Number of send neighbors.
/// @param recv_ranks  Array of recv-neighbor ranks.
/// @param num_recv    Number of recv neighbors.
/// @return Formatted neighbor summary string.
std::string format_neighbor_summary(const int* send_ranks, std::size_t num_send,
                                    const int* recv_ranks, std::size_t num_recv);

} // namespace detail
} // namespace halo

#endif // HALO_ERROR_POLICY_HPP
