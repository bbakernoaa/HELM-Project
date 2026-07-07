// DAGR — logging.hpp
// Public header: configuration entry point for DAGR's shared LOGS logger.
//
// DAGR routes all of its diagnostics through a single LOGS Logger. By default
// that logger is unconfigured (rank sentinel, no sink -> stderr), which under
// MPI causes every rank to emit identical messages. Call configure_logging()
// once at startup to attach the MPI communicator (for correct [RANK:NNNN]
// stamping), install a stdout sink, and set the severity threshold. On
// non-root ranks pass a higher level (e.g. Log_Level::error) to suppress
// per-rank INFO/DEBUG spam while still surfacing FATAL diagnostics.
//
// Only <mpi.h> is exposed here; no lower-tier HELM headers leak through the
// public surface (Req 11.3).
#pragma once

#include <mpi.h>

namespace dagr {

/// Severity levels accepted by configure_logging(), mapped internally onto the
/// LOGS Severity_Level scale. FATAL is always emitted regardless of threshold.
enum class Log_Level {
    debug,    ///< Emit DEBUG and above.
    info,     ///< Emit INFO and above (typical for rank 0).
    warning,  ///< Emit WARNING and above.
    error,    ///< Emit ERROR and above (typical for non-root ranks).
    silent,   ///< Emit FATAL only.
};

/// Configure DAGR's shared logger.
///
/// Idempotent with respect to the stdout sink (installed at most once); the
/// communicator and threshold are (re)applied on every call. Safe to call
/// before or after MPI_Init — if the communicator is not yet valid, rank
/// stamping falls back to the sentinel until reconfigured.
///
/// @param comm  Communicator used for rank stamping (e.g. MPI_COMM_WORLD).
/// @param level Severity threshold for this rank.
void configure_logging(MPI_Comm comm, Log_Level level) noexcept;

}  // namespace dagr
