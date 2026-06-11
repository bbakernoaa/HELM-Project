// ─── LOGS MPI Interposition — Weak-Symbol Overrides ─────────────────────────
// Provides link-time interposition for MPI functions used by LOGS. When this
// translation unit is linked into a test executable, these definitions override
// the real MPI implementations.
//
// Each override:
//   1. Records the call in MPI_Spy (logs::testing::MPI_Spy)
//   2. Returns the configured values from the spy (rank, thread level, etc.)
//   3. Does NOT actually terminate for MPI_Abort — allowing in-process assertions
//
// This approach mirrors HALO's mpi_interposition.cpp: the test executable links
// this .cpp, and the linker resolves these symbols before the MPI library.
//
// Feature: helm-logs
// Requirements: 13.5, 13.6
// ─────────────────────────────────────────────────────────────────────────────

#include "mpi_interposition.hpp"

#include <mpi.h>

using logs::testing::MPI_Spy;

// ─── MPI Function Overrides ─────────────────────────────────────────────────
// These are extern "C" to match the MPI C API linkage.

extern "C" {

// ─── MPI_Comm_rank ──────────────────────────────────────────────────────────
// Override to return the configured rank from MPI_Spy and record the call.
int MPI_Comm_rank(MPI_Comm comm, int* rank) {
    return MPI_Spy::instance().record_comm_rank(comm, rank);
}

// ─── MPI_Query_thread ───────────────────────────────────────────────────────
// Override to return the configured thread level from MPI_Spy and record.
int MPI_Query_thread(int* provided) {
    return MPI_Spy::instance().record_query_thread(provided);
}

// ─── MPI_Initialized ────────────────────────────────────────────────────────
// Override to return the configured initialized state from MPI_Spy and record.
int MPI_Initialized(int* flag) {
    return MPI_Spy::instance().record_initialized(flag);
}

// ─── MPI_Abort ──────────────────────────────────────────────────────────────
// Override to record the abort call in MPI_Spy but NOT actually terminate.
// This allows tests to continue assertions after a FATAL log triggers the
// abort path. The spinning while(true) loop in the logger is handled by
// running the FATAL log on a separate thread.
int MPI_Abort(MPI_Comm comm, int errorcode) {
    return MPI_Spy::instance().record_abort(comm, errorcode);
}

// ─── MPI_Comm_size ──────────────────────────────────────────────────────────
// Override to return a deterministic size without requiring real MPI runtime.
int MPI_Comm_size(MPI_Comm comm, int* size) {
    if (size != nullptr) {
        *size = 4;
    }
    return MPI_SUCCESS;
}

// ─── MPI_Finalized ──────────────────────────────────────────────────────────
// Override to always report MPI as NOT finalized in test mode.
int MPI_Finalized(int* flag) {
    if (flag != nullptr) {
        *flag = 0;
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_test_inter ────────────────────────────────────────────────────
// Override to prevent internal MPI calls from failing. Returns "not inter-comm".
int MPI_Comm_test_inter(MPI_Comm comm, int* flag) {
    if (flag != nullptr) {
        *flag = 0;
    }
    return MPI_SUCCESS;
}

}  // extern "C"
