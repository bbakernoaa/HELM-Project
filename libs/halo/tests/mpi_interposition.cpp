// ─── HALO MPI Interposition — Weak-Symbol Overrides ─────────────────────────
// Provides link-time interposition for target MPI functions. When this
// translation unit is linked into a test executable, these definitions override
// the real MPI implementations (which are declared as weak symbols in most MPI
// implementations, or are overridden via linker priority).
//
// Each override:
//   1. Records the call in MPI_Spy
//   2. Checks for injected errors (returns injected code if set)
//   3. Returns MPI_SUCCESS for mocked mode (no delegation to real MPI)
//
// This approach avoids needing LD_PRELOAD or PMPI — the test executable simply
// links this .cpp, and the linker resolves these symbols before the MPI library.
//
// Feature: helm-halo-microlibrary
// Requirements: 11.7
// ─────────────────────────────────────────────────────────────────────────────

#include "mpi_interposition.hpp"

#include <mpi.h>

#include <cstring>

using halo::testing::MPI_Call_Record;
using halo::testing::MPI_Spy;

// ─── Helper: Cast MPI handle pointer to void* for recording ─────────────────
namespace {

inline void* as_handle(MPI_Comm* p) { return static_cast<void*>(p); }
inline void* as_handle(MPI_Request* p) { return static_cast<void*>(p); }
inline void* as_handle(MPI_Win* p) { return static_cast<void*>(p); }

}  // anonymous namespace

// ─── MPI Function Overrides ─────────────────────────────────────────────────
// These are extern "C" to match the MPI C API linkage.

extern "C" {

// ─── MPI_Finalized ──────────────────────────────────────────────────────────
// Override MPI_Finalized to always report MPI as NOT finalized in test mode.
// This ensures RAII destructors proceed with their cleanup logic.
int MPI_Finalized(int* flag) {
    if (flag != nullptr) {
        *flag = 0;  // MPI is never "finalized" in test mode
    }
    return MPI_SUCCESS;
}

// ─── MPI_Initialized ────────────────────────────────────────────────────────
// Override MPI_Initialized to always report MPI as initialized in test mode.
// This allows Environment::initialize() to proceed without a real MPI runtime.
int MPI_Initialized(int* flag) {
    if (flag != nullptr) {
        *flag = 1;  // MPI is always "initialized" in test mode
    }
    return MPI_SUCCESS;
}

// ─── MPI_Query_thread ───────────────────────────────────────────────────────
// Override MPI_Query_thread to return the configured thread level from MPI_Spy.
// Records the call for verification in property tests.
int MPI_Query_thread(int* provided) {
    auto& spy = MPI_Spy::instance();
    spy.record_query_thread();
    if (provided != nullptr) {
        *provided = spy.thread_level();
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_test_inter ─────────────────────────────────────────────────────
// Override to prevent OpenMPI internal calls from aborting due to MPI not being
// initialized. Returns "not an intercommunicator" for all handles.
int MPI_Comm_test_inter(MPI_Comm comm, int* flag) {
    if (flag != nullptr) {
        *flag = 0;  // Not an intercommunicator
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_rank ──────────────────────────────────────────────────────────
// Override to return a deterministic rank without requiring real MPI runtime.
int MPI_Comm_rank(MPI_Comm comm, int* rank) {
    if (rank != nullptr) {
        *rank = 0;  // Always rank 0 in mock mode
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_size ──────────────────────────────────────────────────────────
// Override to return a deterministic size without requiring real MPI runtime.
int MPI_Comm_size(MPI_Comm comm, int* size) {
    if (size != nullptr) {
        *size = 4;  // Always 4 processes in mock mode
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_split ─────────────────────────────────────────────────────────
// Override to return a synthetic communicator without requiring real MPI runtime.
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm) {
    if (newcomm != nullptr) {
        if (color == MPI_UNDEFINED) {
            *newcomm = MPI_COMM_NULL;
        } else {
            // Return a synthetic non-null communicator
            static char split_sentinel[256];
            static int split_idx = 0;
            *newcomm = reinterpret_cast<MPI_Comm>(&split_sentinel[split_idx++ % 256]);
        }
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_dup ───────────────────────────────────────────────────────────
// Override to return a synthetic communicator without requiring real MPI runtime.
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm* newcomm) {
    if (newcomm != nullptr) {
        // Return a synthetic non-null communicator
        static char dup_sentinel[256];
        static int dup_idx = 0;
        *newcomm = reinterpret_cast<MPI_Comm>(&dup_sentinel[dup_idx++ % 256]);
    }
    return MPI_SUCCESS;
}

// ─── MPI_Error_string ───────────────────────────────────────────────────────
// Override to provide error strings without requiring real MPI runtime.
int MPI_Error_string(int errorcode, char* string, int* resultlen) {
    const char* msg = "Mock MPI error";
    if (string != nullptr) {
        std::strncpy(string, msg, MPI_MAX_ERROR_STRING);
    }
    if (resultlen != nullptr) {
        *resultlen = static_cast<int>(std::strlen(msg));
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_get_attr ──────────────────────────────────────────────────────
// Override to provide MPI_TAG_UB without requiring real MPI runtime.
int MPI_Comm_get_attr(MPI_Comm comm, int keyval, void* attribute_val, int* flag) {
    if (keyval == MPI_TAG_UB) {
        static int tag_ub = 32767;  // Standard minimum MPI_TAG_UB
        if (attribute_val != nullptr) {
            *static_cast<int**>(attribute_val) = &tag_ub;
        }
        if (flag != nullptr) {
            *flag = 1;
        }
    } else {
        if (flag != nullptr) {
            *flag = 0;
        }
    }
    return MPI_SUCCESS;
}

// ─── MPI_Comm_free ──────────────────────────────────────────────────────────
int MPI_Comm_free(MPI_Comm* comm) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Comm_free, as_handle(comm));
    if (err != MPI_SUCCESS) return err;

    // Simulate the free: set handle to MPI_COMM_NULL
    if (comm != nullptr) {
        *comm = MPI_COMM_NULL;
    }
    return MPI_SUCCESS;
}

// ─── MPI_Request_free ───────────────────────────────────────────────────────
int MPI_Request_free(MPI_Request* request) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Request_free, as_handle(request));
    if (err != MPI_SUCCESS) return err;

    // Simulate the free: set handle to MPI_REQUEST_NULL
    if (request != nullptr) {
        *request = MPI_REQUEST_NULL;
    }
    return MPI_SUCCESS;
}

// ─── MPI_Cancel ─────────────────────────────────────────────────────────────
int MPI_Cancel(MPI_Request* request) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Cancel, as_handle(request));
    if (err != MPI_SUCCESS) return err;

    // Cancel is a no-op in mock mode (request remains valid until freed)
    return MPI_SUCCESS;
}

// ─── MPI_Wait ───────────────────────────────────────────────────────────────
int MPI_Wait(MPI_Request* request, MPI_Status* status) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Wait, as_handle(request));
    if (err != MPI_SUCCESS) return err;

    // Simulate completion: set request to MPI_REQUEST_NULL
    if (request != nullptr) {
        *request = MPI_REQUEST_NULL;
    }
    // Fill status with empty/success if provided
    if (status != MPI_STATUS_IGNORE && status != nullptr) {
        std::memset(status, 0, sizeof(MPI_Status));
    }
    return MPI_SUCCESS;
}

// ─── MPI_Test ───────────────────────────────────────────────────────────────
int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Test, as_handle(request));
    if (err != MPI_SUCCESS) return err;

    // In mock mode, operations always complete immediately
    if (flag != nullptr) {
        *flag = 1;  // completed
    }
    if (request != nullptr) {
        *request = MPI_REQUEST_NULL;
    }
    if (status != MPI_STATUS_IGNORE && status != nullptr) {
        std::memset(status, 0, sizeof(MPI_Status));
    }
    return MPI_SUCCESS;
}

// ─── MPI_Win_free ───────────────────────────────────────────────────────────
int MPI_Win_free(MPI_Win* win) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Win_free, as_handle(win));
    if (err != MPI_SUCCESS) return err;

    // Simulate the free: set handle to MPI_WIN_NULL
    if (win != nullptr) {
        *win = MPI_WIN_NULL;
    }
    return MPI_SUCCESS;
}

// ─── MPI_Win_fence ──────────────────────────────────────────────────────────
int MPI_Win_fence(int assert_arg, MPI_Win win) {
    // Record with the assertion value as the arg field.
    // MPI_Win is a pointer type in OpenMPI, so cast directly to void*.
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Win_fence,
        static_cast<void*>(win),
        assert_arg);
    if (err != MPI_SUCCESS) return err;

    return MPI_SUCCESS;
}

// ─── Synthetic Request Sentinels ─────────────────────────────────────────────
// OpenMPI uses pointers for MPI_Request. We allocate small sentinel objects
// so that our synthetic requests are valid, distinct, non-null pointers that
// won't collide with real MPI internals.
namespace {

// Pool of sentinel bytes used as fake request handles.
// Each Irecv/Isend gets a unique address from this pool.
constexpr int SENTINEL_POOL_SIZE = 4096;
alignas(16) char sentinel_pool[SENTINEL_POOL_SIZE];
int sentinel_index = 0;

MPI_Request next_sentinel_request() {
    int idx = sentinel_index++ % SENTINEL_POOL_SIZE;
    return reinterpret_cast<MPI_Request>(&sentinel_pool[idx]);
}

}  // anonymous namespace

// ─── MPI_Irecv ──────────────────────────────────────────────────────────────
int MPI_Irecv(void* buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request* request) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Irecv, buf, source);
    if (err != MPI_SUCCESS) return err;

    // Provide a non-null sentinel request so RAII wrappers can track it
    if (request != nullptr) {
        *request = next_sentinel_request();
    }
    return MPI_SUCCESS;
}

// ─── MPI_Isend ──────────────────────────────────────────────────────────────
int MPI_Isend(const void* buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Isend, const_cast<void*>(buf), dest);
    if (err != MPI_SUCCESS) return err;

    // Provide a non-null sentinel request value
    if (request != nullptr) {
        *request = next_sentinel_request();
    }
    return MPI_SUCCESS;
}

// ─── MPI_Waitall ────────────────────────────────────────────────────────────
int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]) {
    int err = MPI_Spy::instance().record(
        MPI_Call_Record::Type::Waitall,
        static_cast<void*>(array_of_requests), count);
    if (err != MPI_SUCCESS) return err;

    // Simulate completion: set all requests to MPI_REQUEST_NULL
    if (array_of_requests != nullptr) {
        for (int i = 0; i < count; ++i) {
            array_of_requests[i] = MPI_REQUEST_NULL;
        }
    }
    // Fill statuses if provided
    if (array_of_statuses != MPI_STATUSES_IGNORE && array_of_statuses != nullptr) {
        std::memset(array_of_statuses, 0, count * sizeof(MPI_Status));
    }
    return MPI_SUCCESS;
}

}  // extern "C"
