// --- Property-Based Tests: C-Interop Destroy Invalidates Handle --------------
// Feature: helm-halo-microlibrary, Property 23: Destroy Invalidates Handle
//
// Uses RapidCheck to verify the extern "C" interop layer
// (src/fortran/halo_c_interop.cpp) honours the destroy-invalidates contract:
//
//   For any opaque handle returned by a creation function, after calling the
//   corresponding destroy function (halo_destroy_plan_c or halo_destroy_comm_c),
//   any subsequent operation using that handle SHALL return HALO_ERR_BAD_HANDLE.
//
// The non-negotiable invariants verified here:
//   - Destroying a live handle returns HALO_SUCCESS (0).
//   - Every reuse of the same token AFTER destruction returns the non-zero
//     HALO_ERR_BAD_HANDLE code.
//   - No C++ exception ever escapes across the language boundary.
//
// Concretely we exercise:
//   (a) Plan lifecycle: create a valid Halo_Plan handle, destroy it, then
//       re-destroy / exchange-blocking / exchange-async on the stale token ->
//       all return HALO_ERR_BAD_HANDLE, never throw.
//   (b) Communicator lifecycle: register a Communicator handle, destroy it via
//       halo_destroy_comm_c, then reuse the stale token (re-destroy,
//       halo_comm_create_c, halo_plan_create_c) -> all return
//       HALO_ERR_BAD_HANDLE, never throw.
//
// These run single-rank with the MPI interposition spy (MPI_Comm_size mock
// returns 4), so no mpirun is required.
//
// **Validates: Requirements 14.13**
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "halo/communicator.hpp"
#include "handle_registry.hpp"
#include "mpi_interposition.hpp"

// --- extern "C" interop forward declarations ---------------------------------
// halo_c_interop.cpp defines these without a public header, so we declare the
// signatures here (matching the definitions exactly) to call across the boundary.

extern "C" {
int halo_init_c(int mpi_comm_int, int* comm_handle_out);
int halo_comm_create_c(int parent_handle, int color, int key,
                       int* child_handle_out);
int halo_plan_create_c(int comm_handle,
                       const int* send_ranks, const int* send_counts, int num_send,
                       const int* recv_ranks, const int* recv_counts, int num_recv,
                       int* plan_handle_out);
int halo_exchange_blocking_c(int plan_handle, void* data,
                             int num_elements, int element_size);
int halo_exchange_async_c(int plan_handle, void* data,
                          int num_elements, int element_size,
                          int* handle_out);
int halo_wait_c(int handle);
int halo_test_c(int handle, int* complete_out);
int halo_destroy_plan_c(int plan_handle);
int halo_destroy_comm_c(int comm_handle);
}

namespace {

// --- Error code mirror -------------------------------------------------------
// The Halo_Error enum lives in an anonymous namespace inside halo_c_interop.cpp
// and is not exported, so we mirror the contract values here for assertions.
constexpr int HALO_SUCCESS         = 0;
constexpr int HALO_ERR_INVALID_ARG = 1;
constexpr int HALO_ERR_BAD_HANDLE  = 4;

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

// --- Valid communicator handle (registered once, intentionally leaked) -------
// halo_plan_create_c needs a live Communicator referenced by an opaque handle.
// We heap-allocate a Communicator wrapping MPI_COMM_WORLD and register it in the
// process-global Handle_Registry. We never call halo_destroy_comm_c on it (which
// would `delete` the pointer), so the handle stays valid for every iteration and
// there is no double-free. The single leak is harmless for a test process.
int valid_comm_handle() {
    static int handle = [] {
        auto* comm = new halo::Communicator(MPI_COMM_WORLD);
        return halo::fortran::Handle_Registry::instance()
            .register_handle(static_cast<void*>(comm));
    }();
    return handle;
}

/// Generate a valid neighbor set: a random subset of unique ranks in
/// [0, MOCK_COMM_SIZE) with positive counts. Returns parallel (ranks, counts).
std::pair<std::vector<int>, std::vector<int>> gen_valid_neighbors() {
    int n = *rc::gen::inRange(0, MOCK_COMM_SIZE + 1);  // 0..4 neighbors
    std::vector<int> available{0, 1, 2, 3};
    std::vector<int> ranks;
    std::vector<int> counts;
    ranks.reserve(static_cast<std::size_t>(n));
    counts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
        ranks.push_back(available[static_cast<std::size_t>(idx)]);
        available.erase(available.begin() + idx);
        counts.push_back(*rc::gen::inRange(1, 101));
    }
    return {ranks, counts};
}

}  // anonymous namespace

// --- Property 23 (a): Destroying a plan invalidates its handle ---------------
// Feature: helm-halo-microlibrary, Property 23: Destroy Invalidates Handle
//
// For any valid neighbor topology, halo_plan_create_c yields a usable plan
// handle and halo_destroy_plan_c on it returns 0. After destruction, EVERY
// subsequent use of the same token (re-destroy, exchange-blocking,
// exchange-async) SHALL return HALO_ERR_BAD_HANDLE and SHALL NOT throw.
//
// **Validates: Requirements 14.13**

RC_GTEST_PROP(InteropProperty23, DestroyedPlanHandleReturnsBadHandle, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    auto [send_ranks, send_counts] = gen_valid_neighbors();
    auto [recv_ranks, recv_counts] = gen_valid_neighbors();

    // A real buffer for the exchange entry points (never dereferenced because
    // the bad-handle check returns before any data access).
    double buffer[MOCK_COMM_SIZE] = {};

    bool threw = false;
    int create_code = HALO_ERR_INVALID_ARG;
    int destroy_code = HALO_ERR_INVALID_ARG;
    int plan_handle = -1;

    // Reuse-after-destroy outcomes.
    int code_redestroy = HALO_SUCCESS;
    int code_exchange_blocking = HALO_SUCCESS;
    int code_exchange_async = HALO_SUCCESS;
    int async_handle = -1;

    try {
        create_code = halo_plan_create_c(
            comm,
            send_ranks.data(), send_counts.data(),
            static_cast<int>(send_ranks.size()),
            recv_ranks.data(), recv_counts.data(),
            static_cast<int>(recv_ranks.size()),
            &plan_handle);

        // Destroy the live plan; this must succeed and invalidate the token.
        if (create_code == HALO_SUCCESS) {
            destroy_code = halo_destroy_plan_c(plan_handle);

            // Every reuse of the now-stale token must report a bad handle.
            code_redestroy = halo_destroy_plan_c(plan_handle);
            code_exchange_blocking = halo_exchange_blocking_c(
                plan_handle, buffer, MOCK_COMM_SIZE,
                static_cast<int>(sizeof(double)));
            code_exchange_async = halo_exchange_async_c(
                plan_handle, buffer, MOCK_COMM_SIZE,
                static_cast<int>(sizeof(double)), &async_handle);
        }
    } catch (...) {
        threw = true;
    }

    // (1) No C++ exception escaped any extern "C" function.
    RC_ASSERT(!threw);

    // (2) Creation and the first destruction of a live handle both succeed.
    RC_ASSERT(create_code == HALO_SUCCESS);
    RC_ASSERT(destroy_code == HALO_SUCCESS);

    // (3) Every reuse of the destroyed token reports the bad handle (non-zero).
    RC_ASSERT(code_redestroy == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_exchange_blocking == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_exchange_async == HALO_ERR_BAD_HANDLE);

    // (4) Restated as the universal contract: reuse-after-destroy is non-zero.
    RC_ASSERT(code_redestroy != HALO_SUCCESS);
    RC_ASSERT(code_exchange_blocking != HALO_SUCCESS);
    RC_ASSERT(code_exchange_async != HALO_SUCCESS);
}

// --- Property 23 (b): Destroying a communicator invalidates its handle -------
// Feature: helm-halo-microlibrary, Property 23: Destroy Invalidates Handle
//
// For a Communicator handle registered in the registry, halo_destroy_comm_c
// returns 0 and invalidates the token. After destruction, EVERY subsequent use
// of the same token (re-destroy, halo_comm_create_c, halo_plan_create_c) SHALL
// return HALO_ERR_BAD_HANDLE and SHALL NOT throw.
//
// We register a *fresh* Communicator wrapping MPI_COMM_WORLD for each iteration
// (distinct from the shared valid_comm_handle()), so destroying it does not
// disturb other properties. Note: MPI_COMM_WORLD is predefined, so the
// Communicator destructor does not call MPI_Comm_free -- destruction simply
// deletes the wrapper and removes the registry entry.
//
// **Validates: Requirements 14.13**

RC_GTEST_PROP(InteropProperty23, DestroyedCommHandleReturnsBadHandle, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Register a fresh, per-iteration Communicator handle to destroy.
    int comm_handle = halo::fortran::Handle_Registry::instance()
        .register_handle(static_cast<void*>(
            new halo::Communicator(MPI_COMM_WORLD)));

    // Random auxiliary arguments for halo_comm_create_c.
    int color = *rc::gen::inRange(0, 8);
    int key = *rc::gen::inRange(0, 8);

    // Valid neighbor arrays for the post-destroy halo_plan_create_c attempt;
    // the failure must stem from the stale comm handle, not the topology.
    auto [send_ranks, send_counts] = gen_valid_neighbors();
    auto [recv_ranks, recv_counts] = gen_valid_neighbors();

    bool threw = false;
    int destroy_code = HALO_ERR_INVALID_ARG;

    // Reuse-after-destroy outcomes.
    int code_redestroy = HALO_SUCCESS;
    int code_comm_create = HALO_SUCCESS;
    int code_plan_create = HALO_SUCCESS;
    int child_handle = -1;
    int plan_handle = -1;

    try {
        // Destroy the live communicator; this must succeed and free the token.
        destroy_code = halo_destroy_comm_c(comm_handle);

        // Every reuse of the now-stale token must report a bad handle.
        code_redestroy = halo_destroy_comm_c(comm_handle);
        code_comm_create =
            halo_comm_create_c(comm_handle, color, key, &child_handle);
        code_plan_create = halo_plan_create_c(
            comm_handle,
            send_ranks.data(), send_counts.data(),
            static_cast<int>(send_ranks.size()),
            recv_ranks.data(), recv_counts.data(),
            static_cast<int>(recv_ranks.size()),
            &plan_handle);
    } catch (...) {
        threw = true;
    }

    // (1) No C++ exception escaped any extern "C" function.
    RC_ASSERT(!threw);

    // (2) Destroying the live communicator handle succeeds.
    RC_ASSERT(destroy_code == HALO_SUCCESS);

    // (3) Every reuse of the destroyed token reports the bad handle (non-zero).
    RC_ASSERT(code_redestroy == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_comm_create == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_plan_create == HALO_ERR_BAD_HANDLE);

    // (4) Restated as the universal contract: reuse-after-destroy is non-zero.
    RC_ASSERT(code_redestroy != HALO_SUCCESS);
    RC_ASSERT(code_comm_create != HALO_SUCCESS);
    RC_ASSERT(code_plan_create != HALO_SUCCESS);
}

// --- Kokkos Initialization ---------------------------------------------------
// The interop translation unit references Kokkos; initialize it once for the
// whole test binary so any code path that touches a Kokkos::View is safe.

class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

static auto* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
