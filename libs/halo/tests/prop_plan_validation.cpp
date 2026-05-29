// --- Property-Based Tests: C-Interop Plan Creation Validates Neighbor Arrays -
// Feature: helm-halo-microlibrary, Property 24: Plan Creation Validates Neighbor Arrays
//
// Uses RapidCheck to verify the extern "C" interop layer
// (src/fortran/halo_c_interop.cpp) validates neighbor topology when creating a
// Halo_Plan:
//
//   For any set of send-neighbor ranks and receive-neighbor ranks passed to
//   halo_plan_create_c, if all ranks are in [0, comm_size) with no duplicates,
//   the function SHALL return HALO_SUCCESS and a valid plan handle; if any rank
//   is out of range or duplicated, the function SHALL return
//   HALO_ERR_INVALID_ARG without creating a plan.
//
// The non-negotiable invariants verified here:
//   - Valid neighbor arrays    -> HALO_SUCCESS (0) and a usable plan handle.
//   - Invalid neighbor arrays  -> HALO_ERR_INVALID_ARG (non-zero).
//   - No C++ exception ever escapes across the language boundary.
//
// Concretely we exercise:
//   (a) Valid send + valid recv -> halo_plan_create_c returns 0 and a usable
//       (positive) plan handle, which is then destroyed (halo_destroy_plan_c)
//       to avoid leaks. Never throws.
//   (b) An invalid list (an out-of-range rank -- negative or >= comm_size -- OR
//       a duplicate rank within a list) injected into send or recv (random),
//       with the other direction valid -> returns HALO_ERR_INVALID_ARG. Never
//       throws.
//
// These run single-rank with the MPI interposition spy (MPI_Comm_size mock
// returns 4), so no mpirun is required.
//
// **Validates: Requirements 14.12**
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

/// Generate an INVALID neighbor set whose defect is guaranteed to be detected
/// by Halo_Plan construction. The defect is EITHER:
///   (a) an out-of-range rank (negative, or >= MOCK_COMM_SIZE), or
///   (b) a duplicate rank within the list.
/// Returns parallel (ranks, counts) with positive counts so that the ONLY
/// reason for rejection is the rank topology, not the counts.
std::pair<std::vector<int>, std::vector<int>> gen_invalid_neighbors() {
    const bool use_duplicate = *rc::gen::arbitrary<bool>();

    std::vector<int> ranks;
    std::vector<int> counts;

    if (use_duplicate) {
        // A rank that appears twice in the same list -> std::invalid_argument.
        int r = *rc::gen::inRange(0, MOCK_COMM_SIZE);
        ranks = {r, r};
        counts = {*rc::gen::inRange(1, 101), *rc::gen::inRange(1, 101)};
    } else {
        // A rank outside [0, comm_size) -> std::invalid_argument. The invalid
        // rank is placed alongside a random (possibly empty) prefix of valid,
        // unique ranks so the defect is isolated to the out-of-range entry.
        const bool negative = *rc::gen::arbitrary<bool>();
        int invalid_rank = negative
                               ? *rc::gen::inRange(-1000, 0)
                               : *rc::gen::inRange(MOCK_COMM_SIZE,
                                                   MOCK_COMM_SIZE + 1000);

        // Optional leading valid, unique ranks to vary list length.
        int prefix_len = *rc::gen::inRange(0, MOCK_COMM_SIZE);
        std::vector<int> available{0, 1, 2, 3};
        for (int i = 0; i < prefix_len; ++i) {
            int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
            ranks.push_back(available[static_cast<std::size_t>(idx)]);
            available.erase(available.begin() + idx);
            counts.push_back(*rc::gen::inRange(1, 101));
        }
        ranks.push_back(invalid_rank);
        counts.push_back(*rc::gen::inRange(1, 101));
    }

    return {ranks, counts};
}

}  // anonymous namespace

// --- Property 24 (a): Valid neighbor arrays succeed --------------------------
// Feature: helm-halo-microlibrary, Property 24: Plan Creation Validates Neighbor Arrays
//
// For any valid send + valid recv neighbor topology (ranks unique and in
// [0, comm_size) with positive counts), halo_plan_create_c SHALL return
// HALO_SUCCESS (0) and a usable (positive) plan handle. The plan is then
// destroyed via halo_destroy_plan_c (which also returns 0) to avoid leaks.
// Neither call throws across the boundary.
//
// **Validates: Requirements 14.12**

RC_GTEST_PROP(InteropProperty24, ValidArraysSucceed, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    auto [send_ranks, send_counts] = gen_valid_neighbors();
    auto [recv_ranks, recv_counts] = gen_valid_neighbors();

    bool threw = false;
    int create_code = HALO_ERR_INVALID_ARG;
    int destroy_code = HALO_ERR_INVALID_ARG;
    int plan_handle = -1;

    try {
        create_code = halo_plan_create_c(
            comm,
            send_ranks.data(), send_counts.data(),
            static_cast<int>(send_ranks.size()),
            recv_ranks.data(), recv_counts.data(),
            static_cast<int>(recv_ranks.size()),
            &plan_handle);

        // A successful create returns 0; destroying the resulting plan also
        // returns 0 and frees the heap-allocated Halo_Plan.
        if (create_code == HALO_SUCCESS) {
            destroy_code = halo_destroy_plan_c(plan_handle);
        }
    } catch (...) {
        threw = true;
    }

    // RC_ASSERT cannot capture pointer operands; the plan handle is a plain int
    // token, so a simple positivity check is a value comparison.
    const bool handle_is_usable = (plan_handle > 0);

    // (1) No C++ exception escaped the boundary.
    RC_ASSERT(!threw);

    // (2) Valid topology -> HALO_SUCCESS (0) and a usable plan handle.
    RC_ASSERT(create_code == HALO_SUCCESS);
    RC_ASSERT(handle_is_usable);

    // (3) Destroying the freshly-created plan also succeeds (no leak).
    RC_ASSERT(destroy_code == HALO_SUCCESS);
}

// --- Property 24 (b): Invalid neighbor arrays are rejected -------------------
// Feature: helm-halo-microlibrary, Property 24: Plan Creation Validates Neighbor Arrays
//
// For any neighbor topology where exactly one direction (send or recv, chosen
// at random) carries an INVALID list -- a rank outside [0, comm_size) (negative
// or >= comm_size) OR a duplicated rank -- and the other direction is valid,
// halo_plan_create_c SHALL return HALO_ERR_INVALID_ARG (non-zero) without
// creating a plan, and SHALL NOT throw across the boundary.
//
// **Validates: Requirements 14.12**

RC_GTEST_PROP(InteropProperty24, InvalidArraysRejected, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    // Decide which direction carries the invalid list.
    const bool inject_in_send = *rc::gen::arbitrary<bool>();

    auto [bad_ranks, bad_counts] = gen_invalid_neighbors();
    auto [good_ranks, good_counts] = gen_valid_neighbors();

    const int* send_ranks;
    const int* send_counts;
    int num_send;
    const int* recv_ranks;
    const int* recv_counts;
    int num_recv;

    if (inject_in_send) {
        send_ranks = bad_ranks.data();
        send_counts = bad_counts.data();
        num_send = static_cast<int>(bad_ranks.size());
        recv_ranks = good_ranks.data();
        recv_counts = good_counts.data();
        num_recv = static_cast<int>(good_ranks.size());
    } else {
        send_ranks = good_ranks.data();
        send_counts = good_counts.data();
        num_send = static_cast<int>(good_ranks.size());
        recv_ranks = bad_ranks.data();
        recv_counts = bad_counts.data();
        num_recv = static_cast<int>(bad_ranks.size());
    }

    bool threw = false;
    int code = HALO_SUCCESS;
    int plan_handle = -1;

    try {
        code = halo_plan_create_c(comm, send_ranks, send_counts, num_send,
                                  recv_ranks, recv_counts, num_recv,
                                  &plan_handle);
    } catch (...) {
        threw = true;
    }

    // (1) No C++ exception escaped the boundary.
    RC_ASSERT(!threw);

    // (2) Invalid topology -> the dedicated invalid-argument code.
    RC_ASSERT(code == HALO_ERR_INVALID_ARG);

    // (3) Restated as the universal contract: rejection is non-zero.
    RC_ASSERT(code != HALO_SUCCESS);
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
