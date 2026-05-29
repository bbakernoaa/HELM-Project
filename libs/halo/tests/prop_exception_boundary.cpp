// --- Property-Based Tests: C-Interop Exception Boundary ----------------------
// Feature: helm-halo-microlibrary, Property 22: Exception Boundary Returns Error Code
//
// Uses RapidCheck to verify the extern "C" interop layer
// (src/fortran/halo_c_interop.cpp) honours its language-boundary contract:
//
//   For any C_Interop_Layer function call that triggers a C++ exception
//   (std::invalid_argument, std::runtime_error, or any other exception), the
//   function SHALL return a non-zero integer error code WITHOUT propagating the
//   exception across the language boundary; and for any valid operation that
//   completes successfully, the function SHALL return 0.
//
// The non-negotiable invariant verified here: every extern "C" function returns
// an int error code and NEVER lets a C++ exception escape. success == 0,
// failure != 0.
//
// Concretely we exercise:
//   (a) Invalid/bogus opaque handles  -> HALO_ERR_BAD_HANDLE (non-zero), no throw
//   (b) std::invalid_argument inside   -> HALO_ERR_INVALID_ARG (non-zero), no throw
//       (halo_plan_create_c with out-of-range or duplicate ranks)
//   (c) A successful operation         -> HALO_SUCCESS (0)
//       (halo_plan_create_c + halo_destroy_plan_c with valid inputs)
//
// These run single-rank with the MPI interposition spy (MPI_Comm_size mock
// returns 4), so no mpirun is required.
//
// **Validates: Requirements 14.9, 14.10**
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

// --- Property 22 (a): Bogus handles return HALO_ERR_BAD_HANDLE, never throw ---
// Feature: helm-halo-microlibrary, Property 22: Exception Boundary Returns Error Code
//
// For any opaque handle token that is not registered (negative or zero tokens
// are never issued by the registry), every handle-consuming extern "C" function
// SHALL return the non-zero HALO_ERR_BAD_HANDLE code and SHALL NOT throw across
// the boundary.
//
// **Validates: Requirements 14.9, 14.10**

RC_GTEST_PROP(InteropProperty22, BogusHandleReturnsBadHandleNeverThrows, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Tokens in [-100000, 0] are guaranteed invalid: the registry issues only
    // positive tokens and reserves 0 as HALO_HANDLE_INVALID.
    int bogus = *rc::gen::inRange(-100000, 1);

    // Random auxiliary arguments for the functions that take them.
    int color = *rc::gen::inRange(0, 8);
    int key = *rc::gen::inRange(0, 8);

    // A real buffer for the exchange entry points (never dereferenced because
    // the bad-handle check returns before any data access).
    double buffer[MOCK_COMM_SIZE] = {};

    bool threw = false;
    int child = -1;
    int async_handle = -1;
    int complete = -1;
    int plan_out = -1;

    int code_destroy_plan = 1;
    int code_destroy_comm = 1;
    int code_wait = 1;
    int code_test = 1;
    int code_comm_create = 1;
    int code_plan_create = 1;
    int code_exchange_blocking = 1;
    int code_exchange_async = 1;

    try {
        code_destroy_plan = halo_destroy_plan_c(bogus);
        code_destroy_comm = halo_destroy_comm_c(bogus);
        code_wait = halo_wait_c(bogus);
        code_test = halo_test_c(bogus, &complete);
        code_comm_create = halo_comm_create_c(bogus, color, key, &child);

        // Valid neighbor arrays; the failure must stem from the bogus comm handle.
        int send_ranks[1] = {0};
        int send_counts[1] = {1};
        int recv_ranks[1] = {1};
        int recv_counts[1] = {1};
        code_plan_create = halo_plan_create_c(
            bogus, send_ranks, send_counts, 1, recv_ranks, recv_counts, 1,
            &plan_out);

        code_exchange_blocking = halo_exchange_blocking_c(
            bogus, buffer, MOCK_COMM_SIZE, static_cast<int>(sizeof(double)));
        code_exchange_async = halo_exchange_async_c(
            bogus, buffer, MOCK_COMM_SIZE, static_cast<int>(sizeof(double)),
            &async_handle);
    } catch (...) {
        threw = true;
    }

    // (1) No C++ exception escaped any extern "C" function.
    RC_ASSERT(!threw);

    // (2) Every handle-consuming function reported the bad handle (non-zero).
    RC_ASSERT(code_destroy_plan == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_destroy_comm == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_wait == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_test == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_comm_create == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_plan_create == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_exchange_blocking == HALO_ERR_BAD_HANDLE);
    RC_ASSERT(code_exchange_async == HALO_ERR_BAD_HANDLE);

    // (3) Restated as the universal contract: failure codes are non-zero.
    RC_ASSERT(code_destroy_plan != HALO_SUCCESS);
    RC_ASSERT(code_destroy_comm != HALO_SUCCESS);
    RC_ASSERT(code_wait != HALO_SUCCESS);
    RC_ASSERT(code_test != HALO_SUCCESS);
    RC_ASSERT(code_comm_create != HALO_SUCCESS);
    RC_ASSERT(code_plan_create != HALO_SUCCESS);
    RC_ASSERT(code_exchange_blocking != HALO_SUCCESS);
    RC_ASSERT(code_exchange_async != HALO_SUCCESS);
}

// --- Property 22 (b): Internal std::invalid_argument -> non-zero, never throw -
// Feature: helm-halo-microlibrary, Property 22: Exception Boundary Returns Error Code
//
// For any halo_plan_create_c call whose neighbor topology triggers an internal
// std::invalid_argument (a rank outside [0, comm_size) or a duplicated rank),
// the function SHALL catch it, map it to the non-zero HALO_ERR_INVALID_ARG code,
// and SHALL NOT throw across the boundary.
//
// **Validates: Requirements 14.9, 14.10**

RC_GTEST_PROP(InteropProperty22, InvalidArgReturnsNonZeroNeverThrows, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    // Decide which direction carries the invalid list and what kind of defect.
    const bool inject_in_send = *rc::gen::arbitrary<bool>();
    const bool use_duplicate = *rc::gen::arbitrary<bool>();

    std::vector<int> bad_ranks;
    std::vector<int> bad_counts;

    if (use_duplicate) {
        // A rank that appears twice in the same list -> std::invalid_argument.
        int r = *rc::gen::inRange(0, MOCK_COMM_SIZE);
        bad_ranks = {r, r};
        bad_counts = {*rc::gen::inRange(1, 101), *rc::gen::inRange(1, 101)};
    } else {
        // A rank outside [0, comm_size) -> std::invalid_argument.
        bool negative = *rc::gen::arbitrary<bool>();
        int invalid_rank = negative
                               ? *rc::gen::inRange(-1000, 0)
                               : *rc::gen::inRange(MOCK_COMM_SIZE,
                                                   MOCK_COMM_SIZE + 1000);
        bad_ranks = {invalid_rank};
        bad_counts = {*rc::gen::inRange(1, 101)};
    }

    // The opposite direction is a valid (possibly empty) neighbor list.
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
    int plan_out = -1;
    try {
        code = halo_plan_create_c(comm, send_ranks, send_counts, num_send,
                                  recv_ranks, recv_counts, num_recv, &plan_out);
    } catch (...) {
        threw = true;
    }

    // No exception crossed the boundary, and the invalid argument was mapped to
    // the dedicated non-zero code.
    RC_ASSERT(!threw);
    RC_ASSERT(code == HALO_ERR_INVALID_ARG);
    RC_ASSERT(code != HALO_SUCCESS);
}

// --- Property 22 (c): Successful operations return HALO_SUCCESS (0) ----------
// Feature: helm-halo-microlibrary, Property 22: Exception Boundary Returns Error Code
//
// For any valid neighbor topology, halo_plan_create_c SHALL return 0 and yield a
// usable plan handle, and halo_destroy_plan_c on that handle SHALL also return 0.
// Neither call throws across the boundary.
//
// **Validates: Requirements 14.9, 14.10**

RC_GTEST_PROP(InteropProperty22, SuccessReturnsZeroNeverThrows, ()) {
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

    RC_ASSERT(!threw);
    RC_ASSERT(create_code == HALO_SUCCESS);
    RC_ASSERT(destroy_code == HALO_SUCCESS);
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
