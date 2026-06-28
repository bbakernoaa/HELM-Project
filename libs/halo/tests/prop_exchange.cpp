// ─── Property-Based Tests: halo::exchange_blocking ───────────────────────────
// Feature: helm-halo-microlibrary
//
// Uses RapidCheck to verify exchange_blocking properties using the MPI spy
// interposition layer for deterministic call-sequence verification.
//
// Property 12: MPI Tag Determinism and Boundedness
//   Validates: Requirements 6.2
//
// Property 13: Exchange Posts All Receives Before Any Send
//   Validates: Requirements 6.1
//
// Property 16: MPI Errors During Exchange Throw With Context
//   Validates: Requirements 6.6, 7.8
//
// Property 17: Async Exchange Handle Owns Correct Request Count
//   Validates: Requirements 7.1, 7.2
//
// Property 18: Completion Triggers Post-Receive Deep-Copy When Staged
//   Validates: Requirements 7.3, 7.4
//
// Property 19: Halo_Handle Destructor Ensures Completion
//   Validates: Requirements 7.5
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/exchange.hpp"
#include "halo/halo_plan.hpp"
#include "mpi_interposition.hpp"

// ─── RapidCheck Generators ───────────────────────────────────────────────────

namespace {

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

/// Generate a non-empty valid Neighbor_Info list: unique ranks in [0, comm_size),
/// random counts > 0. Guarantees at least 1 neighbor.
rc::Gen<std::vector<halo::Neighbor_Info>> genNonEmptyNeighborList() {
    return rc::gen::exec([]() -> std::vector<halo::Neighbor_Info> {
        // Generate 1 to MOCK_COMM_SIZE neighbors (at least 1)
        int num_neighbors = *rc::gen::inRange(1, MOCK_COMM_SIZE + 1);

        // Create all possible ranks and pick a random subset
        std::vector<int> available(MOCK_COMM_SIZE);
        std::iota(available.begin(), available.end(), 0);

        std::vector<int> selected;
        selected.reserve(num_neighbors);

        for (int i = 0; i < num_neighbors; ++i) {
            int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
            selected.push_back(available[idx]);
            available.erase(available.begin() + idx);
        }

        // Generate random counts for each selected rank
        std::vector<halo::Neighbor_Info> neighbors;
        neighbors.reserve(selected.size());
        for (int rank : selected) {
            std::size_t count = static_cast<std::size_t>(*rc::gen::inRange(1, 101));
            neighbors.push_back({rank, count});
        }

        return neighbors;
    });
}

}  // anonymous namespace

// ─── Property 12: MPI Tag Determinism and Boundedness ────────────────────────
// Feature: helm-halo-microlibrary, Property 12: MPI Tag Determinism and Boundedness
//
// For any (sender_rank, receiver_rank, comm_size) triple where both ranks are
// in [0, comm_size), the computed MPI tag SHALL be deterministic (same inputs
// always produce same output) and SHALL be in the range [0, MPI_TAG_UB).
//
// **Validates: Requirements 6.2**

RC_GTEST_PROP(ExchangeProperty12, TagDeterminismAndBoundedness, ()) {
    // The mock MPI_Comm_get_attr returns MPI_TAG_UB = 32767
    constexpr int MOCK_TAG_UB = 32767;

    // Generate a random comm_size in [2, 10000] (at least 2 ranks for exchange)
    int comm_size = *rc::gen::inRange(2, 10001);

    // Generate sender and receiver ranks in [0, comm_size)
    int sender = *rc::gen::inRange(0, comm_size);
    int receiver = *rc::gen::inRange(0, comm_size);

    // Compute the tag
    int tag1 = halo::detail::compute_tag(sender, receiver, comm_size);

    // Property 1: Boundedness — tag must be in [0, MPI_TAG_UB)
    RC_ASSERT(tag1 >= 0);
    RC_ASSERT(tag1 < MOCK_TAG_UB);

    // Property 2: Determinism — same inputs must produce same output
    int tag2 = halo::detail::compute_tag(sender, receiver, comm_size);
    RC_ASSERT(tag1 == tag2);

    // Additional determinism check: call a third time to be thorough
    int tag3 = halo::detail::compute_tag(sender, receiver, comm_size);
    RC_ASSERT(tag1 == tag3);
}

// ─── Property 13: Exchange Posts All Receives Before Any Send ─────────────────
// Feature: helm-halo-microlibrary, Property 13: Exchange Posts All Receives Before Any Send
//
// For any Halo_Plan with at least one send-neighbor and one receive-neighbor,
// exchange_blocking SHALL post all MPI_Irecv calls before posting any MPI_Isend
// call.
//
// **Validates: Requirements 6.1**

RC_GTEST_PROP(ExchangeProperty13, AllReceivesBeforeAnySend, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    // Preconditions: both lists must be non-empty
    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction (which may call MPI functions)
    spy.reset();

    // Execute the blocking exchange
    halo::exchange_blocking(plan, view);

    // Inspect the MPI_Spy call records
    auto const &calls = spy.calls();

    // Find the index of the LAST MPI_Irecv call
    int last_irecv_index = -1;
    // Find the index of the FIRST MPI_Isend call
    int first_isend_index = -1;

    for (int i = 0; i < static_cast<int>(calls.size()); ++i) {
        if (calls[i].type == halo::testing::MPI_Call_Record::Type::Irecv) {
            last_irecv_index = i;
        }
        if (calls[i].type == halo::testing::MPI_Call_Record::Type::Isend) {
            if (first_isend_index == -1) {
                first_isend_index = i;
            }
        }
    }

    // Both Irecv and Isend must have been called
    RC_ASSERT(last_irecv_index >= 0);
    RC_ASSERT(first_isend_index >= 0);

    // The last Irecv must appear BEFORE the first Isend
    RC_ASSERT(last_irecv_index < first_isend_index);
}

// ─── Property 13b: Correct count of Irecv and Isend calls ───────────────────
// Feature: helm-halo-microlibrary, Property 13: Exchange Posts All Receives Before Any Send
//
// Verify that the number of MPI_Irecv calls equals the number of receive
// neighbors and the number of MPI_Isend calls equals the number of send
// neighbors.
//
// **Validates: Requirements 6.1**

RC_GTEST_PROP(ExchangeProperty13, CorrectIrecvAndIsendCounts, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction
    spy.reset();

    halo::exchange_blocking(plan, view);

    // Count Irecv and Isend calls
    std::size_t irecv_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Irecv);
    std::size_t isend_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Isend);

    RC_ASSERT(irecv_count == recv_neighbors.size());
    RC_ASSERT(isend_count == send_neighbors.size());
}

// ─── Property 17: Async Exchange Handle Owns Correct Request Count ───────────
// Feature: helm-halo-microlibrary, Property 17: Async Exchange Handle Owns Correct Request Count
//
// For any Halo_Plan with S send-neighbors and R receive-neighbors,
// exchange_async SHALL return a Halo_Handle owning exactly S + R
// Request_Guard objects. We verify via MPI spy that exactly S Isend + R Irecv
// calls were made, and that the returned handle is not empty().
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(ExchangeProperty17, AsyncHandleOwnsCorrectRequestCount, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    // Preconditions: both lists must be non-empty
    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    const std::size_t S = send_neighbors.size();
    const std::size_t R = recv_neighbors.size();

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction (which may call MPI functions)
    spy.reset();

    // Execute the async exchange
    auto handle = halo::exchange_async(plan, view);

    // Verify the handle is NOT empty (it owns pending operations)
    RC_ASSERT(!handle.empty());

    // Verify via spy that exactly R Irecv calls were made
    std::size_t irecv_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Irecv);
    RC_ASSERT(irecv_count == R);

    // Verify via spy that exactly S Isend calls were made
    std::size_t isend_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Isend);
    RC_ASSERT(isend_count == S);

    // Total requests owned by the handle should be S + R
    // (verified indirectly: the spy recorded exactly S + R non-blocking calls)
    RC_ASSERT(irecv_count + isend_count == S + R);
}

// ─── Property 19: Halo_Handle Destructor Ensures Completion ──────────────────
// Feature: helm-halo-microlibrary, Property 19: Halo_Handle Destructor Ensures Completion
//
// For any Halo_Plan with S send-neighbors and R receive-neighbors, destroying
// a Halo_Handle returned by exchange_async WITHOUT calling wait() SHALL invoke
// MPI_Wait for each pending request (S + R times total), ensuring all MPI
// operations complete before resources are released.
//
// **Validates: Requirements 7.5**

RC_GTEST_PROP(ExchangeProperty19, DestructorEnsuresCompletion, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    const std::size_t S = send_neighbors.size();
    const std::size_t R = recv_neighbors.size();

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction (which may call MPI functions)
    spy.reset();

    // Call exchange_async and let the handle go out of scope WITHOUT calling wait()
    {
        auto handle = halo::exchange_async(plan, view);
        // DO NOT call handle.wait() — let the destructor handle completion
    }
    // At this point, the Halo_Handle destructor has been called

    // Count MPI_Wait calls recorded by the spy
    std::size_t wait_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Wait);

    // The destructor must have called wait() on each Request_Guard,
    // resulting in exactly S + R MPI_Wait calls (one per request)
    RC_ASSERT(wait_count == S + R);
}

// ─── Property 16: MPI Errors During Exchange Throw With Context ──────────────
// Feature: helm-halo-microlibrary, Property 16: MPI Errors During Exchange Throw With Context
//
// For any MPI send or receive operation that returns a non-success error code
// during a halo exchange, the exchange function SHALL throw std::runtime_error
// containing both the MPI error string and the failing neighbor rank.
//
// **Validates: Requirements 6.6, 7.8**

RC_GTEST_PROP(ExchangeProperty16, BlockingMpiErrorThrowsWithContext, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction
    spy.reset();

    // Generate a random non-success MPI error code (non-zero)
    int error_code = *rc::gen::inRange(1, 100);

    // Inject the error so the NEXT MPI call (first MPI_Irecv) will fail
    spy.set_next_error(error_code);

    // The first recv neighbor's rank should appear in the error message
    int expected_failing_rank = recv_neighbors[0].rank;

    // Execute exchange_blocking — should throw std::runtime_error
    bool caught_runtime_error = false;
    std::string error_message;

    try {
        halo::exchange_blocking(plan, view);
    } catch (const std::runtime_error &e) {
        caught_runtime_error = true;
        error_message = e.what();
    }

    // Verify std::runtime_error was thrown
    RC_ASSERT(caught_runtime_error);

    // Verify the error message contains the MPI error string
    // (MPI_Error_string mock returns "Mock MPI error")
    RC_ASSERT(error_message.find("Mock MPI error") != std::string::npos);

    // Verify the error message contains the failing neighbor rank
    RC_ASSERT(error_message.find(std::to_string(expected_failing_rank)) != std::string::npos);
}

RC_GTEST_PROP(ExchangeProperty16, AsyncMpiErrorThrowsWithContext, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction
    spy.reset();

    // Generate a random non-success MPI error code (non-zero)
    int error_code = *rc::gen::inRange(1, 100);

    // Inject the error so the NEXT MPI call (first MPI_Irecv) will fail
    spy.set_next_error(error_code);

    // The first recv neighbor's rank should appear in the error message
    int expected_failing_rank = recv_neighbors[0].rank;

    // Execute exchange_async — should throw std::runtime_error
    bool caught_runtime_error = false;
    std::string error_message;

    try {
        auto handle = halo::exchange_async(plan, view);
    } catch (const std::runtime_error &e) {
        caught_runtime_error = true;
        error_message = e.what();
    }

    // Verify std::runtime_error was thrown
    RC_ASSERT(caught_runtime_error);

    // Verify the error message contains the MPI error string
    RC_ASSERT(error_message.find("Mock MPI error") != std::string::npos);

    // Verify the error message contains the failing neighbor rank
    RC_ASSERT(error_message.find(std::to_string(expected_failing_rank)) != std::string::npos);
}

RC_GTEST_PROP(ExchangeProperty16, IsendErrorThrowsWithContext, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate non-empty send and receive neighbor lists
    auto send_neighbors = *genNonEmptyNeighborList();
    auto recv_neighbors = *genNonEmptyNeighborList();

    RC_PRE(!send_neighbors.empty());
    RC_PRE(!recv_neighbors.empty());

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct a valid Halo_Plan
    halo::Halo_Plan plan(comm, send_neighbors, recv_neighbors);

    // Allocate a Kokkos host view large enough for all send + recv data
    std::size_t total_elements = plan.total_send_elements() + plan.total_recv_elements();
    Kokkos::View<double *, Kokkos::HostSpace> view("test_view", total_elements);

    // Reset spy after plan construction
    spy.reset();

    // Generate a random non-success MPI error code (non-zero)
    int error_code = *rc::gen::inRange(1, 100);

    // We want the error to hit on the first MPI_Isend, not MPI_Irecv.
    // The spy consumes the error on the NEXT call. We need to let all Irecv
    // calls succeed first, then inject the error before the first Isend.
    // Strategy: execute exchange_blocking, but inject error after all Irecvs.
    // Since set_next_error only affects the NEXT call, we need to time it.
    // Alternative: just let the Irecvs succeed naturally (no error injected),
    // then inject before Isend. But the spy only supports one-shot injection.
    //
    // Workaround: We know there are recv_neighbors.size() Irecv calls before
    // the first Isend. We can't easily inject at a specific call index with
    // the current spy API. Instead, we test with an empty recv list so the
    // first MPI call is Isend.

    // Create a plan with NO recv neighbors so first MPI call is Isend
    std::vector<halo::Neighbor_Info> empty_recv;
    halo::Halo_Plan send_only_plan(comm, send_neighbors, empty_recv);

    std::size_t send_total = send_only_plan.total_send_elements();
    Kokkos::View<double *, Kokkos::HostSpace> send_view("send_view", send_total);

    spy.reset();
    spy.set_next_error(error_code);

    // The first send neighbor's rank should appear in the error message
    int expected_failing_rank = send_neighbors[0].rank;

    // Execute exchange_blocking — should throw on first Isend
    bool caught_runtime_error = false;
    std::string error_message;

    try {
        halo::exchange_blocking(send_only_plan, send_view);
    } catch (const std::runtime_error &e) {
        caught_runtime_error = true;
        error_message = e.what();
    }

    // Verify std::runtime_error was thrown
    RC_ASSERT(caught_runtime_error);

    // Verify the error message contains the MPI error string
    RC_ASSERT(error_message.find("Mock MPI error") != std::string::npos);

    // Verify the error message contains the failing neighbor rank
    RC_ASSERT(error_message.find(std::to_string(expected_failing_rank)) != std::string::npos);
}

// ─── Property 18: Completion Triggers Post-Receive Deep-Copy When Staged ─────
// Feature: helm-halo-microlibrary, Property 18: Completion Triggers Post-Receive Deep-Copy When Staged
//
// For any Halo_Handle associated with a staged (non-GPU-aware) exchange, when
// test() returns true or wait() completes, the handle SHALL perform the
// post-receive deep-copy callback before returning to the caller.
//
// We test this by injecting a staged_recv callback via the Halo_Handle_Test_Access
// friend class and verifying the callback is invoked upon completion.
//
// **Validates: Requirements 7.3, 7.4**

namespace halo {

/// @brief Test-only friend class that provides access to Halo_Handle internals
/// for injecting staged_recv state in property tests.
class Halo_Handle_Test_Access {
   public:
    /// Inject Request_Guard objects into a Halo_Handle.
    static void add_request(Halo_Handle &handle, MPI_Request &req) {
        handle.requests_.emplace_back(req);
    }

    /// Inject a staged_recv callback into a Halo_Handle.
    static void set_staged_recv(Halo_Handle &handle, std::function<void()> callback) {
        auto staged = std::make_unique<Halo_Handle::Staged_Recv>();
        staged->post_recv_copy = std::move(callback);
        staged->completed = false;
        handle.staged_recv_ = std::move(staged);
    }

    /// Check if staged_recv has been marked completed.
    static bool is_staged_completed(const Halo_Handle &handle) {
        if (!handle.staged_recv_) return false;
        return handle.staged_recv_->completed;
    }
};

}  // namespace halo

RC_GTEST_PROP(ExchangeProperty18, WaitTriggersPostReceiveDeepCopy, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a random number of pending requests [1, 8]
    int num_requests = *rc::gen::inRange(1, 9);

    // Construct a Halo_Handle and inject requests + staged_recv callback
    halo::Halo_Handle handle;

    for (int i = 0; i < num_requests; ++i) {
        // Create a sentinel request (mock MPI_Test/Wait will complete it)
        MPI_Request req = MPI_REQUEST_NULL;
        // Use MPI_Irecv to get a non-null sentinel request from the mock
        MPI_Irecv(nullptr, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        halo::Halo_Handle_Test_Access::add_request(handle, req);
    }

    // Inject a staged_recv callback that sets a flag when called
    bool deep_copy_called = false;
    halo::Halo_Handle_Test_Access::set_staged_recv(handle, [&deep_copy_called]() { deep_copy_called = true; });

    // Reset spy to only track wait() calls
    spy.reset();

    // Call wait() — this should complete all requests and trigger the callback
    handle.wait();

    // Verify: the post-receive deep-copy callback was invoked
    RC_ASSERT(deep_copy_called);

    // Verify: staged_recv is marked as completed
    RC_ASSERT(halo::Halo_Handle_Test_Access::is_staged_completed(handle));
}

RC_GTEST_PROP(ExchangeProperty18, TestTrueTriggersPostReceiveDeepCopy, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a random number of pending requests [1, 8]
    int num_requests = *rc::gen::inRange(1, 9);

    // Construct a Halo_Handle and inject requests + staged_recv callback
    halo::Halo_Handle handle;

    for (int i = 0; i < num_requests; ++i) {
        // Create a sentinel request (mock MPI_Test will complete it immediately)
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Irecv(nullptr, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        halo::Halo_Handle_Test_Access::add_request(handle, req);
    }

    // Inject a staged_recv callback that sets a flag when called
    bool deep_copy_called = false;
    halo::Halo_Handle_Test_Access::set_staged_recv(handle, [&deep_copy_called]() { deep_copy_called = true; });

    // Reset spy to only track test() calls
    spy.reset();

    // Call test() — in mock mode, MPI_Test always returns completed,
    // so test() should return true and trigger the callback
    bool completed = handle.test();

    // Verify: test() returned true (all operations completed in mock mode)
    RC_ASSERT(completed);

    // Verify: the post-receive deep-copy callback was invoked
    RC_ASSERT(deep_copy_called);

    // Verify: staged_recv is marked as completed
    RC_ASSERT(halo::Halo_Handle_Test_Access::is_staged_completed(handle));
}

RC_GTEST_PROP(ExchangeProperty18, DeepCopyCalledExactlyOnce, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a random number of pending requests [1, 8]
    int num_requests = *rc::gen::inRange(1, 9);

    // Construct a Halo_Handle and inject requests + staged_recv callback
    halo::Halo_Handle handle;

    for (int i = 0; i < num_requests; ++i) {
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Irecv(nullptr, 0, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        halo::Halo_Handle_Test_Access::add_request(handle, req);
    }

    // Inject a staged_recv callback that counts invocations
    int call_count = 0;
    halo::Halo_Handle_Test_Access::set_staged_recv(handle, [&call_count]() { ++call_count; });

    spy.reset();

    // Call wait() to trigger the callback
    handle.wait();
    RC_ASSERT(call_count == 1);

    // Call wait() again — should NOT trigger the callback a second time
    // (staged_recv_.completed is already true)
    handle.wait();
    RC_ASSERT(call_count == 1);

    // Call test() — should also NOT trigger the callback again
    (void)handle.test();
    RC_ASSERT(call_count == 1);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation.

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

// Register the Kokkos environment with GTest
static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
