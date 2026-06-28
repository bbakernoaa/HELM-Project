// --- Property-Based Tests: halo::Request_Guard ---
// Feature: helm-halo-microlibrary, Property 4 & 5: Request_Guard Destruction
//
// Uses RapidCheck to verify that:
//   Property 4: Normal destruction calls MPI_Wait exactly once.
//   Property 5: Unwinding destruction calls MPI_Cancel then MPI_Request_free.
//
// The MPI_Spy interposition layer intercepts MPI calls and records them
// for deterministic verification without relying on MPI runtime side effects.
//
// **Validates: Requirements 2.2, 2.7**
// ---

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "halo/request_guard.hpp"
#include "mpi_interposition.hpp"

// --- RapidCheck Generators ---
// MPI_Request in OpenMPI is a pointer type (ompi_request_t*). We generate
// arbitrary non-null uintptr_t values and reinterpret_cast them to MPI_Request
// to simulate a range of opaque handle values.

namespace {

/// Generate a non-null MPI_Request value suitable for testing.
/// Returns a uintptr_t that can be cast to MPI_Request.
rc::Gen<std::uintptr_t> genNonNullRequestValue() {
    return rc::gen::map(rc::gen::inRange<std::intptr_t>(1, 100000), [](std::intptr_t val) -> std::uintptr_t {
        // Offset to avoid accidental collision with MPI_REQUEST_NULL
        return static_cast<std::uintptr_t>(val * 16 + 0x200000);
    });
}

/// Helper: cast uintptr_t to MPI_Request
inline MPI_Request toRequest(std::uintptr_t val) {
    return reinterpret_cast<MPI_Request>(val);
}

}  // anonymous namespace

// --- Property 4: Request_Guard Normal Destruction Completes Operation ---
// Feature: helm-halo-microlibrary, Property 4: Request_Guard Normal Destruction Completes Operation
//
// For any Request_Guard holding a non-null MPI_Request when destruction occurs
// outside of stack unwinding (std::uncaught_exceptions() unchanged from
// construction), the destructor SHALL invoke MPI_Wait exactly once to complete
// the pending operation.
//
// **Validates: Requirements 2.2**

RC_GTEST_PROP(RequestGuardProperty4, NormalDestructionCallsWaitExactlyOnce, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    MPI_Request req = toRequest(*genNonNullRequestValue());

    {
        halo::Request_Guard guard(req);
    }

    auto wait_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Wait);
    RC_ASSERT(wait_count == 1u);

    auto cancel_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Cancel);
    RC_ASSERT(cancel_count == 0u);

    auto free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Request_free);
    RC_ASSERT(free_count == 0u);
}

// --- Property 4b: Normal destruction - total call count ---
// Feature: helm-halo-microlibrary, Property 4: Request_Guard Normal Destruction Completes Operation
//
// **Validates: Requirements 2.2**

RC_GTEST_PROP(RequestGuardProperty4, NormalDestructionWaitOnlyCall, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    MPI_Request req = toRequest(*genNonNullRequestValue());

    {
        halo::Request_Guard guard(req);
    }

    RC_ASSERT(spy.call_count() == 1u);

    auto const &calls = spy.calls();
    RC_ASSERT(calls[0].type == halo::testing::MPI_Call_Record::Type::Wait);
}

// --- Property 4c: MPI_REQUEST_NULL skips MPI_Wait ---
// Feature: helm-halo-microlibrary, Property 4: Request_Guard Normal Destruction Completes Operation
//
// **Validates: Requirements 2.2**

RC_GTEST_PROP(RequestGuardProperty4, NullRequestSkipsWait, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    {
        halo::Request_Guard guard;
    }

    RC_ASSERT(spy.call_count() == 0u);
}

// --- Property 5: Request_Guard Unwinding Destruction Cancels Operation ---
// Feature: helm-halo-microlibrary, Property 5: Request_Guard Unwinding Destruction Cancels Operation
//
// For any Request_Guard holding a non-null MPI_Request when destruction occurs
// during stack unwinding (std::uncaught_exceptions() increased since
// construction), the destructor SHALL invoke MPI_Cancel followed by
// MPI_Request_free, in that order.
//
// **Validates: Requirements 2.7**

RC_GTEST_PROP(RequestGuardProperty5, UnwindingDestructionCallsCancelThenFree, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    MPI_Request req = toRequest(*genNonNullRequestValue());

    try {
        halo::Request_Guard guard(req);
        throw std::runtime_error("trigger stack unwinding");
    } catch (std::runtime_error const &) {
    }

    auto cancel_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Cancel);
    RC_ASSERT(cancel_count == 1u);

    auto free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Request_free);
    RC_ASSERT(free_count == 1u);

    auto wait_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Wait);
    RC_ASSERT(wait_count == 0u);
}

// --- Property 5b: Cancel precedes Request_free in call sequence ---
// Feature: helm-halo-microlibrary, Property 5: Request_Guard Unwinding Destruction Cancels Operation
//
// Verify that MPI_Cancel is called BEFORE MPI_Request_free during unwinding
// destruction, ensuring the correct ordering of cancellation operations.
//
// **Validates: Requirements 2.7**

RC_GTEST_PROP(RequestGuardProperty5, CancelPrecedesFreeInCallOrder, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    MPI_Request req = toRequest(*genNonNullRequestValue());

    try {
        halo::Request_Guard guard(req);
        throw std::runtime_error("trigger stack unwinding");
    } catch (std::runtime_error const &) {
    }

    auto const &calls = spy.calls();
    RC_ASSERT(calls.size() == 2u);

    RC_ASSERT(calls[0].type == halo::testing::MPI_Call_Record::Type::Cancel);
    RC_ASSERT(calls[1].type == halo::testing::MPI_Call_Record::Type::Request_free);
}

// --- Property 5c: Null request during unwinding is no-op ---
// Feature: helm-halo-microlibrary, Property 5: Request_Guard Unwinding Destruction Cancels Operation
//
// Verify that a Request_Guard holding MPI_REQUEST_NULL does NOT call
// MPI_Cancel or MPI_Request_free during unwinding destruction (null-handle
// case is always a no-op regardless of unwinding state).
//
// **Validates: Requirements 2.7**

RC_GTEST_PROP(RequestGuardProperty5, NullRequestDuringUnwindingIsNoOp, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    try {
        halo::Request_Guard guard;
        throw std::runtime_error("trigger stack unwinding");
    } catch (std::runtime_error const &) {
    }

    RC_ASSERT(spy.call_count() == 0u);
}
