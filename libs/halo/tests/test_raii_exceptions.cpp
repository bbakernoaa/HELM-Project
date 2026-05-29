// ─── HALO RAII Destructor Exception-Path Unit Tests ─────────────────────────
// Feature: helm-halo-microlibrary
//
// Proves that RAII destructors (Communicator, Request_Guard) execute their
// cleanup logic correctly on ALL exit paths — normal scope exit and exception
// stack unwinding — so that no MPI handles are leaked in error paths.
//
// All verification is deterministic via the MPI interposition spy
// (mpi_interposition.{hpp,cpp}), which records MPI_Comm_free, MPI_Cancel,
// MPI_Request_free, etc. and nullifies handles exactly as the real MPI C API
// does. This avoids any reliance on a live MPI runtime.
//
// Requirements covered:
//   11.1 — Communicator freed during exception unwinding
//   11.2 — Request_Guard cancelled + freed during exception unwinding
//   11.3 — Nested scopes destroy in reverse construction order
//   11.4 — Move-constructed source is NULL; source destruction is a no-op
//   11.5 — Communicator wrapping MPI_COMM_WORLD is never freed
//   11.6 — Communicator via split/dup is freed on destruction (round-trip)
//   11.7 — Deterministic verification via MPI interposition spy
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "halo/communicator.hpp"
#include "halo/request_guard.hpp"
#include "mpi_interposition.hpp"

using halo::testing::MPI_Call_Record;
using halo::testing::MPI_Spy;

namespace {

/// Cast an integer value to a synthetic, non-null MPI_Request handle.
/// MPI_Request is a pointer type in OpenMPI; a distinct non-null pointer value
/// lets the Request_Guard treat it as a live, pending operation.
inline MPI_Request synthetic_request(std::uintptr_t value) {
    return reinterpret_cast<MPI_Request>(value);
}

}  // anonymous namespace

// ─── Test Fixture ────────────────────────────────────────────────────────────
// Resets the spy before and after each test so call records never leak between
// tests (the spy is a process-wide singleton).

class RAIIExceptionTest : public ::testing::Test {
protected:
    void SetUp() override { MPI_Spy::instance().reset(); }
    void TearDown() override { MPI_Spy::instance().reset(); }
};

// ─── Requirement 11.1 ────────────────────────────────────────────────────────
// Construct a Communicator (via duplicate) inside a try block, throw a
// std::runtime_error, and after the catch confirm MPI_Comm_free was invoked
// during stack unwinding. The spy's MPI_Comm_free nullifies its handle argument
// (mirroring the real MPI C API), so the destroyed object's internal handle is
// MPI_COMM_NULL after the free — the leak-free post-condition this test asserts.

TEST_F(RAIIExceptionTest, CommunicatorFreedDuringExceptionUnwinding) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);
    spy.reset();  // ignore setup; focus on what unwinding does

    bool caught = false;
    try {
        // Constructed inside the try via duplicate() → owns a synthetic,
        // non-null, non-predefined handle that must be freed on any exit.
        halo::Communicator dup = world.duplicate();
        ASSERT_NE(dup.handle(), MPI_COMM_NULL);

        throw std::runtime_error("trigger stack unwinding");
    } catch (const std::runtime_error&) {
        caught = true;
    }

    EXPECT_TRUE(caught);
    // Destructor ran during unwinding and freed the owned handle exactly once.
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 1u);
}

// ─── Requirement 11.2 ────────────────────────────────────────────────────────
// Construct a Request_Guard wrapping a pending non-blocking operation inside a
// try block, throw, and confirm the request was cancelled (MPI_Cancel) and
// freed (MPI_Request_free) during stack unwinding — in that order — rather than
// waited on.

TEST_F(RAIIExceptionTest, RequestGuardCancelledAndFreedDuringUnwinding) {
    auto& spy = MPI_Spy::instance();

    bool caught = false;
    try {
        MPI_Request raw = synthetic_request(0x200000);
        halo::Request_Guard guard(raw);
        // Ownership transferred: the source handle is nullified on construction.
        EXPECT_EQ(raw, MPI_REQUEST_NULL);

        throw std::runtime_error("trigger stack unwinding");
    } catch (const std::runtime_error&) {
        caught = true;
    }

    EXPECT_TRUE(caught);

    // Unwinding path: cancel + free, never wait.
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Cancel), 1u);
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Request_free), 1u);
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Wait), 0u);

    // Verify ordering: MPI_Cancel must precede MPI_Request_free.
    auto calls = spy.calls_copy();
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Cancel);
    EXPECT_EQ(calls[1].type, MPI_Call_Record::Type::Request_free);
}

// ─── Requirement 11.3 ────────────────────────────────────────────────────────
// Two nested scopes: a Communicator in the outer scope (constructed first) and
// a Request_Guard in the inner scope (constructed second). Throwing from the
// innermost scope unwinds the stack, which must destroy objects in reverse
// construction order: the Request_Guard first (Cancel + Request_free), then the
// Communicator (Comm_free). The recorded call sequence proves the ordering.

TEST_F(RAIIExceptionTest, NestedScopesDestroyInReverseOrder) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);

    bool caught = false;
    try {
        // Outer scope resource: constructed FIRST.
        halo::Communicator outer = world.duplicate();
        ASSERT_NE(outer.handle(), MPI_COMM_NULL);

        // Clear setup noise; only destruction calls should remain.
        spy.reset();

        {
            // Inner scope resource: constructed SECOND.
            MPI_Request raw = synthetic_request(0x300000);
            halo::Request_Guard inner(raw);

            throw std::runtime_error("throw from innermost scope");
        }
    } catch (const std::runtime_error&) {
        caught = true;
    }

    EXPECT_TRUE(caught);

    // Reverse-order destruction: inner Request_Guard (Cancel, Request_free)
    // unwinds before the outer Communicator (Comm_free).
    auto calls = spy.calls_copy();
    ASSERT_EQ(calls.size(), 3u);
    EXPECT_EQ(calls[0].type, MPI_Call_Record::Type::Cancel);
    EXPECT_EQ(calls[1].type, MPI_Call_Record::Type::Request_free);
    EXPECT_EQ(calls[2].type, MPI_Call_Record::Type::Comm_free);
}

// ─── Requirement 11.4 ────────────────────────────────────────────────────────
// Move-construct a Communicator from a source. The source's handle must become
// MPI_COMM_NULL, the destination must own the original handle, and destroying
// the moved-from source must NOT call MPI_Comm_free.

TEST_F(RAIIExceptionTest, MoveConstructedSourceIsNullAndDestructionIsNoOp) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);

    // Destination kept alive in the outer scope so its eventual free does not
    // interfere with the source-destruction measurement below.
    std::optional<halo::Communicator> dst;

    {
        halo::Communicator src = world.duplicate();
        const MPI_Comm original = src.handle();
        ASSERT_NE(original, MPI_COMM_NULL);

        dst.emplace(std::move(src));  // move-construct destination from source

        // Source nullified; destination assumed ownership of the handle.
        EXPECT_EQ(src.handle(), MPI_COMM_NULL);
        EXPECT_EQ(dst->handle(), original);

        spy.reset();  // measure ONLY the moved-from source's destruction
        // `src` (moved-from, holds MPI_COMM_NULL) is destroyed at scope exit.
    }

    // Destroying the moved-from source must be a no-op (no MPI_Comm_free).
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);

    // `dst` still owns the real handle and frees it on destruction; that is
    // expected cleanup and not part of this assertion.
    dst.reset();
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 1u);
}

// ─── Requirement 11.5 ────────────────────────────────────────────────────────
// A Communicator wrapping the predefined MPI_COMM_WORLD must NEVER call
// MPI_Comm_free on destruction.

TEST_F(RAIIExceptionTest, PredefinedCommWorldNotFreed) {
    auto& spy = MPI_Spy::instance();

    {
        halo::Communicator world(MPI_COMM_WORLD);
        EXPECT_EQ(world.handle(), MPI_COMM_WORLD);
    }  // destructor runs here

    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);
}

// MPI_COMM_SELF is likewise predefined and must not be freed.
TEST_F(RAIIExceptionTest, PredefinedCommSelfNotFreed) {
    auto& spy = MPI_Spy::instance();

    {
        halo::Communicator self(MPI_COMM_SELF);
        EXPECT_EQ(self.handle(), MPI_COMM_SELF);
    }

    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);
}

// ─── Requirement 11.6 ────────────────────────────────────────────────────────
// Create a Communicator via duplicate(), confirm it owns a live (non-null)
// handle, then destroy it and confirm MPI_Comm_free was invoked exactly once.
// The spy nullifies the handle on free, so no MPI handle is leaked — the
// round-trip (acquire → release) property.

TEST_F(RAIIExceptionTest, DuplicateRoundTripFreesHandle) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);
    spy.reset();

    {
        halo::Communicator dup = world.duplicate();
        ASSERT_NE(dup.handle(), MPI_COMM_NULL);  // acquired a live handle
        EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);
    }  // destructor releases the handle here

    // Round-trip complete: exactly one free, no leaked handle.
    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 1u);
}

// The same round-trip property holds for split-derived sub-communicators.
TEST_F(RAIIExceptionTest, SplitRoundTripFreesHandle) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);
    spy.reset();

    {
        halo::Communicator sub = world.split(/*color=*/0, /*key=*/0);
        ASSERT_NE(sub.handle(), MPI_COMM_NULL);  // acquired a live handle
        EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);
    }  // destructor releases the handle here

    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 1u);
}

// A split with MPI_UNDEFINED yields MPI_COMM_NULL, which must not be freed.
TEST_F(RAIIExceptionTest, SplitUndefinedYieldsNullAndIsNotFreed) {
    auto& spy = MPI_Spy::instance();

    halo::Communicator world(MPI_COMM_WORLD);
    spy.reset();

    {
        halo::Communicator empty = world.split(MPI_UNDEFINED, 0);
        EXPECT_EQ(empty.handle(), MPI_COMM_NULL);
    }

    EXPECT_EQ(spy.count_of(MPI_Call_Record::Type::Comm_free), 0u);
}
