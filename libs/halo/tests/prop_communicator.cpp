// ─── Property-Based Tests: halo::Communicator ───────────────────────────────
// Feature: helm-halo-microlibrary
//
// Uses RapidCheck to verify RAII construction, move semantics, and destructor
// cleanup properties of the Communicator class.
//
// Property 1: RAII Construction Round-Trip
//   Validates: Requirements 1.1, 1.5
//
// Property 3: Communicator Destructor Cleanup
//   Validates: Requirements 1.2, 1.6
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "halo/communicator.hpp"
#include "mpi_interposition.hpp"

// ─── RapidCheck Generators ───────────────────────────────────────────────────
// MPI_Comm in OpenMPI is a pointer type (ompi_communicator_t*). We generate
// arbitrary uintptr_t values and reinterpret_cast them to MPI_Comm to simulate
// a range of opaque handle values. We generate integers (not MPI_Comm directly)
// to avoid RapidCheck's ShowType attempting typeid on the incomplete struct.

namespace {

/// Generate a non-null, non-predefined MPI_Comm value suitable for testing.
/// Returns a uintptr_t that can be cast to MPI_Comm.
rc::Gen<std::uintptr_t> genNonNullCommValue() {
    return rc::gen::suchThat(rc::gen::map(rc::gen::inRange<std::intptr_t>(1, 100000),
                                          [](std::intptr_t val) -> std::uintptr_t {
                                              // Offset to avoid accidental collision with predefined comms
                                              return static_cast<std::uintptr_t>(val * 16 + 0x100000);
                                          }),
                             [](std::uintptr_t val) {
                                 auto comm = reinterpret_cast<MPI_Comm>(val);
                                 return comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF;
                             });
}

/// Generate an arbitrary MPI_Comm value (including NULL and predefined).
/// Returns a uintptr_t that can be cast to MPI_Comm.
rc::Gen<std::uintptr_t> genArbitraryCommValue() {
    return rc::gen::oneOf(rc::gen::just(reinterpret_cast<std::uintptr_t>(MPI_COMM_NULL)),
                          rc::gen::just(reinterpret_cast<std::uintptr_t>(MPI_COMM_WORLD)),
                          rc::gen::just(reinterpret_cast<std::uintptr_t>(MPI_COMM_SELF)), genNonNullCommValue());
}

/// Helper: cast uintptr_t to MPI_Comm
inline MPI_Comm toComm(std::uintptr_t val) {
    return reinterpret_cast<MPI_Comm>(val);
}

}  // anonymous namespace

// ─── Property 1: RAII Construction Round-Trip ────────────────────────────────
// Feature: helm-halo-microlibrary, Property 1: RAII Construction Round-Trip
//
// For any valid MPI_Comm handle value, constructing a Communicator and
// immediately querying handle() SHALL return the original value unchanged.
//
// **Validates: Requirements 1.1, 1.5**

RC_GTEST_PROP(CommunicatorProperty1, ConstructionRoundTrip, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate an arbitrary MPI_Comm handle (NULL, predefined, or synthetic)
    MPI_Comm original = toComm(*genArbitraryCommValue());

    // Construct the Communicator (takes exclusive ownership of the handle)
    halo::Communicator comm(original);

    // Verify handle() returns the original value unchanged
    RC_ASSERT(comm.handle() == original);
}

// ─── Feature: helm-halo-microlibrary, Property 3: Communicator Destructor Cleanup
// ─────────────────────────────────────────────────────────────────────────────
// For any Communicator holding a non-predefined, non-null MPI_Comm handle where
// MPI has not been finalized, destruction (whether via normal scope exit or
// exception unwinding) SHALL invoke MPI_Comm_free exactly once on that handle.
//
// **Validates: Requirements 1.2, 1.6**
// ─────────────────────────────────────────────────────────────────────────────

// ─── Property 3a: Destructor calls MPI_Comm_free exactly once (normal exit) ──

RC_GTEST_PROP(CommunicatorProperty3, DestructorCleanupNormalExit, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a non-predefined, non-null handle
    MPI_Comm comm_handle = toComm(*genNonNullCommValue());

    // Construct and destroy in a nested scope
    {
        halo::Communicator communicator(comm_handle);
        // Communicator goes out of scope here — destructor should call MPI_Comm_free
    }

    // Verify MPI_Comm_free was called exactly once
    auto comm_free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Comm_free);
    RC_ASSERT(comm_free_count == 1u);
}

// ─── Property 3b: Destructor calls MPI_Comm_free during exception unwinding ──

RC_GTEST_PROP(CommunicatorProperty3, DestructorCleanupExceptionUnwinding, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a non-predefined, non-null handle
    MPI_Comm comm_handle = toComm(*genNonNullCommValue());

    // Construct inside a try block and throw to trigger stack unwinding
    try {
        halo::Communicator communicator(comm_handle);
        throw std::runtime_error("test exception to trigger stack unwinding");
    } catch (const std::runtime_error &) {
        // Exception caught — destructor should have already run
    }

    // Verify MPI_Comm_free was called exactly once during unwinding
    auto comm_free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Comm_free);
    RC_ASSERT(comm_free_count == 1u);
}

// ─── Property 3c: Predefined communicators must NOT be freed ─────────────────

RC_GTEST_PROP(CommunicatorProperty3, DestructorSkipsPredefined, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Pick one of the predefined communicators randomly
    auto predefined_val = *rc::gen::element(reinterpret_cast<std::uintptr_t>(MPI_COMM_WORLD), reinterpret_cast<std::uintptr_t>(MPI_COMM_SELF));
    MPI_Comm predefined = toComm(predefined_val);

    {
        halo::Communicator communicator(predefined);
    }

    // Verify MPI_Comm_free was NOT called
    auto comm_free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Comm_free);
    RC_ASSERT(comm_free_count == 0u);
}

// ─── Property 3d: MPI_COMM_NULL must NOT be freed ────────────────────────────

RC_GTEST_PROP(CommunicatorProperty3, DestructorSkipsNull, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    {
        halo::Communicator communicator(MPI_COMM_NULL);
    }

    // Verify MPI_Comm_free was NOT called
    auto comm_free_count = spy.count_of(halo::testing::MPI_Call_Record::Type::Comm_free);
    RC_ASSERT(comm_free_count == 0u);
}
