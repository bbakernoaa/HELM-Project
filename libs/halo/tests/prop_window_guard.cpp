// ─── Property-Based Tests: Window_Guard (RAII MPI_Win wrapper) ──────────────
// Feature: helm-halo-microlibrary
//
// Uses the MPI interposition spy to verify Window_Guard's destructor behavior
// deterministically — without relying on a real MPI runtime or RMA fabric.
//
// Property 6: Window_Guard Fence-Before-Free on Active Epoch
//   For any Window_Guard with epoch_active==true and holding a non-null MPI_Win,
//   destruction SHALL invoke MPI_Win_fence(0, win) BEFORE MPI_Win_free(win).
//   Validates: Requirements 3.7
//
// Property 7: Window_Guard Destructor Never Throws
//   For any error code returned by MPI_Win_fence or MPI_Win_free during
//   Window_Guard destruction, the destructor SHALL complete without throwing.
//   Validates: Requirements 3.8
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <type_traits>

#include <mpi.h>

#include "halo/window_guard.hpp"
#include "mpi_interposition.hpp"

using halo::testing::MPI_Call_Record;
using halo::testing::MPI_Spy;

namespace {

/// Synthesize a non-null MPI_Win handle from an integer value. MPI_Win is a
/// pointer type in OpenMPI, so a distinct non-null pointer signals a live window.
inline MPI_Win synthetic_win(std::uintptr_t value = 0x10000) {
    return reinterpret_cast<MPI_Win>(value);
}

}  // namespace

// ─── Property 6: Window_Guard Fence-Before-Free on Active Epoch ─────────────
// Feature: helm-halo-microlibrary, Property 6: Window_Guard Fence-Before-Free on Active Epoch
//
// For any Window_Guard with epoch_active set to true and holding a non-null
// MPI_Win, destruction SHALL invoke MPI_Win_fence(0, win) before invoking
// MPI_Win_free(win).
//
// **Validates: Requirements 3.7**

RC_GTEST_PROP(WindowGuardProperty6, FenceCalledBeforeFreeWhenEpochActive, ()) {
    auto& spy = MPI_Spy::instance();
    spy.reset();

    // Generate a random non-null MPI_Win handle value.
    auto raw = *rc::gen::inRange<std::uintptr_t>(0x1000, 0xFFFFFF);
    MPI_Win win = reinterpret_cast<MPI_Win>(raw);

    {
        halo::Window_Guard guard(win);
        guard.set_epoch_active(true);  // signals the destructor to fence first
    }  // ~Window_Guard should: fence(0) then free

    auto calls = spy.calls_copy();
    // Expect exactly two calls: Win_fence then Win_free.
    RC_ASSERT(calls.size() == 2u);
    RC_ASSERT(calls[0].type == MPI_Call_Record::Type::Win_fence);
    RC_ASSERT(calls[0].arg == 0);  // assertion argument is 0
    RC_ASSERT(calls[1].type == MPI_Call_Record::Type::Win_free);
}

// Additional case: epoch_active == false -> no fence, just free.
RC_GTEST_PROP(WindowGuardProperty6, NoFenceWhenEpochInactive, ()) {
    auto& spy = MPI_Spy::instance();
    spy.reset();

    auto raw = *rc::gen::inRange<std::uintptr_t>(0x1000, 0xFFFFFF);
    MPI_Win win = reinterpret_cast<MPI_Win>(raw);

    {
        halo::Window_Guard guard(win);
        guard.set_epoch_active(false);  // no epoch to close
    }

    auto calls = spy.calls_copy();
    // Only Win_free; no Win_fence.
    RC_ASSERT(calls.size() == 1u);
    RC_ASSERT(calls[0].type == MPI_Call_Record::Type::Win_free);
}

// Additional case: MPI_WIN_NULL -> destructor is a no-op, no MPI calls at all.
RC_GTEST_PROP(WindowGuardProperty6, NullWindowIsNoOp, ()) {
    auto& spy = MPI_Spy::instance();
    spy.reset();

    {
        halo::Window_Guard guard(MPI_WIN_NULL);
        guard.set_epoch_active(true);  // epoch flag is irrelevant for null
    }

    // No MPI calls should be recorded for a null window.
    RC_ASSERT(spy.call_count() == 0u);
}

// ─── Property 7: Window_Guard Destructor Never Throws ───────────────────────
// Feature: helm-halo-microlibrary, Property 7: Window_Guard Destructor Never Throws
//
// For any error code returned by MPI_Win_fence or MPI_Win_free during
// Window_Guard destruction, the destructor SHALL complete without throwing an
// exception and SHALL NOT propagate the error.
//
// **Validates: Requirements 3.8**

// Compile-time check: the destructor must be declared noexcept.
static_assert(std::is_nothrow_destructible_v<halo::Window_Guard>,
              "Window_Guard destructor must be noexcept");

RC_GTEST_PROP(WindowGuardProperty7, DestructorSwallowsFenceError, ()) {
    auto& spy = MPI_Spy::instance();
    spy.reset();

    // Inject an error on the NEXT MPI call (which will be Win_fence inside the
    // destructor when epoch_active==true).
    spy.set_next_error(MPI_ERR_WIN);

    {
        halo::Window_Guard guard(synthetic_win());
        guard.set_epoch_active(true);
    }  // destructor calls fence (gets error), then free — must not throw.

    // The destructor ran to completion; verify both calls were still made even
    // though fence returned an error.
    auto calls = spy.calls_copy();
    // Depending on the implementation, it may stop after the fence error or
    // continue to free. Either way, it must NOT throw. At minimum the fence
    // was attempted.
    bool fence_attempted = false;
    for (auto const& c : calls) {
        if (c.type == MPI_Call_Record::Type::Win_fence) fence_attempted = true;
    }
    RC_ASSERT(fence_attempted);
}

RC_GTEST_PROP(WindowGuardProperty7, DestructorSwallowsFreeError, ()) {
    auto& spy = MPI_Spy::instance();
    spy.reset();

    // No epoch active, so the destructor goes straight to Win_free. Inject an
    // error into that call.
    spy.set_next_error(MPI_ERR_WIN);

    {
        halo::Window_Guard guard(synthetic_win());
        guard.set_epoch_active(false);
    }  // destructor calls free (gets error) — must not throw.

    auto calls = spy.calls_copy();
    bool free_attempted = false;
    for (auto const& c : calls) {
        if (c.type == MPI_Call_Record::Type::Win_free) free_attempted = true;
    }
    RC_ASSERT(free_attempted);
}
