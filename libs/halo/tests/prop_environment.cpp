// --- Property-Based Tests: halo::Environment ---------------------------------
// Feature: helm-halo-microlibrary, Property 20: Environment Initialization Idempotence
//
// Uses RapidCheck to verify that calling Environment::initialize() multiple
// times preserves the thread support level and does not re-query MPI.
//
// Since std::once_flag is process-global and cannot be reset, the property test
// verifies idempotence by:
//   1. Initializing once (first call triggers MPI_Query_thread)
//   2. Calling initialize() N additional times (random N per iteration)
//   3. Verifying thread_support_level() returns the same value after each call
//   4. Verifying MPI_Query_thread is NOT called again after the first init
//
// **Validates: Requirements 9.6**
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>

#include "halo/environment.hpp"
#include "mpi_interposition.hpp"

// --- Test Fixture ------------------------------------------------------------
// We use a fixture to perform the initial Environment::initialize() call once
// (in SetUpTestSuite) and then verify idempotence across property iterations.

namespace {

/// Track whether the initial initialization has been performed.
bool g_initial_init_done = false;
int g_expected_thread_level = -1;

/// Perform the one-time initialization before any property iterations run.
/// This sets up the MPI_Spy thread level and calls initialize() once.
void ensure_initialized() {
    if (!g_initial_init_done) {
        auto &spy = halo::testing::MPI_Spy::instance();
        spy.reset();
        // Configure the spy to report MPI_THREAD_SERIALIZED
        spy.set_thread_level(MPI_THREAD_SERIALIZED);

        // First call -- this triggers MPI_Query_thread exactly once
        halo::Environment::initialize();

        g_expected_thread_level = halo::Environment::thread_support_level();
        g_initial_init_done = true;
    }
}

}  // anonymous namespace

// --- Property 20: Environment Initialization Idempotence ---------------------
// Feature: helm-halo-microlibrary, Property 20: Environment Initialization Idempotence
//
// For any sequence of calls to Environment::initialize() after the first
// successful call, subsequent calls SHALL return without re-querying MPI and
// SHALL preserve the previously stored thread support level unchanged.
//
// **Validates: Requirements 9.6**

RC_GTEST_PROP(EnvironmentProperty20, InitializationIdempotence, ()) {
    // Ensure the one-time initialization has occurred
    ensure_initialized();

    auto &spy = halo::testing::MPI_Spy::instance();

    // Record the query_thread call count BEFORE our repeated calls
    std::size_t calls_before = spy.query_thread_call_count();

    // Generate a random number of additional initialize() calls (1 to 50)
    int n = *rc::gen::inRange(1, 51);

    // Call initialize() N times and verify thread_support_level() after each
    for (int i = 0; i < n; ++i) {
        halo::Environment::initialize();

        // After each call, thread_support_level() must return the same value
        int level = halo::Environment::thread_support_level();
        RC_ASSERT(level == g_expected_thread_level);
    }

    // Verify MPI_Query_thread was NOT called again (idempotence -- no re-query)
    std::size_t calls_after = spy.query_thread_call_count();
    RC_ASSERT(calls_after == calls_before);
}

// --- Property 20b: thread_support_level() consistency across calls -----------
// Feature: helm-halo-microlibrary, Property 20: Environment Initialization Idempotence
//
// Verify that thread_support_level() returns the same value on every call,
// regardless of how many times it is queried.
//
// **Validates: Requirements 9.6**

RC_GTEST_PROP(EnvironmentProperty20, ThreadLevelConsistency, ()) {
    // Ensure the one-time initialization has occurred
    ensure_initialized();

    // Generate a random number of thread_support_level() queries (2 to 100)
    int n = *rc::gen::inRange(2, 101);

    int first_level = halo::Environment::thread_support_level();

    for (int i = 1; i < n; ++i) {
        int level = halo::Environment::thread_support_level();
        RC_ASSERT(level == first_level);
    }
}

// --- Property 20c: is_thread_multiple() consistency --------------------------
// Feature: helm-halo-microlibrary, Property 20: Environment Initialization Idempotence
//
// Verify that is_thread_multiple() returns a value consistent with
// thread_support_level() and remains stable across repeated initialize() calls.
//
// **Validates: Requirements 9.6**

RC_GTEST_PROP(EnvironmentProperty20, IsThreadMultipleConsistency, ()) {
    // Ensure the one-time initialization has occurred
    ensure_initialized();

    // Call initialize() a random number of times
    int n = *rc::gen::inRange(1, 30);
    for (int i = 0; i < n; ++i) {
        halo::Environment::initialize();
    }

    // Verify is_thread_multiple() is consistent with thread_support_level()
    int level = halo::Environment::thread_support_level();
    bool expected_multiple = (level == MPI_THREAD_MULTIPLE);
    RC_ASSERT(halo::Environment::is_thread_multiple() == expected_multiple);
}
