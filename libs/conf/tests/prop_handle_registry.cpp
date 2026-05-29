// ─── Property-Based Tests: conf::fortran::Handle_Registry ────────────────────
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// Uses RapidCheck to verify that for ANY interleaving of register_handle /
// release operations on the opaque Handle_Registry:
//
//   * every issued token is a distinct positive integer (> 0),
//   * 0 (CONF_HANDLE_INVALID) is never issued,
//   * once a token is released it is never valid() / never resolves via
//     lookup() again, and is never re-issued by a later register_handle.
//
// The registry is a process-wide singleton with monotonically increasing
// tokens, so we track the tokens this run issues in a std::set to verify
// global distinctness and non-reuse across the whole generated sequence.
//
// **Validates: Requirements 31.1, 31.2, 31.3**
// ─────────────────────────────────────────────────────────────────────────────

#include "../src/fortran/handle_registry.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <map>
#include <set>
#include <vector>

using conf::fortran::CONF_HANDLE_INVALID;
using conf::fortran::Handle_Registry;

// ─── Property 5: distinct positive tokens, 0 never issued, released stays dead ─
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// Drives an arbitrary interleaving of register/release operations and asserts
// the full set of invariants after every step. A boolean per step selects
// register (true) vs release (false); a release with nothing live degrades to
// a register so the sequence always makes progress.
//
// **Validates: Requirements 31.1, 31.2, 31.3**

RC_GTEST_PROP(HandleRegistryProperty5,
              IssuedTokensDistinctPositiveAndReleasedNeverReused,
              ()) {
    auto& reg = Handle_Registry::instance();

    // Pointer comparisons are funneled through these boolean helpers so that
    // RapidCheck's RC_ASSERT only ever captures/shows a `bool`. RapidCheck's
    // expression renderer cannot stringify a bare `void*` (it attempts to
    // dereference it), so a raw `RC_ASSERT(p == q)` over `void*` would fail to
    // compile; the helpers sidestep that without weakening the checks.
    auto same_ptr = [](const void* a, const void* b) { return a == b; };
    auto is_null = [](const void* p) { return p == nullptr; };

    // Arbitrary interleaving: true == register, false == release.
    const auto ops = *rc::gen::container<std::vector<bool>>(
        rc::gen::arbitrary<bool>());

    std::set<int> issued;          // every token issued during this run
    std::set<int> released;        // every token released during this run
    std::map<int, void*> live;     // token -> ptr currently registered

    // Invariant that must hold continuously: nothing ever released is valid.
    auto assert_released_stay_dead = [&]() {
        for (int t : released) {
            RC_ASSERT(reg.valid(t) == false);       // 31.2
            RC_ASSERT(is_null(reg.lookup(t)));      // 31.2
        }
    };

    for (bool do_register : ops) {
        if (do_register || live.empty()) {
            // ── register ────────────────────────────────────────────────
            void* ptr = static_cast<void*>(new int(0));
            const int token = reg.register_handle(ptr);

            // 31.1: token is a distinct positive integer; 0 is never issued.
            RC_ASSERT(token != CONF_HANDLE_INVALID);
            RC_ASSERT(token > 0);
            RC_ASSERT(issued.count(token) == 0);     // globally distinct
            // 31.2 / 31.3: a released token is never handed out again.
            RC_ASSERT(released.count(token) == 0);

            issued.insert(token);
            live.emplace(token, ptr);

            // A freshly issued token resolves to exactly what we registered.
            RC_ASSERT(reg.valid(token));
            RC_ASSERT(same_ptr(reg.lookup(token), ptr));
        } else {
            // ── release ─────────────────────────────────────────────────
            const auto it = live.begin();
            const int token = it->first;
            void* const ptr = it->second;

            void* const returned = reg.release(token);
            RC_ASSERT(same_ptr(returned, ptr));

            delete static_cast<int*>(ptr);
            live.erase(it);
            released.insert(token);

            // 31.2: immediately after release the token is dead.
            RC_ASSERT(reg.valid(token) == false);
            RC_ASSERT(is_null(reg.lookup(token)));
        }

        // Every token released so far remains dead for the rest of the run.
        assert_released_stay_dead();
    }

    // ── cleanup: drain any handles still live so the test leaks nothing ──
    for (auto& [token, ptr] : live) {
        reg.release(token);
        delete static_cast<int*>(ptr);
    }
}

// ─── Property 5 (focused): register → release → register issues a fresh token ─
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// Directly exercises Requirement 31.3: after a handle is registered and
// released, a batch of subsequent registrations must each yield a token
// distinct from the released one (and from each other), and the released
// token must stay invalid throughout.
//
// **Validates: Requirements 31.1, 31.2, 31.3**

RC_GTEST_PROP(HandleRegistryProperty5,
              ReleasedTokenIsNeverReissuedBySubsequentRegister,
              ()) {
    auto& reg = Handle_Registry::instance();

    // See note in the first property: keep `void*` out of RC_ASSERT's captured
    // expression (RapidCheck cannot render a bare void*), so compare through a
    // boolean helper instead.
    auto same_ptr = [](const void* a, const void* b) { return a == b; };
    auto is_null = [](const void* p) { return p == nullptr; };

    // Number of follow-up registrations after the release (1..32).
    const int follow_ups = *rc::gen::inRange<int>(1, 33);

    // Register one handle then release it.
    int first_payload = 0;
    const int released_token =
        reg.register_handle(static_cast<void*>(&first_payload));
    RC_ASSERT(released_token > 0);

    RC_ASSERT(same_ptr(reg.release(released_token),
                       static_cast<void*>(&first_payload)));
    RC_ASSERT(reg.valid(released_token) == false);

    std::set<int> seen;
    std::vector<int> tokens;
    tokens.reserve(static_cast<std::size_t>(follow_ups));

    for (int i = 0; i < follow_ups; ++i) {
        void* ptr = static_cast<void*>(new int(0));
        const int token = reg.register_handle(ptr);
        tokens.push_back(token);

        RC_ASSERT(token > 0);                       // 31.1
        RC_ASSERT(token != released_token);         // 31.3: not re-issued
        RC_ASSERT(seen.insert(token).second);       // 31.1: distinct
        RC_ASSERT(reg.valid(token));

        // The released token stays invalid no matter how many we register.
        RC_ASSERT(reg.valid(released_token) == false);   // 31.2
        RC_ASSERT(is_null(reg.lookup(released_token)));   // 31.2

        delete static_cast<int*>(ptr);
        reg.release(token);
    }
}
