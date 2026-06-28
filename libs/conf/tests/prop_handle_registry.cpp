// --- Property-Based Tests: conf::fortran::Handle_Registry --------------------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify the opaque handle registry that maps integer
// tokens to C++ object pointers for the Fortran C-interop layer. The registry
// is a process-global singleton with monotonically increasing positive tokens
// that are never reused (token 0 reserved as CONF_HANDLE_INVALID).
//
// Property 5: Handle Uniqueness & Non-Reuse
//   Validates: Requirements 31.1, 31.2, 31.3
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "handle_registry.hpp"

using conf::fortran::CONF_HANDLE_INVALID;
using conf::fortran::Handle_Registry;

namespace {

// The registry is a process-global singleton whose tokens are monotonically
// increasing and NEVER reused across the lifetime of the process. To verify
// global uniqueness and the "never reissued" guarantee across all property
// iterations, we track every token the registry has ever issued.
std::set<int> g_all_issued_tokens;
std::set<int> g_all_released_tokens;

}  // anonymous namespace

// --- Property 5a: Handle Uniqueness & Non-Reuse (interleaved ops) ------------
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// For any random interleaving of register/release operations:
//   (1) All tokens ever issued are > 0 (CONF_HANDLE_INVALID is never issued)
//   (2) All tokens ever issued are unique (no duplicates in the full history)
//   (3) After releasing a token, valid(token) returns false
//   (4) No subsequently issued token equals any previously released token
//
// **Validates: Requirements 31.1, 31.2, 31.3**

RC_GTEST_PROP(HandleRegistryProperty5, UniquenessAndNonReuse, ()) {
    auto &reg = Handle_Registry::instance();

    // Number of register/release operations to perform this iteration.
    const int num_ops = *rc::gen::inRange(1, 101);

    // Model state for this iteration:
    //   live_tokens    -> tokens currently registered (for random release)
    //   issued_this_run -> all tokens issued in this iteration (local uniqueness)
    //   released_this_run -> tokens released this iteration (non-reuse check)
    //   storage        -> owns dummy objects so their addresses stay stable
    std::vector<int> live_tokens;
    std::set<int> issued_this_run;
    std::set<int> released_this_run;
    std::vector<std::unique_ptr<int>> storage;

    for (int op = 0; op < num_ops; ++op) {
        // Choose register vs release. Release is only possible when something
        // is live; otherwise we must register.
        const bool do_register = live_tokens.empty() ? true : *rc::gen::arbitrary<bool>();

        if (do_register) {
            // Create a unique, address-stable dummy object to register.
            storage.push_back(std::make_unique<int>(op));
            void *ptr = static_cast<void *>(storage.back().get());

            const int token = reg.register_handle(ptr);

            // (1) Tokens are strictly positive (0 is never issued).
            RC_ASSERT(token > CONF_HANDLE_INVALID);
            RC_ASSERT(token > 0);

            // (2) Tokens are globally unique — never seen before across all
            // iterations of this property test in this process.
            RC_ASSERT(g_all_issued_tokens.find(token) == g_all_issued_tokens.end());
            g_all_issued_tokens.insert(token);

            // (2) Also unique within this iteration.
            RC_ASSERT(issued_this_run.insert(token).second);

            // (4) A newly issued token must never equal any previously released
            // token — within this run or across all previous runs.
            RC_ASSERT(released_this_run.find(token) == released_this_run.end());
            RC_ASSERT(g_all_released_tokens.find(token) == g_all_released_tokens.end());

            live_tokens.push_back(token);
        } else {
            // Release a randomly chosen live token.
            const int idx = *rc::gen::inRange(0, static_cast<int>(live_tokens.size()));
            const int token = live_tokens[static_cast<std::size_t>(idx)];

            reg.release(token);

            // (3) After releasing, valid() returns false.
            RC_ASSERT(!reg.valid(token));

            // Track released tokens for non-reuse verification.
            released_this_run.insert(token);
            g_all_released_tokens.insert(token);

            // Remove from live set.
            live_tokens.erase(live_tokens.begin() + idx);
        }
    }

    // Final check: all still-live tokens remain valid.
    for (int token : live_tokens) {
        RC_ASSERT(reg.valid(token));
    }

    // Cleanup: release everything still live so the singleton registry does not
    // accumulate stale entries across property iterations.
    for (int token : live_tokens) {
        reg.release(token);
        g_all_released_tokens.insert(token);
    }
}

// --- Property 5b: Released tokens are never reissued -------------------------
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// For any number of register-then-release cycles, each fresh registration
// yields a strictly larger token than every previously issued token, proving
// released tokens are never recycled and all tokens are unique positive ints.
//
// **Validates: Requirements 31.1, 31.2, 31.3**

RC_GTEST_PROP(HandleRegistryProperty5, ReleasedTokensNeverReissued, ()) {
    auto &reg = Handle_Registry::instance();

    const int cycles = *rc::gen::inRange(1, 51);

    int prev_token = 0;
    int dummy = 0;
    for (int i = 0; i < cycles; ++i) {
        const int token = reg.register_handle(static_cast<void *>(&dummy));

        // (1) Token is always positive.
        RC_ASSERT(token > 0);
        RC_ASSERT(token > CONF_HANDLE_INVALID);

        // (2)/(4) Strictly increasing means unique and never reissued.
        RC_ASSERT(token > prev_token);

        // (2) Globally unique across all iterations.
        RC_ASSERT(g_all_issued_tokens.find(token) == g_all_issued_tokens.end());
        g_all_issued_tokens.insert(token);

        // Release and verify the contract.
        reg.release(token);

        // (3) After release, valid() returns false.
        RC_ASSERT(!reg.valid(token));

        // (4) Track as released — subsequent iterations will verify non-reuse.
        g_all_released_tokens.insert(token);

        // Double release is safe and returns nullptr.
        const bool double_release_null = (reg.release(token) == nullptr);
        RC_ASSERT(double_release_null);

        prev_token = token;
    }
}

// --- Property 5c: Zero token is never valid ----------------------------------
// Feature: conf-config-parser, Property 5: Handle Uniqueness & Non-Reuse
//
// Token 0 (CONF_HANDLE_INVALID) is never issued, never valid, and lookup
// always returns nullptr regardless of registry state.
//
// **Validates: Requirements 31.1, 31.2, 31.3**

RC_GTEST_PROP(HandleRegistryProperty5, ZeroTokenNeverValid, ()) {
    auto &reg = Handle_Registry::instance();

    // Register some handles to change registry state.
    const int num_regs = *rc::gen::inRange(0, 21);
    std::vector<int> tokens;
    int dummy = 42;

    for (int i = 0; i < num_regs; ++i) {
        const int token = reg.register_handle(static_cast<void *>(&dummy));
        // (1) Issued token is never 0.
        RC_ASSERT(token != CONF_HANDLE_INVALID);
        RC_ASSERT(token > 0);
        tokens.push_back(token);
        g_all_issued_tokens.insert(token);
    }

    // Regardless of how many handles exist, token 0 is never valid.
    RC_ASSERT(!reg.valid(CONF_HANDLE_INVALID));
    const bool lookup_is_null = (reg.lookup(CONF_HANDLE_INVALID) == nullptr);
    RC_ASSERT(lookup_is_null);

    // Release of token 0 returns nullptr (no-op).
    const bool release_is_null = (reg.release(CONF_HANDLE_INVALID) == nullptr);
    RC_ASSERT(release_is_null);

    // Cleanup
    for (int token : tokens) {
        reg.release(token);
        g_all_released_tokens.insert(token);
    }
}
