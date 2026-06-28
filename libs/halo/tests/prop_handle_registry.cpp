// --- Property-Based Tests: halo::fortran::Handle_Registry --------------------
// Feature: helm-halo-microlibrary
//
// Uses RapidCheck to verify the opaque handle registry that maps integer
// tokens to C++ object pointers for the Fortran C-interop layer. The registry
// is a process-global singleton with monotonically increasing positive tokens
// that are never reused (token 0 reserved as HALO_HANDLE_INVALID).
//
// Property 21: Handle Registry Round-Trip
//   Validates: Requirements 14.3, 14.4, 14.16
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

#include "../src/fortran/handle_registry.hpp"

using halo::fortran::HALO_HANDLE_INVALID;
using halo::fortran::Handle_Registry;

namespace {

// The registry is a process-global singleton whose tokens are monotonically
// increasing and NEVER reused across the lifetime of the process. To verify
// global uniqueness and the "never reissued" guarantee across all property
// iterations, we track every token the registry has ever issued and the
// largest token observed so far.
std::set<int> g_all_issued_tokens;
int g_max_issued_token = HALO_HANDLE_INVALID;

}  // anonymous namespace

// --- Property 21: Handle Registry Round-Trip ---------------------------------
// Feature: helm-halo-microlibrary, Property 21: Handle Registry Round-Trip
//
// For any random sequence of register/release operations against the registry:
//   (a) every token returned is unique (never reissued) and positive,
//   (b) lookup() returns the exact pointer that was registered,
//   (c) released tokens become invalid and are never reissued,
//   (d) lookup() of an unregistered/released token returns nullptr.
//
// **Validates: Requirements 14.3, 14.4, 14.16**

RC_GTEST_PROP(HandleRegistryProperty21, RoundTrip, ()) {
    auto &reg = Handle_Registry::instance();

    // Number of register/release operations to perform this iteration.
    const int num_ops = *rc::gen::inRange(1, 101);

    // Model state for this iteration:
    //   live           -> token -> registered pointer (currently registered)
    //   live_tokens    -> the live tokens, for random release selection
    //   storage        -> owns dummy objects so their addresses stay stable
    std::unordered_map<int, void *> live;
    std::vector<int> live_tokens;
    std::vector<std::unique_ptr<int>> storage;

    for (int op = 0; op < num_ops; ++op) {
        // Choose register vs release. Release is only possible when something
        // is live; otherwise we must register.
        const bool do_register = live_tokens.empty() ? true : *rc::gen::arbitrary<bool>();

        if (do_register) {
            // Create a unique, address-stable dummy object to register.
            const int value = *rc::gen::arbitrary<int>();
            storage.push_back(std::make_unique<int>(value));
            void *ptr = static_cast<void *>(storage.back().get());

            const int token = reg.register_handle(ptr);

            // (a) Tokens are strictly positive (never HALO_HANDLE_INVALID).
            RC_ASSERT(token > HALO_HANDLE_INVALID);

            // (a)/(c) Tokens are globally unique and never reissued -- not even
            // after a previous token was released earlier in the process.
            RC_ASSERT(g_all_issued_tokens.insert(token).second);

            // Tokens are unique within this iteration's live set as well.
            RC_ASSERT(live.find(token) == live.end());

            g_max_issued_token = std::max(g_max_issued_token, token);

            live[token] = ptr;
            live_tokens.push_back(token);

            // (b) Round-trip: lookup() returns the exact registered pointer and
            // the token reports as valid immediately after registration.
            // (void* operands are wrapped in a bool so RapidCheck does not try
            // to dereference/show them.)
            const bool lookup_matches = (reg.lookup(token) == ptr);
            RC_ASSERT(lookup_matches);
            RC_ASSERT(reg.valid(token));
        } else {
            // Release a randomly chosen live token.
            const int idx = *rc::gen::inRange(0, static_cast<int>(live_tokens.size()));
            const int token = live_tokens[static_cast<std::size_t>(idx)];
            void *const expected = live[token];

            // release() returns the originally registered pointer.
            void *const released = reg.release(token);
            const bool released_matches = (released == expected);
            RC_ASSERT(released_matches);

            // (c)/(d) After release the token is invalid and lookup() is null.
            RC_ASSERT(!reg.valid(token));
            const bool lookup_is_null = (reg.lookup(token) == nullptr);
            RC_ASSERT(lookup_is_null);

            // Remove from the model.
            live.erase(token);
            live_tokens.erase(live_tokens.begin() + idx);
        }

        // Invariant after every operation: every still-live token continues to
        // resolve to the exact pointer it was registered with.
        for (const auto &[tok, p] : live) {
            const bool still_matches = (reg.lookup(tok) == p);
            RC_ASSERT(still_matches);
            RC_ASSERT(reg.valid(tok));
        }
    }

    // (d) The reserved invalid token never resolves to a pointer.
    const bool invalid_lookup_null = (reg.lookup(HALO_HANDLE_INVALID) == nullptr);
    RC_ASSERT(invalid_lookup_null);
    RC_ASSERT(!reg.valid(HALO_HANDLE_INVALID));

    // (d) A token the monotonic counter has not yet reached is never-registered
    // and must resolve to nullptr / report invalid.
    const int never_registered = g_max_issued_token + *rc::gen::inRange(1, 1000001);
    const bool never_lookup_null = (reg.lookup(never_registered) == nullptr);
    RC_ASSERT(never_lookup_null);
    RC_ASSERT(!reg.valid(never_registered));

    // Cleanup: release everything still live so the singleton registry does not
    // accumulate stale entries across property iterations.
    for (int token : live_tokens) {
        reg.release(token);
    }
}

// --- Property 21b: Released tokens are never reissued ------------------------
// Feature: helm-halo-microlibrary, Property 21: Handle Registry Round-Trip
//
// For any number of register-then-release cycles, each fresh registration
// yields a strictly larger token than every previously issued token, proving
// released tokens are never recycled.
//
// **Validates: Requirements 14.3, 14.4, 14.16**

RC_GTEST_PROP(HandleRegistryProperty21, ReleasedTokensNeverReissued, ()) {
    auto &reg = Handle_Registry::instance();

    const int cycles = *rc::gen::inRange(1, 51);

    int prev_token = HALO_HANDLE_INVALID;
    int dummy = 0;
    for (int i = 0; i < cycles; ++i) {
        const int token = reg.register_handle(static_cast<void *>(&dummy));

        // Strictly increasing -> never reissued, always positive.
        RC_ASSERT(token > prev_token);
        RC_ASSERT(token > HALO_HANDLE_INVALID);

        // Releasing returns our pointer and invalidates the token.
        const bool released_matches = (reg.release(token) == static_cast<void *>(&dummy));
        RC_ASSERT(released_matches);
        RC_ASSERT(!reg.valid(token));
        const bool lookup_is_null = (reg.lookup(token) == nullptr);
        RC_ASSERT(lookup_is_null);

        // Double release is safe and returns nullptr (token still gone).
        const bool double_release_null = (reg.release(token) == nullptr);
        RC_ASSERT(double_release_null);

        prev_token = token;
    }
}

// --- Property 21c: Invalid tokens always resolve to nullptr ------------------
// Feature: helm-halo-microlibrary, Property 21: Handle Registry Round-Trip
//
// For any non-positive token (zero or negative), lookup() returns nullptr and
// valid() returns false -- such tokens are never produced by the registry.
//
// **Validates: Requirements 14.3, 14.4, 14.16**

RC_GTEST_PROP(HandleRegistryProperty21, NonPositiveTokensAreInvalid, ()) {
    auto &reg = Handle_Registry::instance();

    // Generate any non-positive token: zero or negative.
    const int token = *rc::gen::inRange(-1000000, 1);  // [-1000000, 0]
    RC_ASSERT(token <= HALO_HANDLE_INVALID);

    const bool lookup_is_null = (reg.lookup(token) == nullptr);
    RC_ASSERT(lookup_is_null);
    RC_ASSERT(!reg.valid(token));
    const bool release_is_null = (reg.release(token) == nullptr);
    RC_ASSERT(release_is_null);
}
