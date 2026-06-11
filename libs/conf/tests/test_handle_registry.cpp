// ─── Unit Tests: Handle_Registry ─────────────────────────────────────────────
// Validates the thread-safe opaque handle registry that maps integer tokens to
// C++ object pointers for Fortran C-interop.
//
// Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.7
// ──────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include "handle_registry.hpp"

#include <set>

using conf::fortran::Handle_Registry;
using conf::fortran::CONF_HANDLE_INVALID;

// ─── Test 1: Register a pointer → token > 0 ─────────────────────────────────
// Requirement 18.1: registering a Config pointer returns a unique integer token
// greater than 0.
TEST(HandleRegistry, RegisterReturnsPositiveToken) {
    auto& reg = Handle_Registry::instance();
    int dummy = 42;
    int token = reg.register_handle(static_cast<void*>(&dummy));
    EXPECT_GT(token, 0);
    // Clean up
    reg.release(token);
}

// ─── Test 2: Lookup registered token → returns the pointer ───────────────────
// Requirement 18.3: looking up a currently registered token returns the
// associated pointer.
TEST(HandleRegistry, LookupRegisteredTokenReturnsPointer) {
    auto& reg = Handle_Registry::instance();
    int dummy = 99;
    void* ptr = static_cast<void*>(&dummy);
    int token = reg.register_handle(ptr);

    void* result = reg.lookup(token);
    EXPECT_EQ(result, ptr);

    reg.release(token);
}

// ─── Test 3: Release → returns pointer, then lookup → nullptr, valid → false ─
// Requirement 18.4: releasing a registered token removes the mapping, returns
// the previously associated pointer, and treats that token as permanently
// invalid for subsequent lookup and release.
TEST(HandleRegistry, ReleaseReturnsPointerThenInvalid) {
    auto& reg = Handle_Registry::instance();
    double dummy = 3.14;
    void* ptr = static_cast<void*>(&dummy);
    int token = reg.register_handle(ptr);

    // Release returns the pointer
    void* released = reg.release(token);
    EXPECT_EQ(released, ptr);

    // After release, lookup returns nullptr
    EXPECT_EQ(reg.lookup(token), nullptr);

    // After release, valid returns false
    EXPECT_FALSE(reg.valid(token));
}

// ─── Test 4: A released token remains invalid forever ────────────────────────
// Requirement 18.4: a released token is permanently invalid for every
// subsequent lookup and release.
// Requirement 18.7: releasing an already-released token returns nullptr and
// performs no mapping change.
TEST(HandleRegistry, ReleasedTokenInvalidForever) {
    auto& reg = Handle_Registry::instance();
    char dummy = 'x';
    int token = reg.register_handle(static_cast<void*>(&dummy));
    reg.release(token);

    // Multiple subsequent checks all confirm permanent invalidity
    EXPECT_FALSE(reg.valid(token));
    EXPECT_EQ(reg.lookup(token), nullptr);

    // Re-releasing yields nullptr (no mapping change)
    EXPECT_EQ(reg.release(token), nullptr);

    // Still invalid after the second release attempt
    EXPECT_FALSE(reg.valid(token));
    EXPECT_EQ(reg.lookup(token), nullptr);
}

// ─── Test 5: Token 0 (CONF_HANDLE_INVALID) is never issued ──────────────────
// Requirement 18.2: token 0 is reserved as the invalid sentinel and is never
// issued as a valid token.
TEST(HandleRegistry, TokenZeroNeverIssued) {
    auto& reg = Handle_Registry::instance();
    constexpr int N = 20;
    int tokens[N];
    int dummies[N];

    for (int i = 0; i < N; ++i) {
        tokens[i] = reg.register_handle(static_cast<void*>(&dummies[i]));
        EXPECT_NE(tokens[i], CONF_HANDLE_INVALID)
            << "Token 0 was issued on registration #" << i;
        EXPECT_GT(tokens[i], 0);
    }

    // Clean up
    for (int i = 0; i < N; ++i) {
        reg.release(tokens[i]);
    }
}

// ─── Test 5b: Lookup on token 0 returns nullptr ─────────────────────────────
// Requirement 18.2: token 0 is the invalid sentinel; lookup always returns
// nullptr and valid() always returns false.
TEST(HandleRegistry, LookupTokenZeroReturnsNullptr) {
    auto& reg = Handle_Registry::instance();
    EXPECT_EQ(reg.lookup(CONF_HANDLE_INVALID), nullptr);
    EXPECT_EQ(reg.lookup(0), nullptr);
    EXPECT_FALSE(reg.valid(0));
}

// ─── Test 6: Tokens are never reused ────────────────────────────────────────
// Requirement 18.5: distinct tokens across the lifetime of a run; never reuse
// a previously issued token, including released tokens.
TEST(HandleRegistry, TokensNeverReused) {
    auto& reg = Handle_Registry::instance();
    int dummy1 = 1;
    int dummy2 = 2;

    // Register → release → register again: new token != old token
    int token1 = reg.register_handle(static_cast<void*>(&dummy1));
    reg.release(token1);

    int token2 = reg.register_handle(static_cast<void*>(&dummy2));
    EXPECT_NE(token1, token2)
        << "Token was reused after register/release/register cycle";
    EXPECT_GT(token2, 0);

    // The old token remains invalid even after a new registration
    EXPECT_FALSE(reg.valid(token1));
    EXPECT_EQ(reg.lookup(token1), nullptr);

    reg.release(token2);
}

// ─── Test 7: Multiple simultaneous registrations get unique, monotonic tokens─
// Requirement 18.1, 18.5: each registration gets a unique token; all issued
// tokens are distinct and monotonically increasing.
TEST(HandleRegistry, MultipleRegistrationsGetUniqueMonotonicTokens) {
    auto& reg = Handle_Registry::instance();
    constexpr int N = 50;
    int dummies[N];
    std::set<int> token_set;
    int prev_token = 0;

    for (int i = 0; i < N; ++i) {
        int token = reg.register_handle(static_cast<void*>(&dummies[i]));
        EXPECT_GT(token, 0);
        EXPECT_GT(token, prev_token)
            << "Token " << token << " is not greater than previous " << prev_token
            << " on registration #" << i;
        auto [_, inserted] = token_set.insert(token);
        EXPECT_TRUE(inserted)
            << "Duplicate token " << token << " issued on registration #" << i;
        prev_token = token;
    }

    EXPECT_EQ(token_set.size(), static_cast<size_t>(N));

    // Clean up
    for (int token : token_set) {
        reg.release(token);
    }
}

// ─── Test 8: Register/release/register cycle never reuses the released token ─
// Requirement 18.5: a released token is never reissued to a new registration.
// This test exercises multiple cycles to confirm robustness.
TEST(HandleRegistry, RegisterReleaseCycleNeverReusesToken) {
    auto& reg = Handle_Registry::instance();
    std::set<int> all_tokens_ever_issued;
    constexpr int CYCLES = 10;
    int dummies[CYCLES];

    for (int i = 0; i < CYCLES; ++i) {
        int token = reg.register_handle(static_cast<void*>(&dummies[i]));
        EXPECT_GT(token, 0);
        // Confirm this token was never issued before in any previous cycle
        auto [_, inserted] = all_tokens_ever_issued.insert(token);
        EXPECT_TRUE(inserted)
            << "Token " << token << " was reused on cycle #" << i;
        // Release immediately so it could theoretically be reused
        reg.release(token);
    }

    // All released tokens remain permanently invalid
    for (int token : all_tokens_ever_issued) {
        EXPECT_FALSE(reg.valid(token));
        EXPECT_EQ(reg.lookup(token), nullptr);
    }
}
