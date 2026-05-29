// ─── CONF Handle_Registry Unit Tests ─────────────────────────────────────────
// Verifies the opaque handle registry correctly maps integer tokens to C++
// object pointers with register, lookup, release, and valid operations. Token 0
// is reserved as invalid (CONF_HANDLE_INVALID). Tokens are monotonically
// increasing positive integers that are never reused, even after release.
//
// The registry header lives in the PRIVATE src/ tree, so this file includes it
// via the src/ include path wired up by the PRIVATE_SRC flag in the test
// CMakeLists (see conf_add_test(test_handle_registry ... PRIVATE_SRC)).
//
// Feature: conf-config-parser
// Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.7
// ─────────────────────────────────────────────────────────────────────────────

#include "fortran/handle_registry.hpp"

#include <gtest/gtest.h>

using conf::fortran::Handle_Registry;
using conf::fortran::CONF_HANDLE_INVALID;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class HandleRegistryTest : public ::testing::Test {
protected:
    // Note: The registry is a singleton with monotonically increasing tokens.
    // Tokens are consumed across tests within a single process and are never
    // reused, so each test verifies relative behavior (uniqueness, ordering,
    // post-release invalidity) rather than absolute token values.
};

// ─── Registration (Req 18.1, 18.2) ───────────────────────────────────────────

TEST_F(HandleRegistryTest, RegisterReturnsPositiveToken) {
    int dummy = 42;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    // Req 18.1: a registered pointer yields a token strictly greater than 0.
    EXPECT_GT(token, CONF_HANDLE_INVALID);

    reg.release(token);
}

TEST_F(HandleRegistryTest, RegisterNeverIssuesInvalidSentinel) {
    // Req 18.2: token 0 (CONF_HANDLE_INVALID) is reserved and must never be
    // issued. Register a batch and confirm none collide with the sentinel.
    auto& reg = Handle_Registry::instance();
    int data[64];
    int tokens[64];

    for (int i = 0; i < 64; ++i) {
        tokens[i] = reg.register_handle(static_cast<void*>(&data[i]));
        EXPECT_NE(tokens[i], CONF_HANDLE_INVALID);
        EXPECT_GT(tokens[i], 0);
    }

    for (int i = 0; i < 64; ++i) {
        reg.release(tokens[i]);
    }
}

TEST_F(HandleRegistryTest, RegisterReturnsUniqueTokens) {
    int a = 1, b = 2, c = 3;
    auto& reg = Handle_Registry::instance();

    int t1 = reg.register_handle(static_cast<void*>(&a));
    int t2 = reg.register_handle(static_cast<void*>(&b));
    int t3 = reg.register_handle(static_cast<void*>(&c));

    // Req 18.1: each registration yields a distinct, positive token.
    EXPECT_NE(t1, t2);
    EXPECT_NE(t2, t3);
    EXPECT_NE(t1, t3);
    EXPECT_GT(t1, CONF_HANDLE_INVALID);
    EXPECT_GT(t2, CONF_HANDLE_INVALID);
    EXPECT_GT(t3, CONF_HANDLE_INVALID);

    reg.release(t1);
    reg.release(t2);
    reg.release(t3);
}

TEST_F(HandleRegistryTest, TokensAreMonotonicallyIncreasing) {
    int a = 1, b = 2, c = 3;
    auto& reg = Handle_Registry::instance();

    int t1 = reg.register_handle(static_cast<void*>(&a));
    int t2 = reg.register_handle(static_cast<void*>(&b));
    int t3 = reg.register_handle(static_cast<void*>(&c));

    EXPECT_LT(t1, t2);
    EXPECT_LT(t2, t3);

    reg.release(t1);
    reg.release(t2);
    reg.release(t3);
}

// ─── Lookup (Req 18.3) ───────────────────────────────────────────────────────

TEST_F(HandleRegistryTest, LookupReturnsRegisteredPointer) {
    int dummy = 99;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    // Req 18.3: a registered token maps back to the exact pointer supplied.
    EXPECT_EQ(reg.lookup(token), static_cast<void*>(&dummy));

    reg.release(token);
}

TEST_F(HandleRegistryTest, LookupReturnsNullptrForUnregisteredToken) {
    auto& reg = Handle_Registry::instance();

    // Req 18.3: any unregistered token resolves to a null pointer. This
    // includes the invalid sentinel, a never-issued large token, and a
    // negative token.
    EXPECT_EQ(reg.lookup(CONF_HANDLE_INVALID), nullptr);
    EXPECT_EQ(reg.lookup(999999), nullptr);
    EXPECT_EQ(reg.lookup(-1), nullptr);
}

TEST_F(HandleRegistryTest, LookupReturnsNullptrAfterRelease) {
    int dummy = 77;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    reg.release(token);

    // Req 18.3 / 18.4: a released token resolves to null on lookup.
    EXPECT_EQ(reg.lookup(token), nullptr);
}

TEST_F(HandleRegistryTest, LookupResolvesHeapTargets) {
    auto& reg = Handle_Registry::instance();
    auto* dummy = new int(123);
    int token = reg.register_handle(static_cast<void*>(dummy));

    EXPECT_EQ(reg.lookup(token), static_cast<void*>(dummy));

    // The caller owns the released pointer and is responsible for freeing it.
    void* released = reg.release(token);
    delete static_cast<int*>(released);
}

// ─── Release (Req 18.4, 18.7) ────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ReleaseReturnsPointerAndInvalidatesToken) {
    int dummy = 55;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    // Req 18.4: release returns the previously associated pointer and removes
    // the mapping, leaving the token permanently invalid.
    void* released = reg.release(token);
    EXPECT_EQ(released, static_cast<void*>(&dummy));

    EXPECT_FALSE(reg.valid(token));
    EXPECT_EQ(reg.lookup(token), nullptr);
}

TEST_F(HandleRegistryTest, ReleaseReturnsNullptrForUnknownToken) {
    auto& reg = Handle_Registry::instance();

    // Req 18.7: releasing the sentinel, an unissued token, or a negative token
    // performs no change and returns null.
    EXPECT_EQ(reg.release(CONF_HANDLE_INVALID), nullptr);
    EXPECT_EQ(reg.release(888888), nullptr);
    EXPECT_EQ(reg.release(-5), nullptr);
}

TEST_F(HandleRegistryTest, DoubleReleaseReturnsNullptr) {
    int dummy = 33;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    // Req 18.7: the first release returns the pointer; a second release of the
    // already-released token performs no change and returns null.
    void* first = reg.release(token);
    void* second = reg.release(token);

    EXPECT_EQ(first, static_cast<void*>(&dummy));
    EXPECT_EQ(second, nullptr);
}

// ─── Valid (Req 18.3, 18.4) ──────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ValidReturnsTrueForRegisteredToken) {
    int dummy = 11;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    EXPECT_TRUE(reg.valid(token));

    reg.release(token);
}

TEST_F(HandleRegistryTest, ValidReturnsFalseForUnregisteredToken) {
    auto& reg = Handle_Registry::instance();

    EXPECT_FALSE(reg.valid(CONF_HANDLE_INVALID));
    EXPECT_FALSE(reg.valid(777777));
    EXPECT_FALSE(reg.valid(-10));
}

TEST_F(HandleRegistryTest, ReleasedTokenIsInvalidForever) {
    int dummy = 22;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    EXPECT_TRUE(reg.valid(token));
    reg.release(token);

    // Req 18.4: a released token is permanently invalid for every subsequent
    // lookup and release. Hammer it to confirm it never resurrects.
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(reg.valid(token));
        EXPECT_EQ(reg.lookup(token), nullptr);
        EXPECT_EQ(reg.release(token), nullptr);
    }
}

// ─── Token Non-Reuse (Req 18.5) ──────────────────────────────────────────────

TEST_F(HandleRegistryTest, ReleasedTokensAreNeverReused) {
    auto& reg = Handle_Registry::instance();
    int a = 1, b = 2;

    // Req 18.5: after a register → release → register cycle, the new token must
    // differ from the released one (no reuse). The registry issues strictly
    // increasing tokens, so the replacement is also greater.
    int t1 = reg.register_handle(static_cast<void*>(&a));
    reg.release(t1);

    int t2 = reg.register_handle(static_cast<void*>(&b));
    EXPECT_NE(t1, t2);
    EXPECT_GT(t2, t1);

    reg.release(t2);
}

TEST_F(HandleRegistryTest, RepeatedReleaseRegisterCyclesNeverReuseTokens) {
    auto& reg = Handle_Registry::instance();
    int target = 7;
    int previous = CONF_HANDLE_INVALID;

    // Req 18.5: across many register/release/register cycles, every issued
    // token is strictly larger than the last; a released value is never reissued.
    for (int i = 0; i < 50; ++i) {
        int token = reg.register_handle(static_cast<void*>(&target));
        EXPECT_GT(token, previous);
        EXPECT_GT(token, CONF_HANDLE_INVALID);
        previous = token;
        reg.release(token);
    }
}

// ─── CONF_HANDLE_INVALID Constant (Req 18.2) ─────────────────────────────────

TEST_F(HandleRegistryTest, InvalidHandleConstantIsZero) {
    // Req 18.2: the invalid sentinel is 0.
    EXPECT_EQ(CONF_HANDLE_INVALID, 0);
}
