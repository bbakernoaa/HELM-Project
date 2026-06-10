// ─── HALO Handle_Registry Unit Tests ─────────────────────────────────────────
// Verifies the opaque handle registry correctly maps integer tokens to C++
// object pointers with thread-safe register, lookup, release, and valid
// operations. Token 0 is reserved as invalid (HALO_HANDLE_INVALID).
//
// Feature: helm-halo-microlibrary
// Requirements: 14.3, 14.4, 14.13
// ─────────────────────────────────────────────────────────────────────────────

#include "../src/fortran/handle_registry.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <set>
#include <algorithm>

using halo::fortran::Handle_Registry;
using halo::fortran::HALO_HANDLE_INVALID;

// ─── Test Fixture ────────────────────────────────────────────────────────────

class HandleRegistryTest : public ::testing::Test {
protected:
    // Note: The registry is a singleton with monotonically increasing tokens.
    // Tests must account for tokens being consumed across tests within a
    // single process. We verify relative behavior rather than absolute values.
};

// ─── Basic Registration Tests ────────────────────────────────────────────────

TEST_F(HandleRegistryTest, RegisterReturnsPositiveToken) {
    int dummy = 42;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    EXPECT_GT(token, HALO_HANDLE_INVALID);

    // Cleanup
    reg.release(token);
}

TEST_F(HandleRegistryTest, RegisterReturnsUniqueTokens) {
    int a = 1, b = 2, c = 3;
    auto& reg = Handle_Registry::instance();

    int t1 = reg.register_handle(static_cast<void*>(&a));
    int t2 = reg.register_handle(static_cast<void*>(&b));
    int t3 = reg.register_handle(static_cast<void*>(&c));

    EXPECT_NE(t1, t2);
    EXPECT_NE(t2, t3);
    EXPECT_NE(t1, t3);

    // All tokens are positive (not HALO_HANDLE_INVALID)
    EXPECT_GT(t1, HALO_HANDLE_INVALID);
    EXPECT_GT(t2, HALO_HANDLE_INVALID);
    EXPECT_GT(t3, HALO_HANDLE_INVALID);

    // Cleanup
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

    // Cleanup
    reg.release(t1);
    reg.release(t2);
    reg.release(t3);
}

// ─── Lookup Tests ────────────────────────────────────────────────────────────

TEST_F(HandleRegistryTest, LookupReturnsRegisteredPointer) {
    int dummy = 99;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    void* result = reg.lookup(token);
    EXPECT_EQ(result, static_cast<void*>(&dummy));

    // Cleanup
    reg.release(token);
}

TEST_F(HandleRegistryTest, LookupReturnsNullptrForInvalidToken) {
    auto& reg = Handle_Registry::instance();

    // Token 0 (HALO_HANDLE_INVALID) should never be registered
    EXPECT_EQ(reg.lookup(HALO_HANDLE_INVALID), nullptr);

    // A very large token that was never registered
    EXPECT_EQ(reg.lookup(999999), nullptr);

    // Negative tokens
    EXPECT_EQ(reg.lookup(-1), nullptr);
}

TEST_F(HandleRegistryTest, LookupReturnsNullptrAfterRelease) {
    int dummy = 77;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    reg.release(token);

    EXPECT_EQ(reg.lookup(token), nullptr);
}

// ─── Release Tests ───────────────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ReleaseReturnsPointerAndInvalidatesToken) {
    int dummy = 55;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    void* released = reg.release(token);
    EXPECT_EQ(released, static_cast<void*>(&dummy));

    // Token is now invalid
    EXPECT_FALSE(reg.valid(token));
    EXPECT_EQ(reg.lookup(token), nullptr);
}

TEST_F(HandleRegistryTest, ReleaseReturnsNullptrForUnknownToken) {
    auto& reg = Handle_Registry::instance();

    EXPECT_EQ(reg.release(HALO_HANDLE_INVALID), nullptr);
    EXPECT_EQ(reg.release(888888), nullptr);
    EXPECT_EQ(reg.release(-5), nullptr);
}

TEST_F(HandleRegistryTest, DoubleReleaseReturnsNullptr) {
    int dummy = 33;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    void* first = reg.release(token);
    void* second = reg.release(token);

    EXPECT_EQ(first, static_cast<void*>(&dummy));
    EXPECT_EQ(second, nullptr);
}

// ─── Valid Tests ─────────────────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ValidReturnsTrueForRegisteredToken) {
    int dummy = 11;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    EXPECT_TRUE(reg.valid(token));

    // Cleanup
    reg.release(token);
}

TEST_F(HandleRegistryTest, ValidReturnsFalseForInvalidToken) {
    auto& reg = Handle_Registry::instance();

    EXPECT_FALSE(reg.valid(HALO_HANDLE_INVALID));
    EXPECT_FALSE(reg.valid(777777));
    EXPECT_FALSE(reg.valid(-10));
}

TEST_F(HandleRegistryTest, ValidReturnsFalseAfterRelease) {
    int dummy = 22;
    auto& reg = Handle_Registry::instance();
    int token = reg.register_handle(static_cast<void*>(&dummy));

    EXPECT_TRUE(reg.valid(token));
    reg.release(token);
    EXPECT_FALSE(reg.valid(token));
}

// ─── Token Reuse Prevention ─────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ReleasedTokensAreNeverReused) {
    auto& reg = Handle_Registry::instance();
    int a = 1, b = 2;

    int t1 = reg.register_handle(static_cast<void*>(&a));
    reg.release(t1);

    // New registration should get a different (higher) token
    int t2 = reg.register_handle(static_cast<void*>(&b));
    EXPECT_NE(t1, t2);
    EXPECT_GT(t2, t1);

    // Cleanup
    reg.release(t2);
}

// ─── Thread Safety Tests ─────────────────────────────────────────────────────

TEST_F(HandleRegistryTest, ConcurrentRegistrationsProduceUniqueTokens) {
    auto& reg = Handle_Registry::instance();
    constexpr int num_threads = 8;
    constexpr int registrations_per_thread = 100;

    std::vector<std::vector<int>> thread_tokens(num_threads);
    std::vector<int> dummy_data(num_threads * registrations_per_thread);

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < registrations_per_thread; ++i) {
                int idx = t * registrations_per_thread + i;
                int token = reg.register_handle(static_cast<void*>(&dummy_data[idx]));
                thread_tokens[t].push_back(token);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Collect all tokens and verify uniqueness
    std::set<int> all_tokens;
    for (auto& tokens : thread_tokens) {
        for (int token : tokens) {
            EXPECT_GT(token, HALO_HANDLE_INVALID);
            auto [_, inserted] = all_tokens.insert(token);
            EXPECT_TRUE(inserted) << "Duplicate token: " << token;
        }
    }

    EXPECT_EQ(all_tokens.size(),
              static_cast<size_t>(num_threads * registrations_per_thread));

    // Cleanup
    for (int token : all_tokens) {
        reg.release(token);
    }
}

TEST_F(HandleRegistryTest, ConcurrentLookupAndReleaseAreThreadSafe) {
    auto& reg = Handle_Registry::instance();
    constexpr int num_handles = 50;

    // Register handles
    std::vector<int> tokens;
    std::vector<int> dummy_data(num_handles);
    for (int i = 0; i < num_handles; ++i) {
        tokens.push_back(reg.register_handle(static_cast<void*>(&dummy_data[i])));
    }

    // Concurrent lookups and releases
    std::vector<std::thread> threads;

    // Half the threads do lookups
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int token : tokens) {
                // lookup may return nullptr if another thread released it
                reg.lookup(token);
            }
        });
    }

    // Other half release handles
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread releases a subset
            for (int i = t; i < num_handles; i += 4) {
                reg.release(tokens[i]);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // All handles should be released (some may have been released by multiple
    // threads, but double-release returns nullptr safely)
    for (int token : tokens) {
        EXPECT_FALSE(reg.valid(token));
    }
}

// ─── HALO_HANDLE_INVALID Constant Test ──────────────────────────────────────

TEST_F(HandleRegistryTest, InvalidHandleConstantIsZero) {
    EXPECT_EQ(HALO_HANDLE_INVALID, 0);
}
