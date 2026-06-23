// DAGR — test_rank_pool.cpp
// Unit tests for Rank_Pool allocation, release, and structural operations.
// Requirements: 5.4, 5.6, 5.9

#include <gtest/gtest.h>

#include "dagr/detail/rank_pool.hpp"

#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using dagr::detail::Rank_Pool;

// ─── Req 5.6: Zero-rank allocation request throws std::invalid_argument ──────

TEST(RankPool, ZeroCountAllocationThrowsInvalidArgument) {
    Rank_Pool pool({0, 1, 2, 3});

    EXPECT_THROW(pool.try_allocate(0), std::invalid_argument);
}

TEST(RankPool, ZeroCountAllocationMessageIsDescriptive) {
    Rank_Pool pool({10, 20, 30});

    try {
        [[maybe_unused]] auto result = pool.try_allocate(0);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty()) << "Error message should not be empty";
    }
}

// ─── Req 5.9: Allocate more than available returns empty set ─────────────────

TEST(RankPool, AllocateMoreThanAvailableReturnsEmptySet) {
    Rank_Pool pool({0, 1, 2});

    auto result = pool.try_allocate(4);

    EXPECT_TRUE(result.empty());
    // Pool state unchanged
    EXPECT_EQ(pool.available_ranks(), 3u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);
    EXPECT_EQ(pool.total_ranks(), 3u);
}

TEST(RankPool, AllocateExactlyOneMoreThanAvailableReturnsEmpty) {
    Rank_Pool pool({5, 10, 15, 20});

    // Allocate 2, leaving 2 available
    auto first = pool.try_allocate(2);
    ASSERT_EQ(first.size(), 2u);

    // Now try to allocate 3 from only 2 available
    auto result = pool.try_allocate(3);
    EXPECT_TRUE(result.empty());

    // Available should still be 2 (no partial allocation)
    EXPECT_EQ(pool.available_ranks(), 2u);
}

// ─── Req 5.9: Allocate-then-release restores availability ────────────────────

TEST(RankPool, AllocateThenReleaseRestoresAvailability) {
    Rank_Pool pool({0, 1, 2, 3, 4});

    EXPECT_EQ(pool.available_ranks(), 5u);
    EXPECT_EQ(pool.total_ranks(), 5u);

    auto allocated = pool.try_allocate(3);
    ASSERT_EQ(allocated.size(), 3u);
    EXPECT_EQ(pool.available_ranks(), 2u);
    EXPECT_EQ(pool.allocated_ranks(), 3u);

    // Invariant holds after allocation
    EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());

    pool.release(allocated);
    EXPECT_EQ(pool.available_ranks(), 5u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);

    // Invariant holds after release
    EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());
}

TEST(RankPool, AllocateAllThenReleaseAll) {
    Rank_Pool pool({7, 8, 9});

    auto all = pool.try_allocate(3);
    ASSERT_EQ(all.size(), 3u);
    EXPECT_EQ(pool.available_ranks(), 0u);

    // Allocating 1 more should fail
    auto extra = pool.try_allocate(1);
    EXPECT_TRUE(extra.empty());

    pool.release(all);
    EXPECT_EQ(pool.available_ranks(), 3u);
    EXPECT_EQ(pool.total_ranks(), 3u);
}

TEST(RankPool, AllocatedRanksAreFromPool) {
    std::set<int> initial_ranks = {10, 20, 30, 40, 50};
    Rank_Pool pool(initial_ranks);

    auto allocated = pool.try_allocate(3);
    ASSERT_EQ(allocated.size(), 3u);

    // All allocated ranks must be from the initial set
    for (int rank : allocated) {
        EXPECT_TRUE(initial_ranks.count(rank) > 0)
            << "Allocated rank " << rank << " not in initial pool";
    }
}

// ─── Req 5.9: add_ranks increases total and available ────────────────────────

TEST(RankPool, AddRanksIncreasesTotalAndAvailable) {
    Rank_Pool pool({0, 1});

    EXPECT_EQ(pool.total_ranks(), 2u);
    EXPECT_EQ(pool.available_ranks(), 2u);

    pool.add_ranks({5, 6, 7});

    EXPECT_EQ(pool.total_ranks(), 5u);
    EXPECT_EQ(pool.available_ranks(), 5u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);

    // Invariant holds
    EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());
}

TEST(RankPool, AddRanksDoesNotDuplicateExisting) {
    Rank_Pool pool({0, 1, 2});

    // Adding ranks that already exist should be a no-op for those
    pool.add_ranks({1, 2, 3, 4});

    // Only 3 and 4 are new
    EXPECT_EQ(pool.total_ranks(), 5u);
    EXPECT_EQ(pool.available_ranks(), 5u);
}

TEST(RankPool, AddRanksAfterAllocation) {
    Rank_Pool pool({0, 1});

    auto allocated = pool.try_allocate(2);
    ASSERT_EQ(allocated.size(), 2u);
    EXPECT_EQ(pool.available_ranks(), 0u);

    pool.add_ranks({10, 11});

    EXPECT_EQ(pool.total_ranks(), 4u);
    EXPECT_EQ(pool.available_ranks(), 2u);
    EXPECT_EQ(pool.allocated_ranks(), 2u);

    // Can now allocate from the newly added ranks
    auto more = pool.try_allocate(2);
    ASSERT_EQ(more.size(), 2u);
    EXPECT_EQ(pool.available_ranks(), 0u);
}

// ─── Req 5.9: remove_available only removes non-allocated ranks ──────────────

TEST(RankPool, RemoveAvailableOnlyRemovesNonAllocated) {
    Rank_Pool pool({0, 1, 2, 3, 4});

    // Allocate ranks {0, 1} (or whichever 2 the pool gives)
    auto allocated = pool.try_allocate(2);
    ASSERT_EQ(allocated.size(), 2u);

    // Try to remove all ranks — only the 3 available should be removed
    std::set<int> all_ranks = {0, 1, 2, 3, 4};
    auto removed = pool.remove_available(all_ranks);

    // Removed set should contain exactly the non-allocated ranks
    EXPECT_EQ(removed.size(), 3u);
    for (int r : removed) {
        EXPECT_EQ(allocated.count(r), 0u)
            << "Rank " << r << " was allocated but was removed";
    }

    // After removal, total should be reduced to just the allocated count
    EXPECT_EQ(pool.total_ranks(), 2u);
    EXPECT_EQ(pool.available_ranks(), 0u);
    EXPECT_EQ(pool.allocated_ranks(), 2u);

    // Invariant holds
    EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());
}

TEST(RankPool, RemoveAvailableReturnsActuallyRemoved) {
    Rank_Pool pool({0, 1, 2, 3});

    // Request removal of ranks not in the pool — nothing removed
    auto removed = pool.remove_available({10, 20, 30});
    EXPECT_TRUE(removed.empty());
    EXPECT_EQ(pool.total_ranks(), 4u);
}

TEST(RankPool, RemoveAvailablePartialMatch) {
    Rank_Pool pool({0, 1, 2, 3, 4});

    // Remove a subset of available ranks
    auto removed = pool.remove_available({1, 3});
    EXPECT_EQ(removed.size(), 2u);
    EXPECT_TRUE(removed.count(1) > 0);
    EXPECT_TRUE(removed.count(3) > 0);

    EXPECT_EQ(pool.total_ranks(), 3u);
    EXPECT_EQ(pool.available_ranks(), 3u);

    // Remaining ranks can still be allocated
    auto allocated = pool.try_allocate(3);
    EXPECT_EQ(allocated.size(), 3u);
}

// ─── Req 5.9: Conservation invariant across operations ───────────────────────

TEST(RankPool, InvariantHoldsAcrossMultipleOperations) {
    Rank_Pool pool({0, 1, 2, 3, 4, 5, 6, 7});

    // Check invariant at every step
    auto check_invariant = [&pool]() {
        EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());
    };

    check_invariant();

    auto a1 = pool.try_allocate(3);
    check_invariant();

    auto a2 = pool.try_allocate(2);
    check_invariant();

    pool.release(a1);
    check_invariant();

    pool.add_ranks({100, 101, 102});
    check_invariant();

    auto removed = pool.remove_available({0, 1, 100, 101});
    check_invariant();

    pool.release(a2);
    check_invariant();
}

// ─── Req 5.9: Empty pool construction ────────────────────────────────────────

TEST(RankPool, EmptyPoolConstruction) {
    Rank_Pool pool(std::set<int>{});

    EXPECT_EQ(pool.total_ranks(), 0u);
    EXPECT_EQ(pool.available_ranks(), 0u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);

    // Allocating from empty pool returns empty
    auto result = pool.try_allocate(1);
    EXPECT_TRUE(result.empty());
}

TEST(RankPool, DefaultConstruction) {
    Rank_Pool pool;

    EXPECT_EQ(pool.total_ranks(), 0u);
    EXPECT_EQ(pool.available_ranks(), 0u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);
}

// ─── Req 5.4: Concurrent allocation/release safety ───────────────────────────

TEST(RankPool, ConcurrentAllocateReleaseSafety) {
    // Create a pool with enough ranks for concurrent access
    std::set<int> ranks;
    for (int i = 0; i < 32; ++i) {
        ranks.insert(i);
    }
    Rank_Pool pool(ranks);

    constexpr int num_threads = 8;
    constexpr int iterations_per_thread = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                // Attempt to allocate 1 rank
                auto allocated = pool.try_allocate(1);
                if (!allocated.empty()) {
                    // Small work simulation
                    std::this_thread::yield();
                    // Release back
                    pool.release(allocated);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // After all threads complete, all ranks should be returned
    EXPECT_EQ(pool.available_ranks(), 32u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);
    EXPECT_EQ(pool.total_ranks(), 32u);

    // Conservation invariant holds
    EXPECT_EQ(pool.available_ranks() + pool.allocated_ranks(), pool.total_ranks());
}

TEST(RankPool, ConcurrentMultiRankAllocateRelease) {
    // Larger pool to exercise multi-rank concurrent allocation
    std::set<int> ranks;
    for (int i = 0; i < 64; ++i) {
        ranks.insert(i);
    }
    Rank_Pool pool(ranks);

    constexpr int num_threads = 4;
    constexpr int iterations_per_thread = 50;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                // Attempt to allocate 2-4 ranks
                auto allocated = pool.try_allocate(3);
                if (!allocated.empty()) {
                    std::this_thread::yield();
                    pool.release(allocated);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All ranks returned
    EXPECT_EQ(pool.available_ranks(), 64u);
    EXPECT_EQ(pool.allocated_ranks(), 0u);
    EXPECT_EQ(pool.total_ranks(), 64u);
}
