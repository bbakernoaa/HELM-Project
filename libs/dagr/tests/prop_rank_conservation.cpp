// DAGR — prop_rank_conservation.cpp
// Property 3: Rank-Conservation Invariant
//
// Validates: Requirements 5.1, 5.2, 5.3, 5.9, 13.3
//
// For any scheduling sequence with a generated Rank_Pool of 1 to 64 total
// ranks, the value returned by available_ranks() plus the count of ranks
// currently allocated to in-flight tasks SHALL equal total_ranks() immediately
// before and after every dispatch and completion event.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cstdint>
#include <dagr/detail/rank_pool.hpp>
#include <set>
#include <vector>

#include "generators.hpp"

namespace {

/// Operation types in a scheduling sequence.
enum class Op_Type : std::uint8_t { allocate = 0, release = 1, add_ranks = 2, remove_available = 3 };

/// Verify the rank conservation invariant holds.
void assert_invariant(const dagr::detail::Rank_Pool &pool) {
    RC_ASSERT(pool.available_ranks() + pool.allocated_ranks() == pool.total_ranks());
}

}  // anonymous namespace

/// **Validates: Requirements 5.1, 5.2, 5.3, 5.9, 13.3**
RC_GTEST_PROP(RankConservation, InvariantHoldsAcrossAllOperations, ()) {
    // Generate initial pool size in [1, 64]
    auto pool_size = *dagr::gen::rank_pool_size();

    // Construct initial rank IDs as {0, 1, ..., pool_size-1}
    std::set<int> initial_ranks;
    for (std::uint32_t i = 0; i < pool_size; ++i) {
        initial_ranks.insert(static_cast<int>(i));
    }

    dagr::detail::Rank_Pool pool(initial_ranks);

    // Invariant must hold immediately after construction
    assert_invariant(pool);

    // Track which ranks are currently allocated (so we know what can be released)
    std::set<int> currently_allocated;

    // Next rank ID to use when adding new ranks (start beyond initial set)
    int next_rank_id = static_cast<int>(pool_size);

    // Generate a random sequence of 10–50 operations
    auto op_count = *rc::gen::inRange(10, 51);

    for (int op_idx = 0; op_idx < op_count; ++op_idx) {
        // Choose a random operation
        auto op_type = static_cast<Op_Type>(*rc::gen::inRange<int>(0, 4));

        switch (op_type) {
            case Op_Type::allocate: {
                // Only attempt allocation if there are available ranks
                if (pool.available_ranks() > 0) {
                    auto count = *rc::gen::inRange<std::uint32_t>(1, pool.available_ranks() + 1);

                    auto allocated = pool.try_allocate(count);

                    // If allocation succeeded, track the allocated ranks
                    if (!allocated.empty()) {
                        for (int rank : allocated) {
                            currently_allocated.insert(rank);
                        }
                    }
                }
                break;
            }

            case Op_Type::release: {
                // Only release if we have allocated ranks
                if (!currently_allocated.empty()) {
                    // Pick a random subset of currently allocated ranks to release
                    std::vector<int> allocated_vec(currently_allocated.begin(), currently_allocated.end());

                    auto release_count = *rc::gen::inRange<std::size_t>(1, allocated_vec.size() + 1);

                    // Shuffle and take first N elements as the subset to release
                    std::set<int> to_release;
                    std::vector<std::size_t> indices(allocated_vec.size());
                    for (std::size_t i = 0; i < indices.size(); ++i) {
                        indices[i] = i;
                    }

                    // Use RapidCheck to select random indices
                    for (std::size_t i = 0; i < release_count; ++i) {
                        auto pick = *rc::gen::inRange<std::size_t>(0, indices.size());
                        to_release.insert(allocated_vec[indices[pick]]);
                        // Remove picked index (swap with last)
                        indices[pick] = indices.back();
                        indices.pop_back();
                    }

                    pool.release(to_release);
                    for (int rank : to_release) {
                        currently_allocated.erase(rank);
                    }
                }
                break;
            }

            case Op_Type::add_ranks: {
                // Add 1–8 new ranks to the pool
                auto add_count = *rc::gen::inRange(1, 9);
                std::set<int> new_ranks;
                for (int i = 0; i < add_count; ++i) {
                    new_ranks.insert(next_rank_id++);
                }

                pool.add_ranks(new_ranks);
                break;
            }

            case Op_Type::remove_available: {
                // Only remove if there are available ranks
                if (pool.available_ranks() > 0) {
                    // We need to know which ranks are available to attempt removal.
                    // Available ranks = all ranks in pool minus currently_allocated.
                    // Since we don't have direct access to pool internals, we infer
                    // from our tracking: ranks in the pool are [initial + added] - removed.
                    // We'll just attempt to remove some ranks we know aren't allocated.

                    // Build set of ranks we believe are available (not allocated)
                    // Use a heuristic: try removing up to 4 ranks from range [0, next_rank_id)
                    auto remove_count = *rc::gen::inRange(1, 5);
                    std::set<int> candidates;

                    for (int i = 0; i < remove_count; ++i) {
                        auto rank_id = *rc::gen::inRange(0, next_rank_id);
                        // Only attempt to remove ranks we know aren't allocated
                        if (currently_allocated.find(rank_id) == currently_allocated.end()) {
                            candidates.insert(rank_id);
                        }
                    }

                    if (!candidates.empty()) {
                        pool.remove_available(candidates);
                        // remove_available only removes ranks that are actually
                        // in the pool AND available — it's safe to call with
                        // ranks that don't exist (they'll just not be removed).
                    }
                }
                break;
            }
        }

        // Invariant MUST hold after every operation (Req 5.9)
        assert_invariant(pool);
    }
}
