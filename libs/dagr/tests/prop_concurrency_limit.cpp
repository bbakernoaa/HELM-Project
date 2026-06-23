// DAGR — prop_concurrency_limit.cpp
// Property 5: Concurrency-Limit Enforcement
//
// Validates: Requirements 3.5, 6.5, 13.5
//
// For any scheduling sequence with a generated max-concurrency value N in the
// range 1 to 64, the count of simultaneously dispatched-but-not-completed tasks
// SHALL never exceed N at any point during Event_Loop execution.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <dagr/detail/event_loop.hpp>
#include <dagr/detail/rank_pool.hpp>
#include <dagr/detail/task_node.hpp>
#include "generators.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <set>
#include <vector>

namespace {

/// Build TaskNode vector from a Generated_DAG, computing pending_deps from edges.
std::vector<dagr::detail::TaskNode> build_task_nodes(const dagr::gen::Generated_DAG& dag) {
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    std::vector<dagr::detail::TaskNode> nodes(node_count);

    for (std::uint32_t i = 0; i < node_count; ++i) {
        nodes[i].id = i;
        nodes[i].name = dag.task_names[i];
        nodes[i].status = dagr::detail::Task_Status::pending;
        nodes[i].required_ranks = 1;
        nodes[i].pending_deps.store(0, std::memory_order_relaxed);
    }

    // Build adjacency lists and compute in-degrees
    for (const auto& edge : dag.edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            nodes[edge.producer_id].dependents.push_back(edge.consumer_id);
            nodes[edge.consumer_id].pending_deps.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return nodes;
}

} // anonymous namespace

/// **Validates: Requirements 3.5, 6.5, 13.5**
RC_GTEST_PROP(ConcurrencyLimit, PeakInFlightNeverExceedsMaxConcurrency, ()) {
    // Generate a random acyclic DAG with 2–128 nodes, 1–512 edges
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);

    // Generate max_concurrency N ∈ [1, 64]
    auto max_conc = *dagr::gen::max_concurrency();

    // Build TaskNode vector from the generated DAG
    auto nodes = build_task_nodes(dag);
    const auto node_count = static_cast<std::uint32_t>(nodes.size());

    // Create a Rank_Pool with MORE ranks than nodes so rank availability
    // is never the bottleneck — we want concurrency to be the only limiter
    const std::uint32_t pool_size = node_count + 64;
    std::set<int> rank_ids;
    for (std::uint32_t i = 0; i < pool_size; ++i) {
        rank_ids.insert(static_cast<int>(i));
    }
    dagr::detail::Rank_Pool rank_pool(rank_ids);

    // Configure Event_Loop with the generated max_concurrency
    dagr::detail::Event_Loop::Config cfg;
    cfg.max_concurrency = max_conc;
    cfg.deadlock_timeout_s = 3600; // Effectively disable deadlock detection

    dagr::detail::Event_Loop loop(cfg, nodes, rank_pool);

    // Track peak in-flight count observed at any point
    std::uint32_t peak_in_flight = 0;

    // Queue of in-flight node IDs — we hold tasks for a delay before completing
    // to actually test concurrency pressure (don't complete immediately)
    std::deque<std::uint32_t> in_flight_queue;

    // Set up dispatch callback: record in-flight count, hold task in queue
    loop.set_dispatch_callback([&](std::uint32_t node_id, const std::set<int>& /*ranks*/) -> bool {
        in_flight_queue.push_back(node_id);

        // Track peak in-flight — check AFTER adding this new task
        std::uint32_t current_in_flight = loop.in_flight_count();
        peak_in_flight = std::max(peak_in_flight, current_in_flight);

        return true;
    });

    // Set up completion callback: release N/2 tasks per cycle (minimum 1)
    // This creates sustained concurrency pressure — tasks aren't all completed
    // immediately, forcing the scheduler to respect the cap
    const std::uint32_t release_per_cycle = std::max<std::uint32_t>(1, max_conc / 2);

    loop.set_completion_callback([&]() -> std::vector<std::pair<std::uint32_t, bool>> {
        std::vector<std::pair<std::uint32_t, bool>> completions;

        // Release up to release_per_cycle tasks from the front of the queue
        std::uint32_t to_release = std::min<std::uint32_t>(
            release_per_cycle,
            static_cast<std::uint32_t>(in_flight_queue.size()));

        for (std::uint32_t i = 0; i < to_release; ++i) {
            completions.emplace_back(in_flight_queue.front(), true);
            in_flight_queue.pop_front();
        }

        return completions;
    });

    // Run the Event_Loop until all tasks complete or we hit a safety limit
    // Safety limit: at most (node_count * 4) cycles to prevent infinite loops
    const std::uint32_t max_cycles = node_count * 4;
    std::uint32_t cycle_count = 0;

    while (!loop.all_complete() && cycle_count < max_cycles) {
        loop.cycle();

        // Also check peak after each cycle to capture any intermediate state
        std::uint32_t current_in_flight = loop.in_flight_count();
        peak_in_flight = std::max(peak_in_flight, current_in_flight);

        ++cycle_count;
    }

    // Drain any remaining in-flight tasks (complete everything)
    while (!in_flight_queue.empty()) {
        loop.cycle();
        ++cycle_count;

        // Safety guard
        if (cycle_count > max_cycles * 2) {
            break;
        }
    }

    // The core property: peak in-flight count MUST never exceed max_concurrency
    RC_ASSERT(peak_in_flight <= max_conc);

    // Sanity check: verify all nodes reached terminal state
    for (const auto& node : nodes) {
        RC_ASSERT(node.status == dagr::detail::Task_Status::completed
                || node.status == dagr::detail::Task_Status::failed
                || node.status == dagr::detail::Task_Status::cancelled);
    }
}
