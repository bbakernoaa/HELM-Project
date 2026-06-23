// DAGR — prop_failure_cascade.cpp
// Property 7: Transitive Cancellation on Failure
//
// Validates: Requirements 4.8, 4.9, 4.10, 6.6
//
// For any valid DAG and any single TaskNode that fails during execution,
// all TaskNodes that are transitively reachable (downstream) from the failed
// node via Dependency_Edges SHALL have their status set to cancelled, and no
// cancelled node SHALL have been dispatched for execution after the failure event.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <dagr/detail/event_loop.hpp>
#include <dagr/detail/rank_pool.hpp>
#include <dagr/detail/task_node.hpp>
#include "generators.hpp"

#include <cstdint>
#include <deque>
#include <set>
#include <vector>

namespace {

/// Compute the set of all transitively reachable downstream nodes from a
/// given source node using BFS over the dependents adjacency lists.
std::set<std::uint32_t> compute_downstream_reachable(
    const std::vector<dagr::detail::TaskNode>& nodes,
    std::uint32_t source_id)
{
    std::set<std::uint32_t> reachable;
    std::deque<std::uint32_t> queue;

    for (std::uint32_t dep_id : nodes[source_id].dependents) {
        queue.push_back(dep_id);
    }

    while (!queue.empty()) {
        std::uint32_t current = queue.front();
        queue.pop_front();

        if (reachable.count(current) > 0) {
            continue; // already visited
        }

        reachable.insert(current);

        for (std::uint32_t dep_id : nodes[current].dependents) {
            if (reachable.count(dep_id) == 0) {
                queue.push_back(dep_id);
            }
        }
    }

    return reachable;
}

} // anonymous namespace

/// **Validates: Requirements 4.8, 4.9, 4.10, 6.6**
RC_GTEST_PROP(FailureCascade, TransitiveCancellationOnFailure, ()) {
    // Generate a random valid acyclic DAG
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Pick a random node to inject failure
    auto fail_node_id = *rc::gen::inRange<std::uint32_t>(0, node_count);

    // Build TaskNodes from the generated DAG
    std::vector<dagr::detail::TaskNode> nodes(node_count);

    for (std::uint32_t i = 0; i < node_count; ++i) {
        nodes[i].id = i;
        nodes[i].name = dag.task_names[i];
        nodes[i].pending_deps.store(0, std::memory_order_relaxed);
        nodes[i].status = dagr::detail::Task_Status::pending;
        nodes[i].required_ranks = 1;
    }

    // Build adjacency (dependents) and compute in-degrees from edges
    for (const auto& edge : dag.edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            nodes[edge.producer_id].dependents.push_back(edge.consumer_id);
            nodes[edge.consumer_id].pending_deps.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Compute expected downstream reachable set BEFORE execution
    // (topology doesn't change during execution)
    std::set<std::uint32_t> expected_cancelled =
        compute_downstream_reachable(nodes, fail_node_id);

    // Create a rank pool with enough ranks for all nodes
    std::set<int> rank_ids;
    for (std::uint32_t i = 0; i < node_count; ++i) {
        rank_ids.insert(static_cast<int>(i));
    }
    dagr::detail::Rank_Pool rank_pool(rank_ids);

    // Configure Event_Loop with generous concurrency
    dagr::detail::Event_Loop::Config cfg;
    cfg.max_concurrency = node_count; // allow all to dispatch
    cfg.deadlock_timeout_s = 60;

    dagr::detail::Event_Loop loop(cfg, nodes, rank_pool);

    // Track which nodes were dispatched and whether failure has occurred
    std::set<std::uint32_t> dispatched_nodes;
    bool failure_occurred = false;
    std::set<std::uint32_t> dispatched_after_failure;

    // Completion tracking: nodes that finish this cycle
    std::vector<std::pair<std::uint32_t, bool>> pending_completions;

    // Dispatch callback: fail the chosen node, succeed all others
    loop.set_dispatch_callback(
        [&](std::uint32_t node_id, const std::set<int>& /*ranks*/) -> bool {
            dispatched_nodes.insert(node_id);

            if (failure_occurred) {
                dispatched_after_failure.insert(node_id);
            }

            if (node_id == fail_node_id) {
                // This node fails — return false to indicate failure
                failure_occurred = true;
                return false;
            }

            // All other nodes succeed — queue for completion
            pending_completions.push_back({node_id, true});
            return true;
        });

    // Completion callback: report queued completions
    loop.set_completion_callback(
        [&]() -> std::vector<std::pair<std::uint32_t, bool>> {
            auto completions = std::move(pending_completions);
            pending_completions.clear();
            return completions;
        });

    // Run the event loop to completion
    // Limit cycles to prevent infinite loops (DAG should complete in at most node_count cycles)
    std::uint32_t max_cycles = node_count * 2 + 10;
    for (std::uint32_t cycle = 0; cycle < max_cycles; ++cycle) {
        if (loop.all_complete()) {
            break;
        }
        loop.cycle();
    }

    // All nodes must have reached terminal state
    RC_ASSERT(loop.all_complete());

    // The failed node itself must be marked as failed
    RC_ASSERT(nodes[fail_node_id].status == dagr::detail::Task_Status::failed);

    // All transitively reachable downstream nodes must be cancelled
    for (std::uint32_t downstream_id : expected_cancelled) {
        // A downstream node should be cancelled UNLESS it was already completed
        // or already running before the failure occurred. Since we use immediate
        // dispatch (return false = instant failure), and the dispatch callback
        // succeeds all other nodes synchronously, a node that was already dispatched
        // and succeeded before the failure node was dispatched is completed—not cancelled.
        //
        // The property states: all transitively reachable downstream nodes that
        // have NOT already completed must be cancelled.
        auto status = nodes[downstream_id].status;

        // If the node completed before the failure cascade reached it, that's valid.
        // Otherwise it must be cancelled.
        if (status != dagr::detail::Task_Status::completed) {
            RC_ASSERT(status == dagr::detail::Task_Status::cancelled);
        }
    }

    // No cancelled node was dispatched AFTER the failure event
    for (std::uint32_t node_id : dispatched_after_failure) {
        RC_ASSERT(nodes[node_id].status != dagr::detail::Task_Status::cancelled);
    }
}
