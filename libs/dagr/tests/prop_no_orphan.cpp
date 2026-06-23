// DAGR — prop_no_orphan.cpp
// Property 2: No-Orphan Invariant
//
// Validates: Requirements 3.7, 13.2
//
// For any valid DAG of 2 to 128 nodes and 1 to 512 directed edges, every
// TaskNode SHALL reach a terminal state (completed or cancelled) within a
// number of scheduling cycles no greater than the DAG's longest path length
// plus one.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <dagr/detail/event_loop.hpp>
#include <dagr/detail/rank_pool.hpp>
#include <dagr/detail/task_node.hpp>
#include "generators.hpp"

#include <algorithm>
#include <cstdint>
#include <queue>
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

/// Compute the longest path (critical path length) in the DAG.
/// Uses dynamic programming over topological order via Kahn's algorithm.
/// Returns the number of edges on the longest path (i.e., longest_path_length).
/// A single-node DAG has longest path 0. A chain of N nodes has longest path N-1.
std::uint32_t compute_longest_path(const dagr::gen::Generated_DAG& dag) {
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Build adjacency list and in-degree array
    std::vector<std::vector<std::uint32_t>> adj(node_count);
    std::vector<std::uint32_t> in_degree(node_count, 0);

    for (const auto& edge : dag.edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            adj[edge.producer_id].push_back(edge.consumer_id);
            ++in_degree[edge.consumer_id];
        }
    }

    // Kahn's algorithm with DP for longest path
    // dist[i] = length of longest path ending at node i (in edges)
    std::vector<std::uint32_t> dist(node_count, 0);
    std::queue<std::uint32_t> queue;

    for (std::uint32_t i = 0; i < node_count; ++i) {
        if (in_degree[i] == 0) {
            queue.push(i);
        }
    }

    while (!queue.empty()) {
        auto u = queue.front();
        queue.pop();

        for (std::uint32_t v : adj[u]) {
            dist[v] = std::max(dist[v], dist[u] + 1);
            if (--in_degree[v] == 0) {
                queue.push(v);
            }
        }
    }

    // The longest path is the maximum value in dist[]
    std::uint32_t longest = 0;
    for (std::uint32_t d : dist) {
        longest = std::max(longest, d);
    }

    return longest;
}

} // anonymous namespace

/// **Validates: Requirements 3.7, 13.2**
RC_GTEST_PROP(NoOrphan, AllNodesReachTerminalWithinLongestPathPlusOne, ()) {
    // Generate a random acyclic DAG with 2–128 nodes, 1–512 edges
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Compute the longest path (critical path) before building nodes
    std::uint32_t longest_path = compute_longest_path(dag);

    // Build TaskNode vector from the generated DAG
    auto nodes = build_task_nodes(dag);

    // Create a rank pool with enough ranks so rank availability never bottlenecks
    // max_concurrency is set to node_count so concurrency isn't the limiter either
    const std::uint32_t pool_size = node_count + 64;
    std::set<int> rank_ids;
    for (std::uint32_t i = 0; i < pool_size; ++i) {
        rank_ids.insert(static_cast<int>(i));
    }
    dagr::detail::Rank_Pool rank_pool(rank_ids);

    // Configure Event_Loop: max_concurrency = node_count (no concurrency bottleneck)
    // so the only limiting factor is dependency structure (longest path)
    dagr::detail::Event_Loop::Config cfg;
    cfg.max_concurrency = std::min<std::uint32_t>(node_count, 1024);
    cfg.deadlock_timeout_s = 3600; // Effectively disable deadlock detection

    dagr::detail::Event_Loop loop(cfg, nodes, rank_pool);

    // All dispatched tasks succeed immediately — the completion callback returns
    // them in the same cycle they were dispatched. This ensures scheduling speed
    // is limited only by the DAG's dependency depth (longest path).
    std::vector<std::uint32_t> dispatched_this_cycle;

    loop.set_dispatch_callback(
        [&](std::uint32_t node_id, const std::set<int>& /*ranks*/) -> bool {
            dispatched_this_cycle.push_back(node_id);
            return true;
        });

    loop.set_completion_callback(
        [&]() -> std::vector<std::pair<std::uint32_t, bool>> {
            std::vector<std::pair<std::uint32_t, bool>> completions;
            completions.reserve(dispatched_this_cycle.size());
            for (std::uint32_t id : dispatched_this_cycle) {
                completions.emplace_back(id, true);
            }
            dispatched_this_cycle.clear();
            return completions;
        });

    // Run the Event_Loop, counting scheduling cycles.
    // The maximum allowed is (longest_path + 1) cycles.
    const std::uint32_t max_allowed_cycles = longest_path + 1;

    // Safety limit to prevent infinite loops (generous: 2x allowed + padding)
    const std::uint32_t safety_limit = max_allowed_cycles * 2 + 10;
    std::uint32_t cycle_count = 0;

    while (!loop.all_complete() && cycle_count < safety_limit) {
        loop.cycle();
        ++cycle_count;
    }

    // Property assertion 1: ALL nodes must reach terminal state (no orphans)
    for (const auto& node : nodes) {
        RC_ASSERT(node.status == dagr::detail::Task_Status::completed
                || node.status == dagr::detail::Task_Status::failed
                || node.status == dagr::detail::Task_Status::cancelled);
    }

    // Property assertion 2: Completion must happen within (longest_path + 1) cycles
    // This verifies the scheduler is optimally progressing through dependency layers
    RC_ASSERT(cycle_count <= max_allowed_cycles);
}
