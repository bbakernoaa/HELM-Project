// DAGR — prop_topological_order.cpp
// Property 1: Topological-Order Invariant
//
// Validates: Requirements 3.1, 3.3, 3.4, 3.6, 13.1
//
// For any valid DAG of 2 to 128 nodes and 1 to 512 directed edges, every
// TaskNode's recorded completion sequence number SHALL be strictly greater
// than the completion sequence numbers of all its upstream dependencies.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <dagr/detail/event_loop.hpp>
#include <dagr/detail/rank_pool.hpp>
#include <dagr/detail/task_node.hpp>
#include <deque>
#include <set>
#include <vector>

#include "generators.hpp"

namespace {

/// Build TaskNode vector from a Generated_DAG, computing pending_deps from edges.
std::vector<dagr::detail::TaskNode> build_task_nodes(const dagr::gen::Generated_DAG &dag) {
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
    for (const auto &edge : dag.edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            nodes[edge.producer_id].dependents.push_back(edge.consumer_id);
            nodes[edge.consumer_id].pending_deps.fetch_add(1, std::memory_order_relaxed);
        }
    }

    return nodes;
}

/// Build the reverse adjacency list (upstream dependencies for each node).
/// Returns a vector where upstream[i] contains all producer node IDs for node i.
std::vector<std::vector<std::uint32_t>> build_upstream_map(const dagr::gen::Generated_DAG &dag) {
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());
    std::vector<std::vector<std::uint32_t>> upstream(node_count);

    for (const auto &edge : dag.edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            upstream[edge.consumer_id].push_back(edge.producer_id);
        }
    }

    return upstream;
}

}  // anonymous namespace

/// **Validates: Requirements 3.1, 3.3, 3.4, 3.6, 13.1**
RC_GTEST_PROP(TopologicalOrder, CompletionSequenceRespectsDAGOrder, ()) {
    // Generate a random acyclic DAG with 2–128 nodes, 1–512 edges
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Build TaskNodes from the generated DAG
    auto nodes = build_task_nodes(dag);

    // Build the upstream (reverse) adjacency map for verification
    auto upstream = build_upstream_map(dag);

    // Create a Rank_Pool with enough ranks so rank availability is never
    // the bottleneck — we want to observe pure topological ordering
    const std::uint32_t pool_size = node_count + 64;
    std::set<int> rank_ids;
    for (std::uint32_t i = 0; i < pool_size; ++i) {
        rank_ids.insert(static_cast<int>(i));
    }
    dagr::detail::Rank_Pool rank_pool(rank_ids);

    // Configure Event_Loop with generous concurrency to allow maximal parallelism
    dagr::detail::Event_Loop::Config cfg;
    cfg.max_concurrency = node_count;  // Allow all ready nodes to dispatch
    cfg.deadlock_timeout_s = 3600;     // Effectively disable deadlock detection

    dagr::detail::Event_Loop loop(cfg, nodes, rank_pool);

    // Atomic sequence counter to record completion ordering
    dagr::gen::Completion_Sequence_Counter seq_counter;

    // Per-node completion sequence numbers (0 = not yet completed)
    std::vector<std::uint64_t> completion_seq(node_count, 0);

    // Queue of dispatched nodes awaiting completion
    std::deque<std::uint32_t> in_flight_queue;

    // Set up dispatch callback: queue nodes for completion
    loop.set_dispatch_callback([&](std::uint32_t node_id, const std::set<int> & /*ranks*/) -> bool {
        in_flight_queue.push_back(node_id);
        return true;
    });

    // Set up completion callback: complete all in-flight tasks each cycle,
    // recording their sequence numbers via the atomic counter.
    // Completing all tasks per cycle maximizes parallelism and exercises
    // the topological ordering invariant under concurrent completions.
    loop.set_completion_callback([&]() -> std::vector<std::pair<std::uint32_t, bool>> {
        std::vector<std::pair<std::uint32_t, bool>> completions;

        while (!in_flight_queue.empty()) {
            std::uint32_t node_id = in_flight_queue.front();
            in_flight_queue.pop_front();

            // Record completion sequence number
            completion_seq[node_id] = seq_counter.record_completion();
            completions.emplace_back(node_id, true);
        }

        return completions;
    });

    // Run the Event_Loop until all tasks complete or we hit a safety limit
    const std::uint32_t max_cycles = node_count * 4;
    std::uint32_t cycle_count = 0;

    while (!loop.all_complete() && cycle_count < max_cycles) {
        loop.cycle();
        ++cycle_count;
    }

    // Sanity check: all nodes must have reached terminal state
    RC_ASSERT(loop.all_complete());

    // All nodes must have a recorded completion sequence number
    for (std::uint32_t i = 0; i < node_count; ++i) {
        RC_ASSERT(completion_seq[i] > 0);
    }

    // THE CORE PROPERTY: For every node, its completion sequence number must
    // be strictly greater than the completion sequence numbers of ALL its
    // upstream dependencies (producers).
    for (std::uint32_t consumer = 0; consumer < node_count; ++consumer) {
        for (std::uint32_t producer : upstream[consumer]) {
            RC_ASSERT(completion_seq[consumer] > completion_seq[producer]);
        }
    }
}
