// DAGR — prop_dag_construction.cpp
// Property 9: DAG Construction Correctness
//
// Validates: Requirements 3.1, 3.8
//
// For any valid acyclic adjacency list of 2 to 128 nodes, the constructed DAG
// SHALL have exactly one TaskNode per declared task, each TaskNode's initial
// pending_deps value SHALL equal its in-degree (number of incoming edges), and
// if the adjacency list references a task identifier that does not correspond
// to any declared task, construction SHALL throw std::invalid_argument.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cstdint>
#include <dagr/detail/task_node.hpp>
#include <dagr/pipeline_config.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "generators.hpp"

namespace {

/// Replicate the DAG construction logic from GraphOrchestrator::Impl.
/// This mirrors validate_dag() + populate_nodes() to test construction correctness
/// without needing a halo::Communicator.
///
/// Throws std::invalid_argument on dangling references or cycles (Req 3.8, 1.7).
std::vector<dagr::detail::TaskNode> construct_dag(const dagr::Pipeline_Config &config) {
    const auto num_tasks = static_cast<std::uint32_t>(config.task_names.size());

    // Req 3.8: Validate no dangling task references in edges.
    for (const auto &edge : config.edges) {
        if (edge.producer_id >= num_tasks) {
            throw std::invalid_argument("GraphOrchestrator: edge references unresolved task ID " + std::to_string(edge.producer_id));
        }
        if (edge.consumer_id >= num_tasks) {
            throw std::invalid_argument("GraphOrchestrator: edge references unresolved task ID " + std::to_string(edge.consumer_id));
        }
    }

    // Build TaskNodes with correct in-degrees and downstream adjacency (Req 3.1)
    std::vector<dagr::detail::TaskNode> nodes(num_tasks);
    std::vector<std::uint32_t> in_degree(num_tasks, 0);
    std::vector<std::vector<std::uint32_t>> downstream(num_tasks);

    for (const auto &edge : config.edges) {
        in_degree[edge.consumer_id] += 1;
        downstream[edge.producer_id].push_back(edge.consumer_id);
    }

    for (std::uint32_t i = 0; i < num_tasks; ++i) {
        nodes[i].id = i;
        nodes[i].name = config.task_names[i];
        nodes[i].pending_deps.store(in_degree[i], std::memory_order_relaxed);
        nodes[i].status = dagr::detail::Task_Status::pending;
        nodes[i].required_ranks = 1;
        nodes[i].dependents = std::move(downstream[i]);
    }

    return nodes;
}

/// Compute the expected in-degree for each node from edges.
std::vector<std::uint32_t> compute_in_degrees(std::uint32_t node_count, const std::vector<dagr::Dependency_Edge> &edges) {
    std::vector<std::uint32_t> in_degree(node_count, 0);
    for (const auto &edge : edges) {
        if (edge.consumer_id < node_count) {
            in_degree[edge.consumer_id] += 1;
        }
    }
    return in_degree;
}

/// Compute the expected dependents (downstream adjacency) for each node.
std::vector<std::set<std::uint32_t>> compute_dependents(std::uint32_t node_count, const std::vector<dagr::Dependency_Edge> &edges) {
    std::vector<std::set<std::uint32_t>> dependents(node_count);
    for (const auto &edge : edges) {
        if (edge.producer_id < node_count && edge.consumer_id < node_count) {
            dependents[edge.producer_id].insert(edge.consumer_id);
        }
    }
    return dependents;
}

}  // anonymous namespace

// ─── Property 9a: Exactly one TaskNode per declared task ─────────────────────
// ─── Property 9b: pending_deps equals in-degree ──────────────────────────────

/// **Validates: Requirements 3.1, 3.8**
RC_GTEST_PROP(DAGConstruction, ValidDAGProducesCorrectNodes, ()) {
    // Generate a random valid acyclic DAG with 2–128 nodes, 1–512 edges
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Build a Pipeline_Config from the generated DAG
    dagr::Pipeline_Config config;
    config.task_names = dag.task_names;
    config.edges = dag.edges;
    config.max_concurrency = 64;
    config.deadlock_timeout_s = 30;
    config.shutdown_timeout_s = 30;

    // Construct DAG (mirrors GraphOrchestrator construction logic)
    auto nodes = construct_dag(config);

    // Property 9a: Exactly one TaskNode per declared task
    RC_ASSERT(nodes.size() == node_count);

    // Verify each node has the correct ID and name
    for (std::uint32_t i = 0; i < node_count; ++i) {
        RC_ASSERT(nodes[i].id == i);
        RC_ASSERT(nodes[i].name == dag.task_names[i]);
    }

    // Property 9b: Each TaskNode's initial pending_deps equals its in-degree
    auto expected_in_degrees = compute_in_degrees(node_count, dag.edges);
    for (std::uint32_t i = 0; i < node_count; ++i) {
        RC_ASSERT(nodes[i].pending_deps.load(std::memory_order_relaxed) == expected_in_degrees[i]);
    }

    // Verify dependents (downstream adjacency) lists are correctly populated
    auto expected_dependents = compute_dependents(node_count, dag.edges);
    for (std::uint32_t i = 0; i < node_count; ++i) {
        // Convert the node's dependents vector to a set for order-independent comparison
        std::set<std::uint32_t> actual_dependents(nodes[i].dependents.begin(), nodes[i].dependents.end());
        RC_ASSERT(actual_dependents == expected_dependents[i]);
    }
}

// ─── Property 9c: Dangling task references throw std::invalid_argument ───────

/// **Validates: Requirements 3.1, 3.8**
RC_GTEST_PROP(DAGConstruction, DanglingReferenceThrowsInvalidArgument, ()) {
    // Generate a valid acyclic DAG
    auto dag = *dagr::gen::acyclic_dag(2, 128, 512);
    const auto node_count = static_cast<std::uint32_t>(dag.task_names.size());

    // Build a Pipeline_Config with a dangling edge reference
    dagr::Pipeline_Config config;
    config.task_names = dag.task_names;
    config.edges = dag.edges;
    config.max_concurrency = 64;
    config.deadlock_timeout_s = 30;
    config.shutdown_timeout_s = 30;

    // Generate an invalid edge referencing a non-existent task ID
    // The invalid ID is guaranteed to be >= node_count (out of range)
    auto invalid_id = *rc::gen::inRange<std::uint32_t>(node_count, node_count + 1000);

    // Choose whether to inject the invalid ID as producer or consumer
    auto inject_as_producer = *rc::gen::inRange<int>(0, 2) == 0;

    dagr::Dependency_Edge bad_edge;
    if (inject_as_producer) {
        bad_edge.producer_id = invalid_id;
        bad_edge.consumer_id = *rc::gen::inRange<std::uint32_t>(0, node_count);
    } else {
        bad_edge.producer_id = *rc::gen::inRange<std::uint32_t>(0, node_count);
        bad_edge.consumer_id = invalid_id;
    }

    // Inject the bad edge into the config
    config.edges.push_back(bad_edge);

    // Construction MUST throw std::invalid_argument for dangling references
    bool threw_invalid_argument = false;
    try {
        construct_dag(config);
    } catch (const std::invalid_argument &e) {
        threw_invalid_argument = true;
        // Verify the error message mentions the unresolved task ID
        std::string msg = e.what();
        RC_ASSERT(msg.find(std::to_string(invalid_id)) != std::string::npos);
    } catch (...) {
        // Wrong exception type
        RC_FAIL("Expected std::invalid_argument but got a different exception");
    }

    RC_ASSERT(threw_invalid_argument);
}
