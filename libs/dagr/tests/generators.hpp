#pragma once

/// @file generators.hpp
/// @brief Custom RapidCheck generators for DAGR property-based tests.
///
/// Provides constrained generators that produce valid DAG topologies,
/// Pipeline_Config values, and Stream_Descriptors for scheduling invariant
/// verification. Uses layered construction to guarantee acyclicity by design.
/// Requirements: 13.7, 13.8, 13.9

#include <rapidcheck.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <dagr/pipeline_config.hpp>
#include <set>
#include <string>
#include <tick/duration.hpp>
#include <vector>

namespace dagr::gen {

// ─── DAG Topology Generators ─────────────────────────────────────────────────

/// Generated DAG result containing task names and edges.
/// Provides a valid acyclic topology suitable for Pipeline_Config construction.
struct Generated_DAG {
    std::vector<std::string> task_names;  ///< One name per node (index = node ID)
    std::vector<Dependency_Edge> edges;   ///< Directed edges (producer → consumer)
};

/// Generator for random acyclic DAGs via layered construction.
///
/// Nodes are assigned to topological layers [0, L-1] where L is in [2, 32].
/// Edges are only added from lower-layer nodes to higher-layer nodes,
/// guaranteeing acyclicity by construction (Req 13.8).
///
/// @param min_nodes  Minimum number of nodes (≥ 2)
/// @param max_nodes  Maximum number of nodes (≤ 128)
/// @param max_edges  Maximum number of edges to generate (≤ 512)
inline auto acyclic_dag(std::uint32_t min_nodes, std::uint32_t max_nodes, std::uint32_t max_edges) {
    return rc::gen::exec([min_nodes, max_nodes, max_edges] {
        // Generate node count
        auto node_count = *rc::gen::inRange<std::uint32_t>(min_nodes, max_nodes + 1);

        // Generate number of layers in [2, min(32, node_count)]
        auto max_layers = std::min<std::uint32_t>(32, node_count);
        auto layer_count = *rc::gen::inRange<std::uint32_t>(2, max_layers + 1);

        // Assign each node to a random layer
        std::vector<std::uint32_t> node_layer(node_count);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            node_layer[i] = *rc::gen::inRange<std::uint32_t>(0, layer_count);
        }

        // Ensure at least one node in layer 0 and one in the highest layer
        // so the DAG has proper roots and leaves
        node_layer[0] = 0;
        node_layer[node_count - 1] = layer_count - 1;

        // Build task names
        std::vector<std::string> task_names;
        task_names.reserve(node_count);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            task_names.push_back("task_" + std::to_string(i));
        }

        // Collect candidate edges: all pairs (i, j) where node_layer[i] < node_layer[j]
        // Then sample up to max_edges from them
        std::vector<Dependency_Edge> edges;

        // Generate edges by randomly sampling from valid layer pairs
        auto edge_count = *rc::gen::inRange<std::uint32_t>(1, std::min(max_edges, node_count * (node_count - 1) / 2) + 1);

        // Use a set to avoid duplicate edges
        std::set<std::pair<std::uint32_t, std::uint32_t>> edge_set;

        for (std::uint32_t attempt = 0; attempt < edge_count * 3 && edge_set.size() < edge_count; ++attempt) {
            auto src = *rc::gen::inRange<std::uint32_t>(0, node_count);
            auto dst = *rc::gen::inRange<std::uint32_t>(0, node_count);

            // Only add edge from lower layer to higher layer (guarantees acyclicity)
            if (node_layer[src] < node_layer[dst]) {
                edge_set.emplace(src, dst);
            }
        }

        edges.reserve(edge_set.size());
        for (const auto &[src, dst] : edge_set) {
            edges.push_back(Dependency_Edge{src, dst});
        }

        // Ensure at least one edge exists
        if (edges.empty()) {
            // Find any two nodes in different layers
            for (std::uint32_t i = 0; i < node_count && edges.empty(); ++i) {
                for (std::uint32_t j = 0; j < node_count && edges.empty(); ++j) {
                    if (node_layer[i] < node_layer[j]) {
                        edges.push_back(Dependency_Edge{i, j});
                    }
                }
            }
        }

        return Generated_DAG{std::move(task_names), std::move(edges)};
    });
}

/// Generator for directed graphs guaranteed to contain at least one cycle.
///
/// Constructs an acyclic DAG via layered construction, then adds a single
/// back-edge from a higher-layer node to a lower-layer node (or same layer),
/// introducing a cycle.
///
/// @param min_nodes  Minimum number of nodes (≥ 2)
/// @param max_nodes  Maximum number of nodes (≤ 128)
inline auto cyclic_graph(std::uint32_t min_nodes, std::uint32_t max_nodes) {
    return rc::gen::exec([min_nodes, max_nodes] {
        // Generate node count
        auto node_count = *rc::gen::inRange<std::uint32_t>(min_nodes, max_nodes + 1);

        // Generate number of layers in [2, min(32, node_count)]
        auto max_layers = std::min<std::uint32_t>(32, node_count);
        auto layer_count = *rc::gen::inRange<std::uint32_t>(2, max_layers + 1);

        // Assign each node to a random layer
        std::vector<std::uint32_t> node_layer(node_count);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            node_layer[i] = *rc::gen::inRange<std::uint32_t>(0, layer_count);
        }
        node_layer[0] = 0;
        node_layer[node_count - 1] = layer_count - 1;

        // Build task names
        std::vector<std::string> task_names;
        task_names.reserve(node_count);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            task_names.push_back("task_" + std::to_string(i));
        }

        // Generate a few forward edges
        std::vector<Dependency_Edge> edges;
        std::set<std::pair<std::uint32_t, std::uint32_t>> edge_set;

        auto edge_count = *rc::gen::inRange<std::uint32_t>(1, node_count + 1);
        for (std::uint32_t attempt = 0; attempt < edge_count * 3 && edge_set.size() < edge_count; ++attempt) {
            auto src = *rc::gen::inRange<std::uint32_t>(0, node_count);
            auto dst = *rc::gen::inRange<std::uint32_t>(0, node_count);
            if (node_layer[src] < node_layer[dst]) {
                edge_set.emplace(src, dst);
            }
        }

        for (const auto &[src, dst] : edge_set) {
            edges.push_back(Dependency_Edge{src, dst});
        }

        // Add a back-edge to introduce a cycle: from a higher-layer node to
        // a lower-layer node (or same-layer for self-loop avoidance, pick
        // strictly higher → lower)
        std::uint32_t back_src = node_count - 1;  // highest layer
        std::uint32_t back_dst = 0;               // lowest layer

        // Try to find a random back-edge pair
        for (std::uint32_t attempt = 0; attempt < node_count * 2; ++attempt) {
            auto s = *rc::gen::inRange<std::uint32_t>(0, node_count);
            auto d = *rc::gen::inRange<std::uint32_t>(0, node_count);
            if (s != d && node_layer[s] >= node_layer[d]) {
                back_src = s;
                back_dst = d;
                break;
            }
        }

        edges.push_back(Dependency_Edge{back_src, back_dst});

        return Generated_DAG{std::move(task_names), std::move(edges)};
    });
}

// ─── Stream Descriptor Generators ────────────────────────────────────────────

/// Generator for a valid Stream_Descriptor with random valid fields.
///
/// Produces stream names of 1–32 alphanumeric characters, random valid
/// temporal profiles and out-of-bounds policies, non-empty dataset paths,
/// and positive snapshot intervals.
inline auto valid_stream_descriptor() {
    return rc::gen::exec([] {
        // Generate a stream name: 1–32 lowercase alphabetic characters
        auto name_len = *rc::gen::inRange(1, 33);
        std::string name;
        name.reserve(name_len);
        for (int i = 0; i < name_len; ++i) {
            name.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }

        // Random temporal profile
        auto profile = *rc::gen::inRange<int>(0, 2) == 0 ? Temporal_Profile::linear : Temporal_Profile::step;

        // Random out-of-bounds policy
        auto oob = *rc::gen::inRange<int>(0, 2) == 0 ? OutOfBounds_Policy::clamp : OutOfBounds_Policy::cycle;

        // Generate a non-empty dataset path
        auto path_len = *rc::gen::inRange(3, 32);
        std::string path_str = "/data/";
        for (int i = 0; i < path_len; ++i) {
            path_str.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        path_str += ".zarr";

        // Generate a positive snapshot interval (1 hour to 30 days in nanoseconds)
        auto interval_hours = *rc::gen::inRange<std::int64_t>(1, 721);  // 1 hour to 30 days
        auto snapshot_interval = tick::Duration{interval_hours * 3'600'000'000'000LL};

        return Stream_Descriptor{std::move(name), profile, oob, std::filesystem::path{std::move(path_str)}, snapshot_interval};
    });
}

// ─── Pipeline_Config Generators ──────────────────────────────────────────────

/// Generator for a complete valid Pipeline_Config.
///
/// Produces a configuration with:
/// - 1 to max_streams valid Stream_Descriptors (unique names guaranteed)
/// - A valid acyclic DAG topology matching the stream count
/// - Valid max_concurrency and deadlock_timeout_s values
///
/// @param min_streams  Minimum number of streams/tasks (≥ 2)
/// @param max_streams  Maximum number of streams/tasks (≤ 32)
inline auto valid_pipeline_config(std::uint32_t min_streams, std::uint32_t max_streams) {
    return rc::gen::exec([min_streams, max_streams] {
        // Generate stream count
        auto stream_count = *rc::gen::inRange<std::uint32_t>(min_streams, max_streams + 1);

        // Generate unique stream descriptors
        std::vector<Stream_Descriptor> streams;
        streams.reserve(stream_count);
        std::set<std::string> used_names;

        for (std::uint32_t i = 0; i < stream_count; ++i) {
            auto sd = *valid_stream_descriptor();
            // Ensure unique names by appending index
            sd.name = "stream_" + std::to_string(i) + "_" + sd.name;
            // Truncate to 128 chars max
            if (sd.name.size() > 128) {
                sd.name.resize(128);
            }
            streams.push_back(std::move(sd));
        }

        // Generate a matching acyclic DAG
        auto dag = *acyclic_dag(stream_count, stream_count, std::min<std::uint32_t>(stream_count * 2, 512));

        // Generate valid concurrency and timeout values
        auto concurrency = *rc::gen::inRange<std::uint32_t>(1, 65);
        auto deadlock_timeout = *rc::gen::inRange<std::uint32_t>(1, 3601);
        auto shutdown_timeout = *rc::gen::inRange<std::uint32_t>(1, 61);

        return Pipeline_Config{std::move(streams), std::move(dag.task_names), std::move(dag.edges), concurrency, deadlock_timeout, shutdown_timeout};
    });
}

// ─── Scalar Value Generators ─────────────────────────────────────────────────

/// Generator for rank pool sizes in [1, 64].
/// Used by rank-conservation property tests (Req 13.3).
inline auto rank_pool_size() {
    return rc::gen::inRange<std::uint32_t>(1, 65);
}

/// Generator for max concurrency values in [1, 64].
/// Used by concurrency-limit property tests (Req 13.5).
inline auto max_concurrency() {
    return rc::gen::inRange<std::uint32_t>(1, 65);
}

// ─── Completion Ordering Utility ─────────────────────────────────────────────

/// Atomic sequence counter for recording task completion ordering.
///
/// Each TaskNode's completion callback increments this counter and records
/// its value, providing the observable evidence for topological-order invariant
/// verification (Req 13.9).
///
/// Thread-safe: uses std::memory_order_seq_cst for all operations.
class Completion_Sequence_Counter {
   public:
    Completion_Sequence_Counter() noexcept = default;

    /// Record a completion event and return its sequence number.
    /// @return Monotonically increasing sequence number (starting at 1)
    [[nodiscard]] std::uint64_t record_completion() noexcept {
        return counter_.fetch_add(1, std::memory_order_seq_cst) + 1;
    }

    /// Get the current sequence counter value (number of completions recorded).
    [[nodiscard]] std::uint64_t current() const noexcept {
        return counter_.load(std::memory_order_seq_cst);
    }

    /// Reset the counter to zero. NOT thread-safe — call only between tests.
    void reset() noexcept {
        counter_.store(0, std::memory_order_seq_cst);
    }

   private:
    std::atomic<std::uint64_t> counter_{0};
};

}  // namespace dagr::gen
