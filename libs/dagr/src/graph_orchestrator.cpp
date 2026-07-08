// DAGR — graph_orchestrator.cpp
// GraphOrchestrator facade implementation.
// Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 3.1, 3.2, 3.8,
//               5.1, 5.5, 5.7, 5.10, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7,
//               10.1, 10.2, 10.3, 10.4, 10.8, 10.9

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dagr/dagr.hpp"
#include "dagr/detail/event_loop.hpp"
#include "dagr/detail/rank_pool.hpp"
#include "dagr/detail/task_node.hpp"
#include "dagr/pipeline_config.hpp"
#include "halo/communicator.hpp"
#include "logs/logs.hpp"
#include "shared_logger.hpp"

namespace dagr {

namespace {

/// GraphOrchestrator diagnostics route through DAGR's single shared LOGS
/// logger (configured once via dagr::configure_logging), so rank stamping and
/// severity thresholds apply uniformly across all DAGR translation units.
logs::Logger &logger() {
    return detail::shared_logger();
}

/// Format a set of integers as a comma-separated bracket-enclosed string.
std::string format_rank_set(const std::set<int> &ranks) {
    std::string result = "{";
    bool first = true;
    for (int r : ranks) {
        if (!first) result += ", ";
        result += std::to_string(r);
        first = false;
    }
    result += "}";
    return result;
}

}  // anonymous namespace

// ─── pImpl Definition ────────────────────────────────────────────────────────

struct GraphOrchestrator::Impl {
    Pipeline_Config config;
    halo::Communicator world;
    std::vector<detail::TaskNode> nodes;
    detail::Rank_Pool rank_pool;
    detail::Event_Loop event_loop;
    bool shut_down{false};

    /// Ranks that have been requested for deferred reclamation (Req 5.10).
    /// These are removed from the pool upon task completion when they become
    /// available again.
    std::set<int> deferred_reclaim;

    // ─── RAII MPI Communicator Registry (Req 9.1, 9.2, 9.3, 9.4) ────────
    //
    // Stores sub-communicators created for in-flight tasks as RAII objects.
    // Each entry maps a node_id to its owned halo::Communicator.
    // The creation_order vector records the order in which communicators were
    // created so the destructor can destroy them in reverse order (Req 9.4).
    //
    // No raw MPI communicator handles are stored anywhere in dagr source (Req 9.3).
    // All MPI operations are mediated through HALO (Req 9.6).
    std::vector<std::optional<halo::Communicator>> task_communicators;
    std::vector<std::uint32_t> comm_creation_order;

    /// Construct Impl: validate the DAG, build nodes, initialize subsystems.
    /// Validation (dangling refs, acyclicity) is performed first via
    /// validated_node_count(), which throws on failure. The TaskNode vector
    /// is then constructed with the validated count (avoids moves of atomics).
    Impl(Pipeline_Config cfg, halo::Communicator &&comm)
        : config(std::move(cfg)),
          world(std::move(comm)),
          nodes(validated_node_count(config)),
          rank_pool(build_rank_pool(world)),
          event_loop(build_event_loop(config, nodes, rank_pool)) {
        // Populate node fields in-place (vector already default-constructed to size).
        populate_nodes(config, nodes);

        // Pre-allocate the communicator registry slots (one per node).
        // All start as std::nullopt (no communicator allocated yet).
        task_communicators.resize(nodes.size());
    }

    /// Destructor: destroy owned sub-communicators in reverse creation order (Req 9.4).
    /// The world communicator is destroyed last (it's a member, so it's destroyed
    /// after all other members in normal C++ destruction order).
    ~Impl() {
        // Destroy sub-communicators in reverse creation order (Req 9.4).
        // RAII destruction of halo::Communicator frees the handle internally
        // via HALO — no raw MPI calls in dagr (Req 9.6).
        for (auto it = comm_creation_order.rbegin(); it != comm_creation_order.rend(); ++it) {
            const std::uint32_t node_id = *it;
            if (node_id < task_communicators.size() && task_communicators[node_id].has_value()) {
                task_communicators[node_id].reset();
            }
        }
        comm_creation_order.clear();
    }

    // ─── Communicator Management Methods (Req 9.1, 9.5, 9.7) ────────────

    /// Create a sub-communicator for a dispatched task via halo::Communicator::split.
    /// Stores the result as an RAII object in the registry (Req 9.1).
    /// If split throws, emits FATAL via LOGS and allows exception to propagate (Req 9.5).
    ///
    /// @param node_id  The task node receiving the sub-communicator.
    /// @param color    MPI split color for this task's rank group.
    /// @param key      MPI split key for rank ordering within the sub-communicator.
    void create_task_communicator(std::uint32_t node_id, int color, int key) {
        try {
            halo::Communicator sub_comm = world.split(color, key);
            task_communicators[node_id] = std::move(sub_comm);
            comm_creation_order.push_back(node_id);
        } catch (...) {
            // Req 9.5: Emit FATAL diagnostic and let exception propagate.
            // HALO's RAII has already cleaned up any partially-constructed communicator.
            logger().log(logs::Severity_Level::FATAL, "GraphOrchestrator: halo::Communicator::split failed for node " + std::to_string(node_id) +
                                                          " '" + (node_id < nodes.size() ? nodes[node_id].name : "unknown") +
                                                          "'; exception propagating");
            throw;
        }
    }

    /// Destroy a task's pre-allocated communicator on cancellation (Req 9.7).
    /// RAII destruction releases the MPI handle without invoking any MPI collective.
    /// The halo::Communicator destructor frees the handle which is local (non-collective).
    ///
    /// @param node_id  The cancelled task node whose communicator should be released.
    void destroy_cancelled_communicator(std::uint32_t node_id) {
        if (node_id < task_communicators.size() && task_communicators[node_id].has_value()) {
            // RAII drop — halo::Communicator destructor frees the handle
            // without MPI collective (Req 9.7). Communicator free is a local operation.
            task_communicators[node_id].reset();

            // Remove from creation order tracking
            comm_creation_order.erase(std::remove(comm_creation_order.begin(), comm_creation_order.end(), node_id), comm_creation_order.end());
        }
    }

    /// Release a task's communicator on normal completion (Req 9.2).
    /// RAII destruction releases the MPI handle regardless of exit path.
    ///
    /// @param node_id  The completed task node whose communicator should be released.
    void release_task_communicator(std::uint32_t node_id) {
        if (node_id < task_communicators.size() && task_communicators[node_id].has_value()) {
            task_communicators[node_id].reset();

            // Remove from creation order tracking
            comm_creation_order.erase(std::remove(comm_creation_order.begin(), comm_creation_order.end(), node_id), comm_creation_order.end());
        }
    }

    /// Validate the config's DAG (dangling refs + acyclicity) and return node count.
    /// Throws std::invalid_argument on validation failure.
    static std::uint32_t validated_node_count(const Pipeline_Config &config) {
        validate_dag(config);
        return static_cast<std::uint32_t>(config.task_names.size());
    }

    // ── Static factory helpers ───────────────────────────────────────────────

    /// Validate the DAG topology: dangling references and acyclicity (Req 1.7, 3.8).
    /// Throws std::invalid_argument on failure.
    static void validate_dag(const Pipeline_Config &config) {
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

        // Build adjacency list for Kahn's algorithm.
        std::vector<std::vector<std::uint32_t>> downstream(num_tasks);
        std::vector<std::uint32_t> in_degree(num_tasks, 0);

        for (const auto &edge : config.edges) {
            in_degree[edge.consumer_id] += 1;
            downstream[edge.producer_id].push_back(edge.consumer_id);
        }

        // Req 1.7: Validate acyclicity via Kahn's algorithm.
        std::deque<std::uint32_t> queue;
        for (std::uint32_t i = 0; i < num_tasks; ++i) {
            if (in_degree[i] == 0) {
                queue.push_back(i);
            }
        }

        std::uint32_t sorted_count = 0;
        while (!queue.empty()) {
            std::uint32_t node = queue.front();
            queue.pop_front();
            ++sorted_count;

            for (std::uint32_t dep : downstream[node]) {
                in_degree[dep] -= 1;
                if (in_degree[dep] == 0) {
                    queue.push_back(dep);
                }
            }
        }

        if (sorted_count != num_tasks) {
            // Cycle detected — find at least two nodes with remaining in-degree > 0.
            std::string cycle_path;
            for (std::uint32_t i = 0; i < num_tasks; ++i) {
                if (in_degree[i] > 0) {
                    if (!cycle_path.empty()) {
                        cycle_path += " -> ";
                    }
                    cycle_path += config.task_names[i];
                    if (cycle_path.find(" -> ") != std::string::npos) {
                        break;
                    }
                }
            }

            logger().log(logs::Severity_Level::WARNING, "GraphOrchestrator: cycle detected in DAG: " + cycle_path);
            throw std::invalid_argument("GraphOrchestrator: cycle detected in task dependencies: " + cycle_path);
        }
    }

    /// Populate TaskNode fields in-place (Req 3.1, 3.2).
    /// The nodes vector must already be sized to config.task_names.size().
    static void populate_nodes(const Pipeline_Config &config, std::vector<detail::TaskNode> &nodes) {
        const auto num_tasks = static_cast<std::uint32_t>(config.task_names.size());

        // Compute in-degrees and downstream adjacency.
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
            nodes[i].status = detail::Task_Status::pending;
            nodes[i].required_ranks = 1;
            nodes[i].dependents = std::move(downstream[i]);
        }
    }

    /// Build a Rank_Pool from the world communicator size.
    static detail::Rank_Pool build_rank_pool(const halo::Communicator &world) {
        const int world_size = world.size();
        std::set<int> initial_ranks;
        for (int r = 0; r < world_size; ++r) {
            initial_ranks.insert(r);
        }
        return detail::Rank_Pool(std::move(initial_ranks));
    }

    /// Build an Event_Loop with config parameters.
    static detail::Event_Loop build_event_loop(const Pipeline_Config &config, std::vector<detail::TaskNode> &nodes, detail::Rank_Pool &rank_pool) {
        detail::Event_Loop::Config el_cfg;
        el_cfg.max_concurrency = config.max_concurrency;
        el_cfg.deadlock_timeout_s = config.deadlock_timeout_s;
        return detail::Event_Loop(el_cfg, nodes, rank_pool);
    }

    /// Process deferred reclamation: remove any deferred ranks that are now
    /// available in the pool (Req 5.10).
    void process_deferred_reclamation() {
        if (deferred_reclaim.empty()) return;

        auto removed = rank_pool.remove_available(deferred_reclaim);
        for (int r : removed) {
            deferred_reclaim.erase(r);
        }

        if (!removed.empty()) {
            logs::Scoped_Context ctx("reclaim");
            logger().log(logs::Severity_Level::DEBUG, "GraphOrchestrator: deferred reclamation completed for ranks " + format_rank_set(removed) +
                                                          "; pool state: available=" + std::to_string(rank_pool.available_ranks()) +
                                                          " total=" + std::to_string(rank_pool.total_ranks()));
        }
    }

    /// Check if any node has failed status.
    [[nodiscard]] bool has_failed_node() const noexcept {
        for (const auto &node : nodes) {
            if (node.status == detail::Task_Status::failed) {
                return true;
            }
        }
        return false;
    }

    /// Get the error message for the first failed node.
    [[nodiscard]] std::string failed_node_message() const {
        for (const auto &node : nodes) {
            if (node.status == detail::Task_Status::failed) {
                return "TaskNode '" + node.name + "' (id=" + std::to_string(node.id) + ") failed during execution";
            }
        }
        return "unknown failure";
    }
};

// ─── Constructor (Req 1.2) ───────────────────────────────────────────────────

GraphOrchestrator::GraphOrchestrator(Pipeline_Config config, halo::Communicator &&world)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(world))) {}

// ─── Destructor (Req 1.6) ────────────────────────────────────────────────────

GraphOrchestrator::~GraphOrchestrator() = default;

// ─── Move Operations (Req 1.6) ───────────────────────────────────────────────

GraphOrchestrator::GraphOrchestrator(GraphOrchestrator &&) noexcept = default;
GraphOrchestrator &GraphOrchestrator::operator=(GraphOrchestrator &&) noexcept = default;

// ─── run() (Req 1.3) ────────────────────────────────────────────────────────

void GraphOrchestrator::run() {
    if (!impl_ || impl_->shut_down) return;

    logs::Scoped_Context ctx("run");

    // Set up dispatch callback: emit INFO on dispatch (Req 10.1, 10.8)
    impl_->event_loop.set_dispatch_callback([this](std::uint32_t node_id, const std::set<int> &ranks) -> bool {
        {
            logs::Scoped_Context dispatch_ctx("dispatch");
            const auto now = std::chrono::steady_clock::now();
            const auto ts_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());

            logger().log(logs::Severity_Level::INFO, "GraphOrchestrator: dispatching node " + std::to_string(node_id) + " '" +
                                                         impl_->nodes[node_id].name + "'" + " on ranks " + format_rank_set(ranks) +
                                                         " at ts=" + std::to_string(ts_ns) + "ns");
        }
        return true;
    });

    // Set up completion callback: emit INFO on success, ERROR on failure (Req 10.2, 10.3, 10.8)
    impl_->event_loop.set_completion_callback([this]() -> std::vector<std::pair<std::uint32_t, bool>> {
        // The Event_Loop internally handles completions; this callback
        // is invoked to poll for externally completed tasks.
        // In the current architecture, the Event_Loop manages completion
        // internally — we return empty here. The diagnostics are emitted
        // by the event loop's own completion processing.
        return {};
    });

    // Enter the Event_Loop: process until all nodes reach terminal state.
    while (!impl_->event_loop.all_complete()) {
        impl_->event_loop.cycle();

        // Process any deferred rank reclamation after completions (Req 5.10)
        impl_->process_deferred_reclamation();
    }

    // After all nodes have drained, check for failures (Req 1.3)
    if (impl_->has_failed_node()) {
        std::string msg = impl_->failed_node_message();

        // Emit ERROR diagnostic for failure (Req 10.3)
        {
            logs::Scoped_Context failure_ctx("failure");
            logger().log(logs::Severity_Level::ERROR, "GraphOrchestrator: run() completed with failure: " + msg);
        }

        throw std::runtime_error(msg);
    }
}

// ─── advance_step() (Req 1.4) ───────────────────────────────────────────────

bool GraphOrchestrator::advance_step() {
    if (!impl_ || impl_->shut_down) return true;

    logs::Scoped_Context ctx("advance_step");

    // Execute one scheduling cycle: resolve → dispatch → complete
    bool all_done = impl_->event_loop.cycle();

    // Process deferred reclamation after each cycle (Req 5.10)
    impl_->process_deferred_reclamation();

    return all_done;
}

// ─── shutdown() (Req 1.5, 5.7) ──────────────────────────────────────────────

void GraphOrchestrator::shutdown() {
    // Idempotent: no-op on already-shut-down instance (Req 1.5)
    if (!impl_ || impl_->shut_down) return;

    logs::Scoped_Context ctx("shutdown");

    logger().log(logs::Severity_Level::INFO, "GraphOrchestrator: shutdown initiated");

    // Drain in-flight tasks subject to timeout (Req 5.7)
    const auto timeout = std::chrono::seconds(impl_->config.shutdown_timeout_s);
    const auto start = std::chrono::steady_clock::now();

    while (impl_->event_loop.in_flight_count() > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            // Drain timeout expired with tasks still in-flight (Req 5.7). This is
            // a *forced* shutdown: cancel the remaining tasks and proceed rather
            // than aborting the whole job. (Logging at FATAL here would trigger
            // LOGS' Synchronized_Abort / MPI-Abort, which contradicts the intent
            // to proceed and would take down every rank.)
            logger().log(logs::Severity_Level::WARNING,
                         "GraphOrchestrator: shutdown drain timeout expired (" + std::to_string(impl_->config.shutdown_timeout_s) + "s) with " +
                             std::to_string(impl_->event_loop.in_flight_count()) + " task(s) still in-flight; " +
                             std::to_string(impl_->rank_pool.allocated_ranks()) + " rank(s) unreturned; forcing cancellation");
            impl_->event_loop.force_cancel_in_flight();
            break;
        }

        impl_->event_loop.cycle();
        impl_->process_deferred_reclamation();

        // Brief yield to avoid busy-spin during drain
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Mark as shut down (Req 1.5)
    impl_->shut_down = true;

    logger().log(logs::Severity_Level::INFO,
                 "GraphOrchestrator: shutdown complete; pool state: available=" + std::to_string(impl_->rank_pool.available_ranks()) +
                     " total=" + std::to_string(impl_->rank_pool.total_ranks()));
}

// ─── yield_ranks() (Req 5.1) ────────────────────────────────────────────────

void GraphOrchestrator::yield_ranks(std::set<int> rank_set) {
    if (!impl_ || impl_->shut_down) return;

    logs::Scoped_Context ctx("yield_ranks");

    // Add ranks to the pool immediately (Req 5.1)
    impl_->rank_pool.add_ranks(rank_set);

    // Emit DEBUG diagnostic for rank operation (Req 10.4)
    logger().log(logs::Severity_Level::DEBUG, "GraphOrchestrator: yielded ranks " + format_rank_set(rank_set) +
                                                  "; pool state: available=" + std::to_string(impl_->rank_pool.available_ranks()) +
                                                  " total=" + std::to_string(impl_->rank_pool.total_ranks()));
}

// ─── reclaim_ranks() (Req 5.10) ─────────────────────────────────────────────

void GraphOrchestrator::reclaim_ranks(std::set<int> rank_set) {
    if (!impl_ || impl_->shut_down) return;

    logs::Scoped_Context ctx("reclaim_ranks");

    // Attempt to remove available ranks immediately.
    // Any ranks that are currently allocated are deferred (Req 5.10).
    auto removed = impl_->rank_pool.remove_available(rank_set);

    // Track ranks that could not be removed (currently allocated) for deferred reclamation
    for (int r : rank_set) {
        if (removed.find(r) == removed.end()) {
            impl_->deferred_reclaim.insert(r);
        }
    }

    // Emit DEBUG diagnostic for rank operation (Req 10.4)
    std::string deferred_msg;
    if (!impl_->deferred_reclaim.empty()) {
        // Build set of only the newly deferred ranks from this call
        std::set<int> newly_deferred;
        for (int r : rank_set) {
            if (removed.find(r) == removed.end()) {
                newly_deferred.insert(r);
            }
        }
        if (!newly_deferred.empty()) {
            deferred_msg = "; deferred ranks (allocated): " + format_rank_set(newly_deferred);
        }
    }

    logger().log(logs::Severity_Level::DEBUG, "GraphOrchestrator: reclaimed ranks " + format_rank_set(removed) + deferred_msg +
                                                  "; pool state: available=" + std::to_string(impl_->rank_pool.available_ranks()) +
                                                  " total=" + std::to_string(impl_->rank_pool.total_ranks()));
}

// ─── Const Accessors (Req 5.9) ──────────────────────────────────────────────

std::uint32_t GraphOrchestrator::total_ranks() const noexcept {
    if (!impl_) return 0;
    return impl_->rank_pool.total_ranks();
}

std::uint32_t GraphOrchestrator::available_ranks() const noexcept {
    if (!impl_) return 0;
    return impl_->rank_pool.available_ranks();
}

std::uint32_t GraphOrchestrator::in_flight_count() const noexcept {
    if (!impl_) return 0;
    return impl_->event_loop.in_flight_count();
}

bool GraphOrchestrator::is_shutdown() const noexcept {
    if (!impl_) return true;
    return impl_->shut_down;
}

}  // namespace dagr
