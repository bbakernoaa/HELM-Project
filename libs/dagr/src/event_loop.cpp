// DAGR — event_loop.cpp
// Asynchronous poll-dispatch-complete event loop implementation.
// Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 10.3, 10.5

#include "dagr/detail/event_loop.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>

#include "logs/logs.hpp"
#include "shared_logger.hpp"

namespace dagr::detail {

namespace {

/// Event_Loop diagnostics route through DAGR's single shared LOGS logger
/// (configured once via dagr::configure_logging).
logs::Logger &logger() {
    return shared_logger();
}

/// Check if a Task_Status is terminal (no further transitions possible).
bool is_terminal(Task_Status s) noexcept {
    return s == Task_Status::completed || s == Task_Status::failed || s == Task_Status::cancelled;
}

}  // anonymous namespace

// ─── Construction ────────────────────────────────────────────────────────────

Event_Loop::Event_Loop(Config cfg, std::vector<TaskNode> &nodes, Rank_Pool &rank_pool)
    : config_(cfg), nodes_(nodes), rank_pool_(rank_pool), last_progress_time_(std::chrono::steady_clock::now()) {
    // Req 6.8: Validate max_concurrency ∈ [1, 1024]
    if (cfg.max_concurrency < 1 || cfg.max_concurrency > 1024) {
        throw std::invalid_argument("Event_Loop: max_concurrency must be in range [1, 1024], got " + std::to_string(cfg.max_concurrency));
    }

    // Req 6.9: Validate deadlock_timeout_s ∈ [1, 3600]
    if (cfg.deadlock_timeout_s < 1 || cfg.deadlock_timeout_s > 3600) {
        throw std::invalid_argument("Event_Loop: deadlock_timeout_s must be in range [1, 3600], got " + std::to_string(cfg.deadlock_timeout_s));
    }

    // Pre-allocate rank tracking per node
    allocated_ranks_.resize(nodes.size());
}

// ─── Callback Setters ────────────────────────────────────────────────────────

void Event_Loop::set_dispatch_callback(Dispatch_Callback cb) {
    dispatch_cb_ = std::move(cb);
}

void Event_Loop::set_completion_callback(Completion_Callback cb) {
    completion_cb_ = std::move(cb);
}

// ─── Main Cycle (Req 6.1) ────────────────────────────────────────────────────

bool Event_Loop::cycle() {
    // Phase 1: Poll — scan for newly ready nodes
    poll_ready_nodes();

    // Phase 2: Dispatch — send ready nodes to available ranks
    dispatch_ready_nodes();

    // Phase 3: Complete — process completion callbacks
    process_completions();

    // Check for deadlock (Req 6.7)
    check_deadlock();

    return all_complete();
}

// ─── Accessors ───────────────────────────────────────────────────────────────

bool Event_Loop::all_complete() const noexcept {
    for (const auto &node : nodes_) {
        if (!is_terminal(node.status)) {
            return false;
        }
    }
    return true;
}

std::uint32_t Event_Loop::in_flight_count() const noexcept {
    return in_flight_;
}

void Event_Loop::force_cancel_in_flight() {
    logs::Scoped_Context ctx("force_cancel");

    std::uint32_t cancelled = 0;
    for (auto &node : nodes_) {
        if (node.status == Task_Status::running) {
            // Release any ranks this task was holding.
            if (node.id < allocated_ranks_.size() && !allocated_ranks_[node.id].empty()) {
                rank_pool_.release(allocated_ranks_[node.id]);
                allocated_ranks_[node.id].clear();
            }
            node.status = Task_Status::cancelled;
            ++cancelled;
        }
    }

    in_flight_ = 0;
    record_progress();

    if (cancelled > 0) {
        logger().log(logs::Severity_Level::WARNING,
                     "Event_Loop: force-cancelled " + std::to_string(cancelled) + " in-flight task(s) during shutdown");
    }
}

// ─── Phase 1: Poll (Req 3.4, 3.6) ───────────────────────────────────────────

void Event_Loop::poll_ready_nodes() {
    for (auto &node : nodes_) {
        if (node.status == Task_Status::pending) {
            // Req 3.4, 3.6: If pending_deps reaches zero, mark as ready
            if (node.pending_deps.load(std::memory_order_acquire) == 0) {
                node.status = Task_Status::ready;
                ready_queue_.push(node.id);
            }
        }
    }
}

// ─── Phase 2: Dispatch (Req 3.5, 6.5) ───────────────────────────────────────

void Event_Loop::dispatch_ready_nodes() {
    logs::Scoped_Context ctx("dispatch");

    while (!ready_queue_.empty()) {
        // Req 6.5: Enforce max_concurrency cap
        if (in_flight_ >= config_.max_concurrency) {
            break;
        }

        const std::uint32_t node_id = ready_queue_.front();
        auto &node = nodes_[node_id];

        // Skip nodes that may have been cancelled between ready and dispatch
        if (node.status != Task_Status::ready) {
            ready_queue_.pop();
            continue;
        }

        // Req 3.5: Attempt rank allocation
        const std::uint32_t needed = node.required_ranks;
        std::set<int> ranks = rank_pool_.try_allocate(needed);

        if (ranks.empty()) {
            // Insufficient ranks available — stop dispatching for this cycle.
            // The node stays at the front of the ready queue for next cycle.
            break;
        }

        // Remove from ready queue
        ready_queue_.pop();

        // Store allocated ranks for release on completion
        allocated_ranks_[node_id] = ranks;

        // Mark as running
        node.status = Task_Status::running;
        ++in_flight_;

        // Record progress — we dispatched something
        record_progress();

        // Invoke the dispatch callback (Req 6.2)
        if (dispatch_cb_) {
            try {
                bool success = dispatch_cb_(node_id, ranks);
                if (!success) {
                    // Immediate failure during dispatch
                    propagate_failure(node_id);
                }
            } catch (...) {
                // Dispatch threw — treat as failure
                propagate_failure(node_id);
            }
        }
    }
}

// ─── Phase 3: Complete (Req 6.3, 3.3) ───────────────────────────────────────

void Event_Loop::process_completions() {
    logs::Scoped_Context ctx("completion");

    if (!completion_cb_) {
        return;
    }

    // Req 6.3: Process completion callbacks (non-blocking)
    auto completions = completion_cb_();

    for (const auto &[node_id, success] : completions) {
        if (node_id >= nodes_.size()) {
            continue;
        }

        auto &node = nodes_[node_id];

        // Only process running nodes
        if (node.status != Task_Status::running) {
            continue;
        }

        if (success) {
            // Mark as completed
            node.status = Task_Status::completed;

            // Release allocated ranks back to pool
            if (!allocated_ranks_[node_id].empty()) {
                rank_pool_.release(allocated_ranks_[node_id]);
                allocated_ranks_[node_id].clear();
            }

            --in_flight_;
            record_progress();

            // Req 3.3: Decrement downstream pending_deps
            for (std::uint32_t dep_id : node.dependents) {
                if (dep_id < nodes_.size()) {
                    nodes_[dep_id].pending_deps.fetch_sub(1, std::memory_order_release);
                }
            }
        } else {
            // Task failed — propagate failure (Req 6.6)
            propagate_failure(node_id);
        }
    }
}

// ─── Failure Propagation (Req 6.6, 10.3) ────────────────────────────────────

void Event_Loop::propagate_failure(std::uint32_t failed_node_id) {
    logs::Scoped_Context ctx("failure");

    auto &node = nodes_[failed_node_id];

    // Mark the failed node
    node.status = Task_Status::failed;

    // Release allocated ranks back to pool
    if (!allocated_ranks_[failed_node_id].empty()) {
        rank_pool_.release(allocated_ranks_[failed_node_id]);
        allocated_ranks_[failed_node_id].clear();
    }

    // Decrement in-flight count if this node was running
    if (in_flight_ > 0) {
        --in_flight_;
    }

    // BFS to find all transitively dependent nodes and cancel them
    std::deque<std::uint32_t> bfs_queue;
    for (std::uint32_t dep_id : node.dependents) {
        bfs_queue.push_back(dep_id);
    }

    std::uint32_t cancelled_count = 0;

    while (!bfs_queue.empty()) {
        std::uint32_t current_id = bfs_queue.front();
        bfs_queue.pop_front();

        if (current_id >= nodes_.size()) {
            continue;
        }

        auto &dep_node = nodes_[current_id];

        // Only cancel nodes that haven't already reached a terminal state
        // and are not currently running (running nodes complete on their own)
        if (is_terminal(dep_node.status)) {
            continue;
        }

        if (dep_node.status == Task_Status::running) {
            // Running nodes can't be cancelled mid-flight; they'll complete
            // or fail on their own. Skip traversal through them.
            continue;
        }

        dep_node.status = Task_Status::cancelled;
        ++cancelled_count;

        // Continue BFS through this node's dependents
        for (std::uint32_t downstream_id : dep_node.dependents) {
            bfs_queue.push_back(downstream_id);
        }
    }

    // Req 10.3: Emit ERROR diagnostic
    logger().log(logs::Severity_Level::ERROR, "Event_Loop: task '" + node.name + "' (id=" + std::to_string(failed_node_id) + ") failed; " +
                                                  std::to_string(cancelled_count) + " dependent node(s) cancelled");

    record_progress();
}

// ─── Deadlock Detection (Req 6.7, 10.5) ─────────────────────────────────────

void Event_Loop::record_progress() {
    last_progress_time_ = std::chrono::steady_clock::now();
    deadlock_warned_ = false;
}

void Event_Loop::check_deadlock() {
    // Only check if there are in-flight tasks but nothing else is progressing
    if (in_flight_ == 0 && ready_queue_.empty()) {
        return;  // Nothing happening — not a deadlock, just idle/complete
    }

    if (all_complete()) {
        return;  // Everything done
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_time_);

    if (elapsed.count() >= static_cast<std::int64_t>(config_.deadlock_timeout_s)) {
        if (!deadlock_warned_) {
            // Count blocked nodes (pending but not ready)
            std::uint32_t blocked_count = 0;
            for (const auto &node : nodes_) {
                if (node.status == Task_Status::pending && node.pending_deps.load(std::memory_order_acquire) > 0) {
                    ++blocked_count;
                }
            }

            // Req 10.5: Emit WARNING diagnostic
            logger().log(logs::Severity_Level::WARNING, "Event_Loop: potential deadlock detected — no progress for " +
                                                            std::to_string(elapsed.count()) + "s; in-flight=" + std::to_string(in_flight_) +
                                                            ", blocked=" + std::to_string(blocked_count));

            deadlock_warned_ = true;
        }
    }
}

}  // namespace dagr::detail
