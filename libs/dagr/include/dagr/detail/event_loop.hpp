// DAGR — detail/event_loop.hpp
// Asynchronous poll-dispatch-complete scheduling loop.
// Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 10.3, 10.5
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

#include "dagr/detail/completion_token.hpp"
#include "dagr/detail/rank_pool.hpp"
#include "dagr/detail/task_node.hpp"

namespace dagr::detail {

/// The asynchronous poll-dispatch-complete scheduling loop.
///
/// Operates on a vector of TaskNodes (passed by reference) and a Rank_Pool
/// (passed by reference) for allocation decisions. The Event_Loop is testable
/// standalone without MPI by using a callback/functor interface for task
/// execution and completion notification.
///
/// The cycle() method executes three sequential phases:
///   1. Poll: scan TaskNodes for pending_deps == 0, mark as ready
///   2. Dispatch: dispatch ready nodes to available ranks, enforce max_concurrency
///   3. Complete: process completion callbacks, decrement downstream pending_deps
class Event_Loop {
   public:
    /// Configuration validated at construction time.
    struct Config {
        std::uint32_t max_concurrency{64};     ///< [1, 1024] (Req 6.8)
        std::uint32_t deadlock_timeout_s{30};  ///< [1, 3600] seconds (Req 6.9)
    };

    /// Callback type for task execution. Invoked when a node is dispatched.
    /// The callback receives the node_id and the set of allocated ranks.
    /// It should return true on success, false on failure.
    /// If it throws, the node is treated as failed.
    using Dispatch_Callback = std::function<bool(std::uint32_t node_id, const std::set<int> &ranks)>;

    /// Callback type for checking task completion. Invoked during the
    /// complete phase. Returns a vector of node_ids that have completed
    /// since the last call. Each entry is a pair of (node_id, success).
    using Completion_Callback = std::function<std::vector<std::pair<std::uint32_t, bool>>()>;

    /// Construct an Event_Loop with validated configuration.
    /// @param cfg         Configuration (max_concurrency, deadlock_timeout_s)
    /// @param nodes       Reference to the TaskNode vector (owned externally)
    /// @param rank_pool   Reference to the Rank_Pool (owned externally)
    /// @throws std::invalid_argument if max_concurrency ∉ [1, 1024] (Req 6.8)
    /// @throws std::invalid_argument if deadlock_timeout_s ∉ [1, 3600] (Req 6.9)
    Event_Loop(Config cfg, std::vector<TaskNode> &nodes, Rank_Pool &rank_pool);

    /// Set the dispatch callback. Called when a ready node is dispatched.
    void set_dispatch_callback(Dispatch_Callback cb);

    /// Set the completion callback. Called during the complete phase.
    void set_completion_callback(Completion_Callback cb);

    /// Execute one full poll-dispatch-complete cycle (Req 6.1).
    /// @return true if all nodes reached terminal state, false otherwise
    bool cycle();

    /// Check if all nodes are in terminal state (completed, failed, or cancelled).
    [[nodiscard]] bool all_complete() const noexcept;

    /// Current count of dispatched-but-not-completed tasks.
    [[nodiscard]] std::uint32_t in_flight_count() const noexcept;

   private:
    /// Phase 1: Poll for newly ready nodes (Req 3.4, 3.6)
    void poll_ready_nodes();

    /// Phase 2: Dispatch ready nodes to available ranks (Req 3.5, 6.5)
    void dispatch_ready_nodes();

    /// Phase 3: Process completion callbacks (Req 6.3, 3.3)
    void process_completions();

    /// Mark a node as failed and cancel all transitively dependent nodes (Req 6.6)
    void propagate_failure(std::uint32_t failed_node_id);

    /// Record that progress was made (reset deadlock timer)
    void record_progress();

    /// Check for deadlock condition (Req 6.7, 10.5)
    void check_deadlock();

    Config config_;
    std::vector<TaskNode> &nodes_;
    Rank_Pool &rank_pool_;

    std::queue<std::uint32_t> ready_queue_;  ///< Nodes ready to dispatch (FIFO)
    std::uint32_t in_flight_{0};             ///< Currently dispatched tasks

    /// Tracks allocated ranks per in-flight node (node_id → ranks)
    std::vector<std::set<int>> allocated_ranks_;

    Dispatch_Callback dispatch_cb_;
    Completion_Callback completion_cb_;

    /// Deadlock detection timing
    std::chrono::steady_clock::time_point last_progress_time_;
    bool deadlock_warned_{false};  ///< Avoid repeated warnings
};

}  // namespace dagr::detail
