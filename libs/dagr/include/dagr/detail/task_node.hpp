// DAGR — detail/task_node.hpp
// TaskNode vertex and Task_Status enum for the scheduling DAG.
// Requirements: 3.1, 3.2
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace dagr::detail {

/// Task execution status — models the lifecycle of a DAG vertex.
enum class Task_Status : std::uint8_t {
    pending   = 0,  ///< Waiting for upstream dependencies to complete
    ready     = 1,  ///< All dependencies resolved, awaiting dispatch
    running   = 2,  ///< Currently executing on allocated ranks
    completed = 3,  ///< Finished successfully
    failed    = 4,  ///< Execution error
    cancelled = 5   ///< Cancelled due to upstream failure
};

/// A vertex in the scheduling DAG.
/// Stored in a contiguous std::vector indexed by `id`.
struct TaskNode {
    std::uint32_t              id;               ///< Zero-based sequential index
    std::string                name;             ///< Human-readable identifier
    std::atomic<std::uint32_t> pending_deps{0};  ///< In-degree countdown (decremented on upstream completion)
    Task_Status                status{Task_Status::pending}; ///< Current lifecycle state
    std::uint32_t              required_ranks{1}; ///< MPI ranks needed for execution

    /// Downstream adjacency list (indices into the node vector).
    std::vector<std::uint32_t> dependents;
};

} // namespace dagr::detail
