// DAGR — dagr.hpp
// Public header: GraphOrchestrator class declaration.
// Requirements: 1.1, 1.2, 1.5, 1.6, 1.7, 1.8
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>

// Forward declarations only — no lower-tier headers exposed publicly (Req 11.3)
namespace halo { class Communicator; }

namespace dagr {

struct Pipeline_Config;

/// @brief Tier 3 DAG-driven orchestrator for the HELM ecosystem.
///
/// @details GraphOrchestrator owns the complete DAG lifecycle: pipeline
/// construction from YAML, asynchronous event-loop scheduling, dynamic MPI
/// rank hijacking, and orderly teardown. It dispatches to lower-tier engines
/// (TICK, AMIO, BLEND, HALO) and performs no computation itself — it routes
/// pointers and manages execution dependencies.
///
/// GraphOrchestrator is move-only (non-copyable) and manages RAII resources
/// including halo::Communicator objects for MPI rank pools.
class GraphOrchestrator {
public:
    /// Construct from parsed pipeline config and world communicator.
    /// @param config  Pipeline configuration (taken by value, moved into internal state)
    /// @param world   World communicator (taken by rvalue reference, moved)
    /// @throws std::invalid_argument if config contains cyclic dependencies,
    ///         dangling task references, or invalid parameter ranges
    GraphOrchestrator(Pipeline_Config config, halo::Communicator&& world);

    ~GraphOrchestrator();

    // Move-only semantics (Req 1.6)
    GraphOrchestrator(GraphOrchestrator&&) noexcept;
    GraphOrchestrator& operator=(GraphOrchestrator&&) noexcept;
    GraphOrchestrator(const GraphOrchestrator&) = delete;
    GraphOrchestrator& operator=(const GraphOrchestrator&) = delete;

    /// Enter the event loop and process all tasks to completion.
    /// @throws std::runtime_error if any TaskNode fails (after draining non-cancelled nodes)
    void run();

    /// Execute one scheduling cycle (resolve → dispatch → complete).
    /// @return true if all nodes completed, false if work remains
    bool advance_step();

    /// Drain in-flight tasks, release hijacked ranks, destroy state.
    /// Idempotent: calling on an already-shut-down instance is a no-op.
    void shutdown();

    /// Yield model processors to the framework rank pool.
    /// @param rank_set  Set of MPI rank identifiers to yield
    void yield_ranks(std::set<int> rank_set);

    /// Reclaim previously yielded ranks (deferred if currently allocated).
    /// @param rank_set  Set of MPI rank identifiers to reclaim
    void reclaim_ranks(std::set<int> rank_set);

    // ─── Const Accessors ──────────────────────────────────────────────
    [[nodiscard]] std::uint32_t total_ranks() const noexcept;
    [[nodiscard]] std::uint32_t available_ranks() const noexcept;
    [[nodiscard]] std::uint32_t in_flight_count() const noexcept;
    [[nodiscard]] bool is_shutdown() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dagr
