// DAGR — detail/rank_pool.hpp
// Lock-free MPI rank pool tracking using an atomic bitset.
// Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.9
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace dagr::detail {

/// Lock-free rank tracking using an atomic bitset.
///
/// Each managed MPI rank occupies one slot in a contiguous bitset where
/// `true` indicates the rank is available for allocation and `false`
/// indicates it is currently allocated to an in-flight task.
///
/// Invariant: available_ranks() + allocated_ranks() == total_ranks()
/// at every observable point (Req 5.9).
///
/// Thread safety: all public methods are safe to call concurrently
/// without external synchronization (Req 5.4).
class Rank_Pool {
   public:
    /// Construct a Rank_Pool managing the given set of MPI rank identifiers.
    /// All ranks are initially available.
    /// @param ranks  Set of MPI rank identifiers to manage
    explicit Rank_Pool(std::set<int> ranks);

    /// Default-construct an empty pool (zero ranks managed).
    Rank_Pool() = default;

    /// Attempt to allocate `count` ranks from the pool.
    /// @param count  Number of ranks to allocate (must be > 0)
    /// @return Set of allocated rank IDs, or empty set if insufficient ranks available
    /// @throws std::invalid_argument if count == 0 (Req 5.6)
    [[nodiscard]] std::set<int> try_allocate(std::uint32_t count);

    /// Return previously allocated ranks to the pool.
    /// Flips bits back to available state.
    /// @param ranks  Set of rank IDs to release
    void release(const std::set<int> &ranks);

    /// Add newly yielded ranks to the pool.
    /// Extends the bitset and rank_ids mapping. Increments total and available.
    /// @param ranks  Set of new MPI rank identifiers to add
    void add_ranks(const std::set<int> &ranks);

    /// Remove only available (non-allocated) ranks from the pool.
    /// @param ranks  Set of rank IDs to attempt removal
    /// @return Set of ranks actually removed (subset of input that were available)
    std::set<int> remove_available(const std::set<int> &ranks);

    /// @return Total number of ranks managed by this pool.
    [[nodiscard]] std::uint32_t total_ranks() const noexcept;

    /// @return Number of ranks currently available for allocation.
    [[nodiscard]] std::uint32_t available_ranks() const noexcept;

    /// @return Number of ranks currently allocated to in-flight tasks.
    [[nodiscard]] std::uint32_t allocated_ranks() const noexcept;

   private:
    /// Per-rank availability flag: true = available, false = allocated.
    /// Stored as heap-allocated atomics to allow vector resizing.
    std::vector<std::unique_ptr<std::atomic<bool>>> bitset_;

    /// Mapping: index → MPI rank ID.
    std::vector<int> rank_ids_;

    /// Total number of ranks currently managed.
    std::atomic<std::uint32_t> total_{0};

    /// Number of ranks currently available.
    std::atomic<std::uint32_t> available_{0};

    /// Mutex for structural modifications (add_ranks, remove_available)
    /// that resize the bitset/rank_ids vectors. Individual allocate/release
    /// operations use atomic compare-exchange on the bitset entries.
    mutable std::mutex structural_mutex_;
};

}  // namespace dagr::detail
