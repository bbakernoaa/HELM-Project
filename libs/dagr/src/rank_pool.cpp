// DAGR — rank_pool.cpp
// Lock-free MPI rank pool management.
// Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.9

#include "dagr/detail/rank_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace dagr::detail {

Rank_Pool::Rank_Pool(std::set<int> ranks)
    : rank_ids_(ranks.begin(), ranks.end()), total_(static_cast<std::uint32_t>(ranks.size())), available_(static_cast<std::uint32_t>(ranks.size())) {
    // Construct bitset with all slots marked available (true)
    bitset_.reserve(ranks.size());
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        bitset_.emplace_back(std::make_unique<std::atomic<bool>>(true));
    }
}

std::set<int> Rank_Pool::try_allocate(std::uint32_t count) {
    // Req 5.6: reject zero-rank allocation request
    if (count == 0) {
        throw std::invalid_argument("Rank_Pool::try_allocate: count must be > 0");
    }

    std::lock_guard<std::mutex> lock(structural_mutex_);

    // Quick check: are there enough available ranks?
    if (available_.load(std::memory_order_relaxed) < count) {
        return {};
    }

    std::set<int> allocated;
    const std::size_t pool_size = bitset_.size();

    for (std::size_t i = 0; i < pool_size && allocated.size() < count; ++i) {
        bool expected = true;
        if (bitset_[i]->compare_exchange_strong(expected, false, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            allocated.insert(rank_ids_[i]);
        }
    }

    if (allocated.size() < count) {
        // Insufficient ranks available: roll back all flipped bits
        for (std::size_t i = 0; i < pool_size; ++i) {
            if (allocated.count(rank_ids_[i]) > 0) {
                bitset_[i]->store(true, std::memory_order_release);
            }
        }
        return {};
    }

    // Decrement available count
    available_.fetch_sub(static_cast<std::uint32_t>(allocated.size()), std::memory_order_release);

    return allocated;
}

void Rank_Pool::release(const std::set<int> &ranks) {
    std::lock_guard<std::mutex> lock(structural_mutex_);

    const std::size_t pool_size = bitset_.size();
    std::uint32_t released_count = 0;

    for (std::size_t i = 0; i < pool_size; ++i) {
        if (ranks.count(rank_ids_[i]) > 0) {
            bool expected = false;
            if (bitset_[i]->compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                ++released_count;
            }
        }
    }

    if (released_count > 0) {
        available_.fetch_add(released_count, std::memory_order_release);
    }
}

void Rank_Pool::add_ranks(const std::set<int> &ranks) {
    std::lock_guard<std::mutex> lock(structural_mutex_);

    for (int rank_id : ranks) {
        // Check if rank already exists in pool
        bool already_exists = false;
        for (std::size_t i = 0; i < rank_ids_.size(); ++i) {
            if (rank_ids_[i] == rank_id) {
                already_exists = true;
                break;
            }
        }

        if (!already_exists) {
            rank_ids_.push_back(rank_id);
            bitset_.emplace_back(std::make_unique<std::atomic<bool>>(true));
            total_.fetch_add(1, std::memory_order_release);
            available_.fetch_add(1, std::memory_order_release);
        }
    }
}

std::set<int> Rank_Pool::remove_available(const std::set<int> &ranks) {
    std::lock_guard<std::mutex> lock(structural_mutex_);

    std::set<int> removed;

    // Identify indices of available ranks to remove
    std::vector<std::size_t> indices_to_remove;

    for (std::size_t i = 0; i < rank_ids_.size(); ++i) {
        if (ranks.count(rank_ids_[i]) > 0) {
            // Only remove if currently available
            if (bitset_[i]->load(std::memory_order_acquire)) {
                removed.insert(rank_ids_[i]);
                indices_to_remove.push_back(i);
            }
        }
    }

    // Remove entries from vectors in reverse index order to maintain validity.
    // Use swap-with-last-and-pop to avoid move issues with unique_ptr vector.
    std::sort(indices_to_remove.begin(), indices_to_remove.end(), [](std::size_t a, std::size_t b) { return a > b; });

    for (std::size_t idx : indices_to_remove) {
        const std::size_t last = rank_ids_.size() - 1;
        if (idx != last) {
            // Swap with last element
            std::swap(rank_ids_[idx], rank_ids_[last]);
            std::swap(bitset_[idx], bitset_[last]);
        }
        rank_ids_.pop_back();
        bitset_.pop_back();
    }

    // Update counters
    const auto removed_count = static_cast<std::uint32_t>(removed.size());
    if (removed_count > 0) {
        total_.fetch_sub(removed_count, std::memory_order_release);
        available_.fetch_sub(removed_count, std::memory_order_release);
    }

    return removed;
}

std::uint32_t Rank_Pool::total_ranks() const noexcept {
    return total_.load(std::memory_order_acquire);
}

std::uint32_t Rank_Pool::available_ranks() const noexcept {
    return available_.load(std::memory_order_acquire);
}

std::uint32_t Rank_Pool::allocated_ranks() const noexcept {
    return total_.load(std::memory_order_acquire) - available_.load(std::memory_order_acquire);
}

}  // namespace dagr::detail
