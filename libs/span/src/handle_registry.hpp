// SPDX-License-Identifier: Apache-2.0
// SPAN — Shared Pointer & Array Network
// Copyright (c) HELM Project Contributors

#ifndef SPAN_HANDLE_REGISTRY_HPP
#define SPAN_HANDLE_REGISTRY_HPP

/// @file src/handle_registry.hpp
/// @brief Thread-safe opaque handle registry for SPAN C-interop layer.
///
/// Maps positive integer handles (exposed to Fortran as opaque tokens) to
/// type-erased C++ objects (shared_ptr<void>). Uses a pre-allocated vector
/// with free-list recycling for O(1) lookup, insertion, and deletion.
///
/// Handle 0 is reserved as INVALID and never assigned.
/// Handles are recycled via a free-list queue when unregistered.

#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "span/span_constants.h"

namespace span::detail {

/// Thread-safe singleton registry mapping integer handles to type-erased objects.
///
/// Design decisions (different from AXIS pattern):
///   - Vector-backed with pre-allocated 65,536 slots for O(1) direct indexing.
///   - Free-list recycling: unregistered handle indices go into a queue for reuse.
///   - Capacity-limited: returns HELM_SPAN_ERR_REGISTRY_FULL when full.
///
/// Usage from C interop functions:
///   auto& reg = HandleRegistry::instance();
///   int h = reg.register_view(std::make_shared<FieldView<double,2>>(...));
///   auto obj = reg.lookup(h);
///   reg.unregister(h);
class HandleRegistry {
public:
    /// Maximum number of concurrent active entries.
    static constexpr std::size_t MAX_CAPACITY = 65536;

    /// Access the process-wide singleton instance.
    static HandleRegistry& instance() {
        static HandleRegistry singleton;
        return singleton;
    }

    // Non-copyable, non-movable (singleton)
    HandleRegistry(const HandleRegistry&) = delete;
    HandleRegistry& operator=(const HandleRegistry&) = delete;
    HandleRegistry(HandleRegistry&&) = delete;
    HandleRegistry& operator=(HandleRegistry&&) = delete;

    /// Register a view and return a unique positive integer handle.
    ///
    /// Thread-safe.
    /// @param view  Shared pointer to the object (type-erased).
    /// @return      A positive integer handle (>0) on success,
    ///              or HELM_SPAN_ERR_REGISTRY_FULL (-5) when capacity reached.
    int register_view(std::shared_ptr<void> view) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (active_count_ >= MAX_CAPACITY) {
            return HELM_SPAN_ERR_REGISTRY_FULL;
        }

        int handle;

        if (!free_list_.empty()) {
            // Recycle a previously-freed handle
            handle = free_list_.front();
            free_list_.pop();
            // Overwrite the slot with the new view
            slots_[static_cast<std::size_t>(handle) - 1] = std::move(view);
            occupied_[static_cast<std::size_t>(handle) - 1] = true;
        } else {
            // Allocate the next monotonic handle
            handle = next_handle_++;
            std::size_t idx = static_cast<std::size_t>(handle) - 1;
            slots_[idx] = std::move(view);
            occupied_[idx] = true;
        }

        ++active_count_;
        return handle;
    }

    /// Look up the object associated with a handle.
    ///
    /// Thread-safe.
    /// @param handle  The integer handle returned by register_view.
    /// @return        Shared pointer to the object, or nullptr if invalid.
    [[nodiscard]] std::shared_ptr<void> lookup(int handle) const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!is_valid_unlocked(handle)) {
            return nullptr;
        }

        return slots_[static_cast<std::size_t>(handle) - 1];
    }

    /// Unregister the object associated with a handle.
    ///
    /// The handle is recycled for future reuse. The old shared_ptr is
    /// released (potentially destroying the object if last reference).
    ///
    /// Thread-safe.
    /// @param handle  The integer handle to unregister.
    /// @return        true if successfully unregistered, false if handle was invalid.
    bool unregister(int handle) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!is_valid_unlocked(handle)) {
            return false;
        }

        std::size_t idx = static_cast<std::size_t>(handle) - 1;
        slots_[idx].reset();
        occupied_[idx] = false;
        free_list_.push(handle);
        --active_count_;
        return true;
    }

    /// Check whether a handle currently maps to a live entry.
    ///
    /// Thread-safe.
    /// @param handle  The integer handle to check.
    /// @return        true if the handle is valid and currently occupied.
    [[nodiscard]] bool valid(int handle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_valid_unlocked(handle);
    }

    /// Returns the current number of active entries.
    ///
    /// Thread-safe.
    [[nodiscard]] std::size_t active_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_count_;
    }

private:
    HandleRegistry()
        : slots_(MAX_CAPACITY)
        , occupied_(MAX_CAPACITY, false)
        , next_handle_(1)
        , active_count_(0) {}

    ~HandleRegistry() = default;

    /// Check validity without holding the lock (caller must hold mutex_).
    [[nodiscard]] bool is_valid_unlocked(int handle) const {
        if (handle <= 0) {
            return false;
        }
        std::size_t idx = static_cast<std::size_t>(handle) - 1;
        if (idx >= MAX_CAPACITY) {
            return false;
        }
        return occupied_[idx];
    }

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<void>> slots_;  ///< Pre-allocated, index = handle-1
    std::vector<bool> occupied_;                 ///< Tracks which slots are active
    std::queue<int> free_list_;                  ///< Recycled handle indices
    int next_handle_;                            ///< Monotonic until free_list non-empty
    std::size_t active_count_;                   ///< Current number of active entries
};

} // namespace span::detail

#endif // SPAN_HANDLE_REGISTRY_HPP
