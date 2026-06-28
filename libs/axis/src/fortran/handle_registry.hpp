// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_FORTRAN_HANDLE_REGISTRY_HPP
#define AXIS_FORTRAN_HANDLE_REGISTRY_HPP

/// @file src/fortran/handle_registry.hpp
/// @brief Thread-safe opaque handle registry for Fortran C-interop layer.
///
/// Maps integer tokens (exposed to Fortran as opaque handles) to type-erased
/// C++ objects (shared_ptr<void>). This allows Fortran callers to hold,
/// pass, and release AXIS objects (meshes, matrices) without exposing C++
/// class internals across the language boundary.
///
/// Token 0 is reserved as INVALID and never assigned.
/// Tokens are monotonically increasing — released tokens are NOT recycled.

#include <memory>
#include <mutex>
#include <unordered_map>

namespace axis::fortran {

/// Thread-safe singleton registry mapping integer tokens to type-erased objects.
///
/// Usage from C interop functions:
///   auto& reg = Handle_Registry::instance();
///   int token = reg.register_handle(std::make_shared<Mesh>(...));
///   auto obj  = reg.lookup(token);
///   reg.release(token);
class Handle_Registry {
   public:
    /// Access the process-wide singleton instance.
    static Handle_Registry &instance() {
        static Handle_Registry singleton;
        return singleton;
    }

    // Non-copyable, non-movable (singleton)
    Handle_Registry(const Handle_Registry &) = delete;
    Handle_Registry &operator=(const Handle_Registry &) = delete;
    Handle_Registry(Handle_Registry &&) = delete;
    Handle_Registry &operator=(Handle_Registry &&) = delete;

    /// Register an object and return a unique integer token (> 0).
    ///
    /// Thread-safe. The token increases monotonically and is never recycled.
    /// @param obj  Shared pointer to the object (type-erased).
    /// @return     A positive integer token identifying the object.
    int register_handle(std::shared_ptr<void> obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        int token = next_token_++;
        handles_.emplace(token, std::move(obj));
        return token;
    }

    /// Look up the object associated with a token.
    ///
    /// Thread-safe.
    /// @param token  The integer token returned by register_handle.
    /// @return       Shared pointer to the object, or nullptr if invalid/released.
    std::shared_ptr<void> lookup(int token) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = handles_.find(token);
        if (it == handles_.end()) {
            return nullptr;
        }
        return it->second;
    }

    /// Check whether a token is currently valid (registered and not released).
    ///
    /// Thread-safe.
    /// @param token  The integer token to check.
    /// @return       true if the token maps to a live object.
    bool valid(int token) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return handles_.count(token) > 0;
    }

    /// Release the object associated with a token.
    ///
    /// After release, the token becomes invalid. If the returned shared_ptr was
    /// the last reference, the object is destroyed. No-op if token is invalid.
    ///
    /// Thread-safe.
    /// @param token  The integer token to release.
    void release(int token) {
        std::lock_guard<std::mutex> lock(mutex_);
        handles_.erase(token);
    }

   private:
    Handle_Registry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<int, std::shared_ptr<void>> handles_;
    int next_token_ = 1;  ///< 0 is reserved as INVALID
};

}  // namespace axis::fortran

#endif  // AXIS_FORTRAN_HANDLE_REGISTRY_HPP
