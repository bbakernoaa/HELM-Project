#ifndef HALO_FORTRAN_HANDLE_REGISTRY_HPP
#define HALO_FORTRAN_HANDLE_REGISTRY_HPP

/// @file handle_registry.hpp
/// @brief Thread-safe opaque handle registry for Fortran C-interop layer.
///
/// The Handle_Registry maps integer tokens to C++ object pointers, enabling
/// Fortran code to reference C++ objects (Communicator, Halo_Plan, Halo_Handle)
/// via opaque integer tokens without exposing raw pointers across the language
/// boundary.
///
/// Token 0 is reserved as the invalid handle sentinel (HALO_HANDLE_INVALID).
/// Tokens are monotonically increasing positive integers that are never reused,
/// preventing dangling-handle bugs.
///
/// Thread safety is provided via std::mutex, supporting MPI_THREAD_MULTIPLE
/// environments where multiple threads may create/destroy objects concurrently.

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace halo::fortran {

/// @brief Invalid handle sentinel value.
///
/// Matches Fortran convention where 0/null signals an error or uninitialized
/// state. All valid tokens are positive integers.
inline constexpr int HALO_HANDLE_INVALID = 0;

/// @brief Thread-safe singleton registry mapping integer tokens to C++ object pointers.
///
/// The registry uses monotonically increasing positive integers as tokens.
/// Token 0 is reserved as invalid (HALO_HANDLE_INVALID). Tokens are never
/// reused after release, which prevents use-after-free bugs when Fortran code
/// retains a stale handle value.
///
/// Usage:
/// @code
///   auto& reg = halo::fortran::Handle_Registry::instance();
///   int token = reg.register_handle(static_cast<void*>(my_ptr));
///   void* ptr = reg.lookup(token);
///   void* released = reg.release(token);
///   delete static_cast<MyType*>(released);
/// @endcode
class Handle_Registry {
public:
    /// @brief Access the singleton instance.
    /// @return Reference to the global Handle_Registry.
    static Handle_Registry& instance() {
        static Handle_Registry reg;
        return reg;
    }

    /// @brief Register a pointer and return a unique positive integer token.
    ///
    /// The returned token is guaranteed to be a positive integer that has
    /// never been issued before by this registry instance.
    ///
    /// @param ptr Pointer to the C++ object to register. Must not be nullptr.
    /// @return A unique positive integer token identifying the registered object.
    int register_handle(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        int token = next_token_++;
        handles_[token] = ptr;
        return token;
    }

    /// @brief Retrieve the pointer associated with a token.
    ///
    /// @param token The integer token previously returned by register_handle.
    /// @return The registered pointer, or nullptr if the token is invalid
    ///         or has been released.
    void* lookup(int token) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = handles_.find(token);
        return (it != handles_.end()) ? it->second : nullptr;
    }

    /// @brief Remove a token from the registry and return its pointer.
    ///
    /// After this call, the token is no longer valid. The caller is responsible
    /// for freeing the returned pointer (e.g., via delete with the correct type).
    ///
    /// @param token The integer token to release.
    /// @return The previously registered pointer, or nullptr if the token
    ///         was not found (already released or never registered).
    void* release(int token) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = handles_.find(token);
        if (it == handles_.end()) return nullptr;
        void* ptr = it->second;
        handles_.erase(it);
        return ptr;
    }

    /// @brief Check if a token is currently valid (registered and not released).
    ///
    /// @param token The integer token to check.
    /// @return true if the token maps to a registered pointer, false otherwise.
    bool valid(int token) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return handles_.count(token) > 0;
    }

    // Non-copyable, non-movable singleton.
    Handle_Registry(const Handle_Registry&) = delete;
    Handle_Registry& operator=(const Handle_Registry&) = delete;
    Handle_Registry(Handle_Registry&&) = delete;
    Handle_Registry& operator=(Handle_Registry&&) = delete;

private:
    Handle_Registry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<int, void*> handles_;
    int next_token_{1};  // 0 reserved as HALO_HANDLE_INVALID
};

} // namespace halo::fortran

#endif // HALO_FORTRAN_HANDLE_REGISTRY_HPP
