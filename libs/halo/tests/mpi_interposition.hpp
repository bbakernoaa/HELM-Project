// ─── HALO MPI Interposition Spy Layer ────────────────────────────────────────
// Provides a singleton MPI_Spy that records intercepted MPI calls and supports
// error injection for deterministic testing of RAII destructor behavior.
//
// This layer uses weak-symbol overrides to intercept target MPI C functions.
// When linked into a test executable, these overrides replace the real MPI
// implementations, allowing tests to verify call sequences without relying on
// MPI runtime side effects.
//
// Feature: helm-halo-microlibrary
// Requirements: 11.7
// ─────────────────────────────────────────────────────────────────────────────

#ifndef HALO_TESTS_MPI_INTERPOSITION_HPP
#define HALO_TESTS_MPI_INTERPOSITION_HPP

#include <mpi.h>

#include <mutex>
#include <optional>
#include <vector>

namespace halo::testing {

// ─── MPI Call Record ─────────────────────────────────────────────────────────

/// Records a single intercepted MPI call with its type, handle, and arguments.
struct MPI_Call_Record {
    /// Enumeration of all intercepted MPI function types.
    enum class Type {
        Comm_free,
        Request_free,
        Cancel,
        Wait,
        Test,
        Win_free,
        Win_fence,
        Irecv,
        Isend,
        Waitall,
        Allgather,
        Allgatherv,
        Allreduce
    };

    Type type;
    void *handle;  ///< The primary handle argument (cast from MPI handle pointer)
    int arg;       ///< Additional argument (e.g., assertion for fence, count for Waitall)

    /// Convenience constructor.
    MPI_Call_Record(Type t, void *h, int a = 0) noexcept : type{t}, handle{h}, arg{a} {}
};

// ─── MPI Spy Singleton ──────────────────────────────────────────────────────

/// Thread-safe singleton that records all intercepted MPI calls and supports
/// error injection for testing error-handling paths.
class MPI_Spy {
   public:
    /// Access the singleton instance.
    static MPI_Spy &instance() {
        static MPI_Spy spy;
        return spy;
    }

    /// Clear all recorded calls and reset error injection state.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        calls_.clear();
        next_error_.reset();
    }

    /// Get the list of all recorded MPI calls (thread-safe copy).
    [[nodiscard]] std::vector<MPI_Call_Record> const &calls() const {
        // Note: caller must ensure no concurrent modifications during read,
        // or use calls_copy() for a safe snapshot.
        return calls_;
    }

    /// Get a thread-safe copy of all recorded calls.
    [[nodiscard]] std::vector<MPI_Call_Record> calls_copy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
    }

    /// Set the error code that the NEXT intercepted MPI call will return.
    /// After one call consumes it, subsequent calls return MPI_SUCCESS.
    void set_next_error(int error_code) {
        std::lock_guard<std::mutex> lock(mutex_);
        next_error_ = error_code;
    }

    /// Set the thread level that MPI_Query_thread will report.
    /// Defaults to MPI_THREAD_MULTIPLE.
    void set_thread_level(int level) {
        std::lock_guard<std::mutex> lock(mutex_);
        thread_level_ = level;
    }

    /// Get the configured thread level for MPI_Query_thread.
    [[nodiscard]] int thread_level() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return thread_level_;
    }

    /// Get the number of times MPI_Query_thread was called.
    [[nodiscard]] std::size_t query_thread_call_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return query_thread_calls_;
    }

    /// Record a call to MPI_Query_thread.
    void record_query_thread() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++query_thread_calls_;
    }

    /// Record a call. Returns the error code to use (injected or MPI_SUCCESS).
    int record(MPI_Call_Record::Type type, void *handle, int arg = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        calls_.emplace_back(type, handle, arg);

        if (next_error_.has_value()) {
            int err = next_error_.value();
            next_error_.reset();
            return err;
        }
        return MPI_SUCCESS;
    }

    /// Check if any calls have been recorded.
    [[nodiscard]] bool has_calls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !calls_.empty();
    }

    /// Get the number of recorded calls.
    [[nodiscard]] std::size_t call_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_.size();
    }

    /// Count calls of a specific type.
    [[nodiscard]] std::size_t count_of(MPI_Call_Record::Type type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (auto const &rec : calls_) {
            if (rec.type == type) ++count;
        }
        return count;
    }

    // Non-copyable, non-movable singleton
    MPI_Spy(MPI_Spy const &) = delete;
    MPI_Spy &operator=(MPI_Spy const &) = delete;
    MPI_Spy(MPI_Spy &&) = delete;
    MPI_Spy &operator=(MPI_Spy &&) = delete;

   private:
    MPI_Spy() = default;

    mutable std::mutex mutex_;
    std::vector<MPI_Call_Record> calls_;
    std::optional<int> next_error_;
    int thread_level_{MPI_THREAD_MULTIPLE};
    std::size_t query_thread_calls_{0};
};

}  // namespace halo::testing

#endif  // HALO_TESTS_MPI_INTERPOSITION_HPP
