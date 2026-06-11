#ifndef LOGS_TESTS_MPI_INTERPOSITION_HPP
#define LOGS_TESTS_MPI_INTERPOSITION_HPP

/// @file tests/mpi_interposition.hpp
/// @brief MPI spy/mock layer for deterministic testing of LOGS.
///
/// Provides an MPI_Spy singleton that records all intercepted MPI calls,
/// supports error injection, and allows configuring return values (rank,
/// thread level, initialized state). Tests link against this layer instead
/// of (or in addition to) the real MPI library to gain deterministic control.
///
/// Requirements: 13.5

#include <atomic>
#include <cstdint>
#include <mutex>
#include <variant>
#include <vector>

#include <mpi.h>

namespace logs::testing {

// ─── Call Record Types ──────────────────────────────────────────────────────

/// Enumeration of intercepted MPI function types.
enum class MPI_Call_Type : std::uint8_t {
    COMM_RANK,
    QUERY_THREAD,
    INITIALIZED,
    ABORT,
    GATHER,
    ALLGATHER
};

/// Arguments specific to MPI_Comm_rank calls.
struct Comm_Rank_Args {
    MPI_Comm comm;
    int      rank_out; ///< The rank value that was returned.
};

/// Arguments specific to MPI_Query_thread calls.
struct Query_Thread_Args {
    int provided_out; ///< The thread level that was returned.
};

/// Arguments specific to MPI_Initialized calls.
struct Initialized_Args {
    int flag_out; ///< The initialized flag that was returned.
};

/// Arguments specific to MPI_Abort calls.
struct Abort_Args {
    MPI_Comm comm;
    int      errorcode;
};

/// Arguments specific to MPI_Gather calls.
struct Gather_Args {
    MPI_Comm comm;
    int      root;
    int      sendcount;
    int      recvcount;
};

/// Arguments specific to MPI_Allgather calls.
struct Allgather_Args {
    MPI_Comm comm;
    int      sendcount;
    int      recvcount;
};

/// Variant holding the arguments for any intercepted MPI call.
using MPI_Call_Args = std::variant<
    Comm_Rank_Args,
    Query_Thread_Args,
    Initialized_Args,
    Abort_Args,
    Gather_Args,
    Allgather_Args
>;

/// A single recorded MPI call with type, arguments, and ordering.
struct MPI_Call_Record {
    MPI_Call_Type type;       ///< Which MPI function was called.
    MPI_Call_Args args;       ///< Function-specific arguments.
    std::uint64_t sequence;   ///< Monotonically increasing call order.
};

// ─── MPI_Spy Singleton ──────────────────────────────────────────────────────

/// Thread-safe singleton that intercepts and records MPI calls for testing.
///
/// Usage:
///   auto& spy = MPI_Spy::instance();
///   spy.reset();
///   spy.set_rank(42);
///   // ... exercise code under test ...
///   auto calls = spy.calls();
///   // assert on calls
///
class MPI_Spy {
public:
    /// Access the global singleton instance (thread-safe).
    static MPI_Spy& instance() noexcept {
        static MPI_Spy s_instance;
        return s_instance;
    }

    // ─── Configuration ──────────────────────────────────────────────────

    /// Set the rank value that MPI_Comm_rank will return.
    void set_rank(int rank) noexcept {
        rank_.store(rank, std::memory_order_relaxed);
    }

    /// Set the thread level that MPI_Query_thread will return.
    void set_thread_level(int level) noexcept {
        thread_level_.store(level, std::memory_order_relaxed);
    }

    /// Set the MPI_Initialized result (true = initialized).
    void set_initialized(bool initialized) noexcept {
        initialized_.store(initialized, std::memory_order_relaxed);
    }

    /// Set the error code that MPI_Comm_rank will return (MPI_SUCCESS by default).
    void set_comm_rank_error(int error) noexcept {
        comm_rank_error_.store(error, std::memory_order_relaxed);
    }

    /// Set the error code that MPI_Query_thread will return.
    void set_query_thread_error(int error) noexcept {
        query_thread_error_.store(error, std::memory_order_relaxed);
    }

    /// Set the error code that MPI_Gather will return.
    void set_gather_error(int error) noexcept {
        gather_error_.store(error, std::memory_order_relaxed);
    }

    /// Set the error code that MPI_Allgather will return.
    void set_allgather_error(int error) noexcept {
        allgather_error_.store(error, std::memory_order_relaxed);
    }

    // ─── Query Configuration ────────────────────────────────────────────

    [[nodiscard]] int rank() const noexcept {
        return rank_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] int thread_level() const noexcept {
        return thread_level_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool initialized() const noexcept {
        return initialized_.load(std::memory_order_relaxed);
    }

    // ─── Call Recording ─────────────────────────────────────────────────

    /// Record a MPI_Comm_rank call and return the configured rank/error.
    int record_comm_rank(MPI_Comm comm, int* rank_out) noexcept {
        int error = comm_rank_error_.load(std::memory_order_relaxed);
        int r = rank_.load(std::memory_order_relaxed);

        if (error == MPI_SUCCESS && rank_out) {
            *rank_out = r;
        }

        MPI_Call_Record rec{
            MPI_Call_Type::COMM_RANK,
            Comm_Rank_Args{comm, (error == MPI_SUCCESS) ? r : -1},
            next_sequence()
        };
        push(rec);
        return error;
    }

    /// Record a MPI_Query_thread call and return the configured level/error.
    int record_query_thread(int* provided) noexcept {
        int error = query_thread_error_.load(std::memory_order_relaxed);
        int level = thread_level_.load(std::memory_order_relaxed);

        if (error == MPI_SUCCESS && provided) {
            *provided = level;
        }

        MPI_Call_Record rec{
            MPI_Call_Type::QUERY_THREAD,
            Query_Thread_Args{(error == MPI_SUCCESS) ? level : 0},
            next_sequence()
        };
        push(rec);
        return error;
    }

    /// Record a MPI_Initialized call and return the configured state.
    int record_initialized(int* flag) noexcept {
        bool init = initialized_.load(std::memory_order_relaxed);
        if (flag) {
            *flag = init ? 1 : 0;
        }

        MPI_Call_Record rec{
            MPI_Call_Type::INITIALIZED,
            Initialized_Args{init ? 1 : 0},
            next_sequence()
        };
        push(rec);
        return MPI_SUCCESS; // MPI_Initialized always returns SUCCESS
    }

    /// Record a MPI_Abort call. Does NOT actually terminate.
    int record_abort(MPI_Comm comm, int errorcode) noexcept {
        MPI_Call_Record rec{
            MPI_Call_Type::ABORT,
            Abort_Args{comm, errorcode},
            next_sequence()
        };
        push(rec);
        abort_called_.store(true, std::memory_order_release);
        return MPI_SUCCESS;
    }

    /// Record a MPI_Gather call and return the configured error.
    int record_gather(MPI_Comm comm, int root,
                      int sendcount, int recvcount) noexcept {
        int error = gather_error_.load(std::memory_order_relaxed);

        MPI_Call_Record rec{
            MPI_Call_Type::GATHER,
            Gather_Args{comm, root, sendcount, recvcount},
            next_sequence()
        };
        push(rec);
        return error;
    }

    /// Record a MPI_Allgather call and return the configured error.
    int record_allgather(MPI_Comm comm,
                         int sendcount, int recvcount) noexcept {
        int error = allgather_error_.load(std::memory_order_relaxed);

        MPI_Call_Record rec{
            MPI_Call_Type::ALLGATHER,
            Allgather_Args{comm, sendcount, recvcount},
            next_sequence()
        };
        push(rec);
        return error;
    }

    // ─── Inspection ─────────────────────────────────────────────────────

    /// Return a snapshot of all recorded calls (thread-safe copy).
    [[nodiscard]] std::vector<MPI_Call_Record> calls() const {
        std::lock_guard lock(mutex_);
        return calls_;
    }

    /// Return only calls of a specific type.
    [[nodiscard]] std::vector<MPI_Call_Record> calls_of_type(
            MPI_Call_Type type) const {
        std::lock_guard lock(mutex_);
        std::vector<MPI_Call_Record> result;
        for (const auto& c : calls_) {
            if (c.type == type) {
                result.push_back(c);
            }
        }
        return result;
    }

    /// Return the total number of recorded calls.
    [[nodiscard]] std::size_t call_count() const noexcept {
        std::lock_guard lock(mutex_);
        return calls_.size();
    }

    /// Return the number of calls of a specific type.
    [[nodiscard]] std::size_t call_count(MPI_Call_Type type) const noexcept {
        std::lock_guard lock(mutex_);
        std::size_t count = 0;
        for (const auto& c : calls_) {
            if (c.type == type) { ++count; }
        }
        return count;
    }

    /// Check if MPI_Abort was recorded.
    [[nodiscard]] bool abort_called() const noexcept {
        return abort_called_.load(std::memory_order_acquire);
    }

    // ─── Reset ──────────────────────────────────────────────────────────

    /// Clear all recorded calls and reset configuration to defaults.
    void reset() noexcept {
        std::lock_guard lock(mutex_);
        calls_.clear();
        sequence_.store(0, std::memory_order_relaxed);
        rank_.store(0, std::memory_order_relaxed);
        thread_level_.store(MPI_THREAD_SINGLE, std::memory_order_relaxed);
        initialized_.store(true, std::memory_order_relaxed);
        comm_rank_error_.store(MPI_SUCCESS, std::memory_order_relaxed);
        query_thread_error_.store(MPI_SUCCESS, std::memory_order_relaxed);
        gather_error_.store(MPI_SUCCESS, std::memory_order_relaxed);
        allgather_error_.store(MPI_SUCCESS, std::memory_order_relaxed);
        abort_called_.store(false, std::memory_order_relaxed);
    }

private:
    MPI_Spy() noexcept = default;
    ~MPI_Spy() = default;
    MPI_Spy(const MPI_Spy&) = delete;
    MPI_Spy& operator=(const MPI_Spy&) = delete;

    void push(const MPI_Call_Record& rec) noexcept {
        std::lock_guard lock(mutex_);
        calls_.push_back(rec);
    }

    std::uint64_t next_sequence() noexcept {
        return sequence_.fetch_add(1, std::memory_order_relaxed);
    }

    // ─── State ──────────────────────────────────────────────────────────

    mutable std::mutex          mutex_;
    std::vector<MPI_Call_Record> calls_;
    std::atomic<std::uint64_t>  sequence_{0};

    // Configurable return values
    std::atomic<int>  rank_{0};
    std::atomic<int>  thread_level_{MPI_THREAD_SINGLE};
    std::atomic<bool> initialized_{true};

    // Error injection
    std::atomic<int> comm_rank_error_{MPI_SUCCESS};
    std::atomic<int> query_thread_error_{MPI_SUCCESS};
    std::atomic<int> gather_error_{MPI_SUCCESS};
    std::atomic<int> allgather_error_{MPI_SUCCESS};

    // Abort tracking
    std::atomic<bool> abort_called_{false};
};

} // namespace logs::testing

#endif // LOGS_TESTS_MPI_INTERPOSITION_HPP
