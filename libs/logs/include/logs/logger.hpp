#ifndef LOGS_LOGGER_HPP
#define LOGS_LOGGER_HPP

/// @file logger.hpp
/// @brief Logger central thread-safe entry point.
///
/// Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.7, 3.8,
///               6.1, 6.2, 6.3, 6.6, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6,
///               9.7, 11.1, 11.2

#include <mpi.h>

#include <atomic>
#include <mutex>
#include <span>
#include <vector>

#include "logs/detail/consolidation.hpp"
#include "logs/detail/mpi_environment.hpp"
#include "logs/log_record.hpp"
#include "logs/severity.hpp"
#include "logs/sink.hpp"
#include "logs/stack_trace.hpp"

namespace logs {

namespace detail {

/// Ensures MPI_Abort is invoked at most once per process, even under
/// concurrent or subsequent FATAL submissions (Requirements 5.8, 11.6).
class Abort_Latch {
   public:
    /// Returns true exactly once (to the first caller). All later callers
    /// receive false and must not invoke MPI_Abort.
    [[nodiscard]] bool try_acquire() noexcept {
        bool expected = false;
        return latched_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

   private:
    std::atomic<bool> latched_{false};
};

}  // namespace detail

/// Options for a log submission.
struct Submit_Options {
    std::optional<Source_Location> location{};
    std::span<const Stack_Frame> frames{};
    bool include_stack_trace{false};
};

/// Central thread-safe logging entry point.
///
/// All public entry points are noexcept (Exception Boundary). Internal failures
/// are absorbed, and at most one ERROR diagnostic is attempted on failure
/// (non-recursive via diagnostic_mutex_).
class Logger {
   public:
    /// Construct an unconfigured Logger. Before a communicator is configured,
    /// the stored rank is the sentinel -1 and the thread level defaults to
    /// MPI_THREAD_SINGLE.
    Logger() noexcept;
    ~Logger();

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    // ---- Configuration (programmatic only; never read from any file) ----

    /// Configure the MPI communicator. Calls MPI_Comm_rank exactly once and
    /// stores the rank; queries MPI_Query_thread and stores the thread level.
    void configure_communicator(MPI_Comm comm) noexcept;

    /// Atomically set the active severity threshold (last-writer-wins).
    void set_threshold(Severity_Level level) noexcept;

    /// Query the current severity threshold.
    [[nodiscard]] Severity_Level threshold() const noexcept;

    /// Register an output-stream Sink. Rejected (absorbed) beyond MAX_SINKS.
    void add_sink(Sink sink) noexcept;

    // ---- Query (all thread-safe) ----

    /// Returns stored rank (-1 if unconfigured).
    [[nodiscard]] int rank() const noexcept;

    /// Returns stored MPI thread level constant.
    [[nodiscard]] int thread_support_level() const noexcept;

    // ---- Logging entry points (noexcept Exception Boundary) ----

    /// Submit a record. Applies severity filtering, rank stamping, context
    /// snapshot, formatting, and dispatch. FATAL is never suppressed and
    /// triggers the Synchronized_Abort path.
    void log(Severity_Level severity, std::string_view message, const Submit_Options &opts = {}) noexcept;

    /// Convenience: emit a FATAL record and initiate Synchronized_Abort.
    [[noreturn]] void fatal(std::string_view message, const Submit_Options &opts = {}) noexcept;

    // ---- Consolidation ----

    /// Collective consolidation placeholder. Buffers records for future
    /// wiring to Consolidation_Engine (task 14.2).
    void consolidate() noexcept;

   private:
    /// Format a Log_Record into the standard output format:
    /// [RANK:0042] [INFO] [ctx1 > ctx2] message text\n
    [[nodiscard]] std::string format_record(const Log_Record &record) const;

    /// Dispatch a formatted record to all configured sinks (or stderr if none).
    void dispatch_to_sinks(std::string_view formatted) noexcept;

    /// Flush all configured sinks.
    void flush_all_sinks() noexcept;

    /// Attempt to emit an ERROR diagnostic about an internal failure.
    /// Non-recursive: if diagnostic emission itself fails, silently discard.
    void emit_diagnostic(std::string_view context) noexcept;

    detail::Mpi_Environment mpi_;
    detail::Consolidation_Engine engine_;
    std::atomic<Severity_Level> threshold_{Severity_Level::INFO};

    std::mutex sinks_mutex_;
    std::vector<Sink> sinks_;

    std::mutex buffer_mutex_;
    std::vector<Log_Record> consolidation_buffer_;

    detail::Abort_Latch abort_latch_;
    std::mutex diagnostic_mutex_;
};

/// Fixed non-zero abort exit code, in the inclusive range 1..255
/// (Requirement 5.3). EX_SOFTWARE-style fixed code.
inline constexpr int ABORT_EXIT_CODE = 70;

}  // namespace logs

#endif  // LOGS_LOGGER_HPP
