/// @file logger.cpp
/// @brief Logger implementation — thread-safe, MPI-rank-aware logging.
///
/// Requirements: 2.1, 2.2, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.7, 3.8,
///               6.1, 6.2, 6.3, 6.6, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6,
///               9.7, 11.1, 11.2

#include "logs/logger.hpp"

#include <mpi.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "logs/detail/context_stack.hpp"

namespace logs {

// ─── Construction / Destruction ─────────────────────────────────────────────

Logger::Logger() noexcept = default;

Logger::~Logger() = default;

// ─── Configuration ──────────────────────────────────────────────────────────

void Logger::configure_communicator(MPI_Comm comm) noexcept {
    try {
        mpi_.detect(comm);
    } catch (...) {
        // Exception Boundary: absorb, attempt diagnostic.
        emit_diagnostic("configure_communicator: MPI detection failed");
    }
}

void Logger::set_threshold(Severity_Level level) noexcept {
    threshold_.store(level, std::memory_order_release);
}

Severity_Level Logger::threshold() const noexcept {
    return threshold_.load(std::memory_order_acquire);
}

void Logger::add_sink(Sink sink) noexcept {
    try {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        if (sinks_.size() >= MAX_SINKS) {
            // Requirement 6.1: bounded to MAX_SINKS=64, reject beyond.
            // Absorbed silently — do not propagate.
            return;
        }
        sinks_.push_back(sink);
    } catch (...) {
        // Exception Boundary: absorb allocation failure.
        emit_diagnostic("add_sink: failed to register sink");
    }
}

// ─── Query ──────────────────────────────────────────────────────────────────

int Logger::rank() const noexcept {
    return mpi_.rank();
}

int Logger::thread_support_level() const noexcept {
    return mpi_.thread_level();
}

// ─── Logging Entry Points ───────────────────────────────────────────────────

void Logger::log(Severity_Level severity, std::string_view message, const Submit_Options &opts) noexcept {
    try {
        // Requirement 3.2, 3.3: Severity filtering — accept >= threshold.
        // Requirement 3.7: FATAL is never suppressed regardless of threshold.
        if (severity != Severity_Level::FATAL && severity < threshold_.load(std::memory_order_acquire)) {
            return;
        }

        // Requirement 2.5, 2.6: Rank stamping from Mpi_Environment.
        const int current_rank = mpi_.rank();

        // Context snapshot from thread-local stack.
        std::vector<std::string> context_labels = detail::snapshot_context();

        // Optional stack-trace formatting.
        std::optional<std::string> stack_trace_text;
        if (opts.include_stack_trace && !opts.frames.empty()) {
            stack_trace_text = format_stack_trace(opts.frames);
        }

        // Construct immutable Log_Record.
        Log_Record record(severity, std::string(message), current_rank, opts.location, std::move(context_labels), std::move(stack_trace_text));

        // Buffer for consolidation.
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            consolidation_buffer_.push_back(record);
        }

        // Format and dispatch.
        std::string formatted = format_record(record);
        dispatch_to_sinks(formatted);

        // If FATAL, trigger the Synchronized Abort path.
        if (severity == Severity_Level::FATAL) {
            flush_all_sinks();
            if (abort_latch_.try_acquire()) {
                if (mpi_.has_communicator()) {
                    MPI_Abort(mpi_.communicator(), ABORT_EXIT_CODE);
                } else {
                    std::exit(ABORT_EXIT_CODE);
                }
            }
            // If another thread already latched, we still must not return
            // from a FATAL log. Spin until process terminates.
            while (true) {
                // NOLINTNEXTLINE(hicpp-no-assembler)
            }
        }

    } catch (...) {
        // Exception Boundary: absorb internal failures.
        emit_diagnostic("log: internal failure during record processing");
    }
}

[[noreturn]] void Logger::fatal(std::string_view message, const Submit_Options &opts) noexcept {
    // Delegate to log() which handles the FATAL path.
    log(Severity_Level::FATAL, message, opts);

    // If log() somehow returned (should not happen for FATAL, but
    // Exception Boundary may have caught), ensure we never return.
    flush_all_sinks();
    if (abort_latch_.try_acquire()) {
        if (mpi_.has_communicator()) {
            MPI_Abort(mpi_.communicator(), ABORT_EXIT_CODE);
        } else {
            std::exit(ABORT_EXIT_CODE);
        }
    }
    // Spin if another thread holds the latch.
    while (true) {
        // NOLINTNEXTLINE(hicpp-no-assembler)
    }
}

// ─── Consolidation ──────────────────────────────────────────────────────────

void Logger::consolidate() noexcept {
    try {
        // Requirement 4.7: Operate over buffered records, then clear buffer.
        // Take a snapshot of the consolidation buffer (swap + clear under lock).
        std::vector<Log_Record> buffered;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            buffered.swap(consolidation_buffer_);
        }

        if (buffered.empty()) {
            return;
        }

        // Consolidate: collective when communicator configured, local fallback.
        std::vector<detail::Consolidated_Record> representatives;

        if (mpi_.has_communicator()) {
            // Requirement 4.2: Collective consolidation via MPI gather.
            representatives = engine_.consolidate_collective(std::span<const Log_Record>(buffered), mpi_);
        } else {
            // Requirement 4.9: Local consolidation fallback, no MPI issued.
            representatives = engine_.consolidate_local(std::span<const Log_Record>(buffered));
        }

        // Emit representatives to sinks.
        // For collective path: only root (rank 0) gets non-empty results.
        // For local path: the calling process gets results.
        for (const auto &rep : representatives) {
            // Format consolidated record as:
            // [CONSOLIDATED] [SEVERITY] (N ranks: r0-r1, r5-r9) message\n
            std::string formatted = "[CONSOLIDATED] [";
            formatted += to_string(rep.key.severity);
            formatted += "] (";
            formatted += std::to_string(rep.rank_count);
            formatted += " ranks: ";
            for (std::size_t i = 0; i < rep.ranges.size(); ++i) {
                if (i > 0) {
                    formatted += ", ";
                }
                formatted += rep.ranges[i].to_string();
            }
            formatted += ") ";
            formatted += rep.key.message;
            formatted += "\n";

            dispatch_to_sinks(formatted);
        }

    } catch (...) {
        emit_diagnostic("consolidate: internal failure");
    }
}

// ─── Private Helpers ────────────────────────────────────────────────────────

std::string Logger::format_record(const Log_Record &record) const {
    // Format: [RANK:0042] [INFO] [ctx1 > ctx2] message text\n
    //
    // Rank at fixed position, zero-padded to 4 digits (or more if rank > 9999).
    // Severity label.
    // Context labels separated by " > " (empty section if no context).
    // Message text verbatim.

    std::ostringstream oss;

    // Rank field: zero-padded to at least 4 digits.
    const int r = record.rank();
    if (r < 0) {
        oss << "[RANK:----]";
    } else {
        oss << "[RANK:" << std::setw(4) << std::setfill('0') << r << "]";
    }

    // Severity label.
    oss << " [" << to_string(record.severity()) << "]";

    // Context labels.
    const auto &labels = record.context_labels();
    if (!labels.empty()) {
        oss << " [";
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) {
                oss << " > ";
            }
            oss << labels[i];
        }
        oss << "]";
    }

    // Message text verbatim.
    oss << " " << record.message() << "\n";

    return oss.str();
}

void Logger::dispatch_to_sinks(std::string_view formatted) noexcept {
    try {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        if (sinks_.empty()) {
            // Requirement 6.2: Zero sinks configured → write to stderr.
            std::cerr << formatted;
            std::cerr.flush();
            return;
        }
        // Requirement 6.3: Write to each sink exactly once.
        for (auto &sink : sinks_) {
            // Requirement 6.6: Sink write-failure isolation — absorb per-sink
            // failures so remaining sinks still receive the record.
            // Requirement 6.7: Retain the failing sink (do not remove it).
            (void)sink.write(formatted);
        }
    } catch (...) {
        // Exception Boundary: absorb, attempt stderr fallback.
        try {
            std::cerr << formatted;
        } catch (...) {
            // Completely absorbed.
        }
    }
}

void Logger::flush_all_sinks() noexcept {
    try {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto &sink : sinks_) {
            (void)sink.flush();
        }
    } catch (...) {
        // Absorbed: flushing failures must not prevent abort.
    }
}

void Logger::emit_diagnostic(std::string_view context) noexcept {
    // Requirement 9.3, 9.6: At most one ERROR diagnostic attempted.
    // Non-recursive: if we can't acquire the diagnostic mutex, discard.
    std::unique_lock<std::mutex> lock(diagnostic_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        // Another diagnostic is in progress (recursion guard). Silently discard.
        return;
    }

    try {
        // Format a minimal ERROR diagnostic and write to stderr.
        // We do NOT recurse into log() to avoid infinite loops.
        std::string diagnostic = "[LOGS DIAGNOSTIC] [ERROR] ";
        diagnostic += context;
        diagnostic += "\n";

        // Write directly to stderr, bypassing sink dispatch.
        std::cerr << diagnostic;
        std::cerr.flush();
    } catch (...) {
        // Completely absorbed. Diagnostic failure is silently discarded.
    }
}

}  // namespace logs
