/// @file test_exception_boundary.cpp
/// @brief Property tests for the Exception Boundary of the Logger.
///
/// Property 35: Bounded, Non-Recursive Absorbed-Failure Diagnostic
/// Validates: Requirements 9.3, 9.6
///
/// Tests verify that:
///   - When an internal failure occurs during logging (e.g., allocation failure),
///     the Exception Boundary catches it and emits at most one ERROR diagnostic
///     to stderr via emit_diagnostic().
///   - Recursive diagnostic failures are silently discarded (non-recursive via
///     try_to_lock on diagnostic_mutex_).
///   - Concurrent failures still produce bounded diagnostics (no cascading).

#include <gtest/gtest.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <logs/logger.hpp>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure: RAII stderr capturer using std::cerr rdbuf redirection.
// ─────────────────────────────────────────────────────────────────────────────

/// Redirects std::cerr to an internal stringstream for the lifetime of
/// this object; restores the original streambuf on destruction.
class Stderr_Capturer {
   public:
    Stderr_Capturer() : original_buf_(std::cerr.rdbuf()) {
        std::cerr.rdbuf(capture_stream_.rdbuf());
    }

    ~Stderr_Capturer() {
        std::cerr.rdbuf(original_buf_);
    }

    /// Returns all text written to std::cerr since construction.
    [[nodiscard]] std::string captured() const {
        return capture_stream_.str();
    }

    /// Clear the captured buffer.
    void clear() {
        capture_stream_.str("");
        capture_stream_.clear();
    }

    Stderr_Capturer(const Stderr_Capturer &) = delete;
    Stderr_Capturer &operator=(const Stderr_Capturer &) = delete;

   private:
    std::streambuf *original_buf_;
    std::stringstream capture_stream_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure: A streambuf that always throws on write (for breaking cerr).
// ─────────────────────────────────────────────────────────────────────────────

class Throwing_Streambuf : public std::streambuf {
   protected:
    std::streamsize xsputn(const char * /*s*/, std::streamsize /*n*/) override {
        throw std::runtime_error("injected streambuf failure");
    }

    int_type overflow(int_type /*ch*/) override {
        throw std::runtime_error("injected streambuf failure (overflow)");
    }

    int sync() override {
        return 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure: RAII virtual memory limiter to force bad_alloc in Logger::log.
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard that restricts the process virtual address space to force
/// std::bad_alloc inside Logger::log() when it tries to copy/format a large
/// message. Restores the original limit on destruction.
class Memory_Limiter {
   public:
    explicit Memory_Limiter(rlim_t limit_bytes) {
        getrlimit(RLIMIT_AS, &original_);
        struct rlimit restricted = original_;
        restricted.rlim_cur = limit_bytes;
        setrlimit(RLIMIT_AS, &restricted);
    }

    ~Memory_Limiter() {
        setrlimit(RLIMIT_AS, &original_);
    }

    Memory_Limiter(const Memory_Limiter &) = delete;
    Memory_Limiter &operator=(const Memory_Limiter &) = delete;

   private:
    struct rlimit original_{};
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Count occurrences of a substring in a string.
// ─────────────────────────────────────────────────────────────────────────────

std::size_t count_occurrences(const std::string &text, const std::string &pattern) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos) {
        ++count;
        pos += pattern.size();
    }
    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 35: Bounded, Non-Recursive Absorbed-Failure Diagnostic
// Validates: Requirements 9.3, 9.6
// ─────────────────────────────────────────────────────────────────────────────

/// Test: When an internal allocation failure occurs during logging, the
/// Exception Boundary emits exactly one ERROR diagnostic to stderr.
/// The diagnostic contains "[LOGS DIAGNOSTIC] [ERROR]".
///
/// Strategy: Restrict virtual memory so that Logger::log()'s internal
/// string copy/format operations throw std::bad_alloc, which is caught
/// by the outer try/catch and routed to emit_diagnostic().
///
/// **Validates: Requirements 9.3, 9.6**
TEST(ExceptionBoundaryDiagnostic, SingleFailureEmitsAtMostOneDiagnostic) {
    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Pre-allocate a large message BEFORE restricting memory.
    // This message will be passed as string_view to log(), which
    // internally tries to copy it (std::string(message)) under memory
    // pressure, causing bad_alloc.
    const std::size_t large_size = 50 * 1024 * 1024;  // 50 MB
    std::string large_message;
    try {
        large_message.assign(large_size, 'X');
    } catch (const std::bad_alloc &) {
        GTEST_SKIP() << "Cannot allocate test message; skipping";
    }

    Stderr_Capturer capturer;

    {
        // Restrict to 100 MB — enough for existing allocations but not
        // enough to duplicate the 50 MB message inside log().
        Memory_Limiter limiter(100 * 1024 * 1024);

        // This call triggers bad_alloc inside log()'s try block,
        // which calls emit_diagnostic().
        logger.log(logs::Severity_Level::INFO, large_message);
    }

    const std::string output = capturer.captured();

    // Requirement 9.3: At most one ERROR diagnostic attempted per failure.
    const std::size_t diag_count = count_occurrences(output, "[LOGS DIAGNOSTIC] [ERROR]");
    EXPECT_EQ(diag_count, 1u) << "Expected exactly one diagnostic, got " << diag_count << "\nCaptured stderr:\n" << output;

    // Verify the diagnostic contains the expected context string.
    EXPECT_NE(output.find("log: internal failure"), std::string::npos) << "Diagnostic should mention 'log: internal failure'";
}

/// Test: A second call that fails also produces at most one diagnostic.
/// Each failure event generates at most one diagnostic — no cascading.
///
/// **Validates: Requirements 9.3, 9.6**
TEST(ExceptionBoundaryDiagnostic, RepeatedFailuresEachBoundedToOneDiagnostic) {
    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    const std::size_t large_size = 50 * 1024 * 1024;
    std::string large_message;
    try {
        large_message.assign(large_size, 'X');
    } catch (const std::bad_alloc &) {
        GTEST_SKIP() << "Cannot allocate test message; skipping";
    }

    Stderr_Capturer capturer;

    {
        Memory_Limiter limiter(100 * 1024 * 1024);

        // First failure.
        logger.log(logs::Severity_Level::INFO, large_message);
    }

    const std::string after_first = capturer.captured();
    const std::size_t count_first = count_occurrences(after_first, "[LOGS DIAGNOSTIC] [ERROR]");
    EXPECT_LE(count_first, 1u) << "First call: expected at most one diagnostic, got " << count_first;

    {
        Memory_Limiter limiter(100 * 1024 * 1024);

        // Second failure (separate event).
        logger.log(logs::Severity_Level::WARNING, large_message);
    }

    const std::string after_second = capturer.captured();
    const std::size_t count_total = count_occurrences(after_second, "[LOGS DIAGNOSTIC] [ERROR]");

    // Each failure should produce at most one diagnostic, so total <= 2.
    EXPECT_LE(count_total, 2u) << "Two failures: expected at most two diagnostics, got " << count_total;

    // The increment from first to second call should be at most 1.
    const std::size_t second_increment = count_total - count_first;
    EXPECT_LE(second_increment, 1u) << "Second call alone produced " << second_increment << " diagnostics (expected at most 1)";
}

/// Test: Concurrent failures from multiple threads each produce bounded
/// diagnostics. The try_to_lock non-recursive guard ensures that if a
/// diagnostic is already in progress, concurrent attempts are silently
/// discarded.
///
/// **Validates: Requirements 9.3, 9.6**
TEST(ExceptionBoundaryDiagnostic, ConcurrentFailuresBoundedDiagnostics) {
    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    const std::size_t large_size = 50 * 1024 * 1024;
    std::string large_message;
    try {
        large_message.assign(large_size, 'X');
    } catch (const std::bad_alloc &) {
        GTEST_SKIP() << "Cannot allocate test message; skipping";
    }

    constexpr int num_threads = 4;

    Stderr_Capturer capturer;

    {
        Memory_Limiter limiter(100 * 1024 * 1024);

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&logger, &large_message]() {
                // Each thread attempts one log that will fail.
                logger.log(logs::Severity_Level::INFO, large_message);
            });
        }

        for (auto &th : threads) {
            th.join();
        }
    }

    const std::string output = capturer.captured();
    const std::size_t total_diagnostics = count_occurrences(output, "[LOGS DIAGNOSTIC] [ERROR]");

    // Property: Each failure produces at most one diagnostic.
    // With try_to_lock, concurrent attempts that can't acquire the mutex
    // are silently discarded. So total <= num_threads (one per failure).
    EXPECT_LE(total_diagnostics, static_cast<std::size_t>(num_threads))
        << "Diagnostics (" << total_diagnostics << ") should not exceed thread count (" << num_threads << ")";

    // At least one thread should succeed in emitting a diagnostic.
    EXPECT_GE(total_diagnostics, 1u) << "At least one diagnostic should have been emitted";

    // Due to concurrency and try_to_lock, some diagnostics are discarded.
    // With 4 threads hitting the mutex simultaneously, we expect fewer
    // diagnostics than threads (verifies the non-recursive property).
    // Note: This is probabilistic — under extreme scheduling, all 4 might
    // serialize. The key invariant is total <= num_threads.
}

/// Test: When the diagnostic mechanism itself fails (cerr is broken),
/// the failure is silently discarded — no recursion, no crash,
/// no exception propagation.
///
/// **Validates: Requirements 9.6**
TEST(ExceptionBoundaryDiagnostic, DiagnosticFailureIsSilentlyDiscarded) {
    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    const std::size_t large_size = 50 * 1024 * 1024;
    std::string large_message;
    try {
        large_message.assign(large_size, 'X');
    } catch (const std::bad_alloc &) {
        GTEST_SKIP() << "Cannot allocate test message; skipping";
    }

    // Replace cerr's streambuf with one that throws on write.
    // This means emit_diagnostic()'s attempt to write to cerr will throw
    // internally. emit_diagnostic catches that and silently discards.
    Throwing_Streambuf broken_cerr_buf;
    std::streambuf *original_cerr = std::cerr.rdbuf(&broken_cerr_buf);

    {
        Memory_Limiter limiter(100 * 1024 * 1024);

        // log() catches bad_alloc -> calls emit_diagnostic() ->
        // emit_diagnostic tries to write to cerr -> cerr throws ->
        // emit_diagnostic catches -> silently discards.
        // No crash, no recursion, no exception propagation.
        EXPECT_NO_THROW(logger.log(logs::Severity_Level::INFO, large_message));
    }

    // Restore cerr.
    std::cerr.rdbuf(original_cerr);

    // Logger state remains valid after the failure (Requirement 9.7).
    // Verify by successfully logging a small message.
    Stderr_Capturer capturer;
    // Note: The consolidation_buffer_ may have leftover entries from the
    // failed attempt, but the logger is still usable. A small message
    // should succeed.
    logger.log(logs::Severity_Level::WARNING, "post-failure OK");

    // The message went through log() successfully — if no exception was
    // thrown, the state is valid. We don't check stderr output here because
    // the default behavior (no configured sinks) writes to cerr which is
    // now captured.
    const std::string output = capturer.captured();
    // The record should appear in stderr (default sink behavior when no
    // sinks are configured).
    EXPECT_NE(output.find("post-failure OK"), std::string::npos) << "Logger should remain functional after absorbed diagnostic failure";
}

}  // namespace
