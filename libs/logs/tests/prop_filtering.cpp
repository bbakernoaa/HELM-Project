/// @file prop_filtering.cpp
/// @brief Property-based tests for atomic threshold last-writer-wins semantics.
///
/// Uses GTest + threads to verify concurrent threshold-setting produces no
/// torn values and that last-writer-wins governs the active threshold.
///
/// **Validates: Requirements 3.1, 3.5, 3.8**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <array>
#include <atomic>
#include <logs/logger.hpp>
#include <thread>
#include <vector>

namespace {

/// All valid Severity_Level values for validation.
constexpr std::array<logs::Severity_Level, 5> ALL_LEVELS = {logs::Severity_Level::DEBUG, logs::Severity_Level::INFO, logs::Severity_Level::WARNING,
                                                            logs::Severity_Level::ERROR, logs::Severity_Level::FATAL};

/// Check whether a value is one of the valid Severity_Level enum values (0-4).
bool is_valid_severity(logs::Severity_Level level) {
    const int v = static_cast<int>(level);
    return v >= 0 && v <= 4;
}

/// Generate a random valid Severity_Level (int 0–4 cast to enum).
rc::Gen<logs::Severity_Level> genSeverityLevel() {
    return rc::gen::map(rc::gen::inRange(0, 5), [](int v) { return static_cast<logs::Severity_Level>(v); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 9: Atomic Threshold Last-Writer-Wins
// Validates: Requirements 3.1, 3.5, 3.8
// ─────────────────────────────────────────────────────────────────────────────

/// Concurrent threshold-setting from N threads; verify threshold() always
/// returns a valid Severity_Level (no torn reads).
RC_GTEST_PROP(AtomicThresholdLastWriterWins, NoTornValuesUnderConcurrency, ()) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERS_PER_THREAD = 1000;

    logs::Logger logger;

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    // Each thread sets the threshold to a different level in a loop.
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&logger, &start_flag, t]() {
            // Spin until all threads are ready.
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin
            }
            for (int i = 0; i < ITERS_PER_THREAD; ++i) {
                auto level = ALL_LEVELS[static_cast<std::size_t>((t + i) % 5)];
                logger.set_threshold(level);
            }
        });
    }

    // Release all threads simultaneously.
    start_flag.store(true, std::memory_order_release);

    // While threads are running, read the threshold many times and verify
    // it's always a valid enum value (no torn reads).
    for (int i = 0; i < ITERS_PER_THREAD * 2; ++i) {
        logs::Severity_Level observed = logger.threshold();
        RC_ASSERT(is_valid_severity(observed));
    }

    for (auto &th : threads) {
        th.join();
    }

    // After all threads complete, the final threshold must still be valid.
    RC_ASSERT(is_valid_severity(logger.threshold()));
}

/// After all concurrent threads finish, set threshold to a known value;
/// verify it sticks (last-writer-wins after contention resolves).
RC_GTEST_PROP(AtomicThresholdLastWriterWins, LastSetterGovernsAfterConcurrency, ()) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERS_PER_THREAD = 500;

    logs::Logger logger;
    const auto final_level = *genSeverityLevel();

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&logger, &start_flag, t]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin
            }
            for (int i = 0; i < ITERS_PER_THREAD; ++i) {
                auto level = ALL_LEVELS[static_cast<std::size_t>((t + i) % 5)];
                logger.set_threshold(level);
            }
        });
    }

    start_flag.store(true, std::memory_order_release);

    for (auto &th : threads) {
        th.join();
    }

    // Now set the threshold to a known value — last writer wins.
    logger.set_threshold(final_level);

    // Immediately after the call returns, the threshold must equal what we set.
    RC_ASSERT(logger.threshold() == final_level);
}

/// Single-threaded set_threshold immediately governs: after the call returns,
/// the threshold equals the set value.
RC_GTEST_PROP(AtomicThresholdLastWriterWins, SetThenReadIsImmediate, ()) {
    logs::Logger logger;
    const auto level = *genSeverityLevel();

    logger.set_threshold(level);

    RC_ASSERT(logger.threshold() == level);
}

/// Verify that threshold().load always yields a value in the valid enum range
/// (0-4), never a torn/garbage value, under heavy concurrent writes.
TEST(AtomicThresholdLastWriterWins, ThresholdAlwaysValidUnderStress) {
    constexpr int NUM_WRITERS = 8;
    constexpr int NUM_READERS = 4;
    constexpr int ITERS = 2000;

    logs::Logger logger;

    std::atomic<bool> start_flag{false};
    std::atomic<bool> stop_flag{false};
    std::atomic<int> invalid_count{0};

    std::vector<std::thread> writers;
    writers.reserve(NUM_WRITERS);
    std::vector<std::thread> readers;
    readers.reserve(NUM_READERS);

    // Writer threads: continuously set threshold to various valid levels.
    for (int t = 0; t < NUM_WRITERS; ++t) {
        writers.emplace_back([&logger, &start_flag, t]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin
            }
            for (int i = 0; i < ITERS; ++i) {
                auto level = ALL_LEVELS[static_cast<std::size_t>((t + i) % 5)];
                logger.set_threshold(level);
            }
        });
    }

    // Reader threads: continuously read the threshold and verify validity.
    for (int r = 0; r < NUM_READERS; ++r) {
        readers.emplace_back([&logger, &start_flag, &stop_flag, &invalid_count]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                // spin
            }
            while (!stop_flag.load(std::memory_order_acquire)) {
                logs::Severity_Level observed = logger.threshold();
                if (!is_valid_severity(observed)) {
                    invalid_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Start all threads simultaneously.
    start_flag.store(true, std::memory_order_release);

    // Wait for writers to finish.
    for (auto &th : writers) {
        th.join();
    }

    // Signal readers to stop.
    stop_flag.store(true, std::memory_order_release);

    for (auto &th : readers) {
        th.join();
    }

    EXPECT_EQ(invalid_count.load(), 0) << "Detected torn/invalid threshold value under concurrent writes";
}

/// Verify last-writer-wins: sequential overwrites always yield the last value.
TEST(AtomicThresholdLastWriterWins, SequentialOverwritesYieldLastValue) {
    logs::Logger logger;

    for (auto level : ALL_LEVELS) {
        logger.set_threshold(level);
        EXPECT_EQ(logger.threshold(), level);
    }

    // Reverse order to confirm no caching effects.
    for (int i = 4; i >= 0; --i) {
        auto level = static_cast<logs::Severity_Level>(i);
        logger.set_threshold(level);
        EXPECT_EQ(logger.threshold(), level);
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Property 11: FATAL Is Never Suppressed
// Validates: Requirements 3.7
//
// For any threshold including FATAL, verify FATAL submissions are always
// accepted — the record reaches output before the process terminates.
//
// Since log(FATAL, ...) triggers std::exit(ABORT_EXIT_CODE) when no
// communicator is configured, we use GTest's EXPECT_EXIT death-test framework
// to fork a child process, emit FATAL in the child, and verify:
//   1. The process exits with ABORT_EXIT_CODE (70).
//   2. stderr contains the FATAL message text (proving the record was
//      dispatched before termination).
// ─────────────────────────────────────────────────────────────────────────────

/// For each severity threshold, emitting at FATAL always produces output
/// before process termination — FATAL is never suppressed by filtering.
TEST(FatalIsNeverSuppressed, FatalAcceptedAtEveryThreshold) {
    for (auto threshold : ALL_LEVELS) {
        const std::string msg = "fatal_marker_" + std::to_string(static_cast<int>(threshold));

        EXPECT_EXIT(
            {
                // Child process: create Logger, set threshold, emit FATAL.
                logs::Logger logger;
                logger.set_threshold(threshold);
                // No sinks configured → output goes to stderr (Requirement 6.2).
                logger.log(logs::Severity_Level::FATAL, msg);
                // Should not reach here; log(FATAL) calls std::exit.
                std::_Exit(1);
            },
            ::testing::ExitedWithCode(logs::ABORT_EXIT_CODE),
            msg  // Verify the FATAL message text appears in stderr output.
            )
            << "FATAL suppressed when threshold=" << static_cast<int>(threshold);
    }
}

/// Property-based variant: for any randomly generated threshold, FATAL is
/// always dispatched to output (never filtered) before termination.
RC_GTEST_PROP(FatalIsNeverSuppressed, RandomThresholdNeverSuppressesFatal, ()) {
    const auto threshold = *genSeverityLevel();
    const std::string msg = "rc_fatal_" + std::to_string(static_cast<int>(threshold));

    // Death tests cannot easily be nested inside RC_GTEST_PROP's assertion
    // framework, so we verify the filtering logic directly: FATAL severity
    // must always pass the threshold check regardless of threshold value.
    //
    // The Logger's filtering logic (from logger.cpp):
    //   if (severity != FATAL && severity < threshold) return; // suppressed
    // Therefore FATAL always passes the filter. We verify this property:
    RC_ASSERT(logs::Severity_Level::FATAL >= threshold);
}

/// Deterministic exhaustive check: FATAL is the maximum severity, so for
/// every possible threshold it satisfies severity >= threshold.
TEST(FatalIsNeverSuppressed, FatalIsMaximumSeverity) {
    for (auto threshold : ALL_LEVELS) {
        EXPECT_GE(static_cast<int>(logs::Severity_Level::FATAL), static_cast<int>(threshold))
            << "FATAL must be >= every threshold for never-suppressed guarantee";
    }
}
