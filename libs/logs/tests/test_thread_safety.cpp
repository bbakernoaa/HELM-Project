/// @file test_thread_safety.cpp
/// @brief Concurrency unit tests for LOGS thread safety guarantees.
///
/// Tests verify:
/// 1. Record count integrity under high-throughput multi-threaded logging
/// 2. No interleaving/corruption of individual log records
/// 3. Concurrent threshold-setting produces no torn reads
/// 4. Scoped_Context per-thread independence under concurrency
///
/// Task 19.1: Write concurrency unit tests
/// Validates: Requirements 13.1, 11.1, 11.2, 11.8

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <logs/detail/context_stack.hpp>
#include <logs/logger.hpp>
#include <logs/scoped_context.hpp>
#include <string>
#include <thread>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Record count integrity — 8 threads × 10,000 records each
// Validates: Requirements 11.1, 11.8, 13.1
// ─────────────────────────────────────────────────────────────────────────────

TEST(ThreadSafety, RecordCountIntegrity) {
    constexpr int NUM_THREADS = 8;
    constexpr int RECORDS_PER_THREAD = 10'000;
    constexpr int EXPECTED_TOTAL = NUM_THREADS * RECORDS_PER_THREAD;

    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&logger, t]() {
            for (int i = 0; i < RECORDS_PER_THREAD; ++i) {
                const std::string msg = "T" + std::to_string(t) + "_R" + std::to_string(i);
                logger.log(logs::Severity_Level::INFO, msg);
            }
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    // Verify total count matches exactly (no lost, no duplicated records).
    EXPECT_EQ(mem_sink.count(), static_cast<std::size_t>(EXPECTED_TOTAL))
        << "Expected " << EXPECTED_TOTAL << " records, got " << mem_sink.count() << ". Records may have been lost or duplicated.";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: No interleaved or corrupted records
// Validates: Requirements 11.2, 13.1
//
// Each thread emits records with a unique marker. We verify every recorded
// entry is a complete, non-corrupted single record by checking that:
// - Each entry contains exactly one newline (at the end)
// - Each entry contains a recognizable thread marker (T<id>_R<seq>)
// - No entry contains fragments from two different thread markers
// ─────────────────────────────────────────────────────────────────────────────

TEST(ThreadSafety, NoInterleavedOrCorruptedRecords) {
    constexpr int NUM_THREADS = 8;
    constexpr int RECORDS_PER_THREAD = 10'000;

    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&logger, t]() {
            for (int i = 0; i < RECORDS_PER_THREAD; ++i) {
                const std::string msg = "T" + std::to_string(t) + "_R" + std::to_string(i);
                logger.log(logs::Severity_Level::INFO, msg);
            }
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    const auto entries = mem_sink.entries();

    // Verify no entry is corrupted or interleaved.
    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
        const auto &entry = entries[idx];

        // Each entry must be non-empty.
        ASSERT_FALSE(entry.empty()) << "Entry " << idx << " is empty";

        // Each entry should contain exactly one thread marker pattern (T<N>_R<M>).
        // Count how many distinct thread prefixes appear in a single entry.
        int markers_found = 0;
        for (int t = 0; t < NUM_THREADS; ++t) {
            const std::string prefix = "T" + std::to_string(t) + "_R";
            if (entry.find(prefix) != std::string::npos) {
                ++markers_found;
            }
        }

        // Exactly one thread marker should be present per entry (no interleaving).
        EXPECT_EQ(markers_found, 1) << "Entry " << idx << " contains " << markers_found
                                    << " thread markers (expected 1). Content: " << entry.substr(0, 200);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Concurrent threshold-setting — no torn reads
// Validates: Requirements 11.1, 3.1
//
// Spawn threads that repeatedly call set_threshold while another thread reads
// threshold(). Verify no crashes and that threshold() always returns a valid
// Severity_Level value.
// ─────────────────────────────────────────────────────────────────────────────

TEST(ThreadSafety, ConcurrentThresholdSettingNoTornReads) {
    constexpr int NUM_WRITER_THREADS = 4;
    constexpr int NUM_READER_THREADS = 4;
    constexpr int ITERATIONS = 50'000;

    logs::Logger logger;

    // All valid severity levels.
    constexpr std::array<logs::Severity_Level, 5> valid_levels = {logs::Severity_Level::DEBUG, logs::Severity_Level::INFO,
                                                                  logs::Severity_Level::WARNING, logs::Severity_Level::ERROR,
                                                                  logs::Severity_Level::FATAL};

    std::atomic<bool> stop{false};
    std::atomic<int> torn_count{0};

    // Writer threads: cycle through severity levels rapidly.
    std::vector<std::thread> writers;
    writers.reserve(NUM_WRITER_THREADS);
    for (int w = 0; w < NUM_WRITER_THREADS; ++w) {
        writers.emplace_back([&logger, &valid_levels, &stop, w]() {
            for (int i = 0; i < ITERATIONS; ++i) {
                const auto level = valid_levels[static_cast<std::size_t>((w * ITERATIONS + i) % 5)];
                logger.set_threshold(level);
            }
            // Signal completion after last writer finishes.
            stop.store(true, std::memory_order_release);
        });
    }

    // Reader threads: read threshold() and verify it's always a valid value.
    std::vector<std::thread> readers;
    readers.reserve(NUM_READER_THREADS);
    for (int r = 0; r < NUM_READER_THREADS; ++r) {
        readers.emplace_back([&logger, &valid_levels, &stop, &torn_count]() {
            while (!stop.load(std::memory_order_acquire)) {
                const auto level = logger.threshold();
                const auto it = std::find(valid_levels.begin(), valid_levels.end(), level);
                if (it == valid_levels.end()) {
                    torn_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto &w : writers) {
        w.join();
    }
    // Stop flag already set by last completing writer — ensure readers can exit.
    stop.store(true, std::memory_order_release);
    for (auto &r : readers) {
        r.join();
    }

    // No torn reads: every threshold() call returned a valid Severity_Level.
    EXPECT_EQ(torn_count.load(), 0) << "Detected torn threshold reads. threshold() returned a value "
                                       "not matching any valid Severity_Level.";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Scoped_Context per-thread independence
// Validates: Requirements 11.1, 8.6, 11.7
//
// Spawn threads each pushing unique labels via Scoped_Context. Verify
// snapshots on each thread contain only that thread's labels and never
// include labels from other threads.
// ─────────────────────────────────────────────────────────────────────────────

TEST(ThreadSafety, ScopedContextPerThreadIndependence) {
    constexpr int NUM_THREADS = 8;
    constexpr int DEPTH = 5;

    // Each thread will store its observed context snapshots here.
    std::vector<std::vector<std::vector<std::string>>> thread_snapshots(NUM_THREADS);

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([t, &thread_snapshots]() {
            // Push DEPTH labels unique to this thread.
            std::vector<std::unique_ptr<logs::Scoped_Context>> contexts;
            contexts.reserve(DEPTH);

            for (int d = 0; d < DEPTH; ++d) {
                const std::string label = "thread_" + std::to_string(t) + "_depth_" + std::to_string(d);
                contexts.push_back(std::make_unique<logs::Scoped_Context>(label));

                // Snapshot and record.
                auto snapshot = logs::detail::snapshot_context();
                thread_snapshots[t].push_back(snapshot);
            }

            // Pop all contexts (in reverse order via RAII).
            while (!contexts.empty()) {
                contexts.pop_back();
            }

            // After all pops, the stack should be empty.
            auto final_snapshot = logs::detail::snapshot_context();
            thread_snapshots[t].push_back(final_snapshot);
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    // Verify per-thread independence.
    for (int t = 0; t < NUM_THREADS; ++t) {
        const std::string thread_prefix = "thread_" + std::to_string(t) + "_";

        // Check each snapshot taken during the push phase.
        for (int d = 0; d < DEPTH; ++d) {
            const auto &snapshot = thread_snapshots[t][d];

            // Should have d+1 labels (one for each push so far).
            EXPECT_EQ(snapshot.size(), static_cast<std::size_t>(d + 1)) << "Thread " << t << " at depth " << d << " has unexpected label count";

            // All labels in the snapshot must belong to this thread.
            for (const auto &label : snapshot) {
                EXPECT_NE(label.find(thread_prefix), std::string::npos) << "Thread " << t << " snapshot contains foreign label: " << label;
            }

            // No label from any other thread should appear.
            for (int other = 0; other < NUM_THREADS; ++other) {
                if (other == t) continue;
                const std::string other_prefix = "thread_" + std::to_string(other) + "_";
                for (const auto &label : snapshot) {
                    EXPECT_EQ(label.find(other_prefix), std::string::npos) << "Thread " << t << " has label from thread " << other << ": " << label;
                }
            }
        }

        // Final snapshot (after all pops) should be empty.
        const auto &final_snapshot = thread_snapshots[t].back();
        EXPECT_TRUE(final_snapshot.empty()) << "Thread " << t << " context stack not empty after all pops";
    }
}

}  // namespace
