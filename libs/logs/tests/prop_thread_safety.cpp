/// @file prop_thread_safety.cpp
/// @brief Property-based test: Concurrent Record-Count Integrity and
///        Non-Interleaving.
///
/// Spawns randomized thread counts, each logging randomized message counts to
/// a shared Logger with In_Memory_Sink. Verifies:
///   1. Total emitted count == N_threads * M_messages (no lost records)
///   2. Each recorded entry is a complete, non-interleaved record (contains
///      exactly one newline at the end and has valid structure)
///
/// **Validates: Requirements 11.1, 11.2, 11.8**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <logs/logger.hpp>
#include <string>
#include <thread>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Check whether a recorded entry has valid record structure:
/// - Starts with "[RANK:" prefix
/// - Ends with exactly one newline at the end
/// - Contains no embedded newlines before the trailing one
[[nodiscard]] bool is_valid_record(const std::string &entry) {
    if (entry.empty()) return false;

    // Must end with exactly one newline.
    if (entry.back() != '\n') return false;

    // Must start with the rank field prefix.
    if (entry.size() < 6 || entry.compare(0, 6, "[RANK:") != 0) return false;

    // No embedded newlines (only the trailing one is allowed).
    auto newline_count = std::count(entry.begin(), entry.end(), '\n');
    if (newline_count != 1) return false;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 36: Concurrent Record-Count Integrity and Non-Interleaving
// Validates: Requirements 11.1, 11.2, 11.8
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-threaded submissions to a shared Logger: verify emitted count equals
/// submitted count and no record's text is interleaved with another.
RC_GTEST_PROP(ConcurrentRecordCountIntegrity, CountEqualsSubmittedAndNoInterleaving, ()) {
    // Generate random N threads (2-16) and M messages per thread (100-1000).
    const auto num_threads = *rc::gen::inRange(2, 17);
    const auto msgs_per_thread = *rc::gen::inRange(100, 1001);

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Launch threads; each thread logs msgs_per_thread records.
    // Use distinct message prefixes per thread to verify no interleaving.
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads));

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&logger, t, msgs_per_thread]() {
            for (int m = 0; m < msgs_per_thread; ++m) {
                // Each message is uniquely identifiable by thread and index.
                std::string msg = "T" + std::to_string(t) + "_M" + std::to_string(m);
                logger.log(logs::Severity_Level::INFO, msg);
            }
        });
    }

    // Join all threads.
    for (auto &th : threads) {
        th.join();
    }

    // Property 1: Total count == N * M (no lost, no duplicated records).
    const std::size_t expected_count = static_cast<std::size_t>(num_threads) * static_cast<std::size_t>(msgs_per_thread);

    RC_ASSERT(mem_sink.count() == expected_count);

    // Property 2: Each entry is a complete, non-interleaved record.
    const auto entries = mem_sink.entries();
    for (const auto &entry : entries) {
        RC_ASSERT(is_valid_record(entry));
    }
}

/// Verify that every submitted record appears exactly once in the output
/// (no duplication, no loss) under concurrent submission.
RC_GTEST_PROP(ConcurrentRecordCountIntegrity, AllRecordsAppearExactlyOnce, ()) {
    // Use smaller ranges for this property since we check exact content.
    const auto num_threads = *rc::gen::inRange(2, 9);
    const auto msgs_per_thread = *rc::gen::inRange(100, 501);

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads));

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&logger, t, msgs_per_thread]() {
            for (int m = 0; m < msgs_per_thread; ++m) {
                std::string msg = "T" + std::to_string(t) + "_M" + std::to_string(m);
                logger.log(logs::Severity_Level::INFO, msg);
            }
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    const auto entries = mem_sink.entries();
    const std::size_t expected_count = static_cast<std::size_t>(num_threads) * static_cast<std::size_t>(msgs_per_thread);

    RC_ASSERT(entries.size() == expected_count);

    // Verify each unique message marker appears in exactly one entry.
    // Extract the message portion from each entry and check uniqueness.
    std::vector<std::string> messages;
    messages.reserve(entries.size());

    for (const auto &entry : entries) {
        // The message is after the last ']' + space, before '\n'.
        auto last_bracket = entry.rfind(']');
        if (last_bracket == std::string::npos) {
            RC_FAIL("Entry has no closing bracket");
            continue;
        }
        std::size_t msg_start = last_bracket + 1;
        if (msg_start < entry.size() && entry[msg_start] == ' ') {
            ++msg_start;
        }
        std::size_t msg_end = entry.find('\n', msg_start);
        if (msg_end == std::string::npos) {
            msg_end = entry.size();
        }
        messages.push_back(entry.substr(msg_start, msg_end - msg_start));
    }

    // Sort and check for duplicates.
    std::sort(messages.begin(), messages.end());
    for (std::size_t i = 1; i < messages.size(); ++i) {
        RC_ASSERT(messages[i] != messages[i - 1]);
    }

    // Check that all expected messages are present.
    for (int t = 0; t < num_threads; ++t) {
        for (int m = 0; m < msgs_per_thread; ++m) {
            std::string expected_msg = "T" + std::to_string(t) + "_M" + std::to_string(m);
            RC_ASSERT(std::binary_search(messages.begin(), messages.end(), expected_msg));
        }
    }
}

}  // namespace
