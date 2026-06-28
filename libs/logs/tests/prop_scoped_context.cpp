/// @file prop_scoped_context.cpp
/// @brief Property-based tests for Scoped_Context behavior.
///
/// Uses RapidCheck + Google Test to verify universal correctness properties
/// of the RAII scoped-context mechanism.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <logs/detail/context_stack.hpp>
#include <logs/scoped_context.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Property 33: Context Label Normalization
// Validates: Requirements 8.8, 8.9
// ─────────────────────────────────────────────────────────────────────────────

/// Empty label ("") pushes the fixed placeholder "<unnamed>".
RC_GTEST_PROP(ContextLabelNormalization, EmptyLabelPushesPlaceholder, ()) {
    // Ensure a clean stack before the test.
    auto before = logs::detail::snapshot_context();

    {
        logs::Scoped_Context ctx("");
        auto snapshot = logs::detail::snapshot_context();

        // The new entry should be the placeholder.
        RC_ASSERT(snapshot.size() == before.size() + 1);
        RC_ASSERT(snapshot.back() == "<unnamed>");
    }

    // After destruction, stack is restored.
    auto after = logs::detail::snapshot_context();
    RC_ASSERT(after == before);
}

/// A label exceeding 256 characters is truncated to its first 256 characters.
RC_GTEST_PROP(ContextLabelNormalization, LongLabelTruncatedTo256, ()) {
    // Generate a string with length in [257, 512].
    const auto extra = *rc::gen::inRange(1, 257);  // 1..256 extra chars
    const std::size_t total_len = 256 + static_cast<std::size_t>(extra);

    // Build a string of the desired length using random printable characters.
    auto raw = *rc::gen::container<std::string>(total_len, rc::gen::inRange<char>(32, 127)  // printable ASCII
    );
    RC_PRE(raw.size() > 256);

    auto before = logs::detail::snapshot_context();

    {
        logs::Scoped_Context ctx(raw);
        auto snapshot = logs::detail::snapshot_context();

        RC_ASSERT(snapshot.size() == before.size() + 1);
        // The pushed label must be exactly the first 256 chars of raw.
        RC_ASSERT(snapshot.back().size() == 256);
        RC_ASSERT(snapshot.back() == raw.substr(0, 256));
    }

    auto after = logs::detail::snapshot_context();
    RC_ASSERT(after == before);
}

/// A label with exactly 256 characters is stored as-is (no truncation).
RC_GTEST_PROP(ContextLabelNormalization, Exactly256CharsStoredAsIs, ()) {
    // Generate exactly 256 random printable characters.
    auto label = *rc::gen::container<std::string>(std::size_t{256}, rc::gen::inRange<char>(32, 127));
    RC_PRE(label.size() == 256);

    auto before = logs::detail::snapshot_context();

    {
        logs::Scoped_Context ctx(label);
        auto snapshot = logs::detail::snapshot_context();

        RC_ASSERT(snapshot.size() == before.size() + 1);
        RC_ASSERT(snapshot.back() == label);
    }

    auto after = logs::detail::snapshot_context();
    RC_ASSERT(after == before);
}

/// A label shorter than 256 characters (and non-empty) is stored as-is.
RC_GTEST_PROP(ContextLabelNormalization, ShortLabelStoredAsIs, ()) {
    // Generate a non-empty string with length in [1, 255].
    const auto len = *rc::gen::inRange(1, 256);  // 1..255

    auto label = *rc::gen::container<std::string>(static_cast<std::size_t>(len), rc::gen::inRange<char>(32, 127));
    RC_PRE(!label.empty() && label.size() < 256);

    auto before = logs::detail::snapshot_context();

    {
        logs::Scoped_Context ctx(label);
        auto snapshot = logs::detail::snapshot_context();

        RC_ASSERT(snapshot.size() == before.size() + 1);
        RC_ASSERT(snapshot.back() == label);
    }

    auto after = logs::detail::snapshot_context();
    RC_ASSERT(after == before);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 32: Context Stack Per-Thread Independence
// Validates: Requirements 8.6, 11.7
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn multiple threads each pushing distinct labels; verify that each
/// thread's snapshot contains ONLY its own labels and never labels from another
/// thread. The main thread's stack is unaffected.
TEST(ContextStackPerThreadIndependence, ThreadsNeverShareLabels) {
    constexpr int NUM_THREADS = 8;
    constexpr int LABELS_PER_THREAD = 4;

    // Ensure main thread starts with a clean stack.
    auto main_before = logs::detail::snapshot_context();

    // Each thread stores its snapshot here for post-join verification.
    std::vector<std::vector<std::string>> thread_snapshots(NUM_THREADS);
    std::mutex snapshot_mutex;

    // Build per-thread label sets — each label contains the thread index so
    // labels are globally unique across threads.
    auto make_label = [](int thread_idx, int label_idx) -> std::string {
        return "thread_" + std::to_string(thread_idx) + "_label_" + std::to_string(label_idx);
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            // Push LABELS_PER_THREAD distinct labels on this thread.
            std::vector<std::unique_ptr<logs::Scoped_Context>> contexts;
            contexts.reserve(LABELS_PER_THREAD);

            for (int l = 0; l < LABELS_PER_THREAD; ++l) {
                contexts.push_back(std::make_unique<logs::Scoped_Context>(make_label(t, l)));
            }

            // Take a snapshot of this thread's context stack.
            auto snapshot = logs::detail::snapshot_context();

            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                thread_snapshots[t] = std::move(snapshot);
            }

            // Contexts destroyed here (reverse order) — pops all labels.
        });
    }

    // Join all threads.
    for (auto &th : threads) {
        th.join();
    }

    // Verify: each thread's snapshot contains exactly its own labels.
    for (int t = 0; t < NUM_THREADS; ++t) {
        const auto &snapshot = thread_snapshots[t];

        // Should have exactly LABELS_PER_THREAD entries.
        EXPECT_EQ(snapshot.size(), static_cast<std::size_t>(LABELS_PER_THREAD)) << "Thread " << t << " has wrong number of labels";

        // Each label in the snapshot must belong to this thread.
        for (int l = 0; l < LABELS_PER_THREAD; ++l) {
            EXPECT_EQ(snapshot[static_cast<std::size_t>(l)], make_label(t, l)) << "Thread " << t << " has incorrect label at position " << l;
        }

        // No label from any other thread should be present.
        for (const auto &label : snapshot) {
            // Extract thread index from label format "thread_X_label_Y".
            // If it doesn't start with "thread_<t>_", it's foreign.
            std::string own_prefix = "thread_" + std::to_string(t) + "_";
            EXPECT_TRUE(label.find(own_prefix) == 0) << "Thread " << t << " snapshot contains foreign label: " << label;
        }
    }

    // Verify: main thread's stack is unaffected by child thread activity.
    auto main_after = logs::detail::snapshot_context();
    EXPECT_EQ(main_after, main_before) << "Main thread context stack was modified by child threads";
}

/// Property-based variant: randomize the number of threads and labels per
/// thread, then verify per-thread isolation.
RC_GTEST_PROP(ContextStackPerThreadIndependence, RandomizedThreadIsolation, ()) {
    // Generate a random number of threads [2, 8] and labels per thread [1, 6].
    const auto num_threads = *rc::gen::inRange(2, 9);
    const auto labels_per_thread = *rc::gen::inRange(1, 7);

    auto main_before = logs::detail::snapshot_context();

    std::vector<std::vector<std::string>> thread_snapshots(static_cast<std::size_t>(num_threads));
    std::mutex snapshot_mutex;

    auto make_label = [](int thread_idx, int label_idx) -> std::string {
        return "t" + std::to_string(thread_idx) + "_l" + std::to_string(label_idx);
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads));

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<std::unique_ptr<logs::Scoped_Context>> contexts;
            contexts.reserve(static_cast<std::size_t>(labels_per_thread));

            for (int l = 0; l < labels_per_thread; ++l) {
                contexts.push_back(std::make_unique<logs::Scoped_Context>(make_label(t, l)));
            }

            auto snapshot = logs::detail::snapshot_context();

            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                thread_snapshots[static_cast<std::size_t>(t)] = std::move(snapshot);
            }
        });
    }

    for (auto &th : threads) {
        th.join();
    }

    // Verify each thread's snapshot contains only its own labels.
    for (int t = 0; t < num_threads; ++t) {
        const auto &snapshot = thread_snapshots[static_cast<std::size_t>(t)];

        RC_ASSERT(static_cast<int>(snapshot.size()) == labels_per_thread);

        for (int l = 0; l < labels_per_thread; ++l) {
            RC_ASSERT(snapshot[static_cast<std::size_t>(l)] == make_label(t, l));
        }

        // Verify no foreign labels appear.
        for (const auto &label : snapshot) {
            std::string own_prefix = "t" + std::to_string(t) + "_";
            RC_ASSERT(label.substr(0, own_prefix.size()) == own_prefix);
        }
    }

    // Main thread stack unaffected.
    auto main_after = logs::detail::snapshot_context();
    RC_ASSERT(main_after == main_before);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 31: Context Capture Ordered Outermost-to-Innermost
// Validates: Requirements 1.4, 8.4, 8.10
// ─────────────────────────────────────────────────────────────────────────────

/// With no active Scoped_Context, the context stack snapshot is empty.
RC_GTEST_PROP(ContextCaptureOrdering, EmptyWhenNoContextActive, ()) {
    // No Scoped_Context objects are active on this thread at this point.
    auto snapshot = logs::detail::snapshot_context();
    RC_ASSERT(snapshot.empty());
}

/// For a random sequence of N labels pushed via nested Scoped_Context objects,
/// the snapshot returns labels in outermost-to-innermost order (construction
/// order), matching the input sequence exactly.
RC_GTEST_PROP(ContextCaptureOrdering, NestedContextsOrderedOutermostToInnermost, ()) {
    // Generate between 1 and 10 non-empty labels (within the 256-char limit).
    const auto count = *rc::gen::inRange(1, 11);
    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(count));

    for (int i = 0; i < count; ++i) {
        // Generate a non-empty label of length [1, 64] (well within 256 limit).
        auto label = *rc::gen::container<std::string>(
            static_cast<std::size_t>(*rc::gen::inRange(1, 65)), rc::gen::inRange<char>(33, 127)  // printable non-space ASCII
        );
        RC_PRE(!label.empty());
        labels.push_back(std::move(label));
    }

    // Confirm stack is empty before we begin.
    RC_ASSERT(logs::detail::snapshot_context().empty());

    // Create nested Scoped_Context objects. The first label pushed becomes
    // the outermost; the last becomes the innermost. We use unique_ptr so
    // that we can control the nesting lifetime explicitly.
    std::vector<std::unique_ptr<logs::Scoped_Context>> contexts;
    contexts.reserve(labels.size());

    for (const auto &lbl : labels) {
        contexts.push_back(std::make_unique<logs::Scoped_Context>(lbl));
    }

    // Snapshot the context stack — should match labels in outermost-to-innermost
    // order, which is the same as the construction order (labels[0] .. labels[N-1]).
    auto snapshot = logs::detail::snapshot_context();

    RC_ASSERT(snapshot.size() == labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        RC_ASSERT(snapshot[i] == labels[i]);
    }

    // Destroy all contexts (reverse order via vector destruction).
    contexts.clear();

    // After all contexts destroyed, the stack must be empty again.
    RC_ASSERT(logs::detail::snapshot_context().empty());
}

}  // namespace
