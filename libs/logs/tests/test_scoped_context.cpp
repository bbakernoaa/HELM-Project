/// @file test_scoped_context.cpp
/// @brief Google Test verification for Scoped_Context exception-unwind behavior.
///
/// Validates: Requirement 13.7
///
/// Tests verify that:
///   - When an exception is thrown while a Scoped_Context is active, the
///     destructor correctly pops the label during stack unwinding.
///   - After the exception is caught, the context stack depth and labels
///     are restored to their pre-Scoped_Context state.
///   - Nested Scoped_Contexts are both unwound correctly when an exception
///     is thrown from the innermost scope.

#include <gtest/gtest.h>

#include <logs/detail/context_stack.hpp>
#include <logs/scoped_context.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Test: Single Scoped_Context exception unwind
// Validates: Requirement 13.7
// ─────────────────────────────────────────────────────────────────────────────

TEST(ScopedContextExceptionUnwind, SingleContextRestoredAfterException) {
    // 1. Snapshot the context stack before — should be empty.
    const auto pre_snapshot = logs::detail::snapshot_context();
    ASSERT_TRUE(pre_snapshot.empty()) << "Pre-condition: context stack should be empty at test start";

    // 2-5. Enter try block, create Scoped_Context, throw, catch.
    try {
        logs::Scoped_Context ctx("test_label");

        // 3. Verify the label is now on the stack.
        const auto active_snapshot = logs::detail::snapshot_context();
        ASSERT_EQ(active_snapshot.size(), 1u);
        EXPECT_EQ(active_snapshot[0], "test_label");

        // 4. Throw a std::runtime_error — the Scoped_Context destructor
        //    should pop "test_label" during stack unwinding.
        throw std::runtime_error("intentional test exception");

    } catch (const std::exception &e) {
        // 5. Verify the context stack is restored to pre-context state.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size()) << "Stack depth should be restored after exception unwind";
        EXPECT_EQ(post_snapshot, pre_snapshot) << "Stack labels should match pre-context state after unwind";
    }

    // Double-check outside the try-catch block as well.
    const auto final_snapshot = logs::detail::snapshot_context();
    EXPECT_TRUE(final_snapshot.empty()) << "Context stack should be empty after exception-unwound Scoped_Context";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Nested Scoped_Contexts exception unwind
// Validates: Requirement 13.7
// ─────────────────────────────────────────────────────────────────────────────

TEST(ScopedContextExceptionUnwind, NestedContextsBothRestoredAfterException) {
    // 1. Snapshot the context stack before — should be empty.
    const auto pre_snapshot = logs::detail::snapshot_context();
    ASSERT_TRUE(pre_snapshot.empty()) << "Pre-condition: context stack should be empty at test start";

    // 2. Push context A, then B; throw from B's scope.
    try {
        logs::Scoped_Context ctx_a("context_A");

        // Verify A is on the stack.
        {
            const auto snap_a = logs::detail::snapshot_context();
            ASSERT_EQ(snap_a.size(), 1u);
            EXPECT_EQ(snap_a[0], "context_A");
        }

        {
            logs::Scoped_Context ctx_b("context_B");

            // Verify both A and B are on the stack (outermost to innermost).
            const auto snap_ab = logs::detail::snapshot_context();
            ASSERT_EQ(snap_ab.size(), 2u);
            EXPECT_EQ(snap_ab[0], "context_A");
            EXPECT_EQ(snap_ab[1], "context_B");

            // Throw from B's scope — both ctx_b and ctx_a destructors
            // should fire during stack unwinding.
            throw std::runtime_error("nested exception test");
        }

    } catch (const std::exception &e) {
        // After catching, verify both contexts are popped.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size()) << "Stack depth should be restored after nested exception unwind";
        EXPECT_EQ(post_snapshot, pre_snapshot) << "Stack labels should match pre-context state after nested unwind";
    }

    // Final verification outside try-catch.
    const auto final_snapshot = logs::detail::snapshot_context();
    EXPECT_TRUE(final_snapshot.empty()) << "Context stack should be empty after nested exception-unwound contexts";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Exception unwind with pre-existing context on stack
// Validates: Requirement 13.7
//
// Verifies that the unwind only pops the labels pushed by the scoped contexts
// within the try block, leaving pre-existing labels intact.
// ─────────────────────────────────────────────────────────────────────────────

TEST(ScopedContextExceptionUnwind, PreExistingContextPreservedAfterUnwind) {
    // Push a base context that should survive the exception.
    logs::Scoped_Context base_ctx("base_context");

    const auto pre_snapshot = logs::detail::snapshot_context();
    ASSERT_EQ(pre_snapshot.size(), 1u);
    EXPECT_EQ(pre_snapshot[0], "base_context");

    try {
        logs::Scoped_Context inner_ctx("inner_context");

        const auto active_snapshot = logs::detail::snapshot_context();
        ASSERT_EQ(active_snapshot.size(), 2u);
        EXPECT_EQ(active_snapshot[0], "base_context");
        EXPECT_EQ(active_snapshot[1], "inner_context");

        throw std::runtime_error("exception with base context present");

    } catch (const std::exception &e) {
        // Only "inner_context" should be popped; "base_context" remains.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size()) << "Only the inner context should be unwound";
        EXPECT_EQ(post_snapshot, pre_snapshot) << "Pre-existing base context should be preserved after unwind";
    }

    // Verify base context still present.
    const auto final_snapshot = logs::detail::snapshot_context();
    ASSERT_EQ(final_snapshot.size(), 1u);
    EXPECT_EQ(final_snapshot[0], "base_context");
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Property 31 — Context Capture Ordered Outermost-to-Innermost
//
// Task 18.3: Verify active labels carried in outermost-to-innermost order;
//            empty when no context active.
// **Validates: Requirements 1.4, 8.4, 8.10**
// ═══════════════════════════════════════════════════════════════════════════════

#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <logs/logger.hpp>
#include <regex>
#include <string>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

/// Helper: generate a non-empty printable ASCII string for context capture tests.
static rc::Gen<std::string> genPrintableMessage() {
    return rc::gen::map(rc::gen::nonEmpty(rc::gen::container<std::string>(rc::gen::inRange(33, 127))), [](std::string s) { return s; });
}

/// Helper: Extract context bracket content from a formatted log line.
/// Format is: [RANK:XXXX] [SEVERITY] [label1 > label2 > ...] message\n
/// Returns empty vector if no context bracket is present.
std::vector<std::string> extract_context_labels(const std::string &formatted) {
    // Look for the third bracket group: [label1 > label2 > ...]
    // The format is: [RANK:...] [SEV] [ctx1 > ctx2] message
    // We search for a bracket that is NOT rank or severity.
    std::vector<std::string> labels;

    // Find all [...] sections
    std::size_t pos = 0;
    int bracket_count = 0;
    while (pos < formatted.size()) {
        auto open = formatted.find('[', pos);
        if (open == std::string::npos) break;
        auto close = formatted.find(']', open);
        if (close == std::string::npos) break;

        ++bracket_count;
        // The third bracket (if present) is the context bracket.
        if (bracket_count == 3) {
            std::string content = formatted.substr(open + 1, close - open - 1);
            // Split by " > "
            std::size_t split_pos = 0;
            while (split_pos < content.size()) {
                auto sep = content.find(" > ", split_pos);
                if (sep == std::string::npos) {
                    labels.push_back(content.substr(split_pos));
                    break;
                }
                labels.push_back(content.substr(split_pos, sep - split_pos));
                split_pos = sep + 3;  // skip " > "
            }
            return labels;
        }
        pos = close + 1;
    }

    return labels;  // empty = no context bracket
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 31: Context Capture Ordered Outermost-to-Innermost
// **Validates: Requirements 1.4, 8.4, 8.10**
// ─────────────────────────────────────────────────────────────────────────────

/// For any sequence of nested Scoped_Context objects with random labels,
/// verify the formatted output carries labels in outermost-to-innermost order.
RC_GTEST_PROP(ContextCaptureProperty, NestedLabelsAppearOutermostToInnermost, ()) {
    // Generate 1..8 non-empty labels (avoid labels containing "]" or " > "
    // which would confuse parsing).
    const auto label_count = *rc::gen::inRange(1, 9);
    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(label_count));

    for (int i = 0; i < label_count; ++i) {
        auto label = *rc::gen::suchThat(rc::gen::nonEmpty<std::string>(), [](const std::string &s) {
            // Avoid characters that would break bracket parsing.
            return s.find(']') == std::string::npos && s.find('[') == std::string::npos && s.find(" > ") == std::string::npos &&
                   s.find('\n') == std::string::npos && s.find('\0') == std::string::npos && s.size() <= 64;  // keep labels short for readability
        });
        labels.push_back(std::move(label));
    }

    // Create Logger with an in-memory sink.
    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Create nested Scoped_Context objects (outermost first).
    // We use a vector of unique_ptr to manage their lifetime in order.
    std::vector<std::unique_ptr<logs::Scoped_Context>> contexts;
    contexts.reserve(labels.size());
    for (const auto &lbl : labels) {
        contexts.push_back(std::make_unique<logs::Scoped_Context>(lbl));
    }

    // Emit a record — it should capture all active context labels.
    logger.log(logs::Severity_Level::INFO, "context capture test");

    RC_ASSERT(mem_sink.count() == 1u);

    // Parse the formatted output and extract context labels.
    const auto all_entries = mem_sink.entries();
    const std::string &output = all_entries[0];
    std::vector<std::string> captured = extract_context_labels(output);

    // Verify: captured labels match in outermost-to-innermost order.
    RC_ASSERT(captured.size() == labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        RC_ASSERT(captured[i] == labels[i]);
    }

    // Clean up contexts in reverse order (innermost first) to maintain RAII.
    while (!contexts.empty()) {
        contexts.pop_back();
    }
}

/// With no active labels, verify no context bracket appears in the output.
RC_GTEST_PROP(ContextCaptureProperty, NoActiveContextProducesNoContextBracket, ()) {
    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());

    // Generate a printable message without bracket characters to avoid
    // confusing the bracket-based parsing logic.
    const auto msg =
        *rc::gen::nonEmpty(rc::gen::container<std::string>(rc::gen::suchThat(rc::gen::inRange(33, 127), [](int c) { return c != '[' && c != ']'; })));

    // Emit with no Scoped_Context active.
    logger.log(logs::Severity_Level::INFO, msg);

    RC_ASSERT(mem_sink.count() == 1u);

    const auto all_entries = mem_sink.entries();
    const std::string &output = all_entries[0];
    std::vector<std::string> captured = extract_context_labels(output);

    // With no context, the third bracket should not exist — captured is empty.
    RC_ASSERT(captured.empty());

    // Verify the format has exactly 2 bracket sections: [RANK:...] [SEV]
    // (valid since we excluded brackets from the message itself)
    int bracket_count = 0;
    std::size_t pos = 0;
    while (pos < output.size()) {
        auto open = output.find('[', pos);
        if (open == std::string::npos) break;
        auto close = output.find(']', open);
        if (close == std::string::npos) break;
        ++bracket_count;
        pos = close + 1;
    }
    RC_ASSERT(bracket_count == 2);
}

}  // namespace
