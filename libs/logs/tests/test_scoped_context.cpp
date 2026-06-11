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

#include <logs/scoped_context.hpp>
#include <logs/detail/context_stack.hpp>

#include <gtest/gtest.h>

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
    ASSERT_TRUE(pre_snapshot.empty())
        << "Pre-condition: context stack should be empty at test start";

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

    } catch (const std::exception& e) {
        // 5. Verify the context stack is restored to pre-context state.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size())
            << "Stack depth should be restored after exception unwind";
        EXPECT_EQ(post_snapshot, pre_snapshot)
            << "Stack labels should match pre-context state after unwind";
    }

    // Double-check outside the try-catch block as well.
    const auto final_snapshot = logs::detail::snapshot_context();
    EXPECT_TRUE(final_snapshot.empty())
        << "Context stack should be empty after exception-unwound Scoped_Context";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Nested Scoped_Contexts exception unwind
// Validates: Requirement 13.7
// ─────────────────────────────────────────────────────────────────────────────

TEST(ScopedContextExceptionUnwind, NestedContextsBothRestoredAfterException) {
    // 1. Snapshot the context stack before — should be empty.
    const auto pre_snapshot = logs::detail::snapshot_context();
    ASSERT_TRUE(pre_snapshot.empty())
        << "Pre-condition: context stack should be empty at test start";

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

    } catch (const std::exception& e) {
        // After catching, verify both contexts are popped.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size())
            << "Stack depth should be restored after nested exception unwind";
        EXPECT_EQ(post_snapshot, pre_snapshot)
            << "Stack labels should match pre-context state after nested unwind";
    }

    // Final verification outside try-catch.
    const auto final_snapshot = logs::detail::snapshot_context();
    EXPECT_TRUE(final_snapshot.empty())
        << "Context stack should be empty after nested exception-unwound contexts";
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

    } catch (const std::exception& e) {
        // Only "inner_context" should be popped; "base_context" remains.
        const auto post_snapshot = logs::detail::snapshot_context();
        EXPECT_EQ(post_snapshot.size(), pre_snapshot.size())
            << "Only the inner context should be unwound";
        EXPECT_EQ(post_snapshot, pre_snapshot)
            << "Pre-existing base context should be preserved after unwind";
    }

    // Verify base context still present.
    const auto final_snapshot = logs::detail::snapshot_context();
    ASSERT_EQ(final_snapshot.size(), 1u);
    EXPECT_EQ(final_snapshot[0], "base_context");
}

} // namespace
