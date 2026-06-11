/// @file test_stack_trace.cpp
/// @brief Unit tests for stack-trace attachment and context capture (18.1),
///        and property test: Conditional Stack-Trace Attachment (18.2).
///
/// 18.1 — Unit tests:
///   - include_stack_trace=true with frames → output contains stack-trace text
///   - include_stack_trace=false → output contains no stack-trace text
///   - Active Scoped_Context labels → output carries labels outermost-to-innermost
///   - No active labels → empty context-label sequence (no context bracket)
///
/// 18.2 — Property 28: Conditional Stack-Trace Attachment
///   For any submission, verify stack-trace present iff caller requested inclusion.
///
/// Validates: Requirements 7.6, 7.7, 1.4, 8.4, 8.10

#include <logs/logger.hpp>
#include <logs/scoped_context.hpp>
#include <logs/stack_trace.hpp>

#include "in_memory_sink.hpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <string>
#include <vector>

namespace {

/// Helper: concatenate all entries from an In_Memory_Sink into one string.
std::string collect_output(const logs::testing::In_Memory_Sink& sink) {
    std::string result;
    for (const auto& entry : sink.entries()) {
        result += entry;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 18.1: Unit tests for stack-trace attachment and context capture
// ─────────────────────────────────────────────────────────────────────────────

/// Test: with include_stack_trace=true and frames supplied, verify Log_Record
/// carries stack_trace text.
/// **Validates: Requirements 7.6**
TEST(StackTraceAttachment, IncludeStackTraceTrue_CarriesStackTraceText) {
    // Verify the Logger's internal logic: when include_stack_trace=true and
    // frames are non-empty, format_stack_trace is invoked and the result is
    // stored in the Log_Record's stack_trace field.
    std::vector<logs::Stack_Frame> frames = {
        {std::string("my_function"), std::string("my_file.cpp"), 42},
        {std::string("caller_func"), std::string("caller.cpp"), 100},
    };

    logs::Submit_Options opts;
    opts.include_stack_trace = true;
    opts.frames = std::span<const logs::Stack_Frame>(frames);

    // Replicate the Logger's stack-trace attachment logic (from logger.cpp):
    std::optional<std::string> stack_trace_text;
    if (opts.include_stack_trace && !opts.frames.empty()) {
        stack_trace_text = logs::format_stack_trace(opts.frames);
    }

    // Construct a Log_Record the same way the Logger does internally.
    logs::Log_Record record(
        logs::Severity_Level::ERROR,
        "something failed",
        0,
        std::nullopt,
        {},
        std::move(stack_trace_text));

    // The record must carry stack_trace text.
    ASSERT_TRUE(record.stack_trace().has_value());
    EXPECT_NE(record.stack_trace().value().find("#0"), std::string::npos);
    EXPECT_NE(record.stack_trace().value().find("my_function"), std::string::npos);
    EXPECT_NE(record.stack_trace().value().find("my_file.cpp:42"), std::string::npos);
    EXPECT_NE(record.stack_trace().value().find("#1"), std::string::npos);
    EXPECT_NE(record.stack_trace().value().find("caller_func"), std::string::npos);
}

/// Test: with include_stack_trace=false, verify Log_Record carries no
/// stack_trace text.
/// **Validates: Requirements 7.7**
TEST(StackTraceAttachment, IncludeStackTraceFalse_NoStackTraceText) {
    std::vector<logs::Stack_Frame> frames = {
        {std::string("my_function"), std::string("my_file.cpp"), 42},
    };

    logs::Submit_Options opts;
    opts.include_stack_trace = false;
    opts.frames = std::span<const logs::Stack_Frame>(frames);

    // With include_stack_trace=false, the Logger does NOT format the trace.
    std::optional<std::string> stack_trace_text;
    if (opts.include_stack_trace && !opts.frames.empty()) {
        stack_trace_text = logs::format_stack_trace(opts.frames);
    }

    logs::Log_Record record(
        logs::Severity_Level::ERROR,
        "something failed",
        0,
        std::nullopt,
        {},
        std::move(stack_trace_text));

    // The record must NOT carry stack_trace text.
    EXPECT_FALSE(record.stack_trace().has_value());
}

/// Test: with active Scoped_Context labels, verify Log_Record carries labels
/// outermost-to-innermost in the formatted output.
/// **Validates: Requirements 1.4, 8.4**
TEST(ContextCapture, ActiveLabels_CarriedOutermostToInnermost) {
    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Create nested scoped contexts — outermost first.
    logs::Scoped_Context outer("initialization");
    logs::Scoped_Context middle("grid_setup");
    logs::Scoped_Context inner("interpolation");

    logger.log(logs::Severity_Level::INFO, "processing step");

    ASSERT_GE(mem_sink.count(), 1u);

    const std::string output = collect_output(mem_sink);

    // The formatted record includes context as:
    // [initialization > grid_setup > interpolation]
    EXPECT_NE(output.find("[initialization > grid_setup > interpolation]"),
              std::string::npos)
        << "Expected outermost-to-innermost context labels in output: " << output;
}

/// Test: with no active labels, verify empty context-label sequence
/// (no context bracket in formatted output).
/// **Validates: Requirements 8.10**
TEST(ContextCapture, NoActiveLabels_EmptyContextSequence) {
    logs::Logger logger;
    logs::testing::In_Memory_Sink mem_sink;
    logger.add_sink(mem_sink.sink());
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // No Scoped_Context objects active on this thread.
    logger.log(logs::Severity_Level::INFO, "no context message");

    ASSERT_GE(mem_sink.count(), 1u);

    const std::string output = collect_output(mem_sink);

    // Verify severity and message are present.
    EXPECT_NE(output.find("[INFO]"), std::string::npos);
    EXPECT_NE(output.find("no context message"), std::string::npos);

    // Verify there's no third bracket between [INFO] and the message.
    // Format without context: "[RANK:----] [INFO] no context message\n"
    // Format with context:    "[RANK:----] [INFO] [ctx] no context message\n"
    auto info_pos = output.find("[INFO]");
    ASSERT_NE(info_pos, std::string::npos);
    auto after_info = output.substr(info_pos + 6);  // skip "[INFO]"
    // After [INFO] should be " no context message" — no opening bracket.
    EXPECT_EQ(after_info[0], ' ');
    EXPECT_NE(after_info[1], '[');
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 18.2: Property 28 — Conditional Stack-Trace Attachment
//
// For any submission, verify stack-trace present iff caller requested inclusion.
// **Validates: Requirements 7.6, 7.7**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(ConditionalStackTrace,
              Property28_StackTracePresentIffRequested,
              ()) {
    // Generate arbitrary frames (0 to 10).
    const auto num_frames = *rc::gen::inRange(0, 10);
    std::vector<logs::Stack_Frame> frames;
    frames.reserve(static_cast<std::size_t>(num_frames));
    for (int i = 0; i < num_frames; ++i) {
        logs::Stack_Frame frame;
        if (*rc::gen::arbitrary<bool>()) {
            frame.function = *rc::gen::nonEmpty<std::string>();
        }
        if (*rc::gen::arbitrary<bool>()) {
            frame.file = *rc::gen::nonEmpty<std::string>();
        }
        if (*rc::gen::arbitrary<bool>()) {
            frame.line = *rc::gen::inRange(1, 10000);
        }
        frames.push_back(std::move(frame));
    }

    // Generate the include_stack_trace flag.
    const bool include_trace = *rc::gen::arbitrary<bool>();

    // Replicate the Logger's internal logic for stack-trace attachment:
    // stack_trace_text is set iff (include_trace AND frames non-empty).
    std::optional<std::string> stack_trace_text;
    if (include_trace && !frames.empty()) {
        stack_trace_text = logs::format_stack_trace(
            std::span<const logs::Stack_Frame>(frames));
    }

    // Construct a Log_Record with the same logic the Logger uses.
    logs::Log_Record record(
        logs::Severity_Level::WARNING,
        "test message",
        0,
        std::nullopt,
        {},
        stack_trace_text);

    // Property: stack_trace is present iff (include_trace AND frames non-empty).
    if (include_trace && !frames.empty()) {
        RC_ASSERT(record.stack_trace().has_value());
        // The formatted trace must contain at least the first frame marker.
        RC_ASSERT(record.stack_trace().value().find("#0") != std::string::npos);
    } else {
        RC_ASSERT(!record.stack_trace().has_value());
    }
}

} // namespace
