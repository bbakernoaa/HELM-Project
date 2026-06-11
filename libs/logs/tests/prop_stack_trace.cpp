/// @file prop_stack_trace.cpp
/// @brief Property-based tests for format_stack_trace pure formatter.
///
/// Uses RapidCheck + Google Test to verify universal correctness properties
/// over arbitrary frame sequences.

#include <logs/stack_trace.hpp>

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// Generate a sanitized string (no newlines) suitable for frame fields.
rc::Gen<std::string> genSafeString() {
    return rc::gen::map(
        rc::gen::nonEmpty(rc::gen::string<std::string>()),
        [](std::string s) {
            std::replace(s.begin(), s.end(), '\n', '_');
            std::replace(s.begin(), s.end(), '\r', '_');
            return s;
        });
}

/// Generate a random Stack_Frame with optional fields.
rc::Gen<logs::Stack_Frame> genStackFrame() {
    return rc::gen::exec([]() {
        logs::Stack_Frame frame;
        // Randomly decide whether each field is present.
        if (*rc::gen::arbitrary<bool>()) {
            frame.function = *genSafeString();
        }
        if (*rc::gen::arbitrary<bool>()) {
            frame.file = *genSafeString();
        }
        if (*rc::gen::arbitrary<bool>()) {
            frame.line = *rc::gen::inRange(1, 100000);
        }
        return frame;
    });
}

/// Helper: count lines in a string (number of '\n' characters).
std::size_t countLines(const std::string& s) {
    return static_cast<std::size_t>(std::count(s.begin(), s.end(), '\n'));
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 27: Stack-Trace Frame Cap With Omission Indicator
// Validates: Requirements 7.8
// ─────────────────────────────────────────────────────────────────────────────

/// For any sequence with >256 frames, verify exactly 256 frame lines are
/// rendered followed by exactly one omission line, totalling 257 lines.
RC_GTEST_PROP(StackTraceFrameCap,
              OutputContainsExactly256FrameLinesAndOneOmissionLine,
              ()) {
    // Generate a frame count in [257, 512].
    const auto frameCount = *rc::gen::inRange<std::size_t>(257, 513);

    // Build the frame sequence.
    std::vector<logs::Stack_Frame> frames;
    frames.reserve(frameCount);
    for (std::size_t i = 0; i < frameCount; ++i) {
        frames.push_back(*genStackFrame());
    }

    const std::string output = logs::format_stack_trace(frames);

    // Total lines must be exactly 257: 256 frame lines + 1 omission line.
    const auto totalLines = countLines(output);
    RC_ASSERT(totalLines == 257u);
}

/// Verify the last line matches the omission pattern with correct count.
RC_GTEST_PROP(StackTraceFrameCap,
              OmissionLineContainsCorrectRemainingCount,
              ()) {
    const auto frameCount = *rc::gen::inRange<std::size_t>(257, 513);

    std::vector<logs::Stack_Frame> frames;
    frames.reserve(frameCount);
    for (std::size_t i = 0; i < frameCount; ++i) {
        frames.push_back(*genStackFrame());
    }

    const std::string output = logs::format_stack_trace(frames);

    // Extract the last line.
    auto lastNewline = output.rfind('\n', output.size() - 2);
    std::string lastLine;
    if (lastNewline == std::string::npos) {
        lastLine = output.substr(0, output.size() - 1); // remove trailing \n
    } else {
        lastLine = output.substr(lastNewline + 1);
        // Remove trailing newline if present.
        if (!lastLine.empty() && lastLine.back() == '\n') {
            lastLine.pop_back();
        }
    }

    // The omission line must match: "[... N additional frames omitted]"
    // where N = frameCount - 256.
    const std::size_t omitted = frameCount - logs::MAX_TRACE_FRAMES;
    const std::string expected = "[... " + std::to_string(omitted) + " additional frames omitted]";
    RC_ASSERT(lastLine == expected);
}

/// Verify the first 256 frame lines reference indices #0 through #255.
RC_GTEST_PROP(StackTraceFrameCap,
              First256FrameIndicesPresent,
              ()) {
    const auto frameCount = *rc::gen::inRange<std::size_t>(257, 513);

    std::vector<logs::Stack_Frame> frames;
    frames.reserve(frameCount);
    for (std::size_t i = 0; i < frameCount; ++i) {
        frames.push_back(*genStackFrame());
    }

    const std::string output = logs::format_stack_trace(frames);

    // Split into lines.
    std::vector<std::string> lines;
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(line);
    }

    // Must have exactly 257 lines total.
    RC_ASSERT(lines.size() == 257u);

    // First 256 lines must start with #0, #1, ..., #255.
    for (std::size_t i = 0; i < 256; ++i) {
        const std::string prefix = "#" + std::to_string(i) + "  ";
        RC_ASSERT(lines[i].substr(0, prefix.size()) == prefix);
    }
}

} // namespace
