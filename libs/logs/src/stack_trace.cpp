/// @file stack_trace.cpp
/// @brief format_stack_trace implementation — pure formatting only.
///
/// This file reads/opens/parses NOTHING. It transforms an already-captured
/// frame sequence into deterministic multi-line text.

#include "logs/stack_trace.hpp"

#include <sstream>

namespace logs {

std::string format_stack_trace(std::span<const Stack_Frame> frames) {
    // Empty sequence → fixed single-line indication (Requirement 7.5).
    if (frames.empty()) {
        return "[no frames available]\n";
    }

    std::ostringstream oss;
    const auto limit = std::min(frames.size(), MAX_TRACE_FRAMES);

    for (std::size_t i = 0; i < limit; ++i) {
        const auto& frame = frames[i];
        // Fixed field order: index, function, file, line (Requirements 7.1, 7.3).
        // Missing fields → fixed placeholder tokens (Requirement 7.4).
        oss << '#' << i << "  "
            << (frame.function ? *frame.function : "<unknown>")
            << " at "
            << (frame.file ? *frame.file : "<unknown>")
            << ':'
            << (frame.line ? std::to_string(*frame.line) : "?")
            << '\n';
    }

    // >256 frames → render first 256 + omission indicator (Requirement 7.8).
    if (frames.size() > MAX_TRACE_FRAMES) {
        oss << "[... " << (frames.size() - MAX_TRACE_FRAMES)
            << " additional frames omitted]\n";
    }

    return oss.str();
}

} // namespace logs
