// DAGR — detail/completion_token.hpp
// Opaque handle tracking an in-flight task dispatched by the Event_Loop.
#pragma once

#include <cstdint>

namespace dagr::detail {

/// Completion_Token: opaque handle returned when a TaskNode is dispatched.
/// Used by the Event_Loop to track in-flight tasks and signal downstream
/// dependents upon task completion.
struct Completion_Token {
    std::uint32_t node_id;                ///< Zero-based TaskNode index in the DAG
    std::uint64_t dispatch_timestamp_ns;  ///< Monotonic clock at dispatch (nanoseconds)
};

}  // namespace dagr::detail
