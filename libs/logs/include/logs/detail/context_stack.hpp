#ifndef LOGS_DETAIL_CONTEXT_STACK_HPP
#define LOGS_DETAIL_CONTEXT_STACK_HPP

/// @file detail/context_stack.hpp
/// @brief Thread-local context stack accessors for Scoped_Context.

#include <string>
#include <vector>

namespace logs::detail {

/// Push a label onto the calling thread's context stack.
void push_context(std::string label);

/// Pop the most recently pushed label from the calling thread's context stack.
void pop_context() noexcept;

/// Snapshot the current thread's context stack (outermost to innermost).
[[nodiscard]] std::vector<std::string> snapshot_context();

} // namespace logs::detail

#endif // LOGS_DETAIL_CONTEXT_STACK_HPP
