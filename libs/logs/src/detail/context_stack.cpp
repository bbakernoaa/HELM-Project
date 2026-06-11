/// @file detail/context_stack.cpp
/// @brief Thread-local context stack implementation.

#include "logs/detail/context_stack.hpp"

namespace logs::detail {

namespace {
thread_local std::vector<std::string> tl_context_stack;
} // anonymous namespace

void push_context(std::string label) {
    tl_context_stack.push_back(std::move(label));
}

void pop_context() noexcept {
    if (!tl_context_stack.empty()) {
        tl_context_stack.pop_back();
    }
}

std::vector<std::string> snapshot_context() {
    return tl_context_stack;
}

} // namespace logs::detail
