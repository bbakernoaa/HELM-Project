/// @file scoped_context.cpp
/// @brief Scoped_Context push/pop implementation.

#include "logs/scoped_context.hpp"
#include "logs/detail/context_stack.hpp"

#include <string>

namespace logs {

Scoped_Context::Scoped_Context(std::string_view label) {
    std::string effective_label;
    if (label.empty()) {
        effective_label = "<unnamed>";
    } else if (label.size() > MAX_LABEL_LENGTH) {
        effective_label = std::string(label.substr(0, MAX_LABEL_LENGTH));
    } else {
        effective_label = std::string(label);
    }
    detail::push_context(std::move(effective_label));
}

Scoped_Context::~Scoped_Context() noexcept {
    detail::pop_context();
}

} // namespace logs
