#ifndef LOGS_SCOPED_CONTEXT_HPP
#define LOGS_SCOPED_CONTEXT_HPP

/// @file scoped_context.hpp
/// @brief Scoped_Context RAII thread-local trace-context label.

#include <cstddef>
#include <string_view>

namespace logs {

/// RAII object that pushes a named trace-context label onto the calling
/// thread's context stack at construction and pops it at destruction.
class Scoped_Context {
   public:
    /// Push `label` onto the calling thread's context stack.
    explicit Scoped_Context(std::string_view label);

    /// Pop the single label this object pushed.
    ~Scoped_Context() noexcept;

    Scoped_Context(const Scoped_Context &) = delete;
    Scoped_Context &operator=(const Scoped_Context &) = delete;
    Scoped_Context(Scoped_Context &&) = delete;
    Scoped_Context &operator=(Scoped_Context &&) = delete;

    /// Maximum stored label length in characters.
    static constexpr std::size_t MAX_LABEL_LENGTH = 256;
};

}  // namespace logs

#endif  // LOGS_SCOPED_CONTEXT_HPP
