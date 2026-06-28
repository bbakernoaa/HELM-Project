#ifndef LOGS_SEVERITY_HPP
#define LOGS_SEVERITY_HPP

/// @file severity.hpp
/// @brief Severity_Level ordered enumeration and to_string mapping.

#include <string_view>

namespace logs {

/// Ascending order of severity. The fixed underlying values establish a
/// total ordering such that DEBUG < INFO < WARNING < ERROR < FATAL.
enum class Severity_Level : int { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4 };

/// Map a Severity_Level to its fixed, human-readable label.
/// Returns "DEBUG", "INFO", "WARNING", "ERROR", or "FATAL".
[[nodiscard]] std::string_view to_string(Severity_Level level) noexcept;

}  // namespace logs

#endif  // LOGS_SEVERITY_HPP
