/// @file severity.cpp
/// @brief Severity_Level to_string implementation.

#include "logs/severity.hpp"

namespace logs {

std::string_view to_string(Severity_Level level) noexcept {
    switch (level) {
        case Severity_Level::DEBUG:   return "DEBUG";
        case Severity_Level::INFO:    return "INFO";
        case Severity_Level::WARNING: return "WARNING";
        case Severity_Level::ERROR:   return "ERROR";
        case Severity_Level::FATAL:   return "FATAL";
    }
    return "UNKNOWN";
}

} // namespace logs
