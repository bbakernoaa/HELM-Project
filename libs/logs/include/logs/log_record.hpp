#ifndef LOGS_LOG_RECORD_HPP
#define LOGS_LOG_RECORD_HPP

/// @file log_record.hpp
/// @brief Log_Record immutable value object.

#include "logs/severity.hpp"
#include "logs/source_location.hpp"

#include <optional>
#include <string>
#include <vector>

namespace logs {

/// Immutable log value object. Once constructed, no field can be mutated.
class Log_Record {
public:
    /// Construct an immutable record.
    Log_Record(Severity_Level severity,
               std::string message,
               int mpi_rank,
               std::optional<Source_Location> location,
               std::vector<std::string> context_labels,
               std::optional<std::string> stack_trace_text);

    Log_Record(const Log_Record&) = default;
    Log_Record(Log_Record&&) noexcept = default;
    Log_Record& operator=(const Log_Record&) = default;
    Log_Record& operator=(Log_Record&&) noexcept = default;

    [[nodiscard]] Severity_Level severity() const noexcept;
    [[nodiscard]] const std::string& message() const noexcept;
    [[nodiscard]] int rank() const noexcept;
    [[nodiscard]] const std::optional<Source_Location>& location() const noexcept;
    [[nodiscard]] const std::vector<std::string>& context_labels() const noexcept;
    [[nodiscard]] const std::optional<std::string>& stack_trace() const noexcept;

    /// True when this record was submitted at Severity_Level::FATAL.
    [[nodiscard]] bool triggers_abort() const noexcept;

private:
    Severity_Level severity_;
    std::string message_;
    int rank_;
    std::optional<Source_Location> location_;
    std::vector<std::string> context_labels_;
    std::optional<std::string> stack_trace_;
};

} // namespace logs

#endif // LOGS_LOG_RECORD_HPP
