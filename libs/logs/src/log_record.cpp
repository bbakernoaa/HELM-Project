/// @file log_record.cpp
/// @brief Log_Record immutable value object implementation.

#include "logs/log_record.hpp"

namespace logs {

Log_Record::Log_Record(Severity_Level severity, std::string message, int mpi_rank, std::optional<Source_Location> location,
                       std::vector<std::string> context_labels, std::optional<std::string> stack_trace_text)
    : severity_{severity},
      message_{std::move(message)},
      rank_{mpi_rank},
      location_{std::move(location)},
      context_labels_{std::move(context_labels)},
      stack_trace_{std::move(stack_trace_text)} {}

Severity_Level Log_Record::severity() const noexcept {
    return severity_;
}

const std::string &Log_Record::message() const noexcept {
    return message_;
}

int Log_Record::rank() const noexcept {
    return rank_;
}

const std::optional<Source_Location> &Log_Record::location() const noexcept {
    return location_;
}

const std::vector<std::string> &Log_Record::context_labels() const noexcept {
    return context_labels_;
}

const std::optional<std::string> &Log_Record::stack_trace() const noexcept {
    return stack_trace_;
}

bool Log_Record::triggers_abort() const noexcept {
    return severity_ == Severity_Level::FATAL;
}

}  // namespace logs
