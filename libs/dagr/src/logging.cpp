// DAGR — logging.cpp
// Implementation of the shared LOGS logger accessor and its public
// configuration entry point.
#include "dagr/logging.hpp"

#include <atomic>
#include <iostream>

#include "shared_logger.hpp"

namespace dagr::detail {

logs::Logger &shared_logger() noexcept {
    static logs::Logger instance;
    return instance;
}

}  // namespace dagr::detail

namespace dagr {

namespace {

logs::Severity_Level to_severity(Log_Level level) noexcept {
    switch (level) {
        case Log_Level::debug:
            return logs::Severity_Level::DEBUG;
        case Log_Level::info:
            return logs::Severity_Level::INFO;
        case Log_Level::warning:
            return logs::Severity_Level::WARNING;
        case Log_Level::error:
            return logs::Severity_Level::ERROR;
        case Log_Level::silent:
            // FATAL is never suppressed; using it as the threshold makes every
            // lower severity fall below the bar, i.e. "fatal only".
            return logs::Severity_Level::FATAL;
    }
    return logs::Severity_Level::INFO;
}

}  // namespace

void configure_logging(MPI_Comm comm, Log_Level level) noexcept {
    logs::Logger &lg = detail::shared_logger();

    // Install the stdout sink exactly once. A Sink holds a reference to the
    // stream (not its buffer), so it honours any redirection/tee applied to
    // std::cout by the host application at write time.
    static std::atomic<bool> sink_installed{false};
    bool expected = false;
    if (sink_installed.compare_exchange_strong(expected, true)) {
        lg.add_sink(logs::Sink(std::cout));
    }

    lg.configure_communicator(comm);
    lg.set_threshold(to_severity(level));
}

}  // namespace dagr
