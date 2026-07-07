// DAGR — shared_logger.hpp (internal)
// Provides the single process-wide LOGS Logger shared by all DAGR translation
// units (graph_orchestrator, event_loop, parse_pipeline, ...). Keeping one
// instance means a single configure_logging() call (communicator + threshold +
// sink) applies to every DAGR diagnostic, and rank stamping works consistently.
//
// This header is PRIVATE to the DAGR build — it exposes logs:: types and must
// not be installed or included from public DAGR headers (Req 11.3/11.5).
#pragma once

#include "logs/logs.hpp"

namespace dagr::detail {

/// The one LOGS Logger shared by all DAGR translation units.
logs::Logger &shared_logger() noexcept;

}  // namespace dagr::detail
