#ifndef LOGS_LOGS_HPP
#define LOGS_LOGS_HPP

/// @file logs.hpp
/// @brief Umbrella header for the LOGS (Lightweight Operational Global Status) library.
///
/// Including this header provides access to all public LOGS types and functions.

#include "logs/log_record.hpp"
#include "logs/logger.hpp"
#include "logs/scoped_context.hpp"
#include "logs/severity.hpp"
#include "logs/sink.hpp"
#include "logs/source_location.hpp"
#include "logs/stack_trace.hpp"

#endif  // LOGS_LOGS_HPP
