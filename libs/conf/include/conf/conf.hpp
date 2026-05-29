#ifndef CONF_CONF_HPP
#define CONF_CONF_HPP

/// @file conf.hpp
/// @brief Umbrella header for the CONF (Configuration Parser) library.
///
/// Including this single header provides access to the complete public API of
/// the CONF Tier-1 micro-library: the stable error taxonomy and exception type
/// (conf::Error_Code, conf::Conf_Error), the RAII configuration owner with its
/// typed dotted-path accessors (conf::Config), and the non-owning resolved-node
/// view with its kind enumeration (conf::Value, conf::Node_Kind).
///
/// @note CONF is a Tier-1 component of the HELM ecosystem. Its public headers
/// include only the C++ standard library and CONF's own public headers; they
/// expose no yaml-cpp type and pull in no other HELM header (DAGR, SPAN, AXIS,
/// HALO, TICK, LOGS, AMIO). The yaml-cpp backend remains an invisible private
/// implementation detail, upholding the No-Circular-Dependency law.

#include "conf/error.hpp"
#include "conf/value.hpp"
#include "conf/config.hpp"

#endif // CONF_CONF_HPP
