// SPDX-License-Identifier: Apache-2.0
// SPAN — Shared Pointer & Array Network
// Copyright (c) HELM Project Contributors

#ifndef SPAN_SPAN_HPP
#define SPAN_SPAN_HPP

/// @file span/span.hpp
/// @brief Unified public header for the SPAN Tier 2 micro-library.
///
/// Including this single header provides the complete SPAN template interface:
///   - span::CoherencyState — host/device coherency state enum
///   - span::MemorySpaceToken — runtime memory space selector
///   - span::FieldView<T, Rank> — non-owning dual-pointer view with coherency
///   - span::TripleBuffer<T> — lock-free I/O isolation for AMIO
///
/// No link dependencies beyond Kokkos and the C++ standard library.
/// For the Legacy C-API, link against libspan_c_api and include span_constants.h.
///
/// @note SPAN is a Tier 2 library. It depends only on Kokkos and the C++20
///       standard library. Tier 1 libraries (AXIS, BLEND) MAY consume SPAN
///       types but SPAN SHALL NOT include any Tier 1 or Tier 3 header.

#include "span/coherency_state.hpp"
#include "span/field_view.hpp"
#include "span/triple_buffer.hpp"

#endif  // SPAN_SPAN_HPP
