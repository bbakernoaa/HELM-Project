// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file halo_pattern.cpp
/// @brief HaloPattern implementation.
///
/// HaloPattern is a plain-data struct with all members defined inline in the
/// header. This compilation unit ensures the header compiles cleanly as part
/// of the AXIS static library and provides a home for any future non-inline
/// member implementations.

#include <axis/solver/halo_pattern.hpp>

// HaloPattern is a plain struct with inline accessors (num_remote, num_ranks).
// No non-inline members are needed — all logic lives in the header.
// This TU validates the header compiles and links as part of the AXIS library.

namespace axis::solver {

// Ensure ODR-use of the struct is valid by instantiating a default instance.
// This is a link-time sanity check — optimized away in release builds.
[[maybe_unused]] static const HaloPattern odr_check_{};

} // namespace axis::solver
