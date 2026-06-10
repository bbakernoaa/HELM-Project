// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/interpolation_matrix.cpp
/// @brief Explicit template instantiations for InterpolationMatrix.
///
/// InterpolationMatrix is header-only for its inline accessors. This
/// compilation unit provides explicit template instantiations for common
/// Kokkos memory spaces so that downstream translation units do not need
/// to include the full Kokkos headers to link against the class.

#include <axis/solver/interpolation_matrix.hpp>

#include <Kokkos_Core.hpp>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations for common Kokkos memory spaces
// ─────────────────────────────────────────────────────────────────────────────

template class InterpolationMatrix<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class InterpolationMatrix<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class InterpolationMatrix<Kokkos::HIPSpace>;
#endif

} // namespace axis::solver
