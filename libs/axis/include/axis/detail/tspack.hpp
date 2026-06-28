// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_TSPACK_HPP
#define AXIS_DETAIL_TSPACK_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace axis::detail::tspack {

/// @brief Hermite spline evaluation under tension.
/// @return Interpolated value at coordinate @c x.
KOKKOS_FUNCTION double
evaluate_spline(double x, double x1, double x2, double y1, double y2,
                double d1, double d2, double sigma);

/// @brief Solve the tridiagonal system for a single column using the Thomas algorithm.
///        No heap allocation (HELM Law #2).
template <std::size_t MaxLevels>
KOKKOS_FUNCTION void
solve_column_spline(const double* x, const double* y, std::size_t n,
                    double sigma, double* d, double* temp_scratch);

} // namespace axis::detail::tspack

#endif // AXIS_DETAIL_TSPACK_HPP
