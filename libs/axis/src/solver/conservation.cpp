// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/conservation.cpp
/// @brief Implementation of conservation accounting functions.
///
/// All integral computations use Kokkos::parallel_reduce for hardware
/// portability (HELM Law #2). adjust_by_fraction uses Kokkos::parallel_for.
/// Explicit template instantiation for Kokkos::HostSpace at file bottom.

#include <algorithm>
#include <axis/solver/conservation.hpp>
#include <cmath>
#include <limits>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// source_integral
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
double source_integral(field_view<const double, 1> src_field, field_view<const double, 1> area_a, field_view<const double, 1> frac_a) {
    using exec_space = typename MemorySpace::execution_space;

    const auto n = static_cast<int>(src_field.extent(0));
    const double *src_ptr = src_field.data_handle();
    const double *area_ptr = area_a.data_handle();
    const double *frac_ptr = frac_a.data_handle();

    double result = 0.0;
    Kokkos::parallel_reduce(
        "source_integral", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const int i, double &sum) { sum += src_ptr[i] * area_ptr[i] * frac_ptr[i]; }, result);
    Kokkos::fence("source_integral_fence");

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// destination_integral
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
double destination_integral(field_view<const double, 1> dst_field, field_view<const double, 1> area_b, field_view<const double, 1> frac_b,
                            NormType norm_type) {
    using exec_space = typename MemorySpace::execution_space;

    const auto n = static_cast<int>(dst_field.extent(0));
    const double *dst_ptr = dst_field.data_handle();
    const double *area_ptr = area_b.data_handle();
    const double *frac_ptr = frac_b.data_handle();

    double result = 0.0;

    if (norm_type == NormType::DstArea) {
        // DstArea: Σ dst(j) * area_b(j) over cells with frac_b(j) != 0
        Kokkos::parallel_reduce(
            "destination_integral_DstArea", Kokkos::RangePolicy<exec_space>(0, n),
            KOKKOS_LAMBDA(const int j, double &sum) {
                if (frac_ptr[j] != 0.0) {
                    sum += dst_ptr[j] * area_ptr[j];
                }
            },
            result);
    } else {
        // FracArea: Σ dst(j) * area_b(j) * frac_b(j)
        Kokkos::parallel_reduce(
            "destination_integral_FracArea", Kokkos::RangePolicy<exec_space>(0, n),
            KOKKOS_LAMBDA(const int j, double &sum) { sum += dst_ptr[j] * area_ptr[j] * frac_ptr[j]; }, result);
    }

    Kokkos::fence("destination_integral_fence");
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// check_conservation
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
ConservationReport check_conservation(field_view<const double, 1> src_field, field_view<const double, 1> dst_field,
                                      const InterpolationMatrix<MemorySpace> &matrix, NormType norm_type) {
    ConservationReport report;

    // Compute source integral: Σ src(i) * area_a(i) * frac_a(i)
    report.src_integral = source_integral<MemorySpace>(src_field, matrix.area_a(), matrix.frac_a());

    // Compute destination integral (NormType-aware)
    report.dst_integral = destination_integral<MemorySpace>(dst_field, matrix.area_b(), matrix.frac_b(), norm_type);

    // Compute errors
    report.absolute_error = std::abs(report.src_integral - report.dst_integral);

    // Relative error: absolute / max(|src|, |dst|, epsilon)
    // Using epsilon to avoid division by zero when both integrals are zero.
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double denom = std::max({std::abs(report.src_integral), std::abs(report.dst_integral), eps});
    report.relative_error = report.absolute_error / denom;

    return report;
}

// ─────────────────────────────────────────────────────────────────────────────
// adjust_by_fraction
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void adjust_by_fraction(field_view<double, 1> dst_field, field_view<const double, 1> frac_b) {
    using exec_space = typename MemorySpace::execution_space;

    const auto n = static_cast<int>(dst_field.extent(0));
    double *dst_ptr = dst_field.data_handle();
    const double *frac_ptr = frac_b.data_handle();

    Kokkos::parallel_for(
        "adjust_by_fraction", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const int j) {
            if (frac_ptr[j] != 0.0) {
                dst_ptr[j] /= frac_ptr[j];
            }
        });
    Kokkos::fence("adjust_by_fraction_fence");
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations — HostSpace
// ─────────────────────────────────────────────────────────────────────────────

template double source_integral<Kokkos::HostSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>);

template double destination_integral<Kokkos::HostSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>,
                                                        NormType);

template ConservationReport check_conservation<Kokkos::HostSpace>(field_view<const double, 1>, field_view<const double, 1>,
                                                                  const InterpolationMatrix<Kokkos::HostSpace> &, NormType);

template void adjust_by_fraction<Kokkos::HostSpace>(field_view<double, 1>, field_view<const double, 1>);

#ifdef KOKKOS_ENABLE_CUDA
template double source_integral<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>);

template double destination_integral<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>,
                                                        NormType);

template ConservationReport check_conservation<Kokkos::CudaSpace>(field_view<const double, 1>, field_view<const double, 1>,
                                                                  const InterpolationMatrix<Kokkos::CudaSpace> &, NormType);

template void adjust_by_fraction<Kokkos::CudaSpace>(field_view<double, 1>, field_view<const double, 1>);
#endif

#ifdef KOKKOS_ENABLE_HIP
template double source_integral<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>);

template double destination_integral<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>, field_view<const double, 1>,
                                                       NormType);

template ConservationReport check_conservation<Kokkos::HIPSpace>(field_view<const double, 1>, field_view<const double, 1>,
                                                                 const InterpolationMatrix<Kokkos::HIPSpace> &, NormType);

template void adjust_by_fraction<Kokkos::HIPSpace>(field_view<double, 1>, field_view<const double, 1>);
#endif

}  // namespace axis::solver
