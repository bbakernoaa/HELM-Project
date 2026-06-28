// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_CONSERVATION_HPP
#define AXIS_SOLVER_CONSERVATION_HPP

/// @file axis/solver/conservation.hpp
/// @brief Conservation accounting: source/destination integrals, error report,
///        and fraction adjustment for partially-covered cells.
///
/// Provides:
///   - ConservationReport: aggregate struct with src/dst integrals and errors
///   - source_integral: Σ src(i) * area_a(i) * frac_a(i)
///   - destination_integral: NormType-aware destination sum
///   - check_conservation: one-shot report builder
///   - adjust_by_fraction: divide dst(j) by frac_b(j) for DstArea recovery
///
/// All reductions use Kokkos::parallel_reduce for hardware portability.
/// Templated on MemorySpace; explicit instantiation for HostSpace in .cpp.

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/types.hpp>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// ConservationReport — aggregate result of conservation check
// ─────────────────────────────────────────────────────────────────────────────

/// Global mass/integral accounting for a regridding operation.
///
/// After applying conservative weights, conservation is verified by comparing
/// source and destination integrals using the exact ESMF formulas.
struct ConservationReport {
    double src_integral{0.0};    ///< Σ src(i) * area_a(i) * frac_a(i)
    double dst_integral{0.0};    ///< Σ dst(j) * area_b(j) [* frac_b(j) if FracArea]
    double absolute_error{0.0};  ///< |src_integral - dst_integral|
    double relative_error{0.0};  ///< absolute_error / max(|src|, |dst|, epsilon)
};

// ─────────────────────────────────────────────────────────────────────────────
// source_integral — Kokkos reduction over source field
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the source integral: Σ src(i) * area_a(i) * frac_a(i).
///
/// Uses Kokkos::parallel_reduce for hardware-portable summation.
///
/// @tparam MemorySpace Kokkos memory space (default HostSpace)
/// @param src_field   Source field values [n_src]
/// @param area_a      Source cell areas [n_src]
/// @param frac_a      Source fractions [n_src]
/// @return            The integrated source quantity
template <class MemorySpace = Kokkos::HostSpace>
[[nodiscard]] double source_integral(field_view<const double, 1> src_field, field_view<const double, 1> area_a, field_view<const double, 1> frac_a);

// ─────────────────────────────────────────────────────────────────────────────
// destination_integral — NormType-aware Kokkos reduction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the destination integral consistent with the configured NormType.
///
/// For DstArea:  Σ dst(j) * area_b(j) over cells where frac_b(j) != 0.
/// For FracArea: Σ dst(j) * area_b(j) * frac_b(j).
///
/// @tparam MemorySpace Kokkos memory space (default HostSpace)
/// @param dst_field   Destination field values [n_dst]
/// @param area_b      Destination cell areas [n_dst]
/// @param frac_b      Destination fractions [n_dst]
/// @param norm_type   Normalization type (DstArea or FracArea)
/// @return            The integrated destination quantity
template <class MemorySpace = Kokkos::HostSpace>
[[nodiscard]] double destination_integral(field_view<const double, 1> dst_field, field_view<const double, 1> area_b,
                                          field_view<const double, 1> frac_b, NormType norm_type);

// ─────────────────────────────────────────────────────────────────────────────
// check_conservation — build a full ConservationReport
// ─────────────────────────────────────────────────────────────────────────────

/// Build a full conservation report from source/destination fields after apply.
///
/// Internally calls source_integral and destination_integral, then computes
/// absolute_error = |src_integral - dst_integral| and relative_error =
/// absolute_error / max(|src_integral|, |dst_integral|, epsilon).
///
/// @tparam MemorySpace Kokkos memory space (default HostSpace)
/// @param src_field   Source field values [n_src]
/// @param dst_field   Destination field values [n_dst]
/// @param matrix      The interpolation matrix (provides area_a, area_b, frac_a, frac_b)
/// @param norm_type   Normalization type used in the apply
/// @return            ConservationReport with integrals and error metrics
template <class MemorySpace = Kokkos::HostSpace>
[[nodiscard]] ConservationReport check_conservation(field_view<const double, 1> src_field, field_view<const double, 1> dst_field,
                                                    const InterpolationMatrix<MemorySpace> &matrix, NormType norm_type);

// ─────────────────────────────────────────────────────────────────────────────
// adjust_by_fraction — recover true values from DstArea-normalized output
// ─────────────────────────────────────────────────────────────────────────────

/// Divide dst(j) by frac_b(j) for all cells where frac_b(j) != 0.
///
/// This recovers the true interpolated value from a DstArea-normalized apply
/// result. For fully-covered cells frac_b == 1.0 so the division is identity.
/// Matches ESMF guidance for DstArea normalization recovery.
///
/// Uses Kokkos::parallel_for for hardware-portable element-wise operation.
///
/// @tparam MemorySpace Kokkos memory space (default HostSpace)
/// @param dst_field   Mutable destination field values [n_dst] (modified in place)
/// @param frac_b      Destination fractions [n_dst]
template <class MemorySpace = Kokkos::HostSpace>
void adjust_by_fraction(field_view<double, 1> dst_field, field_view<const double, 1> frac_b);

}  // namespace axis::solver

#endif  // AXIS_SOLVER_CONSERVATION_HPP
