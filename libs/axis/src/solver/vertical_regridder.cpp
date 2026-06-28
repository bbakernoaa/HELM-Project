// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/detail/tspack.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <stdexcept>
#include <string>

namespace axis::solver {

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                 Kokkos::View<const double *, MemorySpace> src_levels,
                                                 Kokkos::View<const double *, MemorySpace> dst_levels, double tension) {
    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(0);
    const std::size_t n_dst = dst_levels.extent(0);

    if (n_col != dst_field.extent(0)) {
        throw std::invalid_argument("VerticalRegridder: Source and destination column extents mismatch");
    }
    if (src_field.extent(1) != n_src) {
        throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
    }
    if (dst_field.extent(1) != n_dst) {
        throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
    }
    if (tension < 0.0) {
        throw std::invalid_argument("VerticalRegridder: Tension parameter must be non-negative");
    }

    constexpr std::size_t MAX_LEVELS = 256;
    if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
        throw std::invalid_argument("VerticalRegridder: Vertical level dimensions exceed hard cap of 256 levels");
    }

    Kokkos::parallel_for(
        "VerticalInterpolate1D", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_col), KOKKOS_LAMBDA(const std::size_t c) {
            double d[MAX_LEVELS];
            double scratch[MAX_LEVELS];
            double src_y[MAX_LEVELS];
            double src_x[MAX_LEVELS];

            for (std::size_t i = 0; i < n_src; ++i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(i);
            }

            axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch);

            for (std::size_t j = 0; j < n_dst; ++j) {
                double target = dst_levels(j);

                // Perform binary search or sequential search to locate target cell
                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx + 1] < target) {
                    idx++;
                }

                dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                        d[idx + 1], tension);
            }
        });
}

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                 Kokkos::View<const double **, MemorySpace> src_levels,
                                                 Kokkos::View<const double **, MemorySpace> dst_levels, double tension) {
    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(1);
    const std::size_t n_dst = dst_levels.extent(1);

    if (n_col != dst_field.extent(0) || n_col != src_levels.extent(0) || n_col != dst_levels.extent(0)) {
        throw std::invalid_argument("VerticalRegridder: Grid column dimensions mismatch across fields and level views");
    }
    if (src_field.extent(1) != n_src) {
        throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
    }
    if (dst_field.extent(1) != n_dst) {
        throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
    }
    if (tension < 0.0) {
        throw std::invalid_argument("VerticalRegridder: Tension parameter must be non-negative");
    }

    constexpr std::size_t MAX_LEVELS = 256;
    if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
        throw std::invalid_argument("VerticalRegridder: Vertical level dimensions exceed hard cap of 256 levels");
    }

    Kokkos::parallel_for(
        "VerticalInterpolate2D", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_col), KOKKOS_LAMBDA(const std::size_t c) {
            double d[MAX_LEVELS];
            double scratch[MAX_LEVELS];
            double src_y[MAX_LEVELS];
            double src_x[MAX_LEVELS];

            for (std::size_t i = 0; i < n_src; ++i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(c, i);
            }

            axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch);

            for (std::size_t j = 0; j < n_dst; ++j) {
                double target = dst_levels(c, j);

                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx + 1] < target) {
                    idx++;
                }

                dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                        d[idx + 1], tension);
            }
        });
}

// Explicit template instantiations
template class VerticalRegridder<Kokkos::HostSpace>;

}  // namespace axis::solver
