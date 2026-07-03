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

    using execution_space = typename MemorySpace::execution_space;
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;

    std::size_t scratch_bytes = 4 * n_src * sizeof(double);
    policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    Kokkos::parallel_for(
        "VerticalInterpolate1D", policy, KOKKOS_LAMBDA(const member_type &team) {
            const std::size_t c = team.league_rank();

            Kokkos::View<double*, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged> scratch_view(team.team_scratch(0), 4 * n_src);
            double *src_x = scratch_view.data();
            double *src_y = src_x + n_src;
            double *d = src_y + n_src;
            double *scratch_arr = d + n_src;

            // Cooperatively fill src_x and src_y
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(i);
            });
            team.team_barrier();

            // Thread 0 solves the column spline
            if (team.team_rank() == 0) {
                axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
            }
            team.team_barrier();

            // Cooperatively evaluate spline for each target destination level
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                double target = dst_levels(j);

                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx + 1] < target) {
                    idx++;
                }

                dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                        d[idx + 1], tension);
            });
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

    using execution_space = typename MemorySpace::execution_space;
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;

    std::size_t scratch_bytes = 4 * n_src * sizeof(double);
    policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    Kokkos::parallel_for(
        "VerticalInterpolate2D", policy, KOKKOS_LAMBDA(const member_type &team) {
            const std::size_t c = team.league_rank();

            Kokkos::View<double*, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged> scratch_view(team.team_scratch(0), 4 * n_src);
            double *src_x = scratch_view.data();
            double *src_y = src_x + n_src;
            double *d = src_y + n_src;
            double *scratch_arr = d + n_src;

            // Cooperatively fill src_x and src_y
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                src_y[i] = src_field(c, i);
                src_x[i] = src_levels(c, i);
            });
            team.team_barrier();

            // Thread 0 solves the column spline
            if (team.team_rank() == 0) {
                axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
            }
            team.team_barrier();

            // Cooperatively evaluate spline for each target destination level
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                double target = dst_levels(c, j);

                std::size_t idx = 0;
                while (idx < n_src - 2 && src_x[idx + 1] < target) {
                    idx++;
                }

                dst_field(c, j) = axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx],
                                                                        d[idx + 1], tension);
            });
        });
}

// Explicit template instantiations
template class VerticalRegridder<Kokkos::HostSpace>;

}  // namespace axis::solver
