// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/solver/vertical_regridder.hpp>

namespace axis::test {

TEST(VerticalRegridderTest, UniformLinearInterpolation) {
    const std::size_t n_col = 5;
    const std::size_t n_src = 11;
    const std::size_t n_dst = 6;

    Kokkos::View<double**, Kokkos::HostSpace> src_field("src_field", n_col, n_src);
    Kokkos::View<double**, Kokkos::HostSpace> dst_field("dst_field", n_col, n_dst);
    Kokkos::View<double*, Kokkos::HostSpace> src_levels("src_levels", n_src);
    Kokkos::View<double*, Kokkos::HostSpace> dst_levels("dst_levels", n_dst);

    for (std::size_t i = 0; i < n_src; ++i) {
        src_levels(i) = static_cast<double>(i);
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        dst_levels(j) = static_cast<double>(j) * 2.0;
    }

    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t i = 0; i < n_src; ++i) {
            src_field(c, i) = 2.0 * src_levels(i) + 5.0; // Perfect linear profile
        }
    }

    axis::solver::VerticalRegridder<Kokkos::HostSpace>::interpolate(
        src_field, dst_field, src_levels, dst_levels, 0.0);

    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t j = 0; j < n_dst; ++j) {
            double expected = 2.0 * dst_levels(j) + 5.0;
            EXPECT_NEAR(dst_field(c, j), expected, 1e-12);
        }
    }
}

} // namespace axis::test
