// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/topology/structured_grid.hpp>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

TEST(TripolarGridTest, FoldedNorthernSeamDetection) {
    const std::size_t ni = 4;
    const std::size_t nj = 3;
    const std::size_t n_points = ni * nj;
    const std::size_t n_nodes = (ni + 1) * (nj + 1);

    // Create center coordinates
    Kokkos::View<double*, MemSpace> cx("cx", n_points);
    Kokkos::View<double*, MemSpace> cy("cy", n_points);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            cx(idx) = -180.0 + (i + 0.5) * 90.0;
            cy(idx) = -60.0 + (j + 0.5) * 40.0;
        }
    }

    topology::StructuredGrid<MemSpace> grid(
        ni, nj, cx, cy, topology::CoordinateSystem::SphericalDeg);

    // Create corner coordinates with a folded northern boundary seam (latitudes of top row are equal)
    Kokkos::View<double*, MemSpace> corners_lon("corners_lon", n_nodes);
    Kokkos::View<double*, MemSpace> corners_lat("corners_lat", n_nodes);

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corners_lon(idx) = -180.0 + i * 90.0;
            if (j == nj) {
                // Top row folded latitude seam at 80.0 degrees North
                corners_lat(idx) = 80.0;
            } else {
                corners_lat(idx) = -80.0 + j * 40.0;
            }
        }
    }

    grid.set_corners(corners_lon, corners_lat);
    auto mesh = grid.to_unstructured();

    // Verify folded tripolar northern seam detection
    auto info = detail::detect_tripolar_grid<MemSpace>(mesh, ni, nj);

    EXPECT_TRUE(info.is_tripolar);
    EXPECT_EQ(info.ni, ni);
    EXPECT_EQ(info.nj, nj);
}

} // namespace axis::test
