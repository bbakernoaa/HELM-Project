// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

TEST(CoastalRenormalization, RenormalizationAndExtrapolation) {
    const std::size_t ni = 4;
    const std::size_t nj = 4;
    const std::size_t n_points = ni * nj;

    // 1. Create a structured source grid
    Kokkos::View<double *, MemSpace> src_cx("src_cx", n_points);
    Kokkos::View<double *, MemSpace> src_cy("src_cy", n_points);
    for (std::size_t idx = 0; idx < n_points; ++idx) {
        src_cx(idx) = static_cast<double>(idx % ni);
        src_cy(idx) = static_cast<double>(idx / ni);
    }

    topology::StructuredGrid<MemSpace> src_grid(ni, nj, src_cx, src_cy, topology::CoordinateSystem::SphericalDeg);

    // Build corners for structured source grid
    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    Kokkos::View<double *, MemSpace> corners_lon("corners_lon", nc_lon * nc_lat);
    Kokkos::View<double *, MemSpace> corners_lat("corners_lat", nc_lon * nc_lat);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * nc_lon;
            corners_lon(idx) = static_cast<double>(i) - 0.5;
            corners_lat(idx) = static_cast<double>(j) - 0.5;
        }
    }
    src_grid.set_corners(corners_lon, corners_lat);

    // 2. Define a land mask on the source mesh:
    //    All cells are dry (0/land) EXCEPT cell index 5 (which is wet/active 1).
    Kokkos::View<int *, MemSpace> src_mask("src_mask", n_points);
    Kokkos::deep_copy(src_mask, 0);  // Default to dry/land
    src_mask(5) = 1;                 // Make cell 5 wet

    auto src_mesh = src_grid.to_unstructured();
    src_mesh = topology::UnstructuredMesh<MemSpace>(src_mesh.node_coords_view(), src_mesh.conn_offsets_view(), src_mesh.conn_indices_view(),
                                                    src_mesh.coord_system(), src_mesh.cell_areas_view(), src_mask);

    // 3. Create a 1-cell destination mesh centered at coordinates (1.2, 1.2) - overlaps source cell 5
    const std::size_t dst_ni = 1;
    const std::size_t dst_nj = 1;
    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = 1.2;
    dst_cy(0) = 1.2;

    topology::StructuredGrid<MemSpace> dst_grid(dst_ni, dst_nj, dst_cx, dst_cy, topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_corners_lon("dst_corners_lon", 4);
    Kokkos::View<double *, MemSpace> dst_corners_lat("dst_corners_lat", 4);
    dst_corners_lon(0) = 0.7;
    dst_corners_lat(0) = 0.7;
    dst_corners_lon(1) = 1.7;
    dst_corners_lat(1) = 0.7;
    dst_corners_lon(2) = 1.7;
    dst_corners_lat(2) = 1.7;
    dst_corners_lon(3) = 0.7;
    dst_corners_lat(3) = 1.7;
    dst_grid.set_corners(dst_corners_lon, dst_corners_lat);
    auto dst_mesh = dst_grid.to_unstructured();

    // 4. Generate weights with default NearestWet extrapolation enabled
    RegridConfig config;
    config.method = InterpolationMethod::Bilinear;
    config.unmapped = UnmappedAction::Ignore;
    config.extrap_method = ExtrapolationAction::NearestWet;

    auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // Since destination cell overlaps only one wet cell (cell 5), its weight
    // must be renormalized to sum exactly to 1.0!
    EXPECT_EQ(W.nnz(), 1);
    EXPECT_EQ(W.factor_index(0).col, 5);
    EXPECT_NEAR(W.factor_list()(0), 1.0, 1e-12);

    // 5. Create a completely dry destination cell at coordinates (3.0, 3.0) - far from wet cell 5
    Kokkos::View<double *, MemSpace> dry_cx("dry_cx", 1);
    Kokkos::View<double *, MemSpace> dry_cy("dry_cy", 1);
    dry_cx(0) = 3.0;
    dry_cy(0) = 3.0;

    topology::StructuredGrid<MemSpace> dry_dst_grid(dst_ni, dst_nj, dry_cx, dry_cy, topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dry_corners_lon("dry_corners_lon", 4);
    Kokkos::View<double *, MemSpace> dry_corners_lat("dry_corners_lat", 4);
    dry_corners_lon(0) = 2.5;
    dry_corners_lat(0) = 2.5;
    dry_corners_lon(1) = 3.5;
    dry_corners_lat(1) = 2.5;
    dry_corners_lon(2) = 3.5;
    dry_corners_lat(2) = 3.5;
    dry_corners_lon(3) = 2.5;
    dry_corners_lat(3) = 3.5;
    dry_dst_grid.set_corners(dry_corners_lon, dry_corners_lat);
    auto dry_dst_mesh = dry_dst_grid.to_unstructured();

    auto W_dry = WeightGenerator::generate<MemSpace>(src_mesh, dry_dst_mesh, config);

    // Under NearestWet policy, the completely unmapped destination row must be
    // extrapolated to the closest wet source cell (which is cell 5) with a weight of 1.0!
    EXPECT_EQ(W_dry.nnz(), 1);
    EXPECT_EQ(W_dry.factor_index(0).col, 5);
    EXPECT_NEAR(W_dry.factor_list()(0), 1.0, 1e-12);
}

}  // namespace axis::test
