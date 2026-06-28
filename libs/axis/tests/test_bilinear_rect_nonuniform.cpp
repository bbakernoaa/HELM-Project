// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file tests/test_bilinear_rect_nonuniform.cpp
/// @brief GTest unit tests for the bilinear non-uniform rectilinear grid fast-path.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static topology::UnstructuredMesh<MemSpace> make_nonuniform_rect_grid(const std::vector<double> &lons, const std::vector<double> &lats) {
    const std::size_t ni = lons.size() - 1;
    const std::size_t nj = lats.size() - 1;

    // Cell centers: average of boundaries
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = 0.5 * (lons[i] + lons[i + 1]);
            cy(i + j * ni) = 0.5 * (lats[j] + lats[j + 1]);
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

    // Corner coordinates
    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_lon * nc_lat);
    Kokkos::View<double *, MemSpace> cry("cry", nc_lon * nc_lat);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_lon) = lons[i];
            cry(i + j * nc_lon) = lats[j];
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));
    return grid.to_unstructured();
}

TEST(BilinearRectNonUniform, KnownAnalyticWeights) {
    // Coordinate vectors with non-uniform spacings
    std::vector<double> src_lons = {0.0, 10.0, 30.0, 40.0};  // sizes: [10, 20, 10]
    std::vector<double> src_lats = {0.0, 10.0, 30.0};        // sizes: [10, 20]

    // Dst has 2x1 cells
    std::vector<double> dst_lons = {0.0, 20.0, 40.0};
    std::vector<double> dst_lats = {10.0, 20.0};

    auto src_mesh = make_nonuniform_rect_grid(src_lons, src_lats);
    auto dst_mesh = make_nonuniform_rect_grid(dst_lons, dst_lats);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    EXPECT_EQ(matrix.n_src(), 3 * 2);
    EXPECT_EQ(matrix.n_dst(), 2 * 1);

    // Dst cell 0 is at (10.0, 15.0).
    // In src lons {0, 10, 30, 40}, 10.0 is on boundary. Let's look at row 0 (center at 10.0, 15.0).
    // Row sums must be 1.0 (partition of unity)
    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();
    double sum = 0.0;
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        if (rows[k] == 0) {
            sum += vals[k];
        }
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

}  // namespace axis::test
