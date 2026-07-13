// SPDX-License-Identifier: Apache-2.0
// AXIS unit tests for InterpolationMethod::Budget

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <vector>
#include <cmath>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;
using solver::RegridConfig;
using solver::InterpolationMethod;

static topology::UnstructuredMesh<MemSpace> make_simple_cartesian_mesh(std::size_t n, double size) {
    const double dx = size / static_cast<double>(n);
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }
    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);
    return grid.to_unstructured();
}

TEST(BudgetInterpolationTest, ConstantFieldPreservation) {
    auto src = make_simple_cartesian_mesh(4, 1.0);
    auto dst = make_simple_cartesian_mesh(2, 1.0);

    for (std::uint32_t subgrid_size : {1, 3, 5}) {
        RegridConfig config;
        config.method = InterpolationMethod::Budget;
        config.budget_subgrid_size = subgrid_size;
        config.budget_min_valid_fraction = 0.5;

        auto matrix = solver::WeightGenerator::generate<MemSpace>(src, dst, config);

        std::vector<double> src_data(src.n_cells(), 1.0);
        std::vector<double> dst_data(dst.n_cells(), 0.0);

        field_view<const double, 1> src_view(src_data.data(), src.n_cells());
        field_view<double, 1> dst_view(dst_data.data(), dst.n_cells());

        solver::apply<MemSpace>(matrix, src_view, dst_view);

        for (std::size_t j = 0; j < dst.n_cells(); ++j) {
            EXPECT_NEAR(dst_data[j], 1.0, 1e-12);
        }
    }
}

TEST(BudgetInterpolationTest, PartitionOfUnity) {
    auto src = make_simple_cartesian_mesh(5, 1.0);
    auto dst = make_simple_cartesian_mesh(3, 1.0);

    RegridConfig config;
    config.method = InterpolationMethod::Budget;
    config.budget_subgrid_size = 5;
    config.budget_min_valid_fraction = 0.5;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src, dst, config);

    auto row = matrix.factor_row_view();
    auto val = matrix.factor_list_view();
    auto nnz = matrix.nnz();

    std::vector<double> row_sums(dst.n_cells(), 0.0);
    for (std::size_t k = 0; k < nnz; ++k) {
        row_sums[row(k)] += val(k);
    }

    for (std::size_t j = 0; j < dst.n_cells(); ++j) {
        EXPECT_NEAR(row_sums[j], 1.0, 1e-12);
    }
}

} // namespace axis::test
