// SPDX-License-Identifier: Apache-2.0
// Property-based testing for InterpolationMethod::Budget

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
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

static topology::UnstructuredMesh<MemSpace> make_prop_mesh(std::size_t n, double size) {
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

RC_GTEST_PROP(PropBudget, RowSumsAreUnityAndIndicesAreInBounds, ()) {
    // Generate random grid size between 2 and 6
    const auto src_size = *rc::gen::inRange<std::size_t>(2, 7);
    const auto dst_size = *rc::gen::inRange<std::size_t>(2, 7);

    auto src = make_prop_mesh(src_size, 1.0);
    auto dst = make_prop_mesh(dst_size, 1.0);

    RegridConfig config;
    config.method = InterpolationMethod::Budget;
    config.budget_subgrid_size = *rc::gen::inRange<std::uint32_t>(2, 6);
    config.budget_min_valid_fraction = 0.5;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src, dst, config);

    const auto row = matrix.factor_row_view();
    const auto col = matrix.factor_col_view();
    const auto val = matrix.factor_list_view();
    const auto nnz = matrix.nnz();

    std::vector<double> row_sums(dst.n_cells(), 0.0);

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(row(k) < dst.n_cells());
        RC_ASSERT(col(k) < src.n_cells());
        row_sums[row(k)] += val(k);
    }

    for (std::size_t j = 0; j < dst.n_cells(); ++j) {
        if (row_sums[j] > 0.0) {
            RC_ASSERT(std::abs(row_sums[j] - 1.0) < 1e-12);
        }
    }
}

} // namespace axis::test
