// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/solver/apply.hpp>
#include <cmath>

namespace axis::test {

using namespace axis::solver;
using MemSpace = Kokkos::HostSpace;

TEST(VectorRegridderTest, CoupledRotationAndInterpolation) {
    const std::size_t ni = 4;
    const std::size_t nj = 4;
    const std::size_t n_points = ni * nj;

    // 1. Create a simple regular source mesh (using identical geometry for source & destination to isolate rotation)
    Kokkos::View<double*, MemSpace> src_cx("src_cx", n_points);
    Kokkos::View<double*, MemSpace> src_cy("src_cy", n_points);
    for (std::size_t idx = 0; idx < n_points; ++idx) {
        src_cx(idx) = static_cast<double>(idx % ni);
        src_cy(idx) = static_cast<double>(idx / ni);
    }

    topology::StructuredGrid<MemSpace> src_grid(
        ni, nj, src_cx, src_cy, topology::CoordinateSystem::SphericalDeg);
    auto src_mesh = src_grid.to_unstructured();

    // 2. Define grid rotation angles (source rotation = 0.0, destination rotation = 0.5 radians)
    Kokkos::View<double*, MemSpace> src_alpha("src_alpha", n_points);
    Kokkos::View<double*, MemSpace> dst_alpha("dst_alpha", n_points);
    Kokkos::deep_copy(src_alpha, 0.0);
    Kokkos::deep_copy(dst_alpha, 0.5); // 0.5 radians rotation

    GridRotation<MemSpace> src_rot{src_alpha};
    GridRotation<MemSpace> dst_rot{dst_alpha};

    RegridConfig config;
    config.method = InterpolationMethod::Bilinear;
    config.unmapped = UnmappedAction::Ignore;

    // 3. Generate coupled vector weights
    auto [W_u, W_v] = VectorWeightGenerator<MemSpace>::generate(
        src_mesh, src_mesh, src_rot, dst_rot, config);

    // 4. Set source field to constant unit vector u = 1.0, v = 0.0 aligned with the grid
    Kokkos::View<double*, MemSpace> src_uv("src_uv", n_points * 2);
    for (std::size_t idx = 0; idx < n_points; ++idx) {
        src_uv(idx) = 1.0;            // u_src = 1.0
        src_uv(idx + n_points) = 0.0; // v_src = 0.0
    }

    Kokkos::View<double*, MemSpace> dst_u("dst_u", n_points);
    Kokkos::View<double*, MemSpace> dst_v("dst_v", n_points);

    // 5. Apply weights (converting Views to field_views)
    axis::field_view<const double, 1> src_uv_view(src_uv.data(), src_uv.extent(0));
    axis::field_view<double, 1> dst_u_view(dst_u.data(), dst_u.extent(0));
    axis::field_view<double, 1> dst_v_view(dst_v.data(), dst_v.extent(0));

    apply(W_u, src_uv_view, dst_u_view);
    apply(W_v, src_uv_view, dst_v_view);

    // 6. Assert rotated components:
    //    u_dst = u_src * cos(0.5) = cos(0.5)
    //    v_dst = u_src * -sin(0.5) = -sin(0.5)
    double expected_u = std::cos(0.5);
    double expected_v = -std::sin(0.5);

    for (std::size_t idx = 0; idx < n_points; ++idx) {
        EXPECT_NEAR(dst_u(idx), expected_u, 1e-12);
        EXPECT_NEAR(dst_v(idx), expected_v, 1e-12);
    }
}

} // namespace axis::test
