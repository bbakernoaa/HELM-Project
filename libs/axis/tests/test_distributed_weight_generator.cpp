// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <axis/distributed/distributed_weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/solver/apply.hpp>
#include <mpi.h>
#include <cmath>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

TEST(DistributedWeightGeneratorTest, GlobalRowMappingTwoRanks) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    bool we_initialized_mpi = false;
    if (!mpi_initialized) {
        MPI_Init(nullptr, nullptr);
        we_initialized_mpi = true;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (we_initialized_mpi) {
            MPI_Finalize();
        }
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks — skipping distributed weight test";
    }

    const std::size_t src_ni = 8;
    const std::size_t src_nj = 4;
    const std::size_t src_n_points = src_ni * src_nj;

    // 1. Create a global source grid on both ranks
    Kokkos::View<double*, MemSpace> src_cx("src_cx", src_n_points);
    Kokkos::View<double*, MemSpace> src_cy("src_cy", src_n_points);
    for (std::size_t idx = 0; idx < src_n_points; ++idx) {
        src_cx(idx) = static_cast<double>(idx % src_ni);
        src_cy(idx) = static_cast<double>(idx / src_ni);
    }
    topology::StructuredGrid<MemSpace> src_grid(
        src_ni, src_nj, src_cx, src_cy, topology::CoordinateSystem::SphericalDeg);
    auto src_mesh = src_grid.to_unstructured();

    // 2. Destination Grid: Total size is 4x2 = 8 cells.
    //    We partition it horizontally across 2 ranks:
    //    - Rank 0 gets cells for x from 0 to 1 (cols 0, 1) -> 2x2 = 4 cells
    //    - Rank 1 gets cells for x from 2 to 3 (cols 2, 3) -> 2x2 = 4 cells
    const std::size_t dst_ni = 2; // Local column count per rank
    const std::size_t dst_nj = 2; // Local row count per rank
    const std::size_t dst_n_points_local = dst_ni * dst_nj;

    Kokkos::View<double*, MemSpace> dst_cx_local("dst_cx_local", dst_n_points_local);
    Kokkos::View<double*, MemSpace> dst_cy_local("dst_cy_local", dst_n_points_local);

    double x_offset = (rank == 0) ? 0.0 : 2.0;

    for (std::size_t j = 0; j < dst_nj; ++j) {
        for (std::size_t i = 0; i < dst_ni; ++i) {
            std::size_t idx = i + j * dst_ni;
            dst_cx_local(idx) = x_offset + static_cast<double>(i) + 0.5;
            dst_cy_local(idx) = static_cast<double>(j) + 0.5;
        }
    }

    topology::StructuredGrid<MemSpace> dst_grid_local(
        dst_ni, dst_nj, dst_cx_local, dst_cy_local, topology::CoordinateSystem::SphericalDeg);
    auto dst_mesh_local = dst_grid_local.to_unstructured();

    // 3. Generate distributed weights using MPI_COMM_WORLD
    RegridConfig config;
    config.method = InterpolationMethod::Bilinear;
    config.unmapped = UnmappedAction::Ignore;

    auto W_dist = distributed::DistributedWeightGenerator<MemSpace>::generate(
        src_mesh, dst_mesh_local, config, MPI_COMM_WORLD);

    // 4. Assert globally mapped matrix properties
    EXPECT_EQ(W_dist.n_src(), src_n_points);
    EXPECT_EQ(W_dist.n_dst(), 8); // Global destination count across both ranks (4 + 4)

    auto rows = W_dist.factor_row_view();
    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);

    // Verify row indices correspond strictly to local partition mapping
    std::size_t expected_offset = (rank == 0) ? 0 : 4;
    for (std::size_t k = 0; k < W_dist.nnz(); ++k) {
        EXPECT_GE(h_rows(k), expected_offset);
        EXPECT_LT(h_rows(k), expected_offset + 4);
    }

    if (we_initialized_mpi) {
        MPI_Finalize();
    }
}

} // namespace axis::test
