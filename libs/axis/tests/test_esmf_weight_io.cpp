// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/io/esmf_weight_io.hpp>
#include <axis/solver/interpolation_matrix.hpp>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

TEST(EsmfWeightIOTest, NetCdfWeightRoundtrip) {
#ifndef AXIS_HAVE_NETCDF
    GTEST_SKIP() << "AXIS built without NetCDF support — skipping ESMF weight IO test";
#endif

    const std::size_t n_src = 8;
    const std::size_t n_dst = 4;
    const std::size_t nnz = 6;

    // 1. Create a mock sparse matrix to write
    Kokkos::View<index_t *, MemSpace> rows("rows", nnz);
    Kokkos::View<index_t *, MemSpace> cols("cols", nnz);
    Kokkos::View<double *, MemSpace> vals("vals", nnz);

    // Mock COO entries: (row, col) = val
    rows(0) = 0;
    cols(0) = 1;
    vals(0) = 0.5;
    rows(1) = 0;
    cols(1) = 2;
    vals(1) = 0.5;
    rows(2) = 1;
    cols(2) = 3;
    vals(2) = 1.0;
    rows(3) = 2;
    cols(3) = 4;
    vals(3) = 0.25;
    rows(4) = 2;
    cols(4) = 5;
    vals(4) = 0.75;
    rows(5) = 3;
    cols(5) = 7;
    vals(5) = 1.0;

    solver::InterpolationMatrix<MemSpace> original_matrix(
        vals, rows, cols, Kokkos::View<double *, MemSpace>("frac_a", n_src), Kokkos::View<double *, MemSpace>("frac_b", n_dst),
        Kokkos::View<double *, MemSpace>("area_a", n_src), Kokkos::View<double *, MemSpace>("area_b", n_dst), n_src, n_dst);

    // 2. Write matrix to NetCDF ESMF weights file
    const std::string filename = "test_esmf_weights.nc";
    io::EsmfWeightIO<MemSpace>::write_esmf(filename, original_matrix);

    // 3. Read back from the NetCDF file
    auto reloaded_matrix = io::EsmfWeightIO<MemSpace>::read_esmf(filename);

    // 4. Assert bitwise and structural identity
    EXPECT_EQ(reloaded_matrix.n_src(), n_src);
    EXPECT_EQ(reloaded_matrix.n_dst(), n_dst);
    EXPECT_EQ(reloaded_matrix.nnz(), nnz);

    auto h_orig_rows = rows;
    auto h_orig_cols = cols;
    auto h_orig_vals = vals;

    auto h_reload_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reloaded_matrix.factor_row_view());
    auto h_reload_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reloaded_matrix.factor_col_view());
    auto h_reload_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reloaded_matrix.factor_list_view());

    for (std::size_t k = 0; k < nnz; ++k) {
        EXPECT_EQ(h_reload_rows(k), h_orig_rows(k));
        EXPECT_EQ(h_reload_cols(k), h_orig_cols(k));
        EXPECT_DOUBLE_EQ(h_reload_vals(k), h_orig_vals(k));
    }

    // Clean up temporary NetCDF file
    std::remove(filename.c_str());
}

}  // namespace axis::test
