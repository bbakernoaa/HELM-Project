// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/weight_generator.hpp>
#include <cmath>
#include <vector>

namespace axis::solver {

template <typename MemorySpace>
std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>> VectorWeightGenerator<MemorySpace>::generate(
    const topology::UnstructuredMesh<MemorySpace> &src_mesh, const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
    const GridRotation<MemorySpace> &src_rotation, const GridRotation<MemorySpace> &dst_rotation, const RegridConfig &config) {
    // 1. Generate the standard scalar spatial interpolation weight matrix
    auto W_scalar = WeightGenerator::generate<MemorySpace>(src_mesh, dst_mesh, config);

    const std::size_t n_src = W_scalar.n_src();
    const std::size_t n_dst = W_scalar.n_dst();
    const std::size_t nnz_scalar = W_scalar.nnz();

    // 2. We extract the scalar sparse matrix underlying Views directly
    const auto &rows = W_scalar.factor_row_view();
    const auto &cols = W_scalar.factor_col_view();
    const auto &vals = W_scalar.factor_list_view();

    // Host mirrors for mapping and trigonometric calculations
    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
    auto h_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cols);
    auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vals);
    auto h_src_alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src_rotation.alpha);
    auto h_dst_alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_rotation.alpha);

    // Each scalar entry W_ji yields exactly 2 entries in W_u and 2 entries in W_v
    std::size_t nnz_vector = nnz_scalar * 2;

    std::vector<index_t> u_rows, u_cols;
    std::vector<double> u_vals;
    u_rows.reserve(nnz_vector);
    u_cols.reserve(nnz_vector);
    u_vals.reserve(nnz_vector);

    std::vector<index_t> v_rows, v_cols;
    std::vector<double> v_vals;
    v_rows.reserve(nnz_vector);
    v_cols.reserve(nnz_vector);
    v_vals.reserve(nnz_vector);

    for (std::size_t k = 0; k < nnz_scalar; ++k) {
        index_t j = h_rows(k);  // dst cell index
        index_t i = h_cols(k);  // src cell index
        double w = h_vals(k);

        double a_src = h_src_alpha(i);
        double a_dst = h_dst_alpha(j);
        double diff_alpha = a_dst - a_src;

        double cos_d = std::cos(diff_alpha);
        double sin_d = std::sin(diff_alpha);

        // --- Coupled Weights for W_u ---
        // u_dst_j += w * cos(a_dst - a_src) * u_src_i
        u_rows.push_back(j);
        u_cols.push_back(i);
        u_vals.push_back(w * cos_d);

        // u_dst_j += w * sin(a_dst - a_src) * v_src_i
        u_rows.push_back(j);
        u_cols.push_back(i + n_src);  // Stacked v component index
        u_vals.push_back(w * sin_d);

        // --- Coupled Weights for W_v ---
        // v_dst_j += w * -sin(a_dst - a_src) * u_src_i
        v_rows.push_back(j);
        v_cols.push_back(i);
        v_vals.push_back(-w * sin_d);

        // v_dst_j += w * cos(a_dst - a_src) * v_src_i
        v_rows.push_back(j);
        v_cols.push_back(i + n_src);  // Stacked v component index
        v_vals.push_back(w * cos_d);
    }

    // 3. Build the final host-space coupled vector matrices
    Kokkos::View<index_t *, Kokkos::HostSpace> host_u_rows("u_rows", nnz_vector);
    Kokkos::View<index_t *, Kokkos::HostSpace> host_u_cols("u_cols", nnz_vector);
    Kokkos::View<double *, Kokkos::HostSpace> host_u_vals("u_vals", nnz_vector);

    Kokkos::View<index_t *, Kokkos::HostSpace> host_v_rows("v_rows", nnz_vector);
    Kokkos::View<index_t *, Kokkos::HostSpace> host_v_cols("v_cols", nnz_vector);
    Kokkos::View<double *, Kokkos::HostSpace> host_v_vals("v_vals", nnz_vector);

    for (std::size_t k = 0; k < nnz_vector; ++k) {
        host_u_rows(k) = u_rows[k];
        host_u_cols(k) = u_cols[k];
        host_u_vals(k) = u_vals[k];

        host_v_rows(k) = v_rows[k];
        host_v_cols(k) = v_cols[k];
        host_v_vals(k) = v_vals[k];
    }

    // Copy to targeted MemorySpace
    auto dev_u_rows = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_u_rows);
    auto dev_u_cols = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_u_cols);
    auto dev_u_vals = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_u_vals);

    auto dev_v_rows = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_v_rows);
    auto dev_v_cols = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_v_cols);
    auto dev_v_vals = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_v_vals);

    // Create stacked fraction and area views matching double n_src
    Kokkos::View<double *, MemorySpace> dev_frac_a("frac_a", n_src * 2);
    Kokkos::View<double *, MemorySpace> dev_area_a("area_a", n_src * 2);

    // Copy the original fractions and areas to the first half, and repeat for the second (stacked) half
    auto h_frac_a = W_scalar.frac_a();
    auto h_area_a = W_scalar.area_a();
    auto dev_orig_frac_a = W_scalar.frac_a_view();
    auto dev_orig_area_a = W_scalar.area_a_view();

    Kokkos::parallel_for(
        "CopyFractions", n_src, KOKKOS_LAMBDA(const std::size_t i) {
            dev_frac_a(i) = dev_orig_frac_a(i);
            dev_frac_a(i + n_src) = dev_orig_frac_a(i);
            dev_area_a(i) = dev_orig_area_a(i);
            dev_area_a(i + n_src) = dev_orig_area_a(i);
        });

    InterpolationMatrix<MemorySpace> W_u(dev_u_vals, dev_u_rows, dev_u_cols, dev_frac_a, W_scalar.frac_b_view(), dev_area_a, W_scalar.area_b_view(),
                                         n_src * 2, n_dst);

    InterpolationMatrix<MemorySpace> W_v(dev_v_vals, dev_v_rows, dev_v_cols, dev_frac_a, W_scalar.frac_b_view(), dev_area_a, W_scalar.area_b_view(),
                                         n_src * 2, n_dst);

    return {std::move(W_u), std::move(W_v)};
}

template class VectorWeightGenerator<Kokkos::HostSpace>;

}  // namespace axis::solver
