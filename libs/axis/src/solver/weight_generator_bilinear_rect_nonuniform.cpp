// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/weight_generator_bilinear_rect_nonuniform.cpp
/// @brief Non-uniform rectilinear grid bilinear interpolation fast-path.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <axis/detail/regular_grid_detector.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace axis::solver {

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
generate_bilinear_rect_nonuniform(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config,
    const detail::RectilinearGridInfo& src_rect_info) {

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t ni = src_rect_info.ni;
    const std::size_t nj = src_rect_info.nj;

    // Retrieve unique cell boundary coordinates (HostSpace)
    auto unique_lons = src_rect_info.unique_lons;
    auto unique_lats = src_rect_info.unique_lats;

    // Retrieve destination centroids
    const auto& offsets = dst_mesh.conn_offsets_view();
    const auto& indices = dst_mesh.conn_indices_view();
    const auto& coords  = dst_mesh.node_coords_view();

    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;
    weights_vec.reserve(n_dst * 4);
    rows_vec.reserve(n_dst * 4);
    cols_vec.reserve(n_dst * 4);

    for (std::size_t c = 0; c < n_dst; ++c) {
        auto start = static_cast<std::size_t>(offsets(c));
        auto end   = static_cast<std::size_t>(offsets(c + 1));
        std::size_t n_verts = end - start;

        double lon_sum = 0.0;
        double lat_sum = 0.0;
        for (std::size_t v = start; v < end; ++v) {
            auto node_idx = static_cast<std::size_t>(indices(v));
            lon_sum += coords(node_idx, 0);
            lat_sum += coords(node_idx, 1);
        }

        double lon_d = lon_sum / static_cast<double>(n_verts);
        double lat_d = lat_sum / static_cast<double>(n_verts);

        // Perform 1D binary search to locate containing bounds
        auto lon_it = std::upper_bound(unique_lons.data(), unique_lons.data() + unique_lons.extent(0), lon_d);
        auto lat_it = std::upper_bound(unique_lats.data(), unique_lats.data() + unique_lats.extent(0), lat_d);

        int i = static_cast<int>(std::distance(unique_lons.data(), lon_it)) - 1;
        int j = static_cast<int>(std::distance(unique_lats.data(), lat_it)) - 1;

        if (i < 0 || i >= static_cast<int>(ni) || j < 0 || j >= static_cast<int>(nj)) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error("Unmapped destination cell in non-uniform bilinear");
            }
            continue;
        }

        double x0 = unique_lons(i);
        double x1 = unique_lons(i + 1);
        double y0 = unique_lats(j);
        double y1 = unique_lats(j + 1);

        double tx = (lon_d - x0) / (x1 - x0);
        double ty = (lat_d - y0) / (y1 - y0);

        tx = std::max(0.0, std::min(1.0, tx));
        ty = std::max(0.0, std::min(1.0, ty));

        std::size_t src_idx[4] = {
            static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i),
            static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i + 1),
            static_cast<std::size_t>(j + 1) * ni + static_cast<std::size_t>(i),
            static_cast<std::size_t>(j + 1) * ni + static_cast<std::size_t>(i + 1)
        };

        double wts[4] = {
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty
        };

        for (int k = 0; k < 4; ++k) {
            weights_vec.push_back(wts[k]);
            rows_vec.push_back(static_cast<index_t>(c));
            cols_vec.push_back(static_cast<index_t>(src_idx[k]));
        }
    }

    // Mirror to Target MemorySpace and build InterpolationMatrix
    const std::size_t nnz = weights_vec.size();
    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k)  = rows_vec[k];
        h_factor_col(k)  = cols_vec[k];
    }
    for (std::size_t i = 0; i < n_src; ++i) h_frac_a(i) = 1.0;
    for (std::size_t j = 0; j < n_dst; ++j) h_frac_b(j) = 1.0;

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}

template InterpolationMatrix<Kokkos::HostSpace>
generate_bilinear_rect_nonuniform<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&,
    const detail::RectilinearGridInfo&);

} // namespace axis::solver
