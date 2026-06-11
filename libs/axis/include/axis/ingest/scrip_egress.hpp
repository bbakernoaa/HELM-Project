// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_INGEST_SCRIP_EGRESS_HPP
#define AXIS_INGEST_SCRIP_EGRESS_HPP

/// @file axis/ingest/scrip_egress.hpp
/// @brief SCRIP-convention egress contract for AXIS interpolation weights.
///
/// ScripEgress is a plain-data struct exposing interpolation weights, grid
/// coordinates, and metadata in the layout expected by SCRIP/ESMF weight files.
/// The key difference from WeightEgress is the 1-based indexing for row/col
/// arrays (matching SCRIP convention) and the inclusion of grid center
/// coordinates and dimensions.
///
/// The scrip_egress() factory function builds ScripEgress views over an
/// InterpolationMatrix and source/destination meshes. The 1-based index offset
/// is applied via a lightweight Kokkos View allocation (nnz index_t values);
/// all other fields are zero-copy views into existing buffers.
///
/// ScripEgress names no Kokkos/AMIO/eckit type in its public interface — only
/// field_view — and is consumable by AMIO or Python for serialization to SCRIP
/// NetCDF weight files. Header-only: no associated .cpp compilation unit.

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <axis/types.hpp>
#include <axis/solver/regrid_config.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────

namespace axis::topology {
template <class MemorySpace>
class UnstructuredMesh;
} // namespace axis::topology

namespace axis::solver {
template <class MemorySpace>
class InterpolationMatrix;
} // namespace axis::solver

namespace axis::ingest {

// ─────────────────────────────────────────────────────────────────────────────
// ScripEgress — SCRIP-convention non-owning view over weight + grid data
// ─────────────────────────────────────────────────────────────────────────────

/// Plain-data struct exposing interpolation weights and grid metadata in the
/// SCRIP convention layout. All index arrays use 1-based addressing. Grid
/// center coordinates, areas, and fractions are non-owning views into the
/// source InterpolationMatrix and UnstructuredMesh buffers.
///
/// A consumer (AMIO, Python) serializes these views to a SCRIP NetCDF file.
struct ScripEgress {
    field_view<const double, 1>  S{};    ///< Interpolation weights [n_s]
    field_view<const index_t, 1> col{};  ///< 1-based source cell indices [n_s]
    field_view<const index_t, 1> row{};  ///< 1-based destination cell indices [n_s]

    field_view<const double, 1> src_grid_center_lon{};  ///< Source cell center longitudes [n_a]
    field_view<const double, 1> src_grid_center_lat{};  ///< Source cell center latitudes [n_a]
    field_view<const double, 1> dst_grid_center_lon{};  ///< Destination cell center longitudes [n_b]
    field_view<const double, 1> dst_grid_center_lat{};  ///< Destination cell center latitudes [n_b]

    field_view<const double, 1> src_grid_area{};   ///< Source cell areas [n_a]
    field_view<const double, 1> dst_grid_area{};   ///< Destination cell areas [n_b]
    field_view<const double, 1> src_grid_frac{};   ///< Source cell fractions [n_a]
    field_view<const double, 1> dst_grid_frac{};   ///< Destination cell fractions [n_b]

    std::size_t src_grid_dims[2]{};  ///< Source grid logical dimensions [ni, nj] (0 if unstructured)
    std::size_t dst_grid_dims[2]{};  ///< Destination grid logical dimensions [ni, nj] (0 if unstructured)

    const char* remap_method{};  ///< "conservative", "bilinear", or "nearest_neighbor"
    const char* norm_option{};   ///< "destarea" or "fracarea"
    const char* map_method{};    ///< "Conservative remapping", "Bilinear remapping", etc.

    std::size_t n_s{0};  ///< Number of nonzero weights
};

// ─────────────────────────────────────────────────────────────────────────────
// ScripEgressResult — owns the lightweight 1-based index buffers + cell centers
// ─────────────────────────────────────────────────────────────────────────────

/// Result type returned by scrip_egress(). Owns the Kokkos Views for the
/// 1-based row/col index arrays and computed cell center coordinates, while
/// exposing a ScripEgress view struct for consumer access.
///
/// The caller holds this result alive while the ScripEgress views are in use.
template <class MemorySpace>
struct ScripEgressResult {
    /// The non-owning ScripEgress view struct — valid while this result lives.
    ScripEgress egress{};

    /// Access the egress view struct.
    [[nodiscard]] const ScripEgress& get() const noexcept { return egress; }

    /// Implicit conversion to ScripEgress for ergonomic use.
    [[nodiscard]] operator const ScripEgress&() const noexcept { return egress; }

private:
    template <class MS>
    friend ScripEgressResult<MS> scrip_egress(
        const solver::InterpolationMatrix<MS>& matrix,
        const topology::UnstructuredMesh<MS>& src,
        const topology::UnstructuredMesh<MS>& dst,
        const solver::RegridConfig& config);

    // Owned storage for 1-based index arrays (lightweight nnz allocation)
    Kokkos::View<index_t*, MemorySpace> col_1based_;
    Kokkos::View<index_t*, MemorySpace> row_1based_;

    // Owned storage for computed cell center coordinates
    Kokkos::View<double*, MemorySpace> src_center_lon_;
    Kokkos::View<double*, MemorySpace> src_center_lat_;
    Kokkos::View<double*, MemorySpace> dst_center_lon_;
    Kokkos::View<double*, MemorySpace> dst_center_lat_;
};

// ─────────────────────────────────────────────────────────────────────────────
// String constants for SCRIP method names
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Map InterpolationMethod enum to SCRIP remap_method string.
inline constexpr const char* scrip_remap_method(solver::InterpolationMethod m) noexcept {
    switch (m) {
        case solver::InterpolationMethod::Conservative1stOrder:
        case solver::InterpolationMethod::Conservative2ndOrder:
            return "conservative";
        case solver::InterpolationMethod::Bilinear:
            return "bilinear";
        case solver::InterpolationMethod::NearestNeighbor:
            return "nearest_neighbor";
        case solver::InterpolationMethod::Bicubic:
            return "bicubic";
        case solver::InterpolationMethod::Patch:
            return "patch";
        default:
            return "unknown";
    }
}

/// Map InterpolationMethod enum to SCRIP map_method description.
inline constexpr const char* scrip_map_method(solver::InterpolationMethod m) noexcept {
    switch (m) {
        case solver::InterpolationMethod::Conservative1stOrder:
            return "Conservative remapping";
        case solver::InterpolationMethod::Conservative2ndOrder:
            return "Conservative remapping 2nd order";
        case solver::InterpolationMethod::Bilinear:
            return "Bilinear remapping";
        case solver::InterpolationMethod::NearestNeighbor:
            return "Nearest neighbor remapping";
        case solver::InterpolationMethod::Bicubic:
            return "Bicubic remapping";
        case solver::InterpolationMethod::Patch:
            return "Patch recovery remapping";
        default:
            return "Unknown remapping";
    }
}

/// Map NormType enum to SCRIP norm_option string.
inline constexpr const char* scrip_norm_option(solver::NormType n) noexcept {
    switch (n) {
        case solver::NormType::DstArea:  return "destarea";
        case solver::NormType::FracArea: return "fracarea";
        default:                         return "destarea";
    }
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// scrip_egress() factory function
// ─────────────────────────────────────────────────────────────────────────────

/// Build a SCRIP-convention egress view over an InterpolationMatrix and meshes.
///
/// Creates 1-based row/col index arrays (lightweight allocation of nnz index_t
/// values each) and computes cell center lon/lat from mesh node coordinates.
/// All other fields (S, areas, fractions) are zero-copy views into the
/// existing InterpolationMatrix buffers.
///
/// @param matrix  The interpolation weight matrix
/// @param src     Source UnstructuredMesh (cell areas must be computed)
/// @param dst     Destination UnstructuredMesh (cell areas must be computed)
/// @param config  RegridConfig used to determine method strings
/// @return ScripEgressResult owning the offset buffers; access .egress or .get()
template <class MS>
[[nodiscard]] ScripEgressResult<MS> scrip_egress(
    const solver::InterpolationMatrix<MS>& matrix,
    const topology::UnstructuredMesh<MS>& src,
    const topology::UnstructuredMesh<MS>& dst,
    const solver::RegridConfig& config)
{
    ScripEgressResult<MS> result;
    const auto n_s = matrix.nnz();
    const auto n_a = matrix.n_src();
    const auto n_b = matrix.n_dst();

    // ─── 1-based index arrays (lightweight allocation + offset) ─────────────

    result.col_1based_ = Kokkos::View<index_t*, MS>("scrip_col_1based", n_s);
    result.row_1based_ = Kokkos::View<index_t*, MS>("scrip_row_1based", n_s);

    // Get internal Kokkos Views for the 0-based indices
    const auto& factor_col_view = matrix.factor_col_view();
    const auto& factor_row_view = matrix.factor_row_view();
    auto col_1based = result.col_1based_;
    auto row_1based = result.row_1based_;

    // Apply +1 offset via Kokkos parallel_for
    Kokkos::parallel_for(
        "scrip_egress_offset_indices",
        Kokkos::RangePolicy<typename MS::execution_space>(0, n_s),
        KOKKOS_LAMBDA(const std::size_t k) {
            col_1based(k) = factor_col_view(k) + 1;
            row_1based(k) = factor_row_view(k) + 1;
        });
    Kokkos::fence("scrip_egress_offset_fence");

    // ─── Cell center coordinates from mesh node_coords ──────────────────────
    // UnstructuredMesh stores node_coords as [n_nodes, ndim] in layout_left.
    // For SCRIP grids (structured grids converted to unstructured), each cell
    // has vertices; we compute the centroid lon/lat for each cell from the
    // CSR connectivity + node coordinates.

    // Source mesh centroids
    {
        const auto n_cells = src.n_cells();
        result.src_center_lon_ = Kokkos::View<double*, MS>("scrip_src_clon", n_cells);
        result.src_center_lat_ = Kokkos::View<double*, MS>("scrip_src_clat", n_cells);

        const auto& nodes = src.node_coords_view();
        const auto& offsets = src.conn_offsets_view();
        const auto& indices = src.conn_indices_view();
        auto clon = result.src_center_lon_;
        auto clat = result.src_center_lat_;

        Kokkos::parallel_for(
            "scrip_egress_src_centroids",
            Kokkos::RangePolicy<typename MS::execution_space>(0, n_cells),
            KOKKOS_LAMBDA(const std::size_t c) {
                const auto start = offsets(c);
                const auto end   = offsets(c + 1);
                const auto nverts = end - start;
                double lon_sum = 0.0;
                double lat_sum = 0.0;
                for (auto v = start; v < end; ++v) {
                    const auto node_idx = indices(v);
                    lon_sum += nodes(node_idx, 0);
                    lat_sum += nodes(node_idx, 1);
                }
                clon(c) = lon_sum / static_cast<double>(nverts);
                clat(c) = lat_sum / static_cast<double>(nverts);
            });
    }

    // Destination mesh centroids
    {
        const auto n_cells = dst.n_cells();
        result.dst_center_lon_ = Kokkos::View<double*, MS>("scrip_dst_clon", n_cells);
        result.dst_center_lat_ = Kokkos::View<double*, MS>("scrip_dst_clat", n_cells);

        const auto& nodes = dst.node_coords_view();
        const auto& offsets = dst.conn_offsets_view();
        const auto& indices = dst.conn_indices_view();
        auto clon = result.dst_center_lon_;
        auto clat = result.dst_center_lat_;

        Kokkos::parallel_for(
            "scrip_egress_dst_centroids",
            Kokkos::RangePolicy<typename MS::execution_space>(0, n_cells),
            KOKKOS_LAMBDA(const std::size_t c) {
                const auto start = offsets(c);
                const auto end   = offsets(c + 1);
                const auto nverts = end - start;
                double lon_sum = 0.0;
                double lat_sum = 0.0;
                for (auto v = start; v < end; ++v) {
                    const auto node_idx = indices(v);
                    lon_sum += nodes(node_idx, 0);
                    lat_sum += nodes(node_idx, 1);
                }
                clon(c) = lon_sum / static_cast<double>(nverts);
                clat(c) = lat_sum / static_cast<double>(nverts);
            });
    }

    Kokkos::fence("scrip_egress_centroids_fence");

    // ─── Populate the ScripEgress view struct ───────────────────────────────

    auto& eg = result.egress;

    // Weights — zero-copy view into InterpolationMatrix
    eg.S = matrix.factor_list();

    // 1-based indices — view into owned offset buffers
    eg.col = field_view<const index_t, 1>{result.col_1based_.data(), n_s};
    eg.row = field_view<const index_t, 1>{result.row_1based_.data(), n_s};

    // Cell center coordinates — view into owned centroid buffers
    eg.src_grid_center_lon = field_view<const double, 1>{result.src_center_lon_.data(), n_a};
    eg.src_grid_center_lat = field_view<const double, 1>{result.src_center_lat_.data(), n_a};
    eg.dst_grid_center_lon = field_view<const double, 1>{result.dst_center_lon_.data(), n_b};
    eg.dst_grid_center_lat = field_view<const double, 1>{result.dst_center_lat_.data(), n_b};

    // Areas and fractions — zero-copy view into InterpolationMatrix
    eg.src_grid_area = matrix.area_a();
    eg.dst_grid_area = matrix.area_b();
    eg.src_grid_frac = matrix.frac_a();
    eg.dst_grid_frac = matrix.frac_b();

    // Grid dimensions — unstructured grids report [n_cells, 1]
    eg.src_grid_dims[0] = n_a;
    eg.src_grid_dims[1] = 1;
    eg.dst_grid_dims[0] = n_b;
    eg.dst_grid_dims[1] = 1;

    // Method strings from config
    eg.remap_method = detail::scrip_remap_method(config.method);
    eg.norm_option  = detail::scrip_norm_option(config.norm_type);
    eg.map_method   = detail::scrip_map_method(config.method);

    // Number of nonzero weights
    eg.n_s = n_s;

    return result;
}

} // namespace axis::ingest

#endif // AXIS_INGEST_SCRIP_EGRESS_HPP
