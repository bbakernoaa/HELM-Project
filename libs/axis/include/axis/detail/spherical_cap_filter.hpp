// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_SPHERICAL_CAP_FILTER_HPP
#define AXIS_DETAIL_SPHERICAL_CAP_FILTER_HPP

/// @file axis/detail/spherical_cap_filter.hpp
/// @brief Spherical cap early-exit filter for conservative remapping.
///
/// Provides:
///   - compute_cell_centroid_xyz(): mean of unit-sphere vertex positions,
///     re-normalized to unit length.
///   - compute_angular_radius(): maximum angular distance from centroid to
///     any vertex of the cell.
///   - spherical_cap_rejects(): returns true when two spherical caps are
///     guaranteed non-overlapping (angular distance between centroids exceeds
///     sum of angular radii).
///   - precompute_cap_data(): batch precomputation of centroids and angular
///     radii for all cells in a mesh, returned as Kokkos Views.
///
/// The filter is conservative: it never rejects a pair that actually overlaps.
/// All functions annotated KOKKOS_FUNCTION for device portability.
/// No heap allocation — safe for Kokkos parallel kernels (HELM Law #2).

#include <Kokkos_Core.hpp>

#include <axis/topology/unstructured_mesh.hpp>
#include "memory_traits.hpp"
#include "spherical_clipper.hpp"  // Vec3, dot, normalize, etc.

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

namespace cap_detail {
    inline constexpr double pi = 3.14159265358979323846;
    inline constexpr double deg2rad = pi / 180.0;
}  // namespace cap_detail

// ─────────────────────────────────────────────────────────────────────────────
// CapData — pre-computed centroid + angular radius arrays for a mesh
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Pre-computed spherical cap data for all cells in a mesh.
///
/// Each cell is bounded by a spherical cap centered at its centroid with
/// angular radius equal to the maximum arc-distance from centroid to any vertex.
template <class MemorySpace>
struct CapData {
    Kokkos::View<Vec3*, MemorySpace>   centroids;      ///< Unit-sphere centroids [n_cells]
    Kokkos::View<double*, MemorySpace> angular_radii;  ///< Angular radii in radians [n_cells]
};

// ─────────────────────────────────────────────────────────────────────────────
// lonlat_to_xyz_device — convert lon/lat to unit-sphere XYZ (device-portable)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Convert (lon, lat) in radians to a unit-sphere Cartesian point.
/// @note Device-portable version using Kokkos math functions.
KOKKOS_INLINE_FUNCTION
Vec3 lonlat_to_xyz_device(double lon_rad, double lat_rad) noexcept {
    double cos_lat = Kokkos::cos(lat_rad);
    return Vec3{cos_lat * Kokkos::cos(lon_rad),
                cos_lat * Kokkos::sin(lon_rad),
                Kokkos::sin(lat_rad)};
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_cell_centroid_xyz — centroid as normalized mean of unit-sphere vertices
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute the unit-sphere centroid of a cell.
///
/// Algorithm: convert each vertex from lon/lat to unit-sphere XYZ, compute
/// the mean of all vertex positions, then re-normalize to unit length.
/// The coordinate system must be SphericalDeg or SphericalRad.
///
/// @param node_coords    Node coordinates [n_nodes, 2] (lon, lat)
/// @param conn_offsets   CSR offsets [n_cells + 1]
/// @param conn_indices   CSR node indices [nnz]
/// @param cell_idx       Index of the cell to compute centroid for
/// @param is_degrees     True if coordinates are in degrees, false if radians
/// @return Unit-sphere centroid (normalized mean of vertex XYZ positions)
KOKKOS_INLINE_FUNCTION
Vec3 compute_cell_centroid_xyz(
    const double* node_coords_ptr,
    const std::size_t n_nodes,
    const axis::index_t* conn_offsets_ptr,
    const axis::index_t* conn_indices_ptr,
    const std::size_t cell_idx,
    bool is_degrees) noexcept {

    const axis::index_t begin = conn_offsets_ptr[cell_idx];
    const axis::index_t end   = conn_offsets_ptr[cell_idx + 1];
    const int n_verts = static_cast<int>(end - begin);

    if (n_verts < 1) return Vec3{0.0, 0.0, 0.0};

    // Accumulate mean of unit-sphere vertex positions
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;

    for (int v = 0; v < n_verts; ++v) {
        const axis::index_t node_idx = conn_indices_ptr[begin + v];
        // node_coords is layout_left: [n_nodes, 2]
        // lon = node_coords(node_idx, 0) = ptr[node_idx]
        // lat = node_coords(node_idx, 1) = ptr[node_idx + n_nodes]
        double lon = node_coords_ptr[node_idx];
        double lat = node_coords_ptr[node_idx + static_cast<axis::index_t>(n_nodes)];

        if (is_degrees) {
            lon *= cap_detail::deg2rad;
            lat *= cap_detail::deg2rad;
        }

        Vec3 xyz = lonlat_to_xyz_device(lon, lat);
        sum_x += xyz.x;
        sum_y += xyz.y;
        sum_z += xyz.z;
    }

    // Mean and re-normalize
    double inv_n = 1.0 / static_cast<double>(n_verts);
    Vec3 mean{sum_x * inv_n, sum_y * inv_n, sum_z * inv_n};
    return normalize(mean);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_angular_radius — max angular distance from centroid to any vertex
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute the angular radius of a cell's bounding spherical cap.
///
/// The angular radius is the maximum great-circle arc distance from the
/// centroid to any vertex of the cell: max_i(acos(clamp(dot(centroid, v_i), -1, 1)))
///
/// @param centroid          Unit-sphere centroid of the cell
/// @param node_coords_ptr   Pointer to node coordinates [n_nodes, 2] (layout_left)
/// @param n_nodes           Number of nodes in the mesh
/// @param conn_offsets_ptr  CSR offsets
/// @param conn_indices_ptr  CSR node indices
/// @param cell_idx          Cell index
/// @param is_degrees        True if coordinates are in degrees
/// @return Angular radius in radians
KOKKOS_INLINE_FUNCTION
double compute_angular_radius(
    const Vec3& centroid,
    const double* node_coords_ptr,
    const std::size_t n_nodes,
    const axis::index_t* conn_offsets_ptr,
    const axis::index_t* conn_indices_ptr,
    const std::size_t cell_idx,
    bool is_degrees) noexcept {

    const axis::index_t begin = conn_offsets_ptr[cell_idx];
    const axis::index_t end   = conn_offsets_ptr[cell_idx + 1];
    const int n_verts = static_cast<int>(end - begin);

    double max_angle = 0.0;

    for (int v = 0; v < n_verts; ++v) {
        const axis::index_t node_idx = conn_indices_ptr[begin + v];
        double lon = node_coords_ptr[node_idx];
        double lat = node_coords_ptr[node_idx + static_cast<axis::index_t>(n_nodes)];

        if (is_degrees) {
            lon *= cap_detail::deg2rad;
            lat *= cap_detail::deg2rad;
        }

        Vec3 vertex = lonlat_to_xyz_device(lon, lat);

        // Angular distance = acos(clamp(dot(centroid, vertex), -1, 1))
        double d = dot(centroid, vertex);
        d = Kokkos::fmin(1.0, Kokkos::fmax(-1.0, d));
        double angle = Kokkos::acos(d);

        if (angle > max_angle) {
            max_angle = angle;
        }
    }

    return max_angle;
}

// ─────────────────────────────────────────────────────────────────────────────
// spherical_cap_rejects — early-exit filter for source-destination cell pairs
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Determine if two spherical caps are guaranteed non-overlapping.
///
/// Returns true when the angular distance between the two centroids exceeds
/// the sum of their angular radii, guaranteeing zero geometric overlap.
/// This is a conservative test: it never rejects a pair that actually overlaps.
///
/// @param centroid_s  Unit-sphere centroid of the source cell
/// @param radius_s    Angular radius of the source cell (radians)
/// @param centroid_d  Unit-sphere centroid of the destination cell
/// @param radius_d    Angular radius of the destination cell (radians)
/// @return true if the pair can be safely skipped (caps are disjoint)
KOKKOS_INLINE_FUNCTION
bool spherical_cap_rejects(const Vec3& centroid_s, double radius_s,
                           const Vec3& centroid_d, double radius_d) noexcept {
    double cos_dist = dot(centroid_s, centroid_d);
    // Clamp for numerical safety
    cos_dist = Kokkos::fmin(1.0, Kokkos::fmax(-1.0, cos_dist));
    double angular_dist = Kokkos::acos(cos_dist);
    return angular_dist > (radius_s + radius_d);
}

// ─────────────────────────────────────────────────────────────────────────────
// precompute_cap_data — batch computation of centroids and angular radii
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Pre-compute spherical cap data (centroids + angular radii) for all
///        cells in a mesh.
///
/// Runs a Kokkos parallel_for over all cells to compute the unit-sphere
/// centroid and angular radius for each cell. Results are returned as Views
/// suitable for use in the overlap loop's early-exit filter.
///
/// @tparam MemorySpace  Kokkos memory space for output Views.
/// @param mesh          The unstructured mesh to precompute cap data for.
/// @return CapData containing Views of centroids [n_cells] and radii [n_cells].
template <class MemorySpace>
CapData<MemorySpace> precompute_cap_data(
    const axis::topology::UnstructuredMesh<MemorySpace>& mesh) {

    using exec_space = exec_space_t<MemorySpace>;

    const std::size_t n_cells = mesh.n_cells();
    const std::size_t n_nodes = mesh.n_nodes();

    // Determine if coordinates are in degrees
    const bool is_degrees =
        (mesh.coord_system() == axis::topology::CoordinateSystem::SphericalDeg);

    // Allocate output views
    Kokkos::View<Vec3*, MemorySpace> centroids("cap_centroids", n_cells);
    Kokkos::View<double*, MemorySpace> angular_radii("cap_angular_radii", n_cells);

    // Get raw pointers for device-portable access
    const auto& node_coords_view = mesh.node_coords_view();
    const auto& offsets_view = mesh.conn_offsets_view();
    const auto& indices_view = mesh.conn_indices_view();

    // Compute centroids and angular radii in parallel
    Kokkos::parallel_for("precompute_cap_data",
        Kokkos::RangePolicy<exec_space>(0, n_cells),
        KOKKOS_LAMBDA(const std::size_t cell_idx) {
            // Compute centroid
            Vec3 centroid = compute_cell_centroid_xyz(
                node_coords_view.data(),
                n_nodes,
                offsets_view.data(),
                indices_view.data(),
                cell_idx,
                is_degrees);

            centroids(cell_idx) = centroid;

            // Compute angular radius
            double radius = compute_angular_radius(
                centroid,
                node_coords_view.data(),
                n_nodes,
                offsets_view.data(),
                indices_view.data(),
                cell_idx,
                is_degrees);

            angular_radii(cell_idx) = radius;
        });

    Kokkos::fence("precompute_cap_data_fence");

    return CapData<MemorySpace>{centroids, angular_radii};
}

}  // namespace axis::detail

#endif  // AXIS_DETAIL_SPHERICAL_CAP_FILTER_HPP
