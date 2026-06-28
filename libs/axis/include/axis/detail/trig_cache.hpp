// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_TRIG_CACHE_HPP
#define AXIS_DETAIL_TRIG_CACHE_HPP

/// @file axis/detail/trig_cache.hpp
/// @brief Pre-computed trigonometric cache for regular lat-lon grids.
///
/// Provides:
///   - TrigCache<MemorySpace>: struct holding sin/cos arrays for all unique
///     longitude and latitude cell centers in a regular grid.
///   - build_trig_cache(): fills the cache via Kokkos::parallel_for over ni and
///     nj, computing sin/cos at cell centers (lon_min + (i + 0.5) * delta_lon).
///   - lonlat_to_xyz_cached(): KOKKOS_INLINE_FUNCTION helper for O(1) index-based
///     lon/lat-to-XYZ conversion using cached trig values.
///
/// Replaces per-vertex sin()/cos() calls with table lookups in the overlap loop,
/// yielding significant speedup for regular-grid conservative remapping.
///
/// All kernels annotated KOKKOS_FUNCTION / KOKKOS_INLINE_FUNCTION for device
/// portability (HELM Law #2). No heap allocation in device code paths.

#include <Kokkos_Core.hpp>
#include <cstddef>

#include "memory_traits.hpp"
#include "regular_grid_detector.hpp"

namespace axis::detail {

// Portable pi constant (M_PI is POSIX-only, not guaranteed in C++20 strict mode).
inline constexpr double trig_cache_pi = 3.14159265358979323846;
inline constexpr double trig_cache_deg2rad = trig_cache_pi / 180.0;

// ─────────────────────────────────────────────────────────────────────────────
// TrigCache<MemorySpace> — pre-computed sin/cos for regular lat-lon grids
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Pre-computed sin/cos values for all unique lon/lat cell centers
///        in a regular lat-lon grid.
///
/// Enables O(1) lookup replacing per-vertex trigonometric function calls.
/// The arrays are stored as Kokkos::Views in the specified MemorySpace for
/// device-portable access.
///
/// @tparam MemorySpace Kokkos memory space (e.g., HostSpace, CudaSpace).
template <class MemorySpace>
struct TrigCache {
    Kokkos::View<double *, MemorySpace> sin_lon;  ///< sin(lon_center) for [0, ni)
    Kokkos::View<double *, MemorySpace> cos_lon;  ///< cos(lon_center) for [0, ni)
    Kokkos::View<double *, MemorySpace> sin_lat;  ///< sin(lat_center) for [0, nj)
    Kokkos::View<double *, MemorySpace> cos_lat;  ///< cos(lat_center) for [0, nj)

    std::size_t ni{0};      ///< Number of longitude cells
    std::size_t nj{0};      ///< Number of latitude cells
    double lon_min{0.0};    ///< Longitude grid origin (degrees)
    double delta_lon{0.0};  ///< Longitude cell width (degrees)
    double lat_min{0.0};    ///< Latitude grid origin (degrees)
    double delta_lat{0.0};  ///< Latitude cell width (degrees)

    bool valid{false};  ///< True if cache was successfully built
};

// ─────────────────────────────────────────────────────────────────────────────
// build_trig_cache — fills sin/cos arrays via Kokkos::parallel_for
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Build a trigonometric cache for a detected regular grid.
///
/// Allocates Kokkos Views for sin_lon, cos_lon, sin_lat, cos_lat and fills
/// them via parallel_for kernels. Each entry corresponds to the cell center
/// coordinate: lon_center(i) = (lon_min + (i + 0.5) * delta_lon) * deg2rad.
///
/// @tparam MemorySpace Kokkos memory space for the cache Views.
/// @param info RegularGridInfo from detect_regular_grid(). If info.is_regular
///             is false, returns a cache with valid=false.
/// @return TrigCache with valid=true if info is regular, valid=false otherwise.
template <class MemorySpace>
TrigCache<MemorySpace> build_trig_cache(const RegularGridInfo &info) {
    using exec_space = exec_space_t<MemorySpace>;

    TrigCache<MemorySpace> cache;

    if (!info.is_regular || info.ni == 0 || info.nj == 0) {
        return cache;  // valid remains false
    }

    cache.ni = info.ni;
    cache.nj = info.nj;
    cache.lon_min = info.lon_min;
    cache.delta_lon = info.delta_lon;
    cache.lat_min = info.lat_min;
    cache.delta_lat = info.delta_lat;

    // Allocate Views
    cache.sin_lon = Kokkos::View<double *, MemorySpace>("trig_cache_sin_lon", info.ni);
    cache.cos_lon = Kokkos::View<double *, MemorySpace>("trig_cache_cos_lon", info.ni);
    cache.sin_lat = Kokkos::View<double *, MemorySpace>("trig_cache_sin_lat", info.nj);
    cache.cos_lat = Kokkos::View<double *, MemorySpace>("trig_cache_cos_lat", info.nj);

    // Capture grid parameters for lambda (avoid capturing 'cache' which holds Views)
    const double lon_min = info.lon_min;
    const double delta_lon = info.delta_lon;
    const double lat_min = info.lat_min;
    const double delta_lat = info.delta_lat;

    auto sin_lon_v = cache.sin_lon;
    auto cos_lon_v = cache.cos_lon;
    auto sin_lat_v = cache.sin_lat;
    auto cos_lat_v = cache.cos_lat;

    // Fill longitude cache: cell centers at (lon_min + (i + 0.5) * delta_lon)
    Kokkos::parallel_for(
        "fill_trig_cache_lon", Kokkos::RangePolicy<exec_space>(0, info.ni), KOKKOS_LAMBDA(const std::size_t i) {
            double lon_rad = (lon_min + (static_cast<double>(i) + 0.5) * delta_lon) * trig_cache_deg2rad;
            sin_lon_v(i) = Kokkos::sin(lon_rad);
            cos_lon_v(i) = Kokkos::cos(lon_rad);
        });

    // Fill latitude cache: cell centers at (lat_min + (j + 0.5) * delta_lat)
    Kokkos::parallel_for(
        "fill_trig_cache_lat", Kokkos::RangePolicy<exec_space>(0, info.nj), KOKKOS_LAMBDA(const std::size_t j) {
            double lat_rad = (lat_min + (static_cast<double>(j) + 0.5) * delta_lat) * trig_cache_deg2rad;
            sin_lat_v(j) = Kokkos::sin(lat_rad);
            cos_lat_v(j) = Kokkos::cos(lat_rad);
        });

    Kokkos::fence("build_trig_cache");

    cache.valid = true;
    return cache;
}

// ─────────────────────────────────────────────────────────────────────────────
// lonlat_to_xyz_cached — device-portable index-based XYZ lookup
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Device-portable 3-component XYZ result from cached trig lookup.
///
/// Matches the layout of axis::detail::spherical::Vec3 but is self-contained
/// here to avoid coupling to spherical_geometry.hpp in device kernels.
struct CachedVec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

/// @brief Convert grid cell (i, j) to unit-sphere XYZ via cached trig values.
///
/// Returns (cos_lat[j]*cos_lon[i], cos_lat[j]*sin_lon[i], sin_lat[j]).
/// Equivalent to lonlat_to_xyz(lon_center(i), lat_center(j)) but with O(1)
/// lookups instead of transcendental function calls.
///
/// @tparam MemorySpace Kokkos memory space of the TrigCache Views.
/// @param cache  A valid TrigCache (cache.valid must be true).
/// @param i      Longitude index in [0, cache.ni).
/// @param j      Latitude index in [0, cache.nj).
/// @return CachedVec3 with unit-sphere Cartesian coordinates.
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION CachedVec3 lonlat_to_xyz_cached(const TrigCache<MemorySpace> &cache, std::size_t i, std::size_t j) noexcept {
    double cos_lat = cache.cos_lat(j);
    return CachedVec3{cos_lat * cache.cos_lon(i), cos_lat * cache.sin_lon(i), cache.sin_lat(j)};
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeTrigCache<MemorySpace> — pre-computed sin/cos for regular grid node positions
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Pre-computed sin/cos values for all unique lon/lat *node* positions
///        in a regular lat-lon grid (ni+1 lons, nj+1 lats at cell edges).
///
/// Used for vertex-level coordinate conversion in the overlap loop. Each cell
/// vertex corresponds to a node at `lon_min + node_i * delta_lon` (node_i in
/// [0, ni]) or `lat_min + node_j * delta_lat` (node_j in [0, nj]).
///
/// @tparam MemorySpace Kokkos memory space (e.g., HostSpace, CudaSpace).
template <class MemorySpace>
struct NodeTrigCache {
    Kokkos::View<double *, MemorySpace> sin_lon;  ///< sin(lon_node) for [0, ni+1)
    Kokkos::View<double *, MemorySpace> cos_lon;  ///< cos(lon_node) for [0, ni+1)
    Kokkos::View<double *, MemorySpace> sin_lat;  ///< sin(lat_node) for [0, nj+1)
    Kokkos::View<double *, MemorySpace> cos_lat;  ///< cos(lat_node) for [0, nj+1)

    std::size_t ni{0};      ///< Number of longitude cells
    std::size_t nj{0};      ///< Number of latitude cells
    double lon_min{0.0};    ///< Longitude grid origin (degrees)
    double delta_lon{0.0};  ///< Longitude cell width (degrees)
    double lat_min{0.0};    ///< Latitude grid origin (degrees)
    double delta_lat{0.0};  ///< Latitude cell width (degrees)

    bool valid{false};  ///< True if cache was successfully built
};

// ─────────────────────────────────────────────────────────────────────────────
// build_node_trig_cache — fills sin/cos arrays at node (vertex) positions
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Build a trigonometric cache at node (vertex) positions for a regular grid.
///
/// For a regular grid with ni cells in lon and nj cells in lat, there are
/// ni+1 unique node longitudes and nj+1 unique node latitudes. Each entry
/// corresponds to a node position: lon_node(i) = (lon_min + i * delta_lon) * deg2rad.
///
/// @tparam MemorySpace Kokkos memory space for the cache Views.
/// @param info RegularGridInfo from detect_regular_grid(). If info.is_regular
///             is false, returns a cache with valid=false.
/// @return NodeTrigCache with valid=true if info is regular, valid=false otherwise.
template <class MemorySpace>
NodeTrigCache<MemorySpace> build_node_trig_cache(const RegularGridInfo &info) {
    using exec_space = exec_space_t<MemorySpace>;

    NodeTrigCache<MemorySpace> cache;

    if (!info.is_regular || info.ni == 0 || info.nj == 0) {
        return cache;  // valid remains false
    }

    cache.ni = info.ni;
    cache.nj = info.nj;
    cache.lon_min = info.lon_min;
    cache.delta_lon = info.delta_lon;
    cache.lat_min = info.lat_min;
    cache.delta_lat = info.delta_lat;

    const std::size_t n_lon_nodes = info.ni + 1;
    const std::size_t n_lat_nodes = info.nj + 1;

    // Allocate Views
    cache.sin_lon = Kokkos::View<double *, MemorySpace>("node_trig_sin_lon", n_lon_nodes);
    cache.cos_lon = Kokkos::View<double *, MemorySpace>("node_trig_cos_lon", n_lon_nodes);
    cache.sin_lat = Kokkos::View<double *, MemorySpace>("node_trig_sin_lat", n_lat_nodes);
    cache.cos_lat = Kokkos::View<double *, MemorySpace>("node_trig_cos_lat", n_lat_nodes);

    // Capture grid parameters for lambda
    const double lon_min = info.lon_min;
    const double delta_lon = info.delta_lon;
    const double lat_min = info.lat_min;
    const double delta_lat = info.delta_lat;

    auto sin_lon_v = cache.sin_lon;
    auto cos_lon_v = cache.cos_lon;
    auto sin_lat_v = cache.sin_lat;
    auto cos_lat_v = cache.cos_lat;

    // Fill longitude node cache: nodes at (lon_min + i * delta_lon)
    Kokkos::parallel_for(
        "fill_node_trig_cache_lon", Kokkos::RangePolicy<exec_space>(0, n_lon_nodes), KOKKOS_LAMBDA(const std::size_t i) {
            double lon_rad = (lon_min + static_cast<double>(i) * delta_lon) * trig_cache_deg2rad;
            sin_lon_v(i) = Kokkos::sin(lon_rad);
            cos_lon_v(i) = Kokkos::cos(lon_rad);
        });

    // Fill latitude node cache: nodes at (lat_min + j * delta_lat)
    Kokkos::parallel_for(
        "fill_node_trig_cache_lat", Kokkos::RangePolicy<exec_space>(0, n_lat_nodes), KOKKOS_LAMBDA(const std::size_t j) {
            double lat_rad = (lat_min + static_cast<double>(j) * delta_lat) * trig_cache_deg2rad;
            sin_lat_v(j) = Kokkos::sin(lat_rad);
            cos_lat_v(j) = Kokkos::cos(lat_rad);
        });

    Kokkos::fence("build_node_trig_cache");

    cache.valid = true;
    return cache;
}

// ─────────────────────────────────────────────────────────────────────────────
// lonlat_to_xyz_node_cached — vertex-level XYZ lookup from node indices
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Convert grid node (lon_idx, lat_idx) to unit-sphere XYZ via cached values.
///
/// Returns (cos_lat[lat_idx]*cos_lon[lon_idx], cos_lat[lat_idx]*sin_lon[lon_idx],
///          sin_lat[lat_idx]).
/// Equivalent to lonlat_to_xyz(lon_node, lat_node) but with O(1) lookups.
///
/// @tparam MemorySpace Kokkos memory space of the NodeTrigCache Views.
/// @param cache    A valid NodeTrigCache (cache.valid must be true).
/// @param lon_idx  Longitude node index in [0, cache.ni].
/// @param lat_idx  Latitude node index in [0, cache.nj].
/// @return CachedVec3 with unit-sphere Cartesian coordinates.
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION CachedVec3 lonlat_to_xyz_node_cached(const NodeTrigCache<MemorySpace> &cache, std::size_t lon_idx,
                                                            std::size_t lat_idx) noexcept {
    double cos_lat = cache.cos_lat(lat_idx);
    return CachedVec3{cos_lat * cache.cos_lon(lon_idx), cos_lat * cache.sin_lon(lon_idx), cache.sin_lat(lat_idx)};
}

}  // namespace axis::detail

#endif  // AXIS_DETAIL_TRIG_CACHE_HPP
