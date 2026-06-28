// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_MORTON_SORT_HPP
#define AXIS_DETAIL_MORTON_SORT_HPP

/// @file axis/detail/morton_sort.hpp
/// @brief Morton/Z-curve encoding and locality-sorted destination permutation.
///
/// Provides:
///   - morton_encode_2d(): bit-interleave two 32-bit coordinates into a 64-bit
///     Morton code (Z-curve index) preserving 2D spatial locality.
///   - normalize_coord(): map a floating-point coordinate to [0, 2^16) integer
///     range for Morton encoding.
///   - morton_sort_indices(): compute Morton codes from centroid arrays and
///     return a permutation View sorting indices by Z-curve order.
///
/// All functions are annotated KOKKOS_INLINE_FUNCTION / KOKKOS_FUNCTION for
/// device portability (HELM Law #2). No heap allocation in device kernels.

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <cstdint>

#include "memory_traits.hpp"

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// morton_encode_2d — bit-interleave two 32-bit integers into 64-bit Morton code
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Interleave bits of two 32-bit integers into a 64-bit Morton code.
///
/// The resulting code places bits of x in even positions (0, 2, 4, ...)
/// and bits of y in odd positions (1, 3, 5, ...), producing a Z-curve
/// index that preserves 2D spatial locality.
///
/// @param x  First coordinate (placed in even bit positions).
/// @param y  Second coordinate (placed in odd bit positions).
/// @return 64-bit Morton code with interleaved bits.
KOKKOS_INLINE_FUNCTION
uint64_t morton_encode_2d(uint32_t x, uint32_t y) noexcept {
    // Spread bits of a 32-bit value: insert a 0 between each bit.
    // After spreading, bits occupy only even positions in the 64-bit result.
    auto spread = [](uint32_t v) -> uint64_t {
        uint64_t r = v;
        r = (r | (r << 16)) & 0x0000FFFF0000FFFF;
        r = (r | (r << 8)) & 0x00FF00FF00FF00FF;
        r = (r | (r << 4)) & 0x0F0F0F0F0F0F0F0F;
        r = (r | (r << 2)) & 0x3333333333333333;
        r = (r | (r << 1)) & 0x5555555555555555;
        return r;
    };
    return spread(x) | (spread(y) << 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// normalize_coord — map a floating-point value to [0, 2^16) integer range
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Normalize a coordinate value to a 16-bit unsigned integer range.
///
/// Maps val from [min_val, max_val] to [0, 65535]. Values outside the range
/// are clamped. If max_val <= min_val (degenerate range), returns 0.
///
/// @param val      The coordinate value to normalize.
/// @param min_val  Minimum of the coordinate range.
/// @param max_val  Maximum of the coordinate range.
/// @return Normalized integer in [0, 65535].
KOKKOS_INLINE_FUNCTION
uint32_t normalize_coord(double val, double min_val, double max_val) noexcept {
    constexpr uint32_t N = (1u << 16) - 1;  // 65535
    if (max_val <= min_val) return 0;
    double t = (val - min_val) / (max_val - min_val);
    t = Kokkos::fmax(0.0, Kokkos::fmin(1.0, t));
    return static_cast<uint32_t>(t * N);
}

// ─────────────────────────────────────────────────────────────────────────────
// morton_sort_indices — compute Morton codes and return sorted permutation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Compute Morton codes from destination cell centroids and return a
///        permutation View that sorts indices by Z-curve order.
///
/// Given arrays of centroid x (longitude) and y (latitude) coordinates,
/// normalizes each to [0, 2^16), computes 64-bit Morton codes, and produces
/// a sorted permutation such that sorted_dst(k) gives the original index
/// of the k-th destination cell in Morton/Z-curve order.
///
/// @tparam MemorySpace  Kokkos memory space for Views.
/// @param centroids_x   View of destination centroid x-coordinates [n_dst].
/// @param centroids_y   View of destination centroid y-coordinates [n_dst].
/// @param n_dst         Number of destination cells.
/// @return View<index_t*, MemorySpace> containing the Morton-sorted permutation.
template <class MemorySpace>
Kokkos::View<int64_t *, MemorySpace> morton_sort_indices(const Kokkos::View<const double *, MemorySpace> &centroids_x,
                                                         const Kokkos::View<const double *, MemorySpace> &centroids_y, const std::size_t n_dst) {
    using exec_space = exec_space_t<MemorySpace>;
    using index_t = int64_t;

    // Handle degenerate case
    Kokkos::View<index_t *, MemorySpace> sorted_dst("sorted_dst", n_dst);
    if (n_dst == 0) return sorted_dst;

    // ── Step 1: Find bounding box of centroids ──

    double lon_min = 0.0, lon_max = 0.0;
    double lat_min = 0.0, lat_max = 0.0;

    Kokkos::parallel_reduce(
        "morton_lon_minmax", Kokkos::RangePolicy<exec_space>(0, n_dst),
        KOKKOS_LAMBDA(const std::size_t j, double &lmin, double &lmax) {
            double v = centroids_x(j);
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        },
        Kokkos::Min<double>(lon_min), Kokkos::Max<double>(lon_max));

    Kokkos::parallel_reduce(
        "morton_lat_minmax", Kokkos::RangePolicy<exec_space>(0, n_dst),
        KOKKOS_LAMBDA(const std::size_t j, double &lmin, double &lmax) {
            double v = centroids_y(j);
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        },
        Kokkos::Min<double>(lat_min), Kokkos::Max<double>(lat_max));

    // ── Step 2: Compute Morton codes ──

    Kokkos::View<uint64_t *, MemorySpace> morton_keys("morton_keys", n_dst);

    Kokkos::parallel_for(
        "compute_morton", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) {
            uint32_t ix = normalize_coord(centroids_x(j), lon_min, lon_max);
            uint32_t iy = normalize_coord(centroids_y(j), lat_min, lat_max);
            morton_keys(j) = morton_encode_2d(ix, iy);
        });

    // ── Step 3: Initialize identity permutation ──

    Kokkos::parallel_for(
        "init_perm", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) { sorted_dst(j) = static_cast<index_t>(j); });

    // ── Step 4: Sort permutation by Morton key ──
    // Use a simple selection approach: create a key-value pair view and sort.
    // We use Kokkos::sort with a custom comparator via a BinSort approach,
    // or we can sort the keys and apply the permutation.

    // Strategy: sort (morton_keys, sorted_dst) together by morton_keys.
    // Kokkos::sort_by_key is available in newer Kokkos versions.
    // For portability, we use a simple approach: copy keys into a sortable
    // structure and produce the permutation on host, or use BinSort.

    // Use Kokkos BinSort for device-portable key-value sorting.
    using KeyViewType = Kokkos::View<uint64_t *, MemorySpace>;

    // Create a combined key-index view for sorting
    // We'll use a simple approach: sort on host for now (morton sort is a
    // pre-processing step before the overlap loop, not in the hot path).

    // Mirror to host for sorting
    auto keys_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, morton_keys);
    auto perm_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sorted_dst);

    // Simple insertion sort for small n, or std::sort for larger
    // Use indices array sorted by key comparison
    std::sort(perm_host.data(), perm_host.data() + n_dst, [&keys_host](index_t a, index_t b) { return keys_host(a) < keys_host(b); });

    // Copy sorted permutation back to device
    Kokkos::deep_copy(sorted_dst, perm_host);

    return sorted_dst;
}

}  // namespace axis::detail

#endif  // AXIS_DETAIL_MORTON_SORT_HPP
