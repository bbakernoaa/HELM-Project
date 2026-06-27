// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TYPES_HPP
#define AXIS_TYPES_HPP

/// @file axis/types.hpp
/// @brief Core type aliases used throughout the AXIS public interface.
///
/// All field and coordinate data crossing the AXIS boundary uses
/// std::mdspan with explicit std::layout_left (column-major / Fortran order)
/// mapping. These aliases state layout and ownership intent up front
/// and satisfy HELM Law #1 (zero-copy non-owning views).

#include <cstddef>
#include <cstdint>
#include <span>

// mdspan: use the C++23 standard header if available, otherwise fall back to
// the kokkos/mdspan reference implementation (provides Kokkos::mdspan et al.).
#if __has_include(<mdspan>) && (__cplusplus > 202002L || defined(__cpp_lib_mdspan))
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
// Bring kokkos/mdspan names into namespace std so the rest of AXIS can use
// std::mdspan, std::dextents, std::layout_left uniformly.
namespace std {
    using Kokkos::mdspan;
    using Kokkos::dextents;
    using Kokkos::extents;
    using Kokkos::layout_left;
    using Kokkos::layout_right;
    using Kokkos::layout_stride;
#ifndef __cpp_lib_span
    using Kokkos::dynamic_extent;
#endif
    using Kokkos::default_accessor;
}
#endif

/// @namespace axis
/// @brief Root namespace for the Arbitrary eXgrid Interpolation Solver (AXIS).
namespace axis {

// ─────────────────────────────────────────────────────────────────────────────
// field_view: the non-owning, column-major view type AXIS uses at all
// boundaries. T is the element type, Rank is the number of dimensions.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Helper template to generate Kokkos dynamic extents based on rank.
/// @tparam IndexType The index type used to define the extent bounds.
/// @tparam Rank The dimensionality (rank) of the span, supporting values 1, 2, 3, or 4.
template <typename IndexType, std::size_t Rank>
using my_dextents = std::conditional_t<Rank == 1,
    Kokkos::extents<IndexType, std::dynamic_extent>,
    std::conditional_t<Rank == 2,
        Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent>,
        std::conditional_t<Rank == 3,
            Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
            Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>
        >
    >
>;

/// @brief Column-major (Fortran-layout) non-owning multidimensional span view.
///
/// This is the canonical layout AXIS accepts at its public boundary, enabling
/// zero-copy interoperability with Fortran scientific models and high-performance layouts.
///
/// @tparam T The element type stored or accessed through the span (e.g. @c double or @c const @c double).
/// @tparam Rank The dimensionality of the field (number of dynamic dimensions).
template <class T, std::size_t Rank>
using field_view = Kokkos::mdspan<T, my_dextents<std::size_t, Rank>, Kokkos::layout_left>;

// ─────────────────────────────────────────────────────────────────────────────
// Common rank shortcuts
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Rank-1 double view: [n_cells] — unstructured fields, 1-D coordinate arrays.
using field1d = field_view<double, 1>;

/// @brief Rank-2 double view: [ni, nj] — structured fields (i fastest, column-major).
using field2d = field_view<double, 2>;

// ─────────────────────────────────────────────────────────────────────────────
// Index type
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Signed 64-bit index type used across solver matrices and connectivity tables.
///
/// Supports global indices in distributed MPI configurations, and allows negative
/// sentinel values (such as -1) to signify invalid or unmapped elements.
using index_t = std::int64_t;

} // namespace axis

#endif // AXIS_TYPES_HPP
