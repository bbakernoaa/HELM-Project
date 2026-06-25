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

namespace axis {

// ─────────────────────────────────────────────────────────────────────────────
// field_view: the non-owning, column-major view type AXIS uses at all
// boundaries. T is the element type, Rank is the number of dimensions.
// ─────────────────────────────────────────────────────────────────────────────

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

/// Column-major (Fortran) non-owning view over a contiguous array.
/// This is the ONLY layout AXIS accepts at its public boundary.
template <class T, std::size_t Rank>
using field_view = Kokkos::mdspan<T, my_dextents<std::size_t, Rank>, Kokkos::layout_left>;

// ─────────────────────────────────────────────────────────────────────────────
// Common rank shortcuts
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-1 double view: [n_cells] — unstructured fields, 1-D coordinate arrays.
using field1d = field_view<double, 1>;

/// Rank-2 double view: [ni, nj] — structured fields (i fastest, column-major).
using field2d = field_view<double, 2>;

// ─────────────────────────────────────────────────────────────────────────────
// Index type
// ─────────────────────────────────────────────────────────────────────────────

/// Index type used throughout the sparse matrix and CSR connectivity tables.
/// Signed 64-bit to support global indices in distributed mode and sentinel
/// values (e.g., -1 for invalid/unmapped).
using index_t = std::int64_t;

} // namespace axis

#endif // AXIS_TYPES_HPP
