// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_INTERPOLATION_MATRIX_HPP
#define AXIS_SOLVER_INTERPOLATION_MATRIX_HPP

/// @file axis/solver/interpolation_matrix.hpp
/// @brief Sparse interpolation operator in COO form (factorList + factorIndexList).
///
/// InterpolationMatrix stores the sparse weights and associated conservation
/// bookkeeping arrays (frac_a, frac_b, area_a, area_b) produced by
/// WeightGenerator. This is the exact data layout that ESMF weight files
/// express: factorList (S), factorIndexList col/row (src/dst), plus per-cell
/// fraction and area arrays for conservative normalization.
///
/// Templated on a Kokkos MemorySpace (HELM Law #2: explicit placement, no UVM).
/// Light inline accessors live in this header; the .cpp provides explicit
/// template instantiations for common memory spaces.

#include <cstddef>

#include <Kokkos_Core.hpp>

#include <axis/types.hpp>

namespace axis::solver {

/// A (destination, source) index pair representing one nonzero entry in the
/// sparse interpolation matrix. Matches ESMF factorIndexList semantics where
/// col = source cell index and row = destination cell index.
struct IndexPair {
    index_t row;  ///< destination cell index
    index_t col;  ///< source cell index
};

/// Sparse interpolation operator dst = S · src in COO form.
///
/// For each nonzero k:
///   factor_list[k] = S(k)         (weight of the k-th nonzero)
///   factor_row[k]  = dst_index    (destination cell)
///   factor_col[k]  = src_index    (source cell)
///
/// Conservation bookkeeping arrays (populated for conservative methods):
///   frac_a[i] = fraction of source cell i covered by destination cells
///   frac_b[j] = fraction of destination cell j covered by source cells
///   area_a[i] = area of source cell i
///   area_b[j] = area of destination cell j
///
/// @tparam MemorySpace Kokkos memory space for internal array storage
///         (Kokkos::HostSpace, CudaSpace, HIPSpace, etc.)
template <class MemorySpace = Kokkos::HostSpace>
class InterpolationMatrix {
public:
    using memory_space = MemorySpace;

    /// Default-construct an empty matrix.
    InterpolationMatrix() = default;

    /// Construct by adopting pre-built Kokkos arrays (moved in, no copy).
    ///
    /// @param factor_list  Interpolation weights [nnz]
    /// @param factor_row   Destination (row) indices [nnz]
    /// @param factor_col   Source (col) indices [nnz]
    /// @param frac_a       Source fractions [n_src]
    /// @param frac_b       Destination fractions [n_dst]
    /// @param area_a       Source cell areas [n_src]
    /// @param area_b       Destination cell areas [n_dst]
    /// @param n_src        Number of source cells (n_a in ESMF terms)
    /// @param n_dst        Number of destination cells (n_b in ESMF terms)
    InterpolationMatrix(
        Kokkos::View<double*, MemorySpace>  factor_list,
        Kokkos::View<index_t*, MemorySpace> factor_row,
        Kokkos::View<index_t*, MemorySpace> factor_col,
        Kokkos::View<double*, MemorySpace>  frac_a,
        Kokkos::View<double*, MemorySpace>  frac_b,
        Kokkos::View<double*, MemorySpace>  area_a,
        Kokkos::View<double*, MemorySpace>  area_b,
        std::size_t n_src,
        std::size_t n_dst)
        : factor_list_(std::move(factor_list))
        , factor_row_(std::move(factor_row))
        , factor_col_(std::move(factor_col))
        , frac_a_(std::move(frac_a))
        , frac_b_(std::move(frac_b))
        , area_a_(std::move(area_a))
        , area_b_(std::move(area_b))
        , n_src_(n_src)
        , n_dst_(n_dst)
    {}

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Number of nonzero entries in the sparse matrix (n_s in ESMF terms).
    [[nodiscard]] std::size_t nnz() const noexcept {
        return factor_list_.extent(0);
    }

    /// Number of source cells (n_a in ESMF terms).
    [[nodiscard]] std::size_t n_src() const noexcept {
        return n_src_;
    }

    /// Number of destination cells (n_b in ESMF terms).
    [[nodiscard]] std::size_t n_dst() const noexcept {
        return n_dst_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Non-owning field_view accessors (zero-copy, layout_left)
    //
    // These return std::mdspan<const T, dextents, layout_left> directly
    // wrapping the internal Kokkos::View data pointers. No allocation or copy.
    // ─────────────────────────────────────────────────────────────────────────

    /// Interpolation weights S: [nnz].
    [[nodiscard]] field_view<const double, 1> factor_list() const noexcept {
        return field_view<const double, 1>{
            factor_list_.data(),
            factor_list_.extent(0)};
    }

    /// Source (column) indices: [nnz]. Each entry is the source cell index
    /// for the corresponding weight in factor_list.
    [[nodiscard]] field_view<const index_t, 1> factor_col() const noexcept {
        return field_view<const index_t, 1>{
            factor_col_.data(),
            factor_col_.extent(0)};
    }

    /// Destination (row) indices: [nnz]. Each entry is the destination cell
    /// index for the corresponding weight in factor_list.
    [[nodiscard]] field_view<const index_t, 1> factor_row() const noexcept {
        return field_view<const index_t, 1>{
            factor_row_.data(),
            factor_row_.extent(0)};
    }

    /// Combined (row, col) index pair accessor for the k-th nonzero.
    /// Provided for ESMF factorIndexList semantics compatibility.
    [[nodiscard]] IndexPair factor_index(std::size_t k) const noexcept {
        return IndexPair{factor_row_.data()[k], factor_col_.data()[k]};
    }

    /// Source fractions: [n_src]. frac_a[i] is the fraction of source cell i
    /// covered by the destination mesh.
    [[nodiscard]] field_view<const double, 1> frac_a() const noexcept {
        return field_view<const double, 1>{
            frac_a_.data(),
            frac_a_.extent(0)};
    }

    /// Destination fractions: [n_dst]. frac_b[j] is the fraction of
    /// destination cell j covered by the source mesh.
    [[nodiscard]] field_view<const double, 1> frac_b() const noexcept {
        return field_view<const double, 1>{
            frac_b_.data(),
            frac_b_.extent(0)};
    }

    /// Source cell areas: [n_src].
    [[nodiscard]] field_view<const double, 1> area_a() const noexcept {
        return field_view<const double, 1>{
            area_a_.data(),
            area_a_.extent(0)};
    }

    /// Destination cell areas: [n_dst].
    [[nodiscard]] field_view<const double, 1> area_b() const noexcept {
        return field_view<const double, 1>{
            area_b_.data(),
            area_b_.extent(0)};
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Kokkos::View accessors (for WeightGenerator, apply, and
    // conservation accounting that need direct View access)
    // ─────────────────────────────────────────────────────────────────────────

    [[nodiscard]] const auto& factor_list_view() const noexcept { return factor_list_; }
    [[nodiscard]] const auto& factor_row_view() const noexcept { return factor_row_; }
    [[nodiscard]] const auto& factor_col_view() const noexcept { return factor_col_; }
    [[nodiscard]] const auto& frac_a_view() const noexcept { return frac_a_; }
    [[nodiscard]] const auto& frac_b_view() const noexcept { return frac_b_; }
    [[nodiscard]] const auto& area_a_view() const noexcept { return area_a_; }
    [[nodiscard]] const auto& area_b_view() const noexcept { return area_b_; }

private:
    Kokkos::View<double*, MemorySpace>  factor_list_;   ///< weights [nnz]
    Kokkos::View<index_t*, MemorySpace> factor_row_;    ///< destination indices [nnz]
    Kokkos::View<index_t*, MemorySpace> factor_col_;    ///< source indices [nnz]
    Kokkos::View<double*, MemorySpace>  frac_a_;        ///< source fractions [n_src]
    Kokkos::View<double*, MemorySpace>  frac_b_;        ///< destination fractions [n_dst]
    Kokkos::View<double*, MemorySpace>  area_a_;        ///< source cell areas [n_src]
    Kokkos::View<double*, MemorySpace>  area_b_;        ///< destination cell areas [n_dst]
    std::size_t n_src_{0};
    std::size_t n_dst_{0};
};

} // namespace axis::solver

#endif // AXIS_SOLVER_INTERPOLATION_MATRIX_HPP
