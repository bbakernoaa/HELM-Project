#ifndef HALO_DETAIL_PACK_UNPACK_HPP
#define HALO_DETAIL_PACK_UNPACK_HPP

/// @file halo/detail/pack_unpack.hpp
/// @brief Device-side pack/unpack kernels for structured halo exchange.
///
/// Provides pack() and unpack() template functions that execute as
/// Kokkos::parallel_for kernels in the view's native execution space.
///
/// - For contiguous subviews: uses RangePolicy with linearized indexing
///   (direct pointer access for maximum bandwidth).
/// - For strided (non-contiguous) subviews: uses MDRangePolicy with
///   multi-dimensional index decomposition.
///
/// Both variants fence after completion to ensure data visibility before
/// MPI send (pack) or before computation resumes (unpack).
///
/// Supports views of rank 1 through 4.

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace halo::detail {

// ─── Contiguity Detection ───────────────────────────────────────────────────

/// @brief Detect at runtime whether a view's data is contiguous in memory.
///
/// A view is contiguous when its span (distance from first to last element + 1)
/// equals its logical size (total number of elements). This holds for views
/// with LayoutLeft/LayoutRight without stride padding, and for rank-1 views
/// with stride(0)==1.
///
/// @tparam ViewType A Kokkos::View type.
/// @param view The view to check.
/// @return true if the view's memory is contiguous.
template <typename ViewType>
KOKKOS_INLINE_FUNCTION
bool is_contiguous(const ViewType& view) {
    // A view is contiguous when its memory span equals its logical element count.
    // This is true for directly constructed LayoutLeft/LayoutRight views, and
    // false for subviews that skip elements (e.g., a column from a row-major 2D view).
    return view.span() == view.size();
}

// ─── Pack: Contiguous (RangePolicy) ────────────────────────────────────────

/// @brief Pack a contiguous subview into a flat buffer using RangePolicy.
///
/// Uses linearized pointer access: buffer(i) = src.data()[i].
/// This is optimal when the source subview is known to be contiguous.
///
/// @tparam SrcView Source view type (the subview to pack from).
/// @tparam DstView Destination buffer type (1D contiguous buffer).
/// @param src  The source subview (must be contiguous in memory).
/// @param dst  The destination 1D buffer (size >= src.size()).
/// @param exec The execution space instance to launch the kernel on.
template <typename SrcView, typename DstView>
void pack_contiguous(const SrcView& src, const DstView& dst,
                     typename SrcView::execution_space exec = {}) {
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");

    using exec_space = typename SrcView::execution_space;
    using size_type = typename SrcView::size_type;
    const size_type n = src.size();

    Kokkos::parallel_for(
        "halo_pack_contiguous",
        Kokkos::RangePolicy<exec_space>(exec, 0, n),
        KOKKOS_LAMBDA(const size_type i) {
            dst(i) = src.data()[i];
        });

    exec.fence("halo::detail::pack_contiguous");
}

// ─── Unpack: Contiguous (RangePolicy) ──────────────────────────────────────

/// @brief Unpack a flat buffer into a contiguous subview using RangePolicy.
///
/// Uses linearized pointer access: dst.data()[i] = src(i).
/// This is optimal when the destination subview is known to be contiguous.
///
/// @tparam SrcView Source buffer type (1D contiguous buffer).
/// @tparam DstView Destination view type (the subview to unpack into).
/// @param src  The source 1D buffer (size >= dst.size()).
/// @param dst  The destination subview (must be contiguous in memory).
/// @param exec The execution space instance to launch the kernel on.
template <typename SrcView, typename DstView>
void unpack_contiguous(const SrcView& src, const DstView& dst,
                       typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");

    using exec_space = typename DstView::execution_space;
    using size_type = typename DstView::size_type;
    const size_type n = dst.size();

    Kokkos::parallel_for(
        "halo_unpack_contiguous",
        Kokkos::RangePolicy<exec_space>(exec, 0, n),
        KOKKOS_LAMBDA(const size_type i) {
            dst.data()[i] = src(i);
        });

    exec.fence("halo::detail::unpack_contiguous");
}

// ─── Pack: Strided / MDRangePolicy (Rank 1-4) ──────────────────────────────

/// @brief Pack a strided rank-1 subview into a flat buffer.
template <typename SrcView, typename DstView>
void pack_strided_rank1(const SrcView& src, const DstView& dst,
                        typename SrcView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source must be rank-1");
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");

    using exec_space = typename SrcView::execution_space;
    using size_type = typename SrcView::size_type;
    const size_type n0 = src.extent(0);

    Kokkos::parallel_for(
        "halo_pack_strided_r1",
        Kokkos::RangePolicy<exec_space>(exec, 0, n0),
        KOKKOS_LAMBDA(const size_type i0) {
            dst(i0) = src(i0);
        });

    exec.fence("halo::detail::pack_strided_rank1");
}

/// @brief Pack a strided rank-2 subview into a flat buffer using MDRangePolicy.
template <typename SrcView, typename DstView>
void pack_strided_rank2(const SrcView& src, const DstView& dst,
                        typename SrcView::execution_space exec = {}) {
    static_assert(SrcView::rank == 2, "Source must be rank-2");
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");

    using exec_space = typename SrcView::execution_space;
    using size_type = typename SrcView::size_type;
    const size_type n0 = src.extent(0);
    const size_type n1 = src.extent(1);

    Kokkos::parallel_for(
        "halo_pack_strided_r2",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(exec, {0, 0}, {n0, n1}),
        KOKKOS_LAMBDA(const size_type i0, const size_type i1) {
            // Linearize using LayoutRight convention (row-major) for the buffer:
            // buffer_idx = i0 * n1 + i1
            dst(i0 * n1 + i1) = src(i0, i1);
        });

    exec.fence("halo::detail::pack_strided_rank2");
}

/// @brief Pack a strided rank-3 subview into a flat buffer using MDRangePolicy.
template <typename SrcView, typename DstView>
void pack_strided_rank3(const SrcView& src, const DstView& dst,
                        typename SrcView::execution_space exec = {}) {
    static_assert(SrcView::rank == 3, "Source must be rank-3");
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");

    using exec_space = typename SrcView::execution_space;
    using size_type = typename SrcView::size_type;
    const size_type n0 = src.extent(0);
    const size_type n1 = src.extent(1);
    const size_type n2 = src.extent(2);

    Kokkos::parallel_for(
        "halo_pack_strided_r3",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(exec, {0, 0, 0}, {n0, n1, n2}),
        KOKKOS_LAMBDA(const size_type i0, const size_type i1, const size_type i2) {
            dst((i0 * n1 + i1) * n2 + i2) = src(i0, i1, i2);
        });

    exec.fence("halo::detail::pack_strided_rank3");
}

/// @brief Pack a strided rank-4 subview into a flat buffer using MDRangePolicy.
template <typename SrcView, typename DstView>
void pack_strided_rank4(const SrcView& src, const DstView& dst,
                        typename SrcView::execution_space exec = {}) {
    static_assert(SrcView::rank == 4, "Source must be rank-4");
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");

    using exec_space = typename SrcView::execution_space;
    using size_type = typename SrcView::size_type;
    const size_type n0 = src.extent(0);
    const size_type n1 = src.extent(1);
    const size_type n2 = src.extent(2);
    const size_type n3 = src.extent(3);

    // Rank-4 MDRangePolicy: Kokkos supports up to rank-6, but we use
    // a nested approach with RangePolicy over linearized outer dims
    // to avoid potential limitations.
    const size_type total = n0 * n1 * n2 * n3;

    Kokkos::parallel_for(
        "halo_pack_strided_r4",
        Kokkos::RangePolicy<exec_space>(exec, 0, total),
        KOKKOS_LAMBDA(const size_type idx) {
            const size_type i3 = idx % n3;
            const size_type rem = idx / n3;
            const size_type i2 = rem % n2;
            const size_type rem2 = rem / n2;
            const size_type i1 = rem2 % n1;
            const size_type i0 = rem2 / n1;
            dst(idx) = src(i0, i1, i2, i3);
        });

    exec.fence("halo::detail::pack_strided_rank4");
}

// ─── Unpack: Strided / MDRangePolicy (Rank 1-4) ────────────────────────────

/// @brief Unpack a flat buffer into a strided rank-1 subview.
template <typename SrcView, typename DstView>
void unpack_strided_rank1(const SrcView& src, const DstView& dst,
                          typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");
    static_assert(DstView::rank == 1, "Destination must be rank-1");

    using exec_space = typename DstView::execution_space;
    using size_type = typename DstView::size_type;
    const size_type n0 = dst.extent(0);

    Kokkos::parallel_for(
        "halo_unpack_strided_r1",
        Kokkos::RangePolicy<exec_space>(exec, 0, n0),
        KOKKOS_LAMBDA(const size_type i0) {
            dst(i0) = src(i0);
        });

    exec.fence("halo::detail::unpack_strided_rank1");
}

/// @brief Unpack a flat buffer into a strided rank-2 subview using MDRangePolicy.
template <typename SrcView, typename DstView>
void unpack_strided_rank2(const SrcView& src, const DstView& dst,
                          typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");
    static_assert(DstView::rank == 2, "Destination must be rank-2");

    using exec_space = typename DstView::execution_space;
    using size_type = typename DstView::size_type;
    const size_type n0 = dst.extent(0);
    const size_type n1 = dst.extent(1);

    Kokkos::parallel_for(
        "halo_unpack_strided_r2",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>(exec, {0, 0}, {n0, n1}),
        KOKKOS_LAMBDA(const size_type i0, const size_type i1) {
            dst(i0, i1) = src(i0 * n1 + i1);
        });

    exec.fence("halo::detail::unpack_strided_rank2");
}

/// @brief Unpack a flat buffer into a strided rank-3 subview using MDRangePolicy.
template <typename SrcView, typename DstView>
void unpack_strided_rank3(const SrcView& src, const DstView& dst,
                          typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");
    static_assert(DstView::rank == 3, "Destination must be rank-3");

    using exec_space = typename DstView::execution_space;
    using size_type = typename DstView::size_type;
    const size_type n0 = dst.extent(0);
    const size_type n1 = dst.extent(1);
    const size_type n2 = dst.extent(2);

    Kokkos::parallel_for(
        "halo_unpack_strided_r3",
        Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>(exec, {0, 0, 0}, {n0, n1, n2}),
        KOKKOS_LAMBDA(const size_type i0, const size_type i1, const size_type i2) {
            dst(i0, i1, i2) = src((i0 * n1 + i1) * n2 + i2);
        });

    exec.fence("halo::detail::unpack_strided_rank3");
}

/// @brief Unpack a flat buffer into a strided rank-4 subview using MDRangePolicy.
template <typename SrcView, typename DstView>
void unpack_strided_rank4(const SrcView& src, const DstView& dst,
                          typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");
    static_assert(DstView::rank == 4, "Destination must be rank-4");

    using exec_space = typename DstView::execution_space;
    using size_type = typename DstView::size_type;
    const size_type n0 = dst.extent(0);
    const size_type n1 = dst.extent(1);
    const size_type n2 = dst.extent(2);
    const size_type n3 = dst.extent(3);

    const size_type total = n0 * n1 * n2 * n3;

    Kokkos::parallel_for(
        "halo_unpack_strided_r4",
        Kokkos::RangePolicy<exec_space>(exec, 0, total),
        KOKKOS_LAMBDA(const size_type idx) {
            const size_type i3 = idx % n3;
            const size_type rem = idx / n3;
            const size_type i2 = rem % n2;
            const size_type rem2 = rem / n2;
            const size_type i1 = rem2 % n1;
            const size_type i0 = rem2 / n1;
            dst(i0, i1, i2, i3) = src(idx);
        });

    exec.fence("halo::detail::unpack_strided_rank4");
}

// ─── Unified Pack Interface ─────────────────────────────────────────────────

/// @brief Pack a subview into a contiguous 1D buffer.
///
/// Detects contiguity at runtime and selects the optimal kernel path:
/// - Contiguous subview → RangePolicy with linearized data()[i] access
/// - Strided subview → MDRangePolicy (rank 2/3) or index decomposition (rank 1/4)
///
/// Fences after completion to ensure the buffer is safe to pass to MPI_Isend.
///
/// @tparam SrcView Source view type (rank 1-4, any layout).
/// @tparam DstView Destination view type (rank 1, contiguous buffer).
/// @param src  The source subview to pack from.
/// @param dst  The destination 1D buffer (size >= src.size()).
/// @param exec The execution space instance to launch the kernel on.
template <typename SrcView, typename DstView>
void pack(const SrcView& src, const DstView& dst,
          typename SrcView::execution_space exec = {}) {
    static_assert(DstView::rank == 1, "Destination buffer must be rank-1");
    static_assert(SrcView::rank >= 1 && SrcView::rank <= 4,
                  "pack() supports source views of rank 1 through 4");

    // Runtime contiguity check: use the fast linearized path when possible
    if (is_contiguous(src)) {
        pack_contiguous(src, dst, exec);
    } else {
        // Dispatch to the appropriate strided kernel based on rank
        if constexpr (SrcView::rank == 1) {
            pack_strided_rank1(src, dst, exec);
        } else if constexpr (SrcView::rank == 2) {
            pack_strided_rank2(src, dst, exec);
        } else if constexpr (SrcView::rank == 3) {
            pack_strided_rank3(src, dst, exec);
        } else if constexpr (SrcView::rank == 4) {
            pack_strided_rank4(src, dst, exec);
        }
    }
}

/// @brief Unpack a contiguous 1D buffer into a subview.
///
/// Detects contiguity at runtime and selects the optimal kernel path:
/// - Contiguous subview → RangePolicy with linearized data()[i] access
/// - Strided subview → MDRangePolicy (rank 2/3) or index decomposition (rank 1/4)
///
/// Fences after completion to ensure the subview data is visible before
/// computation resumes.
///
/// @tparam SrcView Source view type (rank 1, contiguous buffer).
/// @tparam DstView Destination view type (rank 1-4, any layout).
/// @param src  The source 1D buffer (size >= dst.size()).
/// @param dst  The destination subview to unpack into.
/// @param exec The execution space instance to launch the kernel on.
template <typename SrcView, typename DstView>
void unpack(const SrcView& src, const DstView& dst,
            typename DstView::execution_space exec = {}) {
    static_assert(SrcView::rank == 1, "Source buffer must be rank-1");
    static_assert(DstView::rank >= 1 && DstView::rank <= 4,
                  "unpack() supports destination views of rank 1 through 4");

    // Runtime contiguity check: use the fast linearized path when possible
    if (is_contiguous(dst)) {
        unpack_contiguous(src, dst, exec);
    } else {
        // Dispatch to the appropriate strided kernel based on rank
        if constexpr (DstView::rank == 1) {
            unpack_strided_rank1(src, dst, exec);
        } else if constexpr (DstView::rank == 2) {
            unpack_strided_rank2(src, dst, exec);
        } else if constexpr (DstView::rank == 3) {
            unpack_strided_rank3(src, dst, exec);
        } else if constexpr (DstView::rank == 4) {
            unpack_strided_rank4(src, dst, exec);
        }
    }
}

} // namespace halo::detail

#endif // HALO_DETAIL_PACK_UNPACK_HPP
