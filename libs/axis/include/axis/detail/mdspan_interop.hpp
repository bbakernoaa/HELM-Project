// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_MDSPAN_INTEROP_HPP
#define AXIS_DETAIL_MDSPAN_INTEROP_HPP

/// @file axis/detail/mdspan_interop.hpp
/// @brief Zero-copy adapters between std::mdspan (layout_left) and
///        Kokkos::View (LayoutLeft, unmanaged).
///
/// These adapters satisfy HELM Law #1 (zero-copy): they wrap the same
/// underlying memory pointer without allocating or copying. The round-trip
/// property (to_view then to_mdspan yields identical address, extents, and
/// values) is guaranteed by construction and tested as Property 1.
///
/// Both adapters are noexcept — they perform no allocation, no device
/// synchronization, and no validation beyond what the type system enforces.

#include <Kokkos_Core.hpp>
#include <array>
#include <axis/types.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// to_view: field_view (layout_left mdspan) → Kokkos::View (LayoutLeft, unmanaged)
//
// Wraps the mdspan's data pointer in an unmanaged Kokkos::View with the same
// element type, layout, and extents. The View does NOT own the memory —
// lifetime management remains with the original buffer owner.
// ─────────────────────────────────────────────────────────────────────────────

namespace impl {

/// @brief Helper structure template to construct a Kokkos::View of appropriate rank.
///
/// @tparam T          The element type (e.g., double, const double).
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace).
/// @tparam Rank       The number of dimensions (rank) of the view.
template <class T, class MemorySpace, std::size_t Rank>
struct view_builder;

/// @brief Specialization of view_builder for Rank 1.
/// @tparam T          The element type.
/// @tparam MemorySpace The Kokkos memory space.
template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 1> {
    /// @brief Type alias for the unmanaged rank-1 LayoutLeft Kokkos::View.
    using view_type = Kokkos::View<T *, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryUnmanaged>;

    /// @brief Constructs a rank-1 unmanaged LayoutLeft Kokkos::View wrapping a raw pointer.
    /// @param ptr The raw data pointer.
    /// @param ext Array representing the dimension extents (size 1).
    /// @return An unmanaged rank-1 Kokkos::View.
    static view_type build(T *ptr, const std::array<std::size_t, 1> &ext) noexcept {
        return view_type(ptr, ext[0]);
    }
};

/// @brief Specialization of view_builder for Rank 2.
/// @tparam T          The element type.
/// @tparam MemorySpace The Kokkos memory space.
template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 2> {
    /// @brief Type alias for the unmanaged rank-2 LayoutLeft Kokkos::View.
    using view_type = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryUnmanaged>;

    /// @brief Constructs a rank-2 unmanaged LayoutLeft Kokkos::View wrapping a raw pointer.
    /// @param ptr The raw data pointer.
    /// @param ext Array representing the dimension extents (size 2).
    /// @return An unmanaged rank-2 Kokkos::View.
    static view_type build(T *ptr, const std::array<std::size_t, 2> &ext) noexcept {
        return view_type(ptr, ext[0], ext[1]);
    }
};

/// @brief Specialization of view_builder for Rank 3.
/// @tparam T          The element type.
/// @tparam MemorySpace The Kokkos memory space.
template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 3> {
    /// @brief Type alias for the unmanaged rank-3 LayoutLeft Kokkos::View.
    using view_type = Kokkos::View<T ***, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryUnmanaged>;

    /// @brief Constructs a rank-3 unmanaged LayoutLeft Kokkos::View wrapping a raw pointer.
    /// @param ptr The raw data pointer.
    /// @param ext Array representing the dimension extents (size 3).
    /// @return An unmanaged rank-3 Kokkos::View.
    static view_type build(T *ptr, const std::array<std::size_t, 3> &ext) noexcept {
        return view_type(ptr, ext[0], ext[1], ext[2]);
    }
};

/// @brief Specialization of view_builder for Rank 4.
/// @tparam T          The element type.
/// @tparam MemorySpace The Kokkos memory space.
template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 4> {
    /// @brief Type alias for the unmanaged rank-4 LayoutLeft Kokkos::View.
    using view_type = Kokkos::View<T ****, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryUnmanaged>;

    /// @brief Constructs a rank-4 unmanaged LayoutLeft Kokkos::View wrapping a raw pointer.
    /// @param ptr The raw data pointer.
    /// @param ext Array representing the dimension extents (size 4).
    /// @return An unmanaged rank-4 Kokkos::View.
    static view_type build(T *ptr, const std::array<std::size_t, 4> &ext) noexcept {
        return view_type(ptr, ext[0], ext[1], ext[2], ext[3]);
    }
};

}  // namespace impl

/// @brief Convert a layout_left field_view to an unmanaged Kokkos::View in the given MemorySpace.
///
/// This adapter satisfies HELM Law #1 (zero-copy): it wraps the same underlying memory
/// pointer without allocating or copying. The resulting View does NOT own the memory;
/// the lifetime of the underlying buffer must be managed by the caller and must exceed
/// that of the returned View.
///
/// @tparam T          Element type (e.g., double, const double).
/// @tparam MemorySpace Kokkos memory space (e.g., Kokkos::HostSpace).
/// @tparam Rank       Number of dimensions (1..4).
/// @param  m          The source mdspan (layout_left) of type field_view.
/// @return            An unmanaged Kokkos::View with LayoutLeft over the same memory.
///
/// @pre The mdspan's data pointer must address memory that is valid in MemorySpace.
///      This is the caller's responsibility; no runtime checks are performed.
///
/// @note Rank is specified explicitly at the call site because std::dextents
///       non-type template parameters are not always deducible from function arguments in
///       all compiler implementations. Usage example:
///       @code
///       auto view = to_view<double, Kokkos::HostSpace, 1>(fv);
///       @endcode
template <class T, class MemorySpace, std::size_t Rank>
[[nodiscard]] auto to_view(field_view<T, Rank> m) noexcept -> typename impl::view_builder<T, MemorySpace, Rank>::view_type {
    std::array<std::size_t, Rank> ext{};
    for (std::size_t i = 0; i < Rank; ++i) {
        ext[i] = m.extent(i);
    }
    return impl::view_builder<T, MemorySpace, Rank>::build(m.data_handle(), ext);
}

/// @brief Convenience overload of to_view that deduces T and Rank from the mdspan.
///
/// Deduces T and Rank from the mdspan, taking MemorySpace as a tag argument.
/// This form works when the compiler can deduce Rank from the mdspan type.
/// If deduction fails, use the explicit form: to_view<T, Space, Rank>(m).
///
/// @tparam MemorySpace Kokkos memory space (e.g., Kokkos::HostSpace).
/// @tparam T          Element type (deduced).
/// @tparam Rank       Number of dimensions (deduced).
/// @param  m          The source mdspan (layout_left) of type field_view.
/// @param  tag        An instance of MemorySpace used solely for type tagging.
/// @return            An unmanaged Kokkos::View with LayoutLeft over the same memory.
template <class MemorySpace, class T, std::size_t Rank>
[[nodiscard]] auto to_view(field_view<T, Rank> m, MemorySpace /*tag*/) noexcept -> typename impl::view_builder<T, MemorySpace, Rank>::view_type {
    return to_view<T, MemorySpace, Rank>(m);
}

// ─────────────────────────────────────────────────────────────────────────────
// to_mdspan: Kokkos::View (LayoutLeft, unmanaged) → field_view (layout_left mdspan)
//
// Wraps the View's data pointer in an mdspan with matching extents. Zero-copy.
// ─────────────────────────────────────────────────────────────────────────────

namespace impl {

/// @brief Helper to extract rank and extents from a Kokkos::View and build a field_view.
/// @tparam ViewType  The Kokkos::View type.
/// @tparam Is        The dimensions pack indices sequence.
/// @param  v         The source Kokkos::View.
/// @param  seq       Index sequence for parameter pack expansion.
/// @return           A field_view wrapping the same memory.
template <class ViewType, std::size_t... Is>
[[nodiscard]] auto make_mdspan(ViewType v, std::index_sequence<Is...>) noexcept -> field_view<typename ViewType::value_type, sizeof...(Is)> {
    using T = typename ViewType::value_type;
    constexpr std::size_t Rank = sizeof...(Is);
    return field_view<T, Rank>{v.data(), v.extent(Is)...};
}

}  // namespace impl

/// @brief Convert an unmanaged LayoutLeft Kokkos::View to a layout_left field_view.
///
/// Performs a zero-copy transformation by wrapping the same pointer that the View holds.
///
/// @tparam ViewType A Kokkos::View type with LayoutLeft layout.
/// @param  v        The source View (unmanaged, LayoutLeft).
/// @return         A field_view<T, Rank> over the same memory.
///
/// @note Static assertion: The View must use LayoutLeft (matching layout_left).
template <class ViewType>
[[nodiscard]] auto to_mdspan(ViewType v) noexcept -> field_view<typename ViewType::value_type, ViewType::rank> {
    static_assert(std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft>, "to_mdspan requires a LayoutLeft Kokkos::View");

    return impl::make_mdspan(v, std::make_index_sequence<ViewType::rank>{});
}

}  // namespace axis::detail

#endif  // AXIS_DETAIL_MDSPAN_INTEROP_HPP
