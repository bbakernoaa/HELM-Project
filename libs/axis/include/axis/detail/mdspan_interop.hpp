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

#include <axis/types.hpp>

#include <Kokkos_Core.hpp>

#include <array>
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

/// Helper: build a Kokkos::View from a pointer and a runtime extent array.
/// Specializations for Rank 1..4 cover all AXIS use cases.
template <class T, class MemorySpace, std::size_t Rank>
struct view_builder;

template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 1> {
    using view_type = Kokkos::View<T*, Kokkos::LayoutLeft,
                                   MemorySpace, Kokkos::MemoryUnmanaged>;

    static view_type build(T* ptr, const std::array<std::size_t, 1>& ext) noexcept {
        return view_type(ptr, ext[0]);
    }
};

template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 2> {
    using view_type = Kokkos::View<T**, Kokkos::LayoutLeft,
                                   MemorySpace, Kokkos::MemoryUnmanaged>;

    static view_type build(T* ptr, const std::array<std::size_t, 2>& ext) noexcept {
        return view_type(ptr, ext[0], ext[1]);
    }
};

template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 3> {
    using view_type = Kokkos::View<T***, Kokkos::LayoutLeft,
                                   MemorySpace, Kokkos::MemoryUnmanaged>;

    static view_type build(T* ptr, const std::array<std::size_t, 3>& ext) noexcept {
        return view_type(ptr, ext[0], ext[1], ext[2]);
    }
};

template <class T, class MemorySpace>
struct view_builder<T, MemorySpace, 4> {
    using view_type = Kokkos::View<T****, Kokkos::LayoutLeft,
                                   MemorySpace, Kokkos::MemoryUnmanaged>;

    static view_type build(T* ptr, const std::array<std::size_t, 4>& ext) noexcept {
        return view_type(ptr, ext[0], ext[1], ext[2], ext[3]);
    }
};

} // namespace impl

/// Convert a layout_left field_view to an unmanaged Kokkos::View in the given
/// MemorySpace. Zero-copy: the View wraps the same pointer the mdspan holds.
///
/// @tparam T          Element type (e.g., double, const double)
/// @tparam MemorySpace Kokkos memory space (e.g., Kokkos::HostSpace)
/// @tparam Rank       Number of dimensions (1..4)
/// @param  m          The source mdspan (layout_left)
/// @return            An unmanaged Kokkos::View with LayoutLeft over the same memory
///
/// Precondition: the mdspan's data pointer addresses memory in MemorySpace.
/// This is the caller's responsibility (no runtime check performed).
///
/// Note: Rank is specified explicitly at the call site because std::dextents
/// non-type template parameters are not deducible from function arguments in
/// all compiler implementations. Usage:
///   auto view = to_view<double, Kokkos::HostSpace, 1>(fv);
template <class T, class MemorySpace, std::size_t Rank>
[[nodiscard]] auto to_view(field_view<T, Rank> m) noexcept
    -> typename impl::view_builder<T, MemorySpace, Rank>::view_type
{
    std::array<std::size_t, Rank> ext{};
    for (std::size_t i = 0; i < Rank; ++i) {
        ext[i] = m.extent(i);
    }
    return impl::view_builder<T, MemorySpace, Rank>::build(m.data_handle(), ext);
}

/// Convenience overload: deduces T and Rank from the mdspan, takes MemorySpace
/// as a tag argument. This form works when the compiler can deduce Rank from
/// the mdspan type (depends on std::dextents implementation).
/// If deduction fails, use the explicit form: to_view<T, Space, Rank>(m).
template <class MemorySpace, class T, std::size_t Rank>
[[nodiscard]] auto to_view(field_view<T, Rank> m, MemorySpace /*tag*/) noexcept
    -> typename impl::view_builder<T, MemorySpace, Rank>::view_type
{
    return to_view<T, MemorySpace, Rank>(m);
}

// ─────────────────────────────────────────────────────────────────────────────
// to_mdspan: Kokkos::View (LayoutLeft, unmanaged) → field_view (layout_left mdspan)
//
// Wraps the View's data pointer in an mdspan with matching extents. Zero-copy.
// ─────────────────────────────────────────────────────────────────────────────

namespace impl {

/// Helper: extract rank and extents from a Kokkos::View and build a field_view.
template <class ViewType, std::size_t... Is>
[[nodiscard]] auto make_mdspan(ViewType v, std::index_sequence<Is...>) noexcept
    -> field_view<typename ViewType::value_type, sizeof...(Is)>
{
    using T = typename ViewType::value_type;
    constexpr std::size_t Rank = sizeof...(Is);
    return field_view<T, Rank>{v.data(), v.extent(Is)...};
}

} // namespace impl

/// Convert an unmanaged LayoutLeft Kokkos::View to a layout_left field_view.
/// Zero-copy: the mdspan wraps the same pointer the View holds.
///
/// @tparam ViewType A Kokkos::View with LayoutLeft layout
/// @param  v       The source View (unmanaged, LayoutLeft)
/// @return         A field_view<T, Rank> over the same memory
///
/// Static assertion: the View must use LayoutLeft (matching layout_left).
template <class ViewType>
[[nodiscard]] auto to_mdspan(ViewType v) noexcept
    -> field_view<typename ViewType::value_type, ViewType::rank>
{
    static_assert(
        std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft>,
        "to_mdspan requires a LayoutLeft Kokkos::View");

    return impl::make_mdspan(v, std::make_index_sequence<ViewType::rank>{});
}

} // namespace axis::detail

#endif // AXIS_DETAIL_MDSPAN_INTEROP_HPP
