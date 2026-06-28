#ifndef HALO_DETAIL_STAGING_HPP
#define HALO_DETAIL_STAGING_HPP

/// @file halo/detail/staging.hpp
/// @brief Host-staging buffer utilities for non-GPU-aware MPI paths.
///
/// Provides stage_send() and stage_recv() helper functions that perform
/// Kokkos::deep_copy between device views and host mirror buffers. These
/// are used by the exchange functions when the view resides in device memory
/// and GPU-aware MPI is not available (requires_staging_v == true).
///
/// @see halo/detail/memory_traits.hpp for requires_staging_v and host_mirror_t.

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <halo/detail/memory_traits.hpp>

namespace halo::detail {

/// @brief Deep-copy a subview of a device view to a host mirror buffer.
///
/// Creates a host mirror view, takes a subview of the source device view
/// at the specified [offset, offset+count) range, and performs a deep_copy
/// from the device subview to the host mirror. The host mirror is suitable
/// for passing to MPI send operations.
///
/// @tparam ViewType A Kokkos::View type (1D) residing in device memory.
/// @param view     The source device view containing data to send.
/// @param offset   Starting element index within the view.
/// @param count    Number of elements to copy starting from offset.
/// @return A host mirror view containing the copied data, ready for MPI.
///
/// @pre offset + count <= view.extent(0)
/// @post The returned host mirror contains a copy of view[offset..offset+count).
template <typename ViewType>
[[nodiscard]] auto stage_send(const ViewType &view, std::size_t offset, std::size_t count) -> host_mirror_t<ViewType> {
    // Create a subview of the device data at [offset, offset+count)
    auto device_subview = Kokkos::subview(view, Kokkos::make_pair(offset, offset + count));

    // Allocate a host mirror buffer for the subview
    auto host_buffer = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, Kokkos::HostSpace{}, device_subview);

    // Deep-copy device data to host buffer
    Kokkos::deep_copy(host_buffer, device_subview);

    return host_buffer;
}

/// @brief Deep-copy data from a host buffer back to a subview of a device view.
///
/// Takes a subview of the destination device view at the specified
/// [offset, offset+count) range and performs a deep_copy from the host
/// buffer into that device subview. Used after MPI receive operations
/// complete to transfer received data back to device memory.
///
/// @tparam ViewType A Kokkos::View type (1D) residing in device memory.
/// @param host_buffer  The host buffer containing received data from MPI.
/// @param device_view  The destination device view to copy data into.
/// @param offset       Starting element index within the device view.
/// @param count        Number of elements to copy starting from offset.
///
/// @pre offset + count <= device_view.extent(0)
/// @pre host_buffer.extent(0) >= count
/// @post device_view[offset..offset+count) contains the data from host_buffer.
template <typename ViewType>
void stage_recv(const host_mirror_t<ViewType> &host_buffer, ViewType &device_view, std::size_t offset, std::size_t count) {
    // Create a subview of the device view at [offset, offset+count)
    auto device_subview = Kokkos::subview(device_view, Kokkos::make_pair(offset, offset + count));

    // Create a subview of the host buffer for the relevant range [0, count)
    auto host_subview = Kokkos::subview(host_buffer, Kokkos::make_pair(std::size_t{0}, count));

    // Deep-copy from host buffer back to device
    Kokkos::deep_copy(device_subview, host_subview);
}

}  // namespace halo::detail

#endif  // HALO_DETAIL_STAGING_HPP
