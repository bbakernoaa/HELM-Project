// SPDX-License-Identifier: Apache-2.0
// SPAN — FieldView<T, Rank>: non-owning, coherency-tracked dual-pointer view
// Copyright (c) HELM Project Contributors

#ifndef SPAN_FIELD_VIEW_HPP
#define SPAN_FIELD_VIEW_HPP

/// @file span/field_view.hpp
/// @brief The revised FieldView template class implementing zero-copy dual-pointer
///        semantics with coherency state tracking.
///
/// This header provides `span::FieldView<T, Rank>`, a non-owning view over host
/// and/or device memory with explicit coherency state management. Unlike the
/// original span.hpp prototype, this implementation:
///   - Stores TWO raw pointers (host + device), both non-owning
///   - Never allocates device memory (zero-copy, HELM Law #1)
///   - Provides std::mdspan access over the host pointer
///   - Tracks CoherencyState for lazy synchronization
///   - Supports optional TripleBuffer attachment for AMIO I/O isolation

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span/coherency_state.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

// mdspan: use the C++23 standard header if available, otherwise fall back to
// the kokkos/mdspan reference implementation.
#if __has_include(<mdspan>) && (__cplusplus > 202002L || defined(__cpp_lib_mdspan))
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
namespace std {
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::layout_left;
using Kokkos::mdspan;
#ifndef __cpp_lib_span
using Kokkos::dynamic_extent;
#endif
}  // namespace std
#endif

namespace span {

// Forward declaration for triple-buffer attachment (implemented in a later task)
template <typename T>
class TripleBuffer;

// ═══════════════════════════════════════════════════════════════════════════════
// FieldView<T, Rank> — revised dual-pointer non-owning view
// ═══════════════════════════════════════════════════════════════════════════════

/// @brief Non-owning, coherency-tracked view over contiguous host and/or device
///        memory with column-major (Fortran) layout.
///
/// FieldView wraps up to two raw pointers (host and device) in a lightweight
/// value type that provides:
///   - `std::mdspan<T, dextents<size_t, Rank>, layout_left>` access via `view()`
///   - Kokkos::View adapters for kernel integration
///   - CoherencyState tracking for lazy host↔device synchronization
///   - Optional TripleBuffer attachment for AMIO I/O isolation
///
/// @par Zero-Copy Guarantee (HELM Law #1)
/// Construction SHALL NOT allocate or copy field data. Both host and device
/// pointers are externally owned. FieldView is a metadata wrapper only.
///
/// @tparam T     Element type (typically `double` or `float`).
/// @tparam Rank  Number of dimensions (1–7 for typical climate fields).
template <typename T, std::size_t Rank>
class FieldView {
   public:
    // ─────────────────────────────────────────────────────────────────────────
    // Type aliases
    // ─────────────────────────────────────────────────────────────────────────

    template <typename IndexType, std::size_t R>
    using my_dextents = std::conditional_t<
        R == 1, Kokkos::extents<IndexType, std::dynamic_extent>,
        std::conditional_t<
            R == 2, Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent>,
            std::conditional_t<R == 3, Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
                               Kokkos::extents<IndexType, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>>>>;

    /// The mdspan type for host-side access (column-major, non-owning).
    using mdspan_type = Kokkos::mdspan<T, my_dextents<std::size_t, Rank>, Kokkos::layout_left>;

    /// Extent array type (fixed-size array matching Rank).
    using extents_type = std::array<std::size_t, Rank>;

    /// Kokkos unmanaged host view (flat rank-1, wraps host pointer).
    using kokkos_host_view = Kokkos::View<T *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

    /// Kokkos unmanaged device view (flat rank-1, wraps device pointer).
    using kokkos_device_view = Kokkos::View<T *, Kokkos::LayoutLeft, typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::MemoryUnmanaged>;

    // ─────────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Default constructor — creates an empty/invalid FieldView.
    ///
    /// All pointers are null, extents are zero, valid() returns false.
    FieldView() noexcept = default;

    /// @brief Host-only constructor.
    ///
    /// Wraps a host pointer with the given extents. Device pointer is nullptr.
    /// Memory space is set to Host. Initial coherency state is HOST_CLEAN.
    ///
    /// @param host_ptr  Non-null pointer to contiguous host memory.
    /// @param exts      Array of extents; all must be > 0.
    ///
    /// @throws std::invalid_argument if host_ptr is null or any extent is 0.
    explicit FieldView(T *host_ptr, extents_type exts)
        : host_ptr_(host_ptr),
          device_ptr_(nullptr),
          extents_(exts),
          total_size_(compute_total_size(exts)),
          state_(CoherencyState::HOST_CLEAN),
          mem_space_(MemorySpaceToken::Host),
          triple_buf_(nullptr) {
        validate_host_construction(host_ptr, exts);
    }

    /// @brief Device-only constructor.
    ///
    /// Wraps a device pointer with the given extents. Host pointer is nullptr.
    /// Memory space is set to the default device backend (CudaDevice or HipDevice).
    /// Initial coherency state is HOST_CLEAN.
    ///
    /// @param           std::nullptr_t tag to distinguish from host constructor.
    /// @param device_ptr Non-null pointer to contiguous device memory.
    /// @param exts       Array of extents; all must be > 0.
    ///
    /// @throws std::invalid_argument if device_ptr is null or any extent is 0.
    explicit FieldView(std::nullptr_t, T *device_ptr, extents_type exts)
        : host_ptr_(nullptr),
          device_ptr_(device_ptr),
          extents_(exts),
          total_size_(compute_total_size(exts)),
          state_(CoherencyState::HOST_CLEAN),
          mem_space_(determine_device_memory_space()),
          triple_buf_(nullptr) {
        validate_device_construction(device_ptr, exts);
    }

    /// @brief Dual-pointer constructor.
    ///
    /// Wraps both host and device pointers with the given extents.
    /// Memory space is set to Host. Initial coherency state is HOST_CLEAN.
    /// If both pointers are nullptr, the FieldView is invalid (valid() == false).
    ///
    /// @param host_ptr   Pointer to host memory (may be null if device_ptr is non-null).
    /// @param device_ptr Pointer to device memory (may be null if host_ptr is non-null).
    /// @param exts       Array of extents; all must be > 0 if either pointer is non-null.
    ///
    /// @throws std::invalid_argument if both pointers are null with non-zero extents,
    ///         or if any extent is 0 when a valid pointer is provided.
    explicit FieldView(T *host_ptr, T *device_ptr, extents_type exts)
        : host_ptr_(host_ptr),
          device_ptr_(device_ptr),
          extents_(exts),
          total_size_(compute_total_size(exts)),
          state_(CoherencyState::HOST_CLEAN),
          mem_space_(device_ptr != nullptr && host_ptr == nullptr ? determine_device_memory_space() : MemorySpaceToken::Host),
          triple_buf_(nullptr) {
        // Req 4.11: If both are null, this is simply an invalid view
        if (host_ptr == nullptr && device_ptr == nullptr) {
            // Reset to default invalid state
            extents_ = {};
            total_size_ = 0;
            return;
        }
        validate_extents(exts);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // mdspan and raw pointer access
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Obtain a non-owning mdspan view over the host data.
    ///
    /// Returns an mdspan with layout_left (column-major) matching Fortran array
    /// semantics. The mdspan is constructed from the host pointer.
    ///
    /// @return std::mdspan over the host pointer with the registered extents.
    /// @throws std::runtime_error if host pointer is null.
    [[nodiscard]] mdspan_type view() const {
        if (host_ptr_ == nullptr) {
            throw std::runtime_error("FieldView::view(): host pointer is null");
        }
        return make_mdspan(std::make_index_sequence<Rank>{});
    }

    /// @brief Raw pointer to host data. Synonym for host_data().
    [[nodiscard]] T *data() const noexcept {
        return host_ptr_;
    }

    /// @brief Raw pointer to host memory.
    [[nodiscard]] T *host_data() const noexcept {
        return host_ptr_;
    }

    /// @brief Raw pointer to device memory.
    [[nodiscard]] T *device_data() const noexcept {
        return device_ptr_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Metadata queries
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Extent along a specific dimension.
    /// @param dim  Dimension index in [0, Rank).
    /// @throws std::out_of_range if dim >= Rank.
    [[nodiscard]] std::size_t extent(std::size_t dim) const {
        if (dim >= Rank) {
            throw std::out_of_range("FieldView::extent(): dim " + std::to_string(dim) + " >= Rank " + std::to_string(Rank));
        }
        return extents_[dim];
    }

    /// @brief The full extents array.
    [[nodiscard]] const extents_type &extents() const noexcept {
        return extents_;
    }

    /// @brief Total number of elements (product of all extents).
    [[nodiscard]] std::size_t size() const noexcept {
        return total_size_;
    }

    /// @brief Compile-time rank (number of dimensions).
    [[nodiscard]] static constexpr std::size_t rank() noexcept {
        return Rank;
    }

    /// @brief Check if this FieldView wraps at least one valid pointer.
    ///
    /// A FieldView is valid if it has a non-null host or device pointer.
    [[nodiscard]] bool valid() const noexcept {
        return host_ptr_ != nullptr || device_ptr_ != nullptr;
    }

    /// @brief Check if a host pointer is available.
    [[nodiscard]] bool has_host_ptr() const noexcept {
        return host_ptr_ != nullptr;
    }

    /// @brief Check if a device pointer is available.
    [[nodiscard]] bool has_device_ptr() const noexcept {
        return device_ptr_ != nullptr;
    }

    /// @brief Query the memory space where the primary pointer resides.
    ///
    /// Returns CudaDevice (or HipDevice) if only a device pointer is set.
    /// Otherwise returns Host.
    [[nodiscard]] MemorySpaceToken memory_space() const noexcept {
        return mem_space_;
    }

    /// @brief Check if a TripleBuffer is attached to this FieldView.
    [[nodiscard]] bool is_triple_buffered() const noexcept {
        return triple_buf_ != nullptr;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Coherency state
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Query the current coherency state.
    [[nodiscard]] CoherencyState coherency_state() const noexcept {
        return state_;
    }

    /// @brief Mark the host copy as modified (device copy is now stale).
    ///
    /// Call this after writing to host memory. The next sync_for_device()
    /// will transfer updated data to the device.
    void mark_host_dirty() noexcept {
        state_ = CoherencyState::HOST_DIRTY;
    }

    /// @brief Mark the device copy as modified (host copy is now stale).
    ///
    /// Call this after a device kernel writes to device memory. The next
    /// sync_for_host() will transfer results back to the host.
    void mark_device_dirty() noexcept {
        state_ = CoherencyState::DEVICE_DIRTY;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Coherency synchronization (Kokkos::deep_copy)
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Synchronize host data to device.
    ///
    /// IF coherency state is HOST_DIRTY AND both host and device pointers are present,
    /// THEN performs Kokkos::deep_copy from host to device and sets state to HOST_CLEAN.
    /// OTHERWISE this is a no-op.
    void sync_for_device() {
        if (state_ == CoherencyState::HOST_DIRTY && host_ptr_ != nullptr && device_ptr_ != nullptr) {
            kokkos_host_view h_view(host_ptr_, total_size_);
            kokkos_device_view d_view(device_ptr_, total_size_);
            Kokkos::deep_copy(d_view, h_view);
            state_ = CoherencyState::HOST_CLEAN;
        }
        // Single-pointer or non-dirty: no-op
    }

    /// @brief Synchronize device data to host.
    ///
    /// IF coherency state is DEVICE_DIRTY AND both host and device pointers are present,
    /// THEN performs Kokkos::deep_copy from device to host and sets state to HOST_CLEAN.
    /// OTHERWISE this is a no-op.
    void sync_for_host() {
        if (state_ == CoherencyState::DEVICE_DIRTY && host_ptr_ != nullptr && device_ptr_ != nullptr) {
            kokkos_host_view h_view(host_ptr_, total_size_);
            kokkos_device_view d_view(device_ptr_, total_size_);
            Kokkos::deep_copy(h_view, d_view);
            state_ = CoherencyState::HOST_CLEAN;
        }
        // Single-pointer or non-dirty: no-op
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Triple-buffer attachment (stub — full impl in task 5.4)
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Attach a TripleBuffer to this FieldView for AMIO I/O isolation.
    ///
    /// The TripleBuffer is non-owning. The caller must ensure the TripleBuffer
    /// outlives this FieldView.
    void attach_triple_buffer(TripleBuffer<T> *tb) noexcept {
        triple_buf_ = tb;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kokkos View adapters (zero-copy, no allocation, no coherency modification)
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Return a flat rank-1 unmanaged Kokkos::View wrapping the host pointer.
    ///
    /// The returned view shares memory with this FieldView (zero-copy). It is
    /// flattened to rank-1 with total_size_ elements, regardless of FieldView Rank.
    ///
    /// @return Unmanaged host-space Kokkos::View over host_data().
    /// @throws std::invalid_argument if this FieldView is invalid (no valid pointer).
    [[nodiscard]] kokkos_host_view to_kokkos_view() const {
        if (!valid()) {
            throw std::invalid_argument("to_kokkos_view: FieldView is invalid");
        }
        return kokkos_host_view(host_ptr_, total_size_);
    }

    /// @brief Return a flat rank-1 unmanaged Kokkos::View wrapping the device pointer.
    ///
    /// The returned view shares memory with this FieldView's device allocation (zero-copy).
    ///
    /// @return Unmanaged device-space Kokkos::View over device_data().
    /// @throws std::invalid_argument if no device pointer is set.
    [[nodiscard]] kokkos_device_view to_kokkos_device_view() const {
        if (device_ptr_ == nullptr) {
            throw std::invalid_argument("to_kokkos_device_view: no device pointer");
        }
        return kokkos_device_view(device_ptr_, total_size_);
    }

    /// @brief Construct a FieldView from an unmanaged host-space Kokkos::View (zero-copy).
    ///
    /// For Rank == 1, the extent is extracted directly from the view. For higher
    /// ranks, a flat view with extent(0) elements is mapped to a rank-1 FieldView
    /// only if Rank == 1; otherwise this overload is only valid for Rank == 1.
    ///
    /// @param kv  An unmanaged Kokkos::View with non-null data pointer.
    /// @return    A FieldView wrapping kv.data() with matching extents.
    /// @throws std::invalid_argument if kv.data() is null.
    [[nodiscard]] static FieldView from_kokkos_view(kokkos_host_view kv) {
        if (kv.data() == nullptr) {
            throw std::invalid_argument("from_kokkos_view: null data pointer");
        }
        extents_type exts{};
        if constexpr (Rank == 1) {
            exts[0] = kv.extent(0);
        } else {
            // For Rank > 1, the flat view extent is mapped entirely to the first
            // dimension. Caller should use the extents-overload for correct shape.
            exts[0] = kv.extent(0);
            for (std::size_t i = 1; i < Rank; ++i) {
                exts[i] = 1;
            }
        }
        return FieldView(kv.data(), exts);
    }

   private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute total element count from extents array.
    static constexpr std::size_t compute_total_size(const extents_type &exts) noexcept {
        std::size_t s = 1;
        for (auto e : exts) s *= e;
        return s;
    }

    /// Validate host-only construction preconditions.
    static void validate_host_construction(T *host_ptr, const extents_type &exts) {
        if (host_ptr == nullptr) {
            throw std::invalid_argument("FieldView: host pointer must not be null");
        }
        validate_extents(exts);
    }

    /// Validate device-only construction preconditions.
    static void validate_device_construction(T *device_ptr, const extents_type &exts) {
        if (device_ptr == nullptr) {
            throw std::invalid_argument("FieldView: device pointer must not be null");
        }
        validate_extents(exts);
    }

    /// Validate that all extents are > 0.
    static void validate_extents(const extents_type &exts) {
        for (std::size_t i = 0; i < Rank; ++i) {
            if (exts[i] == 0) {
                throw std::invalid_argument("FieldView: extent[" + std::to_string(i) + "] must be > 0");
            }
        }
    }

    /// Determine the device memory space token based on the Kokkos backend.
    static constexpr MemorySpaceToken determine_device_memory_space() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
        return MemorySpaceToken::CudaDevice;
#elif defined(KOKKOS_ENABLE_HIP)
        return MemorySpaceToken::HipDevice;
#else
        // Fallback: if no GPU backend, device pointer still marked as CudaDevice
        // for consistency with the C constants
        return MemorySpaceToken::CudaDevice;
#endif
    }

    /// Build the mdspan from host_ptr_ and extents_ (index_sequence dispatch).
    template <std::size_t... Is>
    mdspan_type make_mdspan(std::index_sequence<Is...>) const {
        return mdspan_type(host_ptr_, extents_[Is]...);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Data members
    // ─────────────────────────────────────────────────────────────────────────

    T *host_ptr_ = nullptr;
    T *device_ptr_ = nullptr;
    extents_type extents_ = {};
    std::size_t total_size_ = 0;
    CoherencyState state_ = CoherencyState::HOST_CLEAN;
    MemorySpaceToken mem_space_ = MemorySpaceToken::Host;
    TripleBuffer<T> *triple_buf_ = nullptr;  // non-owning, optional
};

}  // namespace span

#endif  // SPAN_FIELD_VIEW_HPP
