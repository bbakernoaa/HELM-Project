// SPDX-License-Identifier: Apache-2.0
// SPAN — Legacy C-API implementation for Fortran interoperability
// Copyright (c) HELM Project Contributors

/// @file span_c_api.cpp
/// @brief Implements the extern "C" registration functions that allow Fortran
///        NUOPC caps to register raw pointers with SPAN. All functions use an
///        exception barrier macro so no C++ exception crosses the ABI boundary.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "handle_registry.hpp"
#include "span/field_view.hpp"
#include "span/span_constants.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Exception barrier macro — catches all C++ exceptions at the extern "C" boundary
// ═══════════════════════════════════════════════════════════════════════════════

#define HELM_SPAN_C_TRY(body)          \
    try {                              \
        body                           \
    } catch (...) {                    \
        return HELM_SPAN_ERR_INTERNAL; \
    }

// ═══════════════════════════════════════════════════════════════════════════════
// Internal: RegisteredField — type-erased storage for a FieldView + metadata
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

/// Type-erased record holding a registered FieldView plus retrieval metadata.
struct RegisteredField {
    std::shared_ptr<void> field_view;  ///< Type-erased FieldView<T, Rank>
    int rank;                          ///< Runtime rank (1–7)
    span::MemorySpaceToken mem_space;  ///< Where the pointer resides
    std::vector<int64_t> dims;         ///< Copy of dims for get_view retrieval
    void *raw_ptr;                     ///< Original pointer for get_view
};

// ═══════════════════════════════════════════════════════════════════════════════
// Internal: memory space validation
// ═══════════════════════════════════════════════════════════════════════════════

/// Validate that the memory_space_token is in-range AND that the corresponding
/// Kokkos backend is enabled at compile time (Requirement 7.4, 7.5).
static bool is_memory_space_valid(int token) {
    switch (token) {
        case HELM_SPAN_MEM_HOST:
            return true;

        case HELM_SPAN_MEM_CUDA_DEVICE:
#if defined(KOKKOS_ENABLE_CUDA)
            return true;
#else
            return false;
#endif

        case HELM_SPAN_MEM_HIP_DEVICE:
#if defined(KOKKOS_ENABLE_HIP)
            return true;
#else
            return false;
#endif

        default:
            return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal: FieldView instantiation helpers (rank dispatch)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert raw dims array to std::array<std::size_t, N> for FieldView construction.
template <std::size_t N>
static std::array<std::size_t, N> make_extents(const int64_t *dims) {
    std::array<std::size_t, N> exts{};
    for (std::size_t i = 0; i < N; ++i) {
        exts[i] = static_cast<std::size_t>(dims[i]);
    }
    return exts;
}

/// Instantiate a FieldView<double, N> and wrap in a shared_ptr<void>.
/// Host pointers are stored in the host slot; device pointers in the device slot.
template <std::size_t N>
static std::shared_ptr<void> create_field_view(void *ptr, const int64_t *dims, span::MemorySpaceToken mem_space) {
    auto exts = make_extents<N>(dims);
    auto *typed_ptr = static_cast<double *>(ptr);

    if (mem_space == span::MemorySpaceToken::Host) {
        // Host pointer → host-only FieldView
        return std::make_shared<span::FieldView<double, N>>(typed_ptr, exts);
    } else {
        // Device pointer → device-only FieldView (nullptr for host, ptr for device)
        return std::make_shared<span::FieldView<double, N>>(nullptr, typed_ptr, exts);
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════════
// extern "C" API implementation
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" {

int helm_span_register_legacy_ptr(void *ptr, const int64_t *dims, int rank, int memory_space_token, int *handle_out) {
    HELM_SPAN_C_TRY(
        // ── Priority 1: null pointer check (Requirement 2.6, 12.3) ──
        if (ptr == nullptr) { return HELM_SPAN_ERR_NULL_PTR; }

        // ── Priority 2: invalid rank check (Requirement 2.7, 12.3) ──
        if (rank < 1 || rank > HELM_SPAN_MAX_RANK) { return HELM_SPAN_ERR_INVALID_RANK; }

        // ── Priority 3: invalid memory space check (Requirement 2.9, 7.4, 7.5) ──
        if (!is_memory_space_valid(memory_space_token)) { return HELM_SPAN_ERR_INVALID_MEMORY_SPACE; }

        // ── Priority 4: invalid dims check (Requirement 2.12) ──
        for (int i = 0; i < rank; ++i) {
            if (dims[i] <= 0) {
                return HELM_SPAN_ERR_NULL_PTR;
            }
        }

        // ── Determine memory space enum ──
        auto mem_space = static_cast<span::MemorySpaceToken>(memory_space_token);

        // ── Instantiate FieldView<double, N> based on runtime rank ──
        std::shared_ptr<void> fv_ptr; switch (rank) {
            case 1:
                fv_ptr = create_field_view<1>(ptr, dims, mem_space);
                break;
            case 2:
                fv_ptr = create_field_view<2>(ptr, dims, mem_space);
                break;
            case 3:
                fv_ptr = create_field_view<3>(ptr, dims, mem_space);
                break;
            case 4:
                fv_ptr = create_field_view<4>(ptr, dims, mem_space);
                break;
            case 5:
                fv_ptr = create_field_view<5>(ptr, dims, mem_space);
                break;
            case 6:
                fv_ptr = create_field_view<6>(ptr, dims, mem_space);
                break;
            case 7:
                fv_ptr = create_field_view<7>(ptr, dims, mem_space);
                break;
            default:
                return HELM_SPAN_ERR_INVALID_RANK;  // unreachable
        }

                                      // ── Build the RegisteredField metadata struct ──
                                      auto reg_field = std::make_shared<RegisteredField>();
        reg_field->field_view = std::move(fv_ptr); reg_field->rank = rank; reg_field->mem_space = mem_space;
        reg_field->dims.assign(dims, dims + rank); reg_field->raw_ptr = ptr;

        // ── Register in the handle registry ──
        auto &registry = span::detail::HandleRegistry::instance();
        int handle = registry.register_view(std::static_pointer_cast<void>(reg_field));

        // ── Priority 5: registry full check (Requirement 10.8) ──
        if (handle == HELM_SPAN_ERR_REGISTRY_FULL) { return HELM_SPAN_ERR_REGISTRY_FULL; }

            // ── Success: write handle out ──
            *handle_out = handle;
        return HELM_SPAN_SUCCESS;)
}

int helm_span_get_view(int handle, const void **ptr_out, int64_t *dims_out, int *rank_out) {
    HELM_SPAN_C_TRY(
        auto &registry = span::detail::HandleRegistry::instance(); auto obj = registry.lookup(handle);

        if (!obj) { return HELM_SPAN_ERR_INVALID_HANDLE; }

        auto *reg = static_cast<RegisteredField *>(obj.get());

            // Write outputs
                *ptr_out = reg->raw_ptr;
        *rank_out = reg->rank; for (int i = 0; i < reg->rank; ++i) { dims_out[i] = reg->dims[static_cast<std::size_t>(i)]; }

        return HELM_SPAN_SUCCESS;)
}

int helm_span_unregister(int handle) {
    HELM_SPAN_C_TRY(auto &registry = span::detail::HandleRegistry::instance();

                    if (!registry.unregister(handle)) { return HELM_SPAN_ERR_INVALID_HANDLE; }

                    return HELM_SPAN_SUCCESS;)
}

}  // extern "C"
