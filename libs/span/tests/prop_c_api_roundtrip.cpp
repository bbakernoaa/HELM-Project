// SPDX-License-Identifier: Apache-2.0
// SPAN — Property-based tests for C-API Registration Round-Trip
// Feature: span-field-view, Property 2: C-API Registration Round-Trip
//
// **Validates: Requirements 2.2, 2.3, 2.4, 7.2, 7.3, 8.2**
//
// Property: For any valid non-null pointer p, valid dimensions array d
// (all elements > 0), valid rank r in [1, HELM_SPAN_MAX_RANK], and valid
// memory space token (HOST=0), calling helm_span_register_legacy_ptr(p, d, r,
// token, &h) followed by helm_span_get_view(h, &p_out, d_out, &r_out) SHALL
// return HELM_SPAN_SUCCESS with p_out == p, d_out[i] == d[i] for all i, and
// r_out == r.

#include "span/span_constants.h"
#include "handle_registry.hpp"

// Include the C-API implementation directly for self-contained testing
#include "span_c_api.cpp"

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdint>
#include <vector>

namespace {

/// Custom RapidCheck generator for a valid rank [1, HELM_SPAN_MAX_RANK].
rc::Gen<int> genValidRank() {
    return rc::gen::inRange(1, HELM_SPAN_MAX_RANK + 1);
}

/// Custom RapidCheck generator for a valid dims array of given rank.
/// Each dimension is in [1, 5] to keep total product bounded (max 5^7 = 78125 elements).
rc::Gen<std::vector<int64_t>> genValidDims(int rank) {
    return rc::gen::container<std::vector<int64_t>>(
        static_cast<std::size_t>(rank),
        rc::gen::inRange<int64_t>(1, 6)
    );
}

} // namespace

// ---------------------------------------------------------------------------
// Property 2: C-API Registration Round-Trip
//
// For any valid non-null pointer, dims (all > 0), rank in [1,7], and HOST
// memory space token, register + get_view round-trips all parameters.
//
// Validates: Requirements 2.2, 2.3, 2.4, 7.2, 7.3, 8.2
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiRoundTrip, RegisterAndRetrieve, ()) {
    // Generate random rank
    const int rank = *genValidRank();

    // Generate random dims for that rank
    const auto dims = *genValidDims(rank);

    // Compute total size for the buffer allocation
    int64_t total_size = 1;
    for (int i = 0; i < rank; ++i) {
        total_size *= dims[static_cast<std::size_t>(i)];
    }

    // Allocate a buffer as the "Fortran" pointer (non-null, valid memory)
    std::vector<double> buffer(static_cast<std::size_t>(total_size), 1.0);
    void* ptr = static_cast<void*>(buffer.data());

    // Register with HOST memory space (0) — the only valid one on CPU-only builds
    int handle = -1;
    int rc_reg = helm_span_register_legacy_ptr(
        ptr, dims.data(), rank, HELM_SPAN_MEM_HOST, &handle);

    // Registration must succeed
    RC_ASSERT(rc_reg == HELM_SPAN_SUCCESS);

    // Handle must be positive (Requirement 2.3)
    RC_ASSERT(handle > 0);

    // Retrieve via get_view
    const void* ptr_out = nullptr;
    int64_t dims_out[HELM_SPAN_MAX_RANK] = {};
    int rank_out = 0;

    int rc_get = helm_span_get_view(handle, &ptr_out, dims_out, &rank_out);

    // get_view must succeed
    RC_ASSERT(rc_get == HELM_SPAN_SUCCESS);

    // Round-trip: pointer identity (Requirement 8.2 — zero-copy)
    RC_ASSERT(reinterpret_cast<std::uintptr_t>(ptr_out) == reinterpret_cast<std::uintptr_t>(ptr));

    // Round-trip: rank matches
    RC_ASSERT(rank_out == rank);

    // Round-trip: dims match element-wise
    for (int i = 0; i < rank; ++i) {
        RC_ASSERT(dims_out[i] == dims[static_cast<std::size_t>(i)]);
    }

    // Cleanup: unregister handle
    int rc_unreg = helm_span_unregister(handle);
    RC_ASSERT(rc_unreg == HELM_SPAN_SUCCESS);
}

// ---------------------------------------------------------------------------
// Property 2 (variant): After unregister, get_view returns INVALID_HANDLE.
//
// Validates: Requirements 2.5, 2.8
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiRoundTrip, UnregisterInvalidatesHandle, ()) {
    const int rank = *genValidRank();
    const auto dims = *genValidDims(rank);

    int64_t total_size = 1;
    for (int i = 0; i < rank; ++i) {
        total_size *= dims[static_cast<std::size_t>(i)];
    }

    std::vector<double> buffer(static_cast<std::size_t>(total_size), 0.0);
    void* ptr = static_cast<void*>(buffer.data());

    int handle = -1;
    int rc_reg = helm_span_register_legacy_ptr(
        ptr, dims.data(), rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(rc_reg == HELM_SPAN_SUCCESS);
    RC_ASSERT(handle > 0);

    // Unregister
    int rc_unreg = helm_span_unregister(handle);
    RC_ASSERT(rc_unreg == HELM_SPAN_SUCCESS);

    // Now get_view on the same handle must fail
    const void* ptr_out = nullptr;
    int64_t dims_out[HELM_SPAN_MAX_RANK] = {};
    int rank_out = 0;

    int rc_get = helm_span_get_view(handle, &ptr_out, dims_out, &rank_out);
    RC_ASSERT(rc_get == HELM_SPAN_ERR_INVALID_HANDLE);
}

// ---------------------------------------------------------------------------
// Property 2 (variant): Multiple registrations produce distinct handles.
//
// Validates: Requirements 2.3, 10.2
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiRoundTrip, MultipleRegistrationsDistinctHandles, ()) {
    const int rank = *genValidRank();
    const auto dims = *genValidDims(rank);

    int64_t total_size = 1;
    for (int i = 0; i < rank; ++i) {
        total_size *= dims[static_cast<std::size_t>(i)];
    }

    // Create two buffers
    std::vector<double> buffer1(static_cast<std::size_t>(total_size), 1.0);
    std::vector<double> buffer2(static_cast<std::size_t>(total_size), 2.0);

    int handle1 = -1, handle2 = -1;

    int rc1 = helm_span_register_legacy_ptr(
        buffer1.data(), dims.data(), rank, HELM_SPAN_MEM_HOST, &handle1);
    int rc2 = helm_span_register_legacy_ptr(
        buffer2.data(), dims.data(), rank, HELM_SPAN_MEM_HOST, &handle2);

    RC_ASSERT(rc1 == HELM_SPAN_SUCCESS);
    RC_ASSERT(rc2 == HELM_SPAN_SUCCESS);
    RC_ASSERT(handle1 > 0);
    RC_ASSERT(handle2 > 0);

    // Handles must be distinct
    RC_ASSERT(handle1 != handle2);

    // Each get_view returns the correct pointer
    const void* p1_out = nullptr;
    const void* p2_out = nullptr;
    int64_t d1_out[HELM_SPAN_MAX_RANK] = {};
    int64_t d2_out[HELM_SPAN_MAX_RANK] = {};
    int r1_out = 0, r2_out = 0;

    RC_ASSERT(helm_span_get_view(handle1, &p1_out, d1_out, &r1_out) == HELM_SPAN_SUCCESS);
    RC_ASSERT(helm_span_get_view(handle2, &p2_out, d2_out, &r2_out) == HELM_SPAN_SUCCESS);

    RC_ASSERT(reinterpret_cast<std::uintptr_t>(p1_out) == reinterpret_cast<std::uintptr_t>(buffer1.data()));
    RC_ASSERT(reinterpret_cast<std::uintptr_t>(p2_out) == reinterpret_cast<std::uintptr_t>(buffer2.data()));

    // Cleanup
    helm_span_unregister(handle1);
    helm_span_unregister(handle2);
}
