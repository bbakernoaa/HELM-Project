// SPDX-License-Identifier: Apache-2.0
// C-API smoke tests (Task 10.3)
// Verifies basic helm_span_* C function wrappers for Fortran interop.

#include <span/span_constants.h>
#include "handle_registry.hpp"
#include "span_c_api.cpp"  // Include directly for testing without shared lib

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

TEST(CApiSmoke, RegisterGetUnregisterFlow) {
    std::vector<double> buffer(100, 3.14);
    int64_t dims[] = {10, 10};
    int handle = 0;

    int rc = helm_span_register_legacy_ptr(buffer.data(), dims, 2, HELM_SPAN_MEM_HOST, &handle);
    EXPECT_EQ(rc, HELM_SPAN_SUCCESS);
    EXPECT_GT(handle, 0);

    const void* ptr_out = nullptr;
    int64_t dims_out[7] = {};
    int rank_out = 0;
    rc = helm_span_get_view(handle, &ptr_out, dims_out, &rank_out);
    EXPECT_EQ(rc, HELM_SPAN_SUCCESS);
    EXPECT_EQ(ptr_out, static_cast<void*>(buffer.data()));
    EXPECT_EQ(rank_out, 2);
    EXPECT_EQ(dims_out[0], 10);
    EXPECT_EQ(dims_out[1], 10);

    rc = helm_span_unregister(handle);
    EXPECT_EQ(rc, HELM_SPAN_SUCCESS);

    // After unregister, get_view should fail
    rc = helm_span_get_view(handle, &ptr_out, dims_out, &rank_out);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_HANDLE);
}

TEST(CApiSmoke, NullPointerReturnsError) {
    int64_t dims[] = {10};
    int handle = 0;
    int rc = helm_span_register_legacy_ptr(nullptr, dims, 1, HELM_SPAN_MEM_HOST, &handle);
    EXPECT_EQ(rc, HELM_SPAN_ERR_NULL_PTR);
}

TEST(CApiSmoke, InvalidRankReturnsError) {
    double buf[10];
    int64_t dims[] = {10};
    int handle = 0;

    int rc = helm_span_register_legacy_ptr(buf, dims, 0, HELM_SPAN_MEM_HOST, &handle);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_RANK);

    rc = helm_span_register_legacy_ptr(buf, dims, 8, HELM_SPAN_MEM_HOST, &handle);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_RANK);
}

TEST(CApiSmoke, InvalidMemorySpaceReturnsError) {
    double buf[10];
    int64_t dims[] = {10};
    int handle = 0;
    // On host-only build, CUDA (1) is invalid
    int rc = helm_span_register_legacy_ptr(buf, dims, 1, HELM_SPAN_MEM_CUDA_DEVICE, &handle);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_MEMORY_SPACE);
}

TEST(CApiSmoke, InvalidDimsReturnsError) {
    double buf[10];
    int64_t dims[] = {0};  // zero dim
    int handle = 0;
    int rc = helm_span_register_legacy_ptr(buf, dims, 1, HELM_SPAN_MEM_HOST, &handle);
    EXPECT_LT(rc, 0);  // Some negative error
}

TEST(CApiSmoke, UnregisterInvalidHandleReturnsError) {
    int rc = helm_span_unregister(0);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_HANDLE);

    rc = helm_span_unregister(999999);
    EXPECT_EQ(rc, HELM_SPAN_ERR_INVALID_HANDLE);
}
