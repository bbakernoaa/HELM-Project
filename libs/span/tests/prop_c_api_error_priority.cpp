// SPDX-License-Identifier: Apache-2.0
// SPAN — Property-based tests for C-API Error Code Priority
// Feature: span-field-view, Property 3: C-API Error Code Priority
//
// **Validates: Requirements 2.6, 2.7, 2.9, 2.12, 12.3**
//
// For any input to helm_span_register_legacy_ptr that violates multiple
// preconditions simultaneously, the returned error code SHALL correspond to the
// highest-priority violated condition in the order:
//   null pointer (-1) > invalid rank (-2) > invalid memory space (-4) >
//   invalid dims (-1) > registry full (-5) > internal (-99).

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <span/span_constants.h>

#include <cstdint>
#include <vector>

extern "C" {
int helm_span_register_legacy_ptr(void *ptr, const int64_t *dims, int rank, int memory_space_token, int *handle_out);
int helm_span_unregister(int handle);
}

// ---------------------------------------------------------------------------
// Scenario 1: Null pointer takes priority over everything.
// null ptr + invalid rank + invalid memory space → ERR_NULL_PTR (-1)
// Validates: Requirements 2.6, 12.3
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, NullPtrHighestPriority, ()) {
    int bad_rank = *rc::gen::element(0, -1, 8, 99);
    int bad_mem = *rc::gen::element(-1, 3, 99);
    int64_t dims[] = {10};
    int handle;

    int result = helm_span_register_legacy_ptr(nullptr, dims, bad_rank, bad_mem, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_NULL_PTR);
}

// ---------------------------------------------------------------------------
// Scenario 2: Invalid rank takes priority over memory space and dims.
// valid ptr + invalid rank + invalid memory space → ERR_INVALID_RANK (-2)
// Validates: Requirements 2.7, 12.3
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidRankOverMemSpace, ()) {
    double buf[10];
    int bad_rank = *rc::gen::element(0, -1, 8, 99);
    int bad_mem = *rc::gen::element(-1, 3, 99);
    int64_t dims[] = {10};
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims, bad_rank, bad_mem, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_INVALID_RANK);
}

// ---------------------------------------------------------------------------
// Scenario 3: Invalid memory space takes priority over invalid dims.
// valid ptr + valid rank + invalid mem space + invalid dims → ERR_INVALID_MEMORY_SPACE (-4)
// Validates: Requirements 2.9, 12.3
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidMemSpaceOverDims, ()) {
    double buf[10];
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    int bad_mem = *rc::gen::element(-1, 3, 99);
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank), -1);  // all invalid dims
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims.data(), valid_rank, bad_mem, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_INVALID_MEMORY_SPACE);
}

// ---------------------------------------------------------------------------
// Scenario 4: Invalid dims returns error when all higher-priority checks pass.
// valid ptr + valid rank + valid mem (HOST=0) + bad dims → negative error
// Validates: Requirements 2.12, 12.3
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidDimsReturnsError, ()) {
    double buf[100];
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank), 10);

    // Make one dimension invalid (zero or negative)
    int bad_idx = *rc::gen::inRange(0, valid_rank);
    dims[static_cast<std::size_t>(bad_idx)] = *rc::gen::element<int64_t>(0, -1, -99);
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims.data(), valid_rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(result < 0);  // Some negative error code (ERR_NULL_PTR reused for dims)
}

// ---------------------------------------------------------------------------
// Scenario 5 (individual errors): null ptr alone → ERR_NULL_PTR
// Validates: Requirements 2.6
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, NullPtrAlone, ()) {
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank), 10);
    int handle;

    int result = helm_span_register_legacy_ptr(nullptr, dims.data(), valid_rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_NULL_PTR);
}

// ---------------------------------------------------------------------------
// Scenario 5 (individual errors): invalid rank alone → ERR_INVALID_RANK
// Validates: Requirements 2.7
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidRankAlone, ()) {
    double buf[10];
    int bad_rank = *rc::gen::element(0, -1, 8, 99);
    int64_t dims[] = {10, 10, 10, 10, 10, 10, 10};
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims, bad_rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_INVALID_RANK);
}

// ---------------------------------------------------------------------------
// Scenario 5 (individual errors): invalid memory space alone → ERR_INVALID_MEMORY_SPACE
// Note: On host-only Docker build, tokens 1 (CUDA) and 2 (HIP) are INVALID.
// Validates: Requirements 2.9
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidMemSpaceAlone, ()) {
    double buf[10];
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank), 10);
    int bad_mem = *rc::gen::element(-1, 3, 99);
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims.data(), valid_rank, bad_mem, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_INVALID_MEMORY_SPACE);
}

// ---------------------------------------------------------------------------
// Scenario 5 (individual errors): invalid dims alone → negative error
// Validates: Requirements 2.12
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, InvalidDimsAlone, ()) {
    double buf[100];
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank), 10);

    // Make exactly one dim invalid
    int bad_idx = *rc::gen::inRange(0, valid_rank);
    dims[static_cast<std::size_t>(bad_idx)] = *rc::gen::element<int64_t>(0, -1, -99);
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims.data(), valid_rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(result == HELM_SPAN_ERR_NULL_PTR);  // dims error reuses ERR_NULL_PTR
}

// ---------------------------------------------------------------------------
// Scenario 5 (success case): valid everything → SUCCESS (0)
// Validates: Requirements 2.6, 2.7, 2.9, 2.12
// ---------------------------------------------------------------------------
RC_GTEST_PROP(CApiErrorPriority, SuccessCase, ()) {
    double buf[1024];
    int valid_rank = *rc::gen::inRange(1, 8);  // [1, 7]
    std::vector<int64_t> dims(static_cast<std::size_t>(valid_rank));

    // Generate valid positive dims
    for (auto &d : dims) {
        d = *rc::gen::inRange<int64_t>(1, 100);
    }
    int handle;

    int result = helm_span_register_legacy_ptr(buf, dims.data(), valid_rank, HELM_SPAN_MEM_HOST, &handle);
    RC_ASSERT(result == HELM_SPAN_SUCCESS);
    RC_ASSERT(handle > 0);

    // Cleanup: unregister to keep registry clean between iterations
    helm_span_unregister(handle);
}
