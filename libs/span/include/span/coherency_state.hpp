/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>

namespace span {

/// Tracks host/device synchronization state for a FieldView.
/// At most one of host or device may be considered dirty at any time.
enum class CoherencyState : int32_t {
    HOST_CLEAN = 0,   // Both host and device in sync; host is canonical
    HOST_DIRTY = 1,   // Host written; device copy stale
    DEVICE_DIRTY = 2  // Device written; host copy stale
};

/// Identifies where a registered pointer resides.
/// Values match the C constants in span_constants.h.
enum class MemorySpaceToken : int32_t {
    Host = 0,        // matches HELM_SPAN_MEM_HOST
    CudaDevice = 1,  // matches HELM_SPAN_MEM_CUDA_DEVICE
    HipDevice = 2    // matches HELM_SPAN_MEM_HIP_DEVICE
};

}  // namespace span
