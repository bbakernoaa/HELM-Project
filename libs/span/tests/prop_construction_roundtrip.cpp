// ─── Property-Based Tests: FieldView Construction Round-Trip ─────────────────
// Feature: span-field-view, Property 1: Construction Round-Trip Identity
//
// For any valid non-null pointer p and valid extents array e (all elements > 0),
// constructing a FieldView<T, Rank>(p, e) SHALL produce an instance where
// host_data() == p, extent(i) == e[i] for all i in [0, Rank), size() == product(e),
// and view().data_handle() == p.
//
// **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 4.4, 4.5, 4.6, 8.1, 8.3, 8.6**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <array>
#include <cstddef>
#include <numeric>
#include <span/field_view.hpp>
#include <vector>

namespace {

// Helper: generate a random extent in [1, 1024]
rc::Gen<std::size_t> genExtent() {
    return rc::gen::inRange<std::size_t>(1, 1025);
}

}  // namespace

// ─── Host-Only Rank 1 ────────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, HostOnlyRank1, ()) {
    const auto ext0 = *genExtent();
    std::vector<double> buf(ext0);
    std::array<std::size_t, 1> exts{ext0};

    span::FieldView<double, 1> fv(buf.data(), exts);

    // Core round-trip postconditions
    RC_ASSERT(fv.host_data() == buf.data());
    RC_ASSERT(fv.data() == buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.size() == ext0);
    RC_ASSERT(fv.view().data_handle() == buf.data());

    // Validity and state postconditions
    RC_ASSERT(fv.valid());
    RC_ASSERT(fv.has_host_ptr());
    RC_ASSERT(!fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() == span::MemorySpaceToken::Host);
}

// ─── Host-Only Rank 2 ────────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, HostOnlyRank2, ()) {
    const auto ext0 = *genExtent();
    const auto ext1 = *genExtent();
    const std::size_t total = ext0 * ext1;
    std::vector<double> buf(total);
    std::array<std::size_t, 2> exts{ext0, ext1};

    span::FieldView<double, 2> fv(buf.data(), exts);

    RC_ASSERT(fv.host_data() == buf.data());
    RC_ASSERT(fv.data() == buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.extent(1) == ext1);
    RC_ASSERT(fv.size() == total);
    RC_ASSERT(fv.view().data_handle() == buf.data());

    RC_ASSERT(fv.valid());
    RC_ASSERT(fv.has_host_ptr());
    RC_ASSERT(!fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() == span::MemorySpaceToken::Host);
}

// ─── Host-Only Rank 3 ────────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, HostOnlyRank3, ()) {
    const auto ext0 = *genExtent();
    const auto ext1 = *genExtent();
    const auto ext2 = *genExtent();
    const std::size_t total = ext0 * ext1 * ext2;
    std::vector<double> buf(total);
    std::array<std::size_t, 3> exts{ext0, ext1, ext2};

    span::FieldView<double, 3> fv(buf.data(), exts);

    RC_ASSERT(fv.host_data() == buf.data());
    RC_ASSERT(fv.data() == buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.extent(1) == ext1);
    RC_ASSERT(fv.extent(2) == ext2);
    RC_ASSERT(fv.size() == total);
    RC_ASSERT(fv.view().data_handle() == buf.data());

    RC_ASSERT(fv.valid());
    RC_ASSERT(fv.has_host_ptr());
    RC_ASSERT(!fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() == span::MemorySpaceToken::Host);
}

// ─── Device-Only Rank 1 ─────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, DeviceOnlyRank1, ()) {
    const auto ext0 = *genExtent();
    // Simulate a device pointer with a host buffer (for testing purposes,
    // the constructor accepts any non-null T* as a device pointer)
    std::vector<double> dev_buf(ext0);
    std::array<std::size_t, 1> exts{ext0};

    span::FieldView<double, 1> fv(nullptr, dev_buf.data(), exts);

    RC_ASSERT(fv.host_data() == nullptr);
    RC_ASSERT(fv.device_data() == dev_buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.size() == ext0);

    RC_ASSERT(fv.valid());
    RC_ASSERT(!fv.has_host_ptr());
    RC_ASSERT(fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    // Device-only memory space should be CudaDevice (or HipDevice)
    RC_ASSERT(fv.memory_space() != span::MemorySpaceToken::Host);
}

// ─── Device-Only Rank 2 ─────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, DeviceOnlyRank2, ()) {
    const auto ext0 = *genExtent();
    const auto ext1 = *genExtent();
    const std::size_t total = ext0 * ext1;
    std::vector<double> dev_buf(total);
    std::array<std::size_t, 2> exts{ext0, ext1};

    span::FieldView<double, 2> fv(nullptr, dev_buf.data(), exts);

    RC_ASSERT(fv.host_data() == nullptr);
    RC_ASSERT(fv.device_data() == dev_buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.extent(1) == ext1);
    RC_ASSERT(fv.size() == total);

    RC_ASSERT(fv.valid());
    RC_ASSERT(!fv.has_host_ptr());
    RC_ASSERT(fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() != span::MemorySpaceToken::Host);
}

// ─── Dual-Pointer Rank 1 ────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, DualPointerRank1, ()) {
    const auto ext0 = *genExtent();
    std::vector<double> host_buf(ext0);
    std::vector<double> dev_buf(ext0);
    std::array<std::size_t, 1> exts{ext0};

    span::FieldView<double, 1> fv(host_buf.data(), dev_buf.data(), exts);

    RC_ASSERT(fv.host_data() == host_buf.data());
    RC_ASSERT(fv.device_data() == dev_buf.data());
    RC_ASSERT(fv.data() == host_buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.size() == ext0);
    RC_ASSERT(fv.view().data_handle() == host_buf.data());

    RC_ASSERT(fv.valid());
    RC_ASSERT(fv.has_host_ptr());
    RC_ASSERT(fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() == span::MemorySpaceToken::Host);
}

// ─── Dual-Pointer Rank 3 ────────────────────────────────────────────────────

RC_GTEST_PROP(ConstructionRoundTrip, DualPointerRank3, ()) {
    const auto ext0 = *genExtent();
    const auto ext1 = *genExtent();
    const auto ext2 = *genExtent();
    const std::size_t total = ext0 * ext1 * ext2;
    std::vector<double> host_buf(total);
    std::vector<double> dev_buf(total);
    std::array<std::size_t, 3> exts{ext0, ext1, ext2};

    span::FieldView<double, 3> fv(host_buf.data(), dev_buf.data(), exts);

    RC_ASSERT(fv.host_data() == host_buf.data());
    RC_ASSERT(fv.device_data() == dev_buf.data());
    RC_ASSERT(fv.data() == host_buf.data());
    RC_ASSERT(fv.extent(0) == ext0);
    RC_ASSERT(fv.extent(1) == ext1);
    RC_ASSERT(fv.extent(2) == ext2);
    RC_ASSERT(fv.size() == total);
    RC_ASSERT(fv.view().data_handle() == host_buf.data());

    RC_ASSERT(fv.valid());
    RC_ASSERT(fv.has_host_ptr());
    RC_ASSERT(fv.has_device_ptr());
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
    RC_ASSERT(fv.memory_space() == span::MemorySpaceToken::Host);
}
