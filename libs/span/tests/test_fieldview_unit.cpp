// SPDX-License-Identifier: Apache-2.0
// Unit tests for span::FieldView<T, Rank> construction and accessors (Task 3.1)

#include <span/field_view.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
// Default Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewConstruction, DefaultConstructorIsInvalid) {
    span::FieldView<double, 2> fv;

    EXPECT_FALSE(fv.valid());
    EXPECT_EQ(fv.host_data(), nullptr);
    EXPECT_EQ(fv.device_data(), nullptr);
    EXPECT_EQ(fv.data(), nullptr);
    EXPECT_EQ(fv.size(), 0u);
    EXPECT_FALSE(fv.has_host_ptr());
    EXPECT_FALSE(fv.has_device_ptr());
    EXPECT_FALSE(fv.is_triple_buffered());
    EXPECT_EQ(fv.coherency_state(), span::CoherencyState::HOST_CLEAN);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host-Only Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewConstruction, HostOnlyRank1) {
    std::vector<double> buffer(100, 1.0);
    std::array<std::size_t, 1> exts = {100};

    span::FieldView<double, 1> fv(buffer.data(), exts);

    EXPECT_TRUE(fv.valid());
    EXPECT_EQ(fv.host_data(), buffer.data());
    EXPECT_EQ(fv.data(), buffer.data());
    EXPECT_EQ(fv.device_data(), nullptr);
    EXPECT_EQ(fv.size(), 100u);
    EXPECT_EQ(fv.extent(0), 100u);
    EXPECT_EQ(fv.rank(), 1u);
    EXPECT_TRUE(fv.has_host_ptr());
    EXPECT_FALSE(fv.has_device_ptr());
    EXPECT_EQ(fv.memory_space(), span::MemorySpaceToken::Host);
    EXPECT_EQ(fv.coherency_state(), span::CoherencyState::HOST_CLEAN);
}

TEST(FieldViewConstruction, HostOnlyRank3) {
    std::vector<float> buffer(24);
    std::array<std::size_t, 3> exts = {2, 3, 4};

    span::FieldView<float, 3> fv(buffer.data(), exts);

    EXPECT_TRUE(fv.valid());
    EXPECT_EQ(fv.size(), 24u);
    EXPECT_EQ(fv.extent(0), 2u);
    EXPECT_EQ(fv.extent(1), 3u);
    EXPECT_EQ(fv.extent(2), 4u);
    EXPECT_EQ(fv.rank(), 3u);

    const auto& stored_exts = fv.extents();
    EXPECT_EQ(stored_exts[0], 2u);
    EXPECT_EQ(stored_exts[1], 3u);
    EXPECT_EQ(stored_exts[2], 4u);
}

TEST(FieldViewConstruction, HostOnlyNullPtrThrows) {
    std::array<std::size_t, 2> exts = {10, 10};
    EXPECT_THROW(
        (span::FieldView<double, 2>(nullptr, exts)),
        std::invalid_argument
    );
}

TEST(FieldViewConstruction, HostOnlyZeroExtentThrows) {
    std::vector<double> buffer(10);
    std::array<std::size_t, 2> exts = {10, 0};  // second extent is 0
    EXPECT_THROW(
        (span::FieldView<double, 2>(buffer.data(), exts)),
        std::invalid_argument
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Device-Only Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewConstruction, DeviceOnlyRank1) {
    // Simulate a device pointer (in CPU-only testing, just use a heap pointer)
    std::vector<double> device_buffer(50);
    std::array<std::size_t, 1> exts = {50};

    span::FieldView<double, 1> fv(nullptr, device_buffer.data(), exts);

    EXPECT_TRUE(fv.valid());
    EXPECT_EQ(fv.host_data(), nullptr);
    EXPECT_EQ(fv.data(), nullptr);  // data() == host_data()
    EXPECT_EQ(fv.device_data(), device_buffer.data());
    EXPECT_EQ(fv.size(), 50u);
    EXPECT_FALSE(fv.has_host_ptr());
    EXPECT_TRUE(fv.has_device_ptr());
    // memory_space should be CudaDevice (or HipDevice on HIP backends)
    EXPECT_NE(fv.memory_space(), span::MemorySpaceToken::Host);
    EXPECT_EQ(fv.coherency_state(), span::CoherencyState::HOST_CLEAN);
}

TEST(FieldViewConstruction, DeviceOnlyNullDevicePtrThrows) {
    std::array<std::size_t, 1> exts = {10};
    EXPECT_THROW(
        (span::FieldView<double, 1>(nullptr, static_cast<double*>(nullptr), exts)),
        std::invalid_argument
    );
}

TEST(FieldViewConstruction, DeviceOnlyZeroExtentThrows) {
    std::vector<double> device_buffer(10);
    std::array<std::size_t, 2> exts = {0, 5};
    EXPECT_THROW(
        (span::FieldView<double, 2>(nullptr, device_buffer.data(), exts)),
        std::invalid_argument
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dual-Pointer Construction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewConstruction, DualPointerBothValid) {
    std::vector<double> host_buf(60);
    std::vector<double> device_buf(60);
    std::array<std::size_t, 2> exts = {6, 10};

    span::FieldView<double, 2> fv(host_buf.data(), device_buf.data(), exts);

    EXPECT_TRUE(fv.valid());
    EXPECT_EQ(fv.host_data(), host_buf.data());
    EXPECT_EQ(fv.device_data(), device_buf.data());
    EXPECT_EQ(fv.size(), 60u);
    EXPECT_TRUE(fv.has_host_ptr());
    EXPECT_TRUE(fv.has_device_ptr());
    EXPECT_EQ(fv.memory_space(), span::MemorySpaceToken::Host);
    EXPECT_EQ(fv.coherency_state(), span::CoherencyState::HOST_CLEAN);
}

TEST(FieldViewConstruction, DualPointerBothNull) {
    // Req 4.11: both null → invalid
    std::array<std::size_t, 1> exts = {10};
    span::FieldView<double, 1> fv(
        static_cast<double*>(nullptr),
        static_cast<double*>(nullptr),
        exts
    );

    EXPECT_FALSE(fv.valid());
    EXPECT_EQ(fv.host_data(), nullptr);
    EXPECT_EQ(fv.device_data(), nullptr);
    EXPECT_EQ(fv.size(), 0u);
}

TEST(FieldViewConstruction, DualPointerHostOnlyProvided) {
    std::vector<double> host_buf(30);
    std::array<std::size_t, 1> exts = {30};

    span::FieldView<double, 1> fv(
        host_buf.data(),
        static_cast<double*>(nullptr),
        exts
    );

    EXPECT_TRUE(fv.valid());
    EXPECT_TRUE(fv.has_host_ptr());
    EXPECT_FALSE(fv.has_device_ptr());
    EXPECT_EQ(fv.memory_space(), span::MemorySpaceToken::Host);
}

TEST(FieldViewConstruction, DualPointerDeviceOnlyProvided) {
    std::vector<double> device_buf(30);
    std::array<std::size_t, 1> exts = {30};

    span::FieldView<double, 1> fv(
        static_cast<double*>(nullptr),
        device_buf.data(),
        exts
    );

    EXPECT_TRUE(fv.valid());
    EXPECT_FALSE(fv.has_host_ptr());
    EXPECT_TRUE(fv.has_device_ptr());
    EXPECT_NE(fv.memory_space(), span::MemorySpaceToken::Host);
}

// ═══════════════════════════════════════════════════════════════════════════════
// view() accessor
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewAccessors, ViewReturnsMdspan) {
    std::vector<double> buffer(12, 0.0);
    for (std::size_t i = 0; i < 12; ++i) buffer[i] = static_cast<double>(i);
    std::array<std::size_t, 2> exts = {3, 4};

    span::FieldView<double, 2> fv(buffer.data(), exts);
    auto mdspan_view = fv.view();

    // Verify data_handle points to same memory
    EXPECT_EQ(mdspan_view.data_handle(), buffer.data());

    // Verify extents match
    EXPECT_EQ(mdspan_view.extent(0), 3u);
    EXPECT_EQ(mdspan_view.extent(1), 4u);

    // Verify column-major access (layout_left):
    // buffer[0] = (0,0), buffer[1] = (1,0), buffer[2] = (2,0),
    // buffer[3] = (0,1), buffer[4] = (1,1), ...
    EXPECT_EQ(mdspan_view(0, 0), 0.0);
    EXPECT_EQ(mdspan_view(1, 0), 1.0);
    EXPECT_EQ(mdspan_view(2, 0), 2.0);
    EXPECT_EQ(mdspan_view(0, 1), 3.0);
}

TEST(FieldViewAccessors, ViewThrowsOnNullHost) {
    span::FieldView<double, 1> fv;
    EXPECT_THROW(fv.view(), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════════
// extent(dim) with bounds checking
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewAccessors, ExtentThrowsOnOutOfRange) {
    std::vector<double> buffer(10);
    std::array<std::size_t, 2> exts = {2, 5};

    span::FieldView<double, 2> fv(buffer.data(), exts);

    EXPECT_EQ(fv.extent(0), 2u);
    EXPECT_EQ(fv.extent(1), 5u);

    EXPECT_THROW(fv.extent(2), std::out_of_range);
    EXPECT_THROW(fv.extent(100), std::out_of_range);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zero-copy guarantee
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewZeroCopy, ConstructionDoesNotAllocate) {
    // The test verifies that construction merely stores pointers and metadata,
    // not that it allocates new memory. We verify by checking pointer identity.
    std::vector<double> buffer(1000, 42.0);
    std::array<std::size_t, 1> exts = {1000};

    span::FieldView<double, 1> fv(buffer.data(), exts);

    // Pointer identity proves no copy
    EXPECT_EQ(fv.host_data(), buffer.data());
    EXPECT_EQ(fv.data(), buffer.data());
    EXPECT_EQ(fv.view().data_handle(), buffer.data());

    // Mutating through the original buffer is visible via the view
    buffer[0] = 99.0;
    EXPECT_EQ(fv.view()(0), 99.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// is_triple_buffered() for unattached views
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewAccessors, IsTripleBufferedFalseByDefault) {
    std::vector<double> buffer(10);
    std::array<std::size_t, 1> exts = {10};

    span::FieldView<double, 1> fv(buffer.data(), exts);
    EXPECT_FALSE(fv.is_triple_buffered());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rank static constexpr
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FieldViewAccessors, RankIsCompileTimeConstant) {
    static_assert(span::FieldView<double, 1>::rank() == 1);
    static_assert(span::FieldView<double, 3>::rank() == 3);
    static_assert(span::FieldView<float, 7>::rank() == 7);
}
