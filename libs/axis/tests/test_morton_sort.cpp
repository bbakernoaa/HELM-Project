// AXIS unit test: Morton/Z-curve encoding and locality sort utilities
// Verifies morton_encode_2d, normalize_coord, and morton_sort_indices.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/morton_sort.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

namespace axis::test {

// ─────────────────────────────────────────────────────────────────────────────
// morton_encode_2d tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(MortonSort, EncodeZeroZero) {
    // (0, 0) should produce Morton code 0
    EXPECT_EQ(axis::detail::morton_encode_2d(0, 0), 0u);
}

TEST(MortonSort, EncodeOneZero) {
    // (1, 0): bit 0 of x goes to position 0 → result = 1
    EXPECT_EQ(axis::detail::morton_encode_2d(1, 0), 1u);
}

TEST(MortonSort, EncodeZeroOne) {
    // (0, 1): bit 0 of y goes to position 1 → result = 2
    EXPECT_EQ(axis::detail::morton_encode_2d(0, 1), 2u);
}

TEST(MortonSort, EncodeOneOne) {
    // (1, 1): x contributes bit 0 → pos 0 (value 1), y contributes bit 0 → pos 1 (value 2)
    // result = 1 + 2 = 3
    EXPECT_EQ(axis::detail::morton_encode_2d(1, 1), 3u);
}

TEST(MortonSort, EncodeTwoTwo) {
    // (2, 2): x=0b10 → spread: bit1 at pos 2 = 4; y=0b10 → spread: bit1 at pos 2, shifted left 1 = 8
    // result = 4 + 8 = 12
    EXPECT_EQ(axis::detail::morton_encode_2d(2, 2), 12u);
}

TEST(MortonSort, EncodeMaxValues) {
    // Maximum 16-bit values
    uint32_t max16 = (1u << 16) - 1;  // 65535
    uint64_t code = axis::detail::morton_encode_2d(max16, max16);
    // All 32 bits set in interleaved fashion → all lower 32 bits set
    EXPECT_EQ(code, 0xFFFFFFFFu);
}

TEST(MortonSort, EncodeSpatialOrdering) {
    // Points close in 2D space should have similar Morton codes.
    // (0,0), (1,0), (0,1), (1,1) form a 2×2 block at the finest level
    uint64_t c00 = axis::detail::morton_encode_2d(0, 0);
    uint64_t c10 = axis::detail::morton_encode_2d(1, 0);
    uint64_t c01 = axis::detail::morton_encode_2d(0, 1);
    uint64_t c11 = axis::detail::morton_encode_2d(1, 1);

    // Z-curve order: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
    EXPECT_EQ(c00, 0u);
    EXPECT_EQ(c10, 1u);
    EXPECT_EQ(c01, 2u);
    EXPECT_EQ(c11, 3u);
}

TEST(MortonSort, EncodePreservesInjection) {
    // Different (x, y) pairs produce different Morton codes
    uint64_t c_3_5 = axis::detail::morton_encode_2d(3, 5);
    uint64_t c_5_3 = axis::detail::morton_encode_2d(5, 3);
    EXPECT_NE(c_3_5, c_5_3);
}

// ─────────────────────────────────────────────────────────────────────────────
// normalize_coord tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(MortonSort, NormalizeMinReturnsZero) {
    EXPECT_EQ(axis::detail::normalize_coord(0.0, 0.0, 100.0), 0u);
}

TEST(MortonSort, NormalizeMaxReturnsN) {
    EXPECT_EQ(axis::detail::normalize_coord(100.0, 0.0, 100.0), 65535u);
}

TEST(MortonSort, NormalizeMidReturnsHalf) {
    uint32_t result = axis::detail::normalize_coord(50.0, 0.0, 100.0);
    // Should be approximately 65535/2 ≈ 32767
    EXPECT_NEAR(static_cast<double>(result), 32767.5, 1.0);
}

TEST(MortonSort, NormalizeBelowMinClampsToZero) {
    EXPECT_EQ(axis::detail::normalize_coord(-10.0, 0.0, 100.0), 0u);
}

TEST(MortonSort, NormalizeAboveMaxClampsToN) {
    EXPECT_EQ(axis::detail::normalize_coord(200.0, 0.0, 100.0), 65535u);
}

TEST(MortonSort, NormalizeDegenerateRangeReturnsZero) {
    // When max <= min, should return 0
    EXPECT_EQ(axis::detail::normalize_coord(5.0, 10.0, 10.0), 0u);
    EXPECT_EQ(axis::detail::normalize_coord(5.0, 10.0, 5.0), 0u);
}

TEST(MortonSort, NormalizeNegativeRange) {
    // Longitude range [-180, 180]
    uint32_t result = axis::detail::normalize_coord(0.0, -180.0, 180.0);
    EXPECT_NEAR(static_cast<double>(result), 32767.5, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// morton_sort_indices tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(MortonSort, SortIndicesEmpty) {
    using MemSpace = Kokkos::HostSpace;
    Kokkos::View<double *, MemSpace> cx("cx", 0);
    Kokkos::View<double *, MemSpace> cy("cy", 0);

    auto cx_const = Kokkos::View<const double *, MemSpace>(cx);
    auto cy_const = Kokkos::View<const double *, MemSpace>(cy);

    auto result = axis::detail::morton_sort_indices<MemSpace>(cx_const, cy_const, 0);
    EXPECT_EQ(result.extent(0), 0u);
}

TEST(MortonSort, SortIndicesSingleElement) {
    using MemSpace = Kokkos::HostSpace;
    Kokkos::View<double *, MemSpace> cx("cx", 1);
    Kokkos::View<double *, MemSpace> cy("cy", 1);
    cx(0) = 5.0;
    cy(0) = 3.0;

    auto cx_const = Kokkos::View<const double *, MemSpace>(cx);
    auto cy_const = Kokkos::View<const double *, MemSpace>(cy);

    auto result = axis::detail::morton_sort_indices<MemSpace>(cx_const, cy_const, 1);
    EXPECT_EQ(result.extent(0), 1u);
    EXPECT_EQ(result(0), 0);
}

TEST(MortonSort, SortIndicesProducesValidPermutation) {
    using MemSpace = Kokkos::HostSpace;
    const std::size_t n = 8;

    Kokkos::View<double *, MemSpace> cx("cx", n);
    Kokkos::View<double *, MemSpace> cy("cy", n);

    // Scatter points across a 2D domain
    cx(0) = 10.0;
    cy(0) = 10.0;  // bottom-left
    cx(1) = 90.0;
    cy(1) = 90.0;  // top-right
    cx(2) = 10.0;
    cy(2) = 90.0;  // top-left
    cx(3) = 90.0;
    cy(3) = 10.0;  // bottom-right
    cx(4) = 50.0;
    cy(4) = 50.0;  // center
    cx(5) = 25.0;
    cy(5) = 25.0;  // near bottom-left
    cx(6) = 75.0;
    cy(6) = 75.0;  // near top-right
    cx(7) = 50.0;
    cy(7) = 10.0;  // bottom-center

    auto cx_const = Kokkos::View<const double *, MemSpace>(cx);
    auto cy_const = Kokkos::View<const double *, MemSpace>(cy);

    auto perm = axis::detail::morton_sort_indices<MemSpace>(cx_const, cy_const, n);
    EXPECT_EQ(perm.extent(0), n);

    // Check it's a valid permutation: each index 0..n-1 appears exactly once
    std::vector<bool> seen(n, false);
    for (std::size_t k = 0; k < n; ++k) {
        auto idx = perm(k);
        ASSERT_GE(idx, 0);
        ASSERT_LT(static_cast<std::size_t>(idx), n);
        EXPECT_FALSE(seen[idx]) << "Index " << idx << " appears more than once";
        seen[idx] = true;
    }
}

TEST(MortonSort, SortIndicesGroupsSpatialNeighbors) {
    using MemSpace = Kokkos::HostSpace;
    const std::size_t n = 4;

    Kokkos::View<double *, MemSpace> cx("cx", n);
    Kokkos::View<double *, MemSpace> cy("cy", n);

    // Two clusters: (0,0), (1,1) are near each other; (100,100), (101,101) are near each other.
    // Morton sort should group each cluster consecutively.
    cx(0) = 100.0;
    cy(0) = 100.0;
    cx(1) = 0.0;
    cy(1) = 0.0;
    cx(2) = 101.0;
    cy(2) = 101.0;
    cx(3) = 1.0;
    cy(3) = 1.0;

    auto cx_const = Kokkos::View<const double *, MemSpace>(cx);
    auto cy_const = Kokkos::View<const double *, MemSpace>(cy);

    auto perm = axis::detail::morton_sort_indices<MemSpace>(cx_const, cy_const, n);

    // The two bottom-left points (indices 1, 3) should appear consecutively
    // and the two top-right points (indices 0, 2) should appear consecutively.
    std::vector<int64_t> order(n);
    for (std::size_t k = 0; k < n; ++k) {
        order[k] = perm(k);
    }

    // Find positions of indices 1 and 3 (bottom-left cluster)
    int pos_1 = -1, pos_3 = -1;
    for (int k = 0; k < static_cast<int>(n); ++k) {
        if (order[k] == 1) pos_1 = k;
        if (order[k] == 3) pos_3 = k;
    }
    EXPECT_EQ(std::abs(pos_1 - pos_3), 1) << "Spatially close points should be adjacent in Morton order";

    // Find positions of indices 0 and 2 (top-right cluster)
    int pos_0 = -1, pos_2 = -1;
    for (int k = 0; k < static_cast<int>(n); ++k) {
        if (order[k] == 0) pos_0 = k;
        if (order[k] == 2) pos_2 = k;
    }
    EXPECT_EQ(std::abs(pos_0 - pos_2), 1) << "Spatially close points should be adjacent in Morton order";
}

}  // namespace axis::test
