// AXIS unit test: PlanarPolygon and PlanarClipper (Sutherland-Hodgman 2D)
// Verifies push/clear/empty/area on PlanarPolygon, and overlap_area on
// PlanarClipper for various geometric configurations.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/planar_clipper.hpp>
#include <cmath>

namespace axis::test {

// ─────────────────────────────────────────────────────────────────────────────
// PlanarPolygon basic operations
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarPolygon, DefaultConstructEmpty) {
    axis::detail::PlanarPolygon<32> poly;
    EXPECT_TRUE(poly.empty());
    EXPECT_EQ(poly.n, 0);
    EXPECT_DOUBLE_EQ(poly.area(), 0.0);
}

TEST(PlanarPolygon, PushAndCount) {
    axis::detail::PlanarPolygon<32> poly;
    poly.push(0.0, 0.0);
    poly.push(1.0, 0.0);
    EXPECT_TRUE(poly.empty());  // < 3 vertices
    EXPECT_EQ(poly.n, 2);

    poly.push(1.0, 1.0);
    EXPECT_FALSE(poly.empty());
    EXPECT_EQ(poly.n, 3);
}

TEST(PlanarPolygon, Clear) {
    axis::detail::PlanarPolygon<32> poly;
    poly.push(0.0, 0.0);
    poly.push(1.0, 0.0);
    poly.push(1.0, 1.0);
    EXPECT_FALSE(poly.empty());

    poly.clear();
    EXPECT_TRUE(poly.empty());
    EXPECT_EQ(poly.n, 0);
}

TEST(PlanarPolygon, PushClampsAtMaxVerts) {
    axis::detail::PlanarPolygon<8> poly;
    for (int i = 0; i < 20; ++i) {
        poly.push(static_cast<double>(i), 0.0);
    }
    EXPECT_EQ(poly.n, 8);  // clamped at MaxVerts
}

TEST(PlanarPolygon, AreaUnitSquare) {
    // Unit square: (0,0), (1,0), (1,1), (0,1) → area = 1.0
    axis::detail::PlanarPolygon<32> poly;
    poly.push(0.0, 0.0);
    poly.push(1.0, 0.0);
    poly.push(1.0, 1.0);
    poly.push(0.0, 1.0);
    EXPECT_DOUBLE_EQ(poly.area(), 1.0);
}

TEST(PlanarPolygon, AreaTriangle) {
    // Right triangle: (0,0), (2,0), (0,2) → area = 2.0
    axis::detail::PlanarPolygon<32> poly;
    poly.push(0.0, 0.0);
    poly.push(2.0, 0.0);
    poly.push(0.0, 2.0);
    EXPECT_DOUBLE_EQ(poly.area(), 2.0);
}

TEST(PlanarPolygon, AreaClockwiseEqualsCounterClockwise) {
    // Shoelace area is unsigned, so winding order doesn't matter.
    axis::detail::PlanarPolygon<32> ccw;
    ccw.push(0.0, 0.0);
    ccw.push(1.0, 0.0);
    ccw.push(1.0, 1.0);
    ccw.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> cw;
    cw.push(0.0, 0.0);
    cw.push(0.0, 1.0);
    cw.push(1.0, 1.0);
    cw.push(1.0, 0.0);

    EXPECT_DOUBLE_EQ(ccw.area(), cw.area());
}

TEST(PlanarPolygon, AreaLargerRectangle) {
    // Rectangle 3×4 → area = 12.0
    axis::detail::PlanarPolygon<32> poly;
    poly.push(0.0, 0.0);
    poly.push(3.0, 0.0);
    poly.push(3.0, 4.0);
    poly.push(0.0, 4.0);
    EXPECT_DOUBLE_EQ(poly.area(), 12.0);
}

// Regression: small polygons located far from the origin must not lose
// precision. The shoelace formula translates to the first vertex before
// summing; without that, catastrophic cancellation of the large coordinate
// products would corrupt the (small) area. These cases would fail with the
// naive absolute-coordinate formula but hold to near machine precision now.

TEST(PlanarPolygon, AreaUnitSquareFarFromOrigin) {
    // Unit square offset to (1e6, -1e6). Exact area is still 1.0, but the
    // cross-product terms are ~1e12, so the naive formula loses ~4 digits.
    constexpr double ox = 1.0e6;
    constexpr double oy = -1.0e6;
    axis::detail::PlanarPolygon<32> poly;
    poly.push(ox + 0.0, oy + 0.0);
    poly.push(ox + 1.0, oy + 0.0);
    poly.push(ox + 1.0, oy + 1.0);
    poly.push(ox + 0.0, oy + 1.0);
    EXPECT_NEAR(poly.area(), 1.0, 1.0e-13);
}

TEST(PlanarPolygon, AreaTinySquareFarFromOrigin) {
    // Side 1e-3 square (area 1e-6) centered near (-3, -3), mirroring the
    // property-test failure geometry. Relative error must stay tiny.
    // Theoretical relative error floor is ~1e-12. While this test is
    // deterministic, we leave 10x headroom for cross-toolchain FP variation
    // (e.g. FMA contraction).
    constexpr double ox = -3.0;
    constexpr double oy = -3.0;
    constexpr double s = 1.0e-3;
    axis::detail::PlanarPolygon<32> poly;
    poly.push(ox + 0.0, oy + 0.0);
    poly.push(ox + s, oy + 0.0);
    poly.push(ox + s, oy + s);
    poly.push(ox + 0.0, oy + s);
    const double expected = s * s;  // 1e-6
    EXPECT_NEAR(poly.area(), expected, expected * 1.0e-11);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — non-overlapping cases
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, NonOverlappingSeparatedSquares) {
    // Two unit squares that don't overlap
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(5.0, 5.0);
    b.push(6.0, 5.0);
    b.push(6.0, 6.0);
    b.push(5.0, 6.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_DOUBLE_EQ(area, 0.0);
}

TEST(PlanarClipper, NonOverlappingAdjacentX) {
    // Two squares side by side along X axis (touching but no interior overlap)
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(1.0, 0.0);
    b.push(2.0, 0.0);
    b.push(2.0, 1.0);
    b.push(1.0, 1.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    // Edge-touching polygons should produce zero or near-zero area
    EXPECT_NEAR(area, 0.0, 1.0e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — full overlap cases
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, FullOverlapIdenticalSquares) {
    // Two identical unit squares → overlap area = 1.0
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(0.0, 0.0);
    b.push(1.0, 0.0);
    b.push(1.0, 1.0);
    b.push(0.0, 1.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 1.0, 1.0e-14);
}

TEST(PlanarClipper, FullOverlapSubjectInsideClip) {
    // Small square fully inside a larger square
    axis::detail::PlanarPolygon<32> small;
    small.push(0.25, 0.25);
    small.push(0.75, 0.25);
    small.push(0.75, 0.75);
    small.push(0.25, 0.75);

    axis::detail::PlanarPolygon<32> large;
    large.push(0.0, 0.0);
    large.push(1.0, 0.0);
    large.push(1.0, 1.0);
    large.push(0.0, 1.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(small, large);
    EXPECT_NEAR(area, 0.25, 1.0e-14);  // 0.5×0.5 = 0.25
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — partial overlap cases
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, PartialOverlapHalfSquare) {
    // Two unit squares offset by 0.5 in X → overlap is 0.5 × 1.0 = 0.5
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(0.5, 0.0);
    b.push(1.5, 0.0);
    b.push(1.5, 1.0);
    b.push(0.5, 1.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 0.5, 1.0e-14);
}

TEST(PlanarClipper, PartialOverlapQuarterSquare) {
    // Two unit squares offset by 0.5 in both X and Y → overlap is 0.5×0.5 = 0.25
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(0.5, 0.5);
    b.push(1.5, 0.5);
    b.push(1.5, 1.5);
    b.push(0.5, 1.5);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 0.25, 1.0e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — rectangle-rectangle overlap
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, RectangleRectangleOverlap) {
    // Rectangle [0,3]×[0,2] and [1,4]×[1,3]
    // Overlap: [1,3]×[1,2] = 2×1 = 2.0
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(3.0, 0.0);
    a.push(3.0, 2.0);
    a.push(0.0, 2.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(1.0, 1.0);
    b.push(4.0, 1.0);
    b.push(4.0, 3.0);
    b.push(1.0, 3.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 2.0, 1.0e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — edge-touching cases
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, EdgeTouchingSharedEdge) {
    // Two squares sharing a full edge should produce ~0 overlap area
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(1.0, 0.0);
    b.push(2.0, 0.0);
    b.push(2.0, 1.0);
    b.push(1.0, 1.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 0.0, 1.0e-12);
}

TEST(PlanarClipper, CornerTouching) {
    // Two squares sharing only a corner point
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(1.0, 0.0);
    a.push(1.0, 1.0);
    a.push(0.0, 1.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(1.0, 1.0);
    b.push(2.0, 1.0);
    b.push(2.0, 2.0);
    b.push(1.0, 2.0);

    double area = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    EXPECT_NEAR(area, 0.0, 1.0e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — triangle overlap
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, TriangleInsideSquare) {
    // Triangle fully inside a unit square
    axis::detail::PlanarPolygon<32> tri;
    tri.push(0.25, 0.25);
    tri.push(0.75, 0.25);
    tri.push(0.5, 0.75);

    axis::detail::PlanarPolygon<32> sq;
    sq.push(0.0, 0.0);
    sq.push(1.0, 0.0);
    sq.push(1.0, 1.0);
    sq.push(0.0, 1.0);

    // Triangle area = 0.5 * base * height = 0.5 * 0.5 * 0.5 = 0.125
    double expected_area = 0.5 * 0.5 * 0.5;
    double area = axis::detail::PlanarClipper::overlap_area<32>(tri, sq);
    EXPECT_NEAR(area, expected_area, 1.0e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — empty polygon cases
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, EmptySubjectReturnsZero) {
    axis::detail::PlanarPolygon<32> empty;
    axis::detail::PlanarPolygon<32> sq;
    sq.push(0.0, 0.0);
    sq.push(1.0, 0.0);
    sq.push(1.0, 1.0);
    sq.push(0.0, 1.0);

    EXPECT_DOUBLE_EQ(axis::detail::PlanarClipper::overlap_area<32>(empty, sq), 0.0);
}

TEST(PlanarClipper, EmptyClipReturnsZero) {
    axis::detail::PlanarPolygon<32> sq;
    sq.push(0.0, 0.0);
    sq.push(1.0, 0.0);
    sq.push(1.0, 1.0);
    sq.push(0.0, 1.0);
    axis::detail::PlanarPolygon<32> empty;

    EXPECT_DOUBLE_EQ(axis::detail::PlanarClipper::overlap_area<32>(sq, empty), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — commutativity
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, OverlapAreaCommutative) {
    // overlap_area(A, B) should equal overlap_area(B, A)
    axis::detail::PlanarPolygon<32> a;
    a.push(0.0, 0.0);
    a.push(2.0, 0.0);
    a.push(2.0, 2.0);
    a.push(0.0, 2.0);

    axis::detail::PlanarPolygon<32> b;
    b.push(1.0, 1.0);
    b.push(3.0, 1.0);
    b.push(3.0, 3.0);
    b.push(1.0, 3.0);

    double ab = axis::detail::PlanarClipper::overlap_area<32>(a, b);
    double ba = axis::detail::PlanarClipper::overlap_area<32>(b, a);
    EXPECT_NEAR(ab, ba, 1.0e-14);
    EXPECT_NEAR(ab, 1.0, 1.0e-14);  // overlap is [1,2]×[1,2] = 1.0
}

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper::overlap_area — pentagon clipping
// ─────────────────────────────────────────────────────────────────────────────

TEST(PlanarClipper, PentagonClippedBySquare) {
    // Regular pentagon centered at (0.5, 0.5), radius 0.4
    // Clipped by unit square [0,1]×[0,1] — pentagon fits inside, so
    // overlap area = pentagon area
    axis::detail::PlanarPolygon<32> pent;
    constexpr double pi = 3.14159265358979323846;
    constexpr double r = 0.4;
    for (int i = 0; i < 5; ++i) {
        double angle = 2.0 * pi * i / 5.0 - pi / 2.0;
        pent.push(0.5 + r * std::cos(angle), 0.5 + r * std::sin(angle));
    }

    axis::detail::PlanarPolygon<32> sq;
    sq.push(0.0, 0.0);
    sq.push(1.0, 0.0);
    sq.push(1.0, 1.0);
    sq.push(0.0, 1.0);

    double overlap = axis::detail::PlanarClipper::overlap_area<32>(pent, sq);
    double pent_area = pent.area();
    // Pentagon is fully inside the square, so overlap = pentagon area
    EXPECT_NEAR(overlap, pent_area, 1.0e-14);
}

}  // namespace axis::test
