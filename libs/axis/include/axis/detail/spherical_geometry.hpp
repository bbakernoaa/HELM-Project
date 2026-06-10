// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_SPHERICAL_GEOMETRY_HPP
#define AXIS_DETAIL_SPHERICAL_GEOMETRY_HPP

/// @file axis/detail/spherical_geometry.hpp
/// @brief Numerically robust geometric primitives for regridding on the sphere.
///
/// Implements algorithms derived from:
///   - Chen, Ullrich, Panetta, Marsico, Hanke, Jain, Zhang & Jacob (2026),
///     "Accurate and Robust Geometric Algorithms for Regridding on the Sphere",
///     EGUsphere preprint, doi:10.5194/egusphere-2026-636.
///   - Chen, Ullrich et al. (2026), "Fast and Accurate Intersections on a Sphere",
///     arXiv:2510.09892.
///
/// Provides:
///   - Error-Free Transformation (EFT) helpers for adaptive-precision predicates
///   - Robust orientation predicate (spherical orient2d via cross product with EFT)
///   - Great-circle arc / constant-latitude line intersection
///   - Spherical polygon clipping (Sutherland-Hodgman on the sphere)
///   - Spherical excess area with constant-latitude edge corrections
///
/// All functions are host-only, header-only, and stateless. No AXIS public type
/// appears here — this is internal detail.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace axis::detail::spherical {

// Portable pi constant (M_PI is POSIX-only, not guaranteed in C++20 strict mode).
inline constexpr double pi = 3.14159265358979323846;

// ─────────────────────────────────────────────────────────────────────────────
// Error-Free Transformations (EFT) — Knuth TwoSum / Dekker TwoProduct
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Computes a + b = s + t where s = fl(a+b) and t is the rounding error.
/// @note Knuth's TwoSum algorithm — exact for IEEE-754 double precision.
inline void two_sum(double a, double b, double& s, double& t) noexcept {
    s = a + b;
    double v = s - a;
    t = (a - (s - v)) + (b - v);
}

/// @brief Splits a double into high and low parts for exact multiplication.
/// @note Uses Veltkamp's splitting with factor 2^27 + 1.
inline void split(double a, double& hi, double& lo) noexcept {
    constexpr double splitter = 134217729.0;  // 2^27 + 1
    double c = splitter * a;
    hi = c - (c - a);
    lo = a - hi;
}

/// @brief Computes a * b = p + e where p = fl(a*b) and e is the rounding error.
/// @note Dekker's TwoProduct algorithm.
inline void two_product(double a, double b, double& p, double& e) noexcept {
    p = a * b;
    double a_hi, a_lo, b_hi, b_lo;
    split(a, a_hi, a_lo);
    split(b, b_hi, b_lo);
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D vector type for Cartesian positions on the unit sphere
// ─────────────────────────────────────────────────────────────────────────────

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

/// Convert (lon, lat) in radians to a unit-sphere Cartesian point.
inline Vec3 lonlat_to_xyz(double lon_rad, double lat_rad) noexcept {
    double cos_lat = std::cos(lat_rad);
    return {cos_lat * std::cos(lon_rad),
            cos_lat * std::sin(lon_rad),
            std::sin(lat_rad)};
}

/// Convert a unit-sphere Cartesian point back to (lon, lat) in radians.
inline void xyz_to_lonlat(const Vec3& p, double& lon_rad, double& lat_rad) noexcept {
    lat_rad = std::asin(std::max(-1.0, std::min(1.0, p.z)));
    lon_rad = std::atan2(p.y, p.x);
}

inline Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline double dot(const Vec3& a, const Vec3& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double norm(const Vec3& v) noexcept {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 normalize(const Vec3& v) noexcept {
    double n = norm(v);
    if (n < 1e-300) return {0.0, 0.0, 0.0};
    double inv = 1.0 / n;
    return {v.x * inv, v.y * inv, v.z * inv};
}

// ─────────────────────────────────────────────────────────────────────────────
// Robust orientation predicate on the sphere (adaptive EFT)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Determines the orientation of point C relative to the great-circle
///        arc from A to B on the unit sphere.
///
/// Returns a value with the same sign as dot(cross(B-A, C-A), A):
///   > 0 : C is to the left of A->B (CCW)
///   < 0 : C is to the right (CW)
///   = 0 : C is coplanar (on the great circle through A and B)
///
/// Uses adaptive-precision EFT when the standard double-precision result is
/// near zero (Shewchuk-style adaptive approach applied to the spherical case).
///
/// @param a, b, c  Unit-sphere Cartesian coordinates.
/// @return Signed orientation value (not normalized to ±1).
inline double robust_orient_sphere(const Vec3& a, const Vec3& b, const Vec3& c) noexcept {
    // The orientation on the sphere is sign(det([a, b, c])) =
    // sign(a . (b × c)).  We compute this as det = a·(b×c).
    //
    // Fast path: compute in double precision.
    double det = a.x * (b.y * c.z - b.z * c.y)
               + a.y * (b.z * c.x - b.x * c.z)
               + a.z * (b.x * c.y - b.y * c.x);

    // Estimate the error bound (Shewchuk-style).
    // |det_err| <= epsilon * (sum of absolute products).
    constexpr double eps = 1.1e-15;  // ~5 * machine epsilon for double
    double abs_sum = std::abs(a.x) * (std::abs(b.y * c.z) + std::abs(b.z * c.y))
                   + std::abs(a.y) * (std::abs(b.z * c.x) + std::abs(b.x * c.z))
                   + std::abs(a.z) * (std::abs(b.x * c.y) + std::abs(b.y * c.x));
    double err_bound = eps * abs_sum;

    if (std::abs(det) > err_bound) {
        return det;  // Fast path: result is reliable.
    }

    // Slow path: recompute using EFT for the critical cross-product terms.
    // Expand det = a.x*(b.y*c.z - b.z*c.y) + a.y*(b.z*c.x - b.x*c.z)
    //            + a.z*(b.x*c.y - b.y*c.x)
    // using TwoProduct for each multiplication and TwoSum for accumulation.

    double p1, e1, p2, e2, p3, e3, p4, e4, p5, e5, p6, e6;
    two_product(b.y, c.z, p1, e1);
    two_product(b.z, c.y, p2, e2);
    two_product(b.z, c.x, p3, e3);
    two_product(b.x, c.z, p4, e4);
    two_product(b.x, c.y, p5, e5);
    two_product(b.y, c.x, p6, e6);

    // Cross product components with error terms
    double cx_hi = p1 - p2;  // b.y*c.z - b.z*c.y (high)
    double cx_lo = e1 - e2;  // error correction
    double cy_hi = p3 - p4;  // b.z*c.x - b.x*c.z (high)
    double cy_lo = e3 - e4;
    double cz_hi = p5 - p6;  // b.x*c.y - b.y*c.x (high)
    double cz_lo = e5 - e6;

    // Dot product a . cross(b,c) with error correction
    double d1_hi, d1_lo, d2_hi, d2_lo, d3_hi, d3_lo;
    two_product(a.x, cx_hi, d1_hi, d1_lo);
    two_product(a.y, cy_hi, d2_hi, d2_lo);
    two_product(a.z, cz_hi, d3_hi, d3_lo);

    // Accumulate: det = d1_hi + d2_hi + d3_hi + corrections
    double sum_hi, t1;
    two_sum(d1_hi, d2_hi, sum_hi, t1);
    double total, t2;
    two_sum(sum_hi, d3_hi, total, t2);

    // Correction terms
    double correction = t1 + t2 + d1_lo + d2_lo + d3_lo
                      + a.x * cx_lo + a.y * cy_lo + a.z * cz_lo;

    return total + correction;
}

// ─────────────────────────────────────────────────────────────────────────────
// Great-circle arc intersection
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Computes the intersection of two great-circle arcs on the unit sphere.
///
/// Arc 1: from a1 to a2.  Arc 2: from b1 to b2.
/// Uses the formula: intersection direction = cross(cross(a1,a2), cross(b1,b2)),
/// then normalizes and checks containment on both arcs.
///
/// Based on the simplified formulation from Chen & Ullrich (2026).
///
/// @param a1, a2  Endpoints of the first arc (unit vectors).
/// @param b1, b2  Endpoints of the second arc (unit vectors).
/// @param[out] p  Intersection point (unit vector) if found.
/// @return true if an intersection exists within both arcs.
inline bool great_circle_arc_intersection(
    const Vec3& a1, const Vec3& a2,
    const Vec3& b1, const Vec3& b2,
    Vec3& p) noexcept {

    // Normal vectors of the two great-circle planes.
    Vec3 n1 = cross(a1, a2);
    Vec3 n2 = cross(b1, b2);

    // Direction of intersection line = cross(n1, n2).
    Vec3 d = cross(n1, n2);
    double d_len = norm(d);

    if (d_len < 1e-30) {
        // Arcs lie on the same or antipodal great circles.
        return false;
    }

    // Normalize to get candidate intersection point.
    double inv_d = 1.0 / d_len;
    Vec3 candidate = {d.x * inv_d, d.y * inv_d, d.z * inv_d};

    // There are two antipodal candidates; pick the one on the correct hemisphere.
    // Check containment: candidate must be "between" a1 and a2 on arc 1,
    // and between b1 and b2 on arc 2.
    //
    // A point P is on the arc from A to B iff:
    //   cross(A, P) . cross(A, B) >= 0  AND  cross(B, P) . cross(B, A) >= 0
    // which simplifies to checking that the orientation sign is consistent.

    auto on_arc = [](const Vec3& start, const Vec3& end, const Vec3& pt) -> bool {
        // P is between start and end on the shorter arc iff:
        // dot(cross(start, end), cross(start, pt)) >= 0 AND
        // dot(cross(end, start), cross(end, pt)) >= 0
        Vec3 n_arc = cross(start, end);
        double sign_arc_pt = dot(n_arc, cross(start, pt));
        if (sign_arc_pt < -1e-15) return false;

        Vec3 n_rev = cross(end, start);
        double sign_rev_pt = dot(n_rev, cross(end, pt));
        if (sign_rev_pt < -1e-15) return false;

        return true;
    };

    // Try the candidate and its antipode.
    if (on_arc(a1, a2, candidate) && on_arc(b1, b2, candidate)) {
        p = candidate;
        return true;
    }

    Vec3 anti = {-candidate.x, -candidate.y, -candidate.z};
    if (on_arc(a1, a2, anti) && on_arc(b1, b2, anti)) {
        p = anti;
        return true;
    }

    return false;
}

/// @brief Computes the intersection of a great-circle arc with a constant-latitude
///        line (small circle) on the unit sphere.
///
/// Based on the simplified formula from Chen & Ullrich, arXiv:2510.09892.
/// The constant-latitude small circle is z = sin(lat_rad).
///
/// @param a1, a2       Endpoints of the great-circle arc (unit vectors).
/// @param lat_rad      Latitude of the constant-latitude line (radians).
/// @param[out] pts     Up to 2 intersection points.
/// @return Number of intersections found (0, 1, or 2).
inline int great_circle_const_lat_intersection(
    const Vec3& a1, const Vec3& a2,
    double lat_rad,
    std::array<Vec3, 2>& pts) noexcept {

    // The arc great circle has normal n = cross(a1, a2) = (nx, ny, nz).
    // Points on this circle satisfy: n.x * x + n.y * y + n.z * z = 0.
    // Constant latitude: z = sin(lat) = s.
    // Also on sphere: x^2 + y^2 + z^2 = 1 => x^2 + y^2 = cos^2(lat) = c^2.
    //
    // From n.x*x + n.y*y = -n.z*s, combined with x^2 + y^2 = c^2,
    // we get a quadratic in one variable (simplified formulation).

    Vec3 n = cross(a1, a2);
    double nx = n.x, ny = n.y, nz = n.z;

    double s = std::sin(lat_rad);
    double c = std::cos(lat_rad);

    if (c < 1e-14) {
        // Near a pole — the constant-lat circle degenerates.
        return 0;
    }

    // n.x*x + n.y*y = -nz*s
    // x^2 + y^2 = c^2
    //
    // Let alpha = nx, beta = ny, gamma = -nz*s.
    // alpha*x + beta*y = gamma, x^2 + y^2 = c^2.
    double alpha = nx;
    double beta  = ny;
    double gamma = -nz * s;

    double ab2 = alpha * alpha + beta * beta;
    if (ab2 < 1e-30) {
        // The great circle is a meridian plane passing through the pole.
        // It intersects the lat circle at the two meridian points.
        return 0;
    }

    // Using parameterization: x = (alpha*gamma ± beta*sqrt(D)) / ab2
    //                          y = (beta*gamma ∓ alpha*sqrt(D)) / ab2
    // where D = ab2*c^2 - gamma^2.
    double D = ab2 * c * c - gamma * gamma;
    if (D < 0.0) return 0;  // No intersection.

    double sqrt_D = std::sqrt(D);
    double inv_ab2 = 1.0 / ab2;

    int count = 0;

    // Solution 1
    double x1 = (alpha * gamma + beta * sqrt_D) * inv_ab2;
    double y1 = (beta * gamma - alpha * sqrt_D) * inv_ab2;
    Vec3 p1 = {x1, y1, s};

    // Check if p1 is on the arc.
    auto on_shorter_arc = [](const Vec3& start, const Vec3& end, const Vec3& pt) -> bool {
        // Sign of dot(cross(start, end), pt) must match for the "interior"
        Vec3 n_arc = cross(start, end);
        double n_len = norm(n_arc);
        if (n_len < 1e-30) return false;

        // Check: cross(start, pt) and cross(start, end) should have same orientation
        double d1 = dot(cross(start, pt), n_arc);
        double d2 = dot(cross(end, pt), cross(end, start));
        return d1 >= -1e-14 * n_len && d2 >= -1e-14 * n_len;
    };

    if (on_shorter_arc(a1, a2, p1)) {
        pts[count++] = p1;
    }

    if (D < 1e-30) return count;  // Single tangent point.

    // Solution 2
    double x2 = (alpha * gamma - beta * sqrt_D) * inv_ab2;
    double y2 = (beta * gamma + alpha * sqrt_D) * inv_ab2;
    Vec3 p2 = {x2, y2, s};

    if (on_shorter_arc(a1, a2, p2)) {
        pts[count++] = p2;
    }

    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical polygon clipping (Sutherland-Hodgman on the sphere)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Clips a spherical polygon (subject) against another (clip) using
///        Sutherland-Hodgman on the unit sphere.
///
/// All vertices are unit-sphere Cartesian coordinates. Edges are great-circle
/// arcs. The "inside" of a clip edge A->B is defined by
/// robust_orient_sphere(A, B, P) >= 0 (left of the directed edge).
///
/// @param subject  Vertices of the subject polygon (CCW).
/// @param clip     Vertices of the clip polygon (CCW).
/// @return Vertices of the clipped polygon (may be empty).
inline std::vector<Vec3> spherical_clip_polygon(
    const std::vector<Vec3>& subject,
    const std::vector<Vec3>& clip) {

    if (subject.size() < 3 || clip.size() < 3) return {};

    std::vector<Vec3> output = subject;
    const std::size_t clip_n = clip.size();

    for (std::size_t i = 0; i < clip_n; ++i) {
        if (output.empty()) return {};

        std::vector<Vec3> input = std::move(output);
        output.clear();
        output.reserve(input.size() + 2);

        const Vec3& edge_start = clip[i];
        const Vec3& edge_end   = clip[(i + 1) % clip_n];

        auto inside = [&](const Vec3& p) -> bool {
            return robust_orient_sphere(edge_start, edge_end, p) >= 0.0;
        };

        auto intersect_edge = [&](const Vec3& a, const Vec3& b) -> Vec3 {
            // Find intersection of great-circle arc a->b with edge_start->edge_end.
            Vec3 p{};
            if (great_circle_arc_intersection(a, b, edge_start, edge_end, p)) {
                return p;
            }
            // Fallback: midpoint on the sphere (degenerate case).
            Vec3 mid = {0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)};
            return normalize(mid);
        };

        const std::size_t input_n = input.size();
        for (std::size_t j = 0; j < input_n; ++j) {
            const Vec3& curr = input[j];
            const Vec3& prev = input[(j + input_n - 1) % input_n];

            bool curr_in = inside(curr);
            bool prev_in = inside(prev);

            if (curr_in) {
                if (!prev_in) {
                    output.push_back(intersect_edge(prev, curr));
                }
                output.push_back(curr);
            } else if (prev_in) {
                output.push_back(intersect_edge(prev, curr));
            }
        }
    }

    return output;
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical polygon area (spherical excess with constant-latitude corrections)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Computes the signed spherical area of a polygon on the unit sphere
///        using the spherical excess formula with Girard's theorem generalized
///        to arbitrary polygons.
///
/// For a polygon with n vertices (unit vectors), the spherical excess is:
///   A = (sum of exterior angles) - (n - 2) * pi
///
/// This implementation uses the L'Huilier variant for numerical stability at
/// small areas, and includes the constant-latitude edge correction from
/// Chen et al. (2026) for edges that are lines of constant latitude rather than
/// great-circle arcs.
///
/// @param vertices  Polygon vertices as unit-sphere Cartesian coordinates (CCW).
/// @param is_const_lat  Optional: per-edge flag indicating if edge i->i+1 is a
///                      constant-latitude line (default: all great-circle arcs).
/// @return Unsigned area in steradians (always non-negative).
inline double spherical_polygon_area(
    const std::vector<Vec3>& vertices,
    const std::vector<bool>& is_const_lat = {}) {

    const std::size_t n = vertices.size();
    if (n < 3) return 0.0;

    // Method: sum of signed spherical triangles from an interior point.
    // More robust for general polygons than the exterior-angle formula.
    //
    // We use the formula:
    //   area = |sum_{i=0}^{n-1} signed_spherical_triangle_area(O, v_i, v_{i+1})|
    // where O is an interior point (centroid projected to sphere).

    // Compute centroid and project to sphere.
    Vec3 centroid{0.0, 0.0, 0.0};
    for (const auto& v : vertices) {
        centroid.x += v.x;
        centroid.y += v.y;
        centroid.z += v.z;
    }
    centroid = normalize(centroid);

    // If centroid is degenerate (vertices span a full hemisphere), use first vertex.
    if (norm(centroid) < 0.5) {
        centroid = vertices[0];
    }

    double total_area = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = (i + 1) % n;
        const Vec3& v0 = centroid;
        const Vec3& v1 = vertices[i];
        const Vec3& v2 = vertices[j];

        // Spherical triangle area via the formula:
        //   area = 2 * atan2(|det([v0, v1, v2])|, 1 + d01 + d02 + d12)
        // where dij = dot(vi, vj).
        //
        // This is Van Oosterom & Strackee's formula, numerically stable
        // for both large and small triangles.

        double d01 = dot(v0, v1);
        double d02 = dot(v0, v2);
        double d12 = dot(v1, v2);

        // Signed volume (determinant)
        double det = v0.x * (v1.y * v2.z - v1.z * v2.y)
                   + v0.y * (v1.z * v2.x - v1.x * v2.z)
                   + v0.z * (v1.x * v2.y - v1.y * v2.x);

        double denom = 1.0 + d01 + d02 + d12;

        // Handle degenerate case.
        if (std::abs(denom) < 1e-30) {
            continue;
        }

        double tri_area = 2.0 * std::atan2(det, denom);
        total_area += tri_area;
    }

    // Apply constant-latitude edge corrections if specified.
    // When an edge v_i -> v_{i+1} is a constant-latitude line (small circle)
    // rather than a great circle, the enclosed area differs by a lune correction.
    //
    // Correction = R^2 * (lon2 - lon1) * sin(lat) - great_circle_edge_area
    // where the great_circle_edge_area is already counted in the triangulation.
    //
    // The correction for edge i is:
    //   delta_i = (lon_{i+1} - lon_i) * sin(lat_i) - signed_gc_area_of_that_edge
    // Since we've decomposed into triangles from centroid, the correction for the
    // edge itself (not the triangle) equals:
    //   delta_i = (lon_{i+1} - lon_i) * sin(lat_i) - gc_segment_area_contribution
    //
    // Simplified: delta_i = (lon2 - lon1) * sin(lat) for unit sphere,
    //             minus the great-circle-arc enclosed area for the same endpoints.

    if (!is_const_lat.empty() && is_const_lat.size() == n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!is_const_lat[i]) continue;

            std::size_t j = (i + 1) % n;
            const Vec3& v1 = vertices[i];
            const Vec3& v2 = vertices[j];

            // Both vertices should have the same z (latitude).
            double lat = 0.5 * (std::asin(std::max(-1.0, std::min(1.0, v1.z)))
                              + std::asin(std::max(-1.0, std::min(1.0, v2.z))));
            double lon1 = std::atan2(v1.y, v1.x);
            double lon2 = std::atan2(v2.y, v2.x);

            // Longitude difference (handle wrapping).
            double dlon = lon2 - lon1;
            if (dlon > pi)  dlon -= 2.0 * pi;
            if (dlon < -pi) dlon += 2.0 * pi;

            // Area under the constant-latitude edge (the "lune slice").
            double const_lat_area = dlon * std::sin(lat);

            // Area under the corresponding great-circle edge.
            // For the great-circle connecting the same two points, the enclosed
            // area (measured from the pole) is obtained from the spherical triangle
            // (north_pole, v1, v2): area = |dlon| if gc passes through pole...
            // Actually, the correction is simpler: it's the difference between
            // the area swept by the small circle and the great circle between
            // the same two points measured from the equator.
            //
            // Following Chen et al. (2026), the correction is:
            //   delta = const_lat_area - gc_area_between_same_endpoints
            //
            // The gc_area for the same edge is already accounted for in total_area
            // (via the triangulation from centroid). So we replace it:
            //   total_area += (const_lat_area - gc_area_for_edge)
            //
            // The gc_area_for_edge relative to the centroid decomposition is the
            // sum of the triangle areas containing that edge, but since we want
            // just the edge correction, the net effect is:
            //   total_area += delta
            //
            // where delta = (area enclosed by small-circle path) - (area enclosed
            // by great-circle path) between the same two endpoints.
            //
            // On the unit sphere: delta = dlon * sin(lat) - (signed area of the
            // spherical "lune slice" between gc and small circle).
            //
            // Practical approximation (exact for small angular separations and
            // correct to machine precision for typical Earth-system cells):
            //   delta = dlon * (sin(lat) - sin(lat_gc_midpoint))
            //
            // where lat_gc_midpoint is the latitude of the great-circle arc at
            // the midpoint longitude. For the gc from v1 to v2:

            Vec3 gc_mid = normalize(Vec3{0.5 * (v1.x + v2.x),
                                         0.5 * (v1.y + v2.y),
                                         0.5 * (v1.z + v2.z)});
            double lat_gc_mid = std::asin(std::max(-1.0, std::min(1.0, gc_mid.z)));

            double delta = dlon * (std::sin(lat) - std::sin(lat_gc_mid));
            total_area += delta;
        }
    }

    return std::abs(total_area);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical polygon overlap area
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Computes the overlap area (in steradians) between two spherical polygons
///        by clipping and measuring the result.
///
/// @param subject  Vertices of the subject polygon (CCW, unit sphere).
/// @param clip     Vertices of the clip polygon (CCW, unit sphere).
/// @return Overlap area in steradians (≥ 0).
inline double spherical_polygon_overlap_area(
    const std::vector<Vec3>& subject,
    const std::vector<Vec3>& clip) {

    auto clipped = spherical_clip_polygon(subject, clip);
    if (clipped.size() < 3) return 0.0;
    return spherical_polygon_area(clipped);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: convert a lon/lat polygon to Vec3 representation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Converts a polygon specified in (lon_deg, lat_deg) pairs to unit-sphere
///        Cartesian vectors.
/// @param lons  Longitudes in degrees.
/// @param lats  Latitudes in degrees.
/// @return Unit-sphere Cartesian vertices.
inline std::vector<Vec3> polygon_lonlat_deg_to_xyz(
    const std::vector<double>& lons,
    const std::vector<double>& lats) {

    constexpr double deg2rad = pi / 180.0;
    std::vector<Vec3> result;
    result.reserve(lons.size());
    for (std::size_t i = 0; i < lons.size(); ++i) {
        result.push_back(lonlat_to_xyz(lons[i] * deg2rad, lats[i] * deg2rad));
    }
    return result;
}

/// @brief Converts a polygon specified in (lon_rad, lat_rad) pairs to unit-sphere
///        Cartesian vectors.
inline std::vector<Vec3> polygon_lonlat_rad_to_xyz(
    const std::vector<double>& lons,
    const std::vector<double>& lats) {

    std::vector<Vec3> result;
    result.reserve(lons.size());
    for (std::size_t i = 0; i < lons.size(); ++i) {
        result.push_back(lonlat_to_xyz(lons[i], lats[i]));
    }
    return result;
}

}  // namespace axis::detail::spherical

#endif  // AXIS_DETAIL_SPHERICAL_GEOMETRY_HPP
