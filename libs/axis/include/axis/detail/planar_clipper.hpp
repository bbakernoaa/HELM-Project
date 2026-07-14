// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_PLANAR_CLIPPER_HPP
#define AXIS_DETAIL_PLANAR_CLIPPER_HPP

/// @file axis/detail/planar_clipper.hpp
/// @brief Device-portable Sutherland-Hodgman polygon clipping in 2D.
///
/// Provides:
///   - PlanarPolygon<MaxVerts>: fixed-capacity 2D polygon with stack arrays
///   - PlanarClipper: Sutherland-Hodgman clipping returning overlap area
///
/// All functions are annotated KOKKOS_FUNCTION for device portability.
/// No heap allocation — safe for use in Kokkos parallel kernels on GPU
/// (HELM Law #2).

#include <Kokkos_Core.hpp>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// PlanarPolygon — fixed-capacity 2D polygon for planar clipping
// ─────────────────────────────────────────────────────────────────────────────

/// @brief A fixed-capacity 2D polygon for device-portable Sutherland-Hodgman
///        clipping.
///
/// Vertices are stored as (x, y) coordinate pairs in stack-allocated arrays.
/// No heap allocation, making this safe for use in Kokkos device kernels.
///
/// @tparam MaxVerts Maximum number of vertices this polygon can hold (default 32).
///                  Must be at least 8 to ensure sufficient capacity for clipping.
template <int MaxVerts = 32>
struct PlanarPolygon {
    static_assert(MaxVerts >= 8, "PlanarPolygon requires MaxVerts >= 8 for sufficient clipping capacity");

    double x[MaxVerts];  ///< X-coordinates of vertices
    double y[MaxVerts];  ///< Y-coordinates of vertices
    int n{0};            ///< Current vertex count

    /// @brief Add a vertex to the polygon (clamps at MaxVerts).
    /// @param px X-coordinate of the new vertex.
    /// @param py Y-coordinate of the new vertex.
    KOKKOS_FUNCTION void push(double px, double py) noexcept {
        if (n < MaxVerts) {
            x[n] = px;
            y[n] = py;
            ++n;
        }
    }

    /// @brief Clear all vertices.
    KOKKOS_FUNCTION void clear() noexcept {
        n = 0;
    }

    /// @brief Check if the polygon is empty (fewer than 3 vertices).
    KOKKOS_FUNCTION bool empty() const noexcept {
        return n < 3;
    }

    /// @brief Compute the polygon area using the shoelace formula (unsigned).
    ///
    /// Returns the absolute area of the polygon regardless of vertex winding
    /// order. Returns 0 if the polygon has fewer than 3 vertices.
    ///
    /// @return Non-negative area of the polygon.
    KOKKOS_FUNCTION double area() const noexcept {
        if (n < 3) return 0.0;
        // Translate to the first vertex before applying the shoelace formula.
        // The sum is mathematically translation-invariant, but subtracting a
        // reference point keeps the cross-product terms small and avoids
        // catastrophic cancellation for polygons located far from the origin.
        const double x0 = x[0];
        const double y0 = y[0];
        double a = 0.0;
        for (int i = 1; i < n - 1; ++i) {
            const double dx0 = x[i] - x0;
            const double dy0 = y[i] - y0;
            const double dx1 = x[i + 1] - x0;
            const double dy1 = y[i + 1] - y0;
            a += dx0 * dy1 - dx1 * dy0;
        }
        return Kokkos::fabs(a) * 0.5;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// PlanarClipper — Sutherland-Hodgman polygon clipping in 2D
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Device-portable Sutherland-Hodgman polygon clipping in 2D.
///
/// Clips a subject polygon against each edge of a clip polygon, producing
/// the intersection polygon. The overlap area is computed via the shoelace
/// formula on the result.
///
/// The algorithm handles both convex and concave polygons. Degenerate cases
/// (empty polygons, no overlap) produce an area of 0.
///
/// All operations use fixed-capacity stack buffers — no heap allocation.
struct PlanarClipper {
    /// @brief Compute the overlap area between two 2D polygons.
    ///
    /// Performs full Sutherland-Hodgman clipping of the subject polygon against
    /// the clip polygon, then returns the shoelace area of the intersection.
    ///
    /// @tparam M Maximum vertex capacity of the polygons.
    /// @param subject The polygon to be clipped.
    /// @param clip The polygon defining the clipping boundary.
    /// @return Overlap area (>= 0). Returns 0 if polygons don't intersect.
    template <int M>
    KOKKOS_FUNCTION static double overlap_area(const PlanarPolygon<M> &subject, const PlanarPolygon<M> &clip) noexcept {
        if (subject.empty() || clip.empty()) {
            return 0.0;
        }

        // Use two buffers to alternate input/output during iterative clipping.
        PlanarPolygon<M> buf_a;
        PlanarPolygon<M> buf_b;

        // Copy subject into buf_a as initial input.
        buf_a.n = subject.n;
        for (int i = 0; i < subject.n; ++i) {
            buf_a.x[i] = subject.x[i];
            buf_a.y[i] = subject.y[i];
        }

        // For each edge of the clip polygon, clip the current polygon.
        for (int ci = 0; ci < clip.n; ++ci) {
            int cj = (ci + 1) % clip.n;

            // Clip edge from clip.{x,y}[ci] to clip.{x,y}[cj]
            double ex0 = clip.x[ci];
            double ey0 = clip.y[ci];
            double ex1 = clip.x[cj];
            double ey1 = clip.y[cj];

            // Edge direction vector
            double edx = ex1 - ex0;
            double edy = ey1 - ey0;

            // Skip degenerate edges (zero-length)
            if (edx * edx + edy * edy < 1.0e-30) {
                continue;
            }

            // Clip buf_a against this edge, output to buf_b.
            buf_b.clear();
            clip_against_edge(buf_a, ex0, ey0, edx, edy, buf_b);

            // Swap: buf_b becomes the new input for the next edge.
            buf_a.n = buf_b.n;
            for (int i = 0; i < buf_b.n; ++i) {
                buf_a.x[i] = buf_b.x[i];
                buf_a.y[i] = buf_b.y[i];
            }

            // Early exit if clipped to nothing.
            if (buf_a.n < 3) {
                return 0.0;
            }
        }

        return buf_a.area();
    }

   private:
    /// @brief Compute the signed distance from a point to a directed edge.
    ///
    /// Positive values indicate the point is on the "inside" (left side)
    /// of the directed edge from (ex0, ey0) to (ex0+edx, ey0+edy).
    ///
    /// The signed distance is the cross product of the edge direction with
    /// the vector from the edge start to the point.
    KOKKOS_FUNCTION
    static double signed_distance(double px, double py, double ex0, double ey0, double edx, double edy) noexcept {
        // Cross product: (edge_dir) × (point - edge_start)
        // = edx * (py - ey0) - edy * (px - ex0)
        return edx * (py - ey0) - edy * (px - ex0);
    }

    /// @brief Compute the intersection of a line segment (p0→p1) with the
    ///        infinite line defined by an edge.
    ///
    /// Uses the parametric form: intersection = p0 + t * (p1 - p0)
    /// where t = d0 / (d0 - d1), and d0, d1 are signed distances of p0, p1
    /// from the edge line.
    KOKKOS_FUNCTION
    static void compute_intersection(double p0x, double p0y, double p1x, double p1y, double d0, double d1, double &ix, double &iy) noexcept {
        double t = d0 / (d0 - d1);
        ix = p0x + t * (p1x - p0x);
        iy = p0y + t * (p1y - p0y);
    }

    /// @brief Clip a polygon against a single directed edge.
    ///
    /// This is the core Sutherland-Hodgman step: for each edge of the input
    /// polygon, output vertices based on inside/outside classification relative
    /// to the clipping edge.
    ///
    /// The four cases:
    ///   - Both inside: output end vertex
    ///   - Inside → outside: output intersection
    ///   - Outside → inside: output intersection + end vertex
    ///   - Both outside: output nothing
    template <int M>
    KOKKOS_FUNCTION static void clip_against_edge(const PlanarPolygon<M> &input, double ex0, double ey0, double edx, double edy,
                                                  PlanarPolygon<M> &output) noexcept {
        if (input.n < 1) return;

        for (int i = 0; i < input.n; ++i) {
            int prev_idx = (i + input.n - 1) % input.n;

            double curr_x = input.x[i];
            double curr_y = input.y[i];
            double prev_x = input.x[prev_idx];
            double prev_y = input.y[prev_idx];

            double d_curr = signed_distance(curr_x, curr_y, ex0, ey0, edx, edy);
            double d_prev = signed_distance(prev_x, prev_y, ex0, ey0, edx, edy);

            // "Inside" means on the left side of the directed edge (d >= 0).
            // Use a small negative tolerance for inclusive boundary handling.
            constexpr double eps = -1.0e-14;
            bool curr_inside = (d_curr >= eps);
            bool prev_inside = (d_prev >= eps);

            if (curr_inside) {
                if (!prev_inside) {
                    // Outside → inside: add intersection point
                    double ix, iy;
                    compute_intersection(prev_x, prev_y, curr_x, curr_y, d_prev, d_curr, ix, iy);
                    output.push(ix, iy);
                }
                // Add current vertex
                output.push(curr_x, curr_y);
            } else if (prev_inside) {
                // Inside → outside: add intersection point only
                double ix, iy;
                compute_intersection(prev_x, prev_y, curr_x, curr_y, d_prev, d_curr, ix, iy);
                output.push(ix, iy);
            }
            // Both outside: output nothing.
        }
    }
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_PLANAR_CLIPPER_HPP
