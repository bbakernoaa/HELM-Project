// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_DATELINE_HANDLER_HPP
#define AXIS_DETAIL_DATELINE_HANDLER_HPP

/// @file axis/detail/dateline_handler.hpp
/// @brief GPU-portable dateline and pole handling for spherical polygon
///        operations.
///
/// Provides:
///   - DatelineHandler::crosses_dateline: detect longitude discontinuity
///   - DatelineHandler::normalize: shift longitudes to continuous range
///   - DatelineHandler::contains_pole: winding-number pole containment test
///
/// All functions are annotated KOKKOS_FUNCTION for device portability.
/// No heap allocation — safe for use in Kokkos parallel kernels.

#include <Kokkos_Core.hpp>
#include <axis/detail/spherical_clipper.hpp>

namespace axis::detail {

/// @brief Device-portable handlers for dateline-crossing cells and polar cells.
///
/// Longitudes are assumed to be in radians throughout, in the range [-π, π].
/// The Vec3 type from spherical_clipper.hpp represents Cartesian unit-sphere
/// coordinates.
struct DatelineHandler {

    /// @brief Detect if a polygon's vertices span the dateline (±180° / ±π).
    ///
    /// A dateline crossing is detected when any pair of adjacent vertices has
    /// a longitude gap exceeding π radians (180°).
    ///
    /// @param lons Array of vertex longitudes in radians.
    /// @param n Number of vertices.
    /// @return true if the cell crosses the dateline.
    KOKKOS_FUNCTION
    static bool crosses_dateline(const double* lons, int n) noexcept {
        if (n < 2) return false;

        constexpr double pi = 3.14159265358979323846;

        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            double diff = lons[j] - lons[i];
            // Normalize difference to [-π, π]
            if (diff > pi) diff -= 2.0 * pi;
            if (diff < -pi) diff += 2.0 * pi;
            // If the absolute difference exceeds π, there's a dateline crossing
            // But we already normalized, so check the raw gap:
            double raw_diff = lons[j] - lons[i];
            if (raw_diff > pi || raw_diff < -pi) {
                return true;
            }
        }
        return false;
    }

    /// @brief Normalize longitudes to a continuous range for dateline-crossing
    ///        cells.
    ///
    /// Finds the largest angular gap between consecutive (sorted) longitudes,
    /// then shifts all values so the range is continuous (max - min < 2π).
    /// This ensures polygon operations work correctly across the discontinuity.
    ///
    /// Longitudes are modified in-place. After normalization, the range
    /// [min(lons), max(lons)] will be less than 2π wide.
    ///
    /// @param lons Array of vertex longitudes in radians (modified in-place).
    /// @param n Number of vertices.
    KOKKOS_FUNCTION
    static void normalize(double* lons, int n) noexcept {
        if (n < 2) return;

        constexpr double two_pi = 2.0 * 3.14159265358979323846;

        // First, bring all longitudes to [0, 2π)
        for (int i = 0; i < n; ++i) {
            while (lons[i] < 0.0) lons[i] += two_pi;
            while (lons[i] >= two_pi) lons[i] -= two_pi;
        }

        // Sort longitudes to find the largest gap.
        // Use a simple insertion sort (n is small, typically 3-8 vertices).
        // We work on a copy to preserve vertex order.
        // MaxVerts matches SphericalPolygon capacity.
        constexpr int kMaxVerts = 32;
        double sorted[kMaxVerts];
        int count = (n <= kMaxVerts) ? n : kMaxVerts;
        for (int i = 0; i < count; ++i) {
            sorted[i] = lons[i];
        }
        // Insertion sort
        for (int i = 1; i < count; ++i) {
            double key = sorted[i];
            int j = i - 1;
            while (j >= 0 && sorted[j] > key) {
                sorted[j + 1] = sorted[j];
                --j;
            }
            sorted[j + 1] = key;
        }

        // Find the largest gap between consecutive sorted longitudes.
        // Include the wrap-around gap (from last to first + 2π).
        double max_gap = 0.0;
        double gap_start = 0.0;  // Upper edge of the largest gap

        for (int i = 0; i < count - 1; ++i) {
            double gap = sorted[i + 1] - sorted[i];
            if (gap > max_gap) {
                max_gap = gap;
                gap_start = sorted[i + 1];
            }
        }
        // Wrap-around gap
        double wrap_gap = (sorted[0] + two_pi) - sorted[count - 1];
        if (wrap_gap > max_gap) {
            max_gap = wrap_gap;
            gap_start = sorted[0] + two_pi;  // shift reference
        }

        // Shift all longitudes so they start just after the largest gap.
        // The reference angle is the upper edge of the largest gap.
        // Shift all values to [gap_start, gap_start + 2π).
        double ref = gap_start;
        for (int i = 0; i < n; ++i) {
            // Shift relative to reference
            double shifted = lons[i] - ref;
            // Bring into [0, 2π)
            while (shifted < 0.0) shifted += two_pi;
            while (shifted >= two_pi) shifted -= two_pi;
            lons[i] = shifted + ref;
        }
    }

    /// @brief Test if a geographic pole lies within a polygon using the winding
    ///        number method on the unit sphere.
    ///
    /// Computes the solid angle subtended by the polygon boundary around the
    /// pole direction. If the winding number is non-zero, the pole is inside.
    ///
    /// The winding number is computed by summing the signed azimuthal angles
    /// of the polygon edges as seen from the pole.
    ///
    /// @param verts Array of polygon vertices as unit-sphere Cartesian (Vec3).
    /// @param n Number of vertices.
    /// @param north If true, test the north pole (0,0,1); otherwise south pole
    ///        (0,0,-1).
    /// @return true if the specified pole is contained within the polygon.
    KOKKOS_FUNCTION
    static bool contains_pole(const Vec3* verts, int n, bool north) noexcept {
        if (n < 3) return false;

        constexpr double pi = 3.14159265358979323846;
        constexpr double two_pi = 2.0 * pi;

        // Compute the winding number by summing azimuthal angle changes
        // around the pole as we traverse the polygon boundary.

        // Compute azimuthal angles for each vertex projected onto the
        // equatorial plane perpendicular to the pole axis.
        //
        // For the north pole we look "down" along +z: azimuth = atan2(y, x).
        // For the south pole we look "up" along -z: flip the x-axis to
        // maintain a consistent right-hand orientation when viewed from the
        // pole's side of the sphere.
        double total_angle = 0.0;

        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;

            double xi = north ? verts[i].x : -verts[i].x;
            double yi = verts[i].y;
            double xj = north ? verts[j].x : -verts[j].x;
            double yj = verts[j].y;

            double angle_i = Kokkos::atan2(yi, xi);
            double angle_j = Kokkos::atan2(yj, xj);

            // Compute the angular difference, normalized to [-π, π]
            double delta = angle_j - angle_i;
            if (delta > pi) delta -= two_pi;
            if (delta < -pi) delta += two_pi;

            total_angle += delta;
        }

        // Winding number is total_angle / (2π).
        // If |winding| >= 1 (i.e., total_angle close to ±2π), pole is inside.
        double winding = total_angle / two_pi;

        // Use a threshold to account for floating-point imprecision.
        // A winding number of ±1 means the pole is inside.
        return Kokkos::fabs(winding) > 0.5;
    }
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_DATELINE_HANDLER_HPP
