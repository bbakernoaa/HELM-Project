// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_GNOMONIC_PROJECTOR_HPP
#define AXIS_DETAIL_GNOMONIC_PROJECTOR_HPP

/// @file axis/detail/gnomonic_projector.hpp
/// @brief GPU-portable gnomonic tangent-plane projection and true bilinear
///        interpolation on the unit sphere.
///
/// Provides:
///   - GnomonicProjector: forward/inverse gnomonic projection
///   - bilinear_weights: Newton iteration for isoparametric quad in projected space
///     with IDW fallback on non-convergence
///
/// All functions are annotated KOKKOS_FUNCTION for device portability.
/// No heap allocation — safe for use in Kokkos parallel kernels.

#include <Kokkos_Core.hpp>
#include <axis/detail/spherical_clipper.hpp>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// GnomonicProjector — tangent-plane projection and true bilinear weights
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Device-portable gnomonic projection and bilinear weight computation.
///
/// The gnomonic projection maps points from the unit sphere onto a tangent plane
/// at a given center point. Great circles project to straight lines in the local
/// neighborhood, making it ideal for bilinear interpolation on spherical quads.
///
/// The bilinear weight computation uses Newton iteration to solve for the
/// isoparametric coordinates (xi, eta) of a point within a projected quad,
/// falling back to inverse-distance weighting if Newton fails to converge.
struct GnomonicProjector {
    /// @brief Project a point on the unit sphere onto the tangent plane at center.
    ///
    /// Computes local east/north basis vectors at the center point and projects
    /// the given point onto that plane. Points on the opposite hemisphere from
    /// center (dot(center, point) <= 0) are clamped to (0, 0).
    ///
    /// @param center The tangent point (unit vector).
    /// @param point  The point to project (unit vector).
    /// @param[out] u Eastward coordinate on the tangent plane.
    /// @param[out] v Northward coordinate on the tangent plane.
    KOKKOS_FUNCTION
    static void forward(const Vec3 &center, const Vec3 &point, double &u, double &v) noexcept {
        // Build local tangent-plane basis at center.
        Vec3 east, north;
        compute_basis(center, east, north);

        // Cosine of angle between center and point.
        double d = dot(center, point);

        // Point on opposite hemisphere — cannot project meaningfully.
        if (d <= 0.0) {
            u = 0.0;
            v = 0.0;
            return;
        }

        // Gnomonic projection: project onto tangent plane.
        double inv_d = 1.0 / d;
        u = dot(east, point) * inv_d;
        v = dot(north, point) * inv_d;
    }

    /// @brief Reconstruct a unit-sphere point from tangent-plane coordinates.
    ///
    /// Given (u, v) on the tangent plane at center, reconstructs the 3-D point
    /// and normalizes it back onto the unit sphere.
    ///
    /// @param center The tangent point (unit vector).
    /// @param u Eastward coordinate.
    /// @param v Northward coordinate.
    /// @return The reconstructed unit-sphere point.
    KOKKOS_FUNCTION
    static Vec3 inverse(const Vec3 &center, double u, double v) noexcept {
        Vec3 east, north;
        compute_basis(center, east, north);

        // Reconstruct the 3-D point: center + u*east + v*north
        Vec3 point{center.x + u * east.x + v * north.x, center.y + u * east.y + v * north.y, center.z + u * east.z + v * north.z};

        return normalize(point);
    }

    /// @brief Compute bilinear shape function weights for a point inside a
    ///        projected quadrilateral using Newton iteration.
    ///
    /// Given a quadrilateral with vertices at (quad_u[0..3], quad_v[0..3]) in
    /// projected (tangent-plane) space and a point at (pu, pv), finds the
    /// bilinear shape function weights w[0..3] via Newton iteration on the
    /// isoparametric coordinates (xi, eta) ∈ [-1, 1]².
    ///
    /// Vertex ordering follows the standard bilinear convention:
    ///   0: (-1,-1), 1: (+1,-1), 2: (+1,+1), 3: (-1,+1)
    ///
    /// If Newton converges (residual < 1e-12 within 20 iterations), the weights
    /// are computed from the converged (xi, eta). Otherwise, returns false and
    /// the caller should use an IDW fallback.
    ///
    /// @param quad_u Array of 4 u-coordinates of the quad vertices.
    /// @param quad_v Array of 4 v-coordinates of the quad vertices.
    /// @param pu     The u-coordinate of the target point.
    /// @param pv     The v-coordinate of the target point.
    /// @param[out] weights Array of 4 bilinear weights (sum to ~1.0 if converged).
    /// @return true if Newton converged; false if IDW fallback is needed.
    KOKKOS_FUNCTION
    static bool bilinear_weights(const double quad_u[4], const double quad_v[4], double pu, double pv, double weights[4]) noexcept {
        constexpr int max_iter = 20;
        constexpr double tol = 1.0e-12;

        // Start Newton iteration from center of reference element.
        double xi = 0.0;
        double eta = 0.0;

        for (int iter = 0; iter < max_iter; ++iter) {
            // Evaluate bilinear map at current (xi, eta):
            //   x(xi,eta) = sum_i N_i(xi,eta) * quad_u[i]
            //   y(xi,eta) = sum_i N_i(xi,eta) * quad_v[i]
            // where N_0 = (1-xi)(1-eta)/4, N_1 = (1+xi)(1-eta)/4,
            //       N_2 = (1+xi)(1+eta)/4, N_3 = (1-xi)(1+eta)/4
            double n0 = (1.0 - xi) * (1.0 - eta) * 0.25;
            double n1 = (1.0 + xi) * (1.0 - eta) * 0.25;
            double n2 = (1.0 + xi) * (1.0 + eta) * 0.25;
            double n3 = (1.0 - xi) * (1.0 + eta) * 0.25;

            double x_val = n0 * quad_u[0] + n1 * quad_u[1] + n2 * quad_u[2] + n3 * quad_u[3];
            double y_val = n0 * quad_v[0] + n1 * quad_v[1] + n2 * quad_v[2] + n3 * quad_v[3];

            // Residual: target minus current.
            double rx = pu - x_val;
            double ry = pv - y_val;

            // Check convergence.
            if (rx * rx + ry * ry < tol * tol) {
                // Converged — compute weights from final (xi, eta).
                weights[0] = (1.0 - xi) * (1.0 - eta) * 0.25;
                weights[1] = (1.0 + xi) * (1.0 - eta) * 0.25;
                weights[2] = (1.0 + xi) * (1.0 + eta) * 0.25;
                weights[3] = (1.0 - xi) * (1.0 + eta) * 0.25;
                return true;
            }

            // Jacobian of the bilinear map:
            //   dx/dxi  = dN0/dxi*u0 + dN1/dxi*u1 + dN2/dxi*u2 + dN3/dxi*u3
            //   dx/deta = dN0/deta*u0 + ...
            // dN/dxi:  -( 1-eta)/4, (1-eta)/4, (1+eta)/4, -(1+eta)/4
            // dN/deta: -(1-xi)/4, -(1+xi)/4, (1+xi)/4, (1-xi)/4
            double dn0_dxi = -(1.0 - eta) * 0.25;
            double dn1_dxi = (1.0 - eta) * 0.25;
            double dn2_dxi = (1.0 + eta) * 0.25;
            double dn3_dxi = -(1.0 + eta) * 0.25;

            double dn0_deta = -(1.0 - xi) * 0.25;
            double dn1_deta = -(1.0 + xi) * 0.25;
            double dn2_deta = (1.0 + xi) * 0.25;
            double dn3_deta = (1.0 - xi) * 0.25;

            double j11 = dn0_dxi * quad_u[0] + dn1_dxi * quad_u[1] + dn2_dxi * quad_u[2] + dn3_dxi * quad_u[3];
            double j12 = dn0_deta * quad_u[0] + dn1_deta * quad_u[1] + dn2_deta * quad_u[2] + dn3_deta * quad_u[3];
            double j21 = dn0_dxi * quad_v[0] + dn1_dxi * quad_v[1] + dn2_dxi * quad_v[2] + dn3_dxi * quad_v[3];
            double j22 = dn0_deta * quad_v[0] + dn1_deta * quad_v[1] + dn2_deta * quad_v[2] + dn3_deta * quad_v[3];

            // Solve 2x2 linear system: J * [dxi, deta]^T = [rx, ry]^T
            double det = j11 * j22 - j12 * j21;
            if (Kokkos::fabs(det) < 1.0e-30) {
                // Singular Jacobian — Newton cannot proceed.
                break;
            }

            double inv_det = 1.0 / det;
            double dxi = (j22 * rx - j12 * ry) * inv_det;
            double deta = (-j21 * rx + j11 * ry) * inv_det;

            xi += dxi;
            eta += deta;
        }

        // Newton did not converge — signal failure (caller uses IDW fallback).
        // Set weights to zero to indicate no valid bilinear solution.
        weights[0] = 0.0;
        weights[1] = 0.0;
        weights[2] = 0.0;
        weights[3] = 0.0;
        return false;
    }

    /// @brief Compute inverse-distance weighted (IDW) fallback weights.
    ///
    /// Used when Newton iteration fails to converge for bilinear_weights.
    /// Computes weights proportional to 1/distance from the target point to
    /// each vertex in projected space.
    ///
    /// @param quad_u Array of 4 u-coordinates of the quad vertices.
    /// @param quad_v Array of 4 v-coordinates of the quad vertices.
    /// @param pu     The u-coordinate of the target point.
    /// @param pv     The v-coordinate of the target point.
    /// @param[out] weights Array of 4 IDW weights (non-negative, sum to 1.0).
    KOKKOS_FUNCTION
    static void idw_weights(const double quad_u[4], const double quad_v[4], double pu, double pv, double weights[4]) noexcept {
        constexpr double min_dist = 1.0e-15;

        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            double du = pu - quad_u[i];
            double dv = pv - quad_v[i];
            double dist = Kokkos::sqrt(du * du + dv * dv);

            if (dist < min_dist) {
                // Point coincides with a vertex — assign full weight.
                weights[0] = 0.0;
                weights[1] = 0.0;
                weights[2] = 0.0;
                weights[3] = 0.0;
                weights[i] = 1.0;
                return;
            }

            weights[i] = 1.0 / dist;
            sum += weights[i];
        }

        // Normalize to sum to 1.0.
        if (sum > 0.0) {
            double inv_sum = 1.0 / sum;
            for (int i = 0; i < 4; ++i) {
                weights[i] *= inv_sum;
            }
        }
    }

   private:
    /// @brief Compute a local orthonormal basis (east, north) on the tangent
    ///        plane at the given center point on the unit sphere.
    ///
    /// Uses cross product with the z-axis to get the east direction. If center
    /// is near a pole (z-component close to ±1), uses the x-axis as reference
    /// instead to avoid degeneracy.
    ///
    /// @param center The tangent point (unit vector).
    /// @param[out] east  Eastward basis vector (unit length).
    /// @param[out] north Northward basis vector (unit length).
    KOKKOS_FUNCTION
    static void compute_basis(const Vec3 &center, Vec3 &east, Vec3 &north) noexcept {
        // Choose a reference vector not parallel to center.
        // If center is near a pole (|z| > 0.9), use x-axis; otherwise z-axis.
        Vec3 ref;
        if (Kokkos::fabs(center.z) > 0.9) {
            ref = Vec3{1.0, 0.0, 0.0};
        } else {
            ref = Vec3{0.0, 0.0, 1.0};
        }

        // east = normalize(cross(ref, center))
        east = normalize(cross(ref, center));

        // north = cross(center, east) — already unit length since center and east
        // are orthonormal.
        north = cross(center, east);
    }
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_GNOMONIC_PROJECTOR_HPP
