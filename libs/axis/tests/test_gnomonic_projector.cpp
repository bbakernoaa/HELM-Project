// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file tests/test_gnomonic_projector.cpp
/// @brief Unit tests for GnomonicProjector forward/inverse projection and
///        bilinear weight computation.

#include <gtest/gtest.h>
#include <axis/detail/gnomonic_projector.hpp>
#include <cmath>

using axis::detail::GnomonicProjector;
using axis::detail::Vec3;
using axis::detail::normalize;
using axis::detail::dot;
using axis::detail::length;
using axis::detail::add;
using axis::detail::scale;

namespace {

// Helper: create a unit vector from lat/lon in degrees.
Vec3 latlon_to_xyz(double lat_deg, double lon_deg) {
    constexpr double deg2rad = M_PI / 180.0;
    double lat = lat_deg * deg2rad;
    double lon = lon_deg * deg2rad;
    return {std::cos(lat) * std::cos(lon),
            std::cos(lat) * std::sin(lon),
            std::sin(lat)};
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Forward/Inverse Round-Trip Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(GnomonicProjector, ForwardInverseRoundTrip) {
    // Project a point and then reconstruct — should recover original.
    Vec3 center = latlon_to_xyz(45.0, 30.0);
    Vec3 point  = latlon_to_xyz(46.0, 31.0);

    double u, v;
    GnomonicProjector::forward(center, point, u, v);

    Vec3 recovered = GnomonicProjector::inverse(center, u, v);

    // Should match the original point within tight tolerance.
    EXPECT_NEAR(dot(point, recovered), 1.0, 1e-12);
}

TEST(GnomonicProjector, ForwardInverseAtEquator) {
    Vec3 center = latlon_to_xyz(0.0, 0.0);
    Vec3 point  = latlon_to_xyz(1.0, 1.0);

    double u, v;
    GnomonicProjector::forward(center, point, u, v);

    Vec3 recovered = GnomonicProjector::inverse(center, u, v);
    EXPECT_NEAR(dot(point, recovered), 1.0, 1e-12);
}

TEST(GnomonicProjector, ForwardInverseNearPole) {
    // Requirement 2.4: near-pole robustness.
    Vec3 center = latlon_to_xyz(89.0, 45.0);
    Vec3 point  = latlon_to_xyz(88.5, 50.0);

    double u, v;
    GnomonicProjector::forward(center, point, u, v);

    Vec3 recovered = GnomonicProjector::inverse(center, u, v);
    EXPECT_NEAR(dot(point, recovered), 1.0, 1e-12);
}

TEST(GnomonicProjector, ForwardCenterProjectsToOrigin) {
    // Projecting the center onto its own tangent plane should give (0,0).
    Vec3 center = latlon_to_xyz(30.0, -60.0);

    double u, v;
    GnomonicProjector::forward(center, center, u, v);

    EXPECT_NEAR(u, 0.0, 1e-14);
    EXPECT_NEAR(v, 0.0, 1e-14);
}

TEST(GnomonicProjector, ForwardOppositeHemisphere) {
    // Point on opposite hemisphere should project to (0,0).
    Vec3 center = latlon_to_xyz(45.0, 0.0);
    Vec3 point  = latlon_to_xyz(-46.0, 180.0);  // roughly antipodal

    double u, v;
    GnomonicProjector::forward(center, point, u, v);

    EXPECT_EQ(u, 0.0);
    EXPECT_EQ(v, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear Weights Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(GnomonicProjector, BilinearWeightsAtCenter) {
    // A simple unit square in projected space: vertices at (±1, ±1).
    // Point at center (0, 0) should get equal weights (0.25 each).
    double quad_u[4] = {-1.0, 1.0, 1.0, -1.0};
    double quad_v[4] = {-1.0, -1.0, 1.0, 1.0};

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, 0.0, 0.0, weights);

    EXPECT_TRUE(converged);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(weights[i], 0.25, 1e-12);
    }
}

TEST(GnomonicProjector, BilinearWeightsAtVertex) {
    // Point exactly at vertex 0 should get weight 1.0 for vertex 0.
    double quad_u[4] = {-1.0, 1.0, 1.0, -1.0};
    double quad_v[4] = {-1.0, -1.0, 1.0, 1.0};

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, -1.0, -1.0, weights);

    EXPECT_TRUE(converged);
    EXPECT_NEAR(weights[0], 1.0, 1e-12);
    EXPECT_NEAR(weights[1], 0.0, 1e-12);
    EXPECT_NEAR(weights[2], 0.0, 1e-12);
    EXPECT_NEAR(weights[3], 0.0, 1e-12);
}

TEST(GnomonicProjector, BilinearWeightsSumToOne) {
    // For any interior point, weights should sum to 1.0.
    double quad_u[4] = {-1.0, 1.0, 1.0, -1.0};
    double quad_v[4] = {-1.0, -1.0, 1.0, 1.0};

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, 0.3, -0.5, weights);

    EXPECT_TRUE(converged);
    double sum = weights[0] + weights[1] + weights[2] + weights[3];
    EXPECT_NEAR(sum, 1.0, 1e-12);

    // All weights should be non-negative for interior point.
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(weights[i], -1e-14);
    }
}

TEST(GnomonicProjector, BilinearWeightsNonSquareQuad) {
    // Test with a non-rectangular quad (trapezoidal shape).
    double quad_u[4] = {0.0, 2.0, 1.5, 0.5};
    double quad_v[4] = {0.0, 0.0, 1.0, 1.0};

    // Test at the centroid of the quad.
    double cu = (quad_u[0] + quad_u[1] + quad_u[2] + quad_u[3]) / 4.0;
    double cv = (quad_v[0] + quad_v[1] + quad_v[2] + quad_v[3]) / 4.0;

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, cu, cv, weights);

    EXPECT_TRUE(converged);
    double sum = weights[0] + weights[1] + weights[2] + weights[3];
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(GnomonicProjector, BilinearWeightsReproducePoint) {
    // The weighted sum of vertices should reconstruct the target point.
    double quad_u[4] = {-0.5, 0.5, 0.7, -0.3};
    double quad_v[4] = {-0.4, -0.6, 0.5, 0.8};

    double pu = 0.1;
    double pv = 0.05;

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, pu, pv, weights);

    EXPECT_TRUE(converged);

    // Reconstruct point from weights.
    double recon_u = 0.0, recon_v = 0.0;
    for (int i = 0; i < 4; ++i) {
        recon_u += weights[i] * quad_u[i];
        recon_v += weights[i] * quad_v[i];
    }

    EXPECT_NEAR(recon_u, pu, 1e-10);
    EXPECT_NEAR(recon_v, pv, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// IDW Fallback Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(GnomonicProjector, IdwWeightsSumToOne) {
    double quad_u[4] = {-1.0, 1.0, 1.0, -1.0};
    double quad_v[4] = {-1.0, -1.0, 1.0, 1.0};

    double weights[4];
    GnomonicProjector::idw_weights(quad_u, quad_v, 0.3, -0.2, weights);

    double sum = weights[0] + weights[1] + weights[2] + weights[3];
    EXPECT_NEAR(sum, 1.0, 1e-14);

    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(weights[i], 0.0);
    }
}

TEST(GnomonicProjector, IdwWeightsAtVertex) {
    // Point coinciding with vertex should get full weight there.
    double quad_u[4] = {-1.0, 1.0, 1.0, -1.0};
    double quad_v[4] = {-1.0, -1.0, 1.0, 1.0};

    double weights[4];
    GnomonicProjector::idw_weights(quad_u, quad_v, 1.0, 1.0, weights);

    EXPECT_NEAR(weights[2], 1.0, 1e-14);
    EXPECT_NEAR(weights[0], 0.0, 1e-14);
    EXPECT_NEAR(weights[1], 0.0, 1e-14);
    EXPECT_NEAR(weights[3], 0.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pole Robustness (Requirement 2.4)
// ─────────────────────────────────────────────────────────────────────────────

TEST(GnomonicProjector, NearPoleWeightsNonNegativeAndSumToOne) {
    // Build a quad around a point at 89° latitude.
    Vec3 center = latlon_to_xyz(89.0, 0.0);
    Vec3 v0 = latlon_to_xyz(88.5, -2.0);
    Vec3 v1 = latlon_to_xyz(88.5, 2.0);
    Vec3 v2 = latlon_to_xyz(89.5, 2.0);
    Vec3 v3 = latlon_to_xyz(89.5, -2.0);
    Vec3 target = latlon_to_xyz(89.0, 0.5);

    double quad_u[4], quad_v[4];
    GnomonicProjector::forward(center, v0, quad_u[0], quad_v[0]);
    GnomonicProjector::forward(center, v1, quad_u[1], quad_v[1]);
    GnomonicProjector::forward(center, v2, quad_u[2], quad_v[2]);
    GnomonicProjector::forward(center, v3, quad_u[3], quad_v[3]);

    double pu, pv;
    GnomonicProjector::forward(center, target, pu, pv);

    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(
        quad_u, quad_v, pu, pv, weights);

    // Should converge for a well-formed quad.
    EXPECT_TRUE(converged);

    double sum = weights[0] + weights[1] + weights[2] + weights[3];
    EXPECT_NEAR(sum, 1.0, 1e-14);

    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(weights[i], -1e-14);
    }
}
