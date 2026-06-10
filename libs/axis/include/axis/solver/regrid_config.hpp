// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_REGRID_CONFIG_HPP
#define AXIS_SOLVER_REGRID_CONFIG_HPP

/// @file axis/solver/regrid_config.hpp
/// @brief Solver configuration enumerations and the RegridConfig struct.
///
/// Defines InterpolationMethod, NormType, LineType, and UnmappedAction enums
/// along with the RegridConfig struct that captures all runtime options for
/// weight generation. Header-only — no associated .cpp compilation unit.

#include <cstdint>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// InterpolationMethod — the spatial interpolation algorithm.
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the interpolation method used by WeightGenerator.
enum class InterpolationMethod : std::uint8_t {
    Bilinear,              ///< True bilinear: point-in-cell location + barycentric weights
    NearestNeighbor,       ///< Nearest source cell centroid to each destination point (weight = 1.0)
    Bicubic,               ///< Bicubic: 4×4 stencil cubic interpolation (CDO remapbic equivalent)
    Patch,                 ///< Patch recovery: least-squares polynomial fit over local patch (ESMF REGRID_METHOD_PATCH)
    Conservative1stOrder   ///< Area-weighted first-order conservative (overlap-based via ArborX BVH + Sutherland-Hodgman)
};

// ─────────────────────────────────────────────────────────────────────────────
// NormType — normalization semantics for conservative weights.
// ─────────────────────────────────────────────────────────────────────────────

/// Controls how partially-covered destination cells are normalized.
/// Matches ESMF normalization semantics.
enum class NormType : std::uint8_t {
    DstArea,   ///< Unnormalized: dst_raw = frac_b * dst_true (default)
    FracArea   ///< Fraction baked in: apply yields true value directly
};

// ─────────────────────────────────────────────────────────────────────────────
// LineType — geometry of paths between grid points.
// ─────────────────────────────────────────────────────────────────────────────

/// Determines how lines between grid points are computed for overlap detection.
enum class LineType : std::uint8_t {
    Cartesian,    ///< Straight lines in projected (Cartesian) space
    GreatCircle   ///< Geodesic arcs on the sphere
};

// ─────────────────────────────────────────────────────────────────────────────
// UnmappedAction — behavior for destination cells with no source coverage.
// ─────────────────────────────────────────────────────────────────────────────

/// Controls what happens when a destination cell has no source overlap.
enum class UnmappedAction : std::uint8_t {
    Error,   ///< Throw std::runtime_error identifying the unmapped index
    Ignore   ///< Leave destination value at zero; no entry in the matrix
};

// ─────────────────────────────────────────────────────────────────────────────
// RegridConfig — aggregate configuration for weight generation.
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration struct capturing all runtime options for weight generation.
/// Passed by value to WeightGenerator::generate.
struct RegridConfig {
    InterpolationMethod method    = InterpolationMethod::Bilinear;
    NormType            norm_type = NormType::DstArea;
    LineType            line_type = LineType::GreatCircle;
    UnmappedAction      unmapped  = UnmappedAction::Ignore;
};

} // namespace axis::solver

#endif // AXIS_SOLVER_REGRID_CONFIG_HPP
