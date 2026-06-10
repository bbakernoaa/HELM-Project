// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_INGEST_GRID_DESCRIPTOR_HPP
#define AXIS_INGEST_GRID_DESCRIPTOR_HPP

/// @file axis/ingest/grid_descriptor.hpp
/// @brief The public plain-data ingest contract for AXIS.
///
/// GridDescriptor is the single, standard, plain-data value type through which
/// every file-backed grid enters AXIS. A producer (AMIO today, Python/pybind11
/// tomorrow) populates this descriptor; MeshFactory::from_descriptor consumes it.
///
/// The descriptor holds only plain enums, plain scalar fields, standard library
/// types, and non-owning std::mdspan<layout_left> buffer views. It contains
/// NO Kokkos type, NO AMIO type, NO eckit type, NO file handle, and NO file path.
/// This "trivially mirrorable in Python" property is an explicit design goal.
///
/// Header-only: no associated .cpp compilation unit.

#include <cstddef>
#include <cstdint>
#include <string>

#include <axis/types.hpp>

namespace axis::ingest {

// ─────────────────────────────────────────────────────────────────────────────
// ConventionKind — how to interpret the descriptor's metadata and buffers.
// ─────────────────────────────────────────────────────────────────────────────

/// How to interpret the descriptor's metadata and buffers. The producer sets
/// this after detecting the file's grid convention; AXIS branches on it ONCE
/// inside MeshFactory::from_descriptor.
enum class ConventionKind : std::uint8_t {
    CF,         ///< CF-conventions structured grid (grid_mapping + coord vars)
    UGRID,      ///< UGRID unstructured mesh (node/edge/face topology)
    GRIB,       ///< GRIB2 grid-description keys (gridType, Ni/Nj, Gaussian N, ...)
    Projected,  ///< Regular grid in a PROJ/proj4 projection (AXIS transforms via PROJ)
    NamedGrid,  ///< A registry token (e.g. "O1280"); AXIS generates it in-memory
    GridRules   ///< Rule parameters (kind, bbox, resolution, gaussian_n); AXIS generates
};

// ─────────────────────────────────────────────────────────────────────────────
// CoordinateSystem — shared with topology; defined here so the ingest contract
// is self-contained (no dependency on topology headers).
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinate system for node/cell coordinates.
enum class CoordinateSystem : std::uint8_t {
    SphericalDeg,   ///< Geographic lon/lat in degrees
    SphericalRad,   ///< Geographic lon/lat in radians
    Cartesian3D     ///< 3-D Cartesian (x, y, z)
};

// ─────────────────────────────────────────────────────────────────────────────
// Convention-specific metadata structs — plain data only.
// Only the member matching GridDescriptor::convention is meaningful.
// ─────────────────────────────────────────────────────────────────────────────

/// CF grid_mapping parameters (cf-conventions ch.5.6).
struct CfParams {
    std::string grid_mapping_name;          ///< e.g. "latitude_longitude", "lambert_conformal_conic"
    double earth_radius{0.0};               ///< optional Earth radius (m); 0 = unset
    double semi_major{0.0};                 ///< optional semi-major axis (m); 0 = unset
    double semi_minor{0.0};                 ///< optional semi-minor axis (m); 0 = unset
    double inverse_flattening{0.0};         ///< optional inverse flattening; 0 = unset
};

/// UGRID topology attributes (ugrid-conventions).
struct UgridParams {
    int         topology_dimension{2};      ///< 2 (surface) or 3 (volume)
    int         start_index{0};             ///< 0- or 1-based connectivity offset
    std::string mesh_name;                  ///< mesh topology variable name
};

/// GRIB grid-description keys (as surfaced by AMIO's g2c driver).
struct GribParams {
    std::string grid_type;                  ///< "regular_ll", "regular_gg", "reduced_gg", ...
    std::int64_t ni{0};                     ///< points along a parallel (Ni)
    std::int64_t nj{0};                     ///< points along a meridian (Nj)
    std::int64_t gaussian_n{0};             ///< Gaussian truncation N (0 if not Gaussian)
    std::uint8_t scanning_mode{0};          ///< GRIB scanning mode byte
    int          earth_shape_code{0};       ///< GRIB earth shape code
};

/// Projected-grid parameters. AXIS transforms via PROJ (its only optional dep).
struct ProjectedParams {
    std::string proj_string;                ///< proj4 / PROJ definition string
};

/// Named-grid parameters: just the registry token. AXIS generates in-memory.
struct NamedGridParams {
    std::string name;                       ///< e.g. "O1280", "F128", "N320"
};

/// Rule-based parameters (a GridRules YAML is parsed by the PRODUCER into these
/// plain fields; AXIS never parses YAML or JSON).
struct GridRulesParams {
    std::string kind;                       ///< "RegularLatLon", "GaussianRegular", etc.
    double min_x{-180.0};                   ///< bounding box min x
    double max_x{180.0};                    ///< bounding box max x
    double min_y{-90.0};                    ///< bounding box min y
    double max_y{90.0};                     ///< bounding box max y
    double r_x{1.0};                        ///< resolution in x direction
    double r_y{1.0};                        ///< resolution in y direction
    std::int64_t gaussian_n{0};             ///< Gaussian truncation N (0 if not Gaussian)
    std::string proj_string;                ///< proj4 string when kind == "Projected"
};

// ─────────────────────────────────────────────────────────────────────────────
// BufferViews — non-owning layout_left mdspan views over decoded arrays.
// ─────────────────────────────────────────────────────────────────────────────

/// Non-owning views over the DECODED arrays the producer supplies. AXIS copies
/// out of these into the target Kokkos memory space (explicit deep_copy, no UVM);
/// the descriptor OWNS NOTHING. All views are std::layout_left (HELM lingua
/// franca). Unused members are left empty (extent 0) per ConventionKind.
struct BufferViews {
    // ── Structured (CF / GRIB): 1-D center coordinate arrays ────────────────
    field_view<const double, 1> center_x{};           ///< lon/x centers [n_points]
    field_view<const double, 1> center_y{};           ///< lat/y centers [n_points]

    // ── Structured: corner coordinate arrays (optional) ─────────────────────
    field_view<const double, 1> corner_x{};           ///< vertex x coordinates
    field_view<const double, 1> corner_y{};           ///< vertex y coordinates

    // ── Unstructured (UGRID): node coordinates + CSR connectivity ───────────
    field_view<const double, 2> node_coords{};        ///< [n_nodes, ndim]
    field_view<const index_t, 1> conn_offsets{};      ///< CSR offsets [n_cells + 1]
    field_view<const index_t, 1> conn_indices{};      ///< CSR column indices

    // ── Optional per-cell metadata (any convention) ─────────────────────────
    field_view<const double, 1> cell_areas{};         ///< precomputed cell areas (optional)
    field_view<const int, 1>    cell_mask{};          ///< 0 = masked, 1 = active (optional)

    // ── Structured grid dimensions (for CF / GRIB) ──────────────────────────
    std::size_t ni{0};                                ///< structured dim i (0 if unstructured)
    std::size_t nj{0};                                ///< structured dim j (0 if unstructured)
};

// ─────────────────────────────────────────────────────────────────────────────
// GridDescriptor — THE INGEST CONTRACT
// ─────────────────────────────────────────────────────────────────────────────

/// THE INGEST CONTRACT. A producer fills this; MeshFactory::from_descriptor
/// consumes it. Pure data: copyable, no destructor logic, no owned resources.
///
/// Only the convention-specific params member matching `kind` is meaningful;
/// the rest are default-constructed. This flat layout (instead of std::variant)
/// is chosen so a Python layer can populate by name without variant gymnastics.
struct GridDescriptor {
    /// Which grid convention the producer detected in the file.
    ConventionKind   kind{ConventionKind::CF};

    /// Coordinate system for the buffer data.
    CoordinateSystem coord_system{CoordinateSystem::SphericalDeg};

    // ── Convention-specific metadata (only one is meaningful per `kind`) ─────
    CfParams         cf{};
    UgridParams      ugrid{};
    GribParams       grib{};
    ProjectedParams  projected{};
    NamedGridParams  named_grid{};
    GridRulesParams  grid_rules{};

    // ── Decoded buffers (empty for NamedGrid / GridRules, generated by AXIS) ─
    BufferViews      buffers{};
};

} // namespace axis::ingest

#endif // AXIS_INGEST_GRID_DESCRIPTOR_HPP
