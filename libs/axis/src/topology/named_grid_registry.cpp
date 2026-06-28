// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file named_grid_registry.cpp
/// @brief Implementation of NamedGridRegistry — on-the-fly generation of
///        standard global weather grids via Kokkos parallel kernels.
///
/// Grid generation algorithms (all zero file I/O):
///
///   O (Octahedral Gaussian): 2N latitude circles total (N per hemisphere).
///     Latitude circle j (0-indexed from nearest pole) has 20+4*j points.
///     Total cells connect adjacent latitude circles with quadrilateral elements.
///
///   F (Regular/Full Gaussian): 2N latitude circles, each with 4N points.
///     Total nodes = 2N * 4N. Quadrilateral cells between adjacent latitudes.
///
///   N (Reduced Gaussian): Follows ECMWF octahedral convention (same as O).
///
///   Gaussian latitudes are computed using Newton's method for Legendre
///   polynomial roots on the unit sphere (iterative, pure math, no tables).

#include <axis/topology/named_grid_registry.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/projection_builder.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

namespace axis::topology {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian latitude computation via Newton's method on Legendre polynomials
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Gaussian latitudes (in degrees, north-to-south) for 2N circles.
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x) where
/// n = 2N. Only the northern hemisphere roots are computed; southern hemisphere
/// is symmetric. Returns all 2N latitudes from north pole toward south pole.
std::vector<double> compute_gaussian_latitudes(int N) {
    const int n = 2 * N;  // degree of Legendre polynomial
    std::vector<double> latitudes(static_cast<std::size_t>(n));

    // We need the n roots of P_n(x) in (-1, 1).
    // Initial guesses using the Bretherton-Hoskins approximation.
    for (int i = 0; i < n; ++i) {
        // Initial guess for the i-th root (1-indexed: k = i+1)
        double k = static_cast<double>(i + 1);
        double theta = M_PI * (k - 0.25) / (static_cast<double>(n) + 0.5);
        double x = std::cos(theta);

        // Newton's method to refine the root of P_n(x)
        for (int iter = 0; iter < 100; ++iter) {
            // Evaluate P_n(x) and P_n'(x) via recurrence
            double p0 = 1.0;  // P_0(x)
            double p1 = x;    // P_1(x)

            for (int j = 2; j <= n; ++j) {
                double pj = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / static_cast<double>(j);
                p0 = p1;
                p1 = pj;
            }
            // p1 = P_n(x), p0 = P_{n-1}(x)
            // Derivative: P_n'(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
            //           = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)
            double dp = static_cast<double>(n) * (p0 - x * p1) / (1.0 - x * x);

            double dx = p1 / dp;
            x -= dx;

            if (std::abs(dx) < 1.0e-15) {
                break;
            }
        }

        // x is the cosine of the colatitude; latitude = asin(x) in degrees
        latitudes[static_cast<std::size_t>(i)] = std::asin(x) * (180.0 / M_PI);
    }

    // latitudes are computed from north to south (largest to smallest)
    // The Newton iteration with the given initial guess produces roots
    // in decreasing order of x (i.e., north to south). Verify ordering.
    // Sort descending to ensure north-to-south order.
    std::sort(latitudes.begin(), latitudes.end(), std::greater<double>());

    return latitudes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Points-per-latitude-circle functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the number of longitude points for each latitude circle in an
/// octahedral reduced Gaussian grid with parameter N.
/// ECMWF convention: nlon(j) = 20 + 4 * min(j, 2N-1-j), i.e. symmetric
/// about the equator with minimum 20 points at the poles.
/// Returns a vector of size 2*N with nlon for each circle from north to south.
std::vector<int> octahedral_nlons(int N) {
    const int n_lat = 2 * N;
    std::vector<int> nlons(static_cast<std::size_t>(n_lat));
    for (int j = 0; j < n_lat; ++j) {
        // Distance from nearest pole
        int dist_from_pole = std::min(j, n_lat - 1 - j);
        nlons[static_cast<std::size_t>(j)] = 20 + 4 * dist_from_pole;
    }
    return nlons;
}

/// Compute total number of nodes in an octahedral reduced Gaussian grid.
/// Formula: 4*N*(N+9) for ECMWF convention (minimum 20 points at poles).
std::size_t octahedral_total_nodes(int N) {
    // Sum all nlons
    std::size_t total = 0;
    for (int j = 0; j < 2 * N; ++j) {
        int dist = std::min(j, 2 * N - 1 - j);
        total += static_cast<std::size_t>(20 + 4 * dist);
    }
    return total;
}

/// Regular Gaussian: 2N latitudes, each with 4N longitudes.
/// Total nodes = 2N * 4N = 8*N^2.
std::size_t regular_total_nodes(int N) {
    return static_cast<std::size_t>(2 * N) * static_cast<std::size_t>(4 * N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell count computation
// ─────────────────────────────────────────────────────────────────────────────

/// For reduced Gaussian grids, cells are quadrilaterals connecting adjacent
/// latitude circles. Between circles j and j+1 with nlon_j and nlon_{j+1}
/// points, the number of cells is max(nlon_j, nlon_{j+1}).
/// Total cells = sum_{j=0}^{2N-2} max(nlon[j], nlon[j+1])
std::size_t reduced_total_cells(const std::vector<int>& nlons) {
    std::size_t total = 0;
    for (std::size_t j = 0; j + 1 < nlons.size(); ++j) {
        total += static_cast<std::size_t>(std::max(nlons[j], nlons[j + 1]));
    }
    return total;
}

/// For regular Gaussian grids, cells are simply (2N-1) * 4N quadrilaterals
/// (one ring of cells between each pair of adjacent latitude circles).
std::size_t regular_total_cells(int N) {
    return static_cast<std::size_t>(2 * N - 1) * static_cast<std::size_t>(4 * N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Regular Gaussian (F family)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_regular_gaussian(int N) {
    const int n_lat = 2 * N;
    const int n_lon = 4 * N;
    const std::size_t n_nodes = regular_total_nodes(N);
    const std::size_t n_cells = regular_total_cells(N);

    // Compute Gaussian latitudes
    std::vector<double> lats = compute_gaussian_latitudes(N);

    // ─── Build node coordinates on host ─────────────────────────────────────
    // Nodes are stored as [n_nodes, 2] with col 0 = lon, col 1 = lat (degrees)
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_coords("h_coords", n_nodes, 2);

    // Fill node coordinates: latitude circles from north to south,
    // each with n_lon evenly spaced longitudes in [0, 360)
    std::size_t node_idx = 0;
    for (int j = 0; j < n_lat; ++j) {
        double lat = lats[static_cast<std::size_t>(j)];
        double dlon = 360.0 / static_cast<double>(n_lon);
        for (int i = 0; i < n_lon; ++i) {
            h_coords(node_idx, 0) = static_cast<double>(i) * dlon;
            h_coords(node_idx, 1) = lat;
            ++node_idx;
        }
    }

    // ─── Build CSR connectivity on host ─────────────────────────────────────
    // Each cell is a quadrilateral connecting 4 nodes between adjacent lat
    // circles. Node ordering: bottom-left, bottom-right, top-right, top-left.
    // "Bottom" = farther from north pole (larger j index), "Top" = closer to pole.
    // Between lat circles j and j+1: n_lon cells (one per longitude column).
    // Connectivity wraps around in longitude.
    const std::size_t nnz = n_cells * 4;  // 4 nodes per quad cell

    Kokkos::View<index_t*, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t*, Kokkos::HostSpace> h_indices("h_indices", nnz);

    // Fill offsets: uniform 4 nodes per cell
    for (std::size_t c = 0; c <= n_cells; ++c) {
        h_offsets(c) = static_cast<index_t>(c * 4);
    }

    // Fill connectivity
    std::size_t cell_idx = 0;
    for (int j = 0; j < n_lat - 1; ++j) {
        std::size_t row_start = static_cast<std::size_t>(j) * static_cast<std::size_t>(n_lon);
        std::size_t next_row_start = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(n_lon);

        for (int i = 0; i < n_lon; ++i) {
            int i_next = (i + 1) % n_lon;

            // Quad: top-left, top-right, bottom-right, bottom-left
            // top = current row (j), bottom = next row (j+1)
            std::size_t tl = row_start + static_cast<std::size_t>(i);
            std::size_t tr = row_start + static_cast<std::size_t>(i_next);
            std::size_t br = next_row_start + static_cast<std::size_t>(i_next);
            std::size_t bl = next_row_start + static_cast<std::size_t>(i);

            std::size_t base = cell_idx * 4;
            h_indices(base + 0) = static_cast<index_t>(tl);
            h_indices(base + 1) = static_cast<index_t>(tr);
            h_indices(base + 2) = static_cast<index_t>(br);
            h_indices(base + 3) = static_cast<index_t>(bl);
            ++cell_idx;
        }
    }

    // ─── Copy to target MemorySpace ─────────────────────────────────────────
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return UnstructuredMesh<MemorySpace>(
            std::move(h_coords), std::move(h_offsets), std::move(h_indices),
            CoordinateSystem::SphericalDeg);
    } else {
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>
            d_coords("d_coords", n_nodes, 2);
        Kokkos::View<index_t*, MemorySpace> d_offsets("d_offsets", n_cells + 1);
        Kokkos::View<index_t*, MemorySpace> d_indices("d_indices", nnz);

        Kokkos::deep_copy(d_coords, h_coords);
        Kokkos::deep_copy(d_offsets, h_offsets);
        Kokkos::deep_copy(d_indices, h_indices);

        return UnstructuredMesh<MemorySpace>(
            std::move(d_coords), std::move(d_offsets), std::move(d_indices),
            CoordinateSystem::SphericalDeg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Octahedral / Reduced Gaussian (O and N families)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_octahedral_gaussian(int N) {
    const int n_lat = 2 * N;
    const std::vector<int> nlons = octahedral_nlons(N);
    const std::size_t n_nodes = octahedral_total_nodes(N);

    // For reduced grids, between adjacent latitude circles with different
    // numbers of points, we generate quadrilateral cells that properly
    // connect the two circles. The number of cells between circles j and j+1
    // is max(nlons[j], nlons[j+1]).
    const std::size_t n_cells = reduced_total_cells(nlons);

    // Compute Gaussian latitudes
    std::vector<double> lats = compute_gaussian_latitudes(N);

    // ─── Build node coordinates on host ─────────────────────────────────────
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_coords("h_coords", n_nodes, 2);

    // Compute row start offsets for quick node lookup
    std::vector<std::size_t> row_starts(static_cast<std::size_t>(n_lat) + 1);
    row_starts[0] = 0;
    for (int j = 0; j < n_lat; ++j) {
        row_starts[static_cast<std::size_t>(j) + 1] =
            row_starts[static_cast<std::size_t>(j)] +
            static_cast<std::size_t>(nlons[static_cast<std::size_t>(j)]);
    }

    // Fill node coordinates
    for (int j = 0; j < n_lat; ++j) {
        double lat = lats[static_cast<std::size_t>(j)];
        int nlon_j = nlons[static_cast<std::size_t>(j)];
        double dlon = 360.0 / static_cast<double>(nlon_j);
        std::size_t start = row_starts[static_cast<std::size_t>(j)];

        for (int i = 0; i < nlon_j; ++i) {
            std::size_t idx = start + static_cast<std::size_t>(i);
            h_coords(idx, 0) = static_cast<double>(i) * dlon;
            h_coords(idx, 1) = lat;
        }
    }

    // ─── Build CSR connectivity on host ─────────────────────────────────────
    // For reduced Gaussian grids, the connectivity between two adjacent
    // latitude circles with different nlon counts uses a "zipper" algorithm:
    // we walk both circles simultaneously, advancing the pointer on whichever
    // circle is behind in longitude, creating quadrilateral cells.
    //
    // Each cell connects 4 nodes (quadrilateral). For simplicity and
    // determinism, we use a straightforward proportional mapping approach:
    // for each cell between rows j and j+1, we map longitude indices
    // proportionally between the two rows.

    // First pass: count total connectivity entries (always 4 per quad cell)
    const std::size_t nnz = n_cells * 4;

    Kokkos::View<index_t*, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t*, Kokkos::HostSpace> h_indices("h_indices", nnz);

    // Fill offsets
    for (std::size_t c = 0; c <= n_cells; ++c) {
        h_offsets(c) = static_cast<index_t>(c * 4);
    }

    // Fill connectivity using proportional longitude mapping
    std::size_t cell_idx = 0;
    for (int j = 0; j < n_lat - 1; ++j) {
        int nlon_top = nlons[static_cast<std::size_t>(j)];
        int nlon_bot = nlons[static_cast<std::size_t>(j + 1)];
        std::size_t top_start = row_starts[static_cast<std::size_t>(j)];
        std::size_t bot_start = row_starts[static_cast<std::size_t>(j + 1)];
        int n_cells_ring = std::max(nlon_top, nlon_bot);

        for (int c = 0; c < n_cells_ring; ++c) {
            // Map cell index proportionally to both rows
            // Top row indices
            double frac_top = static_cast<double>(c) / static_cast<double>(n_cells_ring);
            double frac_top_next = static_cast<double>(c + 1) / static_cast<double>(n_cells_ring);

            int top_i = static_cast<int>(std::floor(frac_top * nlon_top)) % nlon_top;
            int top_i_next = static_cast<int>(std::floor(frac_top_next * nlon_top)) % nlon_top;

            // Bottom row indices
            double frac_bot = static_cast<double>(c) / static_cast<double>(n_cells_ring);
            double frac_bot_next = static_cast<double>(c + 1) / static_cast<double>(n_cells_ring);

            int bot_i = static_cast<int>(std::floor(frac_bot * nlon_bot)) % nlon_bot;
            int bot_i_next = static_cast<int>(std::floor(frac_bot_next * nlon_bot)) % nlon_bot;

            // Quad: top-left, top-right, bottom-right, bottom-left
            std::size_t tl = top_start + static_cast<std::size_t>(top_i);
            std::size_t tr = top_start + static_cast<std::size_t>(top_i_next);
            std::size_t br = bot_start + static_cast<std::size_t>(bot_i_next);
            std::size_t bl = bot_start + static_cast<std::size_t>(bot_i);

            std::size_t base = cell_idx * 4;
            h_indices(base + 0) = static_cast<index_t>(tl);
            h_indices(base + 1) = static_cast<index_t>(tr);
            h_indices(base + 2) = static_cast<index_t>(br);
            h_indices(base + 3) = static_cast<index_t>(bl);
            ++cell_idx;
        }
    }

    // ─── Copy to target MemorySpace ─────────────────────────────────────────
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return UnstructuredMesh<MemorySpace>(
            std::move(h_coords), std::move(h_offsets), std::move(h_indices),
            CoordinateSystem::SphericalDeg);
    } else {
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>
            d_coords("d_coords", n_nodes, 2);
        Kokkos::View<index_t*, MemorySpace> d_offsets("d_offsets", n_cells + 1);
        Kokkos::View<index_t*, MemorySpace> d_indices("d_indices", nnz);

        Kokkos::deep_copy(d_coords, h_coords);
        Kokkos::deep_copy(d_offsets, h_offsets);
        Kokkos::deep_copy(d_indices, h_indices);

        return UnstructuredMesh<MemorySpace>(
            std::move(d_coords), std::move(d_offsets), std::move(d_indices),
            CoordinateSystem::SphericalDeg);
    }
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: NOAA NWS GRIB Grids (G family - Extensible)
// ─────────────────────────────────────────────────────────────────────────────

/// @struct NoaaGribDefinition
/// @brief Declarative metadata structure for NOAA GRIB grids.
struct NoaaGribDefinition {
    int number;              ///< Official GRIB grid number.
    std::size_t ni;          ///< Columns count (Ni).
    std::size_t nj;          ///< Rows count (Nj).
    double lon_start;        ///< Leftmost longitude boundary (for regular grids).
    double lat_start;        ///< Southernmost latitude boundary (for regular grids).
    double dlon;             ///< Longitude grid spacing (for regular grids).
    double dlat;             ///< Latitude grid spacing (for regular grids).
    const char* proj_string; ///< PROJ-compliant string (nullptr if regular global grid).
};

/// @brief Global declarative registry of supported NOAA GRIB grids.
/// @details Adding a new grid to AXIS is a simple, single-line addition here!
static const NoaaGribDefinition NOAA_GRIB_GRIDS[] = {
    // grid3: GFS 1.0 degree global grid
    {3, 360, 181, -180.0, -90.0, 1.0, 1.0, nullptr},
    // grid4: GFS 0.5 degree global grid
    {4, 720, 361, -180.0, -90.0, 0.5, 0.5, nullptr},
    // grid218: NAM / RAP 12km ConUS Lambert Conformal grid (requires PROJ)
    {218, 614, 428, 0.0, 0.0, 0.0, 0.0,
     "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"}
};

/// @brief Total count of registered NOAA NWS grids in our static array.
static constexpr std::size_t NOAA_GRIB_GRIDS_COUNT = sizeof(NOAA_GRIB_GRIDS) / sizeof(NOAA_GRIB_GRIDS[0]);

/// @brief Generates a standard regular lat-lon grid as an UnstructuredMesh.
template <class MemorySpace>
inline UnstructuredMesh<MemorySpace> generate_regular_grid(
    std::size_t ni, std::size_t nj,
    double lon_start, double lat_start,
    double dlon, double dlat) {

    const std::size_t n_nodes = (ni + 1) * (nj + 1);
    const std::size_t n_cells = ni * nj;

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
        h_coords("h_coords", n_nodes, 2);

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            h_coords(idx, 0) = lon_start + static_cast<double>(i) * dlon;
            h_coords(idx, 1) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    Kokkos::View<index_t*, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t*, Kokkos::HostSpace> h_indices("h_indices", n_cells * 4);

    h_offsets(0) = 0;
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t cell_idx = i + j * ni;
            h_offsets(cell_idx + 1) = h_offsets(cell_idx) + 4;

            std::size_t n0 = i + j * (ni + 1);
            std::size_t n1 = (i + 1) + j * (ni + 1);
            std::size_t n2 = (i + 1) + (j + 1) * (ni + 1);
            std::size_t n3 = i + (j + 1) * (ni + 1);

            std::size_t indices_start = cell_idx * 4;
            h_indices(indices_start) = n0;
            h_indices(indices_start + 1) = n1;
            h_indices(indices_start + 2) = n2;
            h_indices(indices_start + 3) = n3;
        }
    }

    auto node_coords = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_coords);
    auto conn_offsets = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_offsets);
    auto conn_indices = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_indices);

    return UnstructuredMesh<MemorySpace>(
        std::move(node_coords),
        std::move(conn_offsets),
        std::move(conn_indices),
        CoordinateSystem::SphericalDeg);
}

/// @brief Analytically generates a registered NOAA NWS GRIB grid.
template <class MemorySpace>
inline UnstructuredMesh<MemorySpace> generate_noaa_grib_grid(int number) {
    for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
        const auto& grid_def = NOAA_GRIB_GRIDS[idx];
        if (grid_def.number == number) {
            if (grid_def.proj_string != nullptr) {
#ifndef AXIS_ENABLE_PROJ
                throw std::runtime_error(
                    "NamedGridRegistry::generate: requested projected grid \"grid" +
                    std::to_string(number) + "\" but AXIS was compiled without PROJ support.");
#else
                axis::ingest::ProjectedParams params;
                params.proj_string = grid_def.proj_string;

                const std::size_t ni = grid_def.ni;
                const std::size_t nj = grid_def.nj;
                const std::size_t n_points = ni * nj;

                // Set coordinates in projection space for NAM Grid 218
                double min_x = -3733392.0;
                double max_x =  3733392.0;
                double min_y = -2602779.0;
                double max_y =  2602779.0;
                double dx = (max_x - min_x) / (ni - 1);
                double dy = (max_y - min_y) / (nj - 1);

                std::vector<double> h_cx(n_points);
                std::vector<double> h_cy(n_points);
                for (std::size_t j = 0; j < nj; ++j) {
                    for (std::size_t i = 0; i < ni; ++i) {
                        std::size_t cell_idx = i + j * ni;
                        h_cx[cell_idx] = min_x + i * dx;
                        h_cy[cell_idx] = min_y + j * dy;
                    }
                }

                axis::ingest::BufferViews buffers;
                buffers.ni = ni;
                buffers.nj = nj;
                buffers.center_x = axis::field_view<const double, 1>(h_cx.data(), n_points);
                buffers.center_y = axis::field_view<const double, 1>(h_cy.data(), n_points);

                auto grid = axis::topology::ProjectionBuilder::build<MemorySpace>(params, buffers);
                return grid.to_unstructured();
#endif
            } else {
                return generate_regular_grid<MemorySpace>(
                    grid_def.ni, grid_def.nj,
                    grid_def.lon_start, grid_def.lat_start,
                    grid_def.dlon, grid_def.dlat);
            }
        }
    }

    throw std::invalid_argument(
        "NamedGridRegistry::generate: unregistered NOAA GRIB grid number grid" +
        std::to_string(number));
}

// ─────────────────────────────────────────────────────────────────────────────
// NamedGridRegistry public interface
// ─────────────────────────────────────────────────────────────────────────────

NamedGridRegistry::ParsedName NamedGridRegistry::parse(const std::string& name) {
    if (name.empty()) {
        throw std::invalid_argument(
            "NamedGridRegistry::parse: empty grid name string");
    }

    std::string lower_name = name;
    for (char& c : lower_name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Support case-insensitive NOAA GRIB grid numbers (e.g. "grid218")
    if (lower_name.rfind("grid", 0) == 0) {
        if (name.size() < 5) {
            throw std::invalid_argument(
                "NamedGridRegistry::parse: grid name \"" + name +
                "\" has no number after the 'grid' prefix");
        }
        std::string num_str = name.substr(4);
        for (char ch : num_str) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                throw std::invalid_argument(
                    "NamedGridRegistry::parse: non-numeric character '" +
                    std::string(1, ch) + "' in number portion of name \"" + name + "\"");
            }
        }
        int grid_num = 0;
        try {
            grid_num = std::stoi(num_str);
        } catch (...) {
            throw std::invalid_argument(
                "NamedGridRegistry::parse: cannot parse number from name \"" + name + "\"");
        }
        if (grid_num <= 0) {
            throw std::invalid_argument(
                "NamedGridRegistry::parse: grid number must be positive, got " +
                std::to_string(grid_num) + " in name \"" + name + "\"");
        }
        bool found_grib = false;
        for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
            if (NOAA_GRIB_GRIDS[idx].number == grid_num) {
                found_grib = true;
                break;
            }
        }
        if (!found_grib) {
            throw std::invalid_argument(
                "NamedGridRegistry::parse: unregistered NOAA GRIB grid number grid" +
                std::to_string(grid_num));
        }
        return ParsedName{'G', grid_num};
    }

    char family = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));

    // Validate family
    if (family != 'O' && family != 'F' && family != 'N') {
        throw std::invalid_argument(
            "NamedGridRegistry::parse: unknown grid family '" +
            std::string(1, name[0]) + "' in name \"" + name +
            "\"; registered families are O, F, N, and grid<num>");
    }

    // Parse number
    if (name.size() < 2) {
        throw std::invalid_argument(
            "NamedGridRegistry::parse: grid name \"" + name +
            "\" has no number after the family prefix");
    }

    std::string num_str = name.substr(1);

    // Validate that the remainder is a valid integer
    for (char ch : num_str) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::invalid_argument(
                "NamedGridRegistry::parse: non-numeric character '" +
                std::string(1, ch) + "' in number portion of name \"" + name + "\"");
        }
    }

    int number = 0;
    try {
        number = std::stoi(num_str);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "NamedGridRegistry::parse: cannot parse number from name \"" + name + "\"");
    }

    if (number <= 0) {
        throw std::invalid_argument(
            "NamedGridRegistry::parse: grid number must be positive, got " +
            std::to_string(number) + " in name \"" + name + "\"");
    }

    return ParsedName{family, number};
}

bool NamedGridRegistry::is_registered(const std::string& name) noexcept {
    try {
        parse(name);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<char> NamedGridRegistry::registered_families() {
    return {'F', 'N', 'O'};
}

// Explicit instantiation of generate for HostSpace
template <>
UnstructuredMesh<Kokkos::HostSpace>
NamedGridRegistry::generate<Kokkos::HostSpace>(const std::string& name) {
    ParsedName parsed = parse(name);

    switch (parsed.family) {
        case 'G':
            return generate_noaa_grib_grid<Kokkos::HostSpace>(parsed.number);
        case 'O':
            return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
        case 'N':
            // N (reduced Gaussian) follows ECMWF octahedral convention
            return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
        case 'F':
            return generate_regular_gaussian<Kokkos::HostSpace>(parsed.number);
        default:
            // Should not reach here due to parse() validation
            throw std::invalid_argument(
                "NamedGridRegistry::generate: unknown family '" +
                std::string(1, parsed.family) + "'");
    }
}

} // namespace axis::topology
