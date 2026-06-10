// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/weight_generator.cpp
/// @brief WeightGenerator implementation — single-rank weight generation for
///        Bilinear, NearestNeighbor, Bicubic, Patch, and Conservative1stOrder.
///
/// Spatial queries use ArborX BoundingVolumeHierarchy (BVH) for GPU-portable
/// nearest-neighbor and intersection searches.  Polygon overlap for conservative
/// remapping uses Sutherland-Hodgman clipping on the host.

#include <axis/solver/weight_generator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include <axis/detail/spherical_geometry.hpp>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (anonymous namespace)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// ─────────────────────── Geometry types ──────────────────────────────────────

/// A simple 2-D point used for polygon clipping on the host.
struct Vec2 {
    double x{0.0};
    double y{0.0};
};

// ────────────────────── compute_cell_centroids_xy ────────────────────────────

/// Extract (x, y) centroids for all cells into separate Kokkos host Views.
/// coord column 0 → x (lon), coord column 1 → y (lat).
template <class MemorySpace>
void compute_cell_centroids_xy(
    const topology::UnstructuredMesh<MemorySpace>& mesh,
    Kokkos::View<double*, Kokkos::HostSpace>& cx_out,
    Kokkos::View<double*, Kokkos::HostSpace>& cy_out) {

    const auto n_cells = mesh.n_cells();
    const auto coords  = mesh.node_coords();   // [n_nodes, ndim]
    const auto offsets = mesh.conn_offsets();   // [n_cells + 1]
    const auto indices = mesh.conn_indices();   // [nnz]

    cx_out = Kokkos::View<double*, Kokkos::HostSpace>("cx", n_cells);
    cy_out = Kokkos::View<double*, Kokkos::HostSpace>("cy", n_cells);

    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = static_cast<std::size_t>(offsets[c]);
        auto end   = static_cast<std::size_t>(offsets[c + 1]);
        auto n_verts = end - start;

        double sx = 0.0, sy = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            auto ni = static_cast<std::size_t>(indices[i]);
            sx += coords(ni, 0);
            sy += coords(ni, 1);
        }

        double inv = (n_verts > 0) ? 1.0 / static_cast<double>(n_verts) : 0.0;
        cx_out(c) = sx * inv;
        cy_out(c) = sy * inv;
    }
}

// ─────────────────────── compute_cell_aabbs ─────────────────────────────────

/// Compute axis-aligned bounding boxes for all cells (min_x, min_y, max_x, max_y).
/// Returns a host-space View of ArborX::Box<2>.
template <class MemorySpace>
Kokkos::View<ArborX::Box<2>*, Kokkos::HostSpace>
compute_cell_aabbs(const topology::UnstructuredMesh<MemorySpace>& mesh) {

    const auto n_cells = mesh.n_cells();
    const auto coords  = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();

    Kokkos::View<ArborX::Box<2>*, Kokkos::HostSpace> boxes("cell_aabbs", n_cells);

    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = static_cast<std::size_t>(offsets[c]);
        auto end   = static_cast<std::size_t>(offsets[c + 1]);

        double min_x =  std::numeric_limits<double>::max();
        double min_y =  std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();

        for (std::size_t i = start; i < end; ++i) {
            auto ni = static_cast<std::size_t>(indices[i]);
            double x = coords(ni, 0);
            double y = coords(ni, 1);
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }

        boxes(c) = ArborX::Box<2>{
            {static_cast<float>(min_x), static_cast<float>(min_y)},
            {static_cast<float>(max_x), static_cast<float>(max_y)}};
    }

    return boxes;
}

// ──────────────────────────── get_cell_areas ─────────────────────────────────

/// Compute the area of a single cell via the shoelace formula (2-D polygons).
template <class MemorySpace>
double compute_single_cell_area(const topology::UnstructuredMesh<MemorySpace>& mesh,
                                std::size_t cell_idx) {
    const auto coords  = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();

    auto start = static_cast<std::size_t>(offsets[cell_idx]);
    auto end   = static_cast<std::size_t>(offsets[cell_idx + 1]);
    auto n_nodes = end - start;

    if (n_nodes < 3) return 0.0;

    double area = 0.0;
    for (std::size_t i = 0; i < n_nodes; ++i) {
        auto idx_curr = static_cast<std::size_t>(indices[start + i]);
        auto idx_next = static_cast<std::size_t>(indices[start + (i + 1) % n_nodes]);

        double x0 = coords(idx_curr, 0);
        double y0 = coords(idx_curr, 1);
        double x1 = coords(idx_next, 0);
        double y1 = coords(idx_next, 1);

        area += x0 * y1 - x1 * y0;
    }

    return std::abs(area) * 0.5;
}

/// Get cell areas: use precomputed mesh areas if available, else compute via shoelace.
template <class MemorySpace>
std::vector<double>
get_cell_areas(const topology::UnstructuredMesh<MemorySpace>& mesh) {
    const auto n_cells = mesh.n_cells();
    std::vector<double> areas(n_cells);

    auto mesh_areas = mesh.cell_areas();
    if (mesh_areas.extent(0) == n_cells) {
        for (std::size_t i = 0; i < n_cells; ++i) {
            areas[i] = mesh_areas[i];
        }
    } else {
        for (std::size_t i = 0; i < n_cells; ++i) {
            areas[i] = compute_single_cell_area(mesh, i);
        }
    }

    return areas;
}

// ──────────────────── Sutherland-Hodgman polygon clipping ────────────────────

/// Compute signed area of a polygon (positive = CCW winding).
inline double polygon_signed_area(const std::vector<Vec2>& poly) {
    double area = 0.0;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = (i + 1) % n;
        area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    return area * 0.5;
}

/// Compute the (unsigned) area of intersection between two 2-D polygons using
/// the Sutherland-Hodgman algorithm.
inline double compute_polygon_overlap_area(const std::vector<Vec2>& subject,
                                           const std::vector<Vec2>& clip) {
    if (subject.size() < 3 || clip.size() < 3) return 0.0;

    std::vector<Vec2> output = subject;
    const std::size_t clip_n = clip.size();

    for (std::size_t i = 0; i < clip_n; ++i) {
        if (output.empty()) return 0.0;

        std::vector<Vec2> input = output;
        output.clear();

        const Vec2& edge_start = clip[i];
        const Vec2& edge_end   = clip[(i + 1) % clip_n];

        double ex = edge_end.x - edge_start.x;
        double ey = edge_end.y - edge_start.y;

        auto inside = [&](const Vec2& p) -> bool {
            return (ex * (p.y - edge_start.y) - ey * (p.x - edge_start.x)) >= 0.0;
        };

        auto intersect = [&](const Vec2& a, const Vec2& b) -> Vec2 {
            double ax = b.x - a.x;
            double ay = b.y - a.y;
            double denom = ax * ey - ay * ex;
            if (std::abs(denom) < 1e-30) {
                return {0.5 * (a.x + b.x), 0.5 * (a.y + b.y)};
            }
            double t = (ex * (a.y - edge_start.y) - ey * (a.x - edge_start.x)) / denom;
            return {a.x + t * ax, a.y + t * ay};
        };

        const std::size_t input_n = input.size();
        for (std::size_t j = 0; j < input_n; ++j) {
            const Vec2& curr = input[j];
            const Vec2& prev = input[(j + input_n - 1) % input_n];

            bool curr_in = inside(curr);
            bool prev_in = inside(prev);

            if (curr_in) {
                if (!prev_in) {
                    output.push_back(intersect(prev, curr));
                }
                output.push_back(curr);
            } else if (prev_in) {
                output.push_back(intersect(prev, curr));
            }
        }
    }

    if (output.size() < 3) return 0.0;
    return std::abs(polygon_signed_area(output));
}

// ──────────────────────── extract_cell_polygon ──────────────────────────────

/// Extract the vertex ring of a given cell as a vector of Vec2.
template <class MemorySpace>
std::vector<Vec2>
extract_cell_polygon(const topology::UnstructuredMesh<MemorySpace>& mesh,
                     std::size_t cell_idx) {
    const auto coords  = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();

    auto start = static_cast<std::size_t>(offsets[cell_idx]);
    auto end   = static_cast<std::size_t>(offsets[cell_idx + 1]);

    std::vector<Vec2> poly;
    poly.reserve(end - start);
    for (std::size_t i = start; i < end; ++i) {
        auto ni = static_cast<std::size_t>(indices[i]);
        poly.push_back({coords(ni, 0), coords(ni, 1)});
    }
    return poly;
}

// ─────────────────── Spherical polygon extraction ──────────────────────────

/// Extract the vertex ring of a given cell as a vector of unit-sphere Vec3.
/// Coordinates are interpreted based on coordinate system:
///   - SphericalDeg: lon/lat in degrees → converted to XYZ
///   - SphericalRad: lon/lat in radians → converted to XYZ
///   - Cartesian3D: returned as-is (assumes data is already on unit sphere, or
///     caller is using the flat Cartesian path)
template <class MemorySpace>
std::vector<axis::detail::spherical::Vec3>
extract_cell_polygon_spherical(const topology::UnstructuredMesh<MemorySpace>& mesh,
                               std::size_t cell_idx) {
    using axis::detail::spherical::Vec3;
    using axis::detail::spherical::lonlat_to_xyz;

    const auto coords  = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();

    auto start = static_cast<std::size_t>(offsets[cell_idx]);
    auto end   = static_cast<std::size_t>(offsets[cell_idx + 1]);

    std::vector<Vec3> poly;
    poly.reserve(end - start);

    auto csys = mesh.coord_system();

    for (std::size_t i = start; i < end; ++i) {
        auto ni = static_cast<std::size_t>(indices[i]);
        double c0 = coords(ni, 0);  // lon or x
        double c1 = coords(ni, 1);  // lat or y

        if (csys == topology::CoordinateSystem::SphericalDeg) {
            constexpr double deg2rad = axis::detail::spherical::pi / 180.0;
            poly.push_back(lonlat_to_xyz(c0 * deg2rad, c1 * deg2rad));
        } else if (csys == topology::CoordinateSystem::SphericalRad) {
            poly.push_back(lonlat_to_xyz(c0, c1));
        } else {
            // Cartesian3D: treat (c0, c1) as (x, y) with z=0 projected to sphere.
            // This fallback shouldn't normally be used for spherical path.
            double z = (coords.extent(1) > 2) ? coords(ni, 2) : 0.0;
            double len = std::sqrt(c0 * c0 + c1 * c1 + z * z);
            if (len > 1e-30) {
                poly.push_back({c0 / len, c1 / len, z / len});
            } else {
                poly.push_back({0.0, 0.0, 1.0});
            }
        }
    }
    return poly;
}

/// Compute the spherical area of a single cell using the spherical excess formula.
template <class MemorySpace>
double compute_single_cell_area_spherical(
    const topology::UnstructuredMesh<MemorySpace>& mesh,
    std::size_t cell_idx) {
    auto poly = extract_cell_polygon_spherical(mesh, cell_idx);
    return axis::detail::spherical::spherical_polygon_area(poly);
}

/// Get cell areas on the sphere: use precomputed if available, else compute.
template <class MemorySpace>
std::vector<double>
get_cell_areas_spherical(const topology::UnstructuredMesh<MemorySpace>& mesh) {
    const auto n_cells = mesh.n_cells();
    std::vector<double> areas(n_cells);

    auto mesh_areas = mesh.cell_areas();
    if (mesh_areas.extent(0) == n_cells) {
        for (std::size_t i = 0; i < n_cells; ++i) {
            areas[i] = mesh_areas[i];
        }
    } else {
        for (std::size_t i = 0; i < n_cells; ++i) {
            areas[i] = compute_single_cell_area_spherical(mesh, i);
        }
    }
    return areas;
}

// ─────────────────────── Point-in-polygon test ──────────────────────────────

/// Winding number test: returns true if point (px, py) is inside the polygon.
inline bool point_in_polygon(double px, double py, const std::vector<Vec2>& poly) {
    const std::size_t n = poly.size();
    if (n < 3) return false;

    int winding = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = (i + 1) % n;
        double y0 = poly[i].y;
        double y1 = poly[j].y;

        if (y0 <= py) {
            if (y1 > py) {
                // Upward crossing
                double cross = (poly[j].x - poly[i].x) * (py - poly[i].y)
                             - (px - poly[i].x) * (poly[j].y - poly[i].y);
                if (cross > 0.0) ++winding;
            }
        } else {
            if (y1 <= py) {
                // Downward crossing
                double cross = (poly[j].x - poly[i].x) * (py - poly[i].y)
                             - (px - poly[i].x) * (poly[j].y - poly[i].y);
                if (cross < 0.0) --winding;
            }
        }
    }
    return winding != 0;
}

// ──────────── Bilinear shape functions for quads (Newton iteration) ──────────

/// Map physical point (px, py) to reference coordinates (xi, eta) in [-1,1]^2
/// for a quadrilateral with vertices v0..v3 (in CCW or CW order).
/// Returns true on convergence, false otherwise.
inline bool map_to_reference_quad(double px, double py,
                                  const Vec2& v0, const Vec2& v1,
                                  const Vec2& v2, const Vec2& v3,
                                  double& xi_out, double& eta_out) {
    // Newton iteration to solve:
    //   x(xi,eta) = N0*x0 + N1*x1 + N2*x2 + N3*x3 = px
    //   y(xi,eta) = N0*y0 + N1*y1 + N2*y2 + N3*y3 = py
    // where Ni = (1 ± xi)(1 ± eta)/4

    double xi = 0.0, eta = 0.0;
    constexpr int max_iter = 20;
    constexpr double tol = 1e-12;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Shape functions
        double N0 = 0.25 * (1.0 - xi) * (1.0 - eta);
        double N1 = 0.25 * (1.0 + xi) * (1.0 - eta);
        double N2 = 0.25 * (1.0 + xi) * (1.0 + eta);
        double N3 = 0.25 * (1.0 - xi) * (1.0 + eta);

        // Current mapped position
        double x_cur = N0 * v0.x + N1 * v1.x + N2 * v2.x + N3 * v3.x;
        double y_cur = N0 * v0.y + N1 * v1.y + N2 * v2.y + N3 * v3.y;

        // Residual
        double rx = px - x_cur;
        double ry = py - y_cur;

        if (std::abs(rx) < tol && std::abs(ry) < tol) {
            xi_out = xi;
            eta_out = eta;
            return true;
        }

        // Jacobian: dN/dxi, dN/deta
        double dN0_dxi = -0.25 * (1.0 - eta);
        double dN1_dxi =  0.25 * (1.0 - eta);
        double dN2_dxi =  0.25 * (1.0 + eta);
        double dN3_dxi = -0.25 * (1.0 + eta);

        double dN0_deta = -0.25 * (1.0 - xi);
        double dN1_deta = -0.25 * (1.0 + xi);
        double dN2_deta =  0.25 * (1.0 + xi);
        double dN3_deta =  0.25 * (1.0 - xi);

        double dx_dxi  = dN0_dxi * v0.x + dN1_dxi * v1.x + dN2_dxi * v2.x + dN3_dxi * v3.x;
        double dy_dxi  = dN0_dxi * v0.y + dN1_dxi * v1.y + dN2_dxi * v2.y + dN3_dxi * v3.y;
        double dx_deta = dN0_deta * v0.x + dN1_deta * v1.x + dN2_deta * v2.x + dN3_deta * v3.x;
        double dy_deta = dN0_deta * v0.y + dN1_deta * v1.y + dN2_deta * v2.y + dN3_deta * v3.y;

        // Solve 2x2 system: J * [dxi, deta]^T = [rx, ry]^T
        double det = dx_dxi * dy_deta - dx_deta * dy_dxi;
        if (std::abs(det) < 1e-30) return false;

        double inv_det = 1.0 / det;
        double dxi  = inv_det * ( dy_deta * rx - dx_deta * ry);
        double deta = inv_det * (-dy_dxi  * rx + dx_dxi  * ry);

        xi  += dxi;
        eta += deta;

        // Clamp to prevent divergence
        xi  = std::max(-2.0, std::min(2.0, xi));
        eta = std::max(-2.0, std::min(2.0, eta));
    }

    xi_out = xi;
    eta_out = eta;
    // Check if final result is inside reference element (with tolerance)
    return (std::abs(xi) <= 1.0 + 1e-6 && std::abs(eta) <= 1.0 + 1e-6);
}

/// Compute barycentric coordinates for point (px, py) in triangle (v0, v1, v2).
/// Returns true if the point is inside (all coords in [0,1]).
inline bool barycentric_triangle(double px, double py,
                                 const Vec2& v0, const Vec2& v1, const Vec2& v2,
                                 double& l0, double& l1, double& l2) {
    double denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    if (std::abs(denom) < 1e-30) {
        l0 = l1 = l2 = 1.0 / 3.0;
        return false;
    }
    double inv = 1.0 / denom;
    l0 = ((v1.y - v2.y) * (px - v2.x) + (v2.x - v1.x) * (py - v2.y)) * inv;
    l1 = ((v2.y - v0.y) * (px - v2.x) + (v0.x - v2.x) * (py - v2.y)) * inv;
    l2 = 1.0 - l0 - l1;

    constexpr double eps = -1e-10;
    return (l0 >= eps && l1 >= eps && l2 >= eps);
}

// ──────────── Dense linear algebra helpers (host-only) ──────────────────────

/// Solve a dense linear system A * x = b using Gaussian elimination with
/// partial pivoting. A is n×n stored row-major in a flat vector.
/// b is the RHS vector of length n. Solution overwrites b.
/// Returns true on success.
inline bool dense_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int pivot_row = col;
        double pivot_val = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = row;
            }
        }

        if (pivot_val < 1e-14) return false; // Singular

        // Swap rows
        if (pivot_row != col) {
            for (int j = col; j < n; ++j) {
                std::swap(A[col * n + j], A[pivot_row * n + j]);
            }
            std::swap(b[col], b[pivot_row]);
        }

        // Eliminate below
        double diag = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / diag;
            for (int j = col + 1; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            A[row * n + col] = 0.0;
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[row];
        for (int j = row + 1; j < n; ++j) {
            sum -= A[row * n + j] * b[j];
        }
        b[row] = sum / A[row * n + row];
    }

    return true;
}

/// Solve a least-squares system A * x = b where A is m×n (m >= n).
/// Uses normal equations: (A^T A) x = A^T b.
/// A is stored row-major [m*n], b is [m], x_out is [n].
/// Returns true on success.
inline bool least_squares_solve(const std::vector<double>& A_in, 
                                const std::vector<double>& b_in,
                                int m, int n,
                                std::vector<double>& x_out) {
    // Form A^T * A (n×n) and A^T * b (n)
    std::vector<double> AtA(n * n, 0.0);
    std::vector<double> Atb(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += A_in[k * n + i] * A_in[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
        double sum_b = 0.0;
        for (int k = 0; k < m; ++k) {
            sum_b += A_in[k * n + i] * b_in[k];
        }
        Atb[i] = sum_b;
    }

    // Solve (A^T A) x = A^T b
    if (!dense_solve(AtA, Atb, n)) return false;

    x_out = std::move(Atb);
    return true;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// generate — top-level dispatch
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate(const topology::UnstructuredMesh<MemorySpace>& src_mesh,
                          const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
                          const RegridConfig& config) {
    // Coordinate system mismatch check
    if (src_mesh.coord_system() != dst_mesh.coord_system()) {
        throw std::invalid_argument(
            "WeightGenerator::generate: source and destination meshes have "
            "different CoordinateSystem values (src="
            + std::to_string(static_cast<int>(src_mesh.coord_system()))
            + ", dst="
            + std::to_string(static_cast<int>(dst_mesh.coord_system()))
            + ")");
    }

    switch (config.method) {
        case InterpolationMethod::Bilinear:
            return generate_bilinear(src_mesh, dst_mesh, config);
        case InterpolationMethod::NearestNeighbor:
            return generate_nearest(src_mesh, dst_mesh, config);
        case InterpolationMethod::Bicubic:
            return generate_bicubic(src_mesh, dst_mesh, config);
        case InterpolationMethod::Patch:
            return generate_patch(src_mesh, dst_mesh, config);
        case InterpolationMethod::Conservative1stOrder:
            return generate_conservative(src_mesh, dst_mesh, config);
        default:
            throw std::invalid_argument(
                "WeightGenerator::generate: unknown InterpolationMethod");
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// generate_nearest — ArborX nearest(point, 1) query
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate_nearest(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config) {

    using HostSpace = Kokkos::HostSpace;
    using Point2    = ArborX::Point<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    // ── Compute source centroids and build ArborX BVH ──
    Kokkos::View<double*, HostSpace> src_cx, src_cy;
    compute_cell_centroids_xy(src_mesh, src_cx, src_cy);

    Kokkos::View<Point2*, HostSpace> src_points("src_points", n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_points(i) = Point2{static_cast<float>(src_cx(i)),
                               static_cast<float>(src_cy(i))};
    }

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(
        host_exec, ArborX::Experimental::attach_indices(src_points));

    // ── Build nearest(point, 1) queries for each dst cell ──
    Kokkos::View<double*, HostSpace> dst_cx, dst_cy;
    compute_cell_centroids_xy(dst_mesh, dst_cx, dst_cy);

    Kokkos::View<decltype(ArborX::nearest(Point2{}, 1))*, HostSpace>
        queries("queries", n_dst);
    for (std::size_t j = 0; j < n_dst; ++j) {
        queries(j) = ArborX::nearest(
            Point2{static_cast<float>(dst_cx(j)),
                   static_cast<float>(dst_cy(j))},
            1);
    }

    // ── Execute query ──
    Kokkos::View<typename decltype(tree)::value_type*, HostSpace> values("values", 0);
    Kokkos::View<int*, HostSpace> offsets("offsets", 0);
    tree.query(host_exec, queries, values, offsets);

    // ── Build COO entries ──
    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;
    weights_vec.reserve(n_dst);
    rows_vec.reserve(n_dst);
    cols_vec.reserve(n_dst);

    for (std::size_t j = 0; j < n_dst; ++j) {
        int begin = offsets(j);
        int end   = offsets(j + 1);
        if (begin == end) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error(
                    "WeightGenerator::generate_nearest: unmapped destination cell "
                    + std::to_string(j));
            }
            continue;
        }
        auto src_idx = static_cast<std::size_t>(values(begin).index);
        weights_vec.push_back(1.0);
        rows_vec.push_back(static_cast<index_t>(j));
        cols_vec.push_back(static_cast<index_t>(src_idx));
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);
    auto h_area_a      = Kokkos::create_mirror_view(area_a);
    auto h_area_b      = Kokkos::create_mirror_view(area_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k)  = rows_vec[k];
        h_factor_col(k)  = cols_vec[k];
    }

    auto src_areas = get_cell_areas(src_mesh);
    auto dst_areas = get_cell_areas(dst_mesh);

    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = src_areas[i];
        h_frac_a(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = 1.0;
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}


// ─────────────────────────────────────────────────────────────────────────────
// generate_bilinear — Point-in-cell location + barycentric/shape-function weights
//
// Algorithm:
//   1. For each destination centroid, find the nearest source cell (k=1).
//   2. Test if the point is inside that cell.
//   3. If inside a quad: compute bilinear shape function weights via Newton
//      iteration to reference coordinates (ξ, η).
//   4. If inside a triangle: compute barycentric coordinates directly.
//   5. If not inside the nearest cell: fall back to IDW on the k=4 nearest.
//   6. area_a is set to 0.0 (ESMF bilinear convention).
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate_bilinear(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config) {

    using HostSpace = Kokkos::HostSpace;
    using Point2    = ArborX::Point<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    // Number of neighbors for IDW fallback
    const int k_fallback = static_cast<int>(std::min(static_cast<std::size_t>(4), n_src));

    // ── Compute source centroids and build ArborX BVH ──
    Kokkos::View<double*, HostSpace> src_cx, src_cy;
    compute_cell_centroids_xy(src_mesh, src_cx, src_cy);

    Kokkos::View<Point2*, HostSpace> src_points("src_points", n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_points(i) = Point2{static_cast<float>(src_cx(i)),
                               static_cast<float>(src_cy(i))};
    }

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(
        host_exec, ArborX::Experimental::attach_indices(src_points));

    // ── Compute destination centroids ──
    Kokkos::View<double*, HostSpace> dst_cx, dst_cy;
    compute_cell_centroids_xy(dst_mesh, dst_cx, dst_cy);

    // ── Build nearest(point, k_fallback) queries for IDW fallback ──
    // We query k_fallback neighbors; the first is used for point-in-cell test
    Kokkos::View<decltype(ArborX::nearest(Point2{}, 1))*, HostSpace>
        queries("queries", n_dst);
    for (std::size_t j = 0; j < n_dst; ++j) {
        queries(j) = ArborX::nearest(
            Point2{static_cast<float>(dst_cx(j)),
                   static_cast<float>(dst_cy(j))},
            k_fallback);
    }

    // ── Execute query ──
    Kokkos::View<typename decltype(tree)::value_type*, HostSpace> values("values", 0);
    Kokkos::View<int*, HostSpace> offsets_view("offsets", 0);
    tree.query(host_exec, queries, values, offsets_view);

    // ── Mesh connectivity accessors for point-in-cell ──
    const auto coords  = src_mesh.node_coords();
    const auto conn_off = src_mesh.conn_offsets();
    const auto conn_idx = src_mesh.conn_indices();

    // ── Build COO entries ──
    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    for (std::size_t j = 0; j < n_dst; ++j) {
        int begin = offsets_view(j);
        int end   = offsets_view(j + 1);

        if (begin == end) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error(
                    "WeightGenerator::generate_bilinear: unmapped destination cell "
                    + std::to_string(j));
            }
            continue;
        }

        double px = dst_cx(j);
        double py = dst_cy(j);

        // Try point-in-cell on the nearest source cell
        auto nearest_src = static_cast<std::size_t>(values(begin).index);

        // Get the vertex ring of the nearest source cell
        auto poly = extract_cell_polygon(src_mesh, nearest_src);
        std::size_t n_verts = poly.size();

        bool used_shape_functions = false;

        if (point_in_polygon(px, py, poly)) {
            // Point is inside the nearest cell — use shape functions
            auto cell_start = static_cast<std::size_t>(conn_off[nearest_src]);

            if (n_verts == 4) {
                // Quad cell: bilinear shape functions via Newton iteration
                double xi = 0.0, eta = 0.0;
                if (map_to_reference_quad(px, py, poly[0], poly[1], poly[2], poly[3],
                                          xi, eta)) {
                    // Clamp to [-1, 1] for safety
                    xi  = std::max(-1.0, std::min(1.0, xi));
                    eta = std::max(-1.0, std::min(1.0, eta));

                    double w0 = 0.25 * (1.0 - xi) * (1.0 - eta);
                    double w1 = 0.25 * (1.0 + xi) * (1.0 - eta);
                    double w2 = 0.25 * (1.0 + xi) * (1.0 + eta);
                    double w3 = 0.25 * (1.0 - xi) * (1.0 + eta);

                    // Get the node indices for this cell (these are the vertices
                    // that form the quad — we store weights keyed by cell index,
                    // but ESMF convention is cell-centroid based weights. We store
                    // the weight as if interpolating from the containing cell
                    // using its vertex values. However, the matrix maps
                    // src_cell → dst_cell. For bilinear on cell centroids, we
                    // contribute the single source cell with weight=1 if it's a
                    // coincident point, or distribute among the 4 neighbor cells.
                    //
                    // Actually, for proper bilinear interpolation in an
                    // unstructured cell-centered scheme, the weights w0..w3 apply
                    // to the VERTICES of the source cell. But our matrix is
                    // cell-to-cell. The standard ESMF approach for bilinear on
                    // unstructured meshes is: the destination point is inside a
                    // source cell, and the weights are the shape function values
                    // at the vertices of that cell. The "source" entries in the
                    // matrix are the NODE-based values. But our system is
                    // cell-based. So we use the 4 neighbor cells as sources.
                    //
                    // For cell-centroid data: the containing cell gets weight 1
                    // if the point is at the centroid, or we can use the shape
                    // functions with the 4 surrounding cell centroids as a
                    // higher-order stencil. The simplest correct approach for
                    // cell-centered data on an unstructured mesh: the nearest
                    // cell contributes the value. But ESMF bilinear actually
                    // works on NODAL meshes. For cell-centered bilinear:
                    // find the containing cell, use its vertices' owning cells.
                    //
                    // Since this library operates on cell centroids (like CDO),
                    // and the ArborX query returns the k=4 nearest cell centroids,
                    // we use the shape function weights applied to those 4 cell
                    // centroids as sources. This is the standard cell-centered
                    // bilinear approach used in CDO/SCRIP.

                    // We use the 4 nearest cell centroids as the "quad vertices"
                    // and compute bilinear weights in that local coordinate system.
                    // But first check: did we enter this branch because the dst
                    // point is inside the nearest cell's polygon? If so, the
                    // correct cell-centered bilinear approach is to use the
                    // containing cell with weight 1.0 (nearest-neighbor on cells).
                    //
                    // CORRECTION: For true bilinear on cell-centered data, ESMF
                    // finds the containing source ELEMENT and uses shape function
                    // weights on its CORNER nodes. Our nodes ARE the cell corners.
                    // So the weight for each corner node of the containing cell
                    // is the shape function value. But our factor_col refers to
                    // CELL indices, not NODE indices.
                    //
                    // Resolution: Store weights for the containing source cell's
                    // CORNER NODES treated as cell indices won't work. Instead,
                    // the standard approach for cell-centered bilinear:
                    // - Use the 4 nearest CELL centroids as interpolation points
                    // - The containing cell test validates we have a good stencil
                    // - Apply bilinear weights computed from the reference mapping
                    //   to those 4 cells' centroids.
                    //
                    // FINAL APPROACH: Since the point IS inside this cell, we know
                    // the cell contains it. The bilinear weights w0..w3 apply to
                    // the 4 vertex positions of THIS cell. The source cell itself
                    // provides the value. For cell-centered fields, the correct
                    // bilinear uses the shape functions with vertex NODES to
                    // identify which source cells own those nodes. But in our
                    // simplified model, we store the containing cell with weight 1
                    // if it's a single cell, OR we use the shape-function weights
                    // distributed to the 4 nearest cell centroids.
                    //
                    // We go with: shape function weights on the k=4 nearest cells.

                    // Map dst point in the reference frame of the 4 nearest centroids
                    // Get the 4 nearest source cell indices
                    int n_avail = end - begin;
                    if (n_avail >= 4) {
                        std::size_t c0 = static_cast<std::size_t>(values(begin).index);
                        std::size_t c1 = static_cast<std::size_t>(values(begin + 1).index);
                        std::size_t c2 = static_cast<std::size_t>(values(begin + 2).index);
                        std::size_t c3 = static_cast<std::size_t>(values(begin + 3).index);

                        Vec2 q0{src_cx(c0), src_cy(c0)};
                        Vec2 q1{src_cx(c1), src_cy(c1)};
                        Vec2 q2{src_cx(c2), src_cy(c2)};
                        Vec2 q3{src_cx(c3), src_cy(c3)};

                        double xi2 = 0.0, eta2 = 0.0;
                        if (map_to_reference_quad(px, py, q0, q1, q2, q3, xi2, eta2)) {
                            xi2  = std::max(-1.0, std::min(1.0, xi2));
                            eta2 = std::max(-1.0, std::min(1.0, eta2));

                            double ww0 = 0.25 * (1.0 - xi2) * (1.0 - eta2);
                            double ww1 = 0.25 * (1.0 + xi2) * (1.0 - eta2);
                            double ww2 = 0.25 * (1.0 + xi2) * (1.0 + eta2);
                            double ww3 = 0.25 * (1.0 - xi2) * (1.0 + eta2);

                            weights_vec.push_back(ww0);
                            rows_vec.push_back(static_cast<index_t>(j));
                            cols_vec.push_back(static_cast<index_t>(c0));

                            weights_vec.push_back(ww1);
                            rows_vec.push_back(static_cast<index_t>(j));
                            cols_vec.push_back(static_cast<index_t>(c1));

                            weights_vec.push_back(ww2);
                            rows_vec.push_back(static_cast<index_t>(j));
                            cols_vec.push_back(static_cast<index_t>(c2));

                            weights_vec.push_back(ww3);
                            rows_vec.push_back(static_cast<index_t>(j));
                            cols_vec.push_back(static_cast<index_t>(c3));

                            used_shape_functions = true;
                        }
                    }
                }
            } else if (n_verts == 3) {
                // Triangle: barycentric coordinates on the 3 nearest cell centroids
                int n_avail = end - begin;
                if (n_avail >= 3) {
                    std::size_t c0 = static_cast<std::size_t>(values(begin).index);
                    std::size_t c1 = static_cast<std::size_t>(values(begin + 1).index);
                    std::size_t c2 = static_cast<std::size_t>(values(begin + 2).index);

                    Vec2 q0{src_cx(c0), src_cy(c0)};
                    Vec2 q1{src_cx(c1), src_cy(c1)};
                    Vec2 q2{src_cx(c2), src_cy(c2)};

                    double l0 = 0.0, l1 = 0.0, l2 = 0.0;
                    if (barycentric_triangle(px, py, q0, q1, q2, l0, l1, l2)) {
                        // Clamp weights to [0, 1] and renormalize
                        l0 = std::max(0.0, l0);
                        l1 = std::max(0.0, l1);
                        l2 = std::max(0.0, l2);
                        double sum = l0 + l1 + l2;
                        if (sum > 0.0) {
                            l0 /= sum; l1 /= sum; l2 /= sum;
                        } else {
                            l0 = l1 = l2 = 1.0 / 3.0;
                        }

                        weights_vec.push_back(l0);
                        rows_vec.push_back(static_cast<index_t>(j));
                        cols_vec.push_back(static_cast<index_t>(c0));

                        weights_vec.push_back(l1);
                        rows_vec.push_back(static_cast<index_t>(j));
                        cols_vec.push_back(static_cast<index_t>(c1));

                        weights_vec.push_back(l2);
                        rows_vec.push_back(static_cast<index_t>(j));
                        cols_vec.push_back(static_cast<index_t>(c2));

                        used_shape_functions = true;
                    }
                }
            }
        }

        if (!used_shape_functions) {
            // Fallback: inverse-distance weighting on k nearest cell centroids
            bool has_zero_dist = false;
            int zero_val_idx = -1;

            struct NeighborInfo {
                std::size_t src_idx;
                double dist;
            };
            std::vector<NeighborInfo> neighbors;
            neighbors.reserve(end - begin);

            for (int vi = begin; vi < end; ++vi) {
                auto src_idx = static_cast<std::size_t>(values(vi).index);
                double dx = px - src_cx(src_idx);
                double dy = py - src_cy(src_idx);
                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist <= 0.0) {
                    has_zero_dist = true;
                    zero_val_idx = vi;
                }
                neighbors.push_back({src_idx, dist});
            }

            if (has_zero_dist) {
                auto src_idx = static_cast<std::size_t>(values(zero_val_idx).index);
                weights_vec.push_back(1.0);
                rows_vec.push_back(static_cast<index_t>(j));
                cols_vec.push_back(static_cast<index_t>(src_idx));
            } else {
                double sum_inv_dist = 0.0;
                for (auto& nb : neighbors) {
                    sum_inv_dist += 1.0 / nb.dist;
                }
                for (auto& nb : neighbors) {
                    double w = (1.0 / nb.dist) / sum_inv_dist;
                    weights_vec.push_back(w);
                    rows_vec.push_back(static_cast<index_t>(j));
                    cols_vec.push_back(static_cast<index_t>(nb.src_idx));
                }
            }
        }
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);
    auto h_area_a      = Kokkos::create_mirror_view(area_a);
    auto h_area_b      = Kokkos::create_mirror_view(area_b);

    for (std::size_t idx = 0; idx < nnz; ++idx) {
        h_factor_list(idx) = weights_vec[idx];
        h_factor_row(idx)  = rows_vec[idx];
        h_factor_col(idx)  = cols_vec[idx];
    }

    // Bilinear sets source areas to 0.0 (ESMF convention)
    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = 0.0;
        h_frac_a(i) = 1.0;
    }

    auto dst_areas = get_cell_areas(dst_mesh);
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = 1.0;
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}


// ─────────────────────────────────────────────────────────────────────────────
// generate_bicubic — 4×4 stencil least-squares bicubic interpolation
//
// Algorithm:
//   1. Use ArborX nearest(k=16) to find the 16 nearest source cell centroids.
//   2. Construct a bicubic Vandermonde matrix V[k,m] = x_k^i * y_k^j for
//      i+j <= 3 (10 terms) or full 4×4 (16 terms). We use 16-term full bicubic.
//   3. Solve V * coeffs = identity columns to get weights directly:
//      weights = V^{-1} evaluated at destination point.
//      Equivalently: w = V^{-T} * eval_vec where eval_vec = [x_d^i * y_d^j].
//   4. Simpler: solve the 16×16 system V^T * w = eval_vec for w.
//   5. area_a set to 0.0.
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate_bicubic(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config) {

    using HostSpace = Kokkos::HostSpace;
    using Point2    = ArborX::Point<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    // Stencil size for bicubic: 16 neighbors (4×4)
    const int k_stencil = static_cast<int>(std::min(static_cast<std::size_t>(16), n_src));

    // ── Compute source centroids and build ArborX BVH ──
    Kokkos::View<double*, HostSpace> src_cx, src_cy;
    compute_cell_centroids_xy(src_mesh, src_cx, src_cy);

    Kokkos::View<Point2*, HostSpace> src_points("src_points", n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_points(i) = Point2{static_cast<float>(src_cx(i)),
                               static_cast<float>(src_cy(i))};
    }

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(
        host_exec, ArborX::Experimental::attach_indices(src_points));

    // ── Compute destination centroids ──
    Kokkos::View<double*, HostSpace> dst_cx, dst_cy;
    compute_cell_centroids_xy(dst_mesh, dst_cx, dst_cy);

    // ── Build nearest(point, k_stencil) queries ──
    Kokkos::View<decltype(ArborX::nearest(Point2{}, 1))*, HostSpace>
        queries("queries", n_dst);
    for (std::size_t j = 0; j < n_dst; ++j) {
        queries(j) = ArborX::nearest(
            Point2{static_cast<float>(dst_cx(j)),
                   static_cast<float>(dst_cy(j))},
            k_stencil);
    }

    // ── Execute query ──
    Kokkos::View<typename decltype(tree)::value_type*, HostSpace> values("values", 0);
    Kokkos::View<int*, HostSpace> offsets_view("offsets", 0);
    tree.query(host_exec, queries, values, offsets_view);

    // ── Build COO entries with bicubic polynomial weights ──
    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    // Bicubic polynomial: P(x,y) = sum_{i=0}^{3} sum_{j=0}^{3} a_{ij} x^i y^j
    // Total 16 basis functions. For k < 16 source cells, we fall back to
    // a lower-order polynomial fit.
    constexpr int n_basis_full = 16; // Full bicubic

    for (std::size_t j = 0; j < n_dst; ++j) {
        int begin = offsets_view(j);
        int end_q = offsets_view(j + 1);
        int n_neighbors = end_q - begin;

        if (n_neighbors == 0) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error(
                    "WeightGenerator::generate_bicubic: unmapped destination cell "
                    + std::to_string(j));
            }
            continue;
        }

        double xd = dst_cx(j);
        double yd = dst_cy(j);

        // Collect neighbor info
        std::vector<std::size_t> src_indices(n_neighbors);
        std::vector<double> xs(n_neighbors), ys(n_neighbors);

        // Compute local centroid for numerical conditioning
        double cx_mean = 0.0, cy_mean = 0.0;
        for (int vi = begin; vi < end_q; ++vi) {
            auto si = static_cast<std::size_t>(values(vi).index);
            cx_mean += src_cx(si);
            cy_mean += src_cy(si);
        }
        cx_mean /= n_neighbors;
        cy_mean /= n_neighbors;

        // Compute scale for conditioning
        double scale = 0.0;
        for (int vi = begin; vi < end_q; ++vi) {
            auto si = static_cast<std::size_t>(values(vi).index);
            double dx = src_cx(si) - cx_mean;
            double dy = src_cy(si) - cy_mean;
            scale = std::max(scale, std::max(std::abs(dx), std::abs(dy)));
        }
        if (scale < 1e-30) scale = 1.0;
        double inv_scale = 1.0 / scale;

        for (int vi = begin; vi < end_q; ++vi) {
            int local_i = vi - begin;
            auto si = static_cast<std::size_t>(values(vi).index);
            src_indices[local_i] = si;
            xs[local_i] = (src_cx(si) - cx_mean) * inv_scale;
            ys[local_i] = (src_cy(si) - cy_mean) * inv_scale;
        }

        double xd_local = (xd - cx_mean) * inv_scale;
        double yd_local = (yd - cy_mean) * inv_scale;

        // Determine polynomial order based on available neighbors
        int n_basis = std::min(n_basis_full, n_neighbors);

        // Build Vandermonde matrix V (n_neighbors × n_basis) row-major
        // Basis: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, x^3y, x^2y^2, xy^3, ...
        // For full bicubic (16 terms): x^i * y^j for i=0..3, j=0..3
        std::vector<double> V(n_neighbors * n_basis, 0.0);
        for (int row = 0; row < n_neighbors; ++row) {
            double xr = xs[row];
            double yr = ys[row];
            int col = 0;
            for (int pi = 0; pi <= 3 && col < n_basis; ++pi) {
                for (int pj = 0; pj <= 3 && col < n_basis; ++pj) {
                    double val = 1.0;
                    for (int ii = 0; ii < pi; ++ii) val *= xr;
                    for (int jj = 0; jj < pj; ++jj) val *= yr;
                    V[row * n_basis + col] = val;
                    ++col;
                }
            }
        }

        // Evaluation vector at destination point
        std::vector<double> eval_vec(n_basis, 0.0);
        {
            int col = 0;
            for (int pi = 0; pi <= 3 && col < n_basis; ++pi) {
                for (int pj = 0; pj <= 3 && col < n_basis; ++pj) {
                    double val = 1.0;
                    for (int ii = 0; ii < pi; ++ii) val *= xd_local;
                    for (int jj = 0; jj < pj; ++jj) val *= yd_local;
                    eval_vec[col] = val;
                    ++col;
                }
            }
        }

        // Compute weights: w = A * (A^T A)^{-1} * eval_vec
        // where A = V (the Vandermonde matrix)
        // This gives weights such that sum(w_k * f_k) = P(xd, yd) for polynomial P
        // fitted through the source values.

        // First solve (V^T V) * c = V^T * eval_impossible...
        // Actually: the weight for source k is the dot product of the k-th row of
        // V * (V^T V)^{-1} with eval_vec.
        // Equivalently: solve (V^T V) alpha = eval_vec, then w_k = V[k,:] . alpha

        std::vector<double> alpha(n_basis);
        bool solved = false;

        if (n_neighbors == n_basis) {
            // Square system: solve V^T * w = eval directly
            // V is n×n, solve V * c = delta gives coefficients; then w_k = eval at k
            // Actually: w = (V^{-T}) * eval_vec, i.e., V^T * w = eval_vec
            std::vector<double> Vt(n_basis * n_basis);
            for (int r = 0; r < n_basis; ++r) {
                for (int c = 0; c < n_basis; ++c) {
                    Vt[r * n_basis + c] = V[c * n_basis + r];
                }
            }
            std::vector<double> rhs = eval_vec;
            solved = dense_solve(Vt, rhs, n_basis);
            if (solved) {
                // rhs now contains the weights directly
                for (int k = 0; k < n_neighbors; ++k) {
                    weights_vec.push_back(rhs[k]);
                    rows_vec.push_back(static_cast<index_t>(j));
                    cols_vec.push_back(static_cast<index_t>(src_indices[k]));
                }
            }
        }

        if (!solved) {
            // Overdetermined: use least-squares approach
            // Solve (V^T V) alpha = eval_vec
            // Then w_k = (V * alpha)_k ... no, that's wrong.
            //
            // Correct: weights w satisfy: for any polynomial values f_k,
            // sum(w_k * f_k) = eval P(xd,yd) where P is the LS fit.
            // weights = V * (V^T V)^{-1} * eval_vec
            //
            // Step 1: solve (V^T V) alpha = eval_vec
            std::vector<double> AtA(n_basis * n_basis, 0.0);
            std::vector<double> rhs = eval_vec;

            for (int i = 0; i < n_basis; ++i) {
                for (int jj = 0; jj < n_basis; ++jj) {
                    double sum = 0.0;
                    for (int k = 0; k < n_neighbors; ++k) {
                        sum += V[k * n_basis + i] * V[k * n_basis + jj];
                    }
                    AtA[i * n_basis + jj] = sum;
                }
            }

            solved = dense_solve(AtA, rhs, n_basis);
            if (solved) {
                // Step 2: w_k = V[k,:] . alpha
                for (int k = 0; k < n_neighbors; ++k) {
                    double w = 0.0;
                    for (int m = 0; m < n_basis; ++m) {
                        w += V[k * n_basis + m] * rhs[m];
                    }
                    weights_vec.push_back(w);
                    rows_vec.push_back(static_cast<index_t>(j));
                    cols_vec.push_back(static_cast<index_t>(src_indices[k]));
                }
            } else {
                // Final fallback: IDW
                double sum_inv = 0.0;
                bool has_zero = false;
                int zero_k = -1;
                for (int k = 0; k < n_neighbors; ++k) {
                    double dx = xd - src_cx(src_indices[k]);
                    double dy = yd - src_cy(src_indices[k]);
                    double d = std::sqrt(dx * dx + dy * dy);
                    if (d <= 0.0) { has_zero = true; zero_k = k; break; }
                    sum_inv += 1.0 / d;
                }
                if (has_zero) {
                    weights_vec.push_back(1.0);
                    rows_vec.push_back(static_cast<index_t>(j));
                    cols_vec.push_back(static_cast<index_t>(src_indices[zero_k]));
                } else {
                    for (int k = 0; k < n_neighbors; ++k) {
                        double dx = xd - src_cx(src_indices[k]);
                        double dy = yd - src_cy(src_indices[k]);
                        double d = std::sqrt(dx * dx + dy * dy);
                        double w = (1.0 / d) / sum_inv;
                        weights_vec.push_back(w);
                        rows_vec.push_back(static_cast<index_t>(j));
                        cols_vec.push_back(static_cast<index_t>(src_indices[k]));
                    }
                }
            }
        }
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);
    auto h_area_a      = Kokkos::create_mirror_view(area_a);
    auto h_area_b      = Kokkos::create_mirror_view(area_b);

    for (std::size_t idx = 0; idx < nnz; ++idx) {
        h_factor_list(idx) = weights_vec[idx];
        h_factor_row(idx)  = rows_vec[idx];
        h_factor_col(idx)  = cols_vec[idx];
    }

    // Bicubic sets source areas to 0.0 (ESMF convention)
    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = 0.0;
        h_frac_a(i) = 1.0;
    }

    auto dst_areas = get_cell_areas(dst_mesh);
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = 1.0;
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}


// ─────────────────────────────────────────────────────────────────────────────
// generate_patch — Least-squares polynomial patch recovery (ESMF REGRID_METHOD_PATCH)
//
// Algorithm:
//   1. Use ArborX nearest(k=16) to find the 16 nearest source cell centroids.
//   2. Fit a 2nd-degree polynomial P(x,y) = a + bx + cy + dxy + ex² + fy²
//      via least-squares over the k source cells.
//   3. Weights are derived from the LS solution evaluated at the destination:
//      Build design matrix A[k, m], evaluation vector e at (xd, yd).
//      Weights = A * (A^T A)^{-1} * e
//   4. area_a set to 0.0.
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate_patch(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config) {

    using HostSpace = Kokkos::HostSpace;
    using Point2    = ArborX::Point<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    // Stencil size for patch: 16 neighbors (overdetermined for 6-term quadratic)
    const int k_stencil = static_cast<int>(std::min(static_cast<std::size_t>(16), n_src));

    // Number of polynomial basis functions: 1, x, y, xy, x², y² = 6
    constexpr int n_basis = 6;

    // ── Compute source centroids and build ArborX BVH ──
    Kokkos::View<double*, HostSpace> src_cx, src_cy;
    compute_cell_centroids_xy(src_mesh, src_cx, src_cy);

    Kokkos::View<Point2*, HostSpace> src_points("src_points", n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_points(i) = Point2{static_cast<float>(src_cx(i)),
                               static_cast<float>(src_cy(i))};
    }

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(
        host_exec, ArborX::Experimental::attach_indices(src_points));

    // ── Compute destination centroids ──
    Kokkos::View<double*, HostSpace> dst_cx, dst_cy;
    compute_cell_centroids_xy(dst_mesh, dst_cx, dst_cy);

    // ── Build nearest(point, k_stencil) queries ──
    Kokkos::View<decltype(ArborX::nearest(Point2{}, 1))*, HostSpace>
        queries("queries", n_dst);
    for (std::size_t j = 0; j < n_dst; ++j) {
        queries(j) = ArborX::nearest(
            Point2{static_cast<float>(dst_cx(j)),
                   static_cast<float>(dst_cy(j))},
            k_stencil);
    }

    // ── Execute query ──
    Kokkos::View<typename decltype(tree)::value_type*, HostSpace> values("values", 0);
    Kokkos::View<int*, HostSpace> offsets_view("offsets", 0);
    tree.query(host_exec, queries, values, offsets_view);

    // ── Build COO entries with patch polynomial weights ──
    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    for (std::size_t j = 0; j < n_dst; ++j) {
        int begin = offsets_view(j);
        int end_q = offsets_view(j + 1);
        int n_neighbors = end_q - begin;

        if (n_neighbors == 0) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error(
                    "WeightGenerator::generate_patch: unmapped destination cell "
                    + std::to_string(j));
            }
            continue;
        }

        double xd = dst_cx(j);
        double yd = dst_cy(j);

        // Collect neighbor info
        std::vector<std::size_t> src_indices(n_neighbors);

        // Compute local centroid for numerical conditioning
        double cx_mean = 0.0, cy_mean = 0.0;
        for (int vi = begin; vi < end_q; ++vi) {
            auto si = static_cast<std::size_t>(values(vi).index);
            cx_mean += src_cx(si);
            cy_mean += src_cy(si);
        }
        cx_mean /= n_neighbors;
        cy_mean /= n_neighbors;

        // Compute scale for conditioning
        double scale = 0.0;
        for (int vi = begin; vi < end_q; ++vi) {
            auto si = static_cast<std::size_t>(values(vi).index);
            double dx = src_cx(si) - cx_mean;
            double dy = src_cy(si) - cy_mean;
            scale = std::max(scale, std::max(std::abs(dx), std::abs(dy)));
        }
        if (scale < 1e-30) scale = 1.0;
        double inv_scale = 1.0 / scale;

        std::vector<double> xs(n_neighbors), ys(n_neighbors);
        for (int vi = begin; vi < end_q; ++vi) {
            int local_i = vi - begin;
            auto si = static_cast<std::size_t>(values(vi).index);
            src_indices[local_i] = si;
            xs[local_i] = (src_cx(si) - cx_mean) * inv_scale;
            ys[local_i] = (src_cy(si) - cy_mean) * inv_scale;
        }

        double xd_local = (xd - cx_mean) * inv_scale;
        double yd_local = (yd - cy_mean) * inv_scale;

        // Build design matrix A [n_neighbors × n_basis]
        // Basis: [1, x, y, xy, x², y²]
        int n_basis_actual = std::min(n_basis, n_neighbors);
        std::vector<double> A(n_neighbors * n_basis_actual, 0.0);
        for (int row = 0; row < n_neighbors; ++row) {
            double xr = xs[row];
            double yr = ys[row];
            A[row * n_basis_actual + 0] = 1.0;
            if (n_basis_actual > 1) A[row * n_basis_actual + 1] = xr;
            if (n_basis_actual > 2) A[row * n_basis_actual + 2] = yr;
            if (n_basis_actual > 3) A[row * n_basis_actual + 3] = xr * yr;
            if (n_basis_actual > 4) A[row * n_basis_actual + 4] = xr * xr;
            if (n_basis_actual > 5) A[row * n_basis_actual + 5] = yr * yr;
        }

        // Evaluation vector at destination point
        std::vector<double> eval_vec(n_basis_actual, 0.0);
        eval_vec[0] = 1.0;
        if (n_basis_actual > 1) eval_vec[1] = xd_local;
        if (n_basis_actual > 2) eval_vec[2] = yd_local;
        if (n_basis_actual > 3) eval_vec[3] = xd_local * yd_local;
        if (n_basis_actual > 4) eval_vec[4] = xd_local * xd_local;
        if (n_basis_actual > 5) eval_vec[5] = yd_local * yd_local;

        // Compute weights = A * (A^T A)^{-1} * eval_vec
        // Step 1: Form A^T A (n_basis × n_basis) and solve (A^T A) alpha = eval_vec
        std::vector<double> AtA(n_basis_actual * n_basis_actual, 0.0);
        for (int i = 0; i < n_basis_actual; ++i) {
            for (int jj = 0; jj < n_basis_actual; ++jj) {
                double sum = 0.0;
                for (int k = 0; k < n_neighbors; ++k) {
                    sum += A[k * n_basis_actual + i] * A[k * n_basis_actual + jj];
                }
                AtA[i * n_basis_actual + jj] = sum;
            }
        }

        std::vector<double> alpha = eval_vec;
        bool solved = dense_solve(AtA, alpha, n_basis_actual);

        if (solved) {
            // Step 2: w_k = A[k,:] . alpha
            for (int k = 0; k < n_neighbors; ++k) {
                double w = 0.0;
                for (int m = 0; m < n_basis_actual; ++m) {
                    w += A[k * n_basis_actual + m] * alpha[m];
                }
                weights_vec.push_back(w);
                rows_vec.push_back(static_cast<index_t>(j));
                cols_vec.push_back(static_cast<index_t>(src_indices[k]));
            }
        } else {
            // Fallback: IDW
            double sum_inv = 0.0;
            bool has_zero = false;
            int zero_k = -1;
            for (int k = 0; k < n_neighbors; ++k) {
                double dx = xd - src_cx(src_indices[k]);
                double dy = yd - src_cy(src_indices[k]);
                double d = std::sqrt(dx * dx + dy * dy);
                if (d <= 0.0) { has_zero = true; zero_k = k; break; }
                sum_inv += 1.0 / d;
            }
            if (has_zero) {
                weights_vec.push_back(1.0);
                rows_vec.push_back(static_cast<index_t>(j));
                cols_vec.push_back(static_cast<index_t>(src_indices[zero_k]));
            } else {
                for (int k = 0; k < n_neighbors; ++k) {
                    double dx = xd - src_cx(src_indices[k]);
                    double dy = yd - src_cy(src_indices[k]);
                    double d = std::sqrt(dx * dx + dy * dy);
                    double w = (1.0 / d) / sum_inv;
                    weights_vec.push_back(w);
                    rows_vec.push_back(static_cast<index_t>(j));
                    cols_vec.push_back(static_cast<index_t>(src_indices[k]));
                }
            }
        }
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);
    auto h_area_a      = Kokkos::create_mirror_view(area_a);
    auto h_area_b      = Kokkos::create_mirror_view(area_b);

    for (std::size_t idx = 0; idx < nnz; ++idx) {
        h_factor_list(idx) = weights_vec[idx];
        h_factor_row(idx)  = rows_vec[idx];
        h_factor_col(idx)  = cols_vec[idx];
    }

    // Patch sets source areas to 0.0 (ESMF convention)
    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = 0.0;
        h_frac_a(i) = 1.0;
    }

    auto dst_areas = get_cell_areas(dst_mesh);
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = 1.0;
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}


// ─────────────────────────────────────────────────────────────────────────────
// generate_conservative — ArborX AABB intersection + Sutherland-Hodgman overlap
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
InterpolationMatrix<MemorySpace>
WeightGenerator::generate_conservative(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config) {

    using HostSpace = Kokkos::HostSpace;
    using Box2      = ArborX::Box<2>;

    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    // Determine whether to use the spherical clipping path.
    const bool use_spherical = (config.line_type == LineType::GreatCircle);

    // ── Build ArborX BVH from source cell AABBs ──
    auto src_boxes = compute_cell_aabbs(src_mesh);

    Kokkos::DefaultHostExecutionSpace host_exec;
    ArborX::BoundingVolumeHierarchy tree(
        host_exec, ArborX::Experimental::attach_indices(src_boxes));

    // ── Build intersection queries from destination cell AABBs ──
    auto dst_boxes = compute_cell_aabbs(dst_mesh);

    Kokkos::View<decltype(ArborX::intersects(Box2{}))*, HostSpace>
        queries("queries", n_dst);
    for (std::size_t j = 0; j < n_dst; ++j) {
        queries(j) = ArborX::intersects(dst_boxes(j));
    }

    // ── Execute query ──
    Kokkos::View<typename decltype(tree)::value_type*, HostSpace> values("values", 0);
    Kokkos::View<int*, HostSpace> offsets_view("offsets", 0);
    tree.query(host_exec, queries, values, offsets_view);

    // ── Get cell areas (spherical or flat) ──
    std::vector<double> src_areas;
    std::vector<double> dst_areas;
    if (use_spherical) {
        src_areas = get_cell_areas_spherical(src_mesh);
        dst_areas = get_cell_areas_spherical(dst_mesh);
    } else {
        src_areas = get_cell_areas(src_mesh);
        dst_areas = get_cell_areas(dst_mesh);
    }

    // ── Accumulators for frac_a and frac_b ──
    std::vector<double> frac_a_acc(n_src, 0.0);
    std::vector<double> frac_b_acc(n_dst, 0.0);

    // ── Compute exact overlap for each candidate pair ──
    std::vector<double>  weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    for (std::size_t j = 0; j < n_dst; ++j) {
        int begin = offsets_view(j);
        int end   = offsets_view(j + 1);

        double area_dst = dst_areas[j];
        if (area_dst <= 0.0) {
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error(
                    "WeightGenerator::generate_conservative: unmapped destination cell "
                    + std::to_string(j));
            }
            continue;
        }

        bool has_entry = false;

        if (use_spherical) {
            // ── Spherical path: great-circle Sutherland-Hodgman clipping ──
            auto dst_poly_s = extract_cell_polygon_spherical(dst_mesh, j);

            for (int vi = begin; vi < end; ++vi) {
                auto src_i = static_cast<std::size_t>(values(vi).index);
                double area_src = src_areas[src_i];
                if (area_src <= 0.0) continue;

                auto src_poly_s = extract_cell_polygon_spherical(src_mesh, src_i);
                double overlap_area = axis::detail::spherical::spherical_polygon_overlap_area(
                    src_poly_s, dst_poly_s);

                if (overlap_area <= 0.0) continue;

                double w_ij = overlap_area / area_dst;
                w_ij = std::max(w_ij, 0.0);

                weights_vec.push_back(w_ij);
                rows_vec.push_back(static_cast<index_t>(j));
                cols_vec.push_back(static_cast<index_t>(src_i));
                has_entry = true;

                frac_a_acc[src_i] += overlap_area / area_src;
                frac_b_acc[j] += overlap_area / area_dst;
            }
        } else {
            // ── Cartesian path: flat Sutherland-Hodgman clipping ──
            auto dst_poly = extract_cell_polygon(dst_mesh, j);

            for (int vi = begin; vi < end; ++vi) {
                auto src_i = static_cast<std::size_t>(values(vi).index);
                double area_src = src_areas[src_i];
                if (area_src <= 0.0) continue;

                auto src_poly = extract_cell_polygon(src_mesh, src_i);
                double overlap_area = compute_polygon_overlap_area(src_poly, dst_poly);

                if (overlap_area <= 0.0) continue;

                double w_ij = overlap_area / area_dst;
                w_ij = std::max(w_ij, 0.0);

                weights_vec.push_back(w_ij);
                rows_vec.push_back(static_cast<index_t>(j));
                cols_vec.push_back(static_cast<index_t>(src_i));
                has_entry = true;

                frac_a_acc[src_i] += overlap_area / area_src;
                frac_b_acc[j] += overlap_area / area_dst;
            }
        }

        if (!has_entry && config.unmapped == UnmappedAction::Error) {
            throw std::runtime_error(
                "WeightGenerator::generate_conservative: unmapped destination cell "
                + std::to_string(j));
        }
    }

    // ── Apply FracArea normalization if configured ──
    if (config.norm_type == NormType::FracArea) {
        for (std::size_t k = 0; k < weights_vec.size(); ++k) {
            auto j = static_cast<std::size_t>(rows_vec[k]);
            if (frac_b_acc[j] > 0.0) {
                weights_vec[k] /= frac_b_acc[j];
            }
        }
    }

    // Clamp fractions to [0, 1]
    for (std::size_t i = 0; i < n_src; ++i) {
        frac_a_acc[i] = std::min(frac_a_acc[i], 1.0);
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        frac_b_acc[j] = std::min(frac_b_acc[j], 1.0);
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double*, MemorySpace>  factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t*, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double*, MemorySpace>  frac_a("frac_a", n_src);
    Kokkos::View<double*, MemorySpace>  frac_b("frac_b", n_dst);
    Kokkos::View<double*, MemorySpace>  area_a("area_a", n_src);
    Kokkos::View<double*, MemorySpace>  area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row  = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col  = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a      = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b      = Kokkos::create_mirror_view(frac_b);
    auto h_area_a      = Kokkos::create_mirror_view(area_a);
    auto h_area_b      = Kokkos::create_mirror_view(area_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k)  = rows_vec[k];
        h_factor_col(k)  = cols_vec[k];
    }

    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = src_areas[i];
        h_frac_a(i) = frac_a_acc[i];
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = frac_b_acc[j];
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(
        std::move(factor_list), std::move(factor_row), std::move(factor_col),
        std::move(frac_a), std::move(frac_b),
        std::move(area_a), std::move(area_b),
        n_src, n_dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// generate — distributed mode (with HaloPattern)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
std::pair<InterpolationMatrix<MemorySpace>, HaloPattern>
WeightGenerator::generate(
    const topology::UnstructuredMesh<MemorySpace>& src_mesh,
    const topology::UnstructuredMesh<MemorySpace>& dst_mesh,
    const RegridConfig& config,
    Kokkos::View<const index_t*, MemorySpace> src_global_ids,
    Kokkos::View<const index_t*, MemorySpace> dst_global_ids,
    const std::vector<int>& owner_of_src) {

    // ── Step 1: Produce local InterpolationMatrix via single-rank generate ──
    auto local_matrix = generate(src_mesh, dst_mesh, config);

    const std::size_t n_local_src = src_mesh.n_cells();
    const std::size_t nnz = local_matrix.nnz();

    // ── Step 2: Determine local rank by checking ownership ──
    auto h_src_global_ids = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, src_global_ids);

    std::unordered_map<index_t, std::size_t> global_to_local;
    global_to_local.reserve(n_local_src);
    for (std::size_t i = 0; i < n_local_src; ++i) {
        global_to_local[h_src_global_ids(i)] = i;
    }

    int local_rank = -1;
    if (n_local_src > 0 && static_cast<std::size_t>(h_src_global_ids(0)) < owner_of_src.size()) {
        local_rank = owner_of_src[static_cast<std::size_t>(h_src_global_ids(0))];
    }

    // ── Step 3: Scan factor_col to find off-rank dependencies ──
    auto factor_col_view = local_matrix.factor_col_view();
    auto h_factor_col = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, factor_col_view);

    struct RemoteEntry {
        index_t global_id;
        int     owner_rank;
    };

    std::unordered_map<index_t, RemoteEntry> off_rank_map;
    std::vector<index_t> off_rank_local_indices;

    for (std::size_t k = 0; k < nnz; ++k) {
        auto local_col = static_cast<std::size_t>(h_factor_col(k));
        if (local_col >= n_local_src) continue;

        index_t global_id = h_src_global_ids(local_col);
        auto gid_as_size = static_cast<std::size_t>(global_id);

        if (gid_as_size >= owner_of_src.size()) continue;

        int owner = owner_of_src[gid_as_size];
        if (owner != local_rank) {
            if (off_rank_map.find(static_cast<index_t>(local_col)) == off_rank_map.end()) {
                off_rank_map[static_cast<index_t>(local_col)] = {global_id, owner};
                off_rank_local_indices.push_back(static_cast<index_t>(local_col));
            }
        }
    }

    // ── Step 4: Group off-rank sources by owning rank (CSR form) ──
    std::sort(off_rank_local_indices.begin(), off_rank_local_indices.end(),
        [&](index_t a, index_t b) {
            return off_rank_map[a].owner_rank < off_rank_map[b].owner_rank;
        });

    HaloPattern pattern;

    if (!off_rank_local_indices.empty()) {
        int prev_rank = -1;
        for (std::size_t i = 0; i < off_rank_local_indices.size(); ++i) {
            int rank = off_rank_map[off_rank_local_indices[i]].owner_rank;
            if (rank != prev_rank) {
                pattern.source_ranks.push_back(rank);
                pattern.rank_offsets.push_back(static_cast<index_t>(i));
                prev_rank = rank;
            }
            pattern.needed_global_src_ids.push_back(
                off_rank_map[off_rank_local_indices[i]].global_id);
            pattern.gather_slot.push_back(static_cast<index_t>(i));
        }
        pattern.rank_offsets.push_back(
            static_cast<index_t>(off_rank_local_indices.size()));
    } else {
        pattern.rank_offsets.push_back(0);
    }

    // ── Step 5: Remap matrix column indices ──
    std::unordered_map<index_t, index_t> col_remap;
    for (std::size_t i = 0; i < off_rank_local_indices.size(); ++i) {
        index_t local_col = off_rank_local_indices[i];
        col_remap[local_col] = static_cast<index_t>(n_local_src + i);
    }

    Kokkos::View<index_t*, MemorySpace> new_factor_col("factor_col_remapped", nnz);
    auto h_new_factor_col = Kokkos::create_mirror_view(new_factor_col);

    for (std::size_t k = 0; k < nnz; ++k) {
        index_t col = h_factor_col(k);
        auto it = col_remap.find(col);
        if (it != col_remap.end()) {
            h_new_factor_col(k) = it->second;
        } else {
            h_new_factor_col(k) = col;
        }
    }
    Kokkos::deep_copy(new_factor_col, h_new_factor_col);

    // ── Step 6: Build remapped InterpolationMatrix ──
    auto factor_list_orig = local_matrix.factor_list_view();
    auto factor_row_orig  = local_matrix.factor_row_view();
    auto frac_a_orig      = local_matrix.frac_a_view();
    auto frac_b_orig      = local_matrix.frac_b_view();
    auto area_a_orig      = local_matrix.area_a_view();
    auto area_b_orig      = local_matrix.area_b_view();

    Kokkos::View<double*, MemorySpace>  new_factor_list("factor_list", nnz);
    Kokkos::View<index_t*, MemorySpace> new_factor_row("factor_row", nnz);
    Kokkos::View<double*, MemorySpace>  new_frac_a("frac_a", frac_a_orig.extent(0));
    Kokkos::View<double*, MemorySpace>  new_frac_b("frac_b", frac_b_orig.extent(0));
    Kokkos::View<double*, MemorySpace>  new_area_a("area_a", area_a_orig.extent(0));
    Kokkos::View<double*, MemorySpace>  new_area_b("area_b", area_b_orig.extent(0));

    Kokkos::deep_copy(new_factor_list, factor_list_orig);
    Kokkos::deep_copy(new_factor_row, factor_row_orig);
    Kokkos::deep_copy(new_frac_a, frac_a_orig);
    Kokkos::deep_copy(new_frac_b, frac_b_orig);
    Kokkos::deep_copy(new_area_a, area_a_orig);
    Kokkos::deep_copy(new_area_b, area_b_orig);

    std::size_t n_src_extended = n_local_src + off_rank_local_indices.size();

    InterpolationMatrix<MemorySpace> remapped_matrix(
        std::move(new_factor_list),
        std::move(new_factor_row),
        std::move(new_factor_col),
        std::move(new_frac_a),
        std::move(new_frac_b),
        std::move(new_area_a),
        std::move(new_area_b),
        n_src_extended,
        local_matrix.n_dst());

    return {std::move(remapped_matrix), std::move(pattern)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate_bilinear<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate_nearest<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate_bicubic<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate_patch<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template InterpolationMatrix<Kokkos::HostSpace>
WeightGenerator::generate_conservative<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&);

template std::pair<InterpolationMatrix<Kokkos::HostSpace>, HaloPattern>
WeightGenerator::generate<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const topology::UnstructuredMesh<Kokkos::HostSpace>&,
    const RegridConfig&,
    Kokkos::View<const index_t*, Kokkos::HostSpace>,
    Kokkos::View<const index_t*, Kokkos::HostSpace>,
    const std::vector<int>&);

} // namespace axis::solver
