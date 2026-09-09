// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/structured_grid.cpp
/// @brief StructuredGrid<MemorySpace> implementation.
///
/// Implements the constructor, accessors, set_corners, corner synthesis, and
/// the to_unstructured() conversion via Kokkos parallel kernel. Each logical
/// cell (i,j) maps to one quadrilateral element with 4 corner nodes.

#include <Kokkos_Core.hpp>
#include <axis/detail/memory_traits.hpp>
#include <axis/topology/structured_grid.hpp>
#include <stdexcept>
#include <string>

namespace axis::topology {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
StructuredGrid<MemorySpace>::StructuredGrid(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> center_lon,
                                            Kokkos::View<double *, MemorySpace> center_lat, CoordinateSystem coord_sys)
    : ni_(ni), nj_(nj), center_lon_(std::move(center_lon)), center_lat_(std::move(center_lat)), coord_sys_(coord_sys) {
    if (ni_ == 0 || nj_ == 0) {
        throw std::invalid_argument("StructuredGrid: ni and nj must be positive");
    }
    if (center_lon_.extent(0) != ni_ * nj_) {
        throw std::invalid_argument("StructuredGrid: center_lon extent (" + std::to_string(center_lon_.extent(0)) + ") does not match ni*nj (" +
                                    std::to_string(ni_ * nj_) + ")");
    }
    if (center_lat_.extent(0) != ni_ * nj_) {
        throw std::invalid_argument("StructuredGrid: center_lat extent (" + std::to_string(center_lat_.extent(0)) + ") does not match ni*nj (" +
                                    std::to_string(ni_ * nj_) + ")");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate accessors
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::center_lon() const noexcept {
    return field_view<const double, 1>{center_lon_.data(), center_lon_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::center_lat() const noexcept {
    return field_view<const double, 1>{center_lat_.data(), center_lat_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::corner_lon() const noexcept {
    return field_view<const double, 1>{corner_lon_.data(), corner_lon_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::corner_lat() const noexcept {
    return field_view<const double, 1>{corner_lat_.data(), corner_lat_.extent(0)};
}

// ─────────────────────────────────────────────────────────────────────────────
// set_corners
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void StructuredGrid<MemorySpace>::set_corners(Kokkos::View<double *, MemorySpace> corner_lon, Kokkos::View<double *, MemorySpace> corner_lat) {
    const std::size_t expected = (ni_ + 1) * (nj_ + 1);
    if (corner_lon.extent(0) != expected) {
        throw std::invalid_argument("StructuredGrid::set_corners: corner_lon extent (" + std::to_string(corner_lon.extent(0)) +
                                    ") does not match (ni+1)*(nj+1) (" + std::to_string(expected) + ")");
    }
    if (corner_lat.extent(0) != expected) {
        throw std::invalid_argument("StructuredGrid::set_corners: corner_lat extent (" + std::to_string(corner_lat.extent(0)) +
                                    ") does not match (ni+1)*(nj+1) (" + std::to_string(expected) + ")");
    }
    corner_lon_ = std::move(corner_lon);
    corner_lat_ = std::move(corner_lat);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared corner-synthesis kernel (single source of truth)
//
// Both StructuredGrid::synthesize_corners() (whole grid) and the free
// synthesize_band_corners() (a latitude band, halo-aware) dispatch through the
// routines below so the geometry — the 2×2 midpoint average of surrounding
// centers, the periodic-longitude wrap, and the one-sided boundary convention —
// lives in exactly one place. A band computed here is provably identical to the
// corresponding corner rows of the whole-grid result.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Detect longitude periodicity from a full global center array (column-major,
/// size ni*nj): true when ni * (spacing of the first row) spans ~360 degrees.
/// Identical logic to the original in-line detection in synthesize_corners().
template <class MemorySpace>
bool detect_periodic_lon(std::size_t ni, const Kokkos::View<double *, MemorySpace> &center_lon) {
    if (ni <= 1) {
        return false;
    }
    auto center_lon_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, center_lon);
    double dlon = center_lon_host(1) - center_lon_host(0);
    double full_span = static_cast<double>(ni) * std::abs(dlon);
    return full_span > 350.0 && full_span < 370.0;
}

/// Synthesize `nrows` rows of Cell_Corners for the global corner rows
/// [cj_lo, cj_lo + nrows) over a full grid of ni × nj centers. The output views
/// are (ni+1) * nrows, indexed so corner (ci, cj) lives at
/// ci + (cj - cj_lo) * (ni + 1). This is the exact kernel the whole-grid path
/// uses (cj_lo = 0, nrows = nj+1 reproduces it verbatim).
template <class MemorySpace>
void synthesize_corner_rows(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> center_lon,
                            Kokkos::View<double *, MemorySpace> center_lat, std::size_t cj_lo, std::size_t nrows, bool is_periodic,
                            Kokkos::View<double *, MemorySpace> corner_lon, Kokkos::View<double *, MemorySpace> corner_lat) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;

    const std::size_t nip1 = ni + 1;

    auto clon = corner_lon;
    auto clat = corner_lat;

    // Dispatch one team per requested corner row. Threads in a team handle the
    // elements within the row.
    using TeamPolicy = Kokkos::TeamPolicy<exec_space>;
    using MemberType = typename TeamPolicy::member_type;

    TeamPolicy policy(static_cast<int>(nrows), Kokkos::AUTO);
    Kokkos::parallel_for(
        "synthesize_corners_team", policy, KOKKOS_LAMBDA(const MemberType &team) {
            const std::size_t cj = cj_lo + static_cast<std::size_t>(team.league_rank());

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nip1), [&](const std::size_t ci) {
                const std::size_t idx = ci + (cj - cj_lo) * nip1;
                double sum_lon = 0.0;
                double sum_lat = 0.0;
                int count = 0;

                for (int dj = -1; dj <= 0; ++dj) {
                    for (int di = -1; di <= 0; ++di) {
                        auto cell_i = static_cast<long long>(ci) + di;
                        const auto cell_j = static_cast<long long>(cj) + dj;

                        double cell_lon = 0.0;
                        bool valid_i = false;

                        if (is_periodic) {
                            if (cell_i < 0) {
                                cell_i = ni - 1;
                                cell_lon = -360.0;
                            } else if (cell_i >= static_cast<long long>(ni)) {
                                cell_i = 0;
                                cell_lon = 360.0;
                            }
                            valid_i = true;
                        } else {
                            valid_i = (cell_i >= 0 && cell_i < static_cast<long long>(ni));
                        }

                        if (valid_i && cell_j >= 0 && cell_j < static_cast<long long>(nj)) {
                            const std::size_t cell_idx = static_cast<std::size_t>(cell_i) + static_cast<std::size_t>(cell_j) * ni;
                            sum_lon += (center_lon(cell_idx) + cell_lon);
                            sum_lat += center_lat(cell_idx);
                            ++count;
                        }
                    }
                }

                clon(idx) = sum_lon / static_cast<double>(count);
                clat(idx) = sum_lat / static_cast<double>(count);
            });
        });

    Kokkos::fence("synthesize_corners_fence");
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// synthesize_corners — build corners from midpoints of adjacent centers
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void StructuredGrid<MemorySpace>::synthesize_corners() const {
    const std::size_t nip1 = ni_ + 1;
    const std::size_t njp1 = nj_ + 1;
    const std::size_t n_corners = nip1 * njp1;

    // Allocate corner arrays (mutable cast — this is a lazy-init pattern).
    auto &self = const_cast<StructuredGrid<MemorySpace> &>(*this);
    self.corner_lon_ = Kokkos::View<double *, MemorySpace>("structured_grid_corner_lon", n_corners);
    self.corner_lat_ = Kokkos::View<double *, MemorySpace>("structured_grid_corner_lat", n_corners);

    const bool is_periodic = detect_periodic_lon(ni_, center_lon_);

    // Whole grid: corner rows [0, nj_+1). Identical to the historical in-line
    // kernel (cj_lo = 0, nrows = nj_+1).
    synthesize_corner_rows(ni_, nj_, center_lon_, center_lat_, /*cj_lo=*/0, njp1, is_periodic, self.corner_lon_, self.corner_lat_);
}


// ─────────────────────────────────────────────────────────────────────────────
// to_unstructured — the core conversion (Requirement 18.1, 18.2, 18.3)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> StructuredGrid<MemorySpace>::to_unstructured() const {
    using exec_space = typename detail::exec_space_t<MemorySpace>;

    // If corners have not been set explicitly, synthesize from centers.
    if (corner_lon_.extent(0) == 0) {
        synthesize_corners();
    }

    const std::size_t ni = ni_;
    const std::size_t nj = nj_;
    const std::size_t nip1 = ni + 1;
    const std::size_t njp1 = nj + 1;
    const std::size_t n_cells = ni * nj;
    const std::size_t n_nodes = nip1 * njp1;

    // ── Allocate output arrays ───────────────────────────────────────────────

    // Node coordinates: [n_nodes, 2] (lon, lat)
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("unstructured_node_coords", n_nodes, std::size_t{2});

    // CSR offsets: [n_cells + 1]. For all-quad meshes: [0, 4, 8, 12, ...]
    Kokkos::View<index_t *, MemorySpace> cell_node_offsets("unstructured_cell_offsets", n_cells + 1);

    // CSR indices: [n_cells * 4] (4 nodes per quad cell)
    Kokkos::View<index_t *, MemorySpace> cell_node_indices("unstructured_cell_indices", n_cells * 4);

    auto clon = corner_lon_;
    auto clat = corner_lat_;

    // Optimize node coordinate filling using MDRangePolicy
    using MDRange2D = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>>;
    Kokkos::parallel_for(
        "to_unstructured_fill_nodes_md", MDRange2D({0, 0}, {static_cast<int>(nip1), static_cast<int>(njp1)}),
        KOKKOS_LAMBDA(const int ci, const int cj) {
            const int node_idx = ci + cj * nip1;
            node_coords(node_idx, 0) = clon(node_idx);
            node_coords(node_idx, 1) = clat(node_idx);
        });

    // Optimize connectivity filling using MDRangePolicy
    Kokkos::parallel_for(
        "to_unstructured_fill_connectivity_md", MDRange2D({0, 0}, {static_cast<int>(ni), static_cast<int>(nj)}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const int cell_idx = i + j * ni;
            cell_node_offsets(cell_idx) = static_cast<index_t>(cell_idx * 4);

            const std::size_t base = cell_idx * 4;
            cell_node_indices(base + 0) = static_cast<index_t>(i + j * nip1);              // bottom-left
            cell_node_indices(base + 1) = static_cast<index_t>((i + 1) + j * nip1);        // bottom-right
            cell_node_indices(base + 2) = static_cast<index_t>((i + 1) + (j + 1) * nip1);  // top-right
            cell_node_indices(base + 3) = static_cast<index_t>(i + (j + 1) * nip1);        // top-left
        });

    // ── Fill the sentinel offset at the end ──────────────────────────────────
    // CSR requires offsets[n_cells] = total number of connectivity entries.
    Kokkos::parallel_for(
        "to_unstructured_sentinel_offset", Kokkos::RangePolicy<exec_space>(0, 1),
        KOKKOS_LAMBDA(const int /*unused*/) { cell_node_offsets(n_cells) = static_cast<index_t>(n_cells * 4); });

    Kokkos::fence("to_unstructured_fence");

    return UnstructuredMesh<MemorySpace>(std::move(node_coords), std::move(cell_node_offsets), std::move(cell_node_indices), coord_sys_);
}

// ─────────────────────────────────────────────────────────────────────────────
// synthesize_band_corners — global-context (halo-aware) band corner synthesis
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void synthesize_band_corners(std::size_t ni, std::size_t nj_global, Kokkos::View<double *, MemorySpace> center_lon,
                             Kokkos::View<double *, MemorySpace> center_lat, std::size_t j0, std::size_t j1,
                             Kokkos::View<double *, MemorySpace> &corner_lon, Kokkos::View<double *, MemorySpace> &corner_lat) {
    const std::size_t nip1 = ni + 1;

    if (ni == 0) {
        throw std::invalid_argument("synthesize_band_corners: ni must be positive");
    }
    if (j1 < j0) {
        throw std::invalid_argument("synthesize_band_corners: j1 must be >= j0");
    }
    if (j1 > nj_global) {
        throw std::invalid_argument("synthesize_band_corners: j1 (" + std::to_string(j1) + ") exceeds nj_global (" + std::to_string(nj_global) + ")");
    }
    if (center_lon.extent(0) != ni * nj_global || center_lat.extent(0) != ni * nj_global) {
        throw std::invalid_argument("synthesize_band_corners: center array extent must equal ni * nj_global");
    }

    const std::size_t nrows = j1 - j0 + 1;  // corner rows [j0, j1]
    corner_lon = Kokkos::View<double *, MemorySpace>("band_corner_lon", nip1 * nrows);
    corner_lat = Kokkos::View<double *, MemorySpace>("band_corner_lat", nip1 * nrows);

    // Periodicity is a property of the FULL global longitude layout.
    const bool is_periodic = detect_periodic_lon(ni, center_lon);

    // Same kernel as the whole-grid path, restricted to global corner rows [j0, j1].
    synthesize_corner_rows(ni, nj_global, center_lon, center_lat, j0, nrows, is_periodic, corner_lon, corner_lat);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class StructuredGrid<Kokkos::HostSpace>;
template void synthesize_band_corners<Kokkos::HostSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HostSpace>,
                                                         Kokkos::View<double *, Kokkos::HostSpace>, std::size_t, std::size_t,
                                                         Kokkos::View<double *, Kokkos::HostSpace> &,
                                                         Kokkos::View<double *, Kokkos::HostSpace> &);

#ifdef KOKKOS_ENABLE_CUDA
template class StructuredGrid<Kokkos::CudaSpace>;
template void synthesize_band_corners<Kokkos::CudaSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::CudaSpace>,
                                                         Kokkos::View<double *, Kokkos::CudaSpace>, std::size_t, std::size_t,
                                                         Kokkos::View<double *, Kokkos::CudaSpace> &,
                                                         Kokkos::View<double *, Kokkos::CudaSpace> &);
#endif

#ifdef KOKKOS_ENABLE_HIP
template class StructuredGrid<Kokkos::HIPSpace>;
template void synthesize_band_corners<Kokkos::HIPSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HIPSpace>,
                                                        Kokkos::View<double *, Kokkos::HIPSpace>, std::size_t, std::size_t,
                                                        Kokkos::View<double *, Kokkos::HIPSpace> &,
                                                        Kokkos::View<double *, Kokkos::HIPSpace> &);
#endif

}  // namespace axis::topology
