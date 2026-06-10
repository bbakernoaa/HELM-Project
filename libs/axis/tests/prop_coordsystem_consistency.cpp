// ─── Property-Based Tests: Coordinate-System Consistency Enforcement ─────────
// Feature: helm-axis-microlibrary, Property 21: Coordinate-System Consistency
//
// For src/dst meshes with differing CoordinateSystem values, verify that
// WeightGenerator::generate throws std::invalid_argument.
//
// **Validates: Requirements 11.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/topology/enums.hpp>
#include <axis/types.hpp>

namespace {

/// Build a small mesh with a specific CoordinateSystem.
axis::topology::UnstructuredMesh<Kokkos::HostSpace>
build_mesh_with_coord_system(std::size_t ni, std::size_t nj,
                             axis::topology::CoordinateSystem coord_sys) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double*, Kokkos::HostSpace> center_lon("clon", n_centers);
    Kokkos::View<double*, Kokkos::HostSpace> center_lat("clat", n_centers);
    Kokkos::View<double*, Kokkos::HostSpace> corner_lon("crlon", n_corners);
    Kokkos::View<double*, Kokkos::HostSpace> corner_lat("crlat", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_lon(idx) = static_cast<double>(i) + 0.5;
            center_lat(idx) = static_cast<double>(j) + 0.5;
        }
    }

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_lon(idx) = static_cast<double>(i);
            corner_lat(idx) = static_cast<double>(j);
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(
        ni, nj, center_lon, center_lat, coord_sys);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 21a: Mismatched SphericalDeg vs SphericalRad throws ────────────

RC_GTEST_PROP(PropCoordsystemConsistency, DegVsRadThrows, ()) {
    const auto ni = *rc::gen::inRange<std::size_t>(2, 6);
    const auto nj = *rc::gen::inRange<std::size_t>(2, 6);

    auto src_mesh = build_mesh_with_coord_system(ni, nj,
        axis::topology::CoordinateSystem::SphericalDeg);
    auto dst_mesh = build_mesh_with_coord_system(ni, nj,
        axis::topology::CoordinateSystem::SphericalRad);

    // Try all methods — all should throw
    auto method = *rc::gen::element(
        axis::solver::InterpolationMethod::Bilinear,
        axis::solver::InterpolationMethod::NearestNeighbor,
        axis::solver::InterpolationMethod::Conservative1stOrder);

    axis::solver::RegridConfig config;
    config.method = method;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    bool threw_invalid = false;
    try {
        axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(
            src_mesh, dst_mesh, config);
    } catch (const std::invalid_argument& e) {
        threw_invalid = true;
        // Verify it mentions CoordinateSystem mismatch
        std::string msg = e.what();
        RC_ASSERT(msg.find("CoordinateSystem") != std::string::npos ||
                  msg.find("coordinate") != std::string::npos ||
                  msg.find("src") != std::string::npos);
    }

    RC_ASSERT(threw_invalid);
}

// ─── Property 21b: Mismatched SphericalDeg vs Cartesian3D throws ─────────────

RC_GTEST_PROP(PropCoordsystemConsistency, DegVsCartesianThrows, ()) {
    const auto ni = *rc::gen::inRange<std::size_t>(2, 6);
    const auto nj = *rc::gen::inRange<std::size_t>(2, 6);

    auto src_mesh = build_mesh_with_coord_system(ni, nj,
        axis::topology::CoordinateSystem::SphericalDeg);
    auto dst_mesh = build_mesh_with_coord_system(ni, nj,
        axis::topology::CoordinateSystem::Cartesian3D);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    bool threw_invalid = false;
    try {
        axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(
            src_mesh, dst_mesh, config);
    } catch (const std::invalid_argument&) {
        threw_invalid = true;
    }

    RC_ASSERT(threw_invalid);
}

// ─── Property 21c: Matching CoordinateSystem does NOT throw ──────────────────

RC_GTEST_PROP(PropCoordsystemConsistency, MatchingDoesNotThrow, ()) {
    const auto ni = *rc::gen::inRange<std::size_t>(2, 6);
    const auto nj = *rc::gen::inRange<std::size_t>(2, 6);

    auto coord_sys = *rc::gen::element(
        axis::topology::CoordinateSystem::SphericalDeg,
        axis::topology::CoordinateSystem::SphericalRad);

    auto src_mesh = build_mesh_with_coord_system(ni, nj, coord_sys);
    auto dst_mesh = build_mesh_with_coord_system(ni, nj, coord_sys);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    // Should NOT throw
    bool threw = false;
    try {
        axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(
            src_mesh, dst_mesh, config);
    } catch (...) {
        threw = true;
    }

    RC_ASSERT(!threw);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

static auto* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace
