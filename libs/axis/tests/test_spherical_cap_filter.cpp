// AXIS unit test: Spherical cap early-exit filter
// Verifies compute_cell_centroid_xyz, compute_angular_radius,
// spherical_cap_rejects, and precompute_cap_data.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>

#include <axis/detail/spherical_cap_filter.hpp>

namespace axis::test {

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double deg2rad = pi / 180.0;
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// spherical_cap_rejects tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalCapFilter, RejectsDistantCaps) {
    // Two caps centered at antipodal points with small radii should be rejected
    axis::detail::Vec3 centroid_s{1.0, 0.0, 0.0};   // on equator at lon=0
    axis::detail::Vec3 centroid_d{-1.0, 0.0, 0.0};  // on equator at lon=180

    double radius_s = 0.1;  // ~5.7 degrees
    double radius_d = 0.1;

    EXPECT_TRUE(axis::detail::spherical_cap_rejects(
        centroid_s, radius_s, centroid_d, radius_d));
}

TEST(SphericalCapFilter, AcceptsOverlappingCaps) {
    // Two caps centered at the same point should never be rejected
    axis::detail::Vec3 centroid{1.0, 0.0, 0.0};

    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        centroid, 0.1, centroid, 0.1));
}

TEST(SphericalCapFilter, AcceptsNearCaps) {
    // Two caps whose combined radii exceed their angular distance
    // centroid_s at (0, 0) and centroid_d at (10°, 0°) on equator
    axis::detail::Vec3 centroid_s{1.0, 0.0, 0.0};
    double lon_d = 10.0 * deg2rad;
    axis::detail::Vec3 centroid_d{std::cos(lon_d), std::sin(lon_d), 0.0};

    // Angular distance is ~10° = ~0.1745 rad
    // Sum of radii = 0.2 > 0.1745 → should NOT reject
    double radius_s = 0.1;
    double radius_d = 0.1;

    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        centroid_s, radius_s, centroid_d, radius_d));
}

TEST(SphericalCapFilter, RejectsBarelyDisjointCaps) {
    // Two caps whose combined radii is just less than their angular distance
    axis::detail::Vec3 centroid_s{1.0, 0.0, 0.0};
    // Place centroid_d at 30° away on equator
    double lon_d = 30.0 * deg2rad;
    axis::detail::Vec3 centroid_d{std::cos(lon_d), std::sin(lon_d), 0.0};

    // Angular distance is 30° = ~0.5236 rad
    // Sum of radii = 0.25 + 0.25 = 0.5 < 0.5236 → should reject
    double radius_s = 0.25;
    double radius_d = 0.25;

    EXPECT_TRUE(axis::detail::spherical_cap_rejects(
        centroid_s, radius_s, centroid_d, radius_d));
}

TEST(SphericalCapFilter, HandlesIdenticalCentroids) {
    // Same centroid → angular distance = 0, never rejected
    axis::detail::Vec3 centroid{0.0, 0.0, 1.0};  // north pole

    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        centroid, 0.001, centroid, 0.001));
}

TEST(SphericalCapFilter, HandlesAntipodalCentroidsLargeRadii) {
    // Antipodal centroids with large radii (> pi/2 each) should accept
    axis::detail::Vec3 centroid_s{0.0, 0.0, 1.0};   // north pole
    axis::detail::Vec3 centroid_d{0.0, 0.0, -1.0};  // south pole

    // Angular distance = pi
    // Sum of radii = 2.0 + 2.0 = 4.0 > pi → should NOT reject
    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        centroid_s, 2.0, centroid_d, 2.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_cell_centroid_xyz tests (via precompute_cap_data)
// ─────────────────────────────────────────────────────────────────────────────

// Helper: build a simple mesh with known cell geometry
class SphericalCapMeshTest : public ::testing::Test {
protected:
    using MemSpace = Kokkos::HostSpace;

    // Build a simple mesh with one quadrilateral cell at the equator
    // centered at lon=0, lat=0, spanning ±5° in each direction
    void SetUp() override {
        // 4 nodes: corners of a 10°×10° cell centered at (0°, 0°)
        const std::size_t n_nodes = 4;
        const std::size_t n_cells = 1;

        Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>
            node_coords("node_coords", n_nodes, 2);

        // Node 0: (-5, -5)
        node_coords(0, 0) = -5.0;  node_coords(0, 1) = -5.0;
        // Node 1: ( 5, -5)
        node_coords(1, 0) =  5.0;  node_coords(1, 1) = -5.0;
        // Node 2: ( 5,  5)
        node_coords(2, 0) =  5.0;  node_coords(2, 1) =  5.0;
        // Node 3: (-5,  5)
        node_coords(3, 0) = -5.0;  node_coords(3, 1) =  5.0;

        Kokkos::View<axis::index_t*, MemSpace> offsets("offsets", n_cells + 1);
        offsets(0) = 0;
        offsets(1) = 4;

        Kokkos::View<axis::index_t*, MemSpace> indices("indices", 4);
        indices(0) = 0;
        indices(1) = 1;
        indices(2) = 2;
        indices(3) = 3;

        mesh_ = axis::topology::UnstructuredMesh<MemSpace>(
            std::move(node_coords),
            std::move(offsets),
            std::move(indices),
            axis::topology::CoordinateSystem::SphericalDeg);
    }

    axis::topology::UnstructuredMesh<MemSpace> mesh_;
};

TEST_F(SphericalCapMeshTest, CentroidIsNearCenter) {
    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh_);

    // Centroid of a cell centered at (0°, 0°) should be near (1, 0, 0) on unit sphere
    axis::detail::Vec3 centroid = cap_data.centroids(0);

    // x should be close to 1 (but slightly less due to spherical curvature)
    EXPECT_GT(centroid.x, 0.99);
    EXPECT_NEAR(centroid.y, 0.0, 0.01);
    EXPECT_NEAR(centroid.z, 0.0, 0.01);

    // Should be unit length
    double len = std::sqrt(centroid.x * centroid.x +
                           centroid.y * centroid.y +
                           centroid.z * centroid.z);
    EXPECT_NEAR(len, 1.0, 1e-14);
}

TEST_F(SphericalCapMeshTest, AngularRadiusBoundsAllVertices) {
    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh_);

    axis::detail::Vec3 centroid = cap_data.centroids(0);
    double radius = cap_data.angular_radii(0);

    // The angular radius should be >= angular distance to each vertex
    // For a 10°×10° cell, the farthest corner is ~sqrt(5^2 + 5^2) = ~7.07° away
    // which is roughly 0.123 radians
    EXPECT_GT(radius, 0.0);
    EXPECT_LT(radius, 0.2);  // Should be reasonable for a small cell

    // Verify each vertex is within the angular radius
    auto node_coords = mesh_.node_coords();
    auto conn_offsets = mesh_.conn_offsets();
    auto conn_indices = mesh_.conn_indices();

    for (int v = 0; v < 4; ++v) {
        axis::index_t node_idx = conn_indices[v];
        double lon = node_coords(node_idx, 0) * deg2rad;
        double lat = node_coords(node_idx, 1) * deg2rad;
        double cos_lat = std::cos(lat);
        axis::detail::Vec3 vertex{
            cos_lat * std::cos(lon),
            cos_lat * std::sin(lon),
            std::sin(lat)};

        double d = axis::detail::dot(centroid, vertex);
        d = std::min(1.0, std::max(-1.0, d));
        double angle = std::acos(d);

        EXPECT_LE(angle, radius + 1e-14)
            << "Vertex " << v << " is outside the angular radius";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-cell test: rejection of distant cells, acceptance of overlapping cells
// ─────────────────────────────────────────────────────────────────────────────

class SphericalCapMultiCellTest : public ::testing::Test {
protected:
    using MemSpace = Kokkos::HostSpace;

    void SetUp() override {
        // Two cells:
        //   Cell 0: centered at (0°, 0°), 10°×10°
        //   Cell 1: centered at (180°, 0°), 10°×10° (far away)
        const std::size_t n_nodes = 8;
        const std::size_t n_cells = 2;

        Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>
            node_coords("node_coords", n_nodes, 2);

        // Cell 0 vertices: centered at (0°, 0°)
        node_coords(0, 0) = -5.0;  node_coords(0, 1) = -5.0;
        node_coords(1, 0) =  5.0;  node_coords(1, 1) = -5.0;
        node_coords(2, 0) =  5.0;  node_coords(2, 1) =  5.0;
        node_coords(3, 0) = -5.0;  node_coords(3, 1) =  5.0;

        // Cell 1 vertices: centered at (180°, 0°)
        node_coords(4, 0) = 175.0; node_coords(4, 1) = -5.0;
        node_coords(5, 0) = 185.0; node_coords(5, 1) = -5.0;
        node_coords(6, 0) = 185.0; node_coords(6, 1) =  5.0;
        node_coords(7, 0) = 175.0; node_coords(7, 1) =  5.0;

        Kokkos::View<axis::index_t*, MemSpace> offsets("offsets", n_cells + 1);
        offsets(0) = 0;
        offsets(1) = 4;
        offsets(2) = 8;

        Kokkos::View<axis::index_t*, MemSpace> indices("indices", 8);
        for (int i = 0; i < 8; ++i) indices(i) = i;

        mesh_ = axis::topology::UnstructuredMesh<MemSpace>(
            std::move(node_coords),
            std::move(offsets),
            std::move(indices),
            axis::topology::CoordinateSystem::SphericalDeg);
    }

    axis::topology::UnstructuredMesh<MemSpace> mesh_;
};

TEST_F(SphericalCapMultiCellTest, DistantCellsAreRejected) {
    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh_);

    // Cell 0 at (0°, 0°) and Cell 1 at (180°, 0°) should be rejected
    EXPECT_TRUE(axis::detail::spherical_cap_rejects(
        cap_data.centroids(0), cap_data.angular_radii(0),
        cap_data.centroids(1), cap_data.angular_radii(1)));
}

TEST_F(SphericalCapMultiCellTest, SameCellIsNotRejected) {
    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh_);

    // A cell tested against itself should never be rejected
    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        cap_data.centroids(0), cap_data.angular_radii(0),
        cap_data.centroids(0), cap_data.angular_radii(0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjacent cells test: two cells sharing an edge should NOT be rejected
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalCapFilter, AdjacentCellsNotRejected) {
    using MemSpace = Kokkos::HostSpace;

    // Two adjacent 10° cells sharing the edge at lon=5°
    //   Cell 0: lon [-5, 5], lat [-5, 5]
    //   Cell 1: lon [5, 15], lat [-5, 5]
    const std::size_t n_nodes = 6;  // shared edge nodes
    const std::size_t n_cells = 2;

    Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>
        node_coords("node_coords", n_nodes, 2);

    // Cell 0: nodes 0,1,2,3
    node_coords(0, 0) = -5.0; node_coords(0, 1) = -5.0;
    node_coords(1, 0) =  5.0; node_coords(1, 1) = -5.0;
    node_coords(2, 0) =  5.0; node_coords(2, 1) =  5.0;
    node_coords(3, 0) = -5.0; node_coords(3, 1) =  5.0;
    // Cell 1: nodes 1,4,5,2 (shares nodes 1 and 2 with cell 0)
    node_coords(4, 0) = 15.0; node_coords(4, 1) = -5.0;
    node_coords(5, 0) = 15.0; node_coords(5, 1) =  5.0;

    Kokkos::View<axis::index_t*, MemSpace> offsets("offsets", n_cells + 1);
    offsets(0) = 0;
    offsets(1) = 4;
    offsets(2) = 8;

    Kokkos::View<axis::index_t*, MemSpace> indices("indices", 8);
    indices(0) = 0; indices(1) = 1; indices(2) = 2; indices(3) = 3;
    indices(4) = 1; indices(5) = 4; indices(6) = 5; indices(7) = 2;

    auto mesh = axis::topology::UnstructuredMesh<MemSpace>(
        std::move(node_coords),
        std::move(offsets),
        std::move(indices),
        axis::topology::CoordinateSystem::SphericalDeg);

    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh);

    // Adjacent cells should NOT be rejected (caps overlap)
    EXPECT_FALSE(axis::detail::spherical_cap_rejects(
        cap_data.centroids(0), cap_data.angular_radii(0),
        cap_data.centroids(1), cap_data.angular_radii(1)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Angular radius soundness: radius always >= distance to every vertex
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalCapFilter, AngularRadiusSoundnessLargeCell) {
    using MemSpace = Kokkos::HostSpace;

    // A large cell spanning 60°×60° centered at (0°, 30°)
    const std::size_t n_nodes = 4;
    Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>
        node_coords("node_coords", n_nodes, 2);

    node_coords(0, 0) = -30.0; node_coords(0, 1) =  0.0;
    node_coords(1, 0) =  30.0; node_coords(1, 1) =  0.0;
    node_coords(2, 0) =  30.0; node_coords(2, 1) = 60.0;
    node_coords(3, 0) = -30.0; node_coords(3, 1) = 60.0;

    Kokkos::View<axis::index_t*, MemSpace> offsets("offsets", 2);
    offsets(0) = 0; offsets(1) = 4;

    Kokkos::View<axis::index_t*, MemSpace> indices("indices", 4);
    indices(0) = 0; indices(1) = 1; indices(2) = 2; indices(3) = 3;

    auto mesh = axis::topology::UnstructuredMesh<MemSpace>(
        std::move(node_coords),
        std::move(offsets),
        std::move(indices),
        axis::topology::CoordinateSystem::SphericalDeg);

    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh);

    axis::detail::Vec3 centroid = cap_data.centroids(0);
    double radius = cap_data.angular_radii(0);

    // The centroid should be unit length
    double len = std::sqrt(centroid.x * centroid.x +
                           centroid.y * centroid.y +
                           centroid.z * centroid.z);
    EXPECT_NEAR(len, 1.0, 1e-14);

    // Verify that the angular radius bounds all 4 vertices
    std::vector<std::pair<double, double>> corners = {
        {-30.0, 0.0}, {30.0, 0.0}, {30.0, 60.0}, {-30.0, 60.0}};

    for (const auto& [lon_deg, lat_deg] : corners) {
        double lon = lon_deg * deg2rad;
        double lat = lat_deg * deg2rad;
        double cos_lat = std::cos(lat);
        axis::detail::Vec3 vertex{
            cos_lat * std::cos(lon),
            cos_lat * std::sin(lon),
            std::sin(lat)};

        double d = axis::detail::dot(centroid, vertex);
        d = std::min(1.0, std::max(-1.0, d));
        double angle = std::acos(d);

        EXPECT_LE(angle, radius + 1e-14)
            << "Vertex at (" << lon_deg << "°, " << lat_deg
            << "°) is outside the angular radius";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Radians coordinate system test
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalCapFilter, WorksWithRadianCoordinates) {
    using MemSpace = Kokkos::HostSpace;

    // Same as the basic test but with coordinates in radians
    const std::size_t n_nodes = 4;
    Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>
        node_coords("node_coords", n_nodes, 2);

    // 10°×10° cell centered at (0, 0) in radians
    double half = 5.0 * deg2rad;
    node_coords(0, 0) = -half; node_coords(0, 1) = -half;
    node_coords(1, 0) =  half; node_coords(1, 1) = -half;
    node_coords(2, 0) =  half; node_coords(2, 1) =  half;
    node_coords(3, 0) = -half; node_coords(3, 1) =  half;

    Kokkos::View<axis::index_t*, MemSpace> offsets("offsets", 2);
    offsets(0) = 0; offsets(1) = 4;

    Kokkos::View<axis::index_t*, MemSpace> indices("indices", 4);
    indices(0) = 0; indices(1) = 1; indices(2) = 2; indices(3) = 3;

    auto mesh = axis::topology::UnstructuredMesh<MemSpace>(
        std::move(node_coords),
        std::move(offsets),
        std::move(indices),
        axis::topology::CoordinateSystem::SphericalRad);

    auto cap_data = axis::detail::precompute_cap_data<MemSpace>(mesh);

    axis::detail::Vec3 centroid = cap_data.centroids(0);
    double radius = cap_data.angular_radii(0);

    // Centroid should be near (1, 0, 0) on unit sphere
    EXPECT_GT(centroid.x, 0.99);
    EXPECT_NEAR(centroid.y, 0.0, 0.01);
    EXPECT_NEAR(centroid.z, 0.0, 0.01);

    // Angular radius should be reasonable for a small cell
    EXPECT_GT(radius, 0.0);
    EXPECT_LT(radius, 0.2);
}

}  // namespace axis::test
