// AXIS unit test: StructuredGrid::to_unstructured_band()
// Verifies that extracting j-row bands from a global grid (a) reproduces the
// full-grid mesh when the whole range is taken, and (b) yields bit-identical
// shared boundary vertices between adjacent bands — the property that makes MPI
// row-band decomposition of conservative regridding seam-free.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace {
class KokkosEnv : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
static auto *const kenv = ::testing::AddGlobalTestEnvironment(new KokkosEnv);
}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Build a uniform rectilinear lat-lon grid (centers only; corners synthesized).
static topology::StructuredGrid<MemSpace> make_grid(std::size_t ni, std::size_t nj, double lon0, double dlon, double lat0, double dlat) {
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon0 + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat0 + (static_cast<double>(j) + 0.5) * dlat;
        }
    }
    return topology::StructuredGrid<MemSpace>(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);
}

// Full-range band [0, nj) reproduces the full-grid mesh exactly.
TEST(StructuredBand, FullRangeEqualsToUnstructured) {
    const std::size_t ni = 8;
    const std::size_t nj = 6;
    auto full = make_grid(ni, nj, 0.0, 10.0, -30.0, 10.0).to_unstructured();
    auto band = make_grid(ni, nj, 0.0, 10.0, -30.0, 10.0).to_unstructured_band(0, nj);

    ASSERT_EQ(full.node_coords().extent(0), band.node_coords().extent(0));
    ASSERT_EQ(full.n_cells(), band.n_cells());

    auto fn = full.node_coords();
    auto bn = band.node_coords();
    for (std::size_t k = 0; k < fn.extent(0); ++k) {
        EXPECT_DOUBLE_EQ(fn(k, 0), bn(k, 0));
        EXPECT_DOUBLE_EQ(fn(k, 1), bn(k, 1));
    }
}

// Adjacent bands must share a bit-identical boundary vertex row: the top vertex
// row of band [0, s) equals the bottom vertex row of band [s, nj).
TEST(StructuredBand, AdjacentBandsShareBoundaryVertices) {
    const std::size_t ni = 8;
    const std::size_t nj = 6;
    const std::size_t split = 3;

    auto lower = make_grid(ni, nj, 0.0, 10.0, -30.0, 10.0).to_unstructured_band(0, split);
    auto upper = make_grid(ni, nj, 0.0, 10.0, -30.0, 10.0).to_unstructured_band(split, nj - split);

    auto ln = lower.node_coords();
    auto un = upper.node_coords();

    const std::size_t nip1 = ni + 1;
    const std::size_t lower_top_row = split;  // vertex rows [0..split]; top row index == split
    for (std::size_t i = 0; i < nip1; ++i) {
        const std::size_t lower_idx = i + lower_top_row * nip1;  // top row of lower band
        const std::size_t upper_idx = i;                         // bottom row of upper band
        EXPECT_DOUBLE_EQ(ln(lower_idx, 0), un(upper_idx, 0)) << "lon mismatch at seam vertex i=" << i;
        EXPECT_DOUBLE_EQ(ln(lower_idx, 1), un(upper_idx, 1)) << "lat mismatch at seam vertex i=" << i;
    }
}

// Out-of-range requests are rejected.
TEST(StructuredBand, RejectsInvalidRange) {
    const std::size_t ni = 4;
    const std::size_t nj = 4;
    EXPECT_THROW(make_grid(ni, nj, 0.0, 10.0, -20.0, 10.0).to_unstructured_band(0, 0), std::invalid_argument);
    EXPECT_THROW(make_grid(ni, nj, 0.0, 10.0, -20.0, 10.0).to_unstructured_band(2, 3), std::invalid_argument);
}

}  // namespace axis::test
