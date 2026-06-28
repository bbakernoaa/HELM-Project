// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <stdexcept>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;
using namespace axis::topology;

TEST(NoaaGribNamedGrids, DynamicGenerationAndResolution) {
    // 1. Verify GFS 1.0 degree global grid (grid3)
    EXPECT_TRUE(NamedGridRegistry::is_registered("grid3"));
    auto mesh_grid3 = NamedGridRegistry::generate<MemSpace>("grid3");
    EXPECT_EQ(mesh_grid3.n_cells(), 360 * 181);

    // 2. Verify GFS 0.5 degree global grid (grid4)
    EXPECT_TRUE(NamedGridRegistry::is_registered("grid4"));
    auto mesh_grid4 = NamedGridRegistry::generate<MemSpace>("grid4");
    EXPECT_EQ(mesh_grid4.n_cells(), 720 * 361);

    // 3. Verify NAM ConUS 12km LCC grid (grid218)
#ifdef AXIS_ENABLE_PROJ
    EXPECT_TRUE(NamedGridRegistry::is_registered("grid218"));
    auto mesh_grid218 = NamedGridRegistry::generate<MemSpace>("grid218");
    EXPECT_EQ(mesh_grid218.n_cells(), 614 * 428);
#endif

    // 4. Case-insensitivity verification
    EXPECT_TRUE(NamedGridRegistry::is_registered("GrId3"));
    auto mesh_case = NamedGridRegistry::generate<MemSpace>("GrId3");
    EXPECT_EQ(mesh_case.n_cells(), 360 * 181);

    // 5. Invalid / Unregistered grid parsing checks
    EXPECT_FALSE(NamedGridRegistry::is_registered("grid999"));
    EXPECT_THROW(NamedGridRegistry::generate<MemSpace>("grid999"), std::invalid_argument);
    EXPECT_THROW(NamedGridRegistry::generate<MemSpace>("gridABC"), std::invalid_argument);
    EXPECT_THROW(NamedGridRegistry::generate<MemSpace>("grid0"), std::invalid_argument);
}

}  // namespace axis::test
