# ##############################################################################
# AXIS - Arbitrary eXgrid Interpolation Solver
#
# Tier 1 C++20 micro-library providing stateless spatial interpolation
# (regridding) for Earth-system fields via Kokkos.
#
# AXIS opens no files, links no file-format libraries, and performs no MPI
# communication. All byte-level I/O is delegated to AMIO via the plain-data
# GridDescriptor ingest contract, and all distributed communication is delegated
# to HALO via the neutral HaloPattern exchange-plan descriptor.
#
# Dependencies:
#   REQUIRED : Kokkos (parallelism + memory spaces)
#   OPTIONAL : PROJ   (coordinate projection transforms, guarded by AXIS_ENABLE_PROJ)
#
# Explicitly NOT linked (HELM Law #4 / Requirement 22.7):
#   NetCDF, HDF5, ecCodes, yaml-cpp, TensorStore, g2c, eckit, AMIO, HALO
# ##############################################################################

cmake_minimum_required(VERSION 3.21)

project(
    AXIS
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Arbitrary eXgrid Interpolation Solver — stateless spatial regridding for HELM"
)

# ##############################################################################
# Global build settings
# ##############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ##############################################################################
# Options
# ##############################################################################

option(AXIS_ENABLE_PROJ "Enable PROJ (libproj) for coordinate projection transforms" ON)
option(BUILD_TESTING    "Build the AXIS test suite (GTest + RapidCheck)"              OFF)
option(BUILD_FORTRAN    "Build the Fortran iso_c_binding interop layer"               OFF)
option(BUILD_PYTHON     "Build the nanobind Python bindings"                          OFF)
option(BUILD_EXAMPLES   "Build runnable documentation examples"                       OFF)
option(BUILD_BENCHMARKS "Build the AXIS performance benchmark harness"                OFF)

# ##############################################################################
# Dependency discovery
# ##############################################################################

# Kokkos — the ONLY required third-party dependency (HELM Law #2).
find_package(Kokkos REQUIRED)

# KokkosKernels — OPTIONAL accelerator for sparse linear algebra (SpMV, sort).
# When found, AXIS uses KokkosSparse::spmv for CSR apply instead of hand-rolled
# TeamPolicy loops. Header-only usage (no ETI).
find_package(KokkosKernels QUIET)
if(KokkosKernels_FOUND)
    message(STATUS "AXIS: KokkosKernels found — using KokkosSparse::spmv for CSR apply")
else()
    message(STATUS "AXIS: KokkosKernels not found — using built-in CSR apply")
endif()

# PROJ — the single OPTIONAL third-party dependency for projection-string
# coordinate transforms (pure math, no equivalent in AMIO's stack).
if(AXIS_ENABLE_PROJ)
    find_package(PROJ QUIET)
    if(NOT PROJ_FOUND)
        # Fallback: try pkg-config for older PROJ installations.
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PROJ QUIET IMPORTED_TARGET proj)
        endif()
    endif()
    if(PROJ_FOUND)
        message(STATUS "AXIS: PROJ found — projection_builder enabled")
    else()
        message(WARNING "AXIS: AXIS_ENABLE_PROJ=ON but PROJ not found — projection_builder disabled")
        set(AXIS_ENABLE_PROJ OFF)
    endif()
endif()

# ##############################################################################
# Vendored third-party: ArborX (BSD-3-Clause, header-only BVH spatial index)
# Provides GPU-portable bounding-volume-hierarchy for spatial queries used by
# WeightGenerator (nearest, bilinear point-location, conservative overlap).
# ##############################################################################

add_subdirectory(third_party/ArborX EXCLUDE_FROM_ALL)

# ##############################################################################
# Library sources
# ##############################################################################

# projection_builder.cpp is always compiled; the AXIS_ENABLE_PROJ preprocessor
# guard inside selects the real PROJ implementation or a stub that throws.
set(AXIS_SOURCES
    src/detail/raii_handles.cpp
    src/topology/unstructured_mesh.cpp
    src/topology/structured_grid.cpp
    src/topology/mesh_factory.cpp
    src/topology/named_grid_registry.cpp
    src/topology/rule_generator.cpp
    src/topology/projection_builder.cpp
    src/topology/gmsh_writer.cpp
    src/solver/weight_generator.cpp
    src/solver/interpolation_matrix.cpp
    src/solver/halo_pattern.cpp
    src/solver/conservation.cpp
    src/solver/gradient_reconstructor.cpp
)

# ##############################################################################
# Target: axis (STATIC library)
# ##############################################################################

add_library(axis STATIC ${AXIS_SOURCES})
add_library(HELM::AXIS ALIAS axis)

set_target_properties(axis PROPERTIES
    EXPORT_NAME AXIS
    OUTPUT_NAME axis
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_include_directories(axis
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(axis PUBLIC cxx_std_20)

# ##############################################################################
# Link dependencies
# ##############################################################################

# Kokkos — PUBLIC so downstream consumers get Kokkos transitively.
target_link_libraries(axis PUBLIC Kokkos::kokkos)

# KokkosKernels — PUBLIC for KokkosSparse::spmv in apply.hpp (header-only).
if(KokkosKernels_FOUND)
    target_link_libraries(axis PUBLIC Kokkos::kokkoskernels)
    target_compile_definitions(axis PUBLIC AXIS_HAVE_KOKKOSKERNELS=1)
endif()

# ArborX — PRIVATE (vendored, header-only BVH for spatial search in WeightGenerator).
target_link_libraries(axis PRIVATE ArborX::ArborX)

# PROJ — PRIVATE so downstream consumers do NOT link PROJ.
if(AXIS_ENABLE_PROJ)
    if(TARGET PROJ::proj)
        target_link_libraries(axis PRIVATE PROJ::proj)
    elseif(TARGET PkgConfig::PROJ)
        target_link_libraries(axis PRIVATE PkgConfig::PROJ)
    endif()
    target_compile_definitions(axis PRIVATE AXIS_ENABLE_PROJ=1)
endif()

# ##############################################################################
# Fortran interop (optional)
# ##############################################################################

if(BUILD_FORTRAN)
    enable_language(Fortran)

    add_library(axis_fortran STATIC
        src/fortran/axis_c_interop.cpp
        fortran/axis_mod.f90
    )
    add_library(HELM::AXIS_Fortran ALIAS axis_fortran)

    set_target_properties(axis_fortran PROPERTIES
        EXPORT_NAME AXIS_Fortran
        POSITION_INDEPENDENT_CODE ON
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/modules
    )

    target_include_directories(axis_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran/modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/axis/fortran>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(axis_fortran PUBLIC axis)
endif()

# ##############################################################################
# Python bindings (nanobind)
# ##############################################################################

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ##############################################################################
# Documentation (optional — requires Doxygen)
# ##############################################################################

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating AXIS API documentation with Doxygen..."
        VERBATIM
    )
    message(STATUS "AXIS: Doxygen found — 'docs' target available (cmake --build build --target docs)")
else()
    message(STATUS "AXIS: Doxygen not found — 'docs' target disabled (documentation build skipped)")
endif()

# ##############################################################################
# Tier 1 Isolation Scan (HELM Law #4 — static verification)
# ##############################################################################

# Custom target: run the isolation scan on demand via `cmake --build . --target check_isolation`
add_custom_target(check_isolation
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_tier1_isolation.sh ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running AXIS Tier 1 isolation scan (HELM Law #4 verification)..."
    VERBATIM
)

# Register the isolation scan as a CTest so `ctest -R TierIsolation` picks it up.
# This is unconditional — the isolation scan is a static check, not a unit test.
enable_testing()
add_test(
    NAME TierIsolationScan
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_tier1_isolation.sh ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(TierIsolationScan PROPERTIES
    LABELS "isolation;static-analysis"
    TIMEOUT 60
)

# ##############################################################################
# Examples (runnable documentation)
# ##############################################################################

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ##############################################################################
# Testing
# ##############################################################################

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ##############################################################################
# Install + export
# ##############################################################################

install(
    DIRECTORY include/axis
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

set(AXIS_INSTALL_TARGETS axis)
if(BUILD_FORTRAN)
    list(APPEND AXIS_INSTALL_TARGETS axis_fortran)
endif()

install(
    TARGETS ${AXIS_INSTALL_TARGETS}
    EXPORT AXISTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT AXISTargets
    FILE AXISTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/AXISConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/AXISConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/AXISConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/AXISConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/AXISConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AXIS
)

if(BUILD_FORTRAN)
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/modules/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/axis/fortran
        FILES_MATCHING PATTERN "*.mod"
    )
endif()

# ##############################################################################
# Configure-time summary
# ##############################################################################

message(STATUS "")
message(STATUS "AXIS ${PROJECT_VERSION} configuration summary")
message(STATUS "  C++ standard    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  PROJ enabled    : ${AXIS_ENABLE_PROJ}")
message(STATUS "  Build testing   : ${BUILD_TESTING}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build examples  : ${BUILD_EXAMPLES}")
message(STATUS "  Build Fortran   : ${BUILD_FORTRAN}")
message(STATUS "  Build Python    : ${BUILD_PYTHON}")
message(STATUS "  Doxygen docs    : ${DOXYGEN_FOUND}")
message(STATUS "  Install prefix  : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
