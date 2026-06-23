# ##############################################################################
# BLEND — Stateless Array Blending Kernels
#
# Tier 1 C++20 header-only micro-library providing BLAS-style element-wise
# array blending kernels dispatched via Kokkos execution spaces.
#
# Dependencies:
#   REQUIRED : Kokkos (parallel dispatch + memory spaces)
#
# Tier 1 compliance: BLEND has no knowledge of time, grids, or domain physics.
# ##############################################################################

cmake_minimum_required(VERSION 3.21)

project(
    BLEND
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Stateless array blending kernels for the HELM ecosystem"
)

# ##############################################################################
# Global build settings
# ##############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# ##############################################################################
# Options
# ##############################################################################

option(BUILD_TESTING "Build the BLEND test suite" OFF)

# ##############################################################################
# Dependency discovery
# ##############################################################################

find_package(Kokkos REQUIRED)

# ##############################################################################
# Target: blend (INTERFACE — header-only)
# ##############################################################################

add_library(blend INTERFACE)
add_library(HELM::BLEND ALIAS blend)

target_include_directories(blend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(blend INTERFACE Kokkos::kokkos)
target_compile_features(blend INTERFACE cxx_std_20)

# ##############################################################################
# Testing
# ##############################################################################

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ##############################################################################
# Install + export
# ##############################################################################

install(
    DIRECTORY include/blend
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(
    TARGETS blend
    EXPORT BLENDTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT BLENDTargets
    FILE BLENDTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/BLEND
)
