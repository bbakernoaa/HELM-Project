# ##############################################################################
# SPAN - Shared Pointer & Array Network
#
# Tier 2 C++20 micro-library providing the zero-copy boundary layer between
# legacy Fortran domain models and the modern C++ HELM infrastructure.
#
# SPAN wraps native pointers in std::mdspan views, maintains a coherency state
# machine for host/device synchronization, and provides triple-buffered I/O
# protection for asynchronous background writes.
#
# Dependencies:
#   REQUIRED : Kokkos (parallelism + memory spaces, mdspan reference impl)
#
# Explicitly NOT linked (HELM Tier 2 compliance — Requirement 13.3):
#   HALO, AMIO, TICK, LOGS, DAGR, AXIS, domain-science models
# ##############################################################################

cmake_minimum_required(VERSION 3.21)

project(
    SPAN
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Shared Pointer & Array Network — zero-copy Fortran/C++ boundary layer for HELM"
)

# ##############################################################################
# Global build settings
# ##############################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

# ##############################################################################
# Options
# ##############################################################################

option(SPAN_BUILD_TESTING "Build the SPAN test suite"           OFF)
option(SPAN_BUILD_C_API   "Build the thin C-API shared library" OFF)

# ##############################################################################
# Dependency discovery
# ##############################################################################

# Kokkos — the ONLY required third-party dependency (HELM Law #2).
find_package(Kokkos REQUIRED)

# ##############################################################################
# Target: span (INTERFACE — header-only template layer)
# ##############################################################################

add_library(span INTERFACE)
add_library(HELM::SPAN ALIAS span)

target_include_directories(span INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(span INTERFACE Kokkos::kokkos)
target_compile_features(span INTERFACE cxx_std_20)

# ##############################################################################
# Target: span_c_api (SHARED — optional thin C-API, Requirement 14.3)
# ##############################################################################

if(SPAN_BUILD_C_API)
    add_library(span_c_api SHARED src/span_c_api.cpp)
    add_library(HELM::SPAN_CAPI ALIAS span_c_api)

    set_target_properties(span_c_api PROPERTIES
        EXPORT_NAME SPAN_CAPI
        OUTPUT_NAME span_c_api
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    target_link_libraries(span_c_api PRIVATE span)

    target_include_directories(span_c_api PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

# ##############################################################################
# Testing
# ##############################################################################

if(SPAN_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ##############################################################################
# Install + export
# ##############################################################################

install(
    DIRECTORY include/span
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)

set(SPAN_INSTALL_TARGETS span)
if(SPAN_BUILD_C_API)
    list(APPEND SPAN_INSTALL_TARGETS span_c_api)
endif()

install(
    TARGETS ${SPAN_INSTALL_TARGETS}
    EXPORT SPANTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT SPANTargets
    FILE SPANTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SPAN
)

# ##############################################################################
# Configure-time summary
# ##############################################################################

message(STATUS "")
message(STATUS "SPAN ${PROJECT_VERSION} configuration summary")
message(STATUS "  C++ standard   : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build testing  : ${SPAN_BUILD_TESTING}")
message(STATUS "  Build C-API    : ${SPAN_BUILD_C_API}")
message(STATUS "  Install prefix : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
