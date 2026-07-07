# ####################################################################################################################################################
# DAGR - Directed Acyclic Graph Router
#
# Tier 3 YAML-driven asynchronous event-loop DAG orchestrator for the HELM ecosystem.
# DAGR performs zero mathematical calculations — it routes pointers between lower-tier
# engines (TICK, AMIO, BLEND, HALO) and manages execution dependencies.
#
# Requirements satisfied by this file: 11.1, 11.5
# ####################################################################################################################################################

cmake_minimum_required(VERSION 3.21)

project(
    DAGR
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Directed Acyclic Graph Router for the HELM ecosystem")

# ####################################################################################################################################################
# Global build settings
# ####################################################################################################################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ####################################################################################################################################################
# Library target: dagr (STATIC)
#
# Produces libdagr.a. All lower-tier dependencies are PRIVATE — no transitive leakage
# to consumers (Req 11.5).
# ####################################################################################################################################################

add_library(dagr STATIC
    src/graph_orchestrator.cpp
    src/parse_pipeline.cpp
    src/event_loop.cpp
    src/rank_pool.cpp
    src/transform_node.cpp
    src/logging.cpp
)
add_library(HELM::DAGR ALIAS dagr)

target_include_directories(dagr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(dagr PUBLIC cxx_std_20)

# ####################################################################################################################################################
# Dependencies
#
# All lower-tier dependencies are PRIVATE (Req 11.5 — no upward or transitive leakage).
# Dependencies are discovered via find_package; when building as part of the full HELM
# super-build, parent CMake will have already defined these targets.
# ####################################################################################################################################################

find_package(MPI QUIET)
find_package(SPAN QUIET)
find_package(TICK QUIET)
find_package(HALO QUIET)
find_package(LOGS QUIET)
find_package(BLEND QUIET)
find_package(AMIO QUIET)
find_package(CONF QUIET)

if(TARGET MPI::MPI_CXX)
    target_link_libraries(dagr PRIVATE MPI::MPI_CXX)
elseif(MPI_CXX_FOUND)
    target_include_directories(dagr PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()

# Link only those dependencies that are available. In the full HELM build all will be present.
set(_dagr_deps "")
foreach(_dep HELM::SPAN HELM::TICK HELM::HALO HELM::LOGS HELM::BLEND HELM::AMIO HELM::CONF)
    if(TARGET ${_dep})
        list(APPEND _dagr_deps ${_dep})
    endif()
endforeach()

if(_dagr_deps)
    target_link_libraries(dagr PRIVATE ${_dagr_deps})
endif()
unset(_dagr_deps)

# When building standalone (without full super-build), provide include paths for
# sibling libraries so that headers resolve correctly.
set(_sibling_libs span tick halo logs blend AMIO amio conf)
foreach(_lib ${_sibling_libs})
    set(_inc_dir "${CMAKE_CURRENT_SOURCE_DIR}/../${_lib}/include")
    if(IS_DIRECTORY ${_inc_dir})
        target_include_directories(dagr PRIVATE ${_inc_dir})
    endif()
endforeach()
unset(_sibling_libs)

# ####################################################################################################################################################
# Testing
# ####################################################################################################################################################

option(BUILD_TESTING "Build the DAGR test suite" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ####################################################################################################################################################
# Install and Export
# ####################################################################################################################################################

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS dagr EXPORT DAGRTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/dagr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT DAGRTargets
    FILE DAGRTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DAGR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DAGRConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DAGRConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DAGR
)
