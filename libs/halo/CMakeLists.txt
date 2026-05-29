cmake_minimum_required(VERSION 3.21)
project(HALO
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Hardware-Abstracted Link Operations — RAII MPI with GPU-aware halo exchange"
)

# ─── C++20 Requirement ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ─────────────────────────────────────────────────────────────────
option(HALO_GPU_AWARE_MPI "Enable GPU-aware MPI (pass device pointers directly)" OFF)
option(BUILD_TESTING "Build the HALO test suite (GTest + RapidCheck)" OFF)
option(BUILD_FORTRAN "Build the Fortran C-interop layer (halo_mod)" ON)

# ─── Required Dependencies ───────────────────────────────────────────────────
find_package(MPI REQUIRED COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "HALO requires MPI with CXX component. Please install an MPI implementation (e.g., OpenMPI, MPICH) with C++ support.")
endif()

find_package(Kokkos REQUIRED)
if(NOT Kokkos_FOUND)
    message(FATAL_ERROR "HALO requires Kokkos. Please install Kokkos or set Kokkos_ROOT to the Kokkos installation prefix.")
endif()

# ─── Library Target ──────────────────────────────────────────────────────────
add_library(halo)
add_library(HELM::HALO ALIAS halo)

target_sources(halo PRIVATE
    src/communicator.cpp
    src/environment.cpp
    src/halo_handle.cpp
    src/halo_plan.cpp
    src/request_guard.cpp
    src/window_guard.cpp
    src/detail/staging.cpp
)

target_include_directories(halo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(halo
    PUBLIC
        MPI::MPI_CXX
        Kokkos::kokkos
)

target_compile_features(halo PUBLIC cxx_std_20)

# ─── GPU-Aware MPI Compile Definition ───────────────────────────────────────
if(HALO_GPU_AWARE_MPI)
    target_compile_definitions(halo PUBLIC HALO_GPU_AWARE_MPI)
endif()

# ─── Fortran Interop Layer ───────────────────────────────────────────────────
if(BUILD_FORTRAN)
    enable_language(Fortran)
    # Fortran targets will be added in task 16.2
endif()

# ─── Install and Export Configuration ────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library binary
install(TARGETS halo
    EXPORT HALOTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets with HELM:: namespace
install(EXPORT HALOTargets
    FILE HALOTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# Generate and install the package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HALOConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# Generate and install the version file with SameMajorVersion compatibility
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HALOConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HALO
)

# ─── Testing ─────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)

    if(BUILD_FORTRAN)
        add_subdirectory(tests_fortran)
    endif()
endif()
