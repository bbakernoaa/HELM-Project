cmake_minimum_required(VERSION 3.21)
project(TICK VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Library target ---
add_library(tick STATIC src/sync.cpp)
add_library(HELM::TICK ALIAS tick)

target_include_directories(tick
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(tick PUBLIC cxx_std_20)

# --- C API target (always built) ---
add_library(tick_c STATIC src/tick_c_api.cpp)
add_library(HELM::TICK_C ALIAS tick_c)
target_link_libraries(tick_c PUBLIC tick)
target_include_directories(tick_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# --- Fortran module target (optional, gated) ---
option(TICK_BUILD_FORTRAN "Build the Fortran tick_mod module" OFF)
if(TICK_BUILD_FORTRAN)
    enable_language(Fortran)
    add_library(tick_fortran STATIC fortran/tick_mod.f90)
    add_library(HELM::TICK_Fortran ALIAS tick_fortran)
    target_link_libraries(tick_fortran PUBLIC tick_c)
    # Make the .mod file available to downstream consumers
    target_include_directories(tick_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tick>
    )
endif()

# --- Resolution configuration ---
set(TICK_RESOLUTION "NANOSECONDS" CACHE STRING "Temporal resolution: SECONDS or NANOSECONDS")
set_property(CACHE TICK_RESOLUTION PROPERTY STRINGS SECONDS NANOSECONDS)

if(TICK_RESOLUTION STREQUAL "SECONDS")
    target_compile_definitions(tick PUBLIC TICK_RESOLUTION_SECONDS)
else()
    target_compile_definitions(tick PUBLIC TICK_RESOLUTION_NANOSECONDS)
endif()

# --- Testing ---
option(TICK_BUILD_TESTING "Build the TICK test suite" OFF)
if(TICK_BUILD_TESTING)
    enable_testing()
    find_package(GTest REQUIRED)
    find_package(rapidcheck REQUIRED)
    add_subdirectory(tests)
endif()

# --- Install and Export ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tick tick_c EXPORT TICKTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/tick DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Conditional Fortran install
if(TICK_BUILD_FORTRAN)
    install(TARGETS tick_fortran EXPORT TICKTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    # Install .mod file
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tick
            FILES_MATCHING PATTERN "tick_mod.mod")
endif()
install(EXPORT TICKTargets
    FILE TICKTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TICK
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TICKConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TICKConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TICK
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TICKConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TICKConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TICKConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TICK
)
