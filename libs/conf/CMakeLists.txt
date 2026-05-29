cmake_minimum_required(VERSION 3.21)
project(CONF
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Configuration Object & Notation Framework — stateless RAII YAML config parser"
)

# ─── C++20 Requirement ───────────────────────────────────────────────────────
# C++20 is a non-optional requirement; compiler-specific extensions are disabled.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ─────────────────────────────────────────────────────────────────
option(BUILD_TESTING "Build the CONF test suite (GTest + RapidCheck)" OFF)
option(BUILD_FORTRAN "Build the Fortran C-interop layer (conf_mod)" ON)
option(CONF_USE_SYSTEM_YAMLCPP "Use a system/Spack yaml-cpp via find_package instead of FetchContent" OFF)

# ─── Private YAML Backend Acquisition (yaml-cpp) ─────────────────────────────
# yaml-cpp is a PRIVATE implementation detail. It is acquired either from a
# system/Spack install (find_package, when CONF_USE_SYSTEM_YAMLCPP is ON and a
# compatible 0.8 is discoverable) or fetched privately via FetchContent pinned
# to 0.8.0. It is linked PRIVATE so neither its headers nor its link flags
# propagate to consumers of HELM::CONF, and its install/export is disabled.
set(CONF_YAMLCPP_TARGET "")

if(CONF_USE_SYSTEM_YAMLCPP)
    find_package(yaml-cpp 0.8 QUIET)
    if(yaml-cpp_FOUND)
        message(STATUS "CONF: using system yaml-cpp ${yaml-cpp_VERSION}")
        if(TARGET yaml-cpp::yaml-cpp)
            set(CONF_YAMLCPP_TARGET yaml-cpp::yaml-cpp)
        elseif(TARGET yaml-cpp)
            set(CONF_YAMLCPP_TARGET yaml-cpp)
        endif()
    else()
        message(STATUS "CONF: CONF_USE_SYSTEM_YAMLCPP=ON but no compatible yaml-cpp 0.8 found; falling back to FetchContent")
    endif()
endif()

if(NOT CONF_YAMLCPP_TARGET)
    message(STATUS "CONF: fetching yaml-cpp 0.8.0 privately via FetchContent")
    include(FetchContent)

    # Disable yaml-cpp's own tests, tools, contrib, and install so it is never
    # installed or exported alongside CONF.
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        0.8.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(yaml-cpp)

    if(TARGET yaml-cpp::yaml-cpp)
        set(CONF_YAMLCPP_TARGET yaml-cpp::yaml-cpp)
    else()
        set(CONF_YAMLCPP_TARGET yaml-cpp)
    endif()
endif()

# ─── Library Target ──────────────────────────────────────────────────────────
add_library(conf)
add_library(HELM::CONF ALIAS conf)

# Core C++ sources. (Authored by later tasks; listed here so the build is wired.)
target_sources(conf PRIVATE
    src/config.cpp
    src/value.cpp
    src/detail/yaml_tree.cpp
)

# The Fortran C-interop bridge is compiled into the library only when the
# Fortran layer is enabled.
if(BUILD_FORTRAN)
    target_sources(conf PRIVATE
        src/fortran/conf_c_interop.cpp
    )
endif()

# Public include dir is a PUBLIC usage requirement and is installed.
# The src/ tree (including src/detail/) is a PRIVATE include path that is
# compiled against but never installed or exposed to consumers.
target_include_directories(conf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(conf PUBLIC cxx_std_20)

# yaml-cpp is linked PRIVATE and wrapped in $<BUILD_INTERFACE:...> so the
# dependency is required only while building CONF and is never recorded in the
# exported/installed interface. This keeps yaml-cpp an invisible detail (no
# propagated headers or link flags, no find_dependency in CONFConfig.cmake) and
# allows install(EXPORT) to succeed without exporting the fetched yaml-cpp target.
target_link_libraries(conf
    PRIVATE
        $<BUILD_INTERFACE:${CONF_YAMLCPP_TARGET}>
)

# ─── Fortran Interop Layer ───────────────────────────────────────────────────
if(BUILD_FORTRAN)
    enable_language(Fortran)

    # conf_mod is the public Fortran wrapper module. It compiles fortran/
    # conf_mod.f90 into a static library and publishes the generated .mod files
    # (written to a dedicated module directory) as a usage requirement so that
    # downstream Fortran targets — e.g. the integration test in tests_fortran —
    # can `use conf_mod`. The actual extern "C" symbols it calls live in the
    # `conf` C++ library (compiled above under BUILD_FORTRAN), which Fortran
    # consumers link alongside this target.
    add_library(conf_mod STATIC
        fortran/conf_mod.f90
    )

    set(CONF_FORTRAN_MOD_DIR ${CMAKE_CURRENT_BINARY_DIR}/conf_mod_mod)
    set_target_properties(conf_mod PROPERTIES
        Fortran_MODULE_DIRECTORY ${CONF_FORTRAN_MOD_DIR}
    )
    target_include_directories(conf_mod
        PUBLIC
            $<BUILD_INTERFACE:${CONF_FORTRAN_MOD_DIR}>
    )

    # The Fortran wrapper drives the extern "C" bridge carried by `conf`.
    target_link_libraries(conf_mod PUBLIC conf)
endif()

# ─── Install and Export Configuration ────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library binary and record its public include destination.
install(TARGETS conf
    EXPORT CONFTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install ONLY the public headers under include/conf/. The private src/detail/
# headers are never installed.
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export the conf target under the HELM:: namespace as CONFTargets.cmake.
install(EXPORT CONFTargets
    FILE CONFTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CONF
)

# Generate the package config file. CONFConfig.cmake.in deliberately omits
# find_dependency(yaml-cpp) because yaml-cpp is a PRIVATE, statically-absorbed
# detail that is invisible to downstream consumers.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CONFConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CONFConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CONF
)

# Generate the version file with SameMajorVersion compatibility.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CONFConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CONFConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CONFConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CONF
)

# ─── Testing ─────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)

    # The Fortran integration test is added only when both the Fortran layer
    # and the test suite are enabled.
    if(BUILD_FORTRAN)
        add_subdirectory(tests_fortran)
    endif()
endif()
