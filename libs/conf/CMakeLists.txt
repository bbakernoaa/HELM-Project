cmake_minimum_required(VERSION 3.21)
project(CONF
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Configuration Object & Notation Framework - Stateless RAII YAML parser for HELM"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTING "Build the CONF test suite (GTest + RapidCheck)" OFF)
option(BUILD_FORTRAN "Build the Fortran C-interop layer (conf_mod)" ON)
option(CONF_USE_SYSTEM_YAMLCPP "Use system-installed yaml-cpp (e.g. from Spack) instead of FetchContent" OFF)

if(CONF_USE_SYSTEM_YAMLCPP)
    find_package(yaml-cpp 0.8 REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        0.8.0
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

add_library(conf)
add_library(HELM::CONF ALIAS conf)

target_sources(conf PRIVATE
    src/config.cpp
    src/value.cpp
    src/detail/yaml_tree.cpp
)

if(BUILD_FORTRAN)
    enable_language(Fortran)
    target_sources(conf PRIVATE
        src/fortran/conf_c_interop.cpp
    )

    add_library(conf_fortran fortran/conf_mod.f90)
    add_library(HELM::CONF_Fortran ALIAS conf_fortran)
    target_link_libraries(conf_fortran PUBLIC conf)
    set_target_properties(conf_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran_modules
    )
    target_include_directories(conf_fortran PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran_modules>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran>
    )
endif()

target_include_directories(conf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(conf
    PRIVATE
        yaml-cpp
)

target_compile_features(conf PUBLIC cxx_std_20)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the library binary. When yaml-cpp is built via FetchContent (PRIVATE),
# it must also appear in the export set so CMake install(EXPORT) validation passes.
if(NOT CONF_USE_SYSTEM_YAMLCPP)
    install(TARGETS conf yaml-cpp
        EXPORT CONFTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
else()
    install(TARGETS conf
        EXPORT CONFTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(DIRECTORY include/conf
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "detail" EXCLUDE
)

install(EXPORT CONFTargets
    FILE CONFTargets.cmake
    NAMESPACE HELM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CONF
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CONFConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CONFConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CONF
)

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

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)

    if(BUILD_FORTRAN)
        add_subdirectory(tests_fortran)
    endif()
endif()
