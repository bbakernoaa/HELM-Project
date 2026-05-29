# ─── CONF Test Infrastructure ────────────────────────────────────────────────
# Builds unit tests (Google Test) and property-based tests (RapidCheck) for the
# CONF micro-library. Unlike HALO, CONF is a serial, node-local concern: it links
# neither Kokkos nor MPI, and every test (unit and property) runs single-process
# with no mpirun launcher. Property tests run >= 100 generated iterations.
#
# Each planned test source is guarded with if(EXISTS ...) so that a test target
# is created the moment its source is authored by a later task; absent sources
# are silently skipped, keeping incremental development green.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Locate Google Test ──────────────────────────────────────────────────────
find_package(GTest REQUIRED)

# ─── Locate RapidCheck (property-based testing) ──────────────────────────────
# Mirrors HALO: prefer a pre-built RapidCheck from a container/CI build cache,
# otherwise fetch and build it privately via FetchContent. A cache directory may
# be supplied via -DCONF_RAPIDCHECK_BUILD_DIR=... ; a couple of common locations
# are probed by default.
set(CONF_RAPIDCHECK_BUILD_DIR "" CACHE PATH
    "Optional path to a pre-built RapidCheck build directory (contains librapidcheck.a)")

set(_conf_rc_build_candidates
    "${CONF_RAPIDCHECK_BUILD_DIR}"
    "/tmp/conf-build/_deps/rapidcheck-build"
    "/tmp/halo-build/_deps/rapidcheck-build"
)
set(_conf_rc_src_candidates
    "/tmp/conf-build/_deps/rapidcheck-src"
    "/tmp/halo-build/_deps/rapidcheck-src"
)

find_package(rapidcheck QUIET
    PATHS ${_conf_rc_build_candidates}
    NO_DEFAULT_PATH
)

set(CONF_RAPIDCHECK_SRC_DIR "")
if(NOT rapidcheck_FOUND AND NOT TARGET rapidcheck)
    # Try to adopt a pre-built static RapidCheck from a build cache.
    set(_conf_rc_imported FALSE)
    foreach(_rc_build IN LISTS _conf_rc_build_candidates)
        if(_rc_build AND EXISTS "${_rc_build}/librapidcheck.a")
            # Derive the matching source dir (sibling "-src" of the "-build" dir).
            string(REGEX REPLACE "-build$" "-src" _rc_src "${_rc_build}")
            add_library(rapidcheck STATIC IMPORTED)
            set_target_properties(rapidcheck PROPERTIES
                IMPORTED_LOCATION "${_rc_build}/librapidcheck.a"
                INTERFACE_INCLUDE_DIRECTORIES
                    "${_rc_src}/include;${_rc_src}/extras/gtest/include"
            )
            set(CONF_RAPIDCHECK_SRC_DIR "${_rc_src}")
            set(_conf_rc_imported TRUE)
            message(STATUS "CONF: using pre-built RapidCheck from ${_rc_build}")
            break()
        endif()
    endforeach()

    if(NOT _conf_rc_imported)
        message(STATUS "CONF: fetching RapidCheck via FetchContent")
        include(FetchContent)
        # Ask RapidCheck to build its Google Test integration (rapidcheck_gtest).
        set(RC_ENABLE_GTEST ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES ""
        )
        FetchContent_MakeAvailable(rapidcheck)
        FetchContent_GetProperties(rapidcheck SOURCE_DIR CONF_RAPIDCHECK_SRC_DIR)
    endif()
endif()

# ─── RapidCheck Google Test extras (rapidcheck_gtest) ────────────────────────
# Provides the RC_GTEST_PROP macro so property checks can be expressed as GTest
# test cases. Built from the RapidCheck source tree if not already present.
if(NOT TARGET rapidcheck_gtest)
    foreach(_rc_src IN LISTS CONF_RAPIDCHECK_SRC_DIR ${_conf_rc_src_candidates})
        if(_rc_src AND EXISTS "${_rc_src}/extras/gtest/CMakeLists.txt")
            add_subdirectory(
                "${_rc_src}/extras/gtest"
                "${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest"
            )
            break()
        endif()
    endforeach()
endif()

# ─── Property-test tuning ────────────────────────────────────────────────────
# Requirement 33.2: each property test runs at least 100 generated iterations.
# RapidCheck reads this from the RC_PARAMS environment variable at runtime.
set(CONF_PROPERTY_TEST_ITERATIONS 100 CACHE STRING
    "Minimum number of generated iterations per CONF property test (RapidCheck max_success)")

# ─── Helper: register a CONF test executable ─────────────────────────────────
# Usage:
#   conf_add_test(<name>
#       SOURCES <src...>
#       LABEL    <unit|property>      # default: unit
#       [PROPERTY]                    # link RapidCheck + run >=100 iterations
#       [PRIVATE_SRC]                 # add the private src/ include path
#       [LIBRARIES <extra libs...>])
#
# Every test links conf + Google Test; property tests additionally link
# RapidCheck (and rapidcheck_gtest when available). Tests are registered with
# CTest, run single-process (no MPI / no mpirun), and tagged with LABELS so the
# suite can be filtered via `ctest -L property` / `ctest -L unit`.
function(conf_add_test TEST_NAME)
    cmake_parse_arguments(ARG "PROPERTY;PRIVATE_SRC" "LABEL" "SOURCES;LIBRARIES" ${ARGN})

    add_executable(${TEST_NAME} ${ARG_SOURCES})

    target_link_libraries(${TEST_NAME}
        PRIVATE
            conf
            GTest::gtest
            GTest::gtest_main
            ${ARG_LIBRARIES}
    )

    if(ARG_PROPERTY)
        target_link_libraries(${TEST_NAME} PRIVATE rapidcheck)
        if(TARGET rapidcheck_gtest)
            target_link_libraries(${TEST_NAME} PRIVATE rapidcheck_gtest)
        endif()
    endif()

    # All tests can see the test directory; the registry tests additionally need
    # the PRIVATE src/ tree because handle_registry.hpp lives in src/fortran/.
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(ARG_PRIVATE_SRC)
        target_include_directories(${TEST_NAME}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src
        )
    endif()

    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)

    # Single-process registration — no MPI, no mpirun.
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)

    if(ARG_LABEL)
        set(_label "${ARG_LABEL}")
    elseif(ARG_PROPERTY)
        set(_label "property")
    else()
        set(_label "unit")
    endif()

    if(ARG_PROPERTY)
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 120
            LABELS "${_label}"
            ENVIRONMENT "RC_PARAMS=max_success=${CONF_PROPERTY_TEST_ITERATIONS}"
        )
    else()
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 60
            LABELS "${_label}"
        )
    endif()
endfunction()

# ─── Unit Tests (Google Test, single-process) ────────────────────────────────
# Each guarded behind if(EXISTS ...) so it builds as soon as its source lands.

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_error_model.cpp")
    conf_add_test(test_error_model
        SOURCES test_error_model.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_config_load.cpp")
    conf_add_test(test_config_load
        SOURCES test_config_load.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_malformed_yaml.cpp")
    conf_add_test(test_malformed_yaml
        SOURCES test_malformed_yaml.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_missing_keys.cpp")
    conf_add_test(test_missing_keys
        SOURCES test_missing_keys.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_type_casting.cpp")
    conf_add_test(test_type_casting
        SOURCES test_type_casting.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_dotted_path.cpp")
    conf_add_test(test_dotted_path
        SOURCES test_dotted_path.cpp
        LABEL   unit
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_value.cpp")
    conf_add_test(test_value
        SOURCES test_value.cpp
        LABEL   unit
    )
endif()

# The handle-registry unit test needs the private src/ include path because the
# registry header lives in src/fortran/handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_handle_registry.cpp")
    conf_add_test(test_handle_registry
        SOURCES     test_handle_registry.cpp
        LABEL       unit
        PRIVATE_SRC
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_c_bridge.cpp")
    conf_add_test(test_c_bridge
        SOURCES test_c_bridge.cpp
        LABEL   unit
    )
endif()

# ─── Property-Based Tests (RapidCheck, single-process, >=100 iterations) ─────

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_round_trip.cpp")
    conf_add_test(prop_round_trip
        SOURCES prop_round_trip.cpp
        PROPERTY
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_type_safety.cpp")
    conf_add_test(prop_type_safety
        SOURCES prop_type_safety.cpp
        PROPERTY
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_path_resolution.cpp")
    conf_add_test(prop_path_resolution
        SOURCES prop_path_resolution.cpp
        PROPERTY
    )
endif()

# The handle-registry property test needs the private src/ include path because
# the registry header lives in src/fortran/handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_handle_registry.cpp")
    conf_add_test(prop_handle_registry
        SOURCES     prop_handle_registry.cpp
        PROPERTY
        PRIVATE_SRC
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_no_leak_c_bridge.cpp")
    conf_add_test(prop_no_leak_c_bridge
        SOURCES prop_no_leak_c_bridge.cpp
        PROPERTY
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_error_lockstep.cpp")
    conf_add_test(prop_error_lockstep
        SOURCES prop_error_lockstep.cpp
        PROPERTY
    )
endif()
