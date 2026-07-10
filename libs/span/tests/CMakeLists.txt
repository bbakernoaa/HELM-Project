# ##############################################################################
# tests/CMakeLists.txt
#
# SPAN test infrastructure. Included from the root CMakeLists.txt only when
# SPAN_BUILD_TESTING=ON. Provides two test executable targets:
#
#   span_unit_tests      — GTest-based unit/integration tests
#   span_property_tests  — RapidCheck property-based tests (≥100 iterations)
#
# Both targets link against HELM::SPAN (via the span alias), GTest::gtest_main,
# and (for property tests) rapidcheck.
# ##############################################################################

include(GoogleTest)

# ##############################################################################
# Locate test dependencies
# ##############################################################################

find_package(GTest REQUIRED)

find_package(rapidcheck CONFIG QUIET)
if(NOT rapidcheck_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
        GIT_SUBMODULES ""
    )
    set(RC_ENABLE_GTEST ON CACHE BOOL "Build RapidCheck GTest integration" FORCE)
    FetchContent_MakeAvailable(rapidcheck)
    set(rapidcheck_FOUND TRUE)
endif()

# Ensure rapidcheck_gtest target is available
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            ${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    endif()
endif()

# ##############################################################################
# Unit test sources (GTest)
# ##############################################################################

set(SPAN_UNIT_TEST_SOURCES
    test_fieldview_unit.cpp
    test_triple_buffer_threading.cpp
    test_c_api_smoke.cpp
)

add_executable(span_unit_tests ${SPAN_UNIT_TEST_SOURCES})

target_link_libraries(span_unit_tests
    PRIVATE
        span
        GTest::gtest_main
)

target_include_directories(span_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

set_target_properties(span_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(span_unit_tests
    PROPERTIES LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

# ##############################################################################
# Property-based test sources (RapidCheck + GTest)
# ##############################################################################

if(rapidcheck_FOUND)
    set(SPAN_PROPERTY_TEST_SOURCES
        prop_construction_roundtrip.cpp
        prop_coherency_state_machine.cpp
        prop_sync_data_integrity.cpp
        prop_triple_buffer_construction.cpp
        prop_triple_buffer_rotation.cpp
        prop_kokkos_adapter_roundtrip.cpp
        prop_handle_registry.cpp
        prop_c_api_roundtrip.cpp
        prop_c_api_error_priority.cpp
    )

    add_executable(span_property_tests ${SPAN_PROPERTY_TEST_SOURCES})

    target_link_libraries(span_property_tests
        PRIVATE
            span
            GTest::gtest_main
            rapidcheck
            $<TARGET_NAME_IF_EXISTS:rapidcheck_gtest>
    )

    target_include_directories(span_property_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )

    set_target_properties(span_property_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    gtest_discover_tests(span_property_tests
        PROPERTIES LABELS "property"
        DISCOVERY_TIMEOUT 60
    )
else()
    message(STATUS "SPAN: RapidCheck not found — skipping property-based tests")
endif()
