# ##############################################################################
# tests/CMakeLists.txt
#
# AXIS test infrastructure. Included from the root CMakeLists.txt only when
# BUILD_TESTING=ON. Provides two test executable targets:
#
#   axis_unit_tests      — GTest-based unit/integration tests
#   axis_property_tests  — RapidCheck property-based tests (≥100 iterations)
#
# Both targets link against HELM::AXIS (via the axis alias), GTest::gtest_main,
# and (for property tests) rapidcheck.
# ##############################################################################

include(GoogleTest)

# ##############################################################################
# Locate test dependencies
# ##############################################################################

find_package(GTest REQUIRED)

find_package(rapidcheck CONFIG QUIET)
if(NOT rapidcheck_FOUND)
    # Fall back to FetchContent (matches HALO's pattern).
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

set(AXIS_UNIT_TEST_SOURCES
    test_conservation.cpp
    test_constant_field.cpp
    test_bilinear_exactness.cpp
    test_spmv_apply.cpp
    test_descriptor_roundtrip.cpp
    test_producer_equivalence.cpp
    test_descriptor_validation.cpp
    test_named_and_rules.cpp
    test_gmsh_roundtrip.cpp
    test_distributed_apply.cpp
    test_conservation_norm.cpp
    test_raii_handles.cpp
)

add_executable(axis_unit_tests ${AXIS_UNIT_TEST_SOURCES})

target_link_libraries(axis_unit_tests
    PRIVATE
        axis
        GTest::gtest_main
)

target_include_directories(axis_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(axis_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

gtest_discover_tests(axis_unit_tests
    PROPERTIES LABELS "unit"
    DISCOVERY_TIMEOUT 30
)

# ##############################################################################
# Property-based test sources (RapidCheck + GTest)
# ##############################################################################

if(rapidcheck_FOUND)
    set(AXIS_PROPERTY_TEST_SOURCES
        prop_field_view_roundtrip.cpp
        prop_raii_handles.cpp
        prop_structured_to_unstructured.cpp
        prop_csr_connectivity.cpp
        prop_named_grid_deterministic.cpp
        prop_named_grid_registration.cpp
        prop_rule_generation.cpp
        prop_descriptor_validation.cpp
        prop_producer_equivalence.cpp
        prop_buffer_adoption.cpp
        prop_sparse_matrix_bounds.cpp
        prop_conservative_nonneg.cpp
        prop_partition_of_unity.cpp
        prop_conservation.cpp
        prop_bilinear_exactness.cpp
        prop_spmv_apply.cpp
        prop_apply_extent_validation.cpp
        prop_dstarea_norm.cpp
        prop_fracarea_norm.cpp
        prop_unmapped_handling.cpp
        prop_coordsystem_consistency.cpp
        prop_halo_pattern.cpp
        prop_gmsh_roundtrip.cpp
        prop_descriptor_roundtrip.cpp
        prop_host_device_equivalence.cpp
    )

    add_executable(axis_property_tests ${AXIS_PROPERTY_TEST_SOURCES})

    target_link_libraries(axis_property_tests
        PRIVATE
            axis
            GTest::gtest_main
            rapidcheck
            $<TARGET_NAME_IF_EXISTS:rapidcheck_gtest>
    )

    target_include_directories(axis_property_tests
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_target_properties(axis_property_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    gtest_discover_tests(axis_property_tests
        PROPERTIES LABELS "property"
        DISCOVERY_TIMEOUT 60
    )
else()
    message(STATUS "AXIS: RapidCheck not found — skipping property-based tests")
endif()
