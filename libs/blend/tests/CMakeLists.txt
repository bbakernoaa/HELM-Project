# ##############################################################################
# BLEND Test Suite
#
# GTest unit tests and RapidCheck property tests for LinearBlendKernel and
# StepBlendKernel. Wired into the parent BLEND CMakeLists.txt via the
# BUILD_TESTING gate (add_subdirectory(tests) is guarded by if(BUILD_TESTING)).
#
# CTest labels:
#   unit     — deterministic fixed-input tests (test_linear_blend, test_step_blend)
#   property — randomized property tests      (prop_identity, prop_step_select,
#                                              prop_boundedness)
#
# Requirements: 1.1, 1.10, 1.11
# ##############################################################################

include(GoogleTest)

find_package(GTest REQUIRED)
find_package(rapidcheck REQUIRED)

# ── Unit test targets ─────────────────────────────────────────────────────────

set(BLEND_UNIT_TESTS
    test_linear_blend
    test_step_blend
)

foreach(test_name IN LISTS BLEND_UNIT_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            HELM::BLEND
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    gtest_discover_tests(${test_name}
        PROPERTIES LABELS "unit"
    )
endforeach()

# ── Property test targets (RapidCheck + GTest via RC_GTEST_PROP) ──────────────

set(BLEND_PROPERTY_TESTS
    prop_identity
    prop_step_select
    prop_boundedness
)

foreach(test_name IN LISTS BLEND_PROPERTY_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            HELM::BLEND
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "property"
        TIMEOUT 120
        ENVIRONMENT "RC_PARAMS=max_success=100"
    )
endforeach()
