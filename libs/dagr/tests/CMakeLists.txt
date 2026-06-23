# ####################################################################################################################################################
# DAGR test suite — GTest + RapidCheck integration
#
# Unit tests validate specific examples and edge cases.
# Property tests validate universal correctness properties (Design Properties 1–12)
# using RapidCheck for randomized generation.
#
# Requirements satisfied by this file: 13.7, 13.8
# ####################################################################################################################################################

include(GoogleTest)

find_package(GTest REQUIRED)
find_package(rapidcheck REQUIRED)

# ####################################################################################################################################################
# Sibling library dependencies (pre-built static/shared libs for linking tests)
# ####################################################################################################################################################

# conf (static) + yaml-cpp dependency
set(CONF_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../conf/build/libconf.a")
set(YAMLCPP_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../conf/build/_deps/yaml-cpp-build/libyaml-cppd.a")
set(LOGS_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../logs/build/liblogs.so")

# Create imported targets for sibling libraries
if(EXISTS ${CONF_LIB})
    add_library(sibling_conf STATIC IMPORTED)
    set_target_properties(sibling_conf PROPERTIES IMPORTED_LOCATION ${CONF_LIB})
endif()
if(EXISTS ${YAMLCPP_LIB})
    add_library(sibling_yamlcpp STATIC IMPORTED)
    set_target_properties(sibling_yamlcpp PROPERTIES IMPORTED_LOCATION ${YAMLCPP_LIB})
endif()
if(EXISTS ${LOGS_LIB})
    add_library(sibling_logs SHARED IMPORTED)
    set_target_properties(sibling_logs PROPERTIES IMPORTED_LOCATION ${LOGS_LIB})
endif()

set(DAGR_TEST_EXTRA_LIBS "")
if(TARGET sibling_conf)
    list(APPEND DAGR_TEST_EXTRA_LIBS sibling_conf)
endif()
if(TARGET sibling_yamlcpp)
    list(APPEND DAGR_TEST_EXTRA_LIBS sibling_yamlcpp)
endif()
if(TARGET sibling_logs)
    list(APPEND DAGR_TEST_EXTRA_LIBS sibling_logs)
endif()

# ####################################################################################################################################################
# Unit test targets
# ####################################################################################################################################################

set(DAGR_UNIT_TESTS
    test_yaml_parser
    test_rank_pool
    test_transform_node
)

foreach(test_name IN LISTS DAGR_UNIT_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            dagr
            ${DAGR_TEST_EXTRA_LIBS}
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tick/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../conf/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../logs/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../SPAN/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../HALO/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../BLEND/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../AMIO/include
    )
    gtest_discover_tests(${test_name})
endforeach()

# ####################################################################################################################################################
# Property-based test targets (RapidCheck + GTest via RC_GTEST_PROP)
# ####################################################################################################################################################

set(DAGR_PROPERTY_TESTS
    prop_topological_order
    prop_no_orphan
    prop_rank_conservation
    prop_acyclicity_detection
    prop_concurrency_limit
    prop_idempotent_shutdown
    prop_failure_cascade
    prop_yaml_roundtrip
    prop_dag_construction
    prop_invalid_enum
    prop_missing_field
    prop_duplicate_stream
)

foreach(test_name IN LISTS DAGR_PROPERTY_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            dagr
            ${DAGR_TEST_EXTRA_LIBS}
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../tick/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../conf/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../logs/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../SPAN/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../HALO/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../BLEND/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../../AMIO/include
    )
    gtest_discover_tests(${test_name}
        PROPERTIES
            ENVIRONMENT "RC_PARAMS=max_success=1000"
            TIMEOUT 120
    )
endforeach()
