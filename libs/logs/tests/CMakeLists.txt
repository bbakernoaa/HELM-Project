# ##############################################################################
# LOGS Test Suite — CMakeLists.txt
#
# Activated when BUILD_TESTING=ON. Requires Google Test; uses RapidCheck for
# property-based tests. MPI-based tests are executed via mpirun -np 4.
#
# Requirements: 12.6, 12.7, 12.10
# ##############################################################################

# ─── Google Test (REQUIRED) ──────────────────────────────────────────────────
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
    message(FATAL_ERROR
        "LOGS tests require Google Test (GTest). BUILD_TESTING is ON but GTest "
        "was not found. Please install GoogleTest or set GTest_ROOT. "
        "(Requirement 12.10)")
endif()

# ─── RapidCheck (property-based testing) ─────────────────────────────────────
# Try find_package first; fall back to FetchContent if unavailable.
find_package(rapidcheck QUIET)
if(NOT rapidcheck_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    # Disable RapidCheck tests/examples to speed up the build.
    set(RC_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(RC_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(rapidcheck)
endif()

# ─── MPI test runner ─────────────────────────────────────────────────────────
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
if(NOT MPIEXEC_EXECUTABLE)
    # Fall back to CMake's MPI detection (already found at top level).
    set(MPIEXEC_EXECUTABLE "${MPIEXEC}")
endif()
set(LOGS_MPI_TEST_NUMPROCS 4 CACHE STRING "Number of MPI ranks for LOGS tests")

# ─── Helper: add a LOGS unit test ────────────────────────────────────────────
# Usage: logs_add_unit_test(NAME <name> SOURCES <src...> [MPI])
function(logs_add_unit_test)
    cmake_parse_arguments(ARG "MPI" "NAME" "SOURCES" ${ARGN})

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME}
        PRIVATE
            logs
            GTest::gtest_main
            MPI::MPI_CXX
    )
    target_include_directories(${ARG_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}   # For test infrastructure headers
    )

    if(ARG_MPI AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME ${ARG_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} -np ${LOGS_MPI_TEST_NUMPROCS}
                    $<TARGET_FILE:${ARG_NAME}>
        )
    else()
        add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    endif()
endfunction()

# ─── Helper: add a LOGS property test (RapidCheck) ───────────────────────────
# Usage: logs_add_property_test(NAME <name> SOURCES <src...> [MPI])
function(logs_add_property_test)
    cmake_parse_arguments(ARG "MPI" "NAME" "SOURCES" ${ARGN})

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    target_link_libraries(${ARG_NAME}
        PRIVATE
            logs
            GTest::gtest_main
            rapidcheck
            MPI::MPI_CXX
    )
    target_include_directories(${ARG_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}   # For test infrastructure headers
    )

    if(ARG_MPI AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME ${ARG_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} -np ${LOGS_MPI_TEST_NUMPROCS}
                    $<TARGET_FILE:${ARG_NAME}>
        )
    else()
        add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    endif()
endfunction()

# ─── Unit Tests ──────────────────────────────────────────────────────────────
set(LOGS_UNIT_TESTS
    test_severity
    test_rank_stamping
    test_filtering
    test_consolidation
    test_synchronized_abort
    test_sinks
    test_stack_trace
    test_scoped_context
    test_exception_boundary
    test_thread_safety
    test_mpi_spy
)

foreach(test_name IN LISTS LOGS_UNIT_TESTS)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
    if(EXISTS "${_src}")
        logs_add_unit_test(
            NAME ${test_name}
            SOURCES ${_src}
            MPI
        )
    endif()
endforeach()

# test_sinks uses RapidCheck for property-based lifecycle verification.
if(TARGET test_sinks)
    target_link_libraries(test_sinks PRIVATE rapidcheck)
endif()

# test_scoped_context uses RapidCheck for property-based context capture tests.
if(TARGET test_scoped_context)
    target_link_libraries(test_scoped_context PRIVATE rapidcheck)
endif()

# test_stack_trace uses RapidCheck for property-based stack-trace attachment test.
if(TARGET test_stack_trace)
    target_link_libraries(test_stack_trace PRIVATE rapidcheck)
endif()

# test_exception_boundary uses RapidCheck for property-based exception boundary tests.
if(TARGET test_exception_boundary)
    target_link_libraries(test_exception_boundary PRIVATE rapidcheck)
endif()

# test_consolidation uses RapidCheck for property-based consolidation tests.
if(TARGET test_consolidation)
    target_link_libraries(test_consolidation PRIVATE rapidcheck)
endif()

# test_synchronized_abort uses RapidCheck for property-based abort tests.
if(TARGET test_synchronized_abort)
    target_link_libraries(test_synchronized_abort PRIVATE rapidcheck)
    # Link the MPI interposition source for spy-based abort verification (Task 21.4).
    target_sources(test_synchronized_abort PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_interposition.cpp
    )
endif()

# test_rank_stamping provides its own main() for MPI_Init_thread/MPI_Finalize.
# Replace gtest_main with gtest (no default main).
if(TARGET test_rank_stamping)
    get_target_property(_libs test_rank_stamping LINK_LIBRARIES)
    list(REMOVE_ITEM _libs GTest::gtest_main)
    list(APPEND _libs GTest::gtest)
    set_target_properties(test_rank_stamping PROPERTIES LINK_LIBRARIES "${_libs}")
endif()

# prop_rank_stamping provides its own main() for MPI_Init_thread/MPI_Finalize.
# Replace gtest_main with gtest (no default main).
if(TARGET prop_rank_stamping)
    get_target_property(_libs prop_rank_stamping LINK_LIBRARIES)
    list(REMOVE_ITEM _libs GTest::gtest_main)
    list(APPEND _libs GTest::gtest)
    set_target_properties(prop_rank_stamping PROPERTIES LINK_LIBRARIES "${_libs}")
endif()

# ─── Property Tests (RapidCheck) ─────────────────────────────────────────────
set(LOGS_PROPERTY_TESTS
    test_log_record
    prop_severity_ordering
    prop_rank_stamping
    prop_filtering
    prop_consolidation
    prop_rank_range
    prop_stack_trace
    prop_scoped_context
    prop_format_round_trip
    prop_thread_safety
)

foreach(test_name IN LISTS LOGS_PROPERTY_TESTS)
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.cpp")
    if(EXISTS "${_src}")
        logs_add_property_test(
            NAME ${test_name}
            SOURCES ${_src}
            MPI
        )
    endif()
endforeach()

# ─── Static Verification Tests ───────────────────────────────────────────────

# Tier 1 isolation scan (Requirements 14.1, 14.5, 14.6, 14.7)
set(LOGS_TIER1_ISOLATION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/check_tier1_isolation.sh")
if(EXISTS "${LOGS_TIER1_ISOLATION_SCRIPT}")
    add_test(
        NAME tier1_isolation
        COMMAND bash "${LOGS_TIER1_ISOLATION_SCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    set_tests_properties(tier1_isolation PROPERTIES
        LABELS "static-analysis;isolation"
        TIMEOUT 60
    )
endif()

# No-input-parsing scan (Requirements 10.5, 10.6, 10.7, 10.8)
set(LOGS_NO_INPUT_PARSING_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/check_no_input_parsing.sh")
if(EXISTS "${LOGS_NO_INPUT_PARSING_SCRIPT}")
    add_test(
        NAME no_input_parsing_scan
        COMMAND bash "${LOGS_NO_INPUT_PARSING_SCRIPT}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    set_tests_properties(no_input_parsing_scan PROPERTIES
        LABELS "static-analysis;no-parsing"
        TIMEOUT 60
    )
endif()


