# ─── HALO Test Infrastructure ─────────────────────────────────────────────────
# Builds unit tests (GTest) and property-based tests (RapidCheck) for the HALO
# micro-library. MPI-based tests run with mpirun -np 4; property tests run
# single-rank with mocked MPI via the interposition spy layer.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Locate Google Test ──────────────────────────────────────────────────────
find_package(GTest REQUIRED)

# ─── Locate RapidCheck (property-based testing) ──────────────────────────────
find_package(rapidcheck QUIET
    PATHS /tmp/halo-build/_deps/rapidcheck-build
    NO_DEFAULT_PATH
)
if(NOT rapidcheck_FOUND)
    # Use the pre-built RapidCheck from the container build cache
    if(EXISTS "/tmp/halo-build/_deps/rapidcheck-build/librapidcheck.a")
        add_library(rapidcheck STATIC IMPORTED)
        set_target_properties(rapidcheck PROPERTIES
            IMPORTED_LOCATION "/tmp/halo-build/_deps/rapidcheck-build/librapidcheck.a"
            INTERFACE_INCLUDE_DIRECTORIES "/tmp/halo-build/_deps/rapidcheck-src/include;/tmp/halo-build/_deps/rapidcheck-src/extras/gtest/include"
        )
        message(STATUS "Using pre-built RapidCheck from /tmp/halo-build/_deps/")
    else()
        # Fall back to FetchContent if not found
        include(FetchContent)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
            GIT_SUBMODULES ""
        )
        FetchContent_MakeAvailable(rapidcheck)
    endif()
endif()

# ─── RapidCheck GTest extras ─────────────────────────────────────────────────
# Build the rapidcheck_gtest library if not already available
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "/tmp/halo-build/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            /tmp/halo-build/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    elseif(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
        add_subdirectory(
            ${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest
            ${CMAKE_CURRENT_BINARY_DIR}/rapidcheck_gtest
        )
    endif()
endif()

# ─── Find MPI runtime for test execution ─────────────────────────────────────
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "mpirun/mpiexec not found. MPI-based tests will not be runnable.")
endif()

set(HALO_MPI_TEST_NUMPROCS 4 CACHE STRING "Number of MPI ranks for HALO MPI tests")

# ─── Helper function: Add an MPI-based unit test ─────────────────────────────
function(halo_add_mpi_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})

    add_executable(${TEST_NAME} ${ARG_SOURCES})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            MPI::MPI_CXX
            ${ARG_LIBRARIES}
    )
    target_include_directories(${TEST_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Register as CTest with mpirun launcher
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE}
                --oversubscribe
                -np ${HALO_MPI_TEST_NUMPROCS}
                $<TARGET_FILE:${TEST_NAME}>
    )
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 60
        LABELS "mpi;unit"
    )
endfunction()

# ─── MPI Interposition Spy Library (linked into property tests) ───────────────
# Provides link-time overrides for MPI functions, enabling deterministic testing
# of RAII destructor behavior without relying on MPI runtime side effects.
add_library(halo_mpi_spy STATIC
    mpi_interposition.cpp
)
target_include_directories(halo_mpi_spy
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(halo_mpi_spy
    PUBLIC MPI::MPI_CXX
)
target_compile_features(halo_mpi_spy PUBLIC cxx_std_20)

# ─── Helper function: Add a property-based test (single-rank, mocked MPI) ───
function(halo_add_property_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES" ${ARGN})

    add_executable(${TEST_NAME} ${ARG_SOURCES})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            halo
            halo_mpi_spy
            GTest::gtest
            GTest::gtest_main
            rapidcheck
            MPI::MPI_CXX
            ${ARG_LIBRARIES}
    )
    # Add rapidcheck_gtest if available
    if(TARGET rapidcheck_gtest)
        target_link_libraries(${TEST_NAME} PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(${TEST_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Property tests run single-rank with mocked MPI (no mpirun needed)
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 120
        LABELS "property"
    )
endfunction()

# ─── Unit Test Targets (MPI-based, run with mpirun -np 4) ────────────────────
# Only build tests whose source files exist (incremental development)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_communicator.cpp")
    halo_add_mpi_test(test_communicator
        SOURCES test_communicator.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_request_guard.cpp")
    halo_add_mpi_test(test_request_guard
        SOURCES test_request_guard.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_window_guard.cpp")
    halo_add_mpi_test(test_window_guard
        SOURCES test_window_guard.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_halo_plan.cpp")
    halo_add_mpi_test(test_halo_plan
        SOURCES test_halo_plan.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange.cpp")
    halo_add_mpi_test(test_exchange
        SOURCES test_exchange.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_raii_exceptions.cpp")
    halo_add_property_test(test_raii_exceptions
        SOURCES test_raii_exceptions.cpp
    )
endif()

# ─── MPI Spy Unit Test (single-rank, mocked MPI) ─────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_mpi_spy.cpp")
    halo_add_property_test(test_mpi_spy
        SOURCES test_mpi_spy.cpp
    )
endif()

# ─── Property-Based Test Targets (single-rank, mocked MPI) ───────────────────

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_communicator.cpp")
    halo_add_property_test(prop_communicator
        SOURCES prop_communicator.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_request_guard.cpp")
    halo_add_property_test(prop_request_guard
        SOURCES prop_request_guard.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_window_guard.cpp")
    halo_add_property_test(prop_window_guard
        SOURCES prop_window_guard.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_halo_plan.cpp")
    halo_add_property_test(prop_halo_plan
        SOURCES prop_halo_plan.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange.cpp")
    halo_add_property_test(prop_exchange
        SOURCES prop_exchange.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_environment.cpp")
    halo_add_property_test(prop_environment
        SOURCES prop_environment.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_gpu_dispatch.cpp")
    halo_add_property_test(prop_gpu_dispatch
        SOURCES prop_gpu_dispatch.cpp
    )
endif()

# ─── Handle Registry Unit Test (no MPI needed, single-rank) ──────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_handle_registry.cpp")
    add_executable(test_handle_registry test_handle_registry.cpp)
    target_link_libraries(test_handle_registry
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )
    target_include_directories(test_handle_registry
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    target_compile_features(test_handle_registry PUBLIC cxx_std_20)
    add_test(NAME test_handle_registry COMMAND $<TARGET_FILE:test_handle_registry>)
    set_tests_properties(test_handle_registry PROPERTIES
        TIMEOUT 30
        LABELS "unit"
    )
endif()
