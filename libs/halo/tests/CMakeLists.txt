# ─── HALO Test Infrastructure ─────────────────────────────────────────────────
# Builds unit tests (GTest) and property-based tests (RapidCheck) for the HALO
# micro-library. MPI-based tests run with mpirun -np 4; property tests run
# single-rank with mocked MPI via the interposition spy layer.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Locate Google Test ──────────────────────────────────────────────────────
find_package(GTest REQUIRED)

# ─── Locate RapidCheck (property-based testing) ──────────────────────────────
# RapidCheck is installed globally in the HELM Docker container (see Dockerfile),
# which provides both the `rapidcheck` and `rapidcheck_gtest` targets.
find_package(rapidcheck QUIET)
if(NOT rapidcheck_FOUND)
    # Fall back to FetchContent for environments without a global install.
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
endif()

# ─── RapidCheck GTest extras ─────────────────────────────────────────────────
# The global RapidCheck install (RC_ENABLE_GTEST=ON) already provides the
# rapidcheck_gtest target via find_package. If it is missing (e.g. an older
# install), build it from the FetchContent source tree.
if(NOT TARGET rapidcheck_gtest)
    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/rapidcheck-src/extras/gtest")
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
    # test_window_guard exercises real MPI_Win one-sided RMA (MPI_Win_create +
    # MPI_Put). In this container's Open MPI 4.1 build, the default one-sided
    # (osc) component selection fails MPI_Win_create with MPI_ERR_WIN on
    # MPI_COMM_SELF; the pt2pt component handles it correctly. Pin osc=pt2pt for
    # just this test so the RMA window lifecycle can be validated. This is an
    # MPI-runtime configuration detail, not a HALO requirement.
    set_property(TEST test_window_guard APPEND PROPERTY
        ENVIRONMENT "OMPI_MCA_osc=pt2pt"
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_halo_plan.cpp")
    halo_add_mpi_test(test_halo_plan
        SOURCES test_halo_plan.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_indexed_halo_plan.cpp")
    halo_add_mpi_test(test_indexed_halo_plan
        SOURCES test_indexed_halo_plan.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange.cpp")
    halo_add_mpi_test(test_exchange
        SOURCES test_exchange.cpp
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_persistent_handle.cpp")
    halo_add_mpi_test(test_persistent_handle
        SOURCES test_persistent_handle.cpp
        LIBRARIES Kokkos::kokkos
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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_indexed_halo_plan.cpp")
    halo_add_property_test(prop_indexed_halo_plan
        SOURCES prop_indexed_halo_plan.cpp
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

# ─── C-Interop Exception Boundary Property Test ──────────────────────────────
# Exercises the extern "C" interop layer (src/fortran/halo_c_interop.cpp), which
# is NOT compiled into the `halo` library target, so the interop translation
# unit is added directly to this test target's sources. The src/fortran include
# directory is added so the test can reach handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exception_boundary.cpp")
    halo_add_property_test(prop_exception_boundary
        SOURCES
            prop_exception_boundary.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran/halo_c_interop.cpp
    )
    target_include_directories(prop_exception_boundary
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── C-Interop Destroy Invalidates Handle Property Test ──────────────────────
# Exercises the extern "C" interop layer (src/fortran/halo_c_interop.cpp), which
# is NOT compiled into the `halo` library target, so the interop translation
# unit is added directly to this test target's sources. The src/fortran include
# directory is added so the test can reach handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_destroy_handle.cpp")
    halo_add_property_test(prop_destroy_handle
        SOURCES
            prop_destroy_handle.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran/halo_c_interop.cpp
    )
    target_include_directories(prop_destroy_handle
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── C-Interop Plan Creation Validates Neighbor Arrays Property Test ──────────
# Exercises the extern "C" interop layer (src/fortran/halo_c_interop.cpp), which
# is NOT compiled into the `halo` library target, so the interop translation
# unit is added directly to this test target's sources. The src/fortran include
# directory is added so the test can reach handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_plan_validation.cpp")
    halo_add_property_test(prop_plan_validation
        SOURCES
            prop_plan_validation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran/halo_c_interop.cpp
    )
    target_include_directories(prop_plan_validation
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── C-Interop Exchange Forwarding Preserves Pointer and Size Property Test ───
# Exercises the extern "C" interop layer (src/fortran/halo_c_interop.cpp), which
# is NOT compiled into the `halo` library target, so the interop translation
# unit is added directly to this test target's sources. The src/fortran include
# directory is added so the test can reach handle_registry.hpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_forwarding.cpp")
    halo_add_property_test(prop_exchange_forwarding
        SOURCES
            prop_exchange_forwarding.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran/halo_c_interop.cpp
    )
    target_include_directories(prop_exchange_forwarding
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
endif()

# ─── Pack/Unpack Kernel Unit Test (no MPI needed, single-rank, Kokkos) ───────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pack_unpack.cpp")
    add_executable(test_pack_unpack test_pack_unpack.cpp)
    target_link_libraries(test_pack_unpack
        PRIVATE
            halo
            GTest::gtest
            Kokkos::kokkos
    )
    target_include_directories(test_pack_unpack
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(test_pack_unpack PUBLIC cxx_std_20)
    add_test(NAME test_pack_unpack COMMAND $<TARGET_FILE:test_pack_unpack>)
    set_tests_properties(test_pack_unpack PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endif()

# ─── Structured Halo Plan Unit Test (MPI-based) ─────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_structured_halo_plan.cpp")
    halo_add_mpi_test(test_structured_halo_plan
        SOURCES test_structured_halo_plan.cpp
        LIBRARIES Kokkos::kokkos
    )
endif()

# ─── Structured Exchange Integration Test (MPI-based) ────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_structured_exchange.cpp")
    halo_add_mpi_test(test_structured_exchange
        SOURCES test_structured_exchange.cpp
        LIBRARIES Kokkos::kokkos
    )
endif()

# ─── Neighbor Collective Exchange Integration Test (MPI-based) ────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_neighbor_collective.cpp")
    halo_add_mpi_test(test_neighbor_collective
        SOURCES test_neighbor_collective.cpp
        LIBRARIES Kokkos::kokkos
    )
endif()

# ─── Error Messages Unit Test (MPI-based) ────────────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_error_messages.cpp")
    halo_add_mpi_test(test_error_messages
        SOURCES test_error_messages.cpp
    )
endif()

# ─── Structured Exchange Property Test (MPI-based + RapidCheck) ──────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_structured_exchange.cpp")
    halo_add_mpi_test(prop_structured_exchange
        SOURCES prop_structured_exchange.cpp
        LIBRARIES Kokkos::kokkos rapidcheck
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_structured_exchange PRIVATE rapidcheck_gtest)
    endif()
    set_tests_properties(prop_structured_exchange PROPERTIES
        TIMEOUT 120
        LABELS "mpi;property"
    )
endif()

# ─── Diagnostics Hook Unit Test (MPI-based) ──────────────────────────────────
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_diagnostics.cpp")
    halo_add_mpi_test(test_diagnostics
        SOURCES test_diagnostics.cpp
        LIBRARIES Kokkos::kokkos
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

# ─── Handle Registry Property Test (no MPI needed, single-rank) ──────────────
# The registry is header-only (src/fortran/handle_registry.hpp), so this target
# does not link the halo library or MPI -- only GTest plus the RapidCheck
# property framework.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_handle_registry.cpp")
    add_executable(prop_handle_registry prop_handle_registry.cpp)
    target_link_libraries(prop_handle_registry
        PRIVATE
            GTest::gtest
            GTest::gtest_main
            rapidcheck
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_handle_registry PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(prop_handle_registry
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/fortran
    )
    target_compile_features(prop_handle_registry PUBLIC cxx_std_20)
    add_test(NAME prop_handle_registry COMMAND $<TARGET_FILE:prop_handle_registry>)
    set_tests_properties(prop_handle_registry PROPERTIES
        TIMEOUT 120
        LABELS "property"
    )
endif()
