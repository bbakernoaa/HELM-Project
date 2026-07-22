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

# ─── Indexed Exchange Property Test (single-process, real MPI + RapidCheck) ───
# exchange_indexed performs a real gather/communicate/scatter round-trip, so it
# needs the real MPI runtime (NOT the interposition spy, which records but does
# not move data). The test runs in a single process using a self-send plan (the
# rank's own send indices feed its own recv indices), matching the design's
# in-process paired send/recv index model, so no mpirun launch is required.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_indexed.cpp")
    add_executable(prop_exchange_indexed prop_exchange_indexed.cpp)
    target_link_libraries(prop_exchange_indexed
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            rapidcheck
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_exchange_indexed PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(prop_exchange_indexed
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(prop_exchange_indexed PUBLIC cxx_std_20)
    add_test(NAME prop_exchange_indexed COMMAND $<TARGET_FILE:prop_exchange_indexed>)
    set_tests_properties(prop_exchange_indexed PROPERTIES
        TIMEOUT 120
        LABELS "property"
    )
endif()

# ─── Indexed Exchange No-Op Property Test (single-rank, mocked MPI) ───────────
# Property 4: an empty or neighborless plan leaves the field view bitwise
# unchanged. A no-op transfers zero elements, so the mocked-MPI single-process
# harness validates it deterministically for rank-1 and rank-2 host views.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_indexed_noop.cpp")
    halo_add_property_test(prop_exchange_indexed_noop
        SOURCES prop_exchange_indexed_noop.cpp
        LIBRARIES Kokkos::kokkos
    )
endif()

# ─── Indexed Exchange Round-Trip Property Test (single-process, real MPI) ─────
# Property 3: after an indexed exchange each receive index holds the paired send
# index's source-owned value (rank-1 and rank-2 LayoutLeft fields; all levels).
# Runs single-process with real MPI using an in-process paired self-send index
# model, so no mpirun launch is required.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_indexed_roundtrip.cpp")
    add_executable(prop_exchange_indexed_roundtrip prop_exchange_indexed_roundtrip.cpp)
    target_link_libraries(prop_exchange_indexed_roundtrip
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            rapidcheck
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_exchange_indexed_roundtrip PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(prop_exchange_indexed_roundtrip
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(prop_exchange_indexed_roundtrip PUBLIC cxx_std_20)
    add_test(NAME prop_exchange_indexed_roundtrip COMMAND $<TARGET_FILE:prop_exchange_indexed_roundtrip>)
    set_tests_properties(prop_exchange_indexed_roundtrip PROPERTIES
        TIMEOUT 120
        LABELS "property"
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

# ─── Indexed Exchange Neighbor-Order Independence Property Test (MPI-based) ────
# Property 8: permuting the order of a plan's neighbors (and thus the order in
# which sends/recvs are posted) produces identical field-view results. This
# property is only meaningful with more than one neighbor, which a single-process
# plan cannot provide (Indexed_Halo_Plan rejects duplicate ranks), so it runs
# over the real multi-rank MPI runtime (mpirun -np 4), where each rank has
# several distinct neighbors and a genuine neighbor permutation can be applied.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_indexed_neighbor_order.cpp")
    halo_add_mpi_test(prop_exchange_indexed_neighbor_order
        SOURCES prop_exchange_indexed_neighbor_order.cpp
        LIBRARIES Kokkos::kokkos rapidcheck
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_exchange_indexed_neighbor_order PRIVATE rapidcheck_gtest)
    endif()
    set_tests_properties(prop_exchange_indexed_neighbor_order PROPERTIES
        TIMEOUT 120
        LABELS "mpi;property"
    )
endif()

# ─── Indexed Exchange Idempotence Property Test (single-process, real MPI) ─────
# Property 9: running exchange_indexed twice with the owned values held fixed
# between the two calls yields the same field state as running it once (the
# second exchange does not change the halo values produced by the first). Runs
# single-process with real MPI using an in-process paired self-send index model
# (like prop_exchange_indexed_roundtrip), so no mpirun launch is required.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prop_exchange_indexed_idempotence.cpp")
    add_executable(prop_exchange_indexed_idempotence prop_exchange_indexed_idempotence.cpp)
    target_link_libraries(prop_exchange_indexed_idempotence
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            rapidcheck
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(prop_exchange_indexed_idempotence PRIVATE rapidcheck_gtest)
    endif()
    target_include_directories(prop_exchange_indexed_idempotence
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(prop_exchange_indexed_idempotence PUBLIC cxx_std_20)
    add_test(NAME prop_exchange_indexed_idempotence COMMAND $<TARGET_FILE:prop_exchange_indexed_idempotence>)
    set_tests_properties(prop_exchange_indexed_idempotence PROPERTIES
        TIMEOUT 120
        LABELS "property"
    )
endif()

# ─── Indexed Exchange Infrastructure Unit Test (single-process, real MPI) ─────
# Task 2.7 (real value-movement checks): diagnostics begin/end neighbor count
# and byte volume (Req 4.3), tag equivalence with the contiguous exchange for
# matching rank pairs (Req 2.6), and the single-rank no-neighbor no-op that
# leaves all elements unchanged (Req 6.3). These need genuine data movement, so
# the target links the real MPI runtime and runs single-process using an
# in-process self-send plan (no mpirun launch required).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange_indexed_infra.cpp")
    add_executable(test_exchange_indexed_infra test_exchange_indexed_infra.cpp)
    target_link_libraries(test_exchange_indexed_infra
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    target_include_directories(test_exchange_indexed_infra
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(test_exchange_indexed_infra PUBLIC cxx_std_20)
    add_test(NAME test_exchange_indexed_infra COMMAND $<TARGET_FILE:test_exchange_indexed_infra>)
    set_tests_properties(test_exchange_indexed_infra PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endif()

# ─── Indexed Exchange MPI-Failure Unit Test (single-rank, mocked MPI spy) ─────
# Task 2.7 (error-path check): an MPI operation failure during exchange_indexed
# raises an error identifying the failing operation and neighbor rank (Req 2.5).
# This needs deterministic error injection, so the target links the MPI
# interposition spy (halo_mpi_spy) instead of the real MPI runtime and runs
# single-rank (no mpirun launch required).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange_indexed_infra_mpifail.cpp")
    add_executable(test_exchange_indexed_infra_mpifail test_exchange_indexed_infra_mpifail.cpp)
    target_link_libraries(test_exchange_indexed_infra_mpifail
        PRIVATE
            halo
            halo_mpi_spy
            GTest::gtest
            GTest::gtest_main
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    target_include_directories(test_exchange_indexed_infra_mpifail
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(test_exchange_indexed_infra_mpifail PUBLIC cxx_std_20)
    add_test(NAME test_exchange_indexed_infra_mpifail COMMAND $<TARGET_FILE:test_exchange_indexed_infra_mpifail>)
    set_tests_properties(test_exchange_indexed_infra_mpifail PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endif()

# ─── Indexed Exchange Multi-Rank Integration Test (MPI-based, np=4) ───────────
# Task 4.1: exercises exchange_indexed on a small synthetic multi-rank partition
# over the real MPI runtime (mpirun -np 4). Owned values are seeded as an
# analytic function of (rank, index); each halo cell is verified against the
# source-owned value on the neighbor that owns it, for rank-1 and rank-2
# (nVertLevels x nElements, LayoutLeft) fields. This validates the genuine
# gather / Irecv+Isend+Waitall / scatter path (Property 3 over real MPI).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange_indexed_multirank.cpp")
    halo_add_mpi_test(test_exchange_indexed_multirank
        SOURCES test_exchange_indexed_multirank.cpp
        LIBRARIES Kokkos::kokkos
    )
endif()

# ─── Indexed Exchange Device-View Round-Trip Test (single-process, real MPI) ──
# Task 4.2: exercises exchange_indexed against a field view in the default
# execution space's memory space (Kokkos::DefaultExecutionSpace::memory_space).
# This drives the compile-time memory-space dispatch: on a device build without
# GPU-aware MPI it exercises the host-staged path (Req 3.3); the GPU-aware direct
# path (Req 3.4) is asserted only when Environment::is_gpu_aware_mpi() is true
# and skipped gracefully otherwise (e.g. on this host-only container, where it
# exercises the host-direct path). The gather/scatter kernels run in the view's
# native execution space (Req 4.1). Covers rank-1 and rank-2 (nVertLevels x
# nElements, LayoutLeft) views. Runs single-process with real MPI using an
# in-process paired self-send index model (like prop_exchange_indexed_roundtrip),
# so no mpirun launch is required.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_exchange_indexed_device.cpp")
    add_executable(test_exchange_indexed_device test_exchange_indexed_device.cpp)
    target_link_libraries(test_exchange_indexed_device
        PRIVATE
            halo
            GTest::gtest
            GTest::gtest_main
            MPI::MPI_CXX
            Kokkos::kokkos
    )
    target_include_directories(test_exchange_indexed_device
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_features(test_exchange_indexed_device PUBLIC cxx_std_20)
    add_test(NAME test_exchange_indexed_device COMMAND $<TARGET_FILE:test_exchange_indexed_device>)
    set_tests_properties(test_exchange_indexed_device PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endif()
