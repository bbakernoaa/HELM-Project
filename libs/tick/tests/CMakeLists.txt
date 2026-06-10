# TICK test suite — GTest + RapidCheck integration
# Parent CMakeLists.txt handles find_package(GTest) and find_package(rapidcheck)

include(GoogleTest)

# List of test source files (one executable per file)
set(TICK_TEST_SOURCES
    test_duration
    test_gregorian
    test_noleap
    test_cal360
    test_alarm
    test_sync
    test_window
    test_integration
)

foreach(test_name IN LISTS TICK_TEST_SOURCES)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name}
        PRIVATE
            HELM::TICK
            GTest::gtest_main
            rapidcheck
    )
    target_include_directories(${test_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    gtest_discover_tests(${test_name})
endforeach()

# ── C API tests (test via tick_c.h only, linked against tick_c) ──
add_executable(test_tick_c test_tick_c.cpp)
target_link_libraries(test_tick_c
    PRIVATE
        HELM::TICK_C
        GTest::gtest_main
        rapidcheck
)
target_include_directories(test_tick_c
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
gtest_discover_tests(test_tick_c)

# ── Fortran integration test (gated by TICK_BUILD_FORTRAN) ──
if(TICK_BUILD_FORTRAN)
    add_executable(test_tick_fortran test_tick_fortran.f90)
    target_link_libraries(test_tick_fortran
        PRIVATE
            HELM::TICK_Fortran
    )
    add_test(NAME test_tick_fortran COMMAND test_tick_fortran)
endif()
