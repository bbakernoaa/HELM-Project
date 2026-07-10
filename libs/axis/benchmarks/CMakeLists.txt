# ##############################################################################
# benchmarks/CMakeLists.txt
#
# AXIS performance benchmark harness. Included from the root CMakeLists.txt
# only when AXIS_BUILD_BENCHMARKS=ON. Produces a single executable:
#
#   axis_benchmarks — runs all benchmark cases and outputs JSON results
#
# Uses Kokkos::HostSpace for reproducible timing across CI environments
# (Requirement 13.5).
# ##############################################################################

add_executable(axis_benchmarks axis_benchmark.cpp)

target_link_libraries(axis_benchmarks
    PRIVATE
        axis
)

target_include_directories(axis_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(axis_benchmarks PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
