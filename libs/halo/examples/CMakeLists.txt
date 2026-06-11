# ─────────────────────────────────────────────────────────────────────────────
# Standalone build for HALO adapter examples.
#
# Prerequisites:
#   - HALO installed (or HALO_ROOT / CMAKE_PREFIX_PATH pointing to install tree)
#   - MPI (found automatically via HALO's transitive dependencies)
#   - Kokkos (found automatically via HALO's transitive dependencies)
#
# Build instructions:
#
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH="<halo-install>;/path/to/kokkos/install"
#   cmake --build .
#
# Run:
#   mpirun -np 4 ./mpas_adapter
#
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.20)
project(halo_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HALO — pulls in MPI and Kokkos transitively.
find_package(HALO REQUIRED)
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# ─── MPAS adapter example ────────────────────────────────────────────────────
add_executable(mpas_adapter mpas_adapter.cpp)
target_link_libraries(mpas_adapter PRIVATE HALO::halo Kokkos::kokkos MPI::MPI_CXX)
