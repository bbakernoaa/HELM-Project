# ─── HALO Fortran Integration Tests ──────────────────────────────────────────
# Builds the end-to-end Fortran integration test (test_halo_mod.f90) that drives
# the public `halo_mod` API against a REAL 4-rank MPI run. This complements the
# single-rank, mocked-MPI C++ property tests by validating the iso_c_binding
# wrapper, the C interop layer, and the core HALO library together under mpirun.
#
# Added by task 18.1. The parent CMakeLists only descends here when
# BUILD_TESTING AND BUILD_FORTRAN are both ON.
# ──────────────────────────────────────────────────────────────────────────────

# ─── MPI Fortran component ────────────────────────────────────────────────────
# The top-level project finds MPI with the CXX component; the Fortran test needs
# the Fortran component, which provides the MPI::MPI_Fortran target and the
# `use mpi` module (mpifort wrapper) consumed by the test program.
find_package(MPI REQUIRED COMPONENTS Fortran)

# ─── MPI launcher ─────────────────────────────────────────────────────────────
# Mirror the convention used by tests/CMakeLists.txt: locate mpirun/mpiexec and
# launch with --oversubscribe -np 4 (the container has limited slots).
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "mpirun/mpiexec not found. The Fortran MPI test will not be runnable.")
endif()

set(HALO_FORTRAN_TEST_NUMPROCS 4 CACHE STRING "Number of MPI ranks for the HALO Fortran integration test")

# ─── Test executable ──────────────────────────────────────────────────────────
add_executable(test_halo_mod test_halo_mod.f90)

# Linking HELM::HALO_Fortran transitively brings:
#   - halo_fortran  : the halo_mod module library + its module dir on the
#                     include interface (so halo_mod.mod is discoverable)
#   - halo_c_interop: the extern "C" symbols resolved at link time
#   - halo          : the core C++ library
#   - MPI::MPI_CXX + Kokkos::kokkos (PUBLIC deps of halo)
# MPI::MPI_Fortran provides the `use mpi` module and the Fortran MPI bindings
# (MPI_Init, MPI_Comm_rank, MPI_Allreduce, ...) used directly by the test.
target_link_libraries(test_halo_mod
    PRIVATE
        HELM::HALO_Fortran
        MPI::MPI_Fortran
)

# ─── CTest registration ───────────────────────────────────────────────────────
add_test(
    NAME test_halo_mod
    COMMAND ${MPIEXEC_EXECUTABLE}
            --oversubscribe
            -np ${HALO_FORTRAN_TEST_NUMPROCS}
            $<TARGET_FILE:test_halo_mod>
)
set_tests_properties(test_halo_mod PROPERTIES
    TIMEOUT 60
    LABELS "mpi;fortran"
)
