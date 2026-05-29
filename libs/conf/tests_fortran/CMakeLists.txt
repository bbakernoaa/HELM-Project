# ─── CONF Fortran Integration Test ───────────────────────────────────────────
# Builds the end-to-end Fortran test that drives CONF through the conf_mod
# wrapper (Fortran -> iso_c_binding -> extern "C" bridge -> yaml-cpp), proving
# the full path including leak-free length-first string marshalling.
#
# This directory is only added by the root CMakeLists when BOTH BUILD_FORTRAN
# and BUILD_TESTING are ON (Requirement 25.5), so no option check is repeated
# here. The executable itself is guarded so it is created only once its source
# (authored by task 8.2) and the conf_mod module target (authored by task 8.1)
# both exist. Like the C++ property tests, it runs single-process — no MPI and
# no mpirun launcher.
# ──────────────────────────────────────────────────────────────────────────────

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_conf_fortran.f90" AND TARGET conf_mod)
    add_executable(test_conf_fortran test_conf_fortran.f90)

    # Link the C++ library (carrying the extern "C" bridge) and the conf_mod
    # Fortran module objects. conf_mod is expected to be an OBJECT/static target
    # exposing the .mod files via its usage requirements.
    target_link_libraries(test_conf_fortran
        PRIVATE
            conf
            conf_mod
    )

    # The fixture YAML lives beside the test source; expose its directory so the
    # test can locate the file at runtime.
    target_compile_definitions(test_conf_fortran
        PRIVATE CONF_FORTRAN_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(NAME test_conf_fortran COMMAND $<TARGET_FILE:test_conf_fortran>)
    set_tests_properties(test_conf_fortran PROPERTIES
        TIMEOUT 60
        LABELS "fortran;integration"
        # Run from the source dir so a relative fixture path also resolves.
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
