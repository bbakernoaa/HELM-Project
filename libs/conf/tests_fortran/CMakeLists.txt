# ─── CONF Fortran Integration Tests ──────────────────────────────────────────
# Builds the end-to-end Fortran integration test (test_conf_fortran.f90) that
# drives the public `conf_mod` API. Unlike HALO, CONF is single-process — no
# MPI launcher is needed.
#
# The parent CMakeLists only descends here when CONF_BUILD_TESTING AND CONF_BUILD_FORTRAN
# are both ON.
# ──────────────────────────────────────────────────────────────────────────────

# ─── Guard: only build when the test source exists ────────────────────────────
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_conf_fortran.f90")
    return()
endif()

# ─── Test executable ──────────────────────────────────────────────────────────
add_executable(test_conf_fortran test_conf_fortran.f90)

# Link the conf library (which includes the C interop layer when CONF_BUILD_FORTRAN=ON)
# and the conf_fortran module library (created by task 8.1, provides conf_mod).
# The conf_fortran target exposes the .mod file directory on its INTERFACE include
# path, so the `use conf_mod` statement in the test is satisfied.
if(TARGET conf_fortran)
    target_link_libraries(test_conf_fortran
        PRIVATE
            conf
            conf_fortran
    )
else()
    # Fallback: if conf_fortran target is not yet created (task 8.1 pending),
    # link only conf. The build will fail at link time until conf_fortran exists,
    # but the EXISTS guard above means this path is only reached when the test
    # source file is actually authored (task 8.2).
    target_link_libraries(test_conf_fortran
        PRIVATE
            conf
    )
endif()

# ─── Copy the fixture YAML into the build directory ───────────────────────────
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fixture.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/fixture.yaml
    COPYONLY
)

# ─── CTest registration (single-process, no MPI) ─────────────────────────────
add_test(NAME test_conf_fortran COMMAND $<TARGET_FILE:test_conf_fortran>)
set_tests_properties(test_conf_fortran PROPERTIES
    TIMEOUT 60
    LABELS "fortran"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
