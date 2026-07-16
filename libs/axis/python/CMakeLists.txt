# Python bindings for AXIS via nanobind
cmake_minimum_required(VERSION 3.21)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(axis_py axis_py.cpp)

target_link_libraries(axis_py PRIVATE axis)
if(AXIS_HAVE_NETCDF)
    target_link_libraries(axis_py PRIVATE axis_io)
endif()
target_include_directories(axis_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/ArborX/src
)

# Install the pure-python axis package directory
install(DIRECTORY axis DESTINATION .)

# Install the compiled nanobind binary extension in the axis package folder
install(TARGETS axis_py LIBRARY DESTINATION axis)

# Generate a type stub (.pyi) for the compiled extension at build time so that
# downstream type checkers (mypy, pyright) can resolve axis_py symbols. The stub
# is produced by importing the freshly built module, so it always matches the
# bindings. Requires nanobind >= 2.0 (provides nanobind_add_stub).
nanobind_add_stub(
    axis_py_stub
    MODULE axis_py
    OUTPUT axis_py.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:axis_py>
    DEPENDS axis_py
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/axis_py.pyi DESTINATION axis)
