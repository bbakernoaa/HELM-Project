# Python bindings for AXIS via nanobind
cmake_minimum_required(VERSION 3.21)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(axis_py axis_py.cpp)

target_link_libraries(axis_py PRIVATE axis)
target_include_directories(axis_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/ArborX/src
)

# Install the pure-python axis package directory
install(DIRECTORY axis DESTINATION .)

# Install the compiled nanobind binary extension in the axis package folder
install(TARGETS axis_py LIBRARY DESTINATION axis)

# Also install axis_py to the root destination to support direct top-level import axis_py
install(TARGETS axis_py LIBRARY DESTINATION .)
