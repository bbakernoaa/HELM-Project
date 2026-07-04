
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was TICKConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

####################################################################
# TICKConfig.cmake
#
# Provides the imported target for downstream consumers:
#
#   HELM::TICK  - STATIC library; C++20 fixed-point time arithmetic.
#
# Consumers should write:
#
#   find_package(TICK REQUIRED)
#   target_link_libraries(my_app PRIVATE HELM::TICK)
#
# No external dependencies are required beyond the C++20 standard
# library (Requirement 12.3).
####################################################################

include(CMakeFindDependencyMacro)

# TICK has no external runtime dependencies — only C++20 standard library.

include("${CMAKE_CURRENT_LIST_DIR}/TICKTargets.cmake")

check_required_components(TICK)
