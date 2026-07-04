
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CONFConfig.cmake.in                            ########

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

# ─── CONFConfig.cmake ────────────────────────────────────────────────────────
# CMake package configuration file for CONF (Configuration Object & Notation Framework)
#
# This file is included by find_package(CONF) and imports the CONF targets.
#
# NOTE: Unlike HALO (which has PUBLIC deps like MPI and Kokkos), CONF has NO
# public transitive dependencies. yaml-cpp is linked PRIVATE and statically
# absorbed into the conf library — it is never exposed to downstream consumers.
# Therefore find_dependency(yaml-cpp) is deliberately omitted here.
# ─────────────────────────────────────────────────────────────────────────────

# Import the exported targets (HELM::conf)
include("${CMAKE_CURRENT_LIST_DIR}/CONFTargets.cmake")

# Provide the documented uppercase alias HELM::CONF that downstream consumers
# use in target_link_libraries(). The install-exported target is HELM::conf
# (lowercase) because CMake's NAMESPACE prefix is prepended to the target name
# as-is; the uppercase alias matches the HELM naming convention.
if(NOT TARGET HELM::CONF)
    add_library(HELM::CONF INTERFACE IMPORTED)
    set_target_properties(HELM::CONF PROPERTIES
        INTERFACE_LINK_LIBRARIES HELM::conf
    )
endif()

check_required_components(CONF)
