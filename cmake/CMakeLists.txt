# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(oneflow C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 14)

# Actual source is the ../ directory
get_filename_component(oneflow_source_dir ${oneflow_SOURCE_DIR} PATH)


message(STATUS "oneflow_source_dir: ${oneflow_source_dir}")

# Options
option(oneflow_VERBOSE "Enable for verbose output" OFF)


if (NOT WIN32)
  # find_package(Threads)
endif()

# Force to link static cxx runtime library
if (MSVC)
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
endif (MSVC)

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

SHOW_VARIABLES()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
#set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
#  CACHE PATH "Location where external projects will be downloaded.")
#mark_as_advanced(DOWNLOAD_LOCATION)

# Location where third party headers and libs will be put
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party"
  CACHE PATH "Location where third party headers and libs will be put.")
mark_as_advanced(THIRD_PARTY_DIR)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  message(STATUS "Build on Windows")
endif()

include(zlib)
include(protobuf)
#include(grpc)
#include(googletest)



