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

# [CLEANUP] Remove when done
# For debugging
function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

# SHOW_VARIABLES()

# External dependencies
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

# Location where external projects will be downloaded
set(DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/downloads"
  CACHE PATH "Location where external projects will be downloaded.")
mark_as_advanced(DOWNLOAD_LOCATION)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  message(STATUS "Build on Windows")
endif()

include(zlib)
include(protobuf)
#include(googletest)
include(grpc)



