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
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
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

#SHOW_VARIABLES()

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
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC -D__VERSION__=\"MSVC\")
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS)
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

if(WIN32)
  message(STATUS "Build on Windows")
endif()



include(zlib)
include(protobuf)
include(googletest)
include(glog)
include(gflags)
include(grpc)

set(oneflow_EXTERNAL_LIBRARIES
    ${ZLIB_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
    ${GLOG_STATIC_LIBRARIES}
    ${GOOGLETEST_STATIC_LIBRARIES}
    ${PROTOBUF_STATIC_LIBRARIES}
    ${GRPC_STATIC_LIBRARIES}
)

set(oneflow_EXTERNAL_DEPENDENCIES
    zlib_copy_headers_to_destination
    zlib_copy_libs_to_destination
    gflags_copy_headers_to_destination
    gflags_copy_libs_to_destination
    glog_copy_headers_to_destination
    glog_copy_libs_to_destination
    googletest_copy_headers_to_destination
    googletest_copy_libs_to_destination
    protobuf_copy_headers_to_destination
    protobuf_copy_libs_to_destination
    protobuf_copy_binary_to_destination
    grpc_copy_headers_to_destination
    grpc_copy_libs_to_destination
)

include_directories(
    # Source and generated code.
    ${oneflow_source_dir}
    #${CMAKE_CURRENT_BINARY_DIR}
    # External dependencies.
    ${ZLIB_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIR}
)
