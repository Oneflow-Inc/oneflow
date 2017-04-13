# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(oneflow C CXX)

# Set C++14 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

# Actual source is the ../ directory
#get_filename_component(oneflow_root_dir ${oneflow_SOURCE_DIR} PATH)
set(oneflow_src_dir ${oneflow_SOURCE_DIR}/oneflow)
set(oneflow_cmake_dir ${oneflow_SOURCE_DIR}/cmake)

# Options
option(oneflow_VERBOSE "Enable for verbose output" OFF)

if (NOT WIN32)
  # Threads: defines CMAKE_THREAD_LIBS_INIT and adds -pthread compile option
  # for targets that link ${CMAKE_THREAD_LIBS_INIT}
  find_package(Threads)
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

# [CLEANUP] Remove when done, For debugging
function(SHOW_VARIABLES)
  get_cmake_property(_variableNames VARIABLES)
  foreach(_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

# External dependencies
set(CMAKE_MODULE_PATH ${oneflow_cmake_dir}/external)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/utils)

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
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

include(zlib)
include(protobuf)
include(googletest)
include(glog)
include(gflags)
include(grpc)

set(oneflow_EXTERNAL_LIBRARIES
    ${ZLIB_STATIC_LIBRARIES}
    ${GLOG_STATIC_LIBRARIES}
    ${GFLAGS_STATIC_LIBRARIES}
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
    ${oneflow_src_dir}
    # External dependencies.
    ${ZLIB_INCLUDE_DIR}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${GOOGLETEST_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${GRPC_INCLUDE_DIR}
)

include(util)
include(proto2cpp)

SUBDIRLIST(subdir_list ${oneflow_src_dir})
foreach(subdir ${subdir_list})
  file(GLOB subdir_headers    "${oneflow_src_dir}/${subdir}/*.h")
  file(GLOB subdir_cpps       "${oneflow_src_dir}/${subdir}/*.cpp")
  file(GLOB subdir_test_cpps  "${oneflow_src_dir}/${subdir}/*_test.cpp")
  file(GLOB subdir_protos     "${oneflow_src_dir}/${subdir}/*.proto")
  # Remove *_test.cpp from ${subdir_cpps}
  foreach(test_cpp ${subdir_test_cpps})
    list(REMOVE_ITEM subdir_cpps ${test_cpp})
  endforeach()

  source_group(${subdir} FILES ${subdir_headers} ${subdir_cpps})
  list(APPEND of_all_ccobj ${subdir_headers} ${subdir_cpps} ${subdir_protos})
  list(APPEND of_all_protos ${subdir_protos})
  list(APPEND of_all_test ${subdir_test_cpps})
endforeach()
source_group("proto" FILES ${of_all_protos})

# Get the relative path of all protos, used by RELATIVE_PROTOBUF_GENERATE_CPP function
foreach(proto_name ${of_all_protos})
  file(RELATIVE_PATH proto_rel_name ${oneflow_src_dir} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

# proto obj lib
RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_src_dir}
                               ${of_all_rel_protos})

add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(of_protoobj ${oneflow_EXTERNAL_DEPENDENCIES})
list(APPEND ONEFLOW_LINKER_LIBS of_protoobj)

# cc obj lib
include_directories(${PROJECT_BINARY_DIR})
add_library(of_ccobj ${of_all_ccobj})
add_dependencies(of_ccobj of_protoobj)
list(APPEND ONEFLOW_LINKER_LIBS of_ccobj)


# build test
foreach(cc ${of_all_test})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} PUBLIC ${ONEFLOW_LINKER_LIBS} ${oneflow_EXTERNAL_LIBRARIES})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()

# SHOW_VARIABLES()
message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "APPLE: ${APPLE}")
message(STATUS "WIN32: ${WIN32}")