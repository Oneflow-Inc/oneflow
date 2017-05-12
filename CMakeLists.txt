# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(oneflow C CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(oneflow_src_dir ${PROJECT_SOURCE_DIR}/oneflow)
set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# option
option(DOWNLOAD_THIRD_PARTY "Download and build all third party codes" OFF)

# Different set for different platform
if (MSVC)
  foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()
endif()

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wno-sign-compare")
endif()

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})
include(util)
include(proto2cpp)
include(proto2grpc)
include(third_party)

# main cpp
list(APPEND of_main_cc ${oneflow_src_dir}/job/compiler.cpp)

if(WIN32)
  set(oneflow_platform "windows")
  list(APPEND oneflow_excludes "linux")
else()
  set(oneflow_platform "linux")
  list(APPEND oneflow_excludes "windows")
endif()

file(GLOB_RECURSE oneflow_all_src "${oneflow_src_dir}/*.*")
foreach(oneflow_single_file ${oneflow_all_src})
  # Verify whether this file is for other platforms
  set(exclude_this OFF)
  foreach(oneflow_exclude ${oneflow_excludes})
    string(FIND ${oneflow_single_file} ${oneflow_exclude} platform_pos)
    if(NOT ${platform_pos} EQUAL -1)
      set(exclude_this ON)
    endif()
  endforeach()
  # If this file is for other platforms, just exclude it from current project
  if(exclude_this)
    continue()
  endif()

  file(RELATIVE_PATH oneflow_relative_file ${oneflow_src_dir} ${oneflow_single_file})
  get_filename_component(oneflow_relative_path ${oneflow_relative_file} PATH)
  string(REPLACE "/" "\\" group_name ${oneflow_relative_path})
  source_group("${group_name}" FILES ${oneflow_single_file})

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.h$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.proto")
    list(APPEND of_all_proto ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.cpp")
    if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*_test\\.cpp")
      list(APPEND of_all_test_cc ${oneflow_single_file})
    endif()

    list(FIND of_main_cc ${oneflow_single_file} find_result)
    if(${find_result} EQUAL -1) # not main entry
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
  endif()
endforeach()

# proto obj lib
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${oneflow_src_dir} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

#Message(STATUS "oneflow_src_dir = ${oneflow_src_dir}")
#Message(STATUS "of_all_proto = ${of_all_proto}")
#Message(STATUS "proto_rel_name = ${proto_rel_name}")

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_src_dir}
                               ${of_all_rel_protos})

add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(of_protoobj ${oneflow_third_party_libs})

#grpc obj lib
list(APPEND proto2grpc_rel_name distributed_runtime/oneflow_init.proto distributed_runtime/machine_list.proto)
Message (STATUS "proto2grpc_rel_name = ${proto2grpc_rel_name}")
RELATIVE_GRPC_GENERATE_CPP(GRPC_SRCS GRPC_HDRS
                           ${oneflow_src_dir}
                           ${proto2grpc_rel_name})

add_library(of_grpcobj ${GRPC_SRCS} ${GRPC_HDRS})
target_link_libraries(of_grpcobj ${oneflow_third_party_libs})

if (DOWNLOAD_THIRD_PARTY)
  add_dependencies(of_protoobj ${oneflow_third_party_dependencies})
endif()

# cc obj lib
include_directories(${oneflow_src_dir})
include_directories(${PROJECT_BINARY_DIR})
add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj of_grpcobj)

# build main
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} PUBLIC of_ccobj of_protoobj of_grpcobj ${oneflow_third_party_libs})
endforeach()

# build test
foreach(cc ${of_all_test_cc})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} PUBLIC of_ccobj of_protoobj of_grpcobj ${oneflow_third_party_libs})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()
