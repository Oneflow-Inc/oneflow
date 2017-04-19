# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

# Project
project(oneflow C CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(oneflow_src_dir ${PROJECT_SOURCE_DIR}/oneflow)
set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

# Force to link static cxx runtime library
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

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})
include(util)
include(proto2cpp)
include(third_party)

# main cpp
list(APPEND of_main_cc ${oneflow_src_dir}/job/compiler.cpp)

# source_group
SUBDIRLIST(subdir_list ${oneflow_src_dir})
foreach(subdir ${subdir_list})
  file(GLOB subdir_headers    "${oneflow_src_dir}/${subdir}/*.h")
  file(GLOB subdir_obj_cpps   "${oneflow_src_dir}/${subdir}/*.cpp")
  file(GLOB subdir_test_cpps  "${oneflow_src_dir}/${subdir}/*_test.cpp")
  file(GLOB subdir_protos     "${oneflow_src_dir}/${subdir}/*.proto")
  foreach(test_cpp ${subdir_test_cpps})
    list(REMOVE_ITEM subdir_obj_cpps ${test_cpp})
  endforeach()
  foreach(main_cpp ${of_main_cc})
    list(REMOVE_ITEM subdir_obj_cpps ${main_cpp})
  endforeach()
  source_group(${subdir} FILES ${subdir_headers} ${subdir_obj_cpps} {subdir_protos})
  list(APPEND of_all_obj_cc ${subdir_headers} ${subdir_obj_cpps})
  list(APPEND of_all_proto ${subdir_protos})
  list(APPEND of_all_test_cc ${subdir_test_cpps})
endforeach()

# proto obj lib
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${oneflow_src_dir} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_src_dir}
                               ${of_all_rel_protos})

add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(of_protoobj ${oneflow_third_party_libs})
add_dependencies(of_protoobj ${oneflow_third_party_dependencies})

# cc obj lib
include_directories(${oneflow_src_dir})
include_directories(${PROJECT_BINARY_DIR})
add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)

# build main
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} PUBLIC of_protoobj of_ccobj ${oneflow_third_party_libs})
endforeach()

# build test
foreach(cc ${of_all_test_cc})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} PUBLIC of_protoobj of_ccobj ${oneflow_third_party_libs})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()
