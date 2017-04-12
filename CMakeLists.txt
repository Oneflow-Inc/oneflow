# cmake configure
cmake_minimum_required(VERSION 3.5)
project(oneflow C CXX)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_testing()
include(util)

# global configure
set(oneflow_source_dir "${PROJECT_SOURCE_DIR}/oneflow")
include_directories(${oneflow_source_dir})

# dependencies

if(UNIX AND NOT APPLE)
  set(PROTOBUF_PROTOC_EXECUTABLE "/opt/oneflow/protobuf3/bin/protoc"
      CACHE FILEPATH "")
  set(Protobuf_INCLUDE_DIRS "/opt/oneflow/protobuf3/include" CACHE PATH "")
  set(Protobuf_LIBRARIES "/opt/oneflow/protobuf3/lib/libprotobuf.so" CACHE PATH "")
endif()
include(dependencies)

# proto
include(proto2cpp)
file(GLOB_RECURSE oneflow_protos
     RELATIVE ${oneflow_source_dir}
     "*.proto")
RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_source_dir}
                               ${oneflow_protos})
include_directories(${PROJECT_BINARY_DIR})
add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
list(APPEND ONEFLOW_LINKER_LIBS of_protoobj)

# find all h,object-cpp,test-cpp

function(FindFilesMatch found_files expr)
  file(GLOB_RECURSE ${found_files} ${expr})
  foreach(file_i ${${found_files}})
    file(RELATIVE_PATH relative_path ${oneflow_source_dir} ${file_i})
    string(SUBSTRING ${relative_path} 0 2 relative_path_prefix)
    if(${relative_path_prefix} STREQUAL ..)
      list(REMOVE_ITEM ${found_files} ${file_i})
    endif()
  endforeach()
  set(${found_files} ${${found_files}} PARENT_SCOPE)
endfunction()

FindFilesMatch(oneflow_all_h "*.h")
FindFilesMatch(oneflow_all_cc "*.cpp")

foreach(cc ${oneflow_all_cc})
  get_filename_component(cc_name_we ${cc} NAME_WE)
  STR_ENDSWITH(${cc_name_we} _test is_test_cc)
  if(${is_test_cc})
    list(APPEND oneflow_test_cc ${cc})
  else()
    list(APPEND oneflow_object_cc ${cc})
  endif()
endforeach()

# source_group

SUBDIRLIST(subdir_list ${oneflow_source_dir})
foreach(subdir_name ${subdir_list})
  set(subdir_path "${oneflow_source_dir}/${subdir_name}")
  oneflow_source_group("${subdir_name}" GLOB "${subdir_path}/*")
endforeach()

# build object-cpp
add_library(of_ccobj ${oneflow_object_cc} ${oneflow_all_h})
add_dependencies(of_ccobj of_protoobj)
list(APPEND ONEFLOW_LINKER_LIBS of_ccobj)

# build test
foreach(cc ${oneflow_test_cc})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} ${ONEFLOW_LINKER_LIBS})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()
