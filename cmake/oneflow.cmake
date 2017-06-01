# main cpp
list(APPEND of_main_cc ${oneflow_src_dir}/compile/compiler.cpp)
list(APPEND of_main_cc ${oneflow_src_dir}/runtime/elf_runner.cpp)

if(WIN32)
  set(oneflow_platform "windows")
  list(APPEND oneflow_platform_excludes "linux")
else()
  set(oneflow_platform "linux")
  list(APPEND oneflow_platform_excludes "windows")
endif()

file(GLOB_RECURSE oneflow_all_src "${oneflow_src_dir}/*.*")
foreach(oneflow_single_file ${oneflow_all_src})
  # Verify whether this file is for other platforms
  set(exclude_this OFF)
  foreach(oneflow_platform_exclude ${oneflow_platform_excludes})
    string(FIND ${oneflow_single_file} ${oneflow_platform_exclude} platform_found)
    if(NOT ${platform_found} EQUAL -1)  # the ${oneflow_single_file} is for other platforms
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

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.cuh$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.cu$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.proto")
    list(APPEND of_all_proto ${oneflow_single_file})
    list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
  endif()

  if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*\\.cpp")
    if("${oneflow_single_file}" MATCHES "^${oneflow_src_dir}/.*_test\\.cpp")
      # test file
      list(APPEND of_all_test_cc ${oneflow_single_file})
    else()
      # not test file
      list(FIND of_main_cc ${oneflow_single_file} main_found)
      if(${main_found} EQUAL -1) # not main entry
        list(APPEND of_all_obj_cc ${oneflow_single_file})
      endif()
    endif()
  endif()
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

# cc obj lib
include_directories(${oneflow_src_dir})
include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})
cuda_add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)

if(APPLE)
  set(of_libs -Wl,-force_load of_ccobj of_protoobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive of_ccobj of_protoobj -Wl,--no-whole-archive)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj") 
endif()

# build main
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  cuda_add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} ${of_libs} ${oneflow_third_party_libs})
endforeach()

# build test
foreach(cc ${of_all_test_cc})
  get_filename_component(test_name ${cc} NAME_WE)
  string(CONCAT test_exe_name ${test_name} exe)
  cuda_add_executable(${test_exe_name} ${cc})
  target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs})
  add_test(NAME ${test_name} COMMAND ${test_exe_name})
endforeach()
