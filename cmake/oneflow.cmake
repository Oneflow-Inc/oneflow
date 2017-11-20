# main cpp
list(APPEND of_main_cc ${PROJECT_SOURCE_DIR}/oneflow/core/job/compiler.cpp)
list(APPEND of_main_cc ${PROJECT_SOURCE_DIR}/oneflow/core/job/runtime.cpp)
list(APPEND of_main_cc ${PROJECT_SOURCE_DIR}/oneflow/core/job/scheduler.cpp)

# source_group
if(WIN32)
  set(oneflow_platform "windows")
  list(APPEND oneflow_platform_excludes "linux")
else()
  set(oneflow_platform "linux")
  list(APPEND oneflow_platform_excludes "windows")
endif()

file(GLOB_RECURSE oneflow_all_src "${PROJECT_SOURCE_DIR}/oneflow/core/*.*")
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

  file(RELATIVE_PATH oneflow_relative_file ${PROJECT_SOURCE_DIR}/oneflow/core/ ${oneflow_single_file})
  get_filename_component(oneflow_relative_path ${oneflow_relative_file} PATH)
  string(REPLACE "/" "\\" group_name ${oneflow_relative_path})
  source_group("${group_name}" FILES ${oneflow_single_file})

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.h$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cuh$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cu$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.proto")
    list(APPEND of_all_proto ${oneflow_single_file})
    #list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cpp")
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*_test\\.cpp")
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

# clang format
add_custom_target(of_format)

foreach(source_file ${of_all_obj_cc} ${of_main_cc} ${of_all_test_cc})
    add_custom_command(TARGET of_format PRE_BUILD
    COMMAND clang-format -i -style=file ${source_file})
endforeach()

# proto obj lib
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${PROJECT_SOURCE_DIR}
                               ${of_all_rel_protos})

cuda_add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(of_protoobj ${oneflow_third_party_libs})

# cc obj lib
include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})
cuda_add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)
add_dependencies(of_ccobj of_format)

if(APPLE)
  set(of_libs -Wl,-force_load of_ccobj of_protoobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive of_ccobj of_protoobj -Wl,--no-whole-archive)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj") 
endif()

# build main
set(main_targets "")
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  cuda_add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} ${of_libs} ${oneflow_third_party_libs})
  list(APPEND main_targets ${main_name})
endforeach()

if(WIN32)

else()
# build oneflow.run
#add_custom_target(copy_raw_oneflow_run
#  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/scripts/oneflow.run ${PROJECT_BINARY_DIR})
#
#add_custom_target(oneflow_run ALL
#  COMMAND tar zcf - compiler runtime scheduler -C ${PROJECT_BINARY_DIR} >> ${PROJECT_BINARY_DIR}/oneflow.run
#  DEPENDS ${main_targets} copy_raw_oneflow_run)
endif()

# build test
if(BUILD_TESTING)
  cuda_add_executable(oneflow_testexe ${of_all_test_cc})
  target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs})
  add_test(NAME oneflow_test COMMAND oneflow_testexe)
  foreach(cc ${of_all_test_cc})
    get_filename_component(test_name ${cc} NAME_WE)
    string(CONCAT test_exe_name ${test_name} exe)
    cuda_add_executable(${test_exe_name} ${cc})
    target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs})
  endforeach()
endif()
