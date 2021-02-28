include(python)
# main cpp
list(APPEND of_main_cc ${PROJECT_SOURCE_DIR}/oneflow/core/job/oneflow_worker.cpp)

function(oneflow_add_executable)
  if (BUILD_CUDA)
    cuda_add_executable(${ARGV})
  else()
    add_executable(${ARGV})
  endif()
endfunction()

function(oneflow_add_library)
  if (BUILD_CUDA)
    cuda_add_library(${ARGV})
  else()
    add_library(${ARGV})
  endif()
endfunction()

# source_group
if(WIN32)
  set(oneflow_platform "windows")
  list(APPEND oneflow_platform_excludes "linux")
else()
  set(oneflow_platform "linux")
  list(APPEND oneflow_platform_excludes "windows")
endif()

file(GLOB_RECURSE oneflow_all_hdr_to_be_expanded "${PROJECT_SOURCE_DIR}/oneflow/core/*.e.h" "${PROJECT_SOURCE_DIR}/oneflow/python/*.e.h")
foreach(oneflow_hdr_to_be_expanded ${oneflow_all_hdr_to_be_expanded})
  file(RELATIVE_PATH of_ehdr_rel_path ${PROJECT_SOURCE_DIR} ${oneflow_hdr_to_be_expanded})
  set(of_e_h_expanded "${PROJECT_BINARY_DIR}/${of_ehdr_rel_path}.expanded.h")
  if(WIN32)
    error( "Expanding macro in WIN32 is not supported yet")
  else()
    add_custom_command(OUTPUT ${of_e_h_expanded}
      COMMAND ${CMAKE_C_COMPILER}
      ARGS -E -I"${PROJECT_SOURCE_DIR}" -I"${PROJECT_BINARY_DIR}"
      -o "${of_e_h_expanded}" "${oneflow_hdr_to_be_expanded}"
      DEPENDS ${oneflow_hdr_to_be_expanded}
      COMMENT "Expanding macros in ${oneflow_hdr_to_be_expanded}")
    list(APPEND oneflow_all_hdr_expanded "${of_e_h_expanded}")
  endif()
  set_source_files_properties(${oneflow_all_hdr_expanded} PROPERTIES GENERATED TRUE)
endforeach()

file(GLOB_RECURSE oneflow_all_src "${PROJECT_SOURCE_DIR}/oneflow/core/*.*" "${PROJECT_SOURCE_DIR}/oneflow/python/*.*"
 "${PROJECT_SOURCE_DIR}/oneflow/user/*.*" "${PROJECT_SOURCE_DIR}/oneflow/api/python/*.*"
 "${PROJECT_SOURCE_DIR}/oneflow/extension/python/*.*")
if (WITH_XLA OR WITH_TENSORRT)
  file(GLOB_RECURSE oneflow_xrt_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/*.*")
  if (NOT WITH_XLA)
    file(GLOB_RECURSE xla_removing_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/xla/*.*")
  endif ()
  if (NOT WITH_TENSORRT)
    file(GLOB_RECURSE trt_removing_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/tensorrt/*.*")
  endif ()

  list(APPEND xrt_removing_srcs ${xla_removing_src})
  list(APPEND xrt_removing_srcs ${trt_removing_src})
  # message(STATUS "removing_srcs: ${xrt_removing_srcs}")
  foreach (removing_file ${xrt_removing_srcs})
    list(REMOVE_ITEM oneflow_xrt_src ${removing_file})
  endforeach ()
  list(APPEND oneflow_all_src ${oneflow_xrt_src})
endif()

foreach(oneflow_single_file ${oneflow_all_src})
  # Verify whether this file is for other platforms
  set(exclude_this OFF)
  set(group_this OFF)
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

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/python/.*\\.h$")
    list(APPEND of_python_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.h$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.hpp$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.cuh$")
    if(BUILD_CUDA)
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.cu$")
    if(BUILD_CUDA)
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.proto$")
    list(APPEND of_all_proto ${oneflow_single_file})
    #list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/api/python/.*\\.cpp$")
    list(APPEND of_pybind_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/api/python/.*\\.h$")
    list(APPEND of_pybind_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/extension/python/.*\\.cpp$")
    list(APPEND of_pyext_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/extension/python/.*\\.h$")
    list(APPEND of_pyext_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.cpp$")
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/transport/transport_test_main\\.cpp$")
      list(APPEND of_transport_test_cc ${oneflow_single_file})
    elseif("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*_test\\.cpp$")
      # test file
      list(APPEND of_all_test_cc ${oneflow_single_file})
    elseif("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/graph/.*\\.cpp$")
    else()
      # not test file
      list(FIND of_main_cc ${oneflow_single_file} main_found)
      if(${main_found} EQUAL -1) # not main entry
        list(APPEND of_all_obj_cc ${oneflow_single_file})
      endif()
    endif()
    set(group_this ON)
  endif()
  if(group_this)
    file(RELATIVE_PATH oneflow_relative_file ${PROJECT_SOURCE_DIR}/oneflow/core/ ${oneflow_single_file})
    get_filename_component(oneflow_relative_path ${oneflow_relative_file} PATH)
    string(REPLACE "/" "\\" group_name ${oneflow_relative_path})
    source_group("${group_name}" FILES ${oneflow_single_file})
  endif()
endforeach()

# clang format
add_custom_target(of_format
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --clang_format_binary clang-format --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix --quiet
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_py_format.py --source_dir ${CMAKE_CURRENT_SOURCE_DIR} --fix
  )

# generate version
if(BUILD_GIT_VERSION)
  set(OF_GIT_VERSION_DIR ${CMAKE_CURRENT_BINARY_DIR}/of_git_version)
  set(OF_GIT_VERSION_FILE ${OF_GIT_VERSION_DIR}/version.cpp)
  set(OF_GIT_VERSION_DUMMY_FILE ${OF_GIT_VERSION_DIR}/_version.cpp)
  add_custom_target(of_git_version_create_dir
          COMMAND ${CMAKE_COMMAND} -E make_directory ${OF_GIT_VERSION_DIR})
  add_custom_command(
          OUTPUT ${OF_GIT_VERSION_DUMMY_FILE}
          COMMAND ${CMAKE_COMMAND} -DOF_GIT_VERSION_FILE=${OF_GIT_VERSION_FILE}
            -DOF_GIT_VERSION_ROOT=${PROJECT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
          DEPENDS of_git_version_create_dir)
  add_custom_target(of_git_version
          DEPENDS ${OF_GIT_VERSION_DUMMY_FILE})
  set_source_files_properties(${OF_GIT_VERSION_FILE} PROPERTIES GENERATED TRUE)
  list(APPEND of_all_obj_cc ${OF_GIT_VERSION_FILE})
  add_definitions(-DWITH_GIT_VERSION)
endif()

set(of_proto_python_dir "${PROJECT_BINARY_DIR}/of_proto_python")

# proto obj lib
add_custom_target(make_pyproto_dir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/python_scripts/oneflow/core
  COMMAND ${CMAKE_COMMAND} -E make_directory ${of_proto_python_dir}
	)
add_dependencies(make_pyproto_dir prepare_oneflow_third_party)
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${PROJECT_SOURCE_DIR}
                               ${of_all_rel_protos})

oneflow_add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(of_protoobj ${oneflow_third_party_libs})
add_dependencies(of_protoobj make_pyproto_dir)

include(cfg)
GENERATE_CFG_AND_PYBIND11_CPP(CFG_SRCS CFG_HRCS PYBIND11_SRCS ${PROJECT_SOURCE_DIR})
oneflow_add_library(of_cfgobj ${CFG_SRCS} ${CFG_HRCS})
target_link_libraries(of_cfgobj ${oneflow_third_party_libs})
add_dependencies(of_cfgobj of_protoobj)

# cc obj lib
include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})
add_subdirectory(${PROJECT_SOURCE_DIR}/oneflow/core)
oneflow_add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj of_graph ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)
add_dependencies(of_ccobj of_cfgobj)
if (BUILD_GIT_VERSION)
  add_dependencies(of_ccobj of_git_version)
endif()
if (USE_CLANG_FORMAT)
  add_dependencies(of_ccobj of_format)
endif()

add_library(of_pyext_obj ${of_pyext_obj_cc})
target_include_directories(of_pyext_obj PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(of_pyext_obj of_ccobj)
add_dependencies(of_pyext_obj of_ccobj)

if(APPLE)
  set(of_libs -Wl,-force_load of_ccobj of_protoobj of_cfgobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive of_ccobj of_protoobj of_cfgobj -Wl,--no-whole-archive -ldl)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj of_cfgobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj")
endif()

pybind11_add_module(oneflow_internal ${PYBIND11_SRCS} ${of_pybind_obj_cc} ${of_main_cc} ${PYBIND_REGISTRY_CC})
set_property(TARGET oneflow_internal PROPERTY CXX_VISIBILITY_PRESET "default")
add_dependencies(oneflow_internal of_cfgobj)
set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python_scripts/oneflow")
target_link_libraries(oneflow_internal PRIVATE ${of_libs} ${oneflow_third_party_libs} of_pyext_obj ${oneflow_exe_third_party_libs})
target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

set(of_pyscript_dir "${PROJECT_BINARY_DIR}/python_scripts")
add_custom_target(of_pyscript_copy ALL
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/clean_generated_api.py --root_path=${of_pyscript_dir}
    COMMAND "${CMAKE_COMMAND}" -E copy
        "${PROJECT_SOURCE_DIR}/oneflow/init.py" "${of_pyscript_dir}/oneflow/__init__.py"
    COMMAND rm -rf ${of_pyscript_dir}/oneflow/python
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/oneflow/python" "${of_pyscript_dir}/oneflow/python"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${of_proto_python_dir}/oneflow/core" "${of_pyscript_dir}/oneflow/core"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${of_pyscript_dir}/oneflow/python_gen"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow/python_gen/__init__.py"
    COMMAND ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/generate_oneflow_symbols_export_file.py"
        "${PROJECT_SOURCE_DIR}" "${of_pyscript_dir}/oneflow/python_gen/__export_symbols__.py")

add_dependencies(of_pyscript_copy of_protoobj)
add_custom_target(generate_api ALL
  COMMAND rm -rf ${of_pyscript_dir}/oneflow/generated
  COMMAND export PYTHONPATH=${of_pyscript_dir}:$PYTHONPATH && ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_oneflow_api.py --root_path=${of_pyscript_dir}/oneflow)
add_dependencies(generate_api of_pyscript_copy)
add_dependencies(generate_api oneflow_internal)

file(RELATIVE_PATH PROJECT_BINARY_DIR_RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
add_custom_target(pip_install)
add_dependencies(pip_install generate_api)
add_custom_command(
  TARGET pip_install
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMAND ${Python_EXECUTABLE} -m pip install -e ${PROJECT_SOURCE_DIR} --install-option="--build_dir=${PROJECT_BINARY_DIR_RELATIVE}" --user)

# get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${include_dirs})
#   message("-I'${dir}' ")
# endforeach()

# build main
set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  oneflow_add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs})
  set_target_properties(${main_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endforeach()

# build test
if(BUILD_TESTING)
  if(BUILD_CUDA)
    if (of_all_test_cc)
      oneflow_add_executable(oneflow_testexe ${of_all_test_cc})
      target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs})
      set_target_properties(oneflow_testexe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
      add_test(NAME oneflow_test COMMAND oneflow_testexe)
      #  foreach(cc ${of_all_test_cc})
      #    get_filename_component(test_name ${cc} NAME_WE)
      #    string(CONCAT test_exe_name ${test_name} exe)
      #    oneflow_add_executable(${test_exe_name} ${cc})
      #    target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs})
      #  endforeach()
    endif()
    if (of_separate_test_cc)
      foreach(cc ${of_separate_test_cc})
        get_filename_component(test_name ${cc} NAME_WE)
        string(CONCAT test_exe_name ${test_name} exe)
        oneflow_add_executable(${test_exe_name} ${cc})
        target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs})
      endforeach()
    endif()
  endif()
endif()

# build transport_test
foreach(cc ${of_transport_test_cc})
  get_filename_component(transport_test_name ${cc} NAME_WE)
  string(CONCAT transport_test_exe_name ${transport_test_name} _exe)
  oneflow_add_executable(${transport_test_exe_name} ${cc})
  target_link_libraries(${transport_test_exe_name} ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs})
  set_target_properties(${transport_test_exe_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endforeach()


# build include
set(ONEFLOW_INCLUDE_DIR "${PROJECT_BINARY_DIR}/python_scripts/oneflow/include")
add_custom_target(of_include_copy ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${ONEFLOW_INCLUDE_DIR}")
add_dependencies(of_include_copy of_ccobj)
file(REMOVE_RECURSE "${ONEFLOW_INCLUDE_DIR}")
foreach(of_include_src_dir ${ONEFLOW_INCLUDE_SRC_DIRS})
  set(oneflow_all_include_file)
  #file(GLOB_RECURSE h_files "${of_include_src_dir}/*.h")
  #list(APPEND oneflow_all_include_file ${h_files})
  #file(GLOB_RECURSE hpp_files "${of_include_src_dir}/*.hpp")
  #list(APPEND oneflow_all_include_file ${hpp_files})
  file(GLOB_RECURSE oneflow_all_include_file "${of_include_src_dir}/*.*")
  copy_files("${oneflow_all_include_file}" "${of_include_src_dir}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)
endforeach()

set(DEVICE_REG_HEADERS "${PROJECT_SOURCE_DIR}/oneflow/core/framework/device_register_*.h")
set(AUTO_GEN_DEV_REG_HEADER "${PROJECT_BINARY_DIR}/oneflow/core/framework/auto_gen_device_registry.h")
set(AUTO_GEN_DEV_REG_MACRO_ID "ONEFLOW_CORE_FRAMEWORK_AUTO_GEN_DEVICE_REGISTRY_H_")

message(STATUS "auto generated header file: ${AUTO_GEN_DEV_REG_HEADER}")
set(AUTO_GEN_DEV_REG_HEADER_CONTENT "#ifndef ${AUTO_GEN_DEV_REG_MACRO_ID}\n#define ${AUTO_GEN_DEV_REG_MACRO_ID}\n")
file(GLOB_RECURSE DEVICE_REGISTER_HEADERS ${DEVICE_REG_HEADERS})
foreach(item ${DEVICE_REGISTER_HEADERS})
    file(RELATIVE_PATH item ${PROJECT_SOURCE_DIR} ${item})
    message(STATUS "device register header file found: " ${item})
    set(AUTO_GEN_DEV_REG_HEADER_CONTENT "${AUTO_GEN_DEV_REG_HEADER_CONTENT}#include \"${item}\"\n")
endforeach()
set(AUTO_GEN_DEV_REG_HEADER_CONTENT "${AUTO_GEN_DEV_REG_HEADER_CONTENT}#endif //${AUTO_GEN_DEV_REG_MACRO_ID}\n\n")
write_file_if_different(${AUTO_GEN_DEV_REG_HEADER} ${AUTO_GEN_DEV_REG_HEADER_CONTENT})
list(APPEND PROTO_HDRS ${AUTO_GEN_DEV_REG_HEADER})

copy_files("${PROTO_HDRS}" "${PROJECT_BINARY_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)
copy_files("${CFG_HRCS}" "${PROJECT_BINARY_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)

set(OF_CORE_HDRS)
list(APPEND of_core_dir_name_list "common" "device" "framework" "kernel/util" "persistence")
foreach(of_core_dir_name ${of_core_dir_name_list})
  file(GLOB_RECURSE h_files "${PROJECT_SOURCE_DIR}/oneflow/core/${of_core_dir_name}/*.h")
  list(APPEND OF_CORE_HDRS ${h_files})
  file(GLOB_RECURSE hpp_files "${PROJECT_SOURCE_DIR}/oneflow/core/${of_core_dir_name}/*.hpp")
  list(APPEND OF_CORE_HDRS ${hpp_files})
endforeach()
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/kernel/new_kernel_util.h")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/kernel/kernel_context.h")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/kernel/kernel_util.cuh")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/job/sbp_signature_builder.h")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/job/parallel_desc.h")
copy_files("${OF_CORE_HDRS}" "${PROJECT_SOURCE_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)

add_dependencies(pip_install of_include_copy)
