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
 "${PROJECT_SOURCE_DIR}/oneflow/customized/*.*")
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

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/python/.*\\.i$")
    list(APPEND of_all_swig ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.h$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.hpp$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.cuh$")
    if(BUILD_CUDA) 
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.cu$")
    if(BUILD_CUDA)
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.proto$")
    list(APPEND of_all_proto ${oneflow_single_file})
    #list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
    set(group_this ON)
  endif()
  
  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*\\.cpp$")
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|customized|xrt)/.*_test\\.cpp$")
      # test file
      list(APPEND of_all_test_cc ${oneflow_single_file})
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

if(PY3)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_package(Python3 COMPONENTS Development NumPy)
  if (Python3_Development_FOUND AND Python3_INCLUDE_DIRS)
    set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
  endif()
  if (Python3_NumPy_FOUND AND Python3_NumPy_INCLUDE_DIRS)
    set(Python_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})
  endif()

  message("-- Python3 specified. Version found: " ${Python3_VERSION})
  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
else()
  find_package(Python2 COMPONENTS Interpreter REQUIRED)
  find_package(Python2 COMPONENTS Development NumPy)
  if (Python2_Development_FOUND AND Python2_INCLUDE_DIRS)
    set(Python_INCLUDE_DIRS ${Python2_INCLUDE_DIRS})
  endif()
  if (Python2_NumPy_FOUND AND Python2_NumPy_INCLUDE_DIRS)
    set(Python_NumPy_INCLUDE_DIRS ${Python2_NumPy_INCLUDE_DIRS})
  endif()
  message("-- Python2 specified. Version found: " ${Python2_VERSION})
  set(Python_EXECUTABLE ${Python2_EXECUTABLE})
endif()
message("-- Using Python executable: " ${Python_EXECUTABLE})
if (NOT Python_INCLUDE_DIRS)
  message(STATUS "Getting python include directory from sysconfig..")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['include'])" 
    OUTPUT_VARIABLE Python_INCLUDE_DIRS
    RESULT_VARIABLE ret_code)
  string(STRIP ${Python_INCLUDE_DIRS} Python_INCLUDE_DIRS)
  if ((NOT (ret_code EQUAL "0")) OR (NOT IS_DIRECTORY ${Python_INCLUDE_DIRS})
    OR (NOT EXISTS ${Python_INCLUDE_DIRS}/Python.h))
    set(Python_INCLUDE_DIRS "")
  endif()
endif()
if (NOT Python_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find python include directory")
endif()
message(STATUS "Found python include directory ${Python_INCLUDE_DIRS}")

if (NOT Python_NumPy_INCLUDE_DIRS)
  message(STATUS "Getting numpy include directory by numpy.get_include()..")
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())" 
    OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS
    RESULT_VARIABLE ret_code)
  string(STRIP ${Python_NumPy_INCLUDE_DIRS} Python_NumPy_INCLUDE_DIRS)
  if ((NOT ret_code EQUAL 0) OR (NOT IS_DIRECTORY ${Python_NumPy_INCLUDE_DIRS}) 
    OR (NOT EXISTS ${Python_NumPy_INCLUDE_DIRS}/numpy/arrayobject.h))
    set(Python_NumPy_INCLUDE_DIRS "")
  endif()
endif()
if (NOT Python_NumPy_INCLUDE_DIRS)
  message(FATAL_ERROR "Cannot find numpy include directory")
endif()
message(STATUS "Found numpy include directory ${Python_NumPy_INCLUDE_DIRS}")

# clang format
add_custom_target(of_format
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --clang_format_binary clang-format --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_py_format.py --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/oneflow/python --fix
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

# proto obj lib
add_custom_target(make_pyproto_dir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/python_scripts/oneflow/core
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/python_scripts/oneflow_pyproto
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/python_scripts/oneflow_pyproto/oneflow
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/python_scripts/oneflow_pyproto/oneflow/core
	)
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

# cc obj lib
include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})
oneflow_add_library(of_ccobj ${of_all_obj_cc})
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)
if (BUILD_GIT_VERSION)
  add_dependencies(of_ccobj of_git_version)
endif()
if (USE_CLANG_FORMAT)
  add_dependencies(of_ccobj of_format)
endif()

if(APPLE)
  set(of_libs -Wl,-force_load of_ccobj of_protoobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive of_ccobj of_protoobj -Wl,--no-whole-archive)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj") 
endif()

# build swig
foreach(swig_name ${of_all_swig})
  file(RELATIVE_PATH swig_rel_name ${PROJECT_SOURCE_DIR} ${swig_name})
  list(APPEND of_all_rel_swigs ${swig_rel_name})
endforeach()

RELATIVE_SWIG_GENERATE_CPP(SWIG_SRCS SWIG_HDRS
                              ${PROJECT_SOURCE_DIR}
                              ${of_all_rel_swigs})
oneflow_add_library(oneflow_internal SHARED ${SWIG_SRCS} ${SWIG_HDRS} ${of_main_cc})
set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python_scripts/oneflow")
target_link_libraries(oneflow_internal ${of_libs} ${oneflow_third_party_libs})
target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

set(of_pyscript_dir "${PROJECT_BINARY_DIR}/python_scripts")
file(REMOVE_RECURSE "${of_pyscript_dir}/oneflow/python")
add_custom_target(of_pyscript_copy ALL
    COMMAND "${CMAKE_COMMAND}" -E copy
        "${PROJECT_SOURCE_DIR}/oneflow/init.py" "${of_pyscript_dir}/oneflow/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/oneflow/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${of_pyscript_dir}/oneflow/python"
    COMMAND ${Python_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tools/generate_oneflow_symbols_export_file.py"
        "${PROJECT_SOURCE_DIR}" "${of_pyscript_dir}/oneflow/python/__export_symbols__.py")
file(GLOB_RECURSE oneflow_all_python_file "${PROJECT_SOURCE_DIR}/oneflow/python/*.py")
copy_files("${oneflow_all_python_file}" "${PROJECT_SOURCE_DIR}" "${of_pyscript_dir}" of_pyscript_copy)
add_dependencies(of_pyscript_copy of_protoobj)
add_custom_target(generate_api ALL
  COMMAND rm -rf ${of_pyscript_dir}/oneflow/generated
  COMMAND export PYTHONPATH=${of_pyscript_dir}:$PYTHONPATH && ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_oneflow_api.py --root_path=${of_pyscript_dir}/oneflow/generated)
add_dependencies(generate_api of_pyscript_copy)
add_dependencies(generate_api oneflow_internal)
# get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${include_dirs})
#   message("-I'${dir}' ")
# endforeach()

# build main
set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
foreach(cc ${of_main_cc})
  get_filename_component(main_name ${cc} NAME_WE)
  oneflow_add_executable(${main_name} ${cc})
  target_link_libraries(${main_name} ${of_libs} ${oneflow_third_party_libs})
  set_target_properties(${main_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endforeach()

# build test
if(BUILD_TESTING)
  if(NOT BUILD_CUDA)
    message(FATAL_ERROR "BUILD_TESTING without BUILD_CUDA")
  endif()
  if (of_all_test_cc)
    oneflow_add_executable(oneflow_testexe ${of_all_test_cc})
    target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs})
    set_target_properties(oneflow_testexe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    add_test(NAME oneflow_test COMMAND oneflow_testexe)
    #  foreach(cc ${of_all_test_cc})
    #    get_filename_component(test_name ${cc} NAME_WE)
    #    string(CONCAT test_exe_name ${test_name} exe)
    #    oneflow_add_executable(${test_exe_name} ${cc})
    #    target_link_libraries(${test_exe_name} ${of_libs} ${oneflow_third_party_libs})
    #  endforeach()
  endif()
endif()

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

copy_files("${PROTO_HDRS}" "${PROJECT_BINARY_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)

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
copy_files("${OF_CORE_HDRS}" "${PROJECT_SOURCE_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)
