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

file(GLOB_RECURSE oneflow_all_src "${PROJECT_SOURCE_DIR}/oneflow/core/*.*" "${PROJECT_SOURCE_DIR}/oneflow/python/*.*")
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

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.h$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cuh$")
    if(BUILD_CUDA) 
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cu$")
    if(BUILD_CUDA)
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.proto$")
    list(APPEND of_all_proto ${oneflow_single_file})
    #list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
    set(group_this ON)
  endif()
  
  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*\\.cpp$")
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/.*_test\\.cpp$")
      # test file
      # list(APPEND of_all_test_cc ${oneflow_single_file})
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
add_custom_target(of_format)

foreach(source_file ${of_all_obj_cc} ${of_main_cc} ${of_all_test_cc} ${of_python_obj_cc})
    add_custom_command(TARGET of_format PRE_BUILD
    COMMAND clang-format -i -style=file ${source_file})
endforeach()

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

oneflow_add_library(oneflow_internal SHARED ${SWIG_SRCS} ${SWIG_HDRS} ${of_main_cc})
set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python_scripts/oneflow")
target_link_libraries(oneflow_internal ${of_libs} ${oneflow_third_party_libs})
target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

set(of_pyscript_dir "${PROJECT_BINARY_DIR}/python_scripts")
file(REMOVE_RECURSE "${of_pyscript_dir}/oneflow/python")
add_custom_target(of_pyscript_copy ALL
    COMMAND "${CMAKE_COMMAND}" -E copy
        "${PROJECT_SOURCE_DIR}/oneflow/__init__.py" "${of_pyscript_dir}/oneflow/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/oneflow/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E touch "${of_pyscript_dir}/oneflow_pyproto/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${of_pyscript_dir}/oneflow/python"
    COMMAND python "${PROJECT_SOURCE_DIR}/tools/generate_oneflow_symbols_export_file.py"
        "${PROJECT_SOURCE_DIR}" "${of_pyscript_dir}/oneflow/python/__export_symbols__.py")
file(GLOB_RECURSE oneflow_all_python_file "${PROJECT_SOURCE_DIR}/oneflow/python/*.py")
foreach(oneflow_python_file ${oneflow_all_python_file})
  file(RELATIVE_PATH oneflow_python_rel_file_path "${PROJECT_SOURCE_DIR}" ${oneflow_python_file})
  add_custom_command(TARGET of_pyscript_copy POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
    "${oneflow_python_file}"
    "${of_pyscript_dir}/${oneflow_python_rel_file_path}")
endforeach()

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
