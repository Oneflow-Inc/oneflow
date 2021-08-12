include(python)
include(CheckCXXCompilerFlag)

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

function(target_try_compile_option target flag)
  # We cannot check for -Wno-foo as this won't throw a warning so we must check for the -Wfoo option directly
  # http://stackoverflow.com/questions/38785168/cc1plus-unrecognized-command-line-option-warning-on-any-other-warning
  string(REGEX REPLACE "^-Wno-" "-W" checkedFlag ${flag})
  string(REGEX REPLACE "[-=]" "_" varName CXX_FLAG${checkedFlag})
  # Avoid double checks. A compiler will not magically support a flag it did not before
  if(NOT DEFINED ${varName}_SUPPORTED)
    check_cxx_compiler_flag(${checkedFlag} ${varName}_SUPPORTED)
  endif()
  if (${varName}_SUPPORTED)
    target_compile_options(${target} PRIVATE ${flag})
  endif ()
endfunction()

function(target_try_compile_options target)
  foreach(flag ${ARGN})
    target_try_compile_option(${target} ${flag})
  endforeach()
endfunction()

function(target_treat_warnings_as_errors target)
  if (TREAT_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE -Werror)

    # TODO: remove it while fixing all deprecated call
    target_try_compile_options(${target} -Wno-error=deprecated-declarations)

    # disable unused-* for different compile mode (maybe unused in cpu.cmake, but used in cuda.cmake)
    target_try_compile_options(${target} 
      -Wno-error=unused-const-variable 
      -Wno-error=unused-variable
      -Wno-error=unused-local-typedefs
      -Wno-error=unused-private-field
      -Wno-error=unused-lambda-capture
    )

    target_try_compile_options(${target} -Wno-error=instantiation-after-specialization)

    # the mangled name between `struct X` and `class X` is different in MSVC ABI, remove it while windows is supported (in MSVC/cl or clang-cl)
    target_try_compile_options(${target} -Wno-error=mismatched-tags)

    # disable for pointer operations of intrusive linked lists
    target_try_compile_options(${target} -Wno-error=array-bounds)

    # avoid check of memcpy for non-trivial types in opencv headers
    target_try_compile_options(${target} -Wno-error=class-memaccess)

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
    if((NOT RPC_BACKEND MATCHES "GRPC") AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/control/.*")
      # skip if GRPC not enabled
    elseif(APPLE AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/comm_network/(epoll|ibverbs)/.*")
      # skip if macOS
    else()
      list(APPEND of_all_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.hpp$")
    list(APPEND of_all_obj_cc ${oneflow_single_file})
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.cuh$")
    if(BUILD_CUDA)
      list(APPEND of_cuda_src ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.cu$")
    if(BUILD_CUDA)
      list(APPEND of_cuda_src ${oneflow_single_file})
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
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*_test\\.cpp$")
      # test file
      list(APPEND of_all_test_cc ${oneflow_single_file})
    elseif(APPLE AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/comm_network/(epoll|ibverbs)/.*")
      # skip if macOS
    elseif(APPLE AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/transport/.*")
      # skip if macOS
    elseif((NOT RPC_BACKEND MATCHES "GRPC") AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/control.*")
      # skip if GRPC not enabled
    else()
      list(APPEND of_all_obj_cc ${oneflow_single_file})
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
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i ${ONEFLOW_PYTHON_DIR} --fix --exclude="oneflow/include" --exclude="oneflow/core"
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix --quiet
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_py_format.py --source_dir ${CMAKE_CURRENT_SOURCE_DIR} --fix
  )

# generate version
set(OF_GIT_VERSION_DIR ${CMAKE_CURRENT_BINARY_DIR}/of_git_version)
set(OF_GIT_VERSION_FILE ${OF_GIT_VERSION_DIR}/version.cpp)
set(OF_GIT_VERSION_DUMMY_FILE ${OF_GIT_VERSION_DIR}/_version.cpp)
add_custom_target(of_git_version_create_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OF_GIT_VERSION_DIR})
add_custom_command(
        OUTPUT ${OF_GIT_VERSION_DUMMY_FILE}
        COMMAND ${CMAKE_COMMAND} -DOF_GIT_VERSION_FILE=${OF_GIT_VERSION_FILE}
          -DOF_GIT_VERSION_ROOT=${PROJECT_SOURCE_DIR}
          -DBUILD_GIT_VERSION=${BUILD_GIT_VERSION}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
        DEPENDS of_git_version_create_dir)
add_custom_target(of_git_version
        DEPENDS ${OF_GIT_VERSION_DUMMY_FILE})
set_source_files_properties(${OF_GIT_VERSION_FILE} PROPERTIES GENERATED TRUE)
list(APPEND of_all_obj_cc ${OF_GIT_VERSION_FILE})

set(of_proto_python_dir "${PROJECT_BINARY_DIR}/of_proto_python")

# proto obj lib
add_custom_target(make_pyproto_dir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${of_proto_python_dir}
	)
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${PROJECT_SOURCE_DIR}
                               ${of_all_rel_protos})

oneflow_add_library(of_protoobj ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(of_protoobj make_pyproto_dir protobuf)

# cfg obj lib
include(cfg)
GENERATE_CFG_AND_PYBIND11_CPP(CFG_SRCS CFG_HRCS CFG_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})
oneflow_add_library(of_cfgobj ${CFG_SRCS} ${CFG_HRCS})
add_dependencies(of_cfgobj of_protoobj generate_cfg)
if (BUILD_SHARED_LIBS)
  target_link_libraries(of_protoobj protobuf_imported)
  target_link_libraries(of_cfgobj protobuf_imported)
  target_link_libraries(of_cfgobj of_protoobj)
else()
  # For some unknown reasons, when building static libraries, we have to link of_protoobj and of_cfgobj with oneflow_third_party_libs
  target_link_libraries(of_protoobj ${oneflow_third_party_libs})
  target_link_libraries(of_cfgobj ${oneflow_third_party_libs})
endif()

include(functional)
GENERATE_FUNCTIONAL_API_AND_PYBIND11_CPP(
    FUNCTIONAL_GENERATED_SRCS FUNCTIONAL_GENERATED_HRCS FUNCTIONAL_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})
list(APPEND of_all_obj_cc ${FUNCTIONAL_GENERATED_SRCS})

set(PYBIND11_SRCS ${CFG_PYBIND11_SRCS} ${FUNCTIONAL_PYBIND11_SRCS})

include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})

if(BUILD_CUDA)
  oneflow_add_library(of_cudaobj ${of_cuda_src})
  add_dependencies(of_cudaobj of_protoobj of_cfgobj prepare_oneflow_third_party)
  target_link_libraries(of_cudaobj ${oneflow_third_party_libs})
  set(ONEFLOW_CUDA_LIBS of_cudaobj)

  target_compile_options(of_cudaobj PRIVATE -Werror=return-type)
endif()

# cc obj lib
oneflow_add_library(of_ccobj ${of_all_obj_cc})
add_dependencies(of_ccobj prepare_oneflow_third_party generate_functional)
target_link_libraries(of_ccobj ${oneflow_third_party_libs})
add_dependencies(of_ccobj of_protoobj)
add_dependencies(of_ccobj of_cfgobj)
add_dependencies(of_ccobj of_git_version)
if (USE_CLANG_FORMAT)
  add_dependencies(of_ccobj of_format)
endif()

if (BUILD_SHARED_LIBS)
  target_link_libraries(of_ccobj of_protoobj of_cfgobj ${ONEFLOW_CUDA_LIBS} glog_imported)
endif()

target_compile_options(of_ccobj PRIVATE -Werror=return-type)
target_treat_warnings_as_errors(of_ccobj)

# py ext lib
add_library(of_pyext_obj ${of_pyext_obj_cc})
target_include_directories(of_pyext_obj PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(of_pyext_obj of_ccobj)
if(BUILD_SHARED_LIBS AND APPLE)
  target_link_libraries(of_pyext_obj ${Python3_LIBRARIES})
endif()
add_dependencies(of_pyext_obj of_ccobj)
target_compile_options(of_pyext_obj PRIVATE -Werror=return-type)
target_treat_warnings_as_errors(of_pyext_obj)

if(APPLE)
  set(of_libs -Wl,-force_load ${ONEFLOW_CUDA_LIBS} of_ccobj of_protoobj of_cfgobj)
elseif(UNIX)
  set(of_libs -Wl,--whole-archive ${ONEFLOW_CUDA_LIBS} of_ccobj of_protoobj of_cfgobj -Wl,--no-whole-archive -ldl -lrt)
elseif(WIN32)
  set(of_libs of_ccobj of_protoobj of_cfgobj)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:of_ccobj")
endif()

pybind11_add_module(oneflow_internal ${PYBIND11_SRCS} ${of_pybind_obj_cc} ${PYBIND_REGISTRY_CC})
set_property(TARGET oneflow_internal PROPERTY CXX_VISIBILITY_PRESET "default")
add_dependencies(oneflow_internal of_cfgobj generate_py_cfg)
set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_PYTHON_DIR}/oneflow")
target_link_libraries(oneflow_internal PRIVATE ${of_libs} ${oneflow_third_party_libs} of_pyext_obj ${oneflow_exe_third_party_libs})
target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

target_compile_options(oneflow_internal PRIVATE -Werror=return-type)
target_treat_warnings_as_errors(oneflow_internal)

set(gen_pip_args "")
if (BUILD_CUDA)
  list(APPEND gen_pip_args --cuda=${CUDA_VERSION})
endif()
if (WITH_XLA)
  list(APPEND gen_pip_args --xla)
endif()

add_custom_target(of_pyscript_copy ALL
    COMMAND ${CMAKE_COMMAND} -E touch "${of_proto_python_dir}/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${of_proto_python_dir}/oneflow/core" "${ONEFLOW_PYTHON_DIR}/oneflow/core"
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_pip_version.py ${gen_pip_args} --src=${PROJECT_SOURCE_DIR} --out=${ONEFLOW_PYTHON_DIR}/oneflow/version.py
)

# source this file to add oneflow in PYTHONPATH
file(WRITE "${PROJECT_BINARY_DIR}/source.sh" "export PYTHONPATH=${ONEFLOW_PYTHON_DIR}:$PYTHONPATH")

add_dependencies(of_pyscript_copy of_protoobj)

file(RELATIVE_PATH PROJECT_BINARY_DIR_RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

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
  if (of_all_test_cc)
    oneflow_add_executable(oneflow_testexe ${of_all_test_cc})
    target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs})
    set_target_properties(oneflow_testexe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    add_test(NAME oneflow_test COMMAND oneflow_testexe)
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

# build include
set(ONEFLOW_INCLUDE_DIR "${ONEFLOW_PYTHON_DIR}/oneflow/include")
add_custom_target(of_include_copy
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${ONEFLOW_INCLUDE_DIR}" && ${CMAKE_COMMAND} -E make_directory "${ONEFLOW_INCLUDE_DIR}")
add_dependencies(of_include_copy oneflow_internal)
foreach(of_include_src_dir ${ONEFLOW_INCLUDE_SRC_DIRS})
  set(oneflow_all_include_file)
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
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/common/symbol.h")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/job/parallel_desc.h")
list(APPEND OF_CORE_HDRS "${PROJECT_SOURCE_DIR}/oneflow/core/autograd/autograd_meta.h")
copy_files("${OF_CORE_HDRS}" "${PROJECT_SOURCE_DIR}" "${ONEFLOW_INCLUDE_DIR}" of_include_copy)
add_custom_target(oneflow_py ALL)
add_dependencies(oneflow_py of_include_copy)
