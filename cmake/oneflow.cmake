include(python)

function(oneflow_add_executable)
  add_executable(${ARGV})
  set_compile_options_to_oneflow_target(${ARGV0})
endfunction()

function(oneflow_add_library)
  add_library(${ARGV})
  set_compile_options_to_oneflow_target(${ARGV0})
endfunction()

# source_group
if(WIN32)
  set(oneflow_platform "windows")
  list(APPEND oneflow_platform_excludes "linux")
else()
  set(oneflow_platform "linux")
  list(APPEND oneflow_platform_excludes "windows")
endif()

file(GLOB_RECURSE oneflow_all_hdr_to_be_expanded "${PROJECT_SOURCE_DIR}/oneflow/core/*.e.h"
     "${PROJECT_SOURCE_DIR}/oneflow/python/*.e.h")
foreach(oneflow_hdr_to_be_expanded ${oneflow_all_hdr_to_be_expanded})
  file(RELATIVE_PATH of_ehdr_rel_path ${PROJECT_SOURCE_DIR} ${oneflow_hdr_to_be_expanded})
  set(of_e_h_expanded "${PROJECT_BINARY_DIR}/${of_ehdr_rel_path}.expanded.h")
  if(WIN32)
    error("Expanding macro in WIN32 is not supported yet")
  else()
    add_custom_command(
      OUTPUT ${of_e_h_expanded}
      COMMAND ${CMAKE_C_COMPILER} ARGS -E -I"${PROJECT_SOURCE_DIR}" -I"${PROJECT_BINARY_DIR}" -o
              "${of_e_h_expanded}" "${oneflow_hdr_to_be_expanded}"
      DEPENDS ${oneflow_hdr_to_be_expanded}
      COMMENT "Expanding macros in ${oneflow_hdr_to_be_expanded}")
    list(APPEND oneflow_all_hdr_expanded "${of_e_h_expanded}")
  endif()
  set_source_files_properties(${oneflow_all_hdr_expanded} PROPERTIES GENERATED TRUE)
endforeach()

file(
  GLOB_RECURSE
  oneflow_all_src
  "${PROJECT_SOURCE_DIR}/oneflow/core/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/user/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/api/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/maybe/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/extension/*.*")

foreach(oneflow_single_file ${oneflow_all_src})
  # Verify whether this file is for other platforms
  set(exclude_this OFF)
  set(group_this OFF)
  foreach(oneflow_platform_exclude ${oneflow_platform_excludes})
    string(FIND ${oneflow_single_file} ${oneflow_platform_exclude} platform_found)
    if(NOT ${platform_found} EQUAL -1) # the ${oneflow_single_file} is for other platforms
      set(exclude_this ON)
    endif()
  endforeach()
  # If this file is for other platforms, just exclude it from current project
  if(exclude_this)
    continue()
  endif()

  if("${oneflow_single_file}" MATCHES
     "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|maybe)/.*\\.(h|hpp)$")
    if((NOT RPC_BACKEND MATCHES "GRPC") AND "${oneflow_single_file}" MATCHES
                                            "^${PROJECT_SOURCE_DIR}/oneflow/core/control/.*")
      # skip if GRPC not enabled
    elseif(APPLE AND "${oneflow_single_file}" MATCHES
                     "^${PROJECT_SOURCE_DIR}/oneflow/core/comm_network/(epoll|ibverbs)/.*")
      # skip if macOS
    else()
      list(APPEND of_all_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user)/.*\\.(cuh|cu)$")
    if(BUILD_CUDA)
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user)/.*\\.proto$")
    list(APPEND of_all_proto ${oneflow_single_file})
    #list(APPEND of_all_obj_cc ${oneflow_single_file})   # include the proto file in the project
    set(group_this ON)
  endif()

  if(BUILD_PYTHON)

    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/api/python/.*\\.(h|cpp)$")
      list(APPEND of_pybind_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()

    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/extension/.*\\.(c|h|cpp)$")
      list(APPEND of_pyext_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()
  endif(BUILD_PYTHON)

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|maybe)/.*\\.cpp$")
    if("${oneflow_single_file}" MATCHES
       "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|maybe|thread)/.*_test\\.cpp$")
      # test file
      list(APPEND of_all_test_cc ${oneflow_single_file})
    elseif(APPLE AND "${oneflow_single_file}" MATCHES
                     "^${PROJECT_SOURCE_DIR}/oneflow/core/comm_network/(epoll|ibverbs)/.*")
      # skip if macOS
    elseif(APPLE AND "${oneflow_single_file}" MATCHES
                     "^${PROJECT_SOURCE_DIR}/oneflow/core/transport/.*")
      # skip if macOS
    elseif((NOT RPC_BACKEND MATCHES "GRPC") AND "${oneflow_single_file}" MATCHES
                                                "^${PROJECT_SOURCE_DIR}/oneflow/core/control.*")
      # skip if GRPC not enabled
    else()
      list(APPEND of_all_obj_cc ${oneflow_single_file})
    endif()
    set(group_this ON)
  endif()
  if(group_this)
    file(RELATIVE_PATH oneflow_relative_file ${PROJECT_SOURCE_DIR}/oneflow/core/
         ${oneflow_single_file})
    get_filename_component(oneflow_relative_path ${oneflow_relative_file} PATH)
    string(REPLACE "/" "\\" group_name ${oneflow_relative_path})
    source_group("${group_name}" FILES ${oneflow_single_file})
  endif()
endforeach()

# clang format
add_custom_target(
  of_format
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i
          ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_license_format.py -i
          ${ONEFLOW_PYTHON_DIR} --fix --exclude="oneflow/include" --exclude="oneflow/core"
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --source_dir
          ${CMAKE_CURRENT_SOURCE_DIR}/oneflow --fix --quiet
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_py_format.py --source_dir
          ${CMAKE_CURRENT_SOURCE_DIR}/python --fix
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py
          --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/tools/oneflow-tblgen --fix --quiet)
# clang tidy
set(RUN_CLANG_TIDY_ARGS --build_dir ${CMAKE_BINARY_DIR})
if(MAYBE_NEED_ERROR_MSG_CHECK)
  list(APPEND RUN_CLANG_TIDY_ARGS --check-error-msg)
endif()
message(STATUS "RUN_CLANG_TIDY_ARGS: ${RUN_CLANG_TIDY_ARGS}")
add_custom_target(
  of_tidy COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/ci/check/run_clang_tidy.py
                  ${RUN_CLANG_TIDY_ARGS} DEPENDS of_git_version oneflow_deps of_functional_obj
                                                 of_functional_tensor_obj)
# generate version
set(OF_GIT_VERSION_DIR ${CMAKE_CURRENT_BINARY_DIR}/of_git_version)
set(OF_GIT_VERSION_FILE ${OF_GIT_VERSION_DIR}/version.cpp)
set(OF_GIT_VERSION_DUMMY_FILE ${OF_GIT_VERSION_DIR}/_version.cpp)
add_custom_target(of_git_version_create_dir COMMAND ${CMAKE_COMMAND} -E make_directory
                                                    ${OF_GIT_VERSION_DIR})
add_custom_command(
  OUTPUT ${OF_GIT_VERSION_DUMMY_FILE}
  COMMAND ${CMAKE_COMMAND} -DOF_GIT_VERSION_FILE=${OF_GIT_VERSION_FILE}
          -DOF_GIT_VERSION_ROOT=${PROJECT_SOURCE_DIR} -DBUILD_GIT_VERSION=${BUILD_GIT_VERSION} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  DEPENDS of_git_version_create_dir)
add_custom_target(of_git_version DEPENDS ${OF_GIT_VERSION_DUMMY_FILE})
set_source_files_properties(${OF_GIT_VERSION_FILE} PROPERTIES GENERATED TRUE)
list(APPEND of_all_obj_cc ${OF_GIT_VERSION_FILE})

set(of_proto_python_dir "${PROJECT_BINARY_DIR}/of_proto_python")

# proto obj lib
add_custom_target(make_pyproto_dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory
                                               ${of_proto_python_dir})
foreach(proto_name ${of_all_proto})
  file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
  list(APPEND of_all_rel_protos ${proto_rel_name})
endforeach()

relative_protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR} ${of_all_rel_protos})

oneflow_add_library(of_protoobj SHARED ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(of_protoobj make_pyproto_dir protobuf)
target_link_libraries(of_protoobj protobuf_imported)

include(functional)
generate_functional_api_and_pybind11_cpp(FUNCTIONAL_GENERATED_SRCS FUNCTIONAL_GENERATED_HRCS
                                         FUNCTIONAL_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})
oneflow_add_library(of_functional_obj STATIC ${FUNCTIONAL_GENERATED_SRCS}
                    ${FUNCTIONAL_GENERATED_HRCS})
target_link_libraries(of_functional_obj LLVMSupportWithHeader glog::glog fmt)
add_dependencies(of_functional_obj prepare_oneflow_third_party)

if(BUILD_PYTHON)

  generate_functional_tensor_api_and_pybind11_cpp(
    FUNCTIONAL_TENSOR_GENERATED_SRCS FUNCTIONAL_TENSOR_GENERATED_HRCS
    FUNCTIONAL_TENSOR_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})

  generate_functional_dispatch_stateful_ops_and_pybind11_cpp(
    FUNCTIONAL_OPS_GENERATED_SRCS FUNCTIONAL_OPS_GENERATED_HRCS FUNCTIONAL_OPS_PYBIND11_SRCS
    ${PROJECT_SOURCE_DIR})

  oneflow_add_library(
    of_functional_tensor_obj STATIC ${FUNCTIONAL_TENSOR_GENERATED_SRCS}
    ${FUNCTIONAL_TENSOR_GENERATED_HRCS} ${FUNCTIONAL_OPS_GENERATED_SRCS}
    ${FUNCTIONAL_OPS_GENERATED_HRCS})
  target_link_libraries(of_functional_tensor_obj LLVMSupportWithHeader glog::glog fmt)
  add_dependencies(of_functional_tensor_obj prepare_oneflow_third_party)
  target_include_directories(of_functional_tensor_obj PRIVATE ${Python_INCLUDE_DIRS}
                                                              ${Python_NumPy_INCLUDE_DIRS})

  set(PYBIND11_SRCS ${FUNCTIONAL_PYBIND11_SRCS} ${FUNCTIONAL_TENSOR_PYBIND11_SRCS}
                    ${FUNCTIONAL_OPS_PYBIND11_SRCS})

endif(BUILD_PYTHON)

include_directories(${PROJECT_SOURCE_DIR}) # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})

# cc obj lib
oneflow_add_library(oneflow SHARED ${of_all_obj_cc})

add_dependencies(oneflow of_protoobj)
add_dependencies(oneflow of_functional_obj)
add_dependencies(oneflow of_op_schema)
add_dependencies(oneflow of_git_version)

if(USE_CLANG_FORMAT)
  add_dependencies(oneflow of_format)
endif()
if(USE_CLANG_TIDY)
  add_dependencies(oneflow of_tidy)
endif()

target_compile_definitions(oneflow PRIVATE GOOGLE_LOGGING)

set(ONEFLOW_TOOLS_DIR "${PROJECT_BINARY_DIR}/tools"
    CACHE STRING "dir to put binary for debugging and development")

# clean cache for last LLVM version
if("${LLVM_MONO_REPO_URL}" STREQUAL
   "https://github.com/llvm/llvm-project/archive/c63522e6ba7782c335043893ae7cbd37eca24fe5.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/a0595f8c99a253c65f30a151337e7aadc19ee3a1.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/7eaa84eac3ba935d13f4267d3d533a6c3e1283ed.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/35e60f5de180aea55ed478298f4b40f04dcc57d1.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/6a9bbd9f20dcd700e28738788bb63a160c6c088c.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/32805e60c9de1f82887cd2af30d247dcabd2e1d3.zip"
   OR "${LLVM_MONO_REPO_URL}" STREQUAL
      "https://github.com/llvm/llvm-project/archive/6d6268dcbf0f48e43f6f9fe46b3a28c29ba63c7d.zip"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "f2f17229cf21049663b8ef4f2b6b8062"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "6b7c6506d5922de9632c8ff012b2f945"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "e0ea669a9f0872d35bffda5ec6c5ac6f"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "241a333828bba1efa35aff4c4fc2ce87"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "075fbfdf06cb3f02373ea44971af7b03"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "e412dc61159b5e929b0c94e44b11feb2"
   OR "${LLVM_MONO_REPO_MD5}" STREQUAL "334997b4879aba15d9323a732356cf2a")
  unset(LLVM_MONO_REPO_URL CACHE)
  unset(LLVM_MONO_REPO_MD5 CACHE)
endif()
set(LLVM_MONO_REPO_URL "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-15.0.6.zip"
    CACHE STRING "")
use_mirror(VARIABLE LLVM_MONO_REPO_URL URL ${LLVM_MONO_REPO_URL})
set(LLVM_MONO_REPO_MD5 "1ccc00accc87a1a5d42a275d6e31cd8c" CACHE STRING "")
set(ONEFLOW_BUILD_ROOT_DIR "${PROJECT_BINARY_DIR}")
add_subdirectory(${PROJECT_SOURCE_DIR}/oneflow/ir)
if(WITH_MLIR)
  set(ONEFLOW_MLIR_LIBS -Wl,--no-as-needed MLIROneFlowExtension -Wl,--as-needed)
endif()

if("${LLVM_PROVIDER}" STREQUAL "install")
  get_property(LLVM_INSTALL_DIR GLOBAL PROPERTY LLVM_INSTALL_DIR)
  check_variable_defined(LLVM_INSTALL_DIR)
  find_library(LLVMSupportLib LLVMSupport PATHS ${LLVM_INSTALL_DIR}/lib REQUIRED)
  add_library(LLVMSupportWithHeader UNKNOWN IMPORTED)
  set_property(TARGET LLVMSupportWithHeader PROPERTY IMPORTED_LOCATION ${LLVMSupportLib})
else()
  add_library(LLVMSupportWithHeader INTERFACE IMPORTED)
  target_link_libraries(LLVMSupportWithHeader INTERFACE LLVMSupport)
endif()
check_variable_defined(LLVM_INCLUDE_DIRS)
set_property(TARGET LLVMSupportWithHeader PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                   ${LLVM_INCLUDE_DIRS})

list(APPEND oneflow_third_party_libs LLVMSupportWithHeader)

include(op_schema)

get_property(EXTERNAL_TARGETS GLOBAL PROPERTY EXTERNAL_TARGETS)

if(APPLE)
  set(of_libs -Wl,-force_load oneflow of_op_schema)
  target_link_libraries(oneflow of_protoobj of_functional_obj ${oneflow_third_party_libs})
elseif(UNIX)
  set(of_libs -Wl,--whole-archive oneflow of_op_schema -Wl,--no-whole-archive -ldl -lrt)
  target_link_libraries(
    oneflow
    of_protoobj
    of_functional_obj
    ${oneflow_third_party_libs}
    ${EXTERNAL_TARGETS}
    -Wl,--no-whole-archive
    -Wl,--as-needed
    -ldl
    -lrt)
  if(BUILD_CUDA)
    target_link_libraries(oneflow CUDA::cudart_static)
  endif()
  if(WITH_OMP)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(oneflow OpenMP::OpenMP_CXX)
    endif()
  endif()
elseif(WIN32)
  set(of_libs oneflow of_protoobj of_functional_obj of_op_schema)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:oneflow")
endif()

if(BUILD_CUDA)
  string(JOIN "," CUDA_REAL_ARCHS ${CUDA_REAL_ARCHS_LIST})
  set_source_files_properties(${PROJECT_SOURCE_DIR}/oneflow/core/hardware/cuda_device_descriptor.cpp
                              PROPERTIES COMPILE_FLAGS "-DCUDA_REAL_ARCHS=\"${CUDA_REAL_ARCHS}\"")
endif()

if(BUILD_CUDA AND WITH_CUTLASS)
  if(CUDA_VERSION VERSION_GREATER_EQUAL "10.1")
    add_definitions(-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1)
  endif()

  set_property(SOURCE ${PROJECT_SOURCE_DIR}/oneflow/user/kernels/fused_attention_kernels.cu APPEND
               PROPERTY INCLUDE_DIRECTORIES ${CUTLASS_INSTALL_DIR}/examples/xformers_fmha)
  set_property(SOURCE ${PROJECT_SOURCE_DIR}/oneflow/user/kernels/fused_glu_kernel.cu APPEND
               PROPERTY INCLUDE_DIRECTORIES ${CUTLASS_INSTALL_DIR}/examples/45_dual_gemm)
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    set_property(
      SOURCE
        ${PROJECT_SOURCE_DIR}/oneflow/user/kernels/fused_multi_head_attention_inference_kernel.cu
      APPEND
      PROPERTY COMPILE_OPTIONS "--use_fast_math")
  endif()
endif()

# oneflow api common
if(BUILD_PYTHON OR BUILD_CPP_API)
  file(GLOB_RECURSE of_api_common_files ${PROJECT_SOURCE_DIR}/oneflow/api/common/*.h
       ${PROJECT_SOURCE_DIR}/oneflow/api/common/*.cpp)
  oneflow_add_library(of_api_common OBJECT ${of_api_common_files})
  target_link_libraries(of_api_common oneflow)
  if(WITH_MLIR)
    target_link_libraries(of_api_common ${ONEFLOW_MLIR_LIBS})
  endif()
endif()

if(BUILD_PYTHON)

  # py ext lib
  # This library should be static to make sure all python symbols are included in the final ext shared lib,
  # so that it is safe to do wheel audits of multiple pythons version in parallel.
  oneflow_add_library(of_pyext_obj STATIC ${of_pyext_obj_cc})
  target_include_directories(of_pyext_obj PRIVATE ${Python_INCLUDE_DIRS}
                                                  ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(of_pyext_obj oneflow pybind11::headers)
  if(BUILD_SHARED_LIBS AND APPLE)
    target_link_libraries(of_pyext_obj ${Python3_LIBRARIES})
  endif()
  add_dependencies(of_pyext_obj oneflow)

  pybind11_add_module(oneflow_internal ${PYBIND11_SRCS} ${of_pybind_obj_cc} ${PYBIND_REGISTRY_CC})
  set_property(TARGET oneflow_internal APPEND PROPERTY BUILD_RPATH "\$ORIGIN/../nvidia/cublas/lib")
  set_property(TARGET oneflow_internal APPEND PROPERTY BUILD_RPATH "\$ORIGIN/../nvidia/cudnn/lib")
  set_compile_options_to_oneflow_target(oneflow_internal)
  set_property(TARGET oneflow_internal PROPERTY CXX_VISIBILITY_PRESET "default")
  add_dependencies(oneflow_internal of_functional_obj of_functional_tensor_obj of_op_schema)
  set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
  set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                    "${ONEFLOW_PYTHON_DIR}/oneflow")
  target_link_libraries(
    oneflow_internal PRIVATE ${of_libs} of_functional_tensor_obj of_api_common
                             ${oneflow_third_party_libs} of_pyext_obj glog::glog)
  target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS}
                                                      ${Python_NumPy_INCLUDE_DIRS})

  if(WITH_MLIR)
    add_dependencies(check-oneflow oneflow_internal)
  endif(WITH_MLIR)

  set(gen_pip_args "")
  if(BUILD_CUDA)
    list(APPEND gen_pip_args --cuda=${CUDA_VERSION})
  endif()

  add_custom_target(
    of_pyscript_copy ALL
    COMMAND ${CMAKE_COMMAND} -E touch "${of_proto_python_dir}/oneflow/core/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${of_proto_python_dir}/oneflow/core"
            "${ONEFLOW_PYTHON_DIR}/oneflow/core"
    COMMAND
      ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/generate_pip_version.py ${gen_pip_args}
      --src=${PROJECT_SOURCE_DIR} --cmake_project_binary_dir=${PROJECT_BINARY_DIR}
      --out=${ONEFLOW_PYTHON_DIR}/oneflow/version.py)

  # source this file to add oneflow in PYTHONPATH
  file(WRITE "${PROJECT_BINARY_DIR}/source.sh"
       "export PYTHONPATH=${ONEFLOW_PYTHON_DIR}:$PYTHONPATH")

  add_dependencies(of_pyscript_copy of_protoobj)

endif(BUILD_PYTHON)

if(BUILD_CPP_API)
  file(GLOB_RECURSE of_cpp_api_files ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/*.cpp
       ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/*.h)
  list(FILTER of_cpp_api_files EXCLUDE REGEX "oneflow/api/cpp/tests")
  oneflow_add_library(oneflow_cpp SHARED ${of_cpp_api_files})
  set_target_properties(oneflow_cpp PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIBONEFLOW_LIBRARY_DIR}"
                                               LIBRARY_OUTPUT_DIRECTORY "${LIBONEFLOW_LIBRARY_DIR}")
  target_link_libraries(oneflow_cpp PRIVATE ${of_libs} of_api_common ${oneflow_third_party_libs})
endif()

file(RELATIVE_PATH PROJECT_BINARY_DIR_RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

function(oneflow_add_test target_name)
  cmake_parse_arguments(arg "" "TEST_NAME;WORKING_DIRECTORY" "SRCS" ${ARGN})
  oneflow_add_executable(${target_name} ${arg_SRCS})
  if(BUILD_CUDA)
    target_link_libraries(${target_name} CUDA::cudart_static)
  endif()
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${PROJECT_BINARY_DIR}/bin")
  add_test(NAME ${arg_TEST_NAME} COMMAND ${target_name} WORKING_DIRECTORY ${arg_WORKING_DIRECTORY})
  set_tests_properties(
    ${arg_TEST_NAME} PROPERTIES ENVIRONMENT
                                "HTTP_PROXY='';HTTPS_PROXY='';http_proxy='';https_proxy='';")
endfunction()

# build test
if(BUILD_TESTING)
  if(of_all_test_cc)
    oneflow_add_test(oneflow_testexe SRCS ${of_all_test_cc} TEST_NAME oneflow_test)
    target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs} glog::glog
                          ${oneflow_test_libs})
    if(WITH_MLIR)
      target_link_libraries(oneflow_testexe MLIROneFlowExtension)
    endif()
  endif()

  if(BUILD_CPP_API)
    file(GLOB_RECURSE cpp_api_test_files ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/tests/*.cpp)
    oneflow_add_test(
      oneflow_cpp_api_testexe
      SRCS
      ${cpp_api_test_files}
      TEST_NAME
      oneflow_cpp_api_test
      WORKING_DIRECTORY
      ${PROJECT_SOURCE_DIR})
    find_package(Threads REQUIRED)
    target_link_libraries(oneflow_cpp_api_testexe oneflow_cpp ${oneflow_third_party_libs}
                          ${oneflow_test_libs} Threads::Threads)
  endif()
endif()

# build include
add_custom_target(of_include_copy ALL)

if(BUILD_PYTHON)

  add_dependencies(of_include_copy oneflow_internal of_pyscript_copy)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/oneflow/core
    DESTINATION ${ONEFLOW_INCLUDE_DIR}/oneflow
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL FILES_MATCHING
    PATTERN *.h
    PATTERN *.hpp)
  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/oneflow
    DESTINATION ${ONEFLOW_INCLUDE_DIR}
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL FILES_MATCHING
    REGEX "oneflow/core/common/.+(h|hpp)$"
    REGEX "oneflow/core/device/.+(h|hpp)$"
    REGEX "oneflow/core/framework/.+(h|hpp)$"
    REGEX "oneflow/core/kernel/util/.+(h|hpp)$"
    REGEX "oneflow/core/persistence/.+(h|hpp)$"
    REGEX "oneflow/core/ep/include/.+(h|hpp)$"
    REGEX "oneflow/core/ep/cuda/.+(h|hpp)$"
    REGEX "oneflow/core/job/.+(h|hpp)$"
    REGEX "oneflow/core/.+(proto)$"
    PATTERN "oneflow/core/kernel/new_kernel_util.h"
    PATTERN "oneflow/core/kernel/kernel_context.h"
    PATTERN "oneflow/core/kernel/kernel_observer.h"
    PATTERN "oneflow/core/kernel/kernel_util.cuh"
    PATTERN "oneflow/core/common/symbol.h"
    PATTERN "oneflow/core/autograd/autograd_meta.h"
    PATTERN "oneflow/core/register/blob_desc.h"
    PATTERN "oneflow/core/operator/operator.h"
    PATTERN "oneflow/core/operator/op_conf_util.h"
    PATTERN "oneflow/core/graph/graph.h"
    PATTERN "oneflow/core/graph/node.h"
    PATTERN "oneflow/core/graph/op_graph.h"
    PATTERN "oneflow/core/graph/task_id.h"
    PATTERN "oneflow/core/graph/task_id_generator.h"
    PATTERN "oneflow/core/graph/stream_id.h"
    PATTERN "oneflow/core/vm/vm_sync.h"
    PATTERN "oneflow/api" EXCLUDE
    PATTERN "oneflow/user" EXCLUDE
    PATTERN "oneflow/extension" EXCLUDE
    PATTERN "oneflow/maybe" EXCLUDE
    PATTERN "oneflow/core/lazy" EXCLUDE
    PATTERN "oneflow/core/graph_impl" EXCLUDE
    PATTERN "oneflow/core/job_rewriter" EXCLUDE
    PATTERN "oneflow/core/hardware" EXCLUDE
    PATTERN "oneflow/core/intrusive" EXCLUDE
    PATTERN "oneflow/core/stream" EXCLUDE
    PATTERN "oneflow/core/functional" EXCLUDE
    PATTERN "oneflow/core/platform" EXCLUDE
    PATTERN "oneflow/core/boxing" EXCLUDE
    PATTERN "oneflow/core/rpc" EXCLUDE
    PATTERN "oneflow/core/profiler" EXCLUDE
    PATTERN "oneflow/core/transport" EXCLUDE
    PATTERN "oneflow/core/comm_network" EXCLUDE
    PATTERN "oneflow/ir" EXCLUDE)
  add_custom_target(
    install_oneflow_py_include
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=oneflow_py_include -P
            "${CMAKE_BINARY_DIR}/cmake_install.cmake" DEPENDS oneflow_internal)
  add_custom_target(oneflow_py ALL)
  add_dependencies(oneflow_py of_include_copy install_oneflow_py_include)

endif(BUILD_PYTHON)

if(BUILD_CPP_API)

  set(LIBONEFLOW_DIR ${PROJECT_BINARY_DIR}/liboneflow_cpp)

  install(
    DIRECTORY oneflow/api/cpp/
    COMPONENT oneflow_cpp_all
    DESTINATION include/oneflow
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "tests" EXCLUDE)
  set(LIBONEFLOW_THIRD_PARTY_DIRS)
  checkdirandappendslash(DIR ${PROTOBUF_LIBRARY_DIR} OUTPUT PROTOBUF_LIBRARY_DIR_APPENDED)
  list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${PROTOBUF_LIBRARY_DIR_APPENDED})
  if(BUILD_CUDA)
    checkdirandappendslash(DIR ${NCCL_LIBRARY_DIR} OUTPUT NCCL_LIBRARY_DIR_APPENDED)
    list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${NCCL_LIBRARY_DIR_APPENDED})
    checkdirandappendslash(DIR ${TRT_FLASH_ATTENTION_LIBRARY_DIR} OUTPUT
                           TRT_FLASH_ATTENTION_LIBRARY_DIR_APPENDED)
    list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${TRT_FLASH_ATTENTION_LIBRARY_DIR_APPENDED})
    if(WITH_CUTLASS)
      checkdirandappendslash(DIR ${CUTLASS_LIBRARY_DIR} OUTPUT CUTLASS_LIBRARY_DIR_APPENDED)
      list(APPEND LIBONEFLOW_THIRD_PARTY_DIRS ${CUTLASS_LIBRARY_DIR_APPENDED})
    endif()
  endif()

  install(
    DIRECTORY ${LIBONEFLOW_THIRD_PARTY_DIRS}
    COMPONENT oneflow_cpp_all
    DESTINATION lib
    FILES_MATCHING
    PATTERN "*.so*"
    PATTERN "*.a" EXCLUDE
    PATTERN "libprotobuf-lite.so*" EXCLUDE
    PATTERN "libprotoc.so*" EXCLUDE
    PATTERN "cmake" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE)

  install(FILES ${PROJECT_SOURCE_DIR}/cmake/oneflow-config.cmake COMPONENT oneflow_cpp_all
          DESTINATION share)

  get_property(MLIR_RELATED_TARGETS GLOBAL PROPERTY MLIR_EXPORTS)
  get_property(LLVM_RELATED_TARGETS GLOBAL PROPERTY LLVM_EXPORTS)

  list(
    REMOVE_ITEM
    LLVM_RELATED_TARGETS
    count
    not
    FileCheck
    lli-child-target
    llvm-jitlink-executor
    llvm-PerfectShuffle
    llvm-tblgen
    mlir-tblgen
    mlir-pdll
    obj2yaml
    oneflow_tblgen
    yaml-bench
    yaml2obj)

  set(LIBONEFLOW_TARGETS)
  list(
    APPEND
    LIBONEFLOW_TARGETS
    oneflow_cpp
    oneflow
    of_protoobj
    glog
    ${MLIR_RELATED_TARGETS}
    ${LLVM_RELATED_TARGETS}
    ${EXTERNAL_TARGETS})

  if(BUILD_TESTING AND BUILD_SHARED_LIBS)
    list(APPEND LIBONEFLOW_TARGETS gtest_main gtest)
  endif()

  if(BUILD_TESTING)
    list(APPEND LIBONEFLOW_TARGETS oneflow_cpp_api_testexe)
    list(APPEND LIBONEFLOW_TARGETS oneflow_testexe)
  endif(BUILD_TESTING)

  install(
    TARGETS ${LIBONEFLOW_TARGETS}
    COMPONENT oneflow_cpp_all
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

  add_custom_target(
    install_oneflow_cpp
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=oneflow_cpp_all
            -DCMAKE_INSTALL_PREFIX="${LIBONEFLOW_DIR}" -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    DEPENDS oneflow_cpp)
  if(BUILD_TESTING)
    add_dependencies(install_oneflow_cpp oneflow_cpp_api_testexe oneflow_testexe)
  endif(BUILD_TESTING)
  add_dependencies(of_include_copy install_oneflow_cpp)

  string(TOLOWER ${CMAKE_SYSTEM_NAME} CPACK_SYSTEM_NAME)
  set(CPACK_GENERATOR ZIP)
  set(CPACK_PACKAGE_DIRECTORY ${PROJECT_BINARY_DIR}/cpack)
  set(CPACK_PACKAGE_NAME liboneflow)
  # TODO: by Shenghang, unify python and c++ version genenerating and getting
  set(CPACK_PACKAGE_VERSION ${ONEFLOW_CURRENT_VERSION})
  set(CPACK_INSTALL_CMAKE_PROJECTS ${PROJECT_BINARY_DIR};oneflow;oneflow_cpp_all;/)
  include(CPack)
endif(BUILD_CPP_API)
