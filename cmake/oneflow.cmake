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

file(GLOB_RECURSE oneflow_all_src
  "${PROJECT_SOURCE_DIR}/oneflow/core/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/user/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/api/*.*"
  "${PROJECT_SOURCE_DIR}/oneflow/extension/python/*.*")
if (WITH_XLA OR WITH_TENSORRT OR WITH_OPENVINO)
  file(GLOB_RECURSE oneflow_xrt_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/*.*")
  if (NOT WITH_XLA)
    file(GLOB_RECURSE xla_removing_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/xla/*.*")
  endif ()
  if (NOT WITH_TENSORRT)
    file(GLOB_RECURSE trt_removing_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/tensorrt/*.*")
  endif ()
  if (NOT WITH_OPENVINO)
    file(GLOB_RECURSE openvino_removing_src "${PROJECT_SOURCE_DIR}/oneflow/xrt/openvino/*.*")
  endif ()

  list(APPEND xrt_removing_srcs ${xla_removing_src})
  list(APPEND xrt_removing_srcs ${trt_removing_src})
  list(APPEND xrt_removing_srcs ${openvino_removing_src})
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

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt|maybe)/.*\\.(h|hpp)$")
    if((NOT RPC_BACKEND MATCHES "GRPC") AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/control/.*")
      # skip if GRPC not enabled
    elseif(APPLE AND "${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/core/comm_network/(epoll|ibverbs)/.*")
      # skip if macOS
    else()
      list(APPEND of_all_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()
  endif()

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt)/.*\\.(cuh|cu)$")
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

  if(BUILD_PYTHON)

    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/api/python/.*\\.(h|cpp)$")
      list(APPEND of_pybind_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()

    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/extension/python/.*\\.(h|cpp)$")
      list(APPEND of_pyext_obj_cc ${oneflow_single_file})
      set(group_this ON)
    endif()
  endif(BUILD_PYTHON)

  if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt|maybe)/.*\\.cpp$")
    if("${oneflow_single_file}" MATCHES "^${PROJECT_SOURCE_DIR}/oneflow/(core|user|xrt|maybe)/.*_test\\.cpp$")
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
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ci/check/run_clang_format.py --source_dir ${CMAKE_CURRENT_SOURCE_DIR}/tools/oneflow-tblgen --fix --quiet
  )
# clang tidy
add_custom_target(of_tidy
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/ci/check/run_clang_tidy.py --build_dir ${CMAKE_BINARY_DIR}
  DEPENDS of_git_version oneflow_deps of_cfgobj of_functional_obj of_functional_tensor_obj
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
add_dependencies(of_cfgobj of_protoobj)
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
oneflow_add_library(of_functional_obj STATIC ${FUNCTIONAL_GENERATED_SRCS} ${FUNCTIONAL_GENERATED_HRCS})
add_dependencies(of_functional_obj of_cfgobj)
add_dependencies(of_functional_obj prepare_oneflow_third_party)

if(BUILD_PYTHON)

  GENERATE_FUNCTIONAL_TENSOR_API_AND_PYBIND11_CPP(
      FUNCTIONAL_TENSOR_GENERATED_SRCS FUNCTIONAL_TENSOR_GENERATED_HRCS
      FUNCTIONAL_TENSOR_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})

  GENERATE_FUNCTIONAL_DISPATCH_STATEFUL_OPS_AND_PYBIND11_CPP(
      FUNCTIONAL_OPS_GENERATED_SRCS FUNCTIONAL_OPS_GENERATED_HRCS
      FUNCTIONAL_OPS_PYBIND11_SRCS ${PROJECT_SOURCE_DIR})

  oneflow_add_library(of_functional_tensor_obj STATIC
      ${FUNCTIONAL_TENSOR_GENERATED_SRCS} ${FUNCTIONAL_TENSOR_GENERATED_HRCS}
      ${FUNCTIONAL_OPS_GENERATED_SRCS} ${FUNCTIONAL_OPS_GENERATED_HRCS})
  add_dependencies(of_functional_tensor_obj of_cfgobj)
  add_dependencies(of_functional_tensor_obj prepare_oneflow_third_party)
  target_include_directories(of_functional_tensor_obj PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

  set(PYBIND11_SRCS
      ${CFG_PYBIND11_SRCS}
      ${FUNCTIONAL_PYBIND11_SRCS}
      ${FUNCTIONAL_TENSOR_PYBIND11_SRCS}
      ${FUNCTIONAL_OPS_PYBIND11_SRCS})

endif(BUILD_PYTHON)

include_directories(${PROJECT_SOURCE_DIR})  # TO FIND: third_party/eigen3/..
include_directories(${PROJECT_BINARY_DIR})

# cc obj lib
oneflow_add_library(oneflow ${of_all_obj_cc})

add_dependencies(oneflow of_protoobj)
add_dependencies(oneflow of_cfgobj)
add_dependencies(oneflow of_functional_obj)
add_dependencies(oneflow of_op_schema)
add_dependencies(oneflow of_git_version)

if (USE_CLANG_FORMAT)
  add_dependencies(oneflow of_format)
endif()
if (USE_CLANG_TIDY)
  add_dependencies(oneflow of_tidy)
endif()

target_compile_definitions(oneflow PRIVATE GOOGLE_LOGGING)

set(ONEFLOW_TOOLS_DIR "${PROJECT_BINARY_DIR}/tools" CACHE STRING "dir to put binary for debugging and development")

set(LLVM_MONO_REPO_URL "https://github.com/llvm/llvm-project/archive/649d95371680cbf7f740c990c0357372c2bd4058.zip" CACHE STRING "")
use_mirror(VARIABLE LLVM_MONO_REPO_URL URL ${LLVM_MONO_REPO_URL})
set(LLVM_MONO_REPO_MD5 "9bda804e5cc61899085fb0f0dce1089f" CACHE STRING "")
set(ONEFLOW_BUILD_ROOT_DIR "${PROJECT_BINARY_DIR}")
add_subdirectory(${PROJECT_SOURCE_DIR}/oneflow/ir)
if (WITH_MLIR)
  set(ONEFLOW_MLIR_LIBS -Wl,--no-as-needed MLIROneFlowExtension -Wl,--as-needed)
endif()

include(op_schema)

if(APPLE)
  set(of_libs -Wl,-force_load oneflow of_protoobj of_cfgobj of_functional_obj of_op_schema)
  target_link_libraries(oneflow of_protoobj of_cfgobj of_functional_obj glog_imported gflags_imported ${oneflow_third_party_libs})
elseif(UNIX)
  set(of_libs -Wl,--whole-archive oneflow of_protoobj of_cfgobj of_functional_obj of_op_schema -Wl,--no-whole-archive -ldl -lrt)
  target_link_libraries(oneflow of_protoobj of_cfgobj of_functional_obj glog_imported gflags_imported ${oneflow_third_party_libs} -Wl,--no-whole-archive -ldl -lrt)
  if(BUILD_CUDA)
    target_link_libraries(oneflow CUDA::cudart_static)
  endif()
elseif(WIN32)
  set(of_libs oneflow of_protoobj of_cfgobj of_functional_obj of_op_schema)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WHOLEARCHIVE:oneflow")
endif()

# oneflow api common
if (BUILD_PYTHON OR BUILD_CPP_API)
  file(GLOB_RECURSE of_api_common_files
    ${PROJECT_SOURCE_DIR}/oneflow/api/common/*.h
    ${PROJECT_SOURCE_DIR}/oneflow/api/common/*.cpp)
  oneflow_add_library(of_api_common OBJECT ${of_api_common_files})
  target_link_libraries(of_api_common oneflow)
  if (WITH_MLIR)
    target_link_libraries(of_api_common ${ONEFLOW_MLIR_LIBS})
  endif()
endif()

if(BUILD_PYTHON)

  # py ext lib
  oneflow_add_library(of_pyext_obj ${of_pyext_obj_cc})
  target_include_directories(of_pyext_obj PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
  target_link_libraries(of_pyext_obj oneflow pybind11::headers)
  if(BUILD_SHARED_LIBS AND APPLE)
    target_link_libraries(of_pyext_obj ${Python3_LIBRARIES})
  endif()
  add_dependencies(of_pyext_obj oneflow)

  pybind11_add_module(oneflow_internal ${PYBIND11_SRCS} ${of_pybind_obj_cc} ${PYBIND_REGISTRY_CC})
  set_compile_options_to_oneflow_target(oneflow_internal)
  set_property(TARGET oneflow_internal PROPERTY CXX_VISIBILITY_PRESET "default")
  add_dependencies(oneflow_internal of_cfgobj of_functional_obj of_functional_tensor_obj of_op_schema)
  set_target_properties(oneflow_internal PROPERTIES PREFIX "_")
  set_target_properties(oneflow_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_PYTHON_DIR}/oneflow")
  target_link_libraries(oneflow_internal PRIVATE
                        ${of_libs}
                        of_functional_tensor_obj
                        of_api_common
                        ${oneflow_third_party_libs}
                        of_pyext_obj
                        ${oneflow_exe_third_party_libs})
  target_include_directories(oneflow_internal PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

  target_compile_definitions(oneflow_internal PRIVATE ONEFLOW_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  if(WITH_MLIR)
    add_dependencies(check-oneflow oneflow_internal)
  endif(WITH_MLIR)

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

endif(BUILD_PYTHON)

if (BUILD_CPP_API)
  file(GLOB_RECURSE of_cpp_api_files
    ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/*.cpp
    ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/*.h)
  if(BUILD_MONOLITHIC_LIBONEFLOW_CPP_SO)
    oneflow_add_library(oneflow_cpp SHARED ${of_cpp_api_files})
  else()
    oneflow_add_library(oneflow_cpp ${of_cpp_api_files})
  endif()
  set_target_properties(oneflow_cpp PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIBONEFLOW_LIBRARY_DIR}" LIBRARY_OUTPUT_DIRECTORY "${LIBONEFLOW_LIBRARY_DIR}")
  target_link_libraries(oneflow_cpp PRIVATE ${of_libs} of_api_common ${oneflow_third_party_libs})
endif()

file(RELATIVE_PATH PROJECT_BINARY_DIR_RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

function(oneflow_add_test target_name)
  cmake_parse_arguments(arg "" "TEST_NAME;WORKING_DIRECTORY" "SRCS" ${ARGN})
  oneflow_add_executable(${target_name} ${arg_SRCS})
  if (BUILD_CUDA)
    target_link_libraries(${target_name} CUDA::cudart_static)
  endif()
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  add_test(NAME ${arg_TEST_NAME} COMMAND ${target_name} WORKING_DIRECTORY ${arg_WORKING_DIRECTORY})
  set_tests_properties(
    ${arg_TEST_NAME}
  PROPERTIES
    ENVIRONMENT "HTTP_PROXY='';HTTPS_PROXY='';http_proxy='';https_proxy='';"
  )
endfunction()

# build test
if(BUILD_TESTING)
  if (of_all_test_cc)
    oneflow_add_test(oneflow_testexe SRCS ${of_all_test_cc} TEST_NAME oneflow_test)
    target_link_libraries(oneflow_testexe ${of_libs} ${oneflow_third_party_libs} ${oneflow_exe_third_party_libs} ${oneflow_test_libs})
  endif()

  if (BUILD_CPP_API)
    file(GLOB_RECURSE cpp_api_test_files ${PROJECT_SOURCE_DIR}/oneflow/api/cpp/tests/*.cpp)
    oneflow_add_test(oneflow_cpp_api_testexe SRCS ${cpp_api_test_files} TEST_NAME oneflow_cpp_api_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_link_libraries(oneflow_cpp_api_testexe oneflow_cpp ${oneflow_test_libs})
  endif()
endif()


# build include
add_custom_target(of_include_copy ALL)

if(BUILD_PYTHON)

  add_dependencies(of_include_copy oneflow_internal of_pyscript_copy)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/oneflow/core DESTINATION ${ONEFLOW_INCLUDE_DIR}/oneflow
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL
    FILES_MATCHING
    PATTERN *.h
    PATTERN *.hpp
  )
  install(DIRECTORY ${CFG_INCLUDE_DIR}/oneflow DESTINATION ${ONEFLOW_INCLUDE_DIR}
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/oneflow DESTINATION ${ONEFLOW_INCLUDE_DIR}
    COMPONENT oneflow_py_include
    EXCLUDE_FROM_ALL
    FILES_MATCHING
    REGEX "oneflow/core/common/.+(h|hpp)$"
    REGEX "oneflow/core/device/.+(h|hpp)$"
    REGEX "oneflow/core/framework/.+(h|hpp)$"
    REGEX "oneflow/core/kernel/util/.+(h|hpp)$"
    REGEX "oneflow/core/persistence/.+(h|hpp)$"
    REGEX "oneflow/core/ep/include/.+(h|hpp)$"
    PATTERN "oneflow/core/kernel/new_kernel_util.h"
    PATTERN "oneflow/core/kernel/kernel_context.h"
    PATTERN "oneflow/core/kernel/kernel_observer.h"
    PATTERN "oneflow/core/kernel/kernel_util.cuh"
    PATTERN "oneflow/core/job/sbp_signature_builder.h"
    PATTERN "oneflow/core/common/symbol.h"
    PATTERN "oneflow/core/job/parallel_desc.h"
    PATTERN "oneflow/core/autograd/autograd_meta.h"
    PATTERN "oneflow/api" EXCLUDE
    PATTERN "oneflow/xrt" EXCLUDE
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
    PATTERN "oneflow/ir" EXCLUDE
  )
  add_custom_target(install_oneflow_py_include
    COMMAND
        "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=oneflow_py_include
        -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    DEPENDS oneflow_internal
  )
  add_custom_target(oneflow_py ALL)
  add_dependencies(oneflow_py of_include_copy install_oneflow_py_include)

endif(BUILD_PYTHON)


set(LIBONEFLOW_INCLUDE_DIR "${PROJECT_BINARY_DIR}/liboneflow_cpp/include/oneflow/api")
install(DIRECTORY oneflow/api/cpp DESTINATION ${LIBONEFLOW_INCLUDE_DIR}
  COMPONENT oneflow_cpp_include
  EXCLUDE_FROM_ALL
  FILES_MATCHING
  PATTERN "*.h"
)

add_custom_target(install_oneflow_cpp_include
  COMMAND
      "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=oneflow_cpp_include
      -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
  DEPENDS oneflow_internal
)
if (BUILD_CPP_API)
  add_dependencies(of_include_copy oneflow_cpp)
  add_dependencies(of_include_copy install_oneflow_cpp_include)
  copy_files("${PROJECT_SOURCE_DIR}/cmake/oneflow-config.cmake" "${PROJECT_SOURCE_DIR}/cmake" "${LIBONEFLOW_SHARE_DIR}" of_include_copy)

  if(WITH_MLIR)
    file(GLOB mlir_shared_libs "${PROJECT_BINARY_DIR}/oneflow/ir/llvm_monorepo-build/lib/*.14git")
    copy_files("${mlir_shared_libs}" "${PROJECT_BINARY_DIR}/oneflow/ir/llvm_monorepo-build/lib" "${LIBONEFLOW_LIBRARY_DIR}" of_include_copy)
  endif(WITH_MLIR)
endif(BUILD_CPP_API)
