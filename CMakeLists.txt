# Minimum CMake required
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
cmake_minimum_required(VERSION 3.18.0)

set(CMAKE_INSTALL_MESSAGE LAZY CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

option(THIRD_PARTY "Build third party" ON)
option(ONEFLOW "Build oneflow" ON)

if(NOT THIRD_PARTY AND NOT ONEFLOW)
  message(FATAL_ERROR "at least one of flags THIRD_PARTY and ONEFLOW should be ON")
endif()

option(USE_CLANG_FORMAT "" OFF)
option(USE_CLANG_TIDY "" OFF)
option(BUILD_PYTHON "" ON)
option(BUILD_CPP_API "Option to build OneFlow C++ API (beta)" OFF)
option(BUILD_RDMA "" OFF)
option(BUILD_CUDA "" ON)
option(BUILD_TESTING "" OFF)
option(BUILD_GIT_VERSION "" ON)
option(BUILD_PROFILER "" OFF)
option(BUILD_FOR_CI "" OFF)
option(WITH_COCOAPI "Option to build with COCO API" ON)
option(WITH_ZLIB "" ON)
option(WITH_ONEDNN "" ON)
option(WITH_MLIR "" OFF)
option(WITH_MLIR_CUDA_CODEGEN "" OFF)
option(OF_SOFTMAX_USE_FAST_MATH "" ON)
option(OF_LAYER_NORM_USE_FAST_MATH "" ON)
option(TREAT_WARNINGS_AS_ERRORS "" ON)
option(MAYBE_NEED_ERROR_MSG_CHECK "" OFF)

option(LITE_USE_ASCEND_NPU "" OFF)

# Reference:
# https://medium.com/@alasher/colored-c-compiler-output-with-ninja-clang-gcc-10bfe7f2b949
option(OF_FORCE_COLORED_DIAGNOSTICS "Always produce ANSI-colored diagnostics (GNU/Clang only)." ON)

set(ONEFLOW_CURRENT_VERSION 0.8.1.dev CACHE STRING "")

if(BUILD_FOR_CI)
  set(ONEFLOW_CURRENT_VERSION ci)
endif()

set(LLVM_PROVIDER "in-tree" CACHE STRING "in-tree, install")

if(NOT WITH_MLIR)
  set(LLVM_PROVIDER "install"
      CACHE STRING "in-tree will build LLVM's ALL, not what we want when not building MLIR" FORCE)
endif(NOT WITH_MLIR)

set(RPC_BACKEND "GRPC,LOCAL" CACHE STRING "")
set(THIRD_PARTY_MIRROR "" CACHE STRING "")
set(PIP_INDEX_MIRROR "" CACHE STRING "")
set(CPU_THREADING_RUNTIMES "TBB;OMP" CACHE STRING "")

if(APPLE)
  set(RPC_BACKEND "LOCAL")
  set(BUILD_CUDA OFF)
  set(WITH_COCOAPI OFF)
  set(WITH_ONEDNN OFF)
endif()

set(CUDNN_STATIC OFF CACHE BOOL "")

project(oneflow C CXX)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)$")
  message(
    FATAL_ERROR
      "Expected CMAKE_BUILD_TYPE is Debug, Release, RelWithDebInfo or MinSizeRel, got ${CMAKE_BUILD_TYPE}"
  )
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(COMPILER_VERSION_ERROR_MSG "At least gcc 9, clang 5 or Apple clang 12 is supported.")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 9)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 5)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 12)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
else()
  message(WARNING "Unknown compiler \"${CMAKE_CXX_COMPILER_ID}\".")
endif()

set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

get_filename_component(real_src_dir "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(real_bin_dir "${CMAKE_BINARY_DIR}" REALPATH)

if("${real_src_dir}" STREQUAL "${real_bin_dir}")
  message(FATAL_ERROR "In-source build not allowed")
endif()

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})

include(threading)
include(util)
include(proto2cpp)

if(NOT DEFINED USE_CXX11_ABI)
  check_cxx11_abi(CXX11_ABI_AVAILABLE)
  set(USE_CXX11_ABI ${CXX11_ABI_AVAILABLE})
elseif(USE_CXX11_ABI)
  check_cxx11_abi(CXX11_ABI_AVAILABLE)

  if(NOT CXX11_ABI_AVAILABLE)
    message(FATAL_ERROR "cxx11 abi is not available for current compiler")
  endif()
endif()

message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")

if(WITH_MLIR)
  add_definitions(-DWITH_MLIR)

  if(WITH_MLIR_CUDA_CODEGEN)
    add_definitions(-DWITH_MLIR_CUDA_CODEGEN)
  endif()
endif()

if(WITH_COCOAPI)
  add_definitions(-DWITH_COCOAPI)
endif()

if(USE_CXX11_ABI)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(BUILD_PROFILER)
  add_definitions(-DOF_ENABLE_PROFILER)
endif()

if(OF_SOFTMAX_USE_FAST_MATH)
  add_definitions(-DOF_SOFTMAX_USE_FAST_MATH)
endif()

if(OF_LAYER_NORM_USE_FAST_MATH)
  add_definitions(-DOF_LAYER_NORM_USE_FAST_MATH)
endif()

if(OF_FORCE_COLORED_DIAGNOSTICS)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CUDA_COMPILER_ID:Clang>:-fcolor-diagnostics>>)
endif()

if(RPC_BACKEND MATCHES "GRPC")
  add_definitions(-DRPC_BACKEND_GRPC)
  message(STATUS "RPC backend enabled: gRPC")
  set(SUPPORTED_RPC_BACKEND_FOUND 1)
endif()

if(WITH_ONEDNN)
  add_definitions(-DWITH_ONEDNN)
endif()

add_definitions(-DRPC_BACKEND_LOCAL)
message(STATUS "RPC backend enabled: local")
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
    CACHE PATH "Where to install third party headers and libs")

set(ONEFLOW_PYTHON_DIR "${PROJECT_SOURCE_DIR}/python" CACHE PATH "oneflow python src dir")

include(platform)

if((ENABLE_ASAN OR ENABLE_UBSAN) AND ENABLE_TSAN)
  message(FATAL_ERROR "Only ASAN and UBSAN can be enabled at the same time.")
endif()
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address -fno-omit-frame-pointer)
endif()
if(ENABLE_UBSAN)
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
endif()
if(ENABLE_TSAN)
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
endif()

if(BUILD_PYTHON)
  set(ONEFLOW_INCLUDE_DIR "${ONEFLOW_PYTHON_DIR}/oneflow/include")
endif(BUILD_PYTHON)

set(CUTLASS_URL
    https://github.com/Oneflow-Inc/cutlass/archive/e6f548d80bfdf1167d66adbbbcfc2ee3394f4777.zip)
use_mirror(VARIABLE CUTLASS_URL URL ${CUTLASS_URL})
set(CUTLASS_MD5 425f8cf064ff47c81124e55490135f5c)

include(cuda)
add_subdirectory(external)
include(third_party)

message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

add_custom_target(oneflow_deps ALL DEPENDS prepare_oneflow_third_party)

# skip oneflow cmake to avoid errors caused by the absences of python-dev, proto src
if(ONEFLOW)
  include(oneflow)
endif()

add_subdirectory(ci)
