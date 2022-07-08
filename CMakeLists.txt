# Minimum CMake required
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
set(CPU_THREADING_RUNTIME "SEQ" CACHE STRING "")

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

set(COMPILER_VERSION_ERROR_MSG "At least gcc 7, clang 5 or Apple clang 12 is supported.")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 7)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()

  if(CPU_THREADING_RUNTIME STREQUAL "OMP")
    set(OMP_FLAGS "-fopenmp")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Reference:
  # https://releases.llvm.org/11.0.0/tools/clang/docs/OpenMPSupport.html
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 11)
    if(CPU_THREADING_RUNTIME STREQUAL "OMP")
      message(
        FATAL_ERROR
          "libopenmp is not supported under clang10, please use TBB with '-DCPU_THREADING_RUNTIME=TBB'."
      )
    endif()
  endif()

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

add_definitions(-DONEFLOW_PYTHON_BASE_DIR=\"${PROJECT_SOURCE_DIR}/python\")
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

if(CPU_THREADING_RUNTIME STREQUAL "TBB")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_TBB)
elseif(CPU_THREADING_RUNTIME STREQUAL "OMP")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_OMP)
elseif(CPU_THREADING_RUNTIME STREQUAL "SEQ")
  add_definitions(-DOF_CPU_THREADING_RUNTIME=OF_RUNTIME_SEQ)
else()
  message(FATAL_ERROR "CPU_THREADING_RUNTIME must be one of: TBB, OMP, SEQ")
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
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
    CACHE PATH "Where to install third party headers and libs")

set(ONEFLOW_PYTHON_DIR "${PROJECT_SOURCE_DIR}/python" CACHE PATH "oneflow python src dir")

if(WIN32)
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC
                  -D__VERSION__=\"MSVC\")
  add_definitions(
    -DWIN32
    -DOS_WIN
    -D_MBCS
    -DWIN64
    -DWIN32_LEAN_AND_MEAN
    -DNOGDI
    -DPLATFORM_WINDOWS
    -D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(
    /bigobj
    /nologo
    /EHsc
    /GF
    /FC
    /MP
    /Gm-)
  add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  foreach(
    flag_var
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()

  # set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS} /DEBUG:FASTLINK")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=0")
else()
  set(EXTRA_CXX_FLAGS "-std=c++14 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")

  if(APPLE)
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wno-deprecated-declarations")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_CXX_FLAGS}")
endif()

if(BUILD_PYTHON)
  set(ONEFLOW_INCLUDE_DIR "${ONEFLOW_PYTHON_DIR}/oneflow/include")
endif(BUILD_PYTHON)

if(CPU_THREADING_RUNTIME STREQUAL "TBB")
  set(ONETBB_URL
      https://github.com/oneapi-src/oneTBB/archive/3db67b5ba2a81bd1288325c5847e09e13c46f4d7.zip)
  use_mirror(VARIABLE ONETBB_URL URL ${ONETBB_URL})
  set(ONETBB_MD5 7545d4084baff17af73da2dae5ab8005)
endif()

set(ROBIN_HOOD_HASHING_URL
    https://github.com/martinus/robin-hood-hashing/archive/refs/tags/3.11.5.tar.gz)
use_mirror(VARIABLE ROBIN_HOOD_HASHING_URL URL ${ROBIN_HOOD_HASHING_URL})
set(ROBIN_HOOD_HASHING_MD5 a78bd30a7582f25984f8592652836467)

set(FMT_URL https://github.com/fmtlib/fmt/archive/48b7e3dafb27ece02cd6addc8bd1041c79d59c2c.zip)
use_mirror(VARIABLE FMT_URL URL ${FMT_URL})
set(FMT_MD5 45925a979ed7195e0c88a70be691de09)

set(KINETO_URL
    https://github.com/pytorch/kineto/archive/ff8dba20499a660650632952be76450bd70a52a6.zip)
use_mirror(VARIABLE KINETO_URL URL ${KINETO_URL})
set(KINETO_MD5 f9b550591b3899fb267270c19484933f)

include(cuda)
add_subdirectory(external)
include(third_party)

if(BUILD_CUDA)
  # NOTE: if you want to use source PTX with a version different from produced PTX/binary, you should add flags
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)

    # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
    list(APPEND CMAKE_CUDA_ARCHITECTURES 61-real)

    # V100, TITAN V
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)

    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      # A100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      # GeForce RTX 30xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
    endif()

    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-virtual)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual)
    else()
      list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)
    endif()
  endif()

  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CUDA_SEPARABLE_COMPILATION OFF)

  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
      set(CUDA_NVCC_THREADS_NUMBER "4" CACHE STRING "")
      list(APPEND CUDA_NVCC_FLAGS -t ${CUDA_NVCC_THREADS_NUMBER})
    endif()

    message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
    list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
  endif()
endif()

message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

add_custom_target(oneflow_deps ALL DEPENDS prepare_oneflow_third_party)

# skip oneflow cmake to avoid errors caused by the absences of python-dev, proto src
if(ONEFLOW)
  include(oneflow)
endif()

add_subdirectory(ci)
