# Minimum CMake required
cmake_minimum_required(VERSION 3.14.0)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Expected CMAKE_BUILD_TYPE is Debug, Release, RelWithDebInfo or MinSizeRel, got ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_definitions(-DONEFLOW_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

option(THIRD_PARTY "Build third party" ON)
option(ONEFLOW "Build oneflow" ON)

if (NOT THIRD_PARTY AND NOT ONEFLOW)
  message(FATAL_ERROR "at least one of flags THIRD_PARTY and ONEFLOW should be ON")
endif()

option(USE_CLANG_FORMAT "" OFF)
option(USE_CLANG_TIDY "" OFF)
option(BUILD_RDMA "" OFF)
option(BUILD_CUDA "" ON)
option(BUILD_TESTING "" OFF)
option(WITH_XLA "Option to build with XLA" OFF)
option(WITH_TENSORRT "Option to build with TensorRT" OFF)
option(WITH_COCOAPI "Option to build with COCO API" ON)
option(BUILD_GIT_VERSION "" ON)
option(BUILD_PROFILER "" OFF)
option(OF_SOFTMAX_USE_FAST_MATH "" ON)
option(TREAT_WARNINGS_AS_ERRORS "" ON)
set(RPC_BACKEND "GRPC,LOCAL" CACHE STRING "")
set(THIRD_PARTY_MIRROR "" CACHE STRING "")
set(PIP_INDEX_MIRROR "" CACHE STRING "")

if (APPLE)
  set(RPC_BACKEND "LOCAL")
  set(BUILD_CUDA OFF)
  set(WITH_COCOAPI OFF)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(CUDNN_STATIC OFF CACHE BOOL "")
else ()
  set(CUDNN_STATIC ON CACHE BOOL "")
endif()

project(oneflow C CXX)

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

if (NOT DEFINED USE_CXX11_ABI)
  check_cxx11_abi(CXX11_ABI_AVAILABLE)
  set(USE_CXX11_ABI ${CXX11_ABI_AVAILABLE})
elseif(USE_CXX11_ABI)
  check_cxx11_abi(CXX11_ABI_AVAILABLE)
  if (NOT CXX11_ABI_AVAILABLE)
    message(FATAL_ERROR "cxx11 abi is not available for current compiler")
  endif()
endif()
message(STATUS "USE_CXX11_ABI: ${USE_CXX11_ABI}")

if (WITH_XLA)
  add_definitions(-DWITH_XLA)
endif()
if (WITH_TENSORRT)
  add_definitions(-DWITH_TENSORRT)
endif()
if (WITH_COCOAPI)
  add_definitions(-DWITH_COCOAPI)
endif()
if (USE_CXX11_ABI)
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
  add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
if (BUILD_PROFILER)
  add_definitions(-DOF_ENABLE_PROFILER)
endif()
if (OF_SOFTMAX_USE_FAST_MATH)
  add_definitions(-DOF_SOFTMAX_USE_FAST_MATH)
endif()
if (RPC_BACKEND MATCHES "GRPC")
  add_definitions(-DRPC_BACKEND_GRPC)
  message(STATUS "RPC backend enabled: gRPC")
  set(SUPPORTED_RPC_BACKEND_FOUND 1)
endif()
add_definitions(-DRPC_BACKEND_LOCAL)
message(STATUS "RPC backend enabled: local")
enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
  CACHE PATH "Where to install third party headers and libs")

set(ONEFLOW_PYTHON_DIR "${PROJECT_SOURCE_DIR}/python" CACHE PATH "oneflow python src dir")

if(WIN32)
  set(CMAKE_BUILD_TYPE Debug)
  add_definitions(-DNOMINMAX -D_WIN32_WINNT=0x0A00 -DLANG_CXX11 -DCOMPILER_MSVC -D__VERSION__=\"MSVC\")
  add_definitions(-DWIN32 -DOS_WIN -D_MBCS -DWIN64 -DWIN32_LEAN_AND_MEAN -DNOGDI -DPLATFORM_WINDOWS -D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(/bigobj /nologo /EHsc /GF /FC /MP /Gm-)
  add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()

  #set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS} /DEBUG:FASTLINK")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=0")
else()
  set(EXTRA_CXX_FLAGS "-std=c++11 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_CXX_FLAGS}")
endif()

set(ONEFLOW_INCLUDE_DIR "${ONEFLOW_PYTHON_DIR}/oneflow/include")
include(third_party)

if (BUILD_CUDA)
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
  message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
  set(CUDA_SEPARABLE_COMPILATION OFF)
  list(APPEND CUDA_NVCC_FLAGS -w -Werror cross-execution-space-call -Wno-deprecated-gpu-targets)
  #  half is not fully supported when __CUDA_ARCH__ < 530
  #  list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_30,code=sm_30")
  #  list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_52,code=sm_52")
  # cubin
  # Tesla P100
  list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_60,code=sm_60")
  # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
  list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_61,code=sm_61")
  # V100, TITAN V
  list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_70,code=sm_70")
  if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_75,code=sm_75")
  endif()
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    # A100
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_80,code=sm_80")
  endif()
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
    # GeForce RTX 30xx
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_86,code=sm_86")
  endif()
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_80,code=compute_80")
  elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_75,code=compute_75")
  else()
    list(APPEND DEFAULT_CUDA_NVCC_GENCODES "arch=compute_70,code=compute_70")
  endif()
  if(NOT DEFINED CUDA_NVCC_GENCODES)
    set(CUDA_NVCC_GENCODES ${DEFAULT_CUDA_NVCC_GENCODES})
  endif()
  foreach(CUDA_NVCC_GENCODE ${CUDA_NVCC_GENCODES})
    list(APPEND CUDA_NVCC_FLAGS -gencode ${CUDA_NVCC_GENCODE})
  endforeach()
  message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
  list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
endif()

message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})

add_custom_target(oneflow_deps ALL DEPENDS prepare_oneflow_third_party)
# skip oneflow cmake to avoid errors caused by the absences of python-dev, proto src
if (ONEFLOW)
  include(oneflow)
endif()
