# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

option(THIRD_PARTY "Build third party or oneflow" OFF)

option(USE_CLANG_FORMAT "" OFF)
option(BUILD_RDMA "" OFF)
option(BUILD_CUDA "" ON)
option(RELEASE_VERSION "" ON)
option(PY3 "" OFF)
option(WITH_XLA "Option to build with XLA" OFF)
option(WITH_TENSORRT "Option to build with TensorRT" OFF)

if(NOT RELEASE_VERSION)
  set(CUDNN_STATIC OFF CACHE BOOL "")
endif()

# Project
if(THIRD_PARTY)
  project(third_party C CXX)
else()
  project(oneflow C CXX)
endif()

if (WITH_XLA)
  add_definitions(-DWITH_XLA)
endif()
if (WITH_TENSORRT)
  add_definitions(-DWITH_TENSORRT)
endif()

enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party"
  CACHE PATH "Where the third party headers and libs are put")

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})
include(util)
include(proto2cpp)
include(swig)
include(third_party)

if(NOT RELEASE_VERSION)
  if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    set(CUDA_NVCC_GENCODES "arch=compute_75,code=sm_75" "arch=compute_75,code=compute_75" CACHE STRING "")
  else()
    set(CUDA_NVCC_GENCODES "arch=compute_61,code=sm_61" "arch=compute_61,code=compute_61" CACHE STRING "")
  endif()
endif()

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
  list(APPEND CUDA_NVCC_FLAGS -w -Wno-deprecated-gpu-targets)
  #  half is not fully supported when __CUDA_ARCH__ < 530
  #  list(APPEND __cuda_nvcc_gencodes "arch=compute_30,code=sm_30")
  #  list(APPEND __cuda_nvcc_gencodes "arch=compute_30,code=compute_30")
  #  list(APPEND __cuda_nvcc_gencodes "arch=compute_52,code=sm_52")
  #  list(APPEND __cuda_nvcc_gencodes "arch=compute_52,code=compute_52")
  list(APPEND __cuda_nvcc_gencodes "arch=compute_60,code=sm_60")
  list(APPEND __cuda_nvcc_gencodes "arch=compute_60,code=compute_60")
  list(APPEND __cuda_nvcc_gencodes "arch=compute_61,code=sm_61")
  list(APPEND __cuda_nvcc_gencodes "arch=compute_61,code=compute_61")
  if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
    list(APPEND __cuda_nvcc_gencodes "arch=compute_70,code=sm_70")
    list(APPEND __cuda_nvcc_gencodes "arch=compute_70,code=compute_70")
    list(APPEND __cuda_nvcc_gencodes "arch=compute_75,code=sm_75")
    list(APPEND __cuda_nvcc_gencodes "arch=compute_75,code=compute_75")
  endif()
  set(CUDA_NVCC_GENCODES ${__cuda_nvcc_gencodes} CACHE STRING "")
  foreach(CUDA_NVCC_GENCODE ${CUDA_NVCC_GENCODES})
    list(APPEND CUDA_NVCC_FLAGS -gencode ${CUDA_NVCC_GENCODE})
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
  if (RELEASE_VERSION)
    list(APPEND CUDA_NVCC_FLAGS -O3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  endif()
endif()

if (THIRD_PARTY)
  add_custom_target(prepare_oneflow_third_party ALL DEPENDS ${oneflow_third_party_dependencies})
  set(THIRD_PARTY OFF CACHE BOOL "" FORCE)
else()
  include(oneflow)
  configure_file(${PROJECT_SOURCE_DIR}/setup.py.in ${PROJECT_BINARY_DIR}/setup.py)
endif()
