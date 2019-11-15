# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

option(THIRD_PARTY "Build third party or oneflow" OFF)

option(USE_CLANG_FORMAT "" OFF)
option(BUILD_RDMA "" OFF)
option(BUILD_CUDA "" ON)
option(RELEASE_VERSION "" ON)
option(PY3 "" OFF)

# Project
if(THIRD_PARTY)
  project(third_party C CXX)
else()
  project(oneflow C CXX)
endif()

enable_testing()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(oneflow_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)

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
  list(APPEND CUDA_NVCC_FLAGS -std=c++11 -w -Wno-deprecated-gpu-targets)
  #  half is not fully supported when __CUDA_ARCH__ < 530
  #  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=\"sm_30,compute_30\")
  #  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=\"sm_52,compute_52\")
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=\"sm_60,compute_60\")
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=\"sm_61,compute_61\")
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=\"sm_70,compute_70\")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wno-sign-compare -Wno-unused-function -fPIC")
  if (RELEASE_VERSION)
    list(APPEND CUDA_NVCC_FLAGS -O3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
endif()

set(THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party"
  CACHE PATH "Where the third party headers and libs are put")

# Modules
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir}/third_party)
list(APPEND CMAKE_MODULE_PATH ${oneflow_cmake_dir})
include(util)
include(proto2cpp)
include(swig)
include(third_party)

if (THIRD_PARTY)
  add_custom_target(prepare_oneflow_third_party ALL DEPENDS ${oneflow_third_party_dependencies})
  set(THIRD_PARTY OFF CACHE BOOL "" FORCE)
else()
  include(oneflow)
endif()
