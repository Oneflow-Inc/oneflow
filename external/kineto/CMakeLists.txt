include(FetchContent)

# reference: https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cupti.cmake

set(CUPTI_ROOT "/usr" CACHE PATH "CUPTI ROOT")

set(CUDA_SOURCE_DIR ${CUDAToolkit_TARGET_DIR})

find_path(
  CUPTI_INCLUDE_DIR cupti.h
  PATHS ${CUPTI_ROOT}
        ${CUPTI_ROOT}/include
        $ENV{CUPTI_ROOT}
        $ENV{CUPTI_ROOT}/include
        ${CUDA_SOURCE_DIR}/extras/CUPTI/include
        ${CUDA_SOURCE_DIR}/targets/x86_64-linux/include
        ${CUDA_SOURCE_DIR}/targets/aarch64-linux/include
  NO_DEFAULT_PATH)

set(TARGET_ARCH "x86_64")
if(NOT ${CMAKE_SYSTEM_PROCESSOR})
  set(TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

list(
  APPEND
  CUPTI_CHECK_LIBRARY_DIRS
  ${CUPTI_ROOT}
  ${CUPTI_ROOT}/lib64
  ${CUPTI_ROOT}/lib
  ${CUPTI_ROOT}/lib/${TARGET_ARCH}-linux-gnu
  $ENV{CUPTI_ROOT}
  $ENV{CUPTI_ROOT}/lib64
  $ENV{CUPTI_ROOT}/lib
  /usr/lib
  ${CUDA_SOURCE_DIR}/targets/x86_64-linux/lib64
  ${CUDA_SOURCE_DIR}/targets/x86_64-linux/lib
  ${CUDA_SOURCE_DIR}/extras/CUPTI/lib64
  ${CUDA_SOURCE_DIR}/extras/CUPTI/lib)

find_library(
  CUDA_cupti_LIBRARY
  NAMES libcupti.so libcupti.dylib # libcupti_static.a
  PATHS ${CUPTI_CHECK_LIBRARY_DIRS} ${CUPTI_INCLUDE_DIR}
  NO_DEFAULT_PATH
  DOC "Path to cuPTI library.")

if(NOT USE_HIP)
  list(APPEND CUDA_cupti_LIBRARY CUDA::cudart_static) # for undefined symbol: cudaGetDeviceCount∂
endif()

if(USE_HIP)
  # kineto auto-enables its roctracer backend whenever it detects ROCm (the
  # ROCM_SOURCE_DIR env var) and then demands ROCTRACER_LIBRARY from a wrong HINTS path.
  # oneflow's profiler (BUILD_PROFILER) is off here, so build kineto CPU-only.
  set(LIBKINETO_NOROCTRACER ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
  kineto
  URL ${KINETO_URL}
  URL_HASH MD5=${KINETO_MD5}
  SOURCE_SUBDIR libkineto)

FetchContent_MakeAvailable(kineto)

target_include_directories(kineto PUBLIC $<BUILD_INTERFACE:${kineto_SOURCE_DIR}/libkineto/include>)

# This kineto release predates gcc-13's stricter transitive-include rules: SampleListener.h
# uses uint64_t without including <cstdint> (it came in for free via libstdc++ before
# gcc-13). Force-include <cstdint> so the headers parse; CPU-only, not HIP-specific.
foreach(_kineto_tgt kineto kineto_base kineto_api)
  if(TARGET ${_kineto_tgt})
    target_compile_options(${_kineto_tgt} PRIVATE -include cstdint)
  endif()
endforeach()
