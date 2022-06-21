include(FetchContent)
FetchContent_Declare(
  kineto
  URL ${KINETO_URL}
  URL_HASH MD5=${KINETO_MD5}
  SOURCE_SUBDIR libkineto
)

set(CUDA_SOURCE_DIR ${CUDAToolkit_TARGET_DIR})
set(CUPTI_INCLUDE_DIR ${CUDA_SOURCE_DIR}/extras/CUPTI/include)
set(CUDA_cupti_LIBRARY ${CUDA_SOURCE_DIR}/extras/CUPTI/lib64/libcupti.so)
list(APPEND CUDA_cupti_LIBRARY CUDA::cudart_static) # for undefined symbol: cudaGetDeviceCount

FetchContent_MakeAvailable(kineto)

target_include_directories(kineto PUBLIC
  $<BUILD_INTERFACE:${kineto_SOURCE_DIR}/libkineto/include>)
