include(FetchContent)
FetchContent_Declare(
  kineto
  URL ${KINETO_URL}
  URL_HASH MD5=${KINETO_MD5}
)

FetchContent_GetProperties(kineto)

set(CUDA_SOURCE_DIR ${CUDAToolkit_TARGET_DIR})
set(CUPTI_INCLUDE_DIR ${CUDA_SOURCE_DIR}/extras/CUPTI/include)
set(CUDA_cupti_LIBRARY ${CUDA_SOURCE_DIR}/extras/CUPTI/lib64/libcupti.so)
list(APPEND CUDA_cupti_LIBRARY CUDA::cudart_static) # for undefined symbol: cudaGetDeviceCount

get_property(FMT_SOURCE_DIR GLOBAL PROPERTY FMT_SOURCE_DIR)

if(NOT kineto_POPULATED)
    FetchContent_Populate(kineto)
    add_subdirectory(${kineto_SOURCE_DIR}/libkineto ${kineto_BINARY_DIR})
endif()

set_property(GLOBAL PROPERTY KINETO_INCLUDE_DIR "${kineto_SOURCE_DIR}/libkineto/include")