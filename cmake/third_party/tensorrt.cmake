include (ExternalProject)

if (WITH_TENSORRT)
    #link_libraries(/user/local/cuda/lib64/libcudnn.so.7)
    #link_libraries(/usr/local/cuda/lib64/libcudart.so.10.0)
    #    find_library_create_target(nvinfer nvinfer SHARED ) 
    find_library(CUDNN_LIB cudnn /usr/local/cuda/lib64/libcudnn.so.7)
find_library(CUDART_LIB cudart /usr/local/cuda/lib64/libcudart.so.10.0)
find_library(CUBLAS_LIB cublas /sur/local/cuda/lib64/libcublas.so.10.0)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include
          $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/include
          ${THIRD_PARTY_DIR}/tensorrt/include)

find_library(TENSORRT_LIBRARIES NAMES libnvinfer.so libnvinfer.a
             PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib
             $ENV{TENSORRT_ROOT} $ENV{TENSORRT_ROOT}/lib
             ${THIRD_PARTY_DIR}/tensorrt/lib)

if (TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARIES)
else()
  message(FATAL_ERROR "TensorRT was not found. You can set TENSORRT_ROOT to specify the search path.")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIBRARIES}")

include_directories(${TENSORRT_INCLUDE_DIR})

endif(WITH_TENSORRT)
