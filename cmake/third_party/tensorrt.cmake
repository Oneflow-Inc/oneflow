include (ExternalProject)

if (WITH_TENSORRT)

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
