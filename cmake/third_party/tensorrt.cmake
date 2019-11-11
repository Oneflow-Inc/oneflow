include (ExternalProject)

if (WITH_TENSORRT)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/include
          ${THIRD_PARTY_DIR}/tensorrt/include)
if (NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT include directory was not found. "
                      "You can set TENSORRT_ROOT to specify the search path.")
endif()

find_library(TENSORRT_LIBRARIES NAMES libnvinfer.so libnvinfer.a
             PATHS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib
             ${THIRD_PARTY_DIR}/tensorrt/lib)
if (NOT TENSORRT_LIBRARIES)
  message(FATAL_ERROR "TensorRT library was not found."
                      "You can set TENSORRT_ROOT to specify the search path.")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIBRARIES}")

list(APPEND TENSORRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARIES})

endif(WITH_TENSORRT)
