set(CNCL_ROOT_DIR "" CACHE PATH "Folder Cambricon NVIDIA CNNCCL")

set(CNCL_LIBRARY_NAME libcncl.so)

find_path(CNCL_INCLUDE_DIR cncl.h HINTS ${CNCL_ROOT_DIR} $ENV{NEUWARE_HOME}
          $ENV{NEUWARE_PATH} PATH_SUFFIXES include neuware/include)


if(NOT CNCL_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "Cambricon cncl header files are not found. Please set CNCL_ROOT_DIR to specify the search path."
  )
endif()

find_library(
  CNCL_LIBRARY
  ${CNCL_LIBRARY_NAME}
  HINTS ${CNCL_ROOT_DIR} $ENV{NEUWARE_HOME} $ENV{NEUWARE_PATH}
  PATH_SUFFIXES lib64 neuware/lib64)

if(NOT CNCL_LIBRARY)
  message(
    FATAL_ERROR
      "Cambricon cncl library is not found. Please set CNCL_ROOT_DIR to specify the search path."
  )
endif()

message(STATUS "Cambricon: CNCL_INCLUDE_DIR = ${CNCL_INCLUDE_DIR}")
message(STATUS "Cambricon: CNCL_LIBRARY = ${CNCL_LIBRARY}")
