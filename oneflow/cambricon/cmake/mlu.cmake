# The following are set after configuration is done:
#  NEUWARE_INCLUDE_DIRS
#  NEUWARE_LIBRARIES

set(NEUWARE_ROOT_DIR "" CACHE PATH "Folder contains Cambricon cntoolkit")
set(CNCL_ROOT_DIR "" CACHE PATH "Folder Cambricon CNCL")

find_path(NEUWARE_INCLUDE_DIRS cnrt.h PATHS ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME}
                                            $ENV{NEUWARE_PATH} PATH_SUFFIXES include
                                                                             neuware/include)
if(NOT NEUWARE_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "Cambricon neuware header files are not found. Please set NEUWARE_ROOT_DIR to specify the search path."
  )
endif()

find_library(
  NEUWARE_CNRT_LIBRARY
  NAMES cnrt
  PATHS ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME} $ENV{NEUWARE_PATH}
  PATH_SUFFIXES lib64 neuware/lib64)

if(NOT NEUWARE_CNRT_LIBRARY)
  message(
    FATAL_ERROR
      "Cambricon neuware cnrt library is not found. Please set NEUWARE_ROOT_DIR to specify the search path."
  )
endif()

find_library(
  NEUWARE_CNDRV_LIBRARY
  NAMES cndrv
  PATHS ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME} $ENV{NEUWARE_PATH}
  PATH_SUFFIXES lib64 neuware/lib64)

if(NOT NEUWARE_CNDRV_LIBRARY)
  message(
    FATAL_ERROR
      "Cambricon neuware cndrv library is not found. Please set NEUWARE_ROOT_DIR to specify the search path."
  )
endif()

find_library(
  NEUWARE_CNNL_LIBRARY
  NAMES cnnl
  PATHS ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME} $ENV{NEUWARE_PATH}
  PATH_SUFFIXES lib64 neuware/lib64)

if(NOT NEUWARE_CNNL_LIBRARY)
  message(
    FATAL_ERROR
      "Cambricon neuware cnnl library is not found. Please set NEUWARE_ROOT_DIR to specify the search path."
  )
endif()

find_path(CNCL_INCLUDE_DIR cncl.h PATHS ${CNCL_ROOT_DIR} ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME}
                                        $ENV{NEUWARE_PATH} PATH_SUFFIXES include neuware/include)

if(NOT CNCL_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "Cambricon cncl header files are not found. Please set CNCL_ROOT_DIR to specify the search path."
  )
endif()

find_library(
  CNCL_LIBRARY
  NAMES cncl
  PATHS ${CNCL_ROOT_DIR} ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME} $ENV{NEUWARE_PATH}
  PATH_SUFFIXES lib64 neuware/lib64)

if(NOT CNCL_LIBRARY)
  message(
    FATAL_ERROR
      "Cambricon cncl library is not found. Please set CNCL_ROOT_DIR to specify the search path.")
endif()

if(NOT "${NEUWARE_INCLUDE_DIRS}" STREQUAL "${CNCL_INCLUDE_DIR}")
  set(NEUWARE_INCLUDE_DIRS ${NEUWARE_INCLUDE_DIRS} ${CNCL_INCLUDE_DIR})
endif()

set(NEUWARE_LIBRARIES ${NEUWARE_CNRT_LIBRARY} ${NEUWARE_CNDRV_LIBRARY} ${NEUWARE_CNNL_LIBRARY}
                      ${CNCL_LIBRARY})

message(STATUS "Cambricon: NEUWARE_INCLUDE_DIRS = ${NEUWARE_INCLUDE_DIRS}")
message(STATUS "Cambricon: NEUWARE_LIBRARIES = ${NEUWARE_LIBRARIES}")
