find_path(NEUWARE_MODULE_DIR FindBANG.cmake PATHS ${NEUWARE_ROOT_DIR} $ENV{NEUWARE_HOME}
                                                  $ENV{NEUWARE_PATH}
          PATH_SUFFIXES cmake/modules neuware/cmake/modules)
if(NOT NEUWARE_MODULE_DIR)
  message(
    FATAL_ERROR
      "Cambricon neuware cmake modules are not found. Please set NEUWARE_ROOT_DIR to specify the search path."
  )
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${NEUWARE_MODULE_DIR})

find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
endif()

# cncc gflags
set(BANG_CNCC_FLAGS
    "-Wall -Werror -fPIC -std=c++11 -pthread --neuware-path=${NEUWARE_MODULE_DIR}/../..")
if(CMAKE_BUILD_TYPE MATCHES "debug" OR CMAKE_BUILD_TYPE MATCHES "DEBUG")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O0 -g")
else()
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Xbang-cnas -fno-soft-pipeline")
endif()

set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_372" "--bang-mlu-arch=mtp_592")
