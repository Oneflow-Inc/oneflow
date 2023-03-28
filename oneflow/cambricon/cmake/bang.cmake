set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${NEUWARE_ROOT_DIR}/cmake/modules
                      $ENV{NEUWARE_HOME}/cmake/modules $ENV{NEUWARE_PATH}/cmake/modules)

find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
endif()

# cncc gflags
set(BANG_CNCC_FLAGS "-Wall -Werror -fPIC -std=c++11 -pthread")
if(CMAKE_BUILD_TYPE MATCHES "debug" OR CMAKE_BUILD_TYPE MATCHES "DEBUG")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O0 -g")
else()
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Xbang-cnas -fno-soft-pipeline")
endif()

set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_372" "--bang-mlu-arch=mtp_592")
