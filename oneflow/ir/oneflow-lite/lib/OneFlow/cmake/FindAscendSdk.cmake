find_path(ASCEND_INCLUDE_DIR graph/graph.h
          PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/include $ENV{ASCEND_HOME_PATH}
                $ENV{ASCEND_HOME_PATH}/include)

find_library(
  ASCEND_GRAPH_LIBRARY NAMES graph PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64
                                         $ENV{ASCEND_HOME_PATH} $ENV{ASCEND_HOME_PATH}/lib64)

if(NOT ASCEND_INCLUDE_DIR OR NOT ASCEND_GRAPH_LIBRARY)
  message(
    FATAL_ERROR "Ascend Sdk was not found. You can set ASCEND_HOME_PATH to specify the search path."
  )
endif()

add_library(ascend_graph SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${ASCEND_GRAPH_LIBRARY})

set(ASCEND_LIBRARIES ascend_graph)
