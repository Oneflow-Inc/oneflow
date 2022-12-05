find_path(ASCEND_INCLUDE_DIR graph/graph.h
          PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/include $ENV{ASCEND_HOME_PATH}
                $ENV{ASCEND_HOME_PATH}/include)

find_library(ASCEND_ACL_LIBRARY
  NAMES ascendcl
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

find_library(ASCEND_ACL_OP_COMPILER_LIBRARY
  NAMES acl_op_compiler
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

find_library(
  ASCEND_GE_COMPILER_LIBRARY
  NAMES ge_compiler
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

find_library(
  ASCEND_GRAPH_LIBRARY
  NAMES graph
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

if(NOT ASCEND_INCLUDE_DIR OR
   NOT ASCEND_GE_COMPILER_LIBRARY OR
   NOT ASCEND_ACL_LIBRARY OR
   NOT ASCEND_ACL_OP_COMPILER_LIBRARY OR
   NOT ASCEND_GRAPH_LIBRARY)
  message(
    FATAL_ERROR
      "Ascend Sdk was not found. You can set ASCEND_HOME_PATH to specify the search path."
  )
endif()

add_library(ascend_acl_ascendcl SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_acl_ascendcl PROPERTY IMPORTED_LOCATION ${ASCEND_ACL_LIBRARY})

add_library(ascend_acl_op_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_acl_op_compiler PROPERTY IMPORTED_LOCATION ${ASCEND_ACL_OP_COMPILER_LIBRARY})

add_library(ascend_atc_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_atc_ge_compiler PROPERTY IMPORTED_LOCATION ${ASCEND_GE_COMPILER_LIBRARY})

add_library(ascend_graph SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_graph PROPERTY IMPORTED_LOCATION ${ASCEND_GRAPH_LIBRARY})

set(ASCEND_LIBRARIES ascend_acl_ascendcl ascend_acl_op_compiler ascend_atc_ge_compiler ascend_graph)
