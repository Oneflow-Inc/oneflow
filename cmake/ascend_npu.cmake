# The following are set after configuration is done: 
# ASCEND_INCLUDE_DIRS
# ASCEND_LIBRARIES


if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(WARNING "ASCEND_TOOLKIT_HOME env is not found. Setting default value: /usr/local/Ascend/ascend-toolkit/latest")
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Folder contains Ascend toolkit")
else()
    # get ASCEND_TOOLKIT_HOME from environment
    message(STATUS "ASCEND_TOOLKIT_HOME found: $ENV{ASCEND_TOOLKIT_HOME}")
    set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME} CACHE PATH "Folder contains Ascend toolkit")
endif()


find_path(
  ASCEND_INCLUDE_DIRS
  NAMES acl hccl
  PATHS $ENV{ASCEND_TOOLKIT_HOME}/include $ENV{CPLUS_INCLUDE_PATH}
  PATH_SUFFIXES include)

if(ASCEND_INCLUDE_DIRS)
  message(STATUS "ASCEND_INCLUDE_DIRS found: ${ASCEND_INCLUDE_DIRS}")
  execute_process(COMMAND source ${ASCEND_HOME_DIR}/bin/setenv.bash)
else()
  message(
    FATAL_ERROR
      "Huawei Ascend header files are not found. Please set ASCEND_TOOLKIT_HOME to specify the search path."
  )
endif()

find_library(
  ASCEND_LD_LIBRARIES
  NAMES ascendcl
  PATHS ${ASCEND_TOOLKIT_HOME} $ENV{ASCEND_TOOLKIT_HOME}/lib64
        $ENV{LD_LIBRARY_PATH})


if(ASCEND_LD_LIBRARIES)
  message(STATUS "ASCEND_LD_LIBRARIES found: ${ASCEND_LD_LIBRARIES}")
else()
  message(
    FATAL_ERROR
      "ASCEND_LD_LIBRARIES Ascend lib(ascendcl) is not found. Please set ASCEND_TOOLKIT_HOME to specify the search path."
  )
endif()

find_library(
  ASCEND_OP_COMPILER_LD_LIBRARIE
  NAMES acl_op_compiler
  PATHS ${ASCEND_TOOLKIT_HOME} $ENV{ASCEND_TOOLKIT_HOME}/lib64
        $ENV{LD_LIBRARY_PATH})

if(NOT ASCEND_OP_COMPILER_LD_LIBRARIE)
  message(
    FATAL_ERROR
      "ASCEND_OP_COMPILER_LD_LIBRARIE Ascend lib(acl_op_compiler) is not found. Please set ASCEND_TOOLKIT_HOME to specify the search path."
  )
endif()

find_library(
  ASCEND_HCCL_LD_LIBRARIE
  NAMES hccl
  PATHS ${ASCEND_TOOLKIT_HOME} $ENV{ASCEND_TOOLKIT_HOME}/lib64
        $ENV{LD_LIBRARY_PATH})

if(NOT ASCEND_HCCL_LD_LIBRARIE)
  message(
    FATAL_ERROR
      "ASCEND_HCCL_LD_LIBRARIE Ascend lib(hccl) is not found. Please set ASCEND_TOOLKIT_HOME to specify the search path."
  )
endif()

set(ASCEND_INCLUDE_DIRS ${ASCEND_INCLUDE_DIRS})
set(ASCEND_LIBRARIES ${ASCEND_LD_LIBRARIES} ${ASCEND_HCCL_LD_LIBRARIE}
                     ${ASCEND_OP_COMPILER_LD_LIBRARIE})

message(STATUS "Ascend: ASCEND_INCLUDE_DIRS = ${ASCEND_INCLUDE_DIRS}")
message(STATUS "Ascend: ASCEND_LIBRARIES = ${ASCEND_LIBRARIES}")
