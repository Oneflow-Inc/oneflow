# cmake configure
cmake_minimum_required(VERSION 2.8)
project(oneflow C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# global configure
set(oneflow_source_dir "${PROJECT_SOURCE_DIR}")
include_directories(${oneflow_source_dir})

#protobuf lib
set(PROTOBUF_PROTOC_EXECUTABLE "/opt/oneflow/protobuf3/bin/protoc" CACHE FILEPATH "")
set(Protobuf_INCLUDE_DIRS "/opt/oneflow/protobuf3/include" CACHE PATH "")
set(Protobuf_LIBRARIES "/opt/oneflow/protobuf3/lib/libprotobuf.so" CACHE PATH "")
include_directories(${Protobuf_INCLUDE_DIRS})

#glog lib
include(FindGLOG)

#grpc lib
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
find_library(GRPC_LIBRARY NAMES grpc)
find_library(GRPCPP_LIBRARY NAMES grpc++)
find_library(GPR_LIBRARY NAMES gpr)
set(GRPC_LIBRARIES ${GRPCPP_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
if(GRPC_LIBRARIES)
    message(STATUS "Found GRPC: ${GRPC_LIBRARIES}; plugin - ${GRPC_CPP_PLUGIN}")
endif()

#grpc
include(FindGRPC)
file(GLOB_RECURSE oneflow_protos RELATIVE ${oneflow_source_dir} "*.proto")
MESSAGE(STATUS "oneflow_protos = ${oneflow_protos}")
PROTOBUF_GENERATE_CPP(GRPC_SRCS GRPC_HDRS ${oneflow_source_dir} ${oneflow_protos})
add_library(grpcobj ${GRPC_SRCS} ${GRPC_HDRS})
list(APPEND ONEFLOW_LINKER_LIBS grpcobj)

set(GENERATED_PROTOBUF_PATH "${CMAKE_BINARY_DIR}/oneflow/proto")
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})
include_directories(${GENERATED_PROTOBUF_PATH})

include_directories(${oneflow_source_dir}/oneflow/net/)
MESSAGE(STATUS "oneflow_source_dir = ${oneflow_source_dir}")
file(GLOB_RECURSE oneflow_all_cc "${oneflow_source_dir}/oneflow/net/*.cc")
MESSAGE(STATUS "SRC_FILES = ${oneflow_all_cc}")
add_library(srcobj ${oneflow_all_cc})
add_dependencies(srcobj grpcobj)
list(APPEND ONEFLOW_LINKER_LIBS srcobj)
MESSAGE(STATUS "of_ccobj = ${of_ccobj}")

link_libraries(grpc++_unsecure grpc gpr ${Protobuf_LIBRARIES} ${GRPC_LIBRARIES} ${ONEFLOW_LINKER_LIBS})
ADD_EXECUTABLE(server_lib oneflow/main.cc ${GRPC_SRCS} ${oneflow_all_cc})
