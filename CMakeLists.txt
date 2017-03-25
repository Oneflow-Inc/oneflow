# cmake configure
cmake_minimum_required(VERSION 3.5)
project(oneflow C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# global configure
set(oneflow_source_dir "${PROJECT_SOURCE_DIR}")
include_directories(${oneflow_source_dir})

#protobuf lib
set(PROTOBUF_PROTOC_EXECUTABLE "/opt/oneflow/protobuf3/bin/protoc" CACHE FILEPATH "")
set(PB_INCLUDE_DIRS "/opt/oneflow/protobuf3/include" CACHE PATH "")
set(PB_LIBRARIES "/opt/oneflow/protobuf3/lib/libprotobuf.so" CACHE PATH "")
include_directories(${PB_INCLUDE_DIRS})
list(APPEND ONEFLOW_LINKER_LIBS ${PB_LIBRARIES})

#gtest
enable_testing()
find_package(GTest)
include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
list(APPEND ONEFLOW_LINKER_LIBS ${GTEST_LIBRARIES})

#glog lib
include(FindGLOG)
MESSAGE(STATUS "GLOG_INCLUDE_PATH = ${GLOG_INCLUDE_PATH}")
MESSAGE(STATUS "GLOG_LIBRARY = ${GLOG_LIBRARY}")
include_directories(SYSTEM ${GLOG_INCLUDE_PATH})
list(APPEND ONEFLOW_LINKER_LIBS ${GLOG_LIBRARY})
#gflags 
include(FindGFLAG)
MESSAGE(STATUS "GFLAGS_INCLUDE_PATH = ${GFLAGS_INCLUDE_DIR}")
MESSAGE(STATUS "GFLAGS_LIBRARY = ${GFLAGS_LIBRARY}")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})
list(APPEND ONEFLOW_LINKER_LIBS ${GFLAGS_LIBRARY})
#grpc lib
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
find_library(GRPC_LIBRARY NAMES grpc)
find_library(GRPCPP_LIBRARY NAMES grpc++)
find_library(GPR_LIBRARY NAMES gpr)
set(GRPC_LIBRARIES ${GRPCPP_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY})
list(APPEND ONEFLOW_LINKER_LIBS ${GRPC_LIBRARIES})
#protobuf
include(FindPROTOC)
file(GLOB_RECURSE oneflow_protos RELATIVE ${oneflow_source_dir} "*.proto")
MESSAGE(STATUS "oneflow_protos = ${oneflow_protos}")
PROTOBUF_GENERATE_CPP(PB_SRCS PB_HDRS ${oneflow_source_dir} ${oneflow_protos})
add_library(pbobj ${PB_SRCS} ${PB_HDRS})
list(APPEND ONEFLOW_LINKER_LIBS pbobj)
include_directories(${CMAKE_BINARY_DIR}/oneflow)
set(GENERATED_PB_PATH "${CMAKE_BINARY_DIR}/oneflow/proto")
file(MAKE_DIRECTORY ${GENERATED_PB_PATH})

#grpc
#include(FindGRPC)
#GRPC_GENERATE_CPP(GRPC_SRCS GRPC_HDRS ${oneflow_source_dir} ${oneflow_protos})
#add_library(grpcobj ${GRPC_SRCS} ${GRPC_HDRS})
#list(APPEND ONEFLOW_LINKER_LIBS grpcobj)
#set(GENERATED_GRPC_PATH "${CMAKE_BINARY_DIR}/oneflow/proto")
#file(MAKE_DIRECTORY ${GENERATED_GRPC_PATH})
#include_directories(${GENERATED_GRPC_PATH})

include_directories(${oneflow_source_dir}/oneflow/)
include_directories(/usr/local/cuda/include)
MESSAGE(STATUS "oneflow_source_dir = ${oneflow_source_dir}")
file(GLOB_RECURSE oneflow_all_cc "${oneflow_source_dir}/oneflow/*.cpp")
MESSAGE(STATUS "SRC_FILES = ${oneflow_all_cc}")
add_library(srcobj ${oneflow_all_cc})
MESSAGE(STATUS "srcobj = ${srcobj}")
#add_dependencies(srcobj grpcobj)
list(APPEND ONEFLOW_LINKER_LIBS srcobj)


link_libraries(grpc++_unsecure grpc gpr glog gflags ${ONEFLOW_LINKER_LIBS})
#ADD_EXECUTABLE(server_lib oneflow/main.cc oneflow/proto/proto_io.cpp oneflow/context/one.cpp oneflow/context/resource_descriptor.cpp oneflow/task/node_manager.cpp ${GRPC_SRCS} ${oneflow_all_cc})
ADD_EXECUTABLE(server_lib oneflow/main.cc ${GRPC_SRCS} ${oneflow_all_cc})
