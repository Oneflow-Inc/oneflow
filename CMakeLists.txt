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
PROTOBUF_GENERATE_CPP(GRPC_SRCS GRPC_HDRS ${oneflow_source_dir} ${oneflow_protos})

#set(GENERATED_PROTOBUF_FILES ${GRPC_HDRS} ${GRPC_SRCS})
set(GENERATED_PROTOBUF_PATH "${CMAKE_BINARY_DIR}/protos")
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

set(PB_CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/protos/lr.pb.cc")
set(PB_H_FILE "${CMAKE_CURRENT_BINARY_DIR}/protos/lr.pb.h")
set(GRPC_PB_CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/protos/lr.grpc.pb.cc")
set(GRPC_PB_H_FILE "${CMAKE_CURRENT_BINARY_DIR}/protos/lr.grpc.pb.h")

set(GENERATED_PROTOBUF_FILES ${PB_H_FILE} ${PB_CPP_FILE}
    ${GRPC_PB_H_FILE} ${GRPC_PB_CPP_FILE})

MESSAGE(STATUS "GRPC_SRCS = ${GRPC_SRCS}")
MESSAGE(STATUS "GRPC_HDRS = ${GRPC_HDRS}")
include_directories(${GENERATED_PROTOBUF_PATH})
link_libraries(grpc++_unsecure grpc gpr ${Protobuf_LIBRARIES} ${GRPC_LIBRARIES})
ADD_EXECUTABLE(scheduler src/scheduler.cpp ${GENERATED_PROTOBUF_FILES})

