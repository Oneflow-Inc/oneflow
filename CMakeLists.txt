# cmake configure
cmake_minimum_required(VERSION 3.5)
project(oneflow C CXX)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_testing()
set(oneflow_source_dir "${PROJECT_SOURCE_DIR}/oneflow")

# dependencies

set(PROTOBUF_PROTOC_EXECUTABLE "/opt/oneflow/protobuf3/bin/protoc"
    CACHE FILEPATH "")
set(Protobuf_INCLUDE_DIRS "/opt/oneflow/protobuf3/include" CACHE PATH "")
set(Protobuf_LIBRARIES "/opt/oneflow/protobuf3/lib/libprotobuf.so" CACHE PATH "")
include(dependencies)

# proto

include(proto2cpp)
file(GLOB_RECURSE oneflow_protos
     RELATIVE ${oneflow_source_dir}
     "*.proto")
RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
                               ${oneflow_source_dir}
                               ${oneflow_protos})
include_directories(${PROJECT_BINARY_DIR})
add_library(protocc ${PROTO_SRCS} ${PROTO_HDRS})

# NOTE: temporary solution
# TODO: auto generate by foreach
add_executable(balanced_splitter_test_exe
               ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter.cpp
               ${PROJECT_SOURCE_DIR}/oneflow/common/balanced_splitter_test.cpp)
target_link_libraries(balanced_splitter_test_exe ${ONEFLOW_LINKER_LIBS})
add_test(NAME balanced_splitter_test COMMAND balanced_splitter_test_exe)
