set(ONEFLOW_LINKER_LIBS "")

find_package(GLog REQUIRED)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND ONEFLOW_LINKER_LIBS ${GLOG_LIBRARIES})

find_package(GTest REQUIRED)
include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
list(APPEND ONEFLOW_LINKER_LIBS ${GTEST_LIBRARIES})

if(NOT PROTOBUF_PROTOC_EXECUTABLE)
  find_package(Protobuf REQUIRED)
endif()
include_directories(${Protobuf_INCLUDE_DIRS})
list(APPEND ONEFLOW_LINKER_LIBS ${Protobuf_LIBRARIES})
