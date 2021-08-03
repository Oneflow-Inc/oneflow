find_package(JNI)

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
else ()
    message (FATAL_ERROR "JNI is not found on your machine")
endif()

# build liboneflow.so
add_library(oneflow SHARED library.h library.cpp)

target_include_directories(
    oneflow
    PRIVATE
    ${JNI_INCLUDE_DIRS}
)

# find java
find_package(Java REQUIRED)
include(UseJava)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8")

# build protobuf-java.jar from source
set(PROTOBUF_JAVA_SRC_DIR ${CMAKE_BINARY_DIR}/protobuf/src/protobuf)

add_custom_target(build_protobuf_java_compile_descriptor
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      --java_out "${PROTOBUF_JAVA_SRC_DIR}/java/core/src/main/java"
      -I "${PROTOBUF_JAVA_SRC_DIR}/src"
      "${PROTOBUF_JAVA_SRC_DIR}/src/google/protobuf/descriptor.proto"
    DEPENDS protobuf
  )
file(GLOB_RECURSE protobuf_java_src "${PROTOBUF_JAVA_SRC_DIR}/java/core/src/main/java/*.java")
add_jar(protobuf_java ${protobuf_java_src})
add_dependencies(protobuf_java build_protobuf_java_compile_descriptor)

# find_jar(protobuf-java "protobuf-java" PATHS "${CMAKE_CURRENT_BINARY_DIR}")

# # build proto for java

# # src, the place to put .proto files. dst, to put .java files.
# set(dst_proto_java_dir "${PROJECT_BINARY_DIR}/of_proto_java")
# set(dst_proto_java_dir_src "${PROJECT_BINARY_DIR}/of_proto_java/src")
# set(dst_proto_java_dir_dst "${PROJECT_BINARY_DIR}/of_proto_java/dst")

# add_custom_target(build_proto_for_java
#     COMMAND ${Python_EXECUTABLE} ${BUILD_PROTO_JAVA}
#       --project_source_dir=${PROJECT_SOURCE_DIR}
#       --src_proto_files="${of_all_proto}" 
#       --dst_proto_java_dir=${dst_proto_java_dir}
#       --protoc_exe ${PROTOBUF_PROTOC_EXECUTABLE}
#     DEPENDS ${Python_EXECUTABLE} protobuf
#   )

# # build oneflow-protos jar
# file(GLOB_RECURSE oneflow_java_all_proto_src "${dst_proto_java_dir_dst}/*.java")
# add_jar(oneflow-protos ${oneflow_java_all_proto_src} 
#         INCLUDE_JARS "${protobuf-java}"
#         VERSION 1.1.1-alpha)
# add_dependencies(oneflow-protos build_proto_for_java)

# # build oneflow-api jar
# file(GLOB_RECURSE oneflow_java_all_src "${PROJECT_SOURCE_DIR}/oneflow/java/src/main/java/org/*.java")
# add_jar(oneflow-api ${oneflow_java_all_src} 
#         INCLUDE_JARS "${protobuf-java}"
#         VERSION 1.1.1-alpha)
# add_dependencies(oneflow-api oneflow-protos)
