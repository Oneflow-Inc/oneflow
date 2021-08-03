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
set(PROTOBUF_SRC_DIR ${CMAKE_BINARY_DIR}/protobuf/src/protobuf)
set(PROTOBUF_JAR ${PROTOBUF_SRC_DIR}/java/core/target/protobuf-java-3.9.2.jar)

add_custom_target(build_protobuf_java
    COMMAND cp ${PROTOBUF_PROTOC_EXECUTABLE} "${PROTOBUF_SRC_DIR}/src"
    COMMAND mvn package -Dmaven.test.skip=true
    WORKING_DIRECTORY "${PROTOBUF_SRC_DIR}/java"
    DEPENDS protobuf
  )

# build oneflow-proto.jar
set(dst_proto_java_dir "${PROJECT_BINARY_DIR}/of_proto_java")
set(dst_proto_java_dir_src "${PROJECT_BINARY_DIR}/of_proto_java/src")
set(dst_proto_java_dir_dst "${PROJECT_BINARY_DIR}/of_proto_java/dst")
set(oneflow_proto_jar "${PROJECT_BINARY_DIR}/of_proto_java/dst/target/oneflow-proto.jar")

add_custom_target(build_proto_for_java
    COMMAND ${Python_EXECUTABLE} ${BUILD_PROTO_JAVA}
      --project_source_dir=${PROJECT_SOURCE_DIR}
      --src_proto_files="${of_all_proto}" 
      --dst_proto_java_dir=${dst_proto_java_dir}
      --protoc_exe ${PROTOBUF_PROTOC_EXECUTABLE}
      --protobuf_jar ${PROTOBUF_JAR}
    DEPENDS ${Python_EXECUTABLE} protobuf
  )
add_dependencies(build_proto_for_java build_protobuf_java)

# build oneflow-api jar
file(GLOB_RECURSE oneflow_java_all_src "${PROJECT_SOURCE_DIR}/oneflow/java/src/main/java/org/*.java")
add_jar(oneflow-api ${oneflow_java_all_src} 
        INCLUDE_JARS "${PROTOBUF_JAR}" "${oneflow_proto_jar}"
        VERSION 1.1.1-alpha)
add_dependencies(oneflow-api build_proto_for_java)
