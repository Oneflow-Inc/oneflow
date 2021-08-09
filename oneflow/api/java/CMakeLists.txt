find_package(JNI)
if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
else ()
    message (FATAL_ERROR "JNI is not found on your machine")
endif()

# build liboneflow_java.so
add_library(oneflow_java SHARED library.h library.cpp)
target_include_directories(
    oneflow_java
    PRIVATE
    ${JNI_INCLUDE_DIRS}
)

# export ${CMAKE_CURRENT_BINARY_DIR} to LD_LIBRARY_PATH
add_custom_target(oneflow_java_exportation
    COMMAND echo export LD_LIBRARY_PATH="${CMAKE_CURRENT_BINARY_DIR}:$$LD_LIBRARY_PATH" > source.sh
  )
add_dependencies(oneflow_java oneflow_java_exportation)

# find maven executable
execute_process(
  COMMAND which mvn
  RESULT_VARIABLE STATUS
  OUTPUT_VARIABLE OUTPUT
  ERROR_QUIET
)
if(STATUS EQUAL 0)
  string(STRIP ${OUTPUT} STRIPPED)
  message(STATUS "Using Maven from 'which mvn': ${STRIPPED}")
  set(MAVEN_EXECUTABLE ${STRIPPED})
else()
message (FATAL_ERROR "Maven is not found on your machine")
endif()

# find java
find_package(Java REQUIRED)
include(UseJava)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8")

# build google protobuf-java.jar from source
set(PROTOBUF_SRC_DIR ${CMAKE_BINARY_DIR}/protobuf/src/protobuf)
set(PROTOBUF_JAR ${PROTOBUF_SRC_DIR}/java/core/target/protobuf-java-3.9.2.jar)
# build guide: https://github.com/protocolbuffers/protobuf/tree/master/java
# skip test just for speed
add_custom_target(protobuf_java
    COMMAND cp ${PROTOBUF_PROTOC_EXECUTABLE} "${PROTOBUF_SRC_DIR}/src"
    COMMAND ${MAVEN_EXECUTABLE} package -Dmaven.test.skip=true
    WORKING_DIRECTORY "${PROTOBUF_SRC_DIR}/java"
    DEPENDS protobuf
  )

# build oneflow-proto.jar
set(OF_PROTO_JAVA "${PROJECT_BINARY_DIR}/of_proto_java")
set(ONEFLOW_PROTO_JAR "${PROJECT_BINARY_DIR}/of_proto_java/dst/target/oneflow-proto.jar")
add_custom_target(oneflow_proto_java
    COMMAND ${Python_EXECUTABLE} ${BUILD_ONEFLOW_PROTO_JAVA}
      --project_source_dir=${PROJECT_SOURCE_DIR}
      --proto_files="${of_all_proto}" 
      --of_proto_java=${OF_PROTO_JAVA}
      --protoc_exe ${PROTOBUF_PROTOC_EXECUTABLE}
      --protobuf_jar ${PROTOBUF_JAR}
    DEPENDS ${Python_EXECUTABLE} protobuf
  )
add_dependencies(oneflow_proto_java protobuf_java)

# build oneflow-api jar
file(GLOB_RECURSE of_all_java "${PROJECT_SOURCE_DIR}/oneflow/java/src/main/java/org/*.java")
add_jar(oneflow-api ${of_all_java} 
        INCLUDE_JARS "${PROTOBUF_JAR}" "${ONEFLOW_PROTO_JAR}"
        VERSION 1.1.1-alpha)
add_dependencies(oneflow-api oneflow_proto_java)
