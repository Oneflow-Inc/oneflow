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

# build oneflow-api.jar
find_package(Java REQUIRED)
include(UseJava)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8")
file(GLOB_RECURSE of_all_java "${PROJECT_SOURCE_DIR}/oneflow/java/src/main/java/org/*.java")
add_jar(oneflow-api ${of_all_java} VERSION 1.1.1-alpha)
