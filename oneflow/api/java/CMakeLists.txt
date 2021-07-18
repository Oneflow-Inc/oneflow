find_package(JNI)

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
else ()
    message (FATAL_ERROR "JNI is not found on your machine")
endif()

add_library(oneflow SHARED library.h library.cpp tensor.h tensor.cpp)

target_include_directories(
    oneflow
    PRIVATE
    ${JNI_INCLUDE_DIRS}
)
