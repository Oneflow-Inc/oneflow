find_package(JNI)

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
else ()
    message (FATAL_ERROR "JNI is not found on your machine")
endif()

aux_source_directory(env ENV_SRC)
aux_source_directory(job_build JOB_BUILD_SRC)
add_library(oneflow SHARED ${ENV_SRC} ${JOB_BUILD_SRC})

target_include_directories(
    oneflow
    PRIVATE
    ${JNI_INCLUDE_DIRS}
)
