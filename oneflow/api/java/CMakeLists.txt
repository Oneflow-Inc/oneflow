add_library(oneflow SHARED example.cpp)

target_include_directories(
    oneflow
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    /home/percent1/environment/jdk-11/include/
    /home/percent1/environment/jdk-11/include/linux/
)
