include(cmake/mlu.cmake)

file(GLOB_RECURSE ONEFLOW_MLU_SRCS *.cpp *.cc)
add_library(oneflow_mlu SHARED ${ONEFLOW_MLU_SRCS})
add_dependencies(oneflow_mlu oneflow)
target_include_directories(oneflow_mlu PRIVATE ${NEUWARE_INCLUDE_DIRS})
target_link_libraries(oneflow_mlu oneflow ${NEUWARE_LIBRARIES})

set_property(GLOBAL APPEND PROPERTY EXTERNAL_BACKEND_TARGETS oneflow_mlu)
