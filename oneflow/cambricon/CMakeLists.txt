include(cmake/mlu.cmake)

set(ONEFLOW_MLU_INCLUDE_DIRS ${NEUWARE_INCLUDE_DIRS})
set(ONEFLOW_MLU_LIBS ${NEUWARE_LIBRARIES})

file(GLOB_RECURSE ONEFLOW_MLU_BANG_SRCS *.mlu)
if(ONEFLOW_MLU_BANG_SRCS)
  include(cmake/bang.cmake)
  bang_add_library(custom_bangc STATIC ${ONEFLOW_MLU_BANG_SRCS})
  list(APPEND ONEFLOW_MLU_LIBS custom_bangc)
endif()

file(GLOB_RECURSE ONEFLOW_MLU_SRCS *.cpp *.cc)
add_library(oneflow_mlu SHARED ${ONEFLOW_MLU_SRCS})
add_dependencies(oneflow_mlu oneflow)
target_include_directories(oneflow_mlu PRIVATE ${ONEFLOW_MLU_INCLUDE_DIRS})
target_link_libraries(oneflow_mlu oneflow ${ONEFLOW_MLU_LIBS})

set_property(GLOBAL APPEND PROPERTY ONEFLOW_BACKEND_LIBS oneflow_mlu)
