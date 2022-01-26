if(DEFINED ENV{ONEFLOW_INSTALL_PREFIX})
  set(ONEFLOW_INSTALL_PREFIX $ENV{ONEFLOW_INSTALL_PREFIX})
else()
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(ONEFLOW_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)
endif()

set(ONEFLOW_INCLUDE_DIRS ${ONEFLOW_INSTALL_PREFIX}/include)

find_library(ONEFLOW_LIBRARY NAMES oneflow_cpp PATHS ${ONEFLOW_INSTALL_PREFIX}/lib REQUIRED)

if(NOT TARGET OneFlow::liboneflow)
  add_library(OneFlow::liboneflow INTERFACE IMPORTED)

  set_property(TARGET OneFlow::liboneflow PROPERTY INTERFACE_LINK_LIBRARIES ${ONEFLOW_LIBRARY})
  set_property(TARGET OneFlow::liboneflow PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                   ${ONEFLOW_INCLUDE_DIRS})
endif()
