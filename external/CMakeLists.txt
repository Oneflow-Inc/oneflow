set(EXTERNAL_TARGETS)

if(CPU_THREADING_RUNTIME STREQUAL "TBB")
  add_subdirectory(onetbb)
  list(APPEND EXTERNAL_TARGETS tbb)
endif()

add_subdirectory(robin-hood-hashing)
list(APPEND EXTERNAL_TARGETS robin_hood)

add_subdirectory(fmt)
list(APPEND EXTERNAL_TARGETS fmt)

add_subdirectory(kineto)
list(APPEND EXTERNAL_TARGETS kineto)

mark_targets_as_system(${EXTERNAL_TARGETS})
set_property(GLOBAL PROPERTY EXTERNAL_TARGETS ${EXTERNAL_TARGETS})
