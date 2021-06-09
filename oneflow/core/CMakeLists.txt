DEFINE_PROPERTY(GLOBAL PROPERTY ONEFLOW_CORE_LIBS
    BRIEF_DOCS "oneflow core libs"
    FULL_DOCS "core libs including ops and kernels")

function(setup_core_library)
  add_dependencies(${ARGV0} of_protoobj of_cfgobj)
  GET_PROPERTY(core_libs GLOBAL PROPERTY ONEFLOW_CORE_LIBS)
  list(APPEND core_libs ${ARGV0})
  SET_PROPERTY(GLOBAL PROPERTY ONEFLOW_CORE_LIBS ${core_libs})
endfunction()

function(oneflow_add_core_library)
  add_library(${ARGV})
  setup_core_library(${ARGV})
endfunction()
