function(setup_user_library)
  add_dependencies(${ARGV0} of_protoobj of_cfgobj)
  list(APPEND ONEFLOW_USER_LIBS ${ARGV0})
  set(ONEFLOW_USER_LIBS ${ONEFLOW_USER_LIBS} PARENT_SCOPE)
endfunction()

function(oneflow_add_user_library)
  add_library(${ARGV})
  setup_user_library(${ARGV})
  set(ONEFLOW_USER_LIBS ${ONEFLOW_USER_LIBS} PARENT_SCOPE)
endfunction()

function(oneflow_add_user_cuda_library)
  if (BUILD_CUDA)
    cuda_add_library(${ARGV})
    setup_user_library(${ARGV})
    set(ONEFLOW_USER_LIBS ${ONEFLOW_USER_LIBS} PARENT_SCOPE)
  endif()
endfunction()

set(ONEFLOW_USER_LIBS ${ONEFLOW_USER_LIBS} PARENT_SCOPE)

add_subdirectory(utils)
add_subdirectory(ops)
add_subdirectory(summary)
add_subdirectory(image)
add_subdirectory(data)
add_subdirectory(kernels)

add_dependencies(of_protoobj make_pyproto_dir ${PROTOBUF_COPY_TARGETS})
if (BUILD_SHARED_LIBS)
  get_filename_component(GLOG_RPATH "${GLOG_STATIC_LIBRARIES}" DIRECTORY)
  get_filename_component(PB_RPATH "${PROTOBUF_LIBRARY_DIR}" DIRECTORY)
  target_link_libraries(oneflow_user_kernels oneflow_user_cuda_kernels)
  foreach(lib IN
    ITEMS
        oneflow_user_ops
        oneflow_user_kernels
        # fixme: add cuda optionally
    )
      target_link_libraries(${lib} of_protoobj of_cfgobj "${GLOG_STATIC_LIBRARIES}")
      set_target_properties(${lib} PROPERTIES INSTALL_RPATH "${GLOG_RPATH} ${PB_RPATH}")
  endforeach()
endif()
