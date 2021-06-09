add_subdirectory(utils)
add_subdirectory(ops)
add_subdirectory(summary)
add_subdirectory(image)
add_subdirectory(data)
add_subdirectory(kernels)

set(ONEFLOW_USER_LIBS oneflow_user_ops oneflow_user_kernels)
set(ONEFLOW_USER_LIBS ${ONEFLOW_USER_LIBS} PARENT_SCOPE)

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

foreach(lib IN
    ITEMS
    oneflow_user_kernel_util
    stateful_local_opkernel
    oneflow_user_data
    oneflow_user_summary
    oneflow_user_image
    oneflow_user_kernels
    oneflow_user_ops
    oneflow_user_utils
    # fixme: add cuda optionally
    oneflow_user_cuda_kernels
    oneflow_softmax_cuda_kernel
)
    add_dependencies(${lib} of_protoobj of_cfgobj)
endforeach()
