execute_process(
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=cfg_include_dir
  OUTPUT_VARIABLE CFG_INCLUDE_DIR)

execute_process(
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=template_convert_python_script
  OUTPUT_VARIABLE TEMPLATE_CONVERT_PYTHON_SCRIPT)

execute_process(
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=copy_pyproto_python_script
  OUTPUT_VARIABLE COPY_PYPROTO_PYTHON_SCRIPT)

execute_process(
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=pybind_registry_cc
  OUTPUT_VARIABLE PYBIND_REGISTRY_CC)

execute_process(
  COMMAND python3 ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=template_files
  OUTPUT_VARIABLE TEMPLATE_FILES)

include_directories(${CFG_INCLUDE_DIR})

list(APPEND ONEFLOW_INCLUDE_SRC_DIRS ${CFG_INCLUDE_DIR})

function(GENERATE_CFG_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  list(APPEND CFG_SOURCE_FILE_CONVERT_PROTO
      oneflow/core/common/error.proto
      oneflow/core/vm/instruction.proto
      oneflow/core/eager/eager_symbol.proto
      oneflow/core/job/job_conf.proto
      oneflow/core/job/placement.proto
      oneflow/core/operator/op_conf.proto
      oneflow/core/operator/interface_blob_conf.proto
      oneflow/core/common/shape.proto
      oneflow/core/record/record.proto
      oneflow/core/job/resource.proto
      oneflow/core/register/logical_blob_id.proto
      oneflow/core/register/tensor_slice_view.proto
      oneflow/core/common/range.proto
      oneflow/core/framework/user_op_conf.proto
      oneflow/core/framework/user_op_attr.proto
      oneflow/core/job/sbp_parallel.proto
      oneflow/core/graph/boxing/collective_boxing.proto
      oneflow/core/register/blob_desc.proto
      oneflow/core/register/pod.proto
      oneflow/core/job/scope.proto
      oneflow/core/job/mirrored_parallel.proto
      oneflow/core/operator/op_attribute.proto
      oneflow/core/operator/op_node_signature.proto
      oneflow/core/operator/arg_modifier_signature.proto
      oneflow/core/job/blob_lifetime_signature.proto
      oneflow/core/job/parallel_signature.proto
      oneflow/core/job/parallel_conf_signature.proto
      oneflow/core/eager/eager_instruction.proto
      oneflow/core/job/cluster_instruction.proto
      oneflow/core/job/initializer_conf.proto
      oneflow/core/job/regularizer_conf.proto
      oneflow/core/job/learning_rate_schedule_conf.proto
      oneflow/core/common/cfg_reflection_test.proto
      oneflow/core/common/data_type.proto
      oneflow/core/common/device_type.proto
      oneflow/core/serving/saved_model.proto
  )

  set(of_cfg_proto_python_dir "${PROJECT_BINARY_DIR}/of_cfg_proto_python")

  add_custom_target(copy_pyproto ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${of_cfg_proto_python_dir}"
    COMMAND ${Python_EXECUTABLE} ${COPY_PYPROTO_PYTHON_SCRIPT} --of_proto_python_dir=${of_proto_python_dir}
      --src_proto_files="${CFG_SOURCE_FILE_CONVERT_PROTO}" --dst_proto_python_dir=${of_cfg_proto_python_dir}
    DEPENDS ${Python_EXECUTABLE} of_protoobj
  )

  set(CFG_ARGS "")
  foreach(FIL ${CFG_SOURCE_FILE_CONVERT_PROTO})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
    set(CFG_HPP_FIL ${PROJECT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.h)
    set(CFG_CPP_FIL ${PROJECT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.cpp)
    list(APPEND ${HDRS} ${CFG_HPP_FIL})
    list(APPEND ${SRCS} ${CFG_CPP_FIL})
    list(APPEND CFG_ARGS "--proto_file_path=${FIL}")
  endforeach()

  add_custom_target(
    generate_cfg
    COMMAND ${Python_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT}
      ${CFG_ARGS}
      --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
      --project_build_dir=${PROJECT_BINARY_DIR}
      --generate_file_type=cfg.cpp
    DEPENDS copy_pyproto ${Python_EXECUTABLE} ${ABS_FIL} ${TEMPLATE_FILES}
    VERBATIM
  )

  list(APPEND PYBIND11_FILE_CONVERT_PROTO
      oneflow/core/job/job_conf.proto
      oneflow/core/job/placement.proto
      oneflow/core/framework/user_op_attr.proto
      oneflow/core/framework/user_op_conf.proto
      oneflow/core/job/sbp_parallel.proto
      oneflow/core/job/scope.proto
      oneflow/core/job/mirrored_parallel.proto
      oneflow/core/operator/op_attribute.proto
      oneflow/core/operator/op_node_signature.proto
      oneflow/core/job/parallel_conf_signature.proto
      oneflow/core/job/initializer_conf.proto
      oneflow/core/job/regularizer_conf.proto
      oneflow/core/job/learning_rate_schedule_conf.proto
      oneflow/core/common/data_type.proto
      oneflow/core/common/device_type.proto
      oneflow/core/register/logical_blob_id.proto
      oneflow/core/operator/interface_blob_conf.proto
      oneflow/core/common/shape.proto
      oneflow/core/register/blob_desc.proto
      oneflow/core/register/pod.proto
      oneflow/core/operator/op_conf.proto
  )

  set(PY_CFG_ARGS "")
  foreach(FIL ${PYBIND11_FILE_CONVERT_PROTO})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
    set(CFG_PYBIND_FIL ${PROJECT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.pybind.cpp)
    list(APPEND ${PYBIND_SRCS} ${CFG_PYBIND_FIL})
    list(APPEND PY_CFG_ARGS "--proto_file_path=${FIL}")
  endforeach()

  add_custom_target(
    generate_py_cfg
    COMMAND ${Python_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT}
      ${PY_CFG_ARGS}
      --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
      --project_build_dir=${PROJECT_BINARY_DIR}
      --generate_file_type=cfg.pybind.cpp
    DEPENDS copy_pyproto ${Python_EXECUTABLE} ${ABS_FIL} ${TEMPLATE_FILES}
    VERBATIM
  )

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)
endfunction()
