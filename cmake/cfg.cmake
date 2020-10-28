execute_process( 
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=cfg_include_dir
  OUTPUT_VARIABLE CFG_INCLUDE_DIR)

execute_process( 
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=template_convert_python_script
  OUTPUT_VARIABLE TEMPLATE_CONVERT_PYTHON_SCRIPT)

execute_process( 
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=copy_pyproto_python_script
  OUTPUT_VARIABLE COPY_PYPROTO_PYTHON_SCRIPT)

execute_process( 
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
    --get_message_type=pybind_registry_cc
  OUTPUT_VARIABLE PYBIND_REGISTRY_CC)

include_directories(${CFG_INCLUDE_DIR})

function(GENERATE_CFG_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  list(APPEND ALL_CFG_CONVERT_PROTO
      oneflow/core/common/cfg_reflection_test.proto
      oneflow/core/common/data_type.proto
      oneflow/core/common/device_type.proto
  )

  set(of_cfg_proto_python_dir "${PROJECT_BINARY_DIR}/of_cfg_proto_python")
  set(cfg_workspace_dir "${PROJECT_BINARY_DIR}/cfg_workspace")

  add_custom_target(copy_and_render_pyproto ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${of_cfg_proto_python_dir}"
    COMMAND ${Python_EXECUTABLE} ${COPY_PYPROTO_PYTHON_SCRIPT} --of_proto_python_dir=${of_proto_python_dir}
      --src_proto_files="${ALL_CFG_CONVERT_PROTO}" --dst_proto_python_dir=${of_cfg_proto_python_dir}
    COMMAND ${Python_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT}
      --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
      --project_build_dir=${PROJECT_BINARY_DIR} --cfg_workspace_dir=${cfg_workspace_dir}
      --proto_file_list="${ALL_CFG_CONVERT_PROTO}"
    DEPENDS ${Python_EXECUTABLE} of_protoobj
  )

  foreach(FIL ${ALL_CFG_CONVERT_PROTO})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
    set(CFG_HPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.h)
    set(CFG_CPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.cpp)
    set(CFG_PYBIND_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.pybind.cpp)
    
    # rule to make target ${CFG_HPP_FIL} for of_cfgobj
    add_custom_command(
      OUTPUT
        "${CFG_CPP_FIL}"
      DEPENDS copy_and_render_pyproto
      VERBATIM)

    list(APPEND ${HDRS} ${CFG_HPP_FIL})
    list(APPEND ${SRCS} ${CFG_CPP_FIL})
    list(APPEND ${PYBIND_SRCS} ${CFG_PYBIND_FIL})
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)
endfunction()
