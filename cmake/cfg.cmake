set(CFG_INCLUDE_DIR tools/cfg/include)
execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
          --get_message_type=template_convert_python_script
  OUTPUT_VARIABLE TEMPLATE_CONVERT_PYTHON_SCRIPT)

execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
          --get_message_type=copy_pyproto_python_script OUTPUT_VARIABLE COPY_PYPROTO_PYTHON_SCRIPT)

execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
          --get_message_type=pybind_registry_cc OUTPUT_VARIABLE PYBIND_REGISTRY_CC)

execute_process(
  COMMAND ${CODEGEN_PYTHON_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
          --get_message_type=template_files OUTPUT_VARIABLE TEMPLATE_FILES)

include_directories(${CFG_INCLUDE_DIR})

function(GENERATE_CFG_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(of_cfg_proto_python_dir "${PROJECT_BINARY_DIR}/of_cfg_proto_python")

  list(APPEND CFG_SOURCE_FILE_CONVERT_PROTO oneflow/core/common/cfg_reflection_test.proto)

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

  if(BUILD_PYTHON)
    list(APPEND PYBIND11_FILE_CONVERT_PROTO)

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

    add_custom_command(
      OUTPUT ${${HDRS}} ${${SRCS}} ${${PYBIND_SRCS}}
      COMMAND ${CMAKE_COMMAND} ARGS -E remove_directory "${of_cfg_proto_python_dir}"
      COMMAND
        ${CODEGEN_PYTHON_EXECUTABLE} ${COPY_PYPROTO_PYTHON_SCRIPT}
        --of_proto_python_dir=${of_proto_python_dir}
        --src_proto_files="${CFG_SOURCE_FILE_CONVERT_PROTO}"
        --dst_proto_python_dir=${of_cfg_proto_python_dir}
      COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT} ${CFG_ARGS}
              --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
              --project_build_dir=${PROJECT_BINARY_DIR} --generate_file_type=cfg.cpp
      COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT} ${PY_CFG_ARGS}
              --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
              --project_build_dir=${PROJECT_BINARY_DIR} --generate_file_type=cfg.pybind.cpp
      DEPENDS ${CODEGEN_PYTHON_EXECUTABLE} ${TEMPLATE_FILES} ${CFG_SOURCE_FILE_CONVERT_PROTO}
              ${PYBIND11_FILE_CONVERT_PROTO})
  else() # build_python

    add_custom_command(
      OUTPUT ${${HDRS}} ${${SRCS}}
      COMMAND ${CMAKE_COMMAND} ARGS -E remove_directory "${of_cfg_proto_python_dir}"
      COMMAND
        ${CODEGEN_PYTHON_EXECUTABLE} ${COPY_PYPROTO_PYTHON_SCRIPT}
        --of_proto_python_dir=${of_proto_python_dir}
        --src_proto_files="${CFG_SOURCE_FILE_CONVERT_PROTO}"
        --dst_proto_python_dir=${of_cfg_proto_python_dir}
      COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT} ${CFG_ARGS}
              --of_cfg_proto_python_dir=${of_cfg_proto_python_dir}
              --project_build_dir=${PROJECT_BINARY_DIR} --generate_file_type=cfg.cpp
      DEPENDS ${CODEGEN_PYTHON_EXECUTABLE} ${TEMPLATE_FILES} ${CFG_SOURCE_FILE_CONVERT_PROTO})

  endif(BUILD_PYTHON)

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)

  if(BUILD_PYTHON)
    set_source_files_properties(${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
    set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)
  endif(BUILD_PYTHON)

endfunction()
