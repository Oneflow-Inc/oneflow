execute_process(
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cfg/generate_cfg_head_dir_and_convert_src.py
  OUTPUT_VARIABLE cfg_head_dir_and_convert_srcs
  RESULT_VARIABLE ret_code
  )

string(REPLACE "\n" ";" cfg_head_dir_and_convert_srcs ${cfg_head_dir_and_convert_srcs})
list(GET cfg_head_dir_and_convert_srcs 0  CFG_INCLUDE_DIR)
list(GET cfg_head_dir_and_convert_srcs 1  TEMPLATE_CONVERT_PYTHON_SCRIPT)
list(REMOVE_AT cfg_head_dir_and_convert_srcs 0 1)

set(PYBIND_REGISTRY_CC ${cfg_head_dir_and_convert_srcs})
include_directories(${CFG_INCLUDE_DIR})


function(GENERATE_CFG_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  list(APPEND ALL_CFG_CONVERT_PROTO
      oneflow/core/common/cfg_reflection_test.proto
      oneflow/core/common/data_type.proto
      oneflow/core/common/device_type.proto
  )

  foreach(FIL ${ALL_CFG_CONVERT_PROTO})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
    set(PY_REL_FIL ${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2.py)
    set(PY_REL_MOD ${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2)
    set(CFG_HPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.h)
    set(CFG_CPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.cfg.cpp)
    set(CFG_PYBIND_FIL ${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pybind.cpp)

    add_custom_command(
      OUTPUT "${CFG_HPP_FIL}"
             "${CFG_CPP_FIL}"
             "${CFG_PYBIND_FIL}"
      COMMAND ${Python_EXECUTABLE} ${TEMPLATE_CONVERT_PYTHON_SCRIPT}
      ARGS --dst_hpp_path ${CFG_HPP_FIL} --dst_cpp_path ${CFG_CPP_FIL}
           --dst_pybind_path ${CFG_PYBIND_FIL}
           --proto_py_path ${PY_REL_MOD}  --of_proto_python_dir ${of_proto_python_dir}

      DEPENDS ${Python_EXECUTABLE} ${PY_REL_FIL} ${of_all_rel_pybinds}
      COMMENT "Running Pybind11 Compiler on ${FIL}"
      VERBATIM)

    list(APPEND ${HDRS} "${CFG_HPP_FIL}")
    list(APPEND ${SRCS} "${CFG_CPP_FIL}")
    list(APPEND ${PYBIND_SRCS} "${CFG_PYBIND_FIL}")
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)
endfunction()
