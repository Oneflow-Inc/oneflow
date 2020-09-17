set(CFG_GEN_PATH cfg)
include_directories(${PROJECT_BINARY_DIR}/${CFG_GEN_PATH})
include_directories(${PROJECT_SOURCE_DIR}/tools/cfg/include)

function(RELATIVE_PYBIND11_GENERATE_CPP SRCS HDRS PYBIND_SRC CFG_REL_DIR ROOT_DIR)
  set(${SRCS})
  set(${HDRS})
  set(${PYBIND_SRC})
  list(APPEND ALL_CONVERT_PROTO oneflow/core/common/data_type.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/common/device_type.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/job/sbp_parallel.proto)

  foreach(FIL ${ALL_CONVERT_PROTO})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})
    set(PY_REL_FIL ${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2.py)
    set(PY_REL_MOD ${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2)
    set(CFG_HPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${CFG_REL_DIR}/${REL_DIR}/${FIL_WE}.cfg.h)
    set(CFG_CPP_FIL ${CMAKE_CURRENT_BINARY_DIR}/${CFG_REL_DIR}/${REL_DIR}/${FIL_WE}.cfg.cpp)
    set(CFG_PYBIND_FIL ${CMAKE_CURRENT_BINARY_DIR}/${CFG_REL_DIR}/${REL_DIR}/${FIL_WE}.pybind.cpp)

    add_custom_command(
      OUTPUT "${CFG_HPP_FIL}"
             "${CFG_CPP_FIL}"
             "${CFG_PYBIND_FIL}"
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/cfg/template_convert.py
      ARGS --dst_hpp_path ${CFG_HPP_FIL} --dst_cpp_path ${CFG_CPP_FIL}
           --dst_pybind_path ${CFG_PYBIND_FIL}
          --proto_py_path ${PY_REL_MOD}

      DEPENDS ${Python_EXECUTABLE} ${PY_REL_FIL} ${of_all_rel_pybinds}
      COMMENT "Running Pybind11 Compiler on ${FIL}"
      VERBATIM)

    list(APPEND ${HDRS} "${CFG_HPP_FIL}")
    list(APPEND ${SRCS} "${CFG_CPP_FIL}")
    list(APPEND ${PYBIND_SRC} "${CFG_PYBIND_FIL}")
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRC}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRC} ${${PYBIND_SRC}} PARENT_SCOPE)
endfunction()
