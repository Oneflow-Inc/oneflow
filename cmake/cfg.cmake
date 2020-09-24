include_directories(${PROJECT_SOURCE_DIR}/tools/cfg/include)
set(PYBIND_REGISTRY_CC "${PROJECT_SOURCE_DIR}/tools/cfg/pybind_module_registry.cpp")

function(RELATIVE_PYBIND11_GENERATE_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(${SRCS})
  set(${HDRS})
  set(${PYBIND_SRCS})
  list(APPEND ALL_CONVERT_PROTO oneflow/core/common/data_type.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/common/device_type.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/job/sbp_parallel.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/job/mirrored_parallel.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/job/scope.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/record/record.proto)
  list(APPEND ALL_CONVERT_PROTO oneflow/core/record/image.proto)

  foreach(FIL ${ALL_CONVERT_PROTO})
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
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/cfg/template_convert.py
      ARGS --dst_hpp_path ${CFG_HPP_FIL} --dst_cpp_path ${CFG_CPP_FIL}
           --dst_pybind_path ${CFG_PYBIND_FIL}
          --proto_py_path ${PY_REL_MOD}

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
