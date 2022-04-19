function(GENERATE_FUNCTIONAL_API_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(YAML_FILE ${PROJECT_SOURCE_DIR}/oneflow/core/functional/functional_api.yaml)
  set(GENERATED_API_DIR oneflow/core/functional)

  list(APPEND SRCS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.cpp)
  list(APPEND HDRS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.h)

  if(BUILD_PYTHON)
    set(GENERATED_PYBIND_DIR oneflow/api/python/functional)
    list(APPEND PYBIND_SRCS
         ${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/functional_api.yaml.pybind.cpp)
  endif(BUILD_PYTHON)

  if(BUILD_PYTHON)

    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.cpp"
             "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.h"
             "${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/functional_api.yaml.pybind.cpp"
      COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_API_DIR}
      COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_PYBIND_DIR}
      COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ARGS
              ${PROJECT_SOURCE_DIR}/tools/functional/generate_functional_api.py --project_source_dir
              ${PROJECT_SOURCE_DIR} --export_pybind
      DEPENDS ${CODEGEN_PYTHON_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/tools/functional/generate_functional_api.py
              ${PROJECT_SOURCE_DIR}/tools/functional/generator.py ${YAML_FILE}
      VERBATIM)

  else() # build_python

    add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.cpp"
             "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.h"
      COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_API_DIR}
      COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ARGS
              ${PROJECT_SOURCE_DIR}/tools/functional/generate_functional_api.py --project_source_dir
              ${PROJECT_SOURCE_DIR}
      DEPENDS ${CODEGEN_PYTHON_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/tools/functional/generate_functional_api.py
              ${PROJECT_SOURCE_DIR}/tools/functional/generator.py ${YAML_FILE}
      VERBATIM)

  endif(BUILD_PYTHON)

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)

  if(BUILD_PYTHON)
    set_source_files_properties(${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
    set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)
  endif(BUILD_PYTHON)

endfunction()

function(GENERATE_FUNCTIONAL_TENSOR_API_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(YAML_FILE ${PROJECT_SOURCE_DIR}/oneflow/api/python/functional/tensor_api.yaml)
  set(GENERATED_API_DIR oneflow/api/python/functional)
  set(GENERATED_PYBIND_DIR oneflow/api/python/functional)

  list(APPEND SRCS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/tensor_api.yaml.cpp)
  list(APPEND HDRS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/tensor_api.yaml.h)
  list(APPEND PYBIND_SRCS ${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/tensor_api.yaml.pybind.cpp)

  add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/tensor_api.yaml.cpp"
           "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/tensor_api.yaml.h"
           "${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/tensor_api.yaml.pybind.cpp"
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_API_DIR}
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_PYBIND_DIR}
    COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ARGS
            ${PROJECT_SOURCE_DIR}/tools/functional/generate_tensor_api.py --project_source_dir
            ${PROJECT_SOURCE_DIR}
    DEPENDS ${CODEGEN_PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/functional/generate_tensor_api.py
            ${PROJECT_SOURCE_DIR}/tools/functional/generator.py ${YAML_FILE}
    VERBATIM)

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)

endfunction()

function(GENERATE_FUNCTIONAL_DISPATCH_STATEFUL_OPS_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(YAML_FILE ${PROJECT_SOURCE_DIR}/oneflow/api/python/functional/dispatch_stateful_ops.yaml)
  set(GENERATED_API_DIR oneflow/api/python/functional)
  set(GENERATED_PYBIND_DIR oneflow/api/python/functional)

  list(APPEND SRCS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/dispatch_stateful_ops.yaml.cpp)
  list(APPEND HDRS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/dispatch_stateful_ops.yaml.h)
  list(APPEND PYBIND_SRCS
       ${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/dispatch_stateful_ops.yaml.pybind.cpp)

  add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/dispatch_stateful_ops.yaml.cpp"
           "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/dispatch_stateful_ops.yaml.h"
           "${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/dispatch_stateful_ops.yaml.pybind.cpp"
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_API_DIR}
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${GENERATED_PYBIND_DIR}
    COMMAND ${CODEGEN_PYTHON_EXECUTABLE} ARGS
            ${PROJECT_SOURCE_DIR}/tools/functional/generate_dispatch_stateful_ops.py
            --project_source_dir ${PROJECT_SOURCE_DIR}
    DEPENDS ${CODEGEN_PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/functional/generate_dispatch_stateful_ops.py
            ${PROJECT_SOURCE_DIR}/tools/functional/generator.py ${YAML_FILE}
    VERBATIM)

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)

endfunction()
