function(GENERATE_FUNCTIONAL_API_AND_PYBIND11_CPP SRCS HDRS PYBIND_SRCS ROOT_DIR)
  set(YAML_FILE ${PROJECT_SOURCE_DIR}/oneflow/core/functional/functional_api.yaml)
  set(GENERATED_API_DIR oneflow/core/functional)
  set(GENERATED_PYBIND_DIR oneflow/api/python/functional)

  list(APPEND SRCS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.cpp)
  list(APPEND HDRS ${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.h)
  list(APPEND PYBIND_SRCS ${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/functional_api.yaml.pybind.cpp)

  add_custom_command(
      OUTPUT "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.cpp"
                 "${PROJECT_BINARY_DIR}/${GENERATED_API_DIR}/functional_api.yaml.h"
                 "${PROJECT_BINARY_DIR}/${GENERATED_PYBIND_DIR}/functional_api.yaml.pybind.cpp"
      COMMAND ${CMAKE_COMMAND} 
      ARGS -E make_directory ${GENERATED_API_DIR}
      COMMAND ${CMAKE_COMMAND} 
      ARGS -E make_directory ${GENERATED_PYBIND_DIR}
      COMMAND ${Python_EXECUTABLE} 
      ARGS ${PROJECT_SOURCE_DIR}/tools/generate_functional_api.py
              --yaml_file_path ${YAML_FILE} --generate_pybind
      DEPENDS ${Python_EXECUTABLE} 
              ${PROJECT_SOURCE_DIR}/tools/generate_functional_api.py ${YAML_FILE}
      VERBATIM)

  set_source_files_properties(${${SRCS}} ${${HDRS}} ${${PYBIND_SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  set(${PYBIND_SRCS} ${${PYBIND_SRCS}} PARENT_SCOPE)

endfunction()
