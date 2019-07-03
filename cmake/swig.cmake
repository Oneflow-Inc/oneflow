function(RELATIVE_SWIG_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_SWIG_GENERATE_CPP() called without any .i files")
    return()
  endif()
  
  set(${SRCS})
  set(${HDRS})
  find_package(SWIG REQUIRED)
  find_package(PythonLibs REQUIRED)
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    if(NOT "${FIL_WE}" STREQUAL "oneflow_internal")
      continue()
    endif()

    set(GENERATED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}PYTHON_wrap.cpp")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
    list(APPEND ${SRCS} ${GENERATED_FILE})
    add_custom_command(
      OUTPUT ${GENERATED_FILE}
      COMMAND ${SWIG_EXECUTABLE}
      ARGS -python -c++ -py3
           -module ${FIL_WE}
           -I${ROOT_DIR}
           -outdir "${CMAKE_CURRENT_BINARY_DIR}/python"
           -o ${GENERATED_FILE} 
           ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${of_python_obj_cc} ${of_all_obj_cc} ${of_all_swig}
      COMMENT "Running SWIG on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()
