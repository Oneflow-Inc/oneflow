function(RELATIVE_SWIG_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_SWIG_GENERATE_CPP() called without any .i files")
    return()
  endif()
  
  set(${SRCS})
  set(${HDRS})
  find_package(SWIG REQUIRED)
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    if(NOT "${FIL_WE}" STREQUAL "oneflow_internal")
      continue()
    endif()

    set(GENERATED_CPP "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}PYTHON_wrap.cpp")
    set(GENERATED_H "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}PYTHON_wrap.h")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python_scripts")
    list(APPEND ${SRCS} ${GENERATED_CPP})
    list(APPEND ${HDRS} ${GENERATED_H})
    if (PY3)
      set(PY3_ARG "-py3")
    endif()
    add_custom_command(
      OUTPUT ${GENERATED_CPP} 
             ${GENERATED_H}
      COMMAND ${SWIG_EXECUTABLE}
      ARGS -python -c++ ${PY3_ARG} -threads
           -module ${FIL_WE}
           -I${ROOT_DIR}
           -outdir "${CMAKE_CURRENT_BINARY_DIR}/python_scripts/oneflow"
           -o ${GENERATED_CPP} 
           ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${of_python_obj_cc} ${of_all_obj_cc} ${of_all_swig} ${oneflow_all_hdr_expanded}
      COMMENT "Running SWIG on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()
