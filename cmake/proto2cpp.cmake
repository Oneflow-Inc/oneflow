function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})
  
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb.h"
             "${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2.py"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIR}
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --python_out  ${of_proto_python_dir} -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIR}
      COMMAND ${CMAKE_COMMAND}
      ARGS -E touch ${of_proto_python_dir}/${REL_DIR}/__init__.py
      DEPENDS ${ABS_FIL}
      COMMENT "Running Protocol Buffer Compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(GRPC_GENERATE_PYTHON SRCS ROOT_DIR)
  if(NOT ARGN)
    message(SEND_ERROR "Error: GRPC_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  find_package(PythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    message(SEND_ERROR "Error: Python Interpreter is not found.")
    return()
  endif()

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m grpc_tools.protoc
    ERROR_VARIABLE _pygrpc_output
  )

  if (NOT (${_pygrpc_output} STREQUAL "Missing input file.\n"))
    message(SEND_ERROR 
      "Error: grpcio_tools not installed\ntry: sudo pip install grpcio_tools"
    ) 
    return() 
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    set(ABS_FIL ${ROOT_DIR}/${FIL})
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} PATH)
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

    list(APPEND ${SRCS} 
      "${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2.py"
      "${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2_grpc.py")

    add_custom_command(
      OUTPUT "${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2.py" 
             "${of_proto_python_dir}/${REL_DIR}/${FIL_WE}_pb2_grpc.py"
      COMMAND ${PYTHON_EXECUTABLE} -m grpc_tools.protoc -I ${ROOT_DIR} ${ABS_FIL} -I ${PROTOBUF_INCLUDE_DIR} --python_out ${of_proto_python_dir} --grpc_python_out ${of_proto_python_dir}
      COMMAND ${CMAKE_COMMAND}
      ARGS -E touch ${of_proto_python_dir}/${REL_DIR}/__init__.py
      DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
      COMMENT "Running Python protocol buffer compiler on ${FIL} for grpc ${of_proto_python_dir}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()
