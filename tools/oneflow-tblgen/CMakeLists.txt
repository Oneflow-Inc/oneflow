set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)
set(LLVM_DIR ${LLVM_INSTALL_DIR}/lib/cmake/llvm)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "[tools/oneflow-tblgen] Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_tablegen(oneflow_tblgen oneflow 
  tablegen.cpp 
  op_schema_emitter.cpp

  ADDITIONAL_HEADER_DIRS
  ${INJA_INCLUDE_DIR}
  ${JSON_INCLUDE_DIR}
  
  DEPENDS
  inja_copy_headers_to_destination
)
