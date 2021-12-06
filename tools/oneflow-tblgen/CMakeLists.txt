set(LLVM_INSTALL_DIR ${THIRD_PARTY_DIR}/llvm)
set(LLVM_DIR ${LLVM_INSTALL_DIR}/lib/cmake/llvm)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "[tools/oneflow-tblgen] Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(oneflow-tblgen oneflow
  tablegen.cpp
  op_schema_emitter.cpp
)
