set(LLVM_TARGET_DEFINITIONS OneFlowEnums.td)
mlir_tablegen(OneFlowEnums.h.inc -gen-enum-decls)
mlir_tablegen(OneFlowEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROneFlowEnumsIncGen)

add_mlir_dialect(OneFlowOps oneflow DEPENDS MLIROneFlowEnumsIncGen)
add_mlir_doc(OneFlowDialect -gen-dialect-doc OneFlowDialect OneFlow/)
add_mlir_doc(OneFlowOps -gen-op-doc OneFlowOps OneFlow/)
