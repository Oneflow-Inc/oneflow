set(LLVM_TARGET_DEFINITIONS OneFlowEnums.td)
mlir_tablegen(OneFlowEnums.h.inc -gen-enum-decls)
mlir_tablegen(OneFlowEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROneFlowEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS OneFlowOps.td)
mlir_tablegen(OneFlowPatterns.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIROneFlowPatternsIncGen)

# NOTE: seperate conversion and opt with --name
set(LLVM_TARGET_DEFINITIONS OneFlowOps.td)
mlir_tablegen(OneFlowPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROneFlowPassIncGen)

add_mlir_dialect(OneFlowOps oneflow DEPENDS MLIROneFlowEnumsIncGen MLIROneFlowPatternsIncGen MLIROneFlowPassIncGen)
