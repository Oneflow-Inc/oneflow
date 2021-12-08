set(LLVM_TARGET_DEFINITIONS OneFlowJITPasses.td)
mlir_tablegen(OneFlowJITPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROneFlowJITPassIncGen)

add_mlir_interface(OneFlowJITInterfaces)
add_mlir_dialect(
    OneFlowJITOps oneflow_jit DEPENDS
    MLIROneFlowJITInterfacesIncGen
    MLIROneFlowJITPassIncGen
)
