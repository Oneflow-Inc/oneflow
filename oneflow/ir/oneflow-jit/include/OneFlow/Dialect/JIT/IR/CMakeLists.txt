add_mlir_interface(OneFlowJITInterfaces)
add_mlir_dialect(
    OneFlowJITOps oneflow_jit DEPENDS
    MLIROneFlowJITInterfacesIncGen
)
