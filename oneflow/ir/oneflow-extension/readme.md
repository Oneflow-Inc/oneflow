oneflow.relu
oneflow.relu

->

func @wrap_ops(1): func_type()->()
    2 = oneflow.relu(1)
    3 = oneflow.relu(2)
    return 2, 3


oneflow.kernel_launch @wrap_ops(mlir_asm)

oneflow.kernel_launch
compute
mlir_asm -> module -> lower-oneflow-to-okl (okl type ) -> lower-okl-to-llvm( type.converter) -> llvm module

-> llvm engine (llvm module)

-> liboneflow.so (x)
