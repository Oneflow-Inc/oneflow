//RUN: oneflow-opt -lower-oneflow-to-tosa %s | \
//RUN: ireec --iree-mlir-to-vm-bytecode-module --iree-input-type=tosa --iree-hal-target-backends=dylib-llvm-aot - | \
//RUN: iree-run-module --entry_function=main --function_input="2x2xf32=-1. 0. 1. 2." --driver=dylib | \
//RUN: FileCheck %s

//CHECK: result[0]: hal.buffer_view

module  {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
