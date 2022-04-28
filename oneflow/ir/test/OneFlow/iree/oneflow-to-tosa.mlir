
//RUN: oneflow-opt -lower-oneflow-to-tosa %s | \
//RUN: ireec --iree-mlir-to-vm-bytecode-module --iree-input-type=tosa --iree-hal-target-backends=dylib-llvm-aot | \
//RUN: iree-run-module --entry_function=main --function_input="2x2xf32=-1. 0. 1. 2." --driver=dylib | \
//RUN: FileCheck %s

//CHECK: result[0]: hal.buffer_view

module  {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // %y_267, %indice = "oneflow.maxpool_2d"(%1) {ceil_mode = false, data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu", dilation = [1 : si32, 1 : si32], hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "maxpool_2d-8", output_lbns = ["maxpool_2d-8/y_0", "maxpool_2d-8/indice_0"], padding = [1 : si32, 1 : si32], return_indices = false, scope_symbol_id = 4611686018427416575 : i64, stride = [2 : si32, 2 : si32]} : (tensor<16x64x112x112xf32>) -> (tensor<16x64x56x56xf32>, tensor<16x64x56x56xi64>)
    return %1 : tensor<2x2xf32>
  }
}
