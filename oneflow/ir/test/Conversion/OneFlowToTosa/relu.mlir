//RUN: oneflow-opt -lower-oneflow-to-tosa %s | \
//RUN: FileCheck %s

//  CHECK-LABEL:func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
module  {
  func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
