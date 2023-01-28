// RUN: oneflow-opt %s \
// RUN: -extract-okm-tensor \
// RUN: -trim-return-to-void \
// RUN: -lower-to-okl \
// RUN: | FileCheck %s

 module {
  func.func @subgraph0(%arg0: tensor<2xf32>) -> tensor<2xf32> attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
