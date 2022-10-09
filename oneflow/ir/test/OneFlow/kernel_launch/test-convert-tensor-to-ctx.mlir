// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm  -canonicalize | FileCheck %s

module {
  func.func @wrap0(%arg0: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {compiled = "true", llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "relu-0",
        scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    %1 = "oneflow.relu"(%arg0) {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "relu-0",
        scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    return %0, %1 : tensor<1xf32>, tensor<1xf32>
  }
}
