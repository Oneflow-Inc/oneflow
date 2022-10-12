// RUN: oneflow-opt %s \
// RUN: -trim-return-as-void  -canonicalize | FileCheck %s

module {
  func.func @wrap0(%arg0: !okl.launcher_ctx) -> (tensor<1xf32>, tensor<1xf32>) {
    %0 = "okl.get_tensor_from_arg"(%arg0) {tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<1xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    %3 = "okl.get_tensor_as_ret"(%arg0, %1) {tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<1xf32>) -> tensor<1xf32>
    %4 = "okl.get_tensor_as_ret"(%arg0, %2) {tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<1xf32>) -> tensor<1xf32>
    return %3, %4 : tensor<1xf32>, tensor<1xf32>
  }
}
