// RUN: oneflow-opt %s \
// RUN: -trim-return-to-void \
// RUN: | FileCheck %s

// CHECK: func.func @wrap0(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx)

module {
  func.func @wrap0(%arg0: !okl.launcher_ctx) -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = "okl.get_tensor_from_arg"(%arg0) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %3 = "okl.get_tensor_as_ret"(%arg0, %1) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    %4 = "okl.get_tensor_as_ret"(%arg0, %2) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    return %3, %4 : tensor<2xf32>, tensor<2xf32>
  }
}
