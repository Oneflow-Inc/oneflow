// RUN: oneflow-opt %s \
// RUN: -extract-kernel-launch-tensor \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @wrap0(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) -> (tensor<2xf32>, tensor<2xf32>) {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_arg"(%[[ARG]]) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "oneflow.relu"(%[[ARG0]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "oneflow.tanh"(%[[ARG1]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG1]]) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG2]]) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:     return %[[ARG3]], %[[ARG4]] : tensor<2xf32>, tensor<2xf32>
// CHECK:   }
// CHECK: }

module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}
