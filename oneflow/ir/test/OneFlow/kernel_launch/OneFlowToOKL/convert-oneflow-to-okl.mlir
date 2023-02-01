// RUN: oneflow-opt %s \
// RUN: -extract-kernel-launch-tensor \
// RUN: -trim-return-to-void \
// RUN: -lower-to-okl \
// RUN: | FileCheck %s



// CHECK: module {
// CHECK:   func.func @okl_func(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) {
// CHECK:     "okl.wrapper_kernel"() ({
// CHECK:       %[[ARG0:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_arg"(%[[ARG]]) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG1:[a-zA-Z0-9_]+]] = "oneflow.relu"(%[[ARG0]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG2:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG1]]) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) {index = 0 : i32} : () -> ()
// CHECK:     "okl.wrapper_kernel"() ({
// CHECK:       %[[ARG3:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_ret"(%[[ARG]]) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG4:[a-zA-Z0-9_]+]] = "oneflow.tanh"(%[[ARG0]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG5:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG4]]) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) {index = 1 : i32} : () -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }


 module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}
