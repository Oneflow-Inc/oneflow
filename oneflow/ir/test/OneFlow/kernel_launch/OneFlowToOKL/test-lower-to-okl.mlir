// RUN: oneflow-opt %s \
// RUN: -lower-to-okl \
// RUN: | FileCheck %s


// CHECK: module {
// CHECK:   func.func @okl_func(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) attributes {compiled = "true"} {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okl.build_reg_ctx"() ({
// CHECK:       %[[ARG6:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_arg"(%[[ARG]]) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG7:[a-zA-Z0-9_]+]] = "oneflow.relu"(%[[ARG6]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG8:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG7]]) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) : () -> !okl.reg_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okl.build_run_ctx"(%[[ARG0]]) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okl.build_op_kernel"(%[[ARG0]]) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%[[ARG1]], %[[ARG2]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%[[ARG0]]) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%[[ARG1]]) : (!okl.run_ctx) -> ()
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.build_reg_ctx"() ({
// CHECK:       %[[ARG6]] = "okl.get_tensor_from_ret"(%[[ARG]]) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG7]] = "oneflow.tanh"(%[[ARG6]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG8]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG7]]) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) : () -> !okl.reg_ctx
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = "okl.build_run_ctx"(%[[ARG3]]) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG5:[a-zA-Z0-9_]+]] = "okl.build_op_kernel"(%[[ARG3]]) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%[[ARG4]], %[[ARG5]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%[[ARG3]]) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%[[ARG4]]) : (!okl.run_ctx) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @wrap0(%arg0: !okl.launcher_ctx) {
    %0 = "okl.get_tensor_from_arg"(%arg0) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %3 = "okl.get_tensor_as_ret"(%arg0, %1) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    %4 = "okl.get_tensor_as_ret"(%arg0, %2) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    return
  }
}
