// RUN: oneflow-opt %s \
// RUN: -lower-to-okl \
// RUN: | FileCheck %s


// CHECK: module {
// CHECK:   func.func @okl_func(%arg0: !okl.launcher_ctx) attributes {compiled = "true"} {
// CHECK:     %0 = "okl.build_reg_ctx"() ({
// CHECK:     ^bb0(%arg1: tensor<2xf32>):
// CHECK:       %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return %6 : tensor<2xf32>
// CHECK:     }) {function_type = (tensor<2xf32>) -> tensor<2xf32>} : () -> !okl.reg_ctx
// CHECK:     %1 = "okl.build_run_ctx"(%0) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %2 = "okl.build_op_kernel"(%0) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%1, %2) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%1) : (!okl.run_ctx) -> ()
// CHECK:     %3 = "okl.build_reg_ctx"() ({
// CHECK:     ^bb0(%arg1: tensor<2xf32>):
// CHECK:       %6 = "oneflow.tanh"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return %6 : tensor<2xf32>
// CHECK:     }) {function_type = (tensor<2xf32>) -> tensor<2xf32>} : () -> !okl.reg_ctx
// CHECK:     %4 = "okl.build_run_ctx"(%3) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %5 = "okl.build_op_kernel"(%3) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%4, %5) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%3) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%4) : (!okl.run_ctx) -> ()
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
