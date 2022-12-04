// RUN: oneflow-opt %s \
// RUN: -extract-kernel-launch-tensor \
// RUN: -trim-return-to-void \
// RUN: -lower-to-okl \
// RUN: -split-into-funcs \
// RUN: -fetch-from-launcher \
// RUN: -mlir-print-ir-after-all \
// RUN: | FileCheck %s


// CHECK: module {
// CHECK:   func.func @okl_recycle(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okl.fetch_reg_ctx"(%[[ARG]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okl.fetch_reg_ctx"(%[[ARG]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     "okl.destroy_reg_ctx"(%[[ARG0]]) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%[[ARG1]]) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%[[ARG2]]) : (!okl.run_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%[[ARG3]]) : (!okl.run_ctx) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @okl_compute(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okl.fetch_kernel"(%[[ARG]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.fetch_kernel"(%[[ARG]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%[[ARG0]], %[[ARG2]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%[[ARG1]], %[[ARG3]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @okl_init_context(%[[ARG:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okl.build_reg_ctx"() ({
// CHECK:       %[[ARG6:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_arg"(%[[ARG]]) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG7:[a-zA-Z0-9_]+]] = "oneflow.relu"(%[[ARG6]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG8:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG7]]) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) : () -> !okl.reg_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okl.build_reg_ctx"() ({
// CHECK:       %[[ARG6:[a-zA-Z0-9_]+]] = "okl.get_tensor_from_ret"(%[[ARG]]) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
// CHECK:       %[[ARG7:[a-zA-Z0-9_]+]] = "oneflow.tanh"(%[[ARG6]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG8:[a-zA-Z0-9_]+]] = "okl.get_tensor_as_ret"(%[[ARG]], %[[ARG7]]) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return
// CHECK:     }) : () -> !okl.reg_ctx
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okl.build_run_ctx"(%[[ARG0]]) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.build_run_ctx"(%[[ARG1]]) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = "okl.build_op_kernel"(%[[ARG0]]) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     %[[ARG5:[a-zA-Z0-9_]+]] = "okl.build_op_kernel"(%[[ARG1]]) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     return
// CHECK:   }
// CHECK:   func.func private @get_resources_type_0(!okl.launcher_ctx) -> (!okl.reg_ctx, !okl.reg_ctx)
// CHECK:   func.func private @get_resources_type_1(!okl.launcher_ctx) -> (!okl.run_ctx, !okl.run_ctx)
// CHECK:   func.func private @get_resources_type_2(!okl.launcher_ctx) -> (!okl.kernel, !okl.kernel)
// CHECK: }

 module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}
