// RUN: oneflow-opt %s \
// RUN: -extract-kernel-launch-tensor \
// RUN: -trim-return-to-void \
// RUN: -lower-to-okl \
// RUN: -split-into-funcs \
// RUN: -fetch-from-launcher \
// RUN: -mlir-print-ir-after-all \
// RUN: | FileCheck %s


// CHECK: module {
// CHECK:   func.func @okl_recycle(%arg0: !okl.launcher_ctx) {
// CHECK:     %0 = "okl.fetch_reg_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
// CHECK:     %1 = "okl.fetch_reg_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
// CHECK:     %2 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %3 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_reg_ctx"(%1) : (!okl.reg_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%2) : (!okl.run_ctx) -> ()
// CHECK:     "okl.destroy_run_ctx"(%3) : (!okl.run_ctx) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @okl_compute(%arg0: !okl.launcher_ctx) {
// CHECK:     %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @okl_init_context(%arg0: !okl.launcher_ctx) {
// CHECK:     %0 = "okl.build_reg_ctx"() ({
// CHECK:     ^bb0(%arg1: tensor<2xf32>):
// CHECK:       %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64, tensor_signature = #okl.tensor_signature<[#okl.arg<0>] -> [#okl.ret<0>]>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return %6 : tensor<2xf32>
// CHECK:     }) {function_type = (tensor<2xf32>) -> tensor<2xf32>} : () -> !okl.reg_ctx
// CHECK:     %1 = "okl.build_reg_ctx"() ({
// CHECK:     ^bb0(%arg1: tensor<2xf32>):
// CHECK:       %6 = "oneflow.tanh"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64, tensor_signature = #okl.tensor_signature<[#okl.ret<0>] -> [#okl.ret<1>]>} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       okl.return %6 : tensor<2xf32>
// CHECK:     }) {function_type = (tensor<2xf32>) -> tensor<2xf32>} : () -> !okl.reg_ctx
// CHECK:     %2 = "okl.build_run_ctx"(%0) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %3 = "okl.build_run_ctx"(%1) : (!okl.reg_ctx) -> !okl.run_ctx
// CHECK:     %4 = "okl.build_op_kernel"(%0) : (!okl.reg_ctx) -> !okl.kernel
// CHECK:     %5 = "okl.build_op_kernel"(%1) : (!okl.reg_ctx) -> !okl.kernel
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
