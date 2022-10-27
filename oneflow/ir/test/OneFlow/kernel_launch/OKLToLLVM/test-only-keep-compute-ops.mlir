// RUN: oneflow-opt %s \
// RUN: -only-keep-compute-ops \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @okl_compute(%arg0: !okl.launcher_ctx) {
// CHECK:     %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @okl_recycle(%arg0: !okl.launcher_ctx) {
    %0 = "okl.fetch_reg_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
    %1 = "okl.fetch_reg_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
    %2 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
    "okl.destroy_reg_ctx"(%1) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%2) : (!okl.run_ctx) -> ()
    "okl.destroy_run_ctx"(%3) : (!okl.run_ctx) -> ()
    return
  }
  func.func @okl_compute(%arg0: !okl.launcher_ctx) {
    %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    return
  }
  func.func @okl_init_context(%arg0: !okl.launcher_ctx) {
    %0 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: tensor<1xf32>):
      %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
      okl.return %6 : tensor<1xf32>
    }) {function_type = (tensor<1xf32>) -> tensor<1xf32>} : () -> !okl.reg_ctx
    %1 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: tensor<1xf32>):
      %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
      okl.return %6 : tensor<1xf32>
    }) {function_type = (tensor<1xf32>) -> tensor<1xf32>} : () -> !okl.reg_ctx
    %2 = "okl.build_run_ctx"(%0, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.build_run_ctx"(%1, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %4 = "okl.build_op_kernel"(%0) : (!okl.reg_ctx) -> !okl.kernel
    %5 = "okl.build_op_kernel"(%1) : (!okl.reg_ctx) -> !okl.kernel
    return
  }
  func.func private @get_resources_type_0(!okl.launcher_ctx) -> (!okl.reg_ctx, !okl.reg_ctx)
  func.func private @get_resources_type_1(!okl.launcher_ctx) -> (!okl.run_ctx, !okl.run_ctx)
  func.func private @get_resources_type_2(!okl.launcher_ctx) -> (!okl.kernel, !okl.kernel)
}
