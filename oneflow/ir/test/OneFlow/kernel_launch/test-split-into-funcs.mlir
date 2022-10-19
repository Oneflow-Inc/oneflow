// RUN: oneflow-opt %s \
// RUN: -split-into-funcs | FileCheck %s

module {
  func.func @_mlir__mlir_ciface_okl_func(%arg0: !okl.launcher_ctx) attributes {compiled = "true"} {
    %0 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: tensor<1xf32>):
      %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
      okl.return %6 : tensor<1xf32>
    }) {function_type = (tensor<1xf32>) -> tensor<1xf32>} : () -> !okl.reg_ctx
    %1 = "okl.build_run_ctx"(%0, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.build_op_kernel"(%0) {op_type_name = "relu"} : (!okl.reg_ctx) -> !okl.kernel
    "okl.launch"(%1, %2) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%1) : (!okl.run_ctx) -> ()
    %3 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: tensor<1xf32>):
      %6 = "oneflow.relu"(%arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
      okl.return %6 : tensor<1xf32>
    }) {function_type = (tensor<1xf32>) -> tensor<1xf32>} : () -> !okl.reg_ctx
    %4 = "okl.build_run_ctx"(%3, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %5 = "okl.build_op_kernel"(%3) {op_type_name = "relu"} : (!okl.reg_ctx) -> !okl.kernel
    "okl.launch"(%4, %5) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.destroy_reg_ctx"(%3) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%4) : (!okl.run_ctx) -> ()
    return
  }
}
