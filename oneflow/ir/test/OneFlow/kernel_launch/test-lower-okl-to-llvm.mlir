// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm  -canonicalize | FileCheck %s

module {
module {
  // func.func @"relu-0"(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {compiled = "true"} {
  //   %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
  //   return %0 : tensor<1xf32>
  // }
  // func.func @wrap0(%arg0: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {compiled = "true", llvm.emit_c_interface} {
  //   %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
  //   %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
  //   return %0, %1 : tensor<1xf32>, tensor<1xf32>
  // }
  func.func @_mlir__mlir_ciface_okl_func(%arg0: !okl.launcher_ctx) attributes {compiled = "true"} {
    %0 = "okl.build_reg_ctx"() {mlir_assembly = "\22func.func\22() ({\0A^bb0(%arg0: tensor<1xf32>):\0A  %0 = \22oneflow.relu\22(%arg0) {device_name = [\22@0:0\22], device_tag = \22cpu\22, hierarchy = [1], op_name = \22relu-0\22, scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>\0A  \22func.return\22(%0) : (tensor<1xf32>) -> ()\0A}) {compiled = \22true\22, function_type = (tensor<1xf32>) -> tensor<1xf32>, sym_name = \22relu-0\22} : () -> ()"} : () -> !okl.reg_ctx
    %1 = "okl.build_run_ctx"(%0, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.build_op_kernel"(%0) {op_type_name = "relu"} : (!okl.reg_ctx) -> !okl.kernel
    "okl.launch"(%0, %1, %2) : (!okl.reg_ctx, !okl.run_ctx, !okl.kernel) -> ()
    "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%1) : (!okl.run_ctx) -> ()
    %3 = "okl.build_reg_ctx"() {mlir_assembly = "\22func.func\22() ({\0A^bb0(%arg0: tensor<1xf32>):\0A  %0 = \22oneflow.relu\22(%arg0) {device_name = [\22@0:0\22], device_tag = \22cpu\22, hierarchy = [1], op_name = \22relu-0\22, scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>\0A  \22func.return\22(%0) : (tensor<1xf32>) -> ()\0A}) {compiled = \22true\22, function_type = (tensor<1xf32>) -> tensor<1xf32>, sym_name = \22relu-0\22} : () -> ()"} : () -> !okl.reg_ctx
    %4 = "okl.build_run_ctx"(%3, %arg0) : (!okl.reg_ctx, !okl.launcher_ctx) -> !okl.run_ctx
    %5 = "okl.build_op_kernel"(%3) {op_type_name = "relu"} : (!okl.reg_ctx) -> !okl.kernel
    "okl.launch"(%3, %4, %5) : (!okl.reg_ctx, !okl.run_ctx, !okl.kernel) -> ()
    "okl.destroy_reg_ctx"(%3) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%4) : (!okl.run_ctx) -> ()
    return
  }
}
}
