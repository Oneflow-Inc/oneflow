// RUN: oneflow-opt %s \
// RUN: -lower-launcher-to-llvm-ptr \
// RUN: | FileCheck %s

// CHECK: func.func @okl_func(%[[ARG0:[a-zA-Z0-9_]+]]: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK: %[[ARG1:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ARG0:[a-zA-Z0-9_]+]] : !llvm.ptr<i8> to !okl.launcher_ctx

module {
  func.func @okl_func(%arg0: !okl.launcher_ctx) {
    "okl.wrapper_kernel"() ({
      %0 = "okl.get_tensor_from_arg"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.get_tensor_as_ret"(%arg0, %1) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 0 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %0 = "okl.get_tensor_from_ret"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.get_tensor_as_ret"(%arg0, %1) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 1 : i32} : () -> ()
    return
  }
}