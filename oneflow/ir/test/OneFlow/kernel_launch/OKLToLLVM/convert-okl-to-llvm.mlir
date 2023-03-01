// RUN: oneflow-opt %s \
// RUN: -lower-launcher-to-llvm-ptr \
// RUN: -lower-okl-to-llvm-call \
// RUN: -reconcile-unrealized-casts \
// RUN: -convert-func-to-llvm \
// RUN: | FileCheck %s


// CHECK: module attributes {llvm.data_layout = ""}
// CHECK: llvm.func @_mlir_ciface_okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
// CHECK: llvm.call @okl_func(%arg0) : (!llvm.ptr<i8>) -> ()
// CHECK: llvm.return

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