// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm-call \
// RUN: | FileCheck %s

// CHECK-COUNT-4: llvm.call @okl_llvm_func
module {
  func.func @okl_subgraph(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    "okl.wrapper_kernel"() ({
      %1 = "okl.get_tensor_from_arg"(%0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "oneflow.relu"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = "okl.tensor_to_pool"(%0, %2) {offset = 0 : i64} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 0 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %1 = "okl.pool_to_tensor"(%0) {offset = 0 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = "okl.tensor_to_pool"(%0, %2) {offset = 512 : i64} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 1 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %1 = "okl.pool_to_tensor"(%0) {offset = 512 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "oneflow.arg_sort"(%1) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
      %3 = "okl.get_tensor_as_ret"(%0, %2) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xsi32>) -> tensor<2xsi32>
      okl.return
    }) {index = 2 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %1 = "okl.pool_to_tensor"(%0) {offset = 512 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "okl.get_tensor_from_ret"(%0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xsi32>
      %3 = "oneflow.dim_gather"(%1, %2) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
      %4 = "okl.get_tensor_as_ret"(%0, %3) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 3 : i32} : () -> ()
    return
  }
}

