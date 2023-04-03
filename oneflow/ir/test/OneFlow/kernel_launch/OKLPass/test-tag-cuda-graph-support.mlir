// RUN: oneflow-opt %s \
// RUN: -tag-cuda-graph-support \
// RUN: | FileCheck %s

// CHECK:  func.func @okl_subgraph(%[[ARG0:[a-zA-Z0-9_]+]]: !okl.launcher_ctx) attributes {cuda_graph_support = false, pool_size = 1024 : i64}

module {
  func.func @okl_subgraph(%arg0: !okl.launcher_ctx) attributes {pool_size = 1024 : i64} {
    "okl.wrapper_kernel"() ({
      %0 = "okl.get_tensor_from_arg"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.tensor_to_pool"(%arg0, %1) {offset = 0 : i64} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 0 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %0 = "okl.pool_to_tensor"(%arg0) {offset = 0 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.tensor_to_pool"(%arg0, %1) {offset = 512 : i64} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 1 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %0 = "okl.pool_to_tensor"(%arg0) {offset = 512 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.arg_sort"(%0) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
      %2 = "okl.get_tensor_as_ret"(%arg0, %1) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xsi32>) -> tensor<2xsi32>
      okl.return
    }) {index = 2 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %0 = "okl.pool_to_tensor"(%arg0) {offset = 512 : i64} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "okl.get_tensor_from_ret"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xsi32>
      %2 = "oneflow.dim_gather"(%0, %1) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
      %3 = "okl.get_tensor_as_ret"(%arg0, %2) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 3 : i32} : () -> ()
    return
  }
}

