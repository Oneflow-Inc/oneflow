// RUN: oneflow-opt %s \
// RUN: -extract-okm-tensor \
// RUN: | FileCheck %s

// CHECK: "okm.arg_to_tensor"() {index = 0 : i32} : () -> tensor<2xf32>
// CHECK: "okm.tensor_to_ret"(%[[ARG0:[a-zA-Z0-9_]+]]) {index = 0 : i32} : (tensor<2xsi32>) -> tensor<2xsi32>
// CHECK: "okm.tensor_to_ret"(%[[ARG1:[a-zA-Z0-9_]+]]) {index = 1 : i32} : (tensor<2xf32>) -> tensor<2xf32>

module {
  func.func @_mlir_oneflow_subgraph0(%arg0: tensor<2xf32>) -> (tensor<2xsi32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.arg_sort"(%1) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
    %3 = "oneflow.dim_gather"(%1, %2) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
    return %2, %3 : tensor<2xsi32>, tensor<2xf32>
  }
}
