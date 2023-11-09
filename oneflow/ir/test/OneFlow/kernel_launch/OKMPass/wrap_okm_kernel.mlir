// RUN: oneflow-opt %s \
// RUN: -wrap-okm-kernel \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @okm_wrap_subgraph0() {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = "okm.arg_to_memref"() {index = 0 : i32} : () -> memref<2xf32>
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okm.plan_memref"() : () -> memref<2xf32>
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okm.wrapper_kernel"(%[[ARG0]], %[[ARG1]]) ({
// CHECK:       %[[ARG11:[a-zA-Z0-9_]+]] = bufferization.to_tensor %[[ARG0]] : memref<2xf32>
// CHECK:       %[[ARG12:[a-zA-Z0-9_]+]] = "oneflow.relu"(%[[ARG11]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG13:[a-zA-Z0-9_]+]] = bufferization.to_memref %[[ARG12]] : memref<2xf32>
// CHECK:       okm.return %[[ARG13:[a-zA-Z0-9_]+]] : memref<2xf32>
// CHECK:     }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okm.plan_memref"() : () -> memref<2xf32>
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = "okm.wrapper_kernel"(%[[ARG1]], %[[ARG3]]) ({
// CHECK:       %[[ARG11:[a-zA-Z0-9_]+]] = bufferization.to_tensor %[[ARG1]] : memref<2xf32>
// CHECK:       %[[ARG12:[a-zA-Z0-9_]+]] = "oneflow.tanh"(%[[ARG11]]) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[ARG13:[a-zA-Z0-9_]+]] = bufferization.to_memref %[[ARG12:[a-zA-Z0-9_]+]] : memref<2xf32>
// CHECK:       okm.return %[[ARG13:[a-zA-Z0-9_]+]] : memref<2xf32>
// CHECK:     }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
// CHECK:     %[[ARG5:[a-zA-Z0-9_]+]] = "okm.ret_to_memref"() {index = 0 : i32} : () -> memref<2xsi32>
// CHECK:     %[[ARG6:[a-zA-Z0-9_]+]] = "okm.wrapper_kernel"(%[[ARG3]], %[[ARG5]]) ({
// CHECK:       %[[ARG11:[a-zA-Z0-9_]+]] = bufferization.to_tensor %[[ARG3]] : memref<2xf32>
// CHECK:       %[[ARG12:[a-zA-Z0-9_]+]] = "oneflow.arg_sort"(%[[ARG11]]) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
// CHECK:       %[[ARG13:[a-zA-Z0-9_]+]] = bufferization.to_memref %[[ARG12]] : memref<2xsi32>
// CHECK:       okm.return %[[ARG13:[a-zA-Z0-9_]+]] : memref<2xsi32>
// CHECK:     }) : (memref<2xf32>, memref<2xsi32>) -> memref<2xsi32>
// CHECK:     %[[ARG7:[a-zA-Z0-9_]+]] = "okm.ret_to_memref"() {index = 1 : i32} : () -> memref<2xf32>
// CHECK:     %[[ARG8:[a-zA-Z0-9_]+]] = "okm.wrapper_kernel"(%[[ARG3]], %[[ARG5]], %7) ({
// CHECK:       %[[ARG11:[a-zA-Z0-9_]+]] = bufferization.to_tensor %[[ARG3]] : memref<2xf32>
// CHECK:       %[[ARG12:[a-zA-Z0-9_]+]] = bufferization.to_tensor %[[ARG5]] : memref<2xsi32>
// CHECK:       %[[ARG13:[a-zA-Z0-9_]+]] = "oneflow.dim_gather"(%[[ARG11]], %[[ARG12]]) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
// CHECK:       %[[ARG14:[a-zA-Z0-9_]+]] = bufferization.to_memref %[[ARG13]] : memref<2xf32>
// CHECK:       okm.return %[[ARG14]] : memref<2xf32>
// CHECK:     }) : (memref<2xf32>, memref<2xsi32>, memref<2xf32>) -> memref<2xf32>
// CHECK:     %[[ARG9:[a-zA-Z0-9_]+]] = "okm.memref_to_ret"(%[[ARG6]]) {index = 0 : i32} : (memref<2xsi32>) -> memref<2xsi32>
// CHECK:     %[[ARG10:[a-zA-Z0-9_]+]] = "okm.memref_to_ret"(%[[ARG8]]) {index = 1 : i32} : (memref<2xf32>) -> memref<2xf32>
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @okm_subgraph0(%arg0: tensor<2xf32>) -> (tensor<2xsi32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "okm.arg_to_tensor"() {index = 0 : i32} : () -> tensor<2xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %3 = "oneflow.arg_sort"(%2) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
    %4 = "oneflow.dim_gather"(%2, %3) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
    %5 = "okm.tensor_to_ret"(%3) {index = 0 : i32} : (tensor<2xsi32>) -> tensor<2xsi32>
    %6 = "okm.tensor_to_ret"(%4) {index = 1 : i32} : (tensor<2xf32>) -> tensor<2xf32>
    return %5, %6 : tensor<2xsi32>, tensor<2xf32>
  }
}

