// RUN: oneflow-opt %s \
// RUN: -opt-okm-memref \
// RUN: | FileCheck %s

// CHECK: func.func @okm_alloc_subgraph
// CHECK: "okm.alloc_memref"()
// CHECK: memref.view

module {
  func.func @okm_wrap_subgraph0() {
    %0 = "okm.arg_to_memref"() {index = 0 : i32} : () -> memref<2xf32>
    %1 = "okm.plan_memref"() : () -> memref<2xf32>
    %2 = "okm.wrapper_kernel"(%0, %1) ({
      %11 = bufferization.to_tensor %0 : memref<2xf32>
      %12 = "oneflow.relu"(%11) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %13 = bufferization.to_memref %12 : memref<2xf32>
      okm.return %13 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %3 = "okm.plan_memref"() : () -> memref<2xf32>
    %4 = "okm.wrapper_kernel"(%1, %3) ({
      %11 = bufferization.to_tensor %1 : memref<2xf32>
      %12 = "oneflow.tanh"(%11) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %13 = bufferization.to_memref %12 : memref<2xf32>
      okm.return %13 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %5 = "okm.ret_to_memref"() {index = 0 : i32} : () -> memref<2xsi32>
    %6 = "okm.wrapper_kernel"(%3, %5) ({
      %11 = bufferization.to_tensor %3 : memref<2xf32>
      %12 = "oneflow.arg_sort"(%11) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
      %13 = bufferization.to_memref %12 : memref<2xsi32>
      okm.return %13 : memref<2xsi32>
    }) : (memref<2xf32>, memref<2xsi32>) -> memref<2xsi32>
    %7 = "okm.ret_to_memref"() {index = 1 : i32} : () -> memref<2xf32>
    %8 = "okm.wrapper_kernel"(%3, %5, %7) ({
      %11 = bufferization.to_tensor %3 : memref<2xf32>
      %12 = bufferization.to_tensor %5 : memref<2xsi32>
      %13 = "oneflow.dim_gather"(%11, %12) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
      %14 = bufferization.to_memref %13 : memref<2xf32>
      okm.return %14 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xsi32>, memref<2xf32>) -> memref<2xf32>
    %9 = "okm.memref_to_ret"(%6) {index = 0 : i32} : (memref<2xsi32>) -> memref<2xsi32>
    %10 = "okm.memref_to_ret"(%8) {index = 1 : i32} : (memref<2xf32>) -> memref<2xf32>
    return
  }
}

