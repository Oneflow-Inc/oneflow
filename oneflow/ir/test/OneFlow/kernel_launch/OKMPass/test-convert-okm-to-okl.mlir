// RUN: oneflow-opt %s \
// RUN: -convert-okm-to-okl \
// RUN: | FileCheck %s

// CHECK:  func.func @okl_subgraph(%arg0: !okl.launcher_ctx) attributes {pool_size = 1024 : i64} {
// CHECK-COUNT-4:    "okl.wrapper_kernel"()

module {
  func.func @okm_alloc_subgraph0() {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %0 = "okm.alloc_memref"() : () -> memref<1024xi8>
    %1 = "okm.arg_to_memref"() {index = 0 : i32} : () -> memref<2xf32>
    %2 = memref.view %0[%c0][] : memref<1024xi8> to memref<2xf32>
    %3 = "okm.wrapper_kernel"(%1, %2) ({
      %12 = bufferization.to_tensor %1 : memref<2xf32>
      %13 = "oneflow.relu"(%12) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %14 = bufferization.to_memref %13 : memref<2xf32>
      okm.return %14 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %4 = memref.view %0[%c512][] : memref<1024xi8> to memref<2xf32>
    %5 = "okm.wrapper_kernel"(%2, %4) ({
      %12 = bufferization.to_tensor %2 : memref<2xf32>
      %13 = "oneflow.tanh"(%12) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %14 = bufferization.to_memref %13 : memref<2xf32>
      okm.return %14 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %6 = "okm.ret_to_memref"() {index = 0 : i32} : () -> memref<2xsi32>
    %7 = "okm.wrapper_kernel"(%4, %6) ({
      %12 = bufferization.to_tensor %4 : memref<2xf32>
      %13 = "oneflow.arg_sort"(%12) {device_name = ["@0:0"], device_tag = "cpu", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
      %14 = bufferization.to_memref %13 : memref<2xsi32>
      okm.return %14 : memref<2xsi32>
    }) : (memref<2xf32>, memref<2xsi32>) -> memref<2xsi32>
    %8 = "okm.ret_to_memref"() {index = 1 : i32} : () -> memref<2xf32>
    %9 = "okm.wrapper_kernel"(%4, %6, %8) ({
      %12 = bufferization.to_tensor %4 : memref<2xf32>
      %13 = bufferization.to_tensor %6 : memref<2xsi32>
      %14 = "oneflow.dim_gather"(%12, %13) {device_name = ["@0:0"], device_tag = "cpu", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 12 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
      %15 = bufferization.to_memref %14 : memref<2xf32>
      okm.return %15 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xsi32>, memref<2xf32>) -> memref<2xf32>
    %10 = "okm.memref_to_ret"(%7) {index = 0 : i32} : (memref<2xsi32>) -> memref<2xsi32>
    %11 = "okm.memref_to_ret"(%9) {index = 1 : i32} : (memref<2xf32>) -> memref<2xf32>
    return
  }
}
