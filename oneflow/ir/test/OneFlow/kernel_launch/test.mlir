module {
  func.func @okm_wrap_subgraph0() {
    %0 = "okm.arg_to_memref"() {index = 0 : i32} : () -> memref<2xf32>
    %1 = "okm.alloc_memref"() : () -> memref<2xf32>
    %2 = "okm.wrapper_kernel"(%0, %1) ({
      %12 = bufferization.to_tensor %0 : memref<2xf32>
      %13 = "oneflow.relu"(%12) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 39 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %14 = bufferization.to_memref %13 : memref<2xf32>
      okm.return %14 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %3 = "okm.alloc_memref"() : () -> memref<2xf32>
    %4 = "okm.wrapper_kernel"(%2, %3) ({
      %12 = bufferization.to_tensor %2 : memref<2xf32>
      %13 = "oneflow.tanh"(%12) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 39 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %14 = bufferization.to_memref %13 : memref<2xf32>
      okm.return %14 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xf32>) -> memref<2xf32>
    %5 = "okm.ret_to_memref"() {index = 0 : i32} : () -> memref<2xsi32>
    %6 = "okm.alloc_memref"() : () -> memref<1791xi8>
    %7 = "okm.wrapper_kernel"(%4, %5, %6) ({
      %12 = bufferization.to_tensor %4 : memref<2xf32>
      %13 = "oneflow.arg_sort"(%12) {device_name = ["@0:0"], device_tag = "cuda", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 39 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
      %14 = bufferization.to_memref %13 : memref<2xsi32>
      okm.return %14 : memref<2xsi32>
    }) : (memref<2xf32>, memref<2xsi32>, memref<1791xi8>) -> memref<2xsi32>
    %8 = "okm.ret_to_memref"() {index = 1 : i32} : () -> memref<2xf32>
    %9 = "okm.wrapper_kernel"(%4, %7, %8) ({
      %12 = bufferization.to_tensor %4 : memref<2xf32>
      %13 = bufferization.to_tensor %7 : memref<2xsi32>
      %14 = "oneflow.dim_gather"(%12, %13) {device_name = ["@0:0"], device_tag = "cuda", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 39 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
      %15 = bufferization.to_memref %14 : memref<2xf32>
      okm.return %15 : memref<2xf32>
    }) : (memref<2xf32>, memref<2xsi32>, memref<2xf32>) -> memref<2xf32>
    %10 = "okm.memref_to_ret"(%7) {index = 0 : i32} : (memref<2xsi32>) -> memref<2xsi32>
    %11 = "okm.memref_to_ret"(%9) {index = 1 : i32} : (memref<2xf32>) -> memref<2xf32>
    return
  }
}