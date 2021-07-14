// RUN: oneflow-opt %s -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm  | FileCheck %s
// CHECK: return
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
// This file was edited by handed to write result to last arg in-place
module  {
  func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: memref<96x96xi64>, %arg1: memref<1xf32>, %arg2: memref<96x96xf32>) {
    %0 = linalg.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>
    linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %0 : memref<96x96xi64>, memref<f32>) outs(%arg2 : memref<96x96xf32>) {
    ^bb0(%arg3: i64, %arg4: f32, %arg5: f32):  // no predecessors
      %2 = sitofp %arg3 : i64 to f32
      %3 = mulf %2, %arg4 : f32
      linalg.yield %3 : f32
    }
    return
  }
}
