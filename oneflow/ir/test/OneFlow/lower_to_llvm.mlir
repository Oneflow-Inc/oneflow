// RUN: oneflow-opt %s | FileCheck %s
// CHECK: return
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module  {
  func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xi64>, %arg1: tensor<1xf32>) -> tensor<96x96xf32> {
    %0 = linalg.init_tensor [96, 96] : tensor<96x96xf32>
    %1 = linalg.tensor_collapse_shape %arg1 [] : tensor<1xf32> into tensor<f32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %1 : tensor<96x96xi64>, tensor<f32>) outs(%0 : tensor<96x96xf32>) {
    ^bb0(%arg2: i64, %arg3: f32, %arg4: f32):  // no predecessors
      %3 = sitofp %arg2 : i64 to f32
      %4 = mulf %3, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<96x96xf32>
    return %2 : tensor<96x96xf32>
  }
}
