// RUN: oneflow-opt %s --pass-pipeline="builtin.module(oneflow-transform-dialect-interpreter{transform-file-name=%p/matmul_codegen_spec.mlir})"

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @JITOpGenerated0(%arg0: tensor<2x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<2x10xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<2x5xf32> into tensor<1x2x5xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<5x10xf32> into tensor<1x5x10xf32>
    %0 = tensor.empty() : tensor<1x2x10xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : tensor<1x2x10xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<1x2x10xf32>
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expanded, %expanded_0 : tensor<1x2x5xf32>, tensor<1x5x10xf32>) outs(%1 : tensor<1x2x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.mulf %in, %in_1 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1x2x10xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %arg2 : tensor<1x2x10xf32>, tensor<10xf32>) outs(%0 : tensor<1x2x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<1x2x10xf32>
    %collapsed = tensor.collapse_shape %3 [[0, 1], [2]] : tensor<1x2x10xf32> into tensor<2x10xf32>
    return %collapsed : tensor<2x10xf32>
  }
}

