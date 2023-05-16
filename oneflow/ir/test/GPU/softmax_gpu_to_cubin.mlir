// RUN: oneflow-opt %s -fold-alloc-to-subview=target-gpu=true -gpu-kernel-outlining | \
// RUN: oneflow-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm))'

#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_1x1x128xf32 : memref<1x1x128xf32> = dense<5.000000e+00> {alignment = 64 : i64}
  func.func @softmax() -> memref<16x128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %0 = memref.get_global @__constant_1x1x128xf32 : memref<1x1x128xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x128xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x128x128xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x128x128xf32>
    memref.copy %alloc_1, %alloc_2 : memref<16x128x128xf32> to memref<16x128x128xf32>
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c16, %arg7 = %c32, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c4, %arg11 = %c1) {
      %c0 = arith.constant 0 : index
      %1 = gpu.block_id  x
      %2 = gpu.block_id  y
      %3 = affine.apply #map(%2)
      %subview = memref.subview %alloc[%1, %3] [1, 4] [1, 1] : memref<16x128xf32> to memref<1x4xf32, strided<[128, 1], offset: ?>>
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
      %4 = gpu.thread_id  y
      %subview_4 = memref.subview %alloc_3[%c0, %4] [1, 1] [1, 1] : memref<1x4xf32> to memref<1x1xf32, strided<[4, 1], offset: ?>>
      linalg.fill ins(%cst_0 : f32) outs(%subview_4 : memref<1x1xf32, strided<[4, 1], offset: ?>>)
      gpu.barrier
      %5 = gpu.thread_id  x
      %6 = arith.cmpi ult, %5, %c1 : index
      scf.if %6 {
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : memref<1x1x128xf32>) outs(%subview_4 : memref<1x1xf32, strided<[4, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          %8 = arith.maxf %in, %out : f32
          linalg.yield %8 : f32
        }
      }
      gpu.barrier
      %subview_5 = memref.subview %alloc_2[%1, %3, 0] [1, 4, 128] [1, 1, 1] : memref<16x128x128xf32> to memref<1x4x128xf32, strided<[16384, 128, 1], offset: ?>>
      %subview_6 = memref.subview %subview[%c0, %4] [1, 1] [1, 1] : memref<1x4xf32, strided<[128, 1], offset: ?>> to memref<1x1xf32, strided<[128, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview_6 : memref<1x1xf32, strided<[128, 1], offset: ?>>)
      gpu.barrier
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x128xf32>
      scf.if %6 {
        %subview_11 = memref.subview %alloc_7[%c0, %4, 0] [1, 1, 128] [1, 1, 1] : memref<1x4x128xf32> to memref<1x1x128xf32, strided<[512, 128, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %subview_4 : memref<1x1x128xf32>, memref<1x1xf32, strided<[4, 1], offset: ?>>) outs(%subview_11, %subview_6 : memref<1x1x128xf32, strided<[512, 128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) {
        ^bb0(%in: f32, %in_12: f32, %out: f32, %out_13: f32):
          %8 = arith.subf %in, %in_12 : f32
          %9 = math.exp %8 : f32
          %10 = arith.addf %9, %out_13 : f32
          linalg.yield %9, %10 : f32, f32
        }
      }
      gpu.barrier
      %subview_8 = memref.subview %alloc_1[%1, %3, 0] [1, 4, 128] [1, 1, 1] : memref<16x128x128xf32> to memref<1x4x128xf32, strided<[16384, 128, 1], offset: ?>>
      %7 = affine.apply #map(%5)
      %subview_9 = memref.subview %alloc_7[%c0, %4, %7] [1, 1, 4] [1, 1, 1] : memref<1x4x128xf32> to memref<1x1x4xf32, strided<[512, 128, 1], offset: ?>>
      %subview_10 = memref.subview %subview_8[%c0, %4, %7] [1, 1, 4] [1, 1, 1] : memref<1x4x128xf32, strided<[16384, 128, 1], offset: ?>> to memref<1x1x4xf32, strided<[16384, 128, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview_9, %subview_6 : memref<1x1x4xf32, strided<[512, 128, 1], offset: ?>>, memref<1x1xf32, strided<[128, 1], offset: ?>>) outs(%subview_10 : memref<1x1x4xf32, strided<[16384, 128, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_11: f32, %out: f32):
        %8 = arith.divf %in, %in_11 : f32
        linalg.yield %8 : f32
      }
      gpu.barrier
      gpu.terminator
    }
    return %alloc_1 : memref<16x128x128xf32>
  }
}