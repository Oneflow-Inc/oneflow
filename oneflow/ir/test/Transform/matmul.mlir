// RUN: oneflow-opt %s  --insert-ofmempool  --convert-linalg-to-loops --convert-scf-to-cf --canonicalize --cse --memref-expand  --gpu-kernel-outlining \
// RUN: | oneflow-opt --pass-pipeline='builtin.module(gpu.module(expand-strided-metadata,lower-affine,strip-debuginfo,convert-gpu-to-nvvm,nvvm-to-cubin))'

module {
  func.func @JITOpGenerated0(%arg0: memref<5x10xf32, strided<[?, ?], offset: ?>>, %arg1: memref<2x5xf32, strided<[?, ?], offset: ?>>, %arg2: memref<2x10xf32>) attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<512xi8>
    %c0 = arith.constant 0 : index
    %view = memref.view %alloc[%c0][] : memref<512xi8> to memref<1x2x10xf32>
    %c10 = arith.constant 10 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %cst = arith.constant 0.000000e+00 : f32
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] : memref<5x10xf32, strided<[?, ?], offset: ?>> into memref<1x5x10xf32, strided<[?, ?, ?], offset: ?>>
    %expand_shape_1 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<2x5xf32, strided<[?, ?], offset: ?>> into memref<1x2x5xf32, strided<[?, ?, ?], offset: ?>>
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c2, %arg11 = %c10) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      memref.store %cst, %view[%c0_0, %arg4, %arg5] : memref<1x2x10xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c2, %arg11 = %c10) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      scf.for %arg15 = %c0_0 to %c5 step %c1 {
        %0 = memref.load %expand_shape_1[%c0_0, %arg4, %arg15] : memref<1x2x5xf32, strided<[?, ?, ?], offset: ?>>
        %1 = memref.load %expand_shape[%c0_0, %arg15, %arg5] : memref<1x5x10xf32, strided<[?, ?, ?], offset: ?>>
        %2 = memref.load %view[%c0_0, %arg4, %arg5] : memref<1x2x10xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        memref.store %4, %view[%c0_0, %arg4, %arg5] : memref<1x2x10xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    %collapse_shape = memref.collapse_shape %view [[0, 1], [2]] : memref<1x2x10xf32> into memref<2x10xf32>
    memref.copy %collapse_shape, %arg2 : memref<2x10xf32> to memref<2x10xf32>
    return
  }
}