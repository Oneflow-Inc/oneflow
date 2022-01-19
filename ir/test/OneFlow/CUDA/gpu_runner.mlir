// RUN: oneflow-opt %s --convert-linalg-to-parallel-loops \
// RUN:   -gpu-kernel-outlining -buffer-host-register  \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void
#map0 = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module  {
  func @main() {
    %c96 = constant 96 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %arg0 = memref.alloc() :  memref<96x96xi64>
    %arg1 = memref.alloc() :  memref<1xf32>
    %arg2 = memref.alloc() :  memref<96x96xf32>

    %0 = memref.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>

    // %cast_arg0 = memref.cast %arg0 : memref<96x96xi64> to memref<*xi64>
    // gpu.host_register %cast_arg0 : memref<*xi64>
    // %cast_arg1 = memref.cast %arg1 : memref<1xf32> to memref<*xf32>
    // gpu.host_register %cast_arg1 : memref<*xf32>
    // %cast_arg2 = memref.cast %arg2 : memref<96x96xf32> to memref<*xf32>
    // gpu.host_register %cast_arg2 : memref<*xf32>

    %1 = memref.alloc() : memref<96x96xf32>
    // %cast_1 = memref.cast %1 : memref<96x96xf32> to memref<*xf32>
    // gpu.host_register %cast_1 : memref<*xf32>

    %c1_0 = constant 1 : index
    %2 = affine.apply #map0(%c96)[%c0, %c1]
    %3 = affine.apply #map0(%c96)[%c0, %c1]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %2, %arg10 = %3, %arg11 = %c1_0) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1_0, %arg13 = %c1_0, %arg14 = %c1_0) {
      %6 = affine.apply #map1(%arg3)[%c1, %c0]
      %7 = affine.apply #map1(%arg4)[%c1, %c0]
      %8 = memref.load %arg0[%6, %7] : memref<96x96xi64>
      %9 = memref.load %0[] : memref<f32>
      %10 = sitofp %8 : i64 to f32
      %11 = mulf %10, %9 : f32
      memref.store %11, %1[%6, %7] : memref<96x96xf32>
      gpu.terminator
    }
    %c1_1 = constant 1 : index
    %4 = affine.apply #map0(%c96)[%c0, %c1]
    %5 = affine.apply #map0(%c96)[%c0, %c1]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %4, %arg10 = %5, %arg11 = %c1_1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1_1, %arg13 = %c1_1, %arg14 = %c1_1) {
      %6 = affine.apply #map1(%arg3)[%c1, %c0]
      %7 = affine.apply #map1(%arg4)[%c1, %c0]
      %8 = memref.load %1[%6, %7] : memref<96x96xf32>
      memref.store %8, %arg2[%6, %7] : memref<96x96xf32>
      gpu.terminator
    }
    return
  }
}
