// RUN: oneflow-opt %s -lower-oneflow-to-tosa -tosa-to-linalg -cse --linalg-fuse-elementwise-ops -linalg-bufferize  \
// RUN: -convert-linalg-to-parallel-loops -gpu-greedy-parallel-loop-mapping \
// RUN: -convert-parallel-loops-to-gpu -gpu-kernel-outlining -buffer-host-register -canonicalize \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN: --func-bufferize -buffer-results-to-out-params --tensor-constant-bufferize --tensor-bufferize \
// RUN: --std-bufferize --finalizing-bufferize \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm \
// RUN: -gpu-to-llvm --reconcile-unrealized-casts --print-ir-after-all \
// RUN: | tee %test_exec_root/$(basename %s).lower.mlir \
// RUN: | oneflow-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   --entry-point-result=void

module attributes {gpu.container_module}  {
  func @Cast_289__FUSE__ScalarMulByTensor_290(%arg0: tensor<3x3xi64> , %arg1: tensor<1xf32> , %arg2: tensor<3x3xf32> ) {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_memref %arg1 : memref<1xf32>
    %1 = bufferization.to_memref %arg0 : memref<3x3xi64>
    %3 = bufferization.to_memref %arg2 : memref<3x3xf32>
    %2 = memref.collapse_shape %0 [] : memref<1xf32> into memref<f32>
    gpu.launch_func  @Cast_289__FUSE__ScalarMulByTensor_290_kernel::@Cast_289__FUSE__ScalarMulByTensor_290_kernel blocks in (%c3, %c3, %c1) threads in (%c1, %c1, %c1) args(%1 : memref<3x3xi64>, %2 : memref<f32>, %3 : memref<3x3xf32>)
    return
  }
  gpu.module @Cast_289__FUSE__ScalarMulByTensor_290_kernel {
    gpu.func @Cast_289__FUSE__ScalarMulByTensor_290_kernel(%arg0: memref<3x3xi64>, %arg1: memref<f32>, %arg2: memref<3x3xf32>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.block_id"() {dimension = "y"} : () -> index
      %2 = memref.load %arg0[%0, %1] : memref<3x3xi64>
      %3 = memref.load %arg1[] : memref<f32>
      %4 = arith.sitofp %2 : i64 to f32
      %5 = arith.mulf %4, %3 : f32
      memref.store %5, %arg2[%0, %1] : memref<3x3xf32>
      gpu.return
    }
  }
  func @main() {
    %0 = memref.alloc() : memref<3x3xi64>
    %1 = memref.alloc() : memref<1xf32>
    %res = memref.alloc() : memref<3x3xf32>
    %2 = bufferization.to_tensor %0 {__inplace_results_attr__ = ["false"]} : memref<3x3xi64>
    %3 = bufferization.to_tensor %1 {__inplace_results_attr__ = ["false"]} : memref<1xf32>
    %4 = bufferization.to_tensor %res  {__inplace_results_attr__ = ["false"]} : memref<3x3xf32>
    call @Cast_289__FUSE__ScalarMulByTensor_290(%2, %3, %4) {__inplace_results_attr__ = ["false"]} : (tensor<3x3xi64>, tensor<1xf32>, tensor<3x3xf32>) -> ()
    %5 = bufferization.to_memref %4 : memref<3x3xf32>
    %6 = memref.cast %5 : memref<3x3xf32> to memref<*xf32>
    call @print_memref_f32(%6) : (memref<*xf32>) -> ()
    %7 = memref.cast %0 : memref<3x3xi64> to memref<*xi64>
    %8 = memref.cast %1 : memref<1xf32> to memref<*xf32>
    call @print_memref_i64(%7) : (memref<*xi64>) -> ()
    call @print_memref_f32(%8) : (memref<*xf32>) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_memref_i64(memref<*xi64>)
}
