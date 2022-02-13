// RUN: oneflow-opt %s -lower-oneflow-to-tosa -tosa-to-linalg -cse --linalg-fuse-elementwise-ops -linalg-bufferize -convert-linalg-to-parallel-loops -gpu-greedy-parallel-loop-mapping \
// RUN: -convert-parallel-loops-to-gpu -gpu-kernel-outlining -buffer-host-register -canonicalize \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN: --func-bufferize -buffer-results-to-out-params -gpu-copy-arg \
// RUN: --finalizing-bufferize \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm \
// RUN: -gpu-to-llvm --reconcile-unrealized-casts --print-ir-after-all \
// RUN: | tee %test_exec_root/$(basename %s).lower.mlir \
// RUN: | oneflow-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   --entry-point-result=void

// RUN: oneflow-opt %s -lower-oneflow-to-tosa -tosa-to-linalg -cse --linalg-fuse-elementwise-ops -linalg-bufferize -convert-linalg-to-parallel-loops -gpu-greedy-parallel-loop-mapping \
// RUN: -convert-parallel-loops-to-gpu -gpu-kernel-outlining -buffer-host-register -canonicalize \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN: --func-bufferize \
// RUN: --finalizing-bufferize \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm \
// RUN: -gpu-to-llvm --reconcile-unrealized-casts \
// RUN: | tee %test_exec_root/$(basename %s).lower.mlir \
// RUN: | oneflow-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void

func @Cast_289__FUSE__ScalarMulByTensor_290(%arg0: tensor<3x3xi64>, %arg1: tensor<1xf32>) -> tensor<3x3xf32> {
  %0 = "oneflow.cast"(%arg0) {device_name = ["@0:0"], device_tag = "gpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_289", output_lbns = ["Cast_289/out_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<3x3xi64>) -> tensor<3x3xf32>
  %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "ScalarMulByTensor_290", output_lbns = ["ScalarMulByTensor_290/y_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<3x3xf32>, tensor<1xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

func @main()  {
  %a_data = memref.alloc() : memref<3x3xi64>
  %b_data = memref.alloc() : memref<1xf32>
  %a = bufferization.to_tensor %a_data : memref<3x3xi64>
  %b = bufferization.to_tensor %b_data : memref<1xf32>

  %c = call @Cast_289__FUSE__ScalarMulByTensor_290(%a, %b) : (tensor<3x3xi64>, tensor<1xf32>) -> (tensor<3x3xf32>)
  %c_buffer = bufferization.to_memref %c : memref<3x3xf32>
  %cast_c_buffer = memref.cast %c_buffer : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%cast_c_buffer) : (memref<*xf32>) -> ()
  // TODO: use real number
  // CHECK: [3, 3]

  %cast_a_data = memref.cast %a_data : memref<3x3xi64> to memref<*xi64>
  %cast_b_data = memref.cast %b_data : memref<1xf32> to memref<*xf32>
  call @print_memref_i64(%cast_a_data) : (memref<*xi64>) -> ()
  call @print_memref_f32(%cast_b_data) : (memref<*xf32>) -> ()
  return
}
func private @print_memref_f32(memref<*xf32>)
func private @print_memref_i64(memref<*xi64>)
