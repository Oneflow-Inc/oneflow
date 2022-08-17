// RUN: oneflow-opt %s -lower-oneflow-to-tosa -tosa-make-broadcastable -pass-pipeline="func.func(tosa-to-linalg)" -cse --linalg-fuse-elementwise-ops -linalg-bufferize -convert-linalg-to-parallel-loops -gpu-map-parallel-loops \
// RUN: -convert-parallel-loops-to-gpu -gpu-kernel-outlining -buffer-host-register -canonicalize \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN: --func-bufferize -buffer-results-to-out-params -gpu-copy-arg
func.func @Cast_289__FUSE__ScalarMulByTensor_290(%arg0: tensor<3x3xi64>, %arg1: tensor<1xf32>) -> tensor<3x3xf32> {
  %0 = "oneflow.cast"(%arg0) {device_name = ["@0:0"], device_tag = "cuda", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_289", output_lbns = ["Cast_289/out_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<3x3xi64>) -> tensor<3x3xf32>
  %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "ScalarMulByTensor_290", output_lbns = ["ScalarMulByTensor_290/y_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<3x3xf32>, tensor<1xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// CHECK: gpu.memcpy  %arg2
