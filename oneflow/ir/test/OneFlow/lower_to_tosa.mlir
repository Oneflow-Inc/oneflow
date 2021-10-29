// RUN: oneflow-opt -lower-oneflow-to-tosa -tosa-to-linalg-on-tensors -cse --linalg-fuse-elementwise-ops -linalg-detensorize=aggressive-mode -linalg-bufferize -tensor-bufferize -func-bufferize --tensor-constant-bufferize -buffer-results-to-out-params -convert-linalg-to-loops -convert-scf-to-std -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm  %s | FileCheck %s
// RUN: oneflow-opt -lower-oneflow-to-tosa -tosa-to-linalg-on-tensors -cse --linalg-fuse-elementwise-ops  -linalg-bufferize -tensor-bufferize -func-bufferize --tensor-constant-bufferize -buffer-results-to-out-params  -finalizing-bufferize -canonicalize %s | FileCheck %s
// CHECK: return
module  {
  func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xi64>, %arg1: tensor<1xf32>) -> tensor<96x96xf32> {
    %0 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = "DT_Float", hierarchy = [1], input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Cast_1", op_type_name = "cast", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Cast_1/out_0"], scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["0:0"], device_tag = "cpu", hierarchy = [1], input_lbn_segment_keys = ["scalar", "x"], input_lbn_segment_sizes = [1 : i32, 1 : i32], op_name = "ScalarMulByTensor_2", op_type_name = "scalar_mul_by_tensor", output_lbn_segment_keys = ["y"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["ScalarMulByTensor_2/y_0"], scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf32>, tensor<1xf32>) -> tensor<96x96xf32>
    return %1 : tensor<96x96xf32>
  }
}
