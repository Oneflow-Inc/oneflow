// RUN: oneflow-opt -lower-oneflow-to-affine %s | FileCheck %s
// CHECK: return
module  {
  func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xf32>, %arg1: tensor<1xf64>, %arg2: tensor<96x96xf64>) {
    %0 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = "DT_Double", hierarchy = [1], input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Cast_1", op_type_name = "cast", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Cast_1/out_0"], scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf32>) -> tensor<96x96xf64>
    %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["0:0"], device_tag = "cpu", hierarchy = [1], input_lbn_segment_keys = ["x", "scalar"], input_lbn_segment_sizes = [1 : i32, 1 : i32], op_name = "ScalarMulByTensor_2", op_type_name = "scalar_mul_by_tensor", output_lbn_segment_keys = ["y"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["ScalarMulByTensor_2/y_0"], scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf64>, tensor<1xf64>) -> tensor<96x96xf64>
    return
  }
}
