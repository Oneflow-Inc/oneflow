// RUN: oneflow-opt %s \
// RUN: -cse | FileCheck %s

// CHECK-LABEL: func.func
module  {
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xi64>) -> tensor<96x96xf32> {
    %0 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %1 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_2", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %2 = "oneflow.add_n"(%0, %1) {device_name = ["0:0"], device_tag = "cpu", hierarchy = [1], op_name = "ScalarMulByTensor_2", op_type_name = "scalar_mul_by_tensor", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf32>, tensor<96x96xf32>) -> tensor<96x96xf32>
    return %2 : tensor<96x96xf32>
  }
}
