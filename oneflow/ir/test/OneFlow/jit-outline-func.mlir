// RUN: oneflow-opt -outline-jit-function %s | FileCheck %s
builtin.module  {
  "oneflow.job" () ({
    %data_output = "oneflow.system"() {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], input_bns = [], op_name = "Input_0", op_type_case = 137 : i32, operand_segment_sizes = dense<0> : vector<2xi32>, output_lbns = ["Input_0/out"], result_segment_sizes = dense<[1, 0]> : vector<2xi32>, scope_symbol_id = 4611686018427432958 : i64} : () -> tensor<96x96xi64>
    %data_output_0 = "oneflow.system"() {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], input_bns = [], op_name = "scale", op_type_case = 122 : i32, operand_segment_sizes = dense<0> : vector<2xi32>, output_lbns = ["scale/out"], result_segment_sizes = dense<[1, 0]> : vector<2xi32>, scope_symbol_id = 4611686018427437054 : i64} : () -> tensor<1xf32>
    %0 = "oneflow.cast"(%data_output) {device_name = ["@0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", scope_symbol_id = 4611686018427437054 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    "oneflow.system"(%data_output_0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_4", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427445246 : i64} : (tensor<1xf32>) -> ()
    %1 = "oneflow.scalar_mul_by_tensor"(%0, %data_output_0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "ScalarMulByTensor_2", scope_symbol_id = 4611686018427437054 : i64} : (tensor<96x96xf32>, tensor<1xf32>) -> tensor<96x96xf32>
    "oneflow.system"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_3", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427445246 : i64} : (tensor<96x96xf32>) -> ()
    oneflow.return
  }) {sym_name = "FuseCastScaleJob", function_type = () -> ()} : () -> ()
}
// CHECK: %0 = oneflow.mlir_jit
