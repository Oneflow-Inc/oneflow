// RUN: python3 %s.py | FileCheck %s
module  {
  func @IdempotentJob() {
    %data_output = "oneflow.system"() {device_name = ["0:0-0"], device_tag = "gpu", hierarchy = [1], input_bns = [], op_name = "Input_0", op_type_case = 137 : i32, operand_segment_sizes = dense<0> : vector<2xi32>, output_lbns = ["Input_0/out"], result_segment_sizes = dense<[1, 0]> : vector<2xi32>, scope_symbol_id = 4611686018427420670 : i64} : () -> tensor<96x96xf32>
    // CHECK: %data_output = "oneflow.system"()
    %0 = "oneflow.relu"(%data_output) {device_name = ["0:0-0"], device_tag = "gpu", hierarchy = [1], op_name = "Relu_2", op_type_name = "relu", scope_symbol_id = 4611686018427420670 : i64} : (tensor<96x96xf32>) -> tensor<96x96xf32>
    // CHECK: %0 = "oneflow.relu"(%data_output)
    %1 = "oneflow.relu"(%data_output) {device_name = ["0:0-0"], device_tag = "gpu", hierarchy = [1], op_name = "Relu_1", op_type_name = "relu", scope_symbol_id = 4611686018427420670 : i64} : (tensor<96x96xf32>) -> tensor<96x96xf32>
    // CHECK: %1 = "oneflow.relu"(%data_output)
    "oneflow.system"(%1) {device_name = ["0:0-0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_10", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427432958 : i64} : (tensor<96x96xf32>) -> ()
    // CHECK: "oneflow.system"(%1)
    "oneflow.system"(%0) {device_name = ["0:0-0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_11", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427432958 : i64} : (tensor<96x96xf32>) -> ()
    // CHECK: "oneflow.system"(%0)
    return
    // CHECK: return
  }
}

module  {
  func @InvolutionJob() {
    %data_output = "oneflow.system"() {device_name = ["0:0-0"], device_tag = "gpu", hierarchy = [1], input_bns = [], op_name = "Input_140", op_type_case = 137 : i32, operand_segment_sizes = dense<0> : vector<2xi32>, output_lbns = ["Input_140/out"], result_segment_sizes = dense<[1, 0]> : vector<2xi32>, scope_symbol_id = 4611686018427441150 : i64} : () -> tensor<96x96xf32>
    // CHECK: %data_output = "oneflow.system"()
    "oneflow.system"(%data_output) {device_name = ["0:0-0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_153", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427445246 : i64} : (tensor<96x96xf32>) -> ()
    // CHECK: "oneflow.system"(%data_output)
    "oneflow.system"(%data_output) {device_name = ["0:0-0"], device_tag = "cpu", hierarchy = [1], input_bns = ["in"], op_name = "Return_154", op_type_case = 146 : i32, operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, output_lbns = [], result_segment_sizes = dense<0> : vector<2xi32>, scope_symbol_id = 4611686018427445246 : i64} : (tensor<96x96xf32>) -> ()
    // CHECK: "oneflow.system"(%data_output)
    return
    // CHECK: return
  }
}
