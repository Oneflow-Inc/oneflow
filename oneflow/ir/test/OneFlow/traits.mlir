// RUN: oneflow-opt -test-oneflow-trait-folder %s | FileCheck %s

// CHECK-LABEL: func @testSingleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func @testSingleIdempotent(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_1", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_1/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %0: tensor<f32>
}

// CHECK-LABEL: func @testDoubleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func @testDoubleIdempotent(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_1", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_1/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.relu"(%0) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_2", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_2/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %1: tensor<f32>
}

// CHECK-LABEL: func @testTripleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func @testTripleIdempotent(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_1", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_1/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.relu"(%0) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_2", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_2/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  %2 = "oneflow.relu"(%1) {device = "gpu", input_lbn_segment_keys = ["in"], input_lbn_segment_sizes = [1 : i32], op_name = "Relu_3", op_type_name = "relu", output_lbn_segment_keys = ["out"], output_lbn_segment_sizes = [1 : i32], output_lbns = ["Relu_3/out_0"], placement = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64, trainable = false} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %2: tensor<f32>
}
