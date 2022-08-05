// RUN: oneflow-opt -test-oneflow-trait-folder %s | FileCheck %s

// CHECK-LABEL: func.func @testSingleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testSingleIdempotent(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %0: tensor<f32>
}

// CHECK-LABEL: func.func @testDoubleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testDoubleIdempotent(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.relu"(%0) {device_tag = "cuda", op_name = "Relu_2", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %1: tensor<f32>
}

// CHECK-LABEL: func.func @testTripleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testTripleIdempotent(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[IDEMPOTENT:%.+]] = "oneflow.relu"([[ARG0]])
  %0 = "oneflow.relu"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.relu"(%0) {device_tag = "cuda", op_name = "Relu_2", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %2 = "oneflow.relu"(%1) {device_tag = "cuda", op_name = "Relu_3", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[IDEMPOTENT]]
  return %2: tensor<f32>
}

// CHECK-LABEL: func.func @testDoubleInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testDoubleInvolution(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "oneflow.negative"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.negative"(%0) {device_tag = "cuda", op_name = "Relu_2", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[ARG0]]
  return %1: tensor<f32>
}

// CHECK-LABEL: func.func @testTripleInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testTripleInvolution(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INVOLUTION:%.+]] = "oneflow.negative"([[ARG0]])
  %0 = "oneflow.negative"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.negative"(%0) {device_tag = "cuda", op_name = "Relu_2", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %2 = "oneflow.negative"(%1) {device_tag = "cuda", op_name = "Relu_3", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[INVOLUTION]]
  return %2: tensor<f32>
}

// CHECK-LABEL: func.func @testFailedInvolutionFoldDueToDifferentPlacement
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testFailedInvolutionFoldDueToDifferentPlacement(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "oneflow.negative"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.negative"(%0) {device_tag = "cuda", op_name = "Relu_2", op_type_name = "relu", device_name = ["1:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: [[INVOLUTION:%.+]] = "oneflow.negative"(%1)
  %2 = "oneflow.negative"(%1) {device_tag = "cuda", op_name = "Relu_3", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[INVOLUTION]]
  return %2: tensor<f32>
}

// CHECK-LABEL: func.func @testFailedInvolutionFoldDueToDifferentDevice
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f32>)
func.func @testFailedInvolutionFoldDueToDifferentDevice(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "oneflow.negative"(%arg0) {device_tag = "cuda", op_name = "Relu_1", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  %1 = "oneflow.negative"(%0) {device_tag = "cpu", op_name = "Relu_2", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: [[INVOLUTION:%.+]] = "oneflow.negative"(%1)
  %2 = "oneflow.negative"(%1) {device_tag = "cuda", op_name = "Relu_3", op_type_name = "relu", device_name = ["0:0-0"], scope_symbol_id = 4611686018427420670 : i64} : (tensor<f32>) -> tensor<f32>
  // CHECK: return [[INVOLUTION]]
  return %2: tensor<f32>
}
