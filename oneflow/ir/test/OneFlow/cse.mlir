// RUN: oneflow-opt %s \
// RUN: -cse-with-attributes-ignored -cse -cse-put-attributes -canonicalize | FileCheck %s

// CHECK-LABEL: func.func
module  {
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xi64>) -> tensor<96x96xf32> {
    %0 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %1 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_2", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
  // CHECK: "oneflow.add_n2"(%0, %0)
    %2 = "oneflow.add_n"(%0, %1) {device_name = ["0:0"], device_tag = "cpu", hierarchy = [1], op_name = "ScalarMulByTensor_2", op_type_name = "add_n", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf32>, tensor<96x96xf32>) -> tensor<96x96xf32>
    return %2 : tensor<96x96xf32>
  }
  func.func @f2(%input: tensor<2x64x64x320xf16>, %w: tensor<320x320x3x3xf16>, %bias: tensor<320xf16>) -> (tensor<2x64x64x320xf16>, tensor<2x64x64x320xf16>) {
    %transpose_w = "oneflow.transpose"(%w) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.resnets.0.conv1-conv2d-31_transpose_input_1", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 163 : i64} : (tensor<320x320x3x3xf16>) -> tensor<320x3x3x320xf16>
    %conv2d = "oneflow.conv2d"(%input, %transpose_w, %bias) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 320 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "unet.down_blocks.0.resnets.0.conv1-conv2d-31", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 163 : i64, strides = [1 : si32, 1 : si32]} : (tensor<2x64x64x320xf16>, tensor<320x3x3x320xf16>, tensor<320xf16>) -> tensor<2x64x64x320xf16>
    %transpose_w1 = "oneflow.transpose"(%w) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.resnets.0.conv1-conv2d-31_transpose_input_1", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 163 : i64} : (tensor<320x320x3x3xf16>) -> tensor<320x3x3x320xf16>
    %conv2d_2 = "oneflow.conv2d"(%input, %transpose_w1, %bias) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 320 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "unet.down_blocks.0.resnets.0.conv1-conv2d-31", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 163 : i64, strides = [1 : si32, 1 : si32]} : (tensor<2x64x64x320xf16>, tensor<320x3x3x320xf16>, tensor<320xf16>) -> tensor<2x64x64x320xf16>
    return %conv2d, %conv2d_2 : tensor<2x64x64x320xf16>, tensor<2x64x64x320xf16>
  // CHECK: %0 = "oneflow.transpose"(%arg1)
  // CHECK: %1 = "oneflow.conv2d"(%arg0, %0, %arg2)
  // CHECK: return %1, %1
  }
  func.func @f_broadcast_matmul(%x: tensor<2x4096x320xf16>, %w1: tensor<320x320xf16>, %w2: tensor<320x320xf16>, %w3: tensor<320x320xf16>) -> (tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>) {
    %matmul1 = "oneflow.broadcast_matmul"(%x, %w1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q-broadcast_matmul-16315", scope_symbol_id = 5497 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    %matmul2 = "oneflow.broadcast_matmul"(%x, %w2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k-broadcast_matmul-16316", scope_symbol_id = 5505 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    %matmul3 = "oneflow.broadcast_matmul"(%x, %w3) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v-broadcast_matmul-16317", scope_symbol_id = 5513 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    return %matmul1, %matmul2, %matmul3 : tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>
  }
}
