// RUN: oneflow-opt %s \
// RUN: -fuse-forward-only-ops -fuse-into-existing-op -fuse-normalization-ops -convert-inference-op -fuse-ops-with-backward-impl -canonicalize | FileCheck %s

module  {
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%685: tensor<2x64x64x320xf16>, %output_574: tensor<320xf16>, %output_573: tensor<320xf16>) -> tensor<2x64x64x320xf16> {
    %y_958, %mean_959, %inv_variance_960 = "oneflow.group_norm"(%685, %output_574, %output_573) {activation = "none", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 1.000000e-05 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "unet.up_blocks.3.resnets.0.norm2-group_norm-877", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 5517 : i64} : (tensor<2x64x64x320xf16>, tensor<320xf16>, tensor<320xf16>) -> (tensor<2x64x64x320xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
    %686 = "oneflow.silu"(%y_958) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.resnets.0.nonlinearity-silu-878", scope_symbol_id = 5466 : i64} : (tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
    // CHECK: activation = "silu"
    // CHECK-NOT: oneflow.silu
    return %686 : tensor<2x64x64x320xf16>
  }

  func.func @GraphToRun_bias_add_and_dropout_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>) {
    %0 = "oneflow.bias_add"(%arg0, %arg1) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "bias_add-0", scope_symbol_id = 12 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out, %mask = "oneflow.dropout"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "dropout-dropout-1", rate = 0.750000e+00 : f32, scope_symbol_id = 22 : i64} : (tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    // CHECK: func.func @GraphToRun_bias_add_and_dropout_0(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    // CHECK: %[[MASK:[a-zA-Z0-9_]+]] = "oneflow.random_mask_like"(%[[A]])
    // CHECK: "oneflow.fused_bias_add_mask_scale"(%[[A]], %[[B]], %[[MASK]])
    // CHECK: scale = 4.000000e+00
    return %out, %mask : tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>
  }

  func.func @GraphToRun_bias_add_and_gelu_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x4x5xf32> {
    %0 = "oneflow.bias_add"(%arg0, %arg1) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "bias_add-0", scope_symbol_id = 12 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out = "oneflow.gelu"(%0) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu-gelu-1", scope_symbol_id = 22 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    // CHECK: func.func @GraphToRun_bias_add_and_gelu_0(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<5xf32>) -> tensor<2x3x4x5xf32>
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.fused_bias_add_gelu"(%[[A]], %[[B]]) {axis = 3 : si32
    // CHECK： return %[[OUT0]]
    return %out : tensor<2x3x4x5xf32>
  }

  func.func @fuse_mha(%query: tensor<2x4096x320xf16>, %key: tensor<2x4096x320xf16>, %value: tensor<2x4096x320xf16>) -> tensor<2x4096x320xf16> {
    %query_reshape = "oneflow.reshape"(%query) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-1", scope_symbol_id = 12 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %key_reshape = "oneflow.reshape"(%key) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-3", scope_symbol_id = 12 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %value_reshape = "oneflow.reshape"(%value) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-5", scope_symbol_id = 12 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %query_transpose = "oneflow.transpose"(%query_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-2", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 12 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x4096x40xf16>
    %key_transpose = "oneflow.transpose"(%key_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-4", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 12 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x40x4096xf16>
    %value_transpose = "oneflow.transpose"(%value_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-6", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 12 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x4096x40xf16>
    %scores = "oneflow.batch_matmul"(%query_transpose, %key_transpose) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "batch_matmul-7", scope_symbol_id = 12 : i64, transpose_a = false, transpose_b = false} : (tensor<2x8x4096x40xf16>, tensor<2x8x40x4096xf16>) -> tensor<2x8x4096x4096xf16>
    %scores_scaled = "oneflow.scalar_div"(%scores) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 6.324555320336759 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "scalar_div-8", scope_symbol_id = 12 : i64} : (tensor<2x8x4096x4096xf16>) -> tensor<2x8x4096x4096xf16>
    %attn = "oneflow.softmax"(%scores_scaled) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "softmax-9", scope_symbol_id = 12 : i64} : (tensor<2x8x4096x4096xf16>) -> tensor<2x8x4096x4096xf16>
    %out = "oneflow.batch_matmul"(%attn, %value_transpose) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "batch_matmul-10", scope_symbol_id = 12 : i64, transpose_a = false, transpose_b = false} : (tensor<2x8x4096x4096xf16>, tensor<2x8x4096x40xf16>) -> tensor<2x8x4096x40xf16>
    %out_transpose = "oneflow.transpose"(%out) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-11", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 12 : i64} : (tensor<2x8x4096x40xf16>) -> tensor<2x4096x8x40xf16>
    %out_reshape = "oneflow.reshape"(%out_transpose) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-12", scope_symbol_id = 12 : i64, shape = [2 : si64, 4096 : si64, 320 : si64]} : (tensor<2x4096x8x40xf16>) -> tensor<2x4096x320xf16>
    // CHECK: func.func @fuse_mha(%[[QUERY:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>, %[[KEY:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>, %[[VALUE:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>)
    // CHECK: "oneflow.fused_multi_head_attention_inference"(%[[QUERY]], %[[KEY]], %[[VALUE]]) {attn_mask_type = "none", causal_diagonal_offset = 0 : si64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], key_layout = "BM(HK)", key_max_seq_len = 0 : si64, op_name = [[OP_NAME:".*"]], operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0]> : vector<7xi32>, output_layout = "BM(HK)", query_head_size = 40 : si64, query_layout = "BM(HK)", query_max_seq_len = 0 : si64, scale = 0.15811388300841897 : f64, scope_symbol_id = 12 : i64, value_layout = "BM(HK)"} : (tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>) -> tensor<2x4096x320xf16>
    return %out_reshape : tensor<2x4096x320xf16>
  }

  func.func @fuse_mha2(%query: tensor<2x4096x320xf16>, %key: tensor<2x4096x320xf16>, %value: tensor<2x4096x320xf16>) -> tensor<2x4096x320xf16> {
    %value_reshape = "oneflow.reshape"(%value) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-124", scope_symbol_id = 661 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %key_reshape = "oneflow.reshape"(%key) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-121", scope_symbol_id = 661 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %query_reshape = "oneflow.reshape"(%query) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-116", scope_symbol_id = 661 : i64, shape = [2 : si64, 4096 : si64, 8 : si64, 40 : si64]} : (tensor<2x4096x320xf16>) -> tensor<2x4096x8x40xf16>
    %value_permute = "oneflow.transpose"(%value_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-transpose-125", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 661 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x4096x40xf16>
    %key_permute = "oneflow.transpose"(%key_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-transpose-122", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 661 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x4096x40xf16>
    %query_permute = "oneflow.transpose"(%query_reshape) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-transpose-117", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 661 : i64} : (tensor<2x4096x8x40xf16>) -> tensor<2x8x4096x40xf16>
    %value_reshape_to_batch = "oneflow.reshape"(%value_permute) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-126", scope_symbol_id = 661 : i64, shape = [16 : si64, 4096 : si64, 40 : si64]} : (tensor<2x8x4096x40xf16>) -> tensor<16x4096x40xf16>
    %key_reshape_to_batch = "oneflow.reshape"(%key_permute) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-123", scope_symbol_id = 661 : i64, shape = [16 : si64, 4096 : si64, 40 : si64]} : (tensor<2x8x4096x40xf16>) -> tensor<16x4096x40xf16>
    %query_reshape_to_batch = "oneflow.reshape"(%query_permute) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-118", scope_symbol_id = 661 : i64, shape = [16 : si64, 4096 : si64, 40 : si64]} : (tensor<2x8x4096x40xf16>) -> tensor<16x4096x40xf16>
    %key_transpose = "oneflow.transpose"(%key_reshape_to_batch) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-transpose-128", perm = [0 : si32, 2 : si32, 1 : si32], scope_symbol_id = 661 : i64} : (tensor<16x4096x40xf16>) -> tensor<16x40x4096xf16>
    %scores_scaled = "oneflow.batch_matmul"(%query_reshape_to_batch, %key_transpose) {alpha = 0.15811388300841897 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-batch_matmul-129", scope_symbol_id = 661 : i64, transpose_a = false, transpose_b = false} : (tensor<16x4096x40xf16>, tensor<16x40x4096xf16>) -> tensor<16x4096x4096xf16>
    %attn = "oneflow.softmax"(%scores_scaled) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-softmax-130", scope_symbol_id = 661 : i64} : (tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16>
    %309 = "oneflow.batch_matmul"(%attn, %value_reshape_to_batch) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-batch_matmul-131", scope_symbol_id = 661 : i64, transpose_a = false, transpose_b = false} : (tensor<16x4096x4096xf16>, tensor<16x4096x40xf16>) -> tensor<16x4096x40xf16>
    %310 = "oneflow.reshape"(%309) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-132", scope_symbol_id = 661 : i64, shape = [2 : si64, 8 : si64, 4096 : si64, 40 : si64]} : (tensor<16x4096x40xf16>) -> tensor<2x8x4096x40xf16>
    %311 = "oneflow.transpose"(%310) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-transpose-133", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 661 : i64} : (tensor<2x8x4096x40xf16>) -> tensor<2x4096x8x40xf16>
    %out_reshape_to_heads = "oneflow.reshape"(%311) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.down_blocks.0.attentions.1.transformer_blocks.0.attn1-reshape-134", scope_symbol_id = 661 : i64, shape = [2 : si64, 4096 : si64, 320 : si64]} : (tensor<2x4096x8x40xf16>) -> tensor<2x4096x320xf16>
    // CHECK: func.func @fuse_mha2(%[[QUERY:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>, %[[KEY:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>, %[[VALUE:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>)
    // CHECK: oneflow.fused_multi_head_attention_inference"(%[[QUERY]], %[[KEY]], %[[VALUE]]) {attn_mask_type = "none", causal_diagonal_offset = 0 : si64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], key_layout = "BM(HK)", key_max_seq_len = 0 : si64, op_name = [[OP_NAME:".*"]], operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0]> : vector<7xi32>, output_layout = "BM(HK)", query_head_size = 40 : si64, query_layout = "BM(HK)", query_max_seq_len = 0 : si64, scale = 0.15811388300841897 : f64, scope_symbol_id = 661 : i64, value_layout = "BM(HK)"} : (tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>) -> tensor<2x4096x320xf16>
    return %out_reshape_to_heads : tensor<2x4096x320xf16>
  }

  func.func @GraphToRun_pad_and_conv2d_0(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x5x6xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "conv.weight", output_lbns = ["conv.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 73 : i64, shape = [3 : si64, 3 : si64, 2 : si64, 2 : si64]} : () -> tensor<3x3x2x2xf32>
    %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_2_input.0.0_2", output_lbns = ["_GraphToRun_2_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %0 = "oneflow.pad"(%output_0) {device_name = ["@0:0"], device_tag = "cpu", floating_constant_value = 0.000000e+00 : f64, hierarchy = [1], integral_constant_value = 0 : si64, op_name = "pad-0", padding = [1 : si64, 1 : si64, 1 : si64, 1 : si64], padding_after = [0 : si64, 0 : si64, 1 : si64, 1 : si64], padding_before = [0 : si64, 0 : si64, 1 : si64, 1 : si64], scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x6x7xf32>
    %1 = "oneflow.conv2d"(%0, %output) {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu", dilation_rate = [1 : si32, 1 : si32], filters = 3 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [2 : si32, 2 : si32], op_name = "conv-conv2d-1", operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>, padding_before = [0 : si32, 0 : si32], scope_symbol_id = 76 : i64, strides = [1 : si32, 1 : si32]} : (tensor<2x3x6x7xf32>, tensor<3x3x2x2xf32>) -> tensor<2x3x5x6xf32>
    %output_1 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_2_output.0.0_2", output_lbns = ["_GraphToRun_2_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 5 : si64, 6 : si64]} : (tensor<2x3x5x6xf32>) -> tensor<2x3x5x6xf32>
    // CHECK: func.func @GraphToRun_pad_and_conv2d_0(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x5x6xf32> {
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]])
    // CHECK-NOT: oneflow.pad
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.conv2d"(%[[OUT0]], %[[OUT]])
    // CHECK: %[[OUT2:[a-zA-Z0-9_]+]] = "oneflow.output"
    return %output_1 : tensor<2x3x5x6xf32>
  }

  func.func @GraphToRun_same_dtype_cast_0(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_3_input.0.0_2", output_lbns = ["_GraphToRun_3_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %0 = "oneflow.cast"(%output_0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %output_1 = "oneflow.output"(%0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_3_output.0.0_2", output_lbns = ["_GraphToRun_3_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    // CHECK: func.func @GraphToRun_same_dtype_cast_0(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]])
    // CHECK-NOT: oneflow.cast
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT0]])
    // CHECK：return %[[OUT]] : tensor<2x3x4x5xf32>
    return %output_1 : tensor<2x3x4x5xf32>
  }

  func.func @GraphToRun_same_dtype_cast_1(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32> {
    %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_4_input.0.0_2", output_lbns = ["_GraphToRun_4_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %0 = "oneflow.cast"(%output_0) {device_name = ["0:0"], device_tag = "cpu", dtype = 5 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32>
    %output_1 = "oneflow.output"(%0) {data_type = 5 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_4_output.0.0_2", output_lbns = ["_GraphToRun_4_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
    // CHECK: func.func @GraphToRun_same_dtype_cast_1(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32> {
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]])
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.cast"(%[[OUT0]])
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT1]])
    // CHECK：return %[[OUT]] : tensor<2x3x4x5xi32>
    return %output_1 : tensor<2x3x4x5xi32>
  }

  func.func @GraphToRun_scale_tril_0() -> tensor<5x5xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "FreeEagerTensor-1", output_lbns = ["FreeEagerTensor-1/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 12 : i64, shape = [5 : si64, 5 : si64], trainable = false} : () -> tensor<5x5xf32>
    %0 = "oneflow.scalar_mul"(%output) {device_name = ["@0:0"], device_tag = "cuda", float_operand = -2.300000e+00 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "scalar_mul-0", scope_symbol_id = 12 : i64} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    %1 = "oneflow.tril"(%0) {device_name = ["@0:0"], device_tag = "cuda", diagonal = -1 : si64, floating_fill_value = 0.000000e+00 : f64, hierarchy = [1], integer_fill_value = 0 : si64, is_floating_fill_value = false, op_name = "tril-2", scope_symbol_id = 12 : i64} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    %output_0 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_TestFuseScaleTril_0_output.0.0_2", output_lbns = ["_TestFuseScaleTril_0_output.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [5 : si64, 5 : si64]} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    // CHECK: func.func @GraphToRun_scale_tril_0() -> tensor<5x5xf32> {
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.fused_scale_tril"(%[[OUT0]])
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT1]])
    // CHECK：return %[[OUT]]
    return %output_0 : tensor<5x5xf32>
  }

  func.func @GraphToRun_scale_tril_1() -> tensor<5x5xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "FreeEagerTensor-1", output_lbns = ["FreeEagerTensor-1/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 66 : i64, shape = [5 : si64, 5 : si64], trainable = false} : () -> tensor<5x5xf32>
    %0 = "oneflow.tril"(%output) {device_name = ["@0:0"], device_tag = "cuda", diagonal = -1 : si64, floating_fill_value = 0.000000e+00 : f64, hierarchy = [1], integer_fill_value = 0 : si64, is_floating_fill_value = false, op_name = "tril-0", scope_symbol_id = 66 : i64} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    %1 = "oneflow.scalar_mul"(%0) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 2.000000e+00 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "scalar_mul-2", scope_symbol_id = 66 : i64} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    %output_0 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_TestFuseTrilScale_1_output.0.0_2", output_lbns = ["_TestFuseTrilScale_1_output.0.0_2/out"], scope_symbol_id = 66 : i64, shape = [5 : si64, 5 : si64]} : (tensor<5x5xf32>) -> tensor<5x5xf32>
    // CHECK: func.func @GraphToRun_scale_tril_1() -> tensor<5x5xf32> {
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.fused_scale_tril"(%[[OUT0]])
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT1]])
    // CHECK：return %[[OUT]]
    return %output_0 : tensor<5x5xf32>
  }

  func.func @GraphToRun_normalization_1(%x: tensor<2x3x224x224xf32>, %moving_mean: tensor<3xf32>, %moving_variance: tensor<3xf32>, %gamma: tensor<3xf32>, %beta: tensor<3xf32>, %addend: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
    %y, %mean, %inv_variance = "oneflow.normalization"(%x, %moving_mean, %moving_variance, %gamma, %beta) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cpu", epsilon = 9.99999974E-6 : f32, hierarchy = [1], momentum = 0.899999976 : f32, op_name = "normalization-2", operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>, result_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 12 : i64, training = true} : (tensor<2x3x224x224xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x224x224xf32>, tensor<3xf32>, tensor<3xf32>)
    %0 = "oneflow.add_n2"(%y, %addend) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "add_n-7", op_type_name = "add_n", scope_symbol_id = 12 : i64} : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-8", scope_symbol_id = 12 : i64} : (tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
    // CHECK: func.func @GraphToRun_normalization_1(%[[X:[a-zA-Z0-9_]+]]: tensor<2x3x224x224xf32>, %[[MOVING_MEAN:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[MOVING_VARIANCE:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[GAMMA:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[BETA:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[ADDEND:[a-zA-Z0-9_]+]]: tensor<2x3x224x224xf32>)
    // CHECK: %[[Y:[a-zA-Z0-9_]+]], %[[reserve_space:[a-zA-Z0-9_]+]], %[[mean:[a-zA-Z0-9_]+]], %[[inv_variance:[a-zA-Z0-9_]+]] = "oneflow.normalization_add_relu"(%[[X]], %[[ADDEND]], %[[MOVING_MEAN]], %[[MOVING_VARIANCE]], %[[GAMMA]], %[[BETA]])
    // CHECK： return %[[Y]]
    return %1 : tensor<2x3x224x224xf32>
  }

  func.func @GraphToRun_normalization_2(%x: tensor<2x3x224x224xf32>, %moving_mean: tensor<3xf32>, %moving_variance: tensor<3xf32>, %gamma: tensor<3xf32>, %beta: tensor<3xf32>, %addend: tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32> {
    %y = "oneflow.normalization_infer"(%x, %moving_mean, %moving_variance, %gamma, %beta) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cpu", epsilon = 9.99999974E-6 : f32, hierarchy = [1], momentum = 0.899999976 : f32, op_name = "normalization-2", operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>, result_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 12 : i64, training = true} : (tensor<2x3x224x224xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x224x224xf32>)
    %0 = "oneflow.add_n2"(%y, %addend) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "add_n-7", op_type_name = "add_n", scope_symbol_id = 12 : i64} : (tensor<2x3x224x224xf32>, tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-8", scope_symbol_id = 12 : i64} : (tensor<2x3x224x224xf32>) -> tensor<2x3x224x224xf32>
    // CHECK: func.func @GraphToRun_normalization_2(%[[X:[a-zA-Z0-9_]+]]: tensor<2x3x224x224xf32>, %[[MOVING_MEAN:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[MOVING_VARIANCE:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[GAMMA:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[BETA:[a-zA-Z0-9_]+]]: tensor<3xf32>, %[[ADDEND:[a-zA-Z0-9_]+]]: tensor<2x3x224x224xf32>)
    // CHECK: %[[Y:[a-zA-Z0-9_]+]], %[[reserve_space:[a-zA-Z0-9_]+]], %[[mean:[a-zA-Z0-9_]+]], %[[inv_variance:[a-zA-Z0-9_]+]] = "oneflow.normalization_add_relu"(%[[X]], %[[ADDEND]], %[[MOVING_MEAN]], %[[MOVING_VARIANCE]], %[[GAMMA]], %[[BETA]])
    // CHECK： return %[[Y]]
    return %1 : tensor<2x3x224x224xf32>
  }

  func.func @GraphToRun_conv_bn_1(%arg0: tensor<1x3x224x224xf32>, %moving_mean: tensor<64xf32>, %moving_variance: tensor<64xf32>, %beta: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_conv_bn_1_input.0.0_2", output_lbns = ["_conv_bn_1_input.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [1 : si64, 3 : si64, 224 : si64, 224 : si64]} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    %0 = "oneflow.variable_ir"() {value = dense<1.0> : tensor<64x3x7x7xf32> ,data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "model.conv1.weight", output_lbns = ["model.conv1.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 18 : i64, shape = [64 : si64, 3 : si64, 7 : si64, 7 : si64], nd_sbp = ["B"]} : () -> tensor<64x3x7x7xf32>
    %gamma = "oneflow.variable_ir"() {value = dense<1.0> : tensor<64xf32> ,data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "model.bn.gamma", output_lbns = ["model.bn.gamma/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 18 : i64, shape = [64 : si64], nd_sbp = ["B"]} : () -> tensor<64xf32>
    %1 = "oneflow.conv2d"(%output, %0) {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 64 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [7 : si32, 7 : si32], op_name = "model.conv1-conv2d-0", operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>, padding_before = [3 : si32, 3 : si32], scope_symbol_id = 21 : i64, strides = [2 : si32, 2 : si32]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %2 = "oneflow.normalization_infer"(%1,  %moving_mean, %moving_variance, %gamma, %beta) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.99999974E-6 : f32, hierarchy = [1], momentum = 0.899999976 : f32, op_name = "model.bn1-normalization-1", operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>, result_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, scope_symbol_id = 41 : i64, training = false} : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    // CHECK: func.func @GraphToRun_conv_bn_1(%[[ARG_0:[a-zA-Z0-9_]+]]: tensor<1x3x224x224xf32>, %[[MOVING_MEAN:[a-zA-Z0-9_]+]]: tensor<64xf32>, %[[MOVING_VARIANCE:[a-zA-Z0-9_]+]]: tensor<64xf32>, %[[BETA:[a-zA-Z0-9_]+]]: tensor<64xf32>)
    // CHECK:  %[[GAMMA:[a-zA-Z0-9_]+]] = "oneflow.variable_ir"()
    // CHECK:  %[[WEIGHT:[a-zA-Z0-9_]+]] = "oneflow.variable_ir"()
    // CHECK:  %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[ARG_0]])
    // CHECK:  %[[OUT2:[a-zA-Z0-9_]+]] = "oneflow.scalar_add"(%[[MOVING_VARIANCE]])
    // CHECK:  %[[OUT3:[a-zA-Z0-9_]+]] = "oneflow.sqrt"(%[[OUT2]])
    // CHECK:  %[[OUT4:[a-zA-Z0-9_]+]] = "oneflow.broadcast_div"(%[[GAMMA]], %[[OUT3]])
    // CHECK:  %[[OUT5:[a-zA-Z0-9_]+]] = "oneflow.reshape"(%[[OUT4]])
    // CHECK:  %[[OUT6:[a-zA-Z0-9_]+]] = "oneflow.broadcast_mul"(%[[WEIGHT]], %[[OUT5]])
    // CHECK:  %[[OUT7:[a-zA-Z0-9_]+]] = "oneflow.broadcast_mul"(%[[MOVING_MEAN]], %[[OUT4]])
    // CHECK:  %[[OUT8:[a-zA-Z0-9_]+]] = "oneflow.broadcast_sub"(%[[BETA]], %[[OUT7]])
    // CHECK:  %[[OUT9:[a-zA-Z0-9_]+]] = "oneflow.conv2d"(%[[OUT]], %[[OUT6]], %[[OUT8]])
    // CHECK： return %[[OUT9]]
    return %2 : tensor<1x64x112x112xf32>
  }


  func.func @GraphToRun_broadcastmul_to_scalarmul_1(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1xf32>) -> tensor<64x3x7x7xf32> {
    %output = "oneflow.broadcast_mul"(%arg0, %arg1) {device_name = ["@0:0"], device_tag = "cuda", op_name = "multiply"} : (tensor<64x3x7x7xf32>, tensor<1xf32>) -> tensor<64x3x7x7xf32>
    // CHECK: func.func @GraphToRun_broadcastmul_to_scalarmul_1(%[[ARG_0:[a-zA-Z0-9_]+]]: tensor<64x3x7x7xf32>, %[[ARG_1:[a-zA-Z0-9_]+]]: tensor<1xf32>)
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.scalar_mul_by_tensor"(%[[ARG_0]], %[[ARG_1]]
    return %output : tensor<64x3x7x7xf32>
    }

  func.func @GraphToRun_fused_gelu_1(%arg0: tensor<2x2304x640xf32>) -> tensor<2x2304x5120xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj.weight", output_lbns = ["gelu_mod.proj.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 18 : i64, shape = [10240 : si64, 640 : si64]} : () -> tensor<10240x640xf32>
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj.bias", output_lbns = ["gelu_mod.proj.bias/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 25 : i64, shape = [10240 : si64]} : () -> tensor<10240xf32>
    %output_1 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 2304 : si64, 640 : si64]} : (tensor<2x2304x640xf32>) -> tensor<2x2304x640xf32>
    %matmul_wx = "oneflow.broadcast_matmul"(%output_1, %output) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj-broadcast_matmul-0", scope_symbol_id = 21 : i64, transpose_a = false, transpose_b = true} : (tensor<2x2304x640xf32>, tensor<10240x640xf32>) -> tensor<2x2304x10240xf32>
    %matmul_wx_add = "oneflow.broadcast_add"(%matmul_wx, %output_0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj-broadcast_add-1", scope_symbol_id = 21 : i64} : (tensor<2x2304x10240xf32>, tensor<10240xf32>) -> tensor<2x2304x10240xf32>
    %hidden_states = "oneflow.narrow"(%matmul_wx_add) {device_name = ["@0:0"], device_tag = "cuda", dim = 2 : si64, hierarchy = [1], length = 5120 : si64, op_name = "gelu_mod-narrow-2", scope_symbol_id = 31 : i64, start = 0 : si64} : (tensor<2x2304x10240xf32>) -> tensor<2x2304x5120xf32>
    %gate = "oneflow.narrow"(%matmul_wx_add) {device_name = ["@0:0"], device_tag = "cuda", dim = 2 : si64, hierarchy = [1], length = 5120 : si64, op_name = "gelu_mod-narrow-3", scope_symbol_id = 31 : i64, start = 5120 : si64} : (tensor<2x2304x10240xf32>) -> tensor<2x2304x5120xf32>
    %gate_activate = "oneflow.gelu"(%gate) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod-gelu-4", scope_symbol_id = 31 : i64} : (tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    %y = "oneflow.broadcast_mul"(%hidden_states, %gate_activate) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod-broadcast_mul-5", scope_symbol_id = 31 : i64} : (tensor<2x2304x5120xf32>, tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    %output_2 = "oneflow.output"(%y) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 2304 : si64, 5120 : si64]} : (tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    // CHECK: func.func @GraphToRun_fused_gelu_1(%[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x2304x640xf32>) -> tensor<2x2304x5120xf32> {
    // CHECK:  %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK:  %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK:  %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[ARG_0]])
    // CHECK:  %[[Y:[a-zA-Z0-9_]+]], %[[MATMUL:[a-zA-Z0-9_]+]] = "oneflow.fused_glu"(%[[OUT1]], %[[OUT]], %[[OUT0]])
    // CHECK:  %[[OUT2:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[Y]])
    // CHECK： return %[[OUT2]]
    return %output_2 : tensor<2x2304x5120xf32>
  }

  func.func @GraphToRun_fused_gelu_2(%arg0: tensor<2x2304x640xf32>) -> tensor<2x2304x5120xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj.weight", output_lbns = ["gelu_mod.proj.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 18 : i64, shape = [10240 : si64, 640 : si64]} : () -> tensor<10240x640xf32>
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj.bias", output_lbns = ["gelu_mod.proj.bias/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 25 : i64, shape = [10240 : si64]} : () -> tensor<10240xf32>
    %output_1 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 2304 : si64, 640 : si64]} : (tensor<2x2304x640xf32>) -> tensor<2x2304x640xf32>
    %matmul_wx_add = "oneflow.fused_matmul_bias"(%output_1, %output, %output_0) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod.proj-broadcast_add-1", scope_symbol_id = 21 : i64} : (tensor<2x2304x640xf32>, tensor<10240x640xf32>, tensor<10240xf32>) -> tensor<2x2304x10240xf32>
    %hidden_states = "oneflow.narrow"(%matmul_wx_add) {device_name = ["@0:0"], device_tag = "cuda", dim = 2 : si64, hierarchy = [1], length = 5120 : si64, op_name = "gelu_mod-narrow-2", scope_symbol_id = 31 : i64, start = 0 : si64} : (tensor<2x2304x10240xf32>) -> tensor<2x2304x5120xf32>
    %gate = "oneflow.narrow"(%matmul_wx_add) {device_name = ["@0:0"], device_tag = "cuda", dim = 2 : si64, hierarchy = [1], length = 5120 : si64, op_name = "gelu_mod-narrow-3", scope_symbol_id = 31 : i64, start = 5120 : si64} : (tensor<2x2304x10240xf32>) -> tensor<2x2304x5120xf32>
    %gate_activate = "oneflow.gelu"(%gate) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod-gelu-4", scope_symbol_id = 31 : i64} : (tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    %y = "oneflow.broadcast_mul"(%hidden_states, %gate_activate) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu_mod-broadcast_mul-5", scope_symbol_id = 31 : i64} : (tensor<2x2304x5120xf32>, tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    %output_2 = "oneflow.output"(%y) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 2304 : si64, 5120 : si64]} : (tensor<2x2304x5120xf32>) -> tensor<2x2304x5120xf32>
    // CHECK: func.func @GraphToRun_fused_gelu_2(%[[ARG_0:[a-zA-Z0-9_]+]]: tensor<2x2304x640xf32>) -> tensor<2x2304x5120xf32> {
    // CHECK:  %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK:  %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK:  %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[ARG_0]])
    // CHECK:  %[[Y:[a-zA-Z0-9_]+]], %[[MATMUL:[a-zA-Z0-9_]+]] = "oneflow.fused_glu"(%[[OUT1]], %[[OUT]], %[[OUT0]])
    // CHECK:  %[[OUT2:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[Y]])
    // CHECK： return %[[OUT2]]
    return %output_2 : tensor<2x2304x5120xf32>
  }
}
