// RUN: oneflow-opt %s \
// RUN: -fuse-forward-only-ops -fuse-into-existing-op -canonicalize | FileCheck %s

module  {
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%685: tensor<2x64x64x320xf16>, %output_574: tensor<320xf16>, %output_573: tensor<320xf16>) -> tensor<2x64x64x320xf16> {
    %y_958, %mean_959, %inv_variance_960 = "oneflow.group_norm"(%685, %output_574, %output_573) {activation = "none", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 1.000000e-05 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "unet.up_blocks.3.resnets.0.norm2-group_norm-877", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 5517 : i64} : (tensor<2x64x64x320xf16>, tensor<320xf16>, tensor<320xf16>) -> (tensor<2x64x64x320xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
    %686 = "oneflow.silu"(%y_958) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.resnets.0.nonlinearity-silu-878", scope_symbol_id = 5466 : i64} : (tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
    // CHECK: activation = "silu"
    // CHECK-NOT: oneflow.silu
    return %686 : tensor<2x64x64x320xf16>
  }

  func.func @GraphToRun_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>) {
    %0 = "oneflow.bias_add"(%arg0, %arg1) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "bias_add-0", scope_symbol_id = 12 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out, %mask = "oneflow.dropout"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "dropout-dropout-1", rate = 0.750000e+00 : f32, scope_symbol_id = 22 : i64} : (tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    // CHECK: func.func @GraphToRun_0(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    // CHECK: %[[MASK:[a-zA-Z0-9_]+]] = "oneflow.random_mask_like"(%[[A]])
    // CHECK: "oneflow.fused_bias_add_mask_scale"(%[[A]], %[[B]], %[[MASK]])
    // CHECK: scale = 4.000000e+00
    return %out, %mask : tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>
  }

  func.func @GraphToRun_1(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x4x5xf32> {
    %0 = "oneflow.bias_add"(%arg0, %arg1) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "bias_add-0", scope_symbol_id = 12 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out = "oneflow.gelu"(%0) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "gelu-gelu-1", scope_symbol_id = 22 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    // CHECK: func.func @GraphToRun_1(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<5xf32>) -> tensor<2x3x4x5xf32>
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
    // CHECK: "oneflow.fused_multi_head_attention_inference"(%[[QUERY]], %[[KEY]], %[[VALUE]]) {causal = false, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], key_hidden_slice_end = -1 : si64, key_hidden_slice_start = 0 : si64, num_heads = 8 : si64, op_name = [[OP_NAME:".*"]], query_hidden_slice_end = -1 : si64, query_hidden_slice_start = 0 : si64, scope_symbol_id = 12 : i64, value_hidden_slice_end = -1 : si64, value_hidden_slice_start = 0 : si64} : (tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>) -> tensor<2x4096x320xf16>
    return %out_reshape : tensor<2x4096x320xf16>
  }
  func.func @GraphToRun_2(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x5x6xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "conv.weight", output_lbns = ["conv.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 73 : i64, shape = [3 : si64, 3 : si64, 2 : si64, 2 : si64]} : () -> tensor<3x3x2x2xf32>
    %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_2_input.0.0_2", output_lbns = ["_GraphToRun_2_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %0 = "oneflow.pad"(%output_0) {device_name = ["@0:0"], device_tag = "cpu", floating_constant_value = 0.000000e+00 : f64, hierarchy = [1], integral_constant_value = 0 : si64, op_name = "pad-0", padding = [1 : si64, 1 : si64, 1 : si64, 1 : si64], padding_after = [0 : si64, 0 : si64, 1 : si64, 1 : si64], padding_before = [0 : si64, 0 : si64, 1 : si64, 1 : si64], scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x6x7xf32>
    %1 = "oneflow.conv2d"(%0, %output) {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu", dilation_rate = [1 : si32, 1 : si32], filters = 3 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [2 : si32, 2 : si32], op_name = "conv-conv2d-1", operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>, padding_before = [0 : si32, 0 : si32], scope_symbol_id = 76 : i64, strides = [1 : si32, 1 : si32]} : (tensor<2x3x6x7xf32>, tensor<3x3x2x2xf32>) -> tensor<2x3x5x6xf32>
    %output_1 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_2_output.0.0_2", output_lbns = ["_GraphToRun_2_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 5 : si64, 6 : si64]} : (tensor<2x3x5x6xf32>) -> tensor<2x3x5x6xf32>
    // CHECK: func.func @GraphToRun_2(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x5x6xf32> {
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.variable"()
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]])
    // CHECK-NOT: oneflow.pad
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.conv2d"(%[[OUT0]], %[[OUT]])
    // CHECK: %[[OUT2:[a-zA-Z0-9_]+]] = "oneflow.output"
    return %output_1 : tensor<2x3x5x6xf32>
    }

    func.func @GraphToRun_3(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
      %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_3_input.0.0_2", output_lbns = ["_GraphToRun_3_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
      %0 = "oneflow.cast"(%output_0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
      %output_1 = "oneflow.output"(%0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_3_output.0.0_2", output_lbns = ["_GraphToRun_3_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
      // CHECK: func.func @GraphToRun_3(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> { 
      // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]]) 
      // CHECK-NOT: oneflow.cast
      // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT0]])
      // CHECK：return %[[OUT]] : tensor<2x3x4x5xf32>
      return %output_1 : tensor<2x3x4x5xf32>
    }

    func.func @GraphToRun_4(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32> {
      %output_0 = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_4_input.0.0_2", output_lbns = ["_GraphToRun_4_input.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
      %0 = "oneflow.cast"(%output_0) {device_name = ["0:0"], device_tag = "cpu", dtype = 5 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 65 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32>
      %output_1 = "oneflow.output"(%0) {data_type = 5 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_4_output.0.0_2", output_lbns = ["_GraphToRun_4_output.0.0_2/out"], scope_symbol_id = 65 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xi32>) -> tensor<2x3x4x5xi32>
      // CHECK: func.func @GraphToRun_4(%[[A:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi32> { 
      // CHECK: %[[OUT0:[a-zA-Z0-9_]+]] = "oneflow.input"(%[[A]]) 
      // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.cast"(%[[OUT0]]) 
      // CHECK: %[[OUT:[a-zA-Z0-9_]+]] = "oneflow.output"(%[[OUT1]])
      // CHECK：return %[[OUT]] : tensor<2x3x4x5xi32>
      return %output_1 : tensor<2x3x4x5xi32>
    }
}
