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
}
