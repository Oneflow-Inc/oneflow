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

  oneflow.job @GraphToRun_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x4x5xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %output_0 = "oneflow.input"(%arg1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.1_3", output_lbns = ["_GraphToRun_0_input.0.1_3/out"], scope_symbol_id = 12 : i64, shape = [5 : si64]} : (tensor<5xf32>) -> tensor<5xf32>
    %0 = "oneflow.bias_add"(%output, %output_0) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "bias_add-0", scope_symbol_id = 12 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out, %mask = "oneflow.dropout"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "dropout-dropout-1", rate = 1.000000e+00 : f32, scope_symbol_id = 22 : i64} : (tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    %output_1 = "oneflow.output"(%out) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    oneflow.return %output_1 : tensor<2x3x4x5xf32>
  }
}
