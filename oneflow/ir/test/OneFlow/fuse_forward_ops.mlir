// RUN: oneflow-opt %s \
// RUN: -fuse-forward-only-ops | FileCheck %s

module  {
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%685: tensor<2x64x64x320xf16>, %output_574: tensor<320xf16>, %output_573: tensor<320xf16>) -> tensor<2x64x64x320xf16> {
    %y_958, %mean_959, %inv_variance_960 = "oneflow.group_norm"(%685, %output_574, %output_573) {activation = "none", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 1.000000e-05 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "unet.up_blocks.3.resnets.0.norm2-group_norm-877", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 5517 : i64} : (tensor<2x64x64x320xf16>, tensor<320xf16>, tensor<320xf16>) -> (tensor<2x64x64x320xf16>, tensor<2x32xf32>, tensor<2x32xf32>)
    %686 = "oneflow.silu"(%y_958) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.resnets.0.nonlinearity-silu-878", scope_symbol_id = 5466 : i64} : (tensor<2x64x64x320xf16>) -> tensor<2x64x64x320xf16>
    // CHECK: activation = "silu"
    // CHECK-NOT: oneflow.silu
    return %686 : tensor<2x64x64x320xf16>
  }
}
