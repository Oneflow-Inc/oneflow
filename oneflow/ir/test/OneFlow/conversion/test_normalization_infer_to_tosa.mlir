// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s


module {
// CHECK:    %0 = "tosa.const"() {value = dense<9.99999974E-6> : tensor<f32>} : () -> tensor<f32>
// CHECK:    %1 = "tosa.reshape"(%arg1) {new_shape = [256, 1, 1]} : (tensor<256xf32>) -> tensor<256x1x1xf32>
// CHECK:    %2 = "tosa.reshape"(%arg2) {new_shape = [256, 1, 1]} : (tensor<256xf32>) -> tensor<256x1x1xf32>
// CHECK:    %3 = "tosa.reshape"(%arg3) {new_shape = [256, 1, 1]} : (tensor<256xf32>) -> tensor<256x1x1xf32>
// CHECK:    %4 = "tosa.reshape"(%arg4) {new_shape = [256, 1, 1]} : (tensor<256xf32>) -> tensor<256x1x1xf32>
// CHECK:    %5 = "tosa.sub"(%arg0, %1) : (tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
// CHECK:    %6 = "tosa.add"(%2, %0) : (tensor<256x1x1xf32>, tensor<f32>) -> tensor<256x1x1xf32>
// CHECK:    %7 = "tosa.rsqrt"(%6) : (tensor<256x1x1xf32>) -> tensor<256x1x1xf32>
// CHECK:    %8 = "tosa.mul"(%5, %7) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
// CHECK:    %9 = "tosa.mul"(%8, %3) {shift = 0 : i32} : (tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
// CHECK:    %10 = "tosa.add"(%9, %4) : (tensor<1x256x14x14xf32>, tensor<256x1x1xf32>) -> tensor<1x256x14x14xf32>
  oneflow.job @GraphModule_0(
    %x:               tensor<1x256x14x14xf32>,
    %moving_mean:     tensor<256xf32>,
    %moving_variance: tensor<256xf32>,
    %gamma:           tensor<256xf32>,
    %beta:            tensor<256xf32>) -> tensor<1x256x14x14xf32> {

    %y = "oneflow.normalization_infer"(%x, %moving_mean, %moving_variance, %gamma, %beta)
    {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cpu", epsilon = 9.99999974E-6 : f32, hierarchy = [1],
     momentum = 0.899999976 : f32, op_name = "model.layer3.5.bn2-normalization-134", operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]>
      : vector<6xi32>, output_lbns = ["model.layer3.5.bn2-normalization-134/y_0"], result_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, scope_symbol_id = 4611686018430074879 : i64,
       training = false} :
       (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    oneflow.return %y: tensor<1x256x14x14xf32>
  }
}
