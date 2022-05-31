// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s


module {
  oneflow.job @GraphModule_0(
    %x:               tensor<1x64x112x112xf32>,
    %moving_mean:     tensor<64xf32>,
    %moving_variance: tensor<64xf32>,
    %gamma:           tensor<64xf32>,
    %beta:            tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %y, %mean, %inv_variance = "oneflow.normalization"(%x, %moving_mean, %moving_variance, %gamma, %beta) {axis = 1 : si32, device_name = ["@0:0"],
    device_tag = "cpu",
    epsilon = 9.99999974E-6 : f32, hierarchy = [1],
    momentum = 0.899999976 : f32, op_name = "fw.model.bn1-normalization-2",
    operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>, output_lbns = ["fw.model.bn1-normalization-2/y_0", "fw.model.bn1-normalization-2/mean_0",
    "fw.model.bn1-normalization-2/inv_variance_0"], result_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 4611686018427453439 : i64,
    training = true} :
    (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
//CHECK:    %0 = "tosa.const"() {value = dense<9.99999974E-6> : tensor<f32>} : () -> tensor<f32>
//CHECK:    %1 = "tosa.reshape"(%arg1) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK:    %2 = "tosa.reshape"(%arg2) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK:    %3 = "tosa.reshape"(%arg3) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK:    %4 = "tosa.reshape"(%arg4) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK:    %5 = "tosa.sub"(%arg0, %1) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK:    %6 = "tosa.add"(%2, %0) : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x1xf32>
//CHECK:    %7 = "tosa.rsqrt"(%6) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
//CHECK:    %8 = "tosa.mul"(%5, %7) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK:    %9 = "tosa.mul"(%8, %3) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK:    %10 = "tosa.add"(%9, %4) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
    oneflow.return %y: tensor<1x64x112x112xf32>
  }
}
