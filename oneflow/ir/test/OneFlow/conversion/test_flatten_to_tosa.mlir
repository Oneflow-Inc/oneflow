// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s
module {
  oneflow.job @GraphModule_0(%args: tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32> {
// CHECK:      %0 = "tosa.reshape"(%arg0) {new_shape = [1, 2048]} : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    %0 = "oneflow.flatten"(%args) {device_name = ["@0:0"], device_tag = "cpu", end_dim = -1 : si32, hierarchy = [1], op_name = "fw.model-flatten-226",
     output_lbns = ["fw.model-flatten-226/out_0"], scope_symbol_id = 4611686018431217663 : i64, start_dim = 1 : si32}
      : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    oneflow.return %0 : tensor<1x2048xf32>
  }
}

module {
  oneflow.job @GraphModule_0(%args: tensor<4x3x2x1xf32>) -> tensor<24xf32> {
// CHECK:      %0 = "tosa.reshape"(%arg0) {new_shape = [24]} : (tensor<4x3x2x1xf32>) -> tensor<24xf32>
    %0 = "oneflow.flatten"(%args) {device_name = ["@0:0"], device_tag = "cpu", end_dim = -1 : si32, hierarchy = [1], op_name = "fw.model-flatten-226",
     output_lbns = ["fw.model-flatten-226/out_0"], scope_symbol_id = 4611686018431217663 : i64, start_dim = 0 : si32}
      : (tensor<4x3x2x1xf32>) -> tensor<24xf32>
    oneflow.return %0 : tensor<24xf32>
  }
}

module {
  oneflow.job @GraphModule_0(%args: tensor<4x3x2x1xf32>) -> tensor<4x6x1xf32> {
// CHECK:      %0 = "tosa.reshape"(%arg0) {new_shape = [4, 6, 1]} : (tensor<4x3x2x1xf32>) -> tensor<4x6x1xf32>
    %0 = "oneflow.flatten"(%args) {device_name = ["@0:0"], device_tag = "cpu", end_dim = 2 : si32, hierarchy = [1], op_name = "fw.model-flatten-226",
     output_lbns = ["fw.model-flatten-226/out_0"], scope_symbol_id = 4611686018431217663 : i64, start_dim = 1 : si32}
      : (tensor<4x3x2x1xf32>) -> tensor<4x6x1xf32>
    oneflow.return %0 : tensor<4x6x1xf32>
  }
}
