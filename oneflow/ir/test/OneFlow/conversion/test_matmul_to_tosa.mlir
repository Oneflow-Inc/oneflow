// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

// CHECK:      %0 = "tosa.reshape"(%arg0) {new_shape = [1, 1, 2048]} : (tensor<1x2048xf32>) -> tensor<1x1x2048xf32>
// CHECK:      %1 = "tosa.reshape"(%arg1) {new_shape = [1, 2048, 1000]} : (tensor<2048x1000xf32>) -> tensor<1x2048x1000xf32>
// CHECK:      %2 = "tosa.matmul"(%0, %1) : (tensor<1x1x2048xf32>, tensor<1x2048x1000xf32>) -> tensor<1x1x1000xf32>
// CHECK:      %3 = "tosa.reshape"(%2) {new_shape = [1, 1000]} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
// CHECK:      return %3 : tensor<1x1000xf32>
module {
  oneflow.job @GraphModule_0(%arg0: tensor<1x2048xf32>, %arg1: tensor<2048x1000xf32>) ->tensor<1x1000xf32> {
    %0 = "oneflow.matmul"(%arg0, %arg1)
    {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1],
     op_name = "fw.model.fc-matmul-227", output_lbns = ["fw.model.fc-matmul-227/out_0"], scope_symbol_id = 4611686018431234047 : i64,
      transpose_a = false, transpose_b = false} : (tensor<1x2048xf32>, tensor<2048x1000xf32>) -> tensor<1x1000xf32>
    oneflow.return %0: tensor<1x1000xf32>
  }
}

// CHECK:      %0 = "tosa.reshape"(%arg0) {new_shape = [1, 1, 2048]} : (tensor<1x2048xf32>) -> tensor<1x1x2048xf32>
// CHECK:      %1 = "tosa.const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<6xi32>
// CHECK:      %2 = "tosa.transpose"(%arg1, %1) : (tensor<1000x2048xf32>, tensor<6xi32>) -> tensor<2048x1000xf32>
// CHECK:      %3 = "tosa.reshape"(%2) {new_shape = [1, 2048, 1000]} : (tensor<2048x1000xf32>) -> tensor<1x2048x1000xf32>
// CHECK:      %4 = "tosa.matmul"(%0, %3) : (tensor<1x1x2048xf32>, tensor<1x2048x1000xf32>) -> tensor<1x1x1000xf32>
// CHECK:      %5 = "tosa.reshape"(%4) {new_shape = [1, 1000]} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
// CHECK:      return %5 : tensor<1x1000xf32>
module {
  oneflow.job @GraphModule_1(%arg0: tensor<1x2048xf32>, %arg1: tensor<1000x2048xf32>) ->tensor<1x1000xf32> {
    %0 = "oneflow.matmul"(%arg0, %arg1)
    {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1],
     op_name = "fw.model.fc-matmul-227", output_lbns = ["fw.model.fc-matmul-227/out_0"], scope_symbol_id = 4611686018431234047 : i64,
      transpose_a = false, transpose_b = true} : (tensor<1x2048xf32>, tensor<1000x2048xf32>) -> tensor<1x1000xf32>
    oneflow.return %0:tensor<1x1000xf32>
  }
}
