// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

// CHECK:    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
module {
  oneflow.job @GraphModule_0(%arg0: tensor<1x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<1x1000xf32> {
    %0 = "oneflow.broadcast_add"(%arg0, %arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "fw.model.fc-broadcast_add-228",
     output_lbns = ["fw.model.fc-broadcast_add-228/z_0"], scope_symbol_id = 4611686018431234047 : i64} :
    (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    oneflow.return %0 : tensor<1x1000xf32>
  }
}
