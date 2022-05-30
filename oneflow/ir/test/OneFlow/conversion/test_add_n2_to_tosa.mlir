// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

module {
  oneflow.job @GraphModule_0(%arg0: tensor<1x7x7xf32>, %arg1: tensor<1x7x7xf32>) -> tensor<1x7x7xf32> {
// CHECK:    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x7x7xf32>, tensor<1x7x7xf32>) -> tensor<1x7x7xf32>
    %res = "oneflow.add_n2"(%arg0, %arg1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "fw.model.layer4.2-add_n-223",
    op_type_name = "add_n", output_lbns = ["fw.model.layer4.2-add_n-223/out_0"], scope_symbol_id = 4611686018431205375 : i64} :
    (tensor<1x7x7xf32>, tensor<1x7x7xf32>) -> tensor<1x7x7xf32>
    oneflow.return %res : tensor<1x7x7xf32>
  }
}
