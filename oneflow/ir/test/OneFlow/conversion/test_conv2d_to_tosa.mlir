// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

module {
  oneflow.job @GraphModule_0(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<5x3x1x1xf32>, %arg2: tensor<5xf32>) -> tensor<1x5x224x224xf32> {
    // CHECK:%0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x3x224x224xf32>, tensor<5x3x1x1xf32>, tensor<5xf32>) -> tensor<1x5x224x224xf32>
    %111 = "oneflow.conv2d"(%arg0, %arg1, %arg2)
    {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu",
    dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1],
    kernel_size = [1 : si32, 1 : si32], op_name = "fw.model.layer4.2.conv1-conv2d-212",
    operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, output_lbns = ["fw.model.layer4.2.conv1-conv2d-212/out_0"],
    padding_before = [0 : si32, 0 : si32], scope_symbol_id = 4611686018431012863 : i64, strides = [1 : si32, 1 : si32]} :
    (tensor<1x3x224x224xf32>, tensor<5x3x1x1xf32>, tensor<5xf32>) -> tensor<1x5x224x224xf32>
    oneflow.return %111 : tensor<1x5x224x224xf32>
  }
}


module {
  oneflow.job @GraphModule_1(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<5x3x1x1xf32>) -> tensor<1x5x224x224xf32> {
    // CHECK:%0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<5xf32>} : () -> tensor<5xf32>
    // CHECK:%1 = "tosa.conv2d"(%arg0, %arg1, %0) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x3x224x224xf32>, tensor<5x3x1x1xf32>, tensor<5xf32>) -> tensor<1x5x224x224xf32>
    %111 = "oneflow.conv2d"(%arg0, %arg1)
    {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu",
    dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1],
    kernel_size = [1 : si32, 1 : si32], op_name = "fw.model.layer4.2.conv1-conv2d-212",
    operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>, output_lbns = ["fw.model.layer4.2.conv1-conv2d-212/out_0"],
    padding_before = [0 : si32, 0 : si32], scope_symbol_id = 4611686018431012863 : i64, strides = [1 : si32, 1 : si32]} :
    (tensor<1x3x224x224xf32>, tensor<5x3x1x1xf32>) -> tensor<1x5x224x224xf32>
    oneflow.return %111 : tensor<1x5x224x224xf32>
  }
}
